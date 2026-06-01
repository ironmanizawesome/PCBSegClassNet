"""
trace_seg U-Net 학습용 Dataset.

(사진, GT mask) 쌍을 data/images/{BOARD_ID}/{side}_{source}.* 와
data/masks/{BOARD_ID}/{side}.png 에서 자동 수집.

주의 — 정합(registration):
  현재 사진과 GT mask는 homography 정합이 안 돼 있다 (단순 resize로만 매칭).
  따라서 실제 일반화 학습에는 부적합하고, 정합은 다음 단계 작업이다.
  단 **overfit 검증**(네트워크+손실+루프가 1 샘플을 외울 수 있는지)에는 충분하다.

mask 규약: binary PNG (배경 0, 전경 255) → {0,1} float 텐서.
"""
import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

SIDES = ("front", "back")
SOURCES = ("photo", "web")          # photo=직접촬영, web=roboflow


def list_samples(images_dir, masks_dir, sources=SOURCES, board=None):
    """(img_path, mask_path, label) 리스트. board 지정 시 해당 보드만."""
    images_dir, masks_dir = Path(images_dir), Path(masks_dir)
    samples = []
    if not images_dir.exists():
        return samples
    for board_dir in sorted(p for p in images_dir.iterdir() if p.is_dir()):
        if board and board_dir.name != board:
            continue
        for side in SIDES:
            gt = masks_dir / board_dir.name / f"{side}.png"
            if not gt.exists():
                continue
            for src in sources:
                hits = sorted(board_dir.glob(f"{side}_{src}.*"))
                if hits:
                    samples.append((hits[0], gt, f"{board_dir.name}/{side}/{src}"))
    return samples


class TraceDataset(Dataset):
    def __init__(self, samples, size: int = 512, augment: bool = False):
        self.samples = samples
        self.size = size
        self.augment = augment
        self.p_rawify = 0.0          # rawify(폰사진 흉내)는 현 데이터(6설계)론 역효과 -> 기본 off

    def __len__(self):
        return len(self.samples)

    def _letterbox(self, img, pad_value, interp):
        """비율 유지 리사이즈 + 중앙 패딩 (가로/세로 보드 왜곡 방지)."""
        h, w = img.shape[:2]
        s = self.size / max(h, w)
        nh, nw = max(1, round(h * s)), max(1, round(w * s))
        r = cv2.resize(img, (nw, nh), interpolation=interp)
        top, left = (self.size - nh) // 2, (self.size - nw) // 2
        return cv2.copyMakeBorder(r, top, self.size - nh - top,
                                  left, self.size - nw - left,
                                  cv2.BORDER_CONSTANT, value=pad_value)

    def _load_image(self, path):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)        # BGR
        if img is None:
            raise FileNotFoundError(f"이미지 로드 실패: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)          # native RGB

    def _load_mask(self, path):
        m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(f"마스크 로드 실패: {path}")
        return (m > 127).astype(np.float32)                 # native {0,1}

    def _rawify(self, img, mask):
        """정합(보드 꽉 찬) 이미지를 '폰 사진'처럼 변형: 랜덤 축소+회전+배경+위치.
        raw 사진 분포(배경·여백·기울기·스케일)에 강건하게. img/mask 동일 기하 적용."""
        size = self.size
        h, w = img.shape[:2]
        scale = np.random.uniform(0.35, 0.9) * size / max(h, w)
        bw, bh = max(1, int(w * scale)), max(1, int(h * scale))
        bimg = cv2.resize(img, (bw, bh), interpolation=cv2.INTER_AREA)
        bmask = cv2.resize(mask, (bw, bh), interpolation=cv2.INTER_NEAREST)

        ang = np.random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((bw / 2, bh / 2), ang, 1.0)
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        nw, nh = int(bh * sin + bw * cos), int(bh * cos + bw * sin)
        M[0, 2] += nw / 2 - bw / 2
        M[1, 2] += nh / 2 - bh / 2
        region = np.ones((bh, bw), np.uint8)
        rimg = cv2.warpAffine(bimg, M, (nw, nh), flags=cv2.INTER_AREA)
        rmask = cv2.warpAffine(bmask, M, (nw, nh), flags=cv2.INTER_NEAREST)
        rreg = cv2.warpAffine(region, M, (nw, nh), flags=cv2.INTER_NEAREST)
        if nw > size or nh > size:                       # 회전으로 캔버스 초과 시 축소
            f = min(size / nw, size / nh)
            nw, nh = max(1, int(nw * f)), max(1, int(nh * f))
            rimg = cv2.resize(rimg, (nw, nh), interpolation=cv2.INTER_AREA)
            rmask = cv2.resize(rmask, (nw, nh), interpolation=cv2.INTER_NEAREST)
            rreg = cv2.resize(rreg, (nw, nh), interpolation=cv2.INTER_NEAREST)

        canvas = np.full((size, size, 3), np.random.randint(0, 256, 3), np.uint8)
        cmask = np.zeros((size, size), np.float32)
        oy = np.random.randint(0, size - nh + 1)
        ox = np.random.randint(0, size - nw + 1)
        reg3 = (rreg > 0)[..., None]
        canvas[oy:oy + nh, ox:ox + nw] = np.where(reg3, rimg, canvas[oy:oy + nh, ox:ox + nw])
        cmask[oy:oy + nh, ox:ox + nw] = np.where(rreg > 0, rmask, cmask[oy:oy + nh, ox:ox + nw])
        return canvas, cmask

    def _color_jitter(self, img):
        """밝기/대비/HSV 지터 (image만). photo<->web 색·조명 차이 흡수용."""
        a = 1.0 + np.random.uniform(-0.2, 0.2)        # contrast
        b = np.random.uniform(-20, 20)                # brightness
        img = np.clip(a * img.astype(np.float32) + b, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + np.random.uniform(-10, 10)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * (1 + np.random.uniform(-0.3, 0.3)), 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    def _augment(self, img, mask):
        # geometric (image+mask 동일 적용)
        if np.random.rand() < 0.5:
            img, mask = img[:, ::-1], mask[:, ::-1]
        if np.random.rand() < 0.5:
            img, mask = img[::-1], mask[::-1]
        k = np.random.randint(4)                      # rot90 x k (정사각 입력이라 형상 유지)
        if k:
            img, mask = np.rot90(img, k), np.rot90(mask, k)
        # photometric (image만)
        img = self._color_jitter(np.ascontiguousarray(img))
        return img, np.ascontiguousarray(mask)

    def __getitem__(self, idx):
        img_path, mask_path, label = self.samples[idx]
        img = self._load_image(img_path)      # native RGB
        mask = self._load_mask(mask_path)     # native {0,1}

        if self.augment and np.random.rand() < self.p_rawify:
            img, mask = self._rawify(img, mask)              # 폰 사진 흉내
        else:
            img = self._letterbox(img, (0, 0, 0), cv2.INTER_AREA)
            mask = self._letterbox(mask, 0, cv2.INTER_NEAREST)
        if self.augment:
            img, mask = self._augment(img, mask)             # flips/rot90/color

        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        img = torch.from_numpy(img.transpose(2, 0, 1))       # (3,H,W)
        mask = torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32))[None, ...]
        return img, mask, label
