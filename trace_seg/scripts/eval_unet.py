"""
U-Net 정직 평가 — 학습 없이 체크포인트를 불러와 hold-out 보드에서 평가하고,
trivial baseline과 비교해 IoU 수치의 의미를 본다.

baseline:
  - all-foreground : pred 전부 1 (IoU = 전경 비율). "구리 다 칠하기".
  - mean-train-mask: train 보드 mask 평균을 0.5 임계 → 상수 예측. "평균 Arduino 배선 깔기".
모델 IoU가 이들과 비슷하면 모델이 prior 이상을 못 하는 것 → 수치 과장 신호.

주의: 현재 dataset은 입력을 정사각으로 stretch(가로세로비 왜곡)함. 이 평가도 동일
조건이라 baseline 대비 비교엔 유효하나, 절대 IoU는 letterbox 교체 후 다시 봐야 함.

사용 (trace_seg 디렉터리):
  python scripts/eval_unet.py --val-board NANO_BOARD_A000005
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_TRACE_SEG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TRACE_SEG / "src"))
from training.dataset import TraceDataset, list_samples   # noqa: E402
from models.unet import UNet                                # noqa: E402


def iou_np(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    union = np.logical_or(pred, gt).sum()
    return float(np.logical_and(pred, gt).sum() / union) if union else 1.0


def main():
    ap = argparse.ArgumentParser(description="U-Net 정직 평가 + baseline 비교")
    ap.add_argument("--ckpt", default=str(_TRACE_SEG / "data/unet/unet_best.pt"))
    ap.add_argument("--images", default=str(_TRACE_SEG / "data/images_registered"))
    ap.add_argument("--masks", default=str(_TRACE_SEG / "data/masks"))
    ap.add_argument("--val-board", required=True, help="평가할 hold-out BOARD_ID")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    alls = list_samples(args.images, args.masks)
    vb = set(args.val_board.split(","))         # 쉼표로 여러 보드(같은 설계)
    val_s = [s for s in alls if s[2].split("/")[0] in vb]
    train_s = [s for s in alls if s[2].split("/")[0] not in vb]
    if not val_s:
        print(f"val 샘플 없음: {args.val_board}")
        return

    device = torch.device(args.device)
    ck = torch.load(args.ckpt, map_location=device)
    size = ck.get("size", args.size)
    model = UNet(base=ck.get("base", 32)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    val_ds = TraceDataset(val_s, size=size, augment=False)
    train_ds = TraceDataset(train_s, size=size, augment=False)

    # mean-train-mask 상수 예측
    acc = np.zeros((size, size), np.float64)
    for i in range(len(train_ds)):
        acc += train_ds[i][1][0].numpy()
    const_pred = (acc / len(train_ds) > 0.5).astype(np.uint8)

    print(f"ckpt={Path(args.ckpt).name}  size={size}  val={len(val_s)}  "
          f"train(for baseline)={len(train_s)}\n")
    print(f"{'sample':40s} {'model':>7} {'all-fg':>7} {'mean-mask':>9}")
    m_i, a_i, c_i = [], [], []
    for i in range(len(val_ds)):
        img, mk, label = val_ds[i]
        gt = mk[0].numpy().astype(np.uint8)
        with torch.no_grad():
            pred = (torch.sigmoid(model(img[None].to(device)))[0, 0] > 0.5).cpu().numpy().astype(np.uint8)
        mi, ai, ci = iou_np(pred, gt), iou_np(np.ones_like(gt), gt), iou_np(const_pred, gt)
        m_i.append(mi); a_i.append(ai); c_i.append(ci)
        print(f"{label:40s} {mi*100:6.2f}% {ai*100:6.2f}% {ci*100:8.2f}%")

    print("-" * 68)
    print(f"{'MEAN':40s} {np.mean(m_i)*100:6.2f}% {np.mean(a_i)*100:6.2f}% "
          f"{np.mean(c_i)*100:8.2f}%")
    print(f"\n해석: model이 mean-mask와 비슷하면 prior 이상을 못 하는 것. "
          f"all-fg는 '전부 구리'(전경비율) 하한.")


if __name__ == "__main__":
    main()
