"""
사진 <-> GT mask 수동 정합(registration) 도구 — homography.

자동 보드검출이 헤더/배경 대비 문제로 불안정해서, 사람이 사진과 GT mask에서
공통 랜드마크(마운팅홀, QFP 코너, 커넥터 패드 등)를 ≥4점 클릭해 homography를
구한다. 사진을 mask 픽셀 좌표계로 warp해서 정합된 사진을 저장한다.

대상 쌍은 data/images/{BOARD}/{side}_{source}.* 와 data/masks/{BOARD}/{side}.png.
결과:
  data/images_registered/{BOARD}/{side}_{source}.png        정합된 사진 (mask 해상도)
  data/images_registered/{BOARD}/{side}_{source}.points.json 클릭 좌표 + H (재현/수정용)
  data/images_registered/{BOARD}/{side}_{source}.overlay.png QA용 (red=사진, green=GT)

조작 (로컬 실행, cv2 창 2개):
  - 사진 창에서 1점 클릭 -> mask 창에서 대응 1점 클릭 -> 반복 (번갈아, 같은 순서)
  - u: 마지막 점 취소   r: 전부 리셋   s: 저장(각 ≥4점)   n: 이 쌍 건너뜀   q/ESC: 종료

사용:
  python scripts/register_photo.py --list                 # 대상 쌍 목록
  python scripts/register_photo.py --selftest             # GUI 없이 homography 로직 점검
  python scripts/register_photo.py --source photo         # photo 쌍 전부 순회 정합
  python scripts/register_photo.py --board LEONARDO_BOARD_A000057 --side front --source photo
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_TRACE_SEG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TRACE_SEG / "src"))
from training.dataset import list_samples  # noqa: E402

OUT_DIR = _TRACE_SEG / "data/images_registered"
MAX_DISP = 900   # 화면 표시용 최대 변 길이


class Picker:
    """이미지 1장 + 클릭 점(원본 좌표) 관리. 표시 스케일 보정."""

    def __init__(self, img):
        self.img = img
        h, w = img.shape[:2]
        self.scale = min(1.0, MAX_DISP / max(h, w))
        self.disp_size = (int(w * self.scale), int(h * self.scale))
        self.points = []                       # 원본 좌표 (x, y)

    def add_display_click(self, dx, dy):
        self.points.append((dx / self.scale, dy / self.scale))

    def render(self, status=""):
        vis = cv2.resize(self.img, self.disp_size)
        for i, (x, y) in enumerate(self.points):
            px, py = int(x * self.scale), int(y * self.scale)
            cv2.circle(vis, (px, py), 5, (0, 0, 255), -1)
            cv2.putText(vis, str(i + 1), (px + 6, py - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if status:
            cv2.putText(vis, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)
        return vis


def compute_homography(src_pts, dst_pts):
    src = np.array(src_pts, dtype=np.float64)
    dst = np.array(dst_pts, dtype=np.float64)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return H


def save_result(photo, mask, H, pp, mp, out_prefix: Path):
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    mh, mw = mask.shape[:2]
    warped = cv2.warpPerspective(photo, H, (mw, mh))
    cv2.imwrite(str(out_prefix.with_suffix(".png")), warped)

    # QA overlay: red=warped photo(gray), green=GT mask
    overlay = np.zeros((mh, mw, 3), np.uint8)
    overlay[..., 2] = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    overlay[..., 1] = mask
    cv2.imwrite(str(out_prefix) + ".overlay.png", overlay)

    with open(str(out_prefix) + ".points.json", "w", encoding="utf-8") as f:
        json.dump({"src_pts": pp.points, "dst_pts": mp.points, "H": H.tolist()},
                  f, indent=2)
    print(f"[saved] {out_prefix.name}.png  (+overlay, +points.json)")


def register_pair(photo_path, mask_path, label, out_prefix):
    photo = cv2.imread(str(photo_path), cv2.IMREAD_COLOR)
    mask_g = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if photo is None or mask_g is None:
        print(f"[skip] load fail: {label}")
        return "skip"
    mask_bgr = cv2.cvtColor(mask_g, cv2.COLOR_GRAY2BGR)
    pp, mp = Picker(photo), Picker(mask_bgr)

    win_p, win_m = f"PHOTO  {label}", f"MASK  {label}"
    cv2.namedWindow(win_p)
    cv2.namedWindow(win_m)
    cv2.moveWindow(win_p, 20, 60)
    cv2.moveWindow(win_m, 40 + pp.disp_size[0], 60)

    def on_photo(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and len(pp.points) == len(mp.points):
            pp.add_display_click(x, y)

    def on_mask(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and len(mp.points) == len(pp.points) - 1:
            mp.add_display_click(x, y)

    cv2.setMouseCallback(win_p, on_photo)
    cv2.setMouseCallback(win_m, on_mask)

    while True:
        nxt = "PHOTO" if len(pp.points) == len(mp.points) else "MASK"
        n = min(len(pp.points), len(mp.points))
        cv2.imshow(win_p, pp.render(f"click {nxt} pt {len(pp.points)+ (nxt=='PHOTO')}  pairs={n}  [u]ndo [r]eset [s]ave [n]ext [q]uit"))
        cv2.imshow(win_m, mp.render(f"pairs={n} (need >=4)"))
        k = cv2.waitKey(20) & 0xFF
        if k in (ord("q"), 27):
            cv2.destroyWindow(win_p)
            cv2.destroyWindow(win_m)
            return "quit"
        if k == ord("n"):
            cv2.destroyWindow(win_p)
            cv2.destroyWindow(win_m)
            return "skip"
        if k == ord("u"):
            if len(pp.points) > len(mp.points):
                pp.points.pop()
            elif mp.points:
                mp.points.pop()
        if k == ord("r"):
            pp.points.clear()
            mp.points.clear()
        if k == ord("s"):
            n = min(len(pp.points), len(mp.points))
            if n < 4:
                print(f"[need] >=4 대응점 (현재 {n})")
                continue
            H = compute_homography(pp.points[:n], mp.points[:n])
            if H is None:
                print("[err] homography 계산 실패 (점 분포 확인)")
                continue
            save_result(photo, mask_g, H, pp, mp, out_prefix)
            cv2.destroyWindow(win_p)
            cv2.destroyWindow(win_m)
            return "saved"


def selftest():
    """GUI 없이 homography+warp 로직 점검 (알려진 변환 복원)."""
    photo = np.zeros((400, 600, 3), np.uint8)
    mask = np.zeros((300, 400), np.uint8)
    src = [(50, 40), (550, 60), (560, 360), (40, 350)]
    dst = [(20, 20), (380, 25), (370, 280), (25, 275)]
    H = compute_homography(src, dst)
    assert H is not None, "homography None"
    warped = cv2.warpPerspective(photo, H, (400, 300))
    assert warped.shape[:2] == (300, 400)
    proj = cv2.perspectiveTransform(np.array([src], np.float64), H)[0]
    err = np.abs(proj - np.array(dst)).max()
    print(f"[selftest] OK  max reproj err = {err:.3f} px  (warp shape {warped.shape})")


def main():
    ap = argparse.ArgumentParser(description="사진<->GT mask 수동 정합 (homography)")
    ap.add_argument("--images", default=str(_TRACE_SEG / "data/images"))
    ap.add_argument("--masks", default=str(_TRACE_SEG / "data/masks"))
    ap.add_argument("--board", default=None)
    ap.add_argument("--side", choices=["front", "back"], default=None)
    ap.add_argument("--source", choices=["photo", "web", "both"], default="photo")
    ap.add_argument("--list", action="store_true", help="대상 쌍만 출력")
    ap.add_argument("--selftest", action="store_true", help="GUI 없이 로직 점검")
    ap.add_argument("--overwrite", action="store_true", help="이미 정합된 쌍도 다시")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    sources = ("photo", "web") if args.source == "both" else (args.source,)
    samples = list_samples(args.images, args.masks, sources=sources, board=args.board)
    if args.side:
        samples = [s for s in samples if f"/{args.side}/" in s[2] + "/"]
    if not samples:
        print("대상 쌍 없음.")
        return

    def prefix_for(label):
        board, side, source = label.split("/")
        return OUT_DIR / board / f"{side}_{source}"

    if args.list:
        print(f"대상 {len(samples)} 쌍:")
        for _, _, label in samples:
            done = prefix_for(label).with_suffix(".png").exists()
            print(f"  [{'done' if done else '   '}] {label}")
        return

    saved = skipped = 0
    for photo_path, mask_path, label in samples:
        out_prefix = prefix_for(label)
        if out_prefix.with_suffix(".png").exists() and not args.overwrite:
            print(f"[have] {label} (이미 정합됨, --overwrite로 재작업)")
            continue
        print(f"\n=== {label} ===  사진+mask 공통 랜드마크 ≥4점 클릭")
        result = register_pair(photo_path, mask_path, label, out_prefix)
        if result == "quit":
            break
        saved += result == "saved"
        skipped += result == "skip"
    cv2.destroyAllWindows()
    print(f"\n정합 저장 {saved} / 건너뜀 {skipped} -> {OUT_DIR}")


if __name__ == "__main__":
    main()
