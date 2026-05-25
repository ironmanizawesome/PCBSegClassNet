# ------------------------------------------------------------------------
# Interactive board cropper for the Arduino dataset.
#
# Why: dataset_png/ photos frame the board at ~40% of the image with desk /
# tape rolls around it. Cropping to the board before CVAT annotation
# (a) makes small components easier to label, (b) reduces the domain gap
# vs. FPIC (which is tightly cropped), and (c) gives the segmentation
# model more useful pixels after the 512x512 resize.
#
# Usage (from repo root, in the pscn env):
#   python scripts/crop_boards.py -i dataset_png -o dataset_cropped
#
# Controls per image (cv2 window):
#   - drag a rectangle around the board
#   - ENTER / SPACE to confirm
#   - c to skip this image
#
# By default only Front/Back are processed (Side views are not useful for
# segmentation per project notes). Pass --views Front Back Side ... to
# override.
# ------------------------------------------------------------------------

import argparse
import sys
from pathlib import Path

import cv2

DEFAULT_VIEWS = ("Front", "Back")


def is_target(p: Path, views) -> bool:
    return any(p.stem.endswith(f"_{v}") for v in views)


def select_and_crop(img_path: Path, out_path: Path, max_display: int) -> bool:
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[skip] could not read {img_path}")
        return False

    h, w = img.shape[:2]
    # downscale only for display; we crop on the original resolution
    scale = min(1.0, max_display / max(h, w))
    disp = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1.0 else img

    win = f"crop: {img_path.name}  (drag, ENTER=ok, c=skip)"
    bbox = cv2.selectROI(win, disp, showCrosshair=False, fromCenter=False)
    cv2.destroyWindow(win)

    x, y, bw, bh = bbox
    if bw == 0 or bh == 0:
        print(f"[skip] {img_path.name}")
        return False

    # rescale back to original image coords
    inv = 1.0 / scale if scale < 1.0 else 1.0
    x, y = int(x * inv), int(y * inv)
    bw, bh = int(bw * inv), int(bh * inv)
    crop = img[y:y + bh, x:x + bw]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), crop)
    print(f"[ok]   {img_path.name}  ->  {crop.shape[1]}x{crop.shape[0]}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(prog="crop_boards")
    ap.add_argument("-i", "--input", required=True,
                    help="input dir tree (e.g. dataset_png)")
    ap.add_argument("-o", "--output", required=True,
                    help="output dir (input folder structure is mirrored)")
    ap.add_argument("--views", nargs="+", default=list(DEFAULT_VIEWS),
                    help="filename suffix(es) to include (default: Front Back)")
    ap.add_argument("--max-display", type=int, default=1200,
                    help="max width/height for the on-screen preview")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-crop even if the output already exists")
    args = ap.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    if not src.is_dir():
        sys.exit(f"input dir does not exist: {src}")

    candidates = sorted(p for p in src.rglob("*.png") if is_target(p, args.views))
    if not candidates:
        sys.exit(f"no images matching views {args.views} under {src}")

    print(f"Found {len(candidates)} image(s). views={args.views}")
    print("Controls: drag the board, ENTER=confirm, c=skip.\n")

    done = skipped = 0
    for i, f in enumerate(candidates, 1):
        rel = f.relative_to(src)
        out = dst / rel
        if out.exists() and not args.overwrite:
            print(f"[{i:>3}/{len(candidates)}] [exists] {rel}  (use --overwrite to redo)")
            skipped += 1
            continue
        print(f"[{i:>3}/{len(candidates)}] {rel}")
        if select_and_crop(f, out, args.max_display):
            done += 1
        else:
            skipped += 1

    print(f"\nDone. {done} cropped, {skipped} skipped -> {dst}")


if __name__ == "__main__":
    main()
