# ------------------------------------------------------------------------
# Build an FPIC-style segmentation dataset for the Arduino boards from
# CVAT-derived per-image CSVs.
#
# This is a trimmed clone of src/data/create_mask.py + create_patches.py:
#   - same HLS + CLAHE image processing
#   - same RGB mask colors (FPIC color_values, channel handling included)
#   - same 768px patching with edge-overlap to cover boundaries
#   - same 80/20 train/val split
#   - **classification crops + super-resolution removed** (we only fine-tune
#     PCBSegNet, not PCBClassNet, so the EDSR weight isn't needed)
#
# Usage (from repo root):
#   python scripts/build_seg_dataset.py \
#       -a data/annotations \
#       -i dataset_cropped \
#       -o data/segmentation_arduino \
#       -ps 768
# ------------------------------------------------------------------------

import argparse
import ast
import csv
import sys
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Must match src/data/create_mask.py exactly so that mask values line up
# with src/data/dataloader.py's one-hot lookup against color_values.
COLOR_VALUES = {
    "R": (255, 0, 0),     "C": (255, 255, 0), "U": (0, 234, 255),
    "Q": (170, 0, 255),   "J": (255, 127, 0), "L": (191, 255, 0),
    "RA": (0, 149, 255),  "D": (106, 255, 0), "RN": (0, 64, 255),
    "TP": (237, 185, 185),"IC": (185, 215, 237),"P": (231, 233, 185),
    "CR": (220, 185, 237),"M": (185, 237, 224),"BTN": (143, 35, 35),
    "FB": (35, 98, 143),  "CRA": (143, 106, 35),"SW": (107, 35, 143),
    "T": (79, 143, 35),   "F": (115, 115, 115),"V": (204, 204, 204),
    "LED": (245, 130, 48),"S": (220, 190, 255),"QA": (170, 255, 195),
    "JP": (255, 250, 200),
}

CLAHE = A.Compose(
    [A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0)]
)


def find_image(images_root: Path, filename: str) -> Path | None:
    hits = list(images_root.rglob(filename))
    return hits[0] if hits else None


def build_full(csv_path: Path, images_root: Path):
    """Match create_mask.py's per-image pipeline.

    Returns (img_hls_clahe, mask_bgr) numpy arrays whose channel layout is
    *intentionally* the FPIC one: img is HLS-valued in BGR-shaped numpy and
    mask was filled by cv2.fillPoly with RGB tuples (so channel 0 holds the
    R value, etc.). The final BGR->RGB swap happens at save time.
    """
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    if not rows:
        return None
    img_name = rows[0]["Image File"]
    img_path = find_image(images_root, img_name)
    if img_path is None:
        print(f"[skip] image not found: {img_name}")
        return None

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[skip] cv2.imread failed: {img_path}")
        return None

    img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    mask = np.zeros_like(img)
    img_hls = CLAHE(image=img_hls, mask=mask)["image"]

    for r in rows:
        cat = r["Designator"]
        if cat not in COLOR_VALUES:
            continue
        try:
            pts = (
                np.array(ast.literal_eval(r["Vertices"]))[0]
                .reshape((-1, 1, 2))
                .astype(np.int32)
            )
        except Exception:
            continue
        color = COLOR_VALUES[cat]
        cv2.polylines(mask, [pts], True, color, 2)
        cv2.fillPoly(mask, [pts], color=color)

    return img_hls, mask


def iter_patches(img: np.ndarray, mask: np.ndarray, ps: int):
    """Yield (img_patch, mask_patch) covering the whole image; edge tiles
    are shifted inward to keep them full-size (matches create_patches.py)."""
    h, w = img.shape[:2]
    if h < ps or w < ps:
        return
    for j in range(0, h, ps):
        for k in range(0, w, ps):
            jj, kk = j, k
            if jj + ps > h:
                jj = h - ps
            if kk + ps > w:
                kk = w - ps
            yield (img[jj : jj + ps, kk : kk + ps],
                   mask[jj : jj + ps, kk : kk + ps])


def save_patch(out_img: Path, out_mask: Path, p_img: np.ndarray, p_mask: np.ndarray):
    # FPIC convention: at save time, swap channels so the PNG on disk has
    # R=ch2, G=ch1, B=ch0 (see create_mask.py). The dataloader reads with
    # tf.image.decode_png (RGB) and recovers the original channel order.
    cv2.imwrite(str(out_img), cv2.cvtColor(p_img, cv2.COLOR_BGR2RGB))
    cv2.imwrite(str(out_mask), cv2.cvtColor(p_mask, cv2.COLOR_BGR2RGB))


def main() -> None:
    ap = argparse.ArgumentParser(prog="build_seg_dataset")
    ap.add_argument("-a", "--annotations", required=True,
                    help="dir of per-image CSVs (coco_to_fpic_csv.py output)")
    ap.add_argument("-i", "--images", required=True,
                    help="root dir to search recursively for source images")
    ap.add_argument("-o", "--output", required=True,
                    help="output root; creates {train,val}/{images,masks}/")
    ap.add_argument("-ps", "--patch-size", type=int, default=768)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    csv_dir = Path(args.annotations)
    img_root = Path(args.images)
    out = Path(args.output)

    csv_files = sorted(csv_dir.glob("*.csv"))
    if not csv_files:
        sys.exit(f"no CSVs in {csv_dir}")

    patches = []
    print(f"Processing {len(csv_files)} CSV(s)...")
    for csv_path in tqdm(csv_files):
        result = build_full(csv_path, img_root)
        if result is None:
            continue
        img, mask = result
        for p_img, p_mask in iter_patches(img, mask, args.patch_size):
            patches.append((p_img, p_mask))

    if not patches:
        sys.exit("no patches generated (image smaller than patch size?)")

    print(f"Total patches: {len(patches)}")

    idx = np.arange(len(patches))
    train_idx, val_idx = train_test_split(
        idx, test_size=args.val_frac, random_state=args.seed
    )

    for split, ids in [("train", train_idx), ("val", val_idx)]:
        img_dir = out / split / "images"
        msk_dir = out / split / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        msk_dir.mkdir(parents=True, exist_ok=True)
        for i, j in enumerate(ids):
            p_img, p_mask = patches[j]
            save_patch(
                img_dir / f"image_{i}.png",
                msk_dir / f"image_{i}.png",
                p_img,
                p_mask,
            )

    print(f"Wrote {len(train_idx)} train + {len(val_idx)} val -> {out}")


if __name__ == "__main__":
    main()
