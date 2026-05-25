# ------------------------------------------------------------------------
# Convert a CVAT-exported COCO JSON into FPIC-style per-image CSV files.
#
# FPIC create_mask.py expects, per image, one CSV with columns:
#   Image File, Vertices, Designator
# where Vertices is the string form of a 3-level-nested list:
#   "[[[x1, y1], [x2, y2], ...]]"
# (the outer list lets a single annotation hold multiple disjoint polygons;
# we emit one row per polygon for simplicity).
#
# Usage (from repo root):
#   python scripts/coco_to_fpic_csv.py \
#       -i PCBannotations/NANO33_BLE_REV2_v1/instances_default.json \
#       -o data/annotations
#
# One CSV is written per image, named after the image stem (e.g.
# NANO33_BLE_REV2_BOARD_ABX00071_Front.csv).
# ------------------------------------------------------------------------

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def convert(coco_path: Path, out_dir: Path) -> None:
    coco = json.loads(coco_path.read_text(encoding="utf-8"))

    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    img_name = {i["id"]: i["file_name"] for i in coco["images"]}

    per_image = defaultdict(list)
    for ann in coco["annotations"]:
        per_image[ann["image_id"]].append(ann)

    out_dir.mkdir(parents=True, exist_ok=True)

    n_csv = 0
    n_poly = 0
    n_skipped = 0
    for img_id, anns in per_image.items():
        if img_id not in img_name:
            print(f"[warn] annotation references missing image_id={img_id}, skip")
            continue
        rows = []
        for ann in anns:
            cls = cat_name.get(ann["category_id"])
            if cls is None:
                n_skipped += 1
                continue
            seg = ann.get("segmentation")
            # RLE (iscrowd) not supported — polygon export only
            if not isinstance(seg, list):
                n_skipped += 1
                continue
            for poly_flat in seg:
                if len(poly_flat) < 6:  # < 3 points
                    n_skipped += 1
                    continue
                # COCO: [x1,y1,x2,y2,...] flat  -> FPIC: [[x1,y1],[x2,y2],...]
                # cv2.fillPoly needs int32, so round here once.
                pts = [
                    [int(round(poly_flat[i])), int(round(poly_flat[i + 1]))]
                    for i in range(0, len(poly_flat) - 1, 2)
                ]
                vertices = str([pts])  # outer list wrapper -> 3-level nested
                rows.append(
                    {
                        "Image File": img_name[img_id],
                        "Vertices": vertices,
                        "Designator": cls,
                    }
                )
                n_poly += 1

        if not rows:
            print(f"[warn] no usable polygons for {img_name[img_id]}, skip")
            continue

        csv_path = out_dir / (Path(img_name[img_id]).stem + ".csv")
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f, fieldnames=["Image File", "Vertices", "Designator"]
            )
            w.writeheader()
            w.writerows(rows)
        n_csv += 1
        print(f"[ok]   {csv_path.name}  ({len(rows)} polygon(s))")

    print(
        f"\nDone. {n_csv} CSV(s), {n_poly} polygon(s), "
        f"{n_skipped} skipped -> {out_dir}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(prog="coco_to_fpic_csv")
    ap.add_argument(
        "-i",
        "--input",
        required=True,
        help="CVAT COCO JSON file, OR a directory to scan recursively "
        "for instances_default.json",
    )
    ap.add_argument(
        "-o",
        "--output",
        required=True,
        help="output directory for per-image CSVs",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    if in_path.is_file():
        json_paths = [in_path]
    elif in_path.is_dir():
        json_paths = sorted(in_path.rglob("instances_default.json"))
        if not json_paths:
            sys.exit(f"no instances_default.json under {in_path}")
        print(f"Found {len(json_paths)} JSON file(s).")
    else:
        sys.exit(f"input path does not exist: {in_path}")

    out_dir = Path(args.output)
    for jp in json_paths:
        print(f"\n=== {jp.parent.name} ===")
        convert(jp, out_dir)


if __name__ == "__main__":
    main()
