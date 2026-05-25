# ------------------------------------------------------------------------
# HEIC -> PNG converter for the Arduino board dataset.
#
# The training pipeline (create_mask.py / dataloader.py) only reads PNG via
# cv2 / tf.image.decode_png. iPhone photos come as HEIC, which cv2 cannot
# open, so convert everything up front.
#
# Usage (from repo root, in the pscn env):
#   pip install pillow-heif        # one-time
#   python scripts/convert_heic.py -i dataset -o dataset_png
#
# Mirrors the input folder structure into the output directory:
#   dataset/UNO_R3_BOARD_A000066/UNO_R3_BOARD_A000066_Front.HEIC
#     -> dataset_png/UNO_R3_BOARD_A000066/UNO_R3_BOARD_A000066_Front.png
# ------------------------------------------------------------------------

import argparse
from pathlib import Path

from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()


def convert_tree(src_dir: str, dst_dir: str) -> None:
    src = Path(src_dir)
    dst = Path(dst_dir)

    heics = sorted(
        p for p in src.rglob("*") if p.suffix.lower() == ".heic"
    )
    if not heics:
        print(f"No .HEIC files found under {src}")
        return

    ok, failed = 0, 0
    for f in heics:
        rel = f.relative_to(src).with_suffix(".png")
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            # convert("RGB") drops alpha/orientation metadata so cv2 reads
            # it back consistently
            Image.open(f).convert("RGB").save(out, "PNG")
            ok += 1
            print(f"[ok]  {rel}")
        except Exception as e:  # noqa: BLE001 - report and continue
            failed += 1
            print(f"[ERR] {rel}: {e.__class__.__name__}: {e}")

    print(f"\nDone. {ok} converted, {failed} failed -> {dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="convert_heic")
    parser.add_argument(
        "-i", "--input", required=True,
        help="Input directory tree containing .HEIC files",
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Output directory (input folder structure is mirrored)",
    )
    args = parser.parse_args()
    convert_tree(args.input, args.output)
