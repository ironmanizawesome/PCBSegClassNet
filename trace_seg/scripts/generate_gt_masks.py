"""
6개 보드 일괄 GT mask 생성.

각 보드의 data/gerber/{board}/copper_{top,bottom}.gbr 를 읽어
data/masks/{board}/{front,back}.png 로 출력.

뒷면(bottom)은 사진과 방향 맞추기 위해 좌우 반전 (flip-horizontal).
사진 크기 매칭은 별도 단계 (image registration); 본 script는 원본 dpi로 출력.

사용:
    python scripts/generate_gt_masks.py --dpi 500
    python scripts/generate_gt_masks.py --board NANO_BOARD_A000005  # 한 보드만
"""
import argparse
import sys
from pathlib import Path

# trace_seg/src를 path에 추가하여 segmentation 모듈 import
_THIS = Path(__file__).resolve()
_TRACE_SEG = _THIS.parent.parent
sys.path.insert(0, str(_TRACE_SEG / "src"))

from segmentation.gerber_to_mask import (
    load_gerber, render_copper_layer, to_binary_mask,
    flip_horizontal, save_mask, print_layer_info,
)


BOARDS = [
    "LEONARDO_BOARD_A000057",
    "LEONARDO_BOARD_NH_A000052",
    "MEGA_R3_BOARD_2560_A000067",
    "NANO_BOARD_A000005",
    "UNO_R3_BOARD_A000066",
    "UNO_R3_BOARD_SMD_A000073",
]


def process_board(board: str, gerber_root: Path, mask_root: Path,
                  dpi: int, threshold: int) -> dict:
    """단일 보드의 front/back GT mask 생성. 통계 반환."""
    src = gerber_root / board
    dst = mask_root / board
    dst.mkdir(parents=True, exist_ok=True)

    stats = {"board": board}
    for side, gerber_name, out_name, do_flip in [
        ("front", "copper_top.gbr", "front.png", False),
        ("back",  "copper_bottom.gbr", "back.png", True),
    ]:
        gpath = src / gerber_name
        opath = dst / out_name
        print(f"\n=== {board} / {side} ===")
        print(f"  in : {gpath}")

        layer = load_gerber(str(gpath))
        print_layer_info(layer)

        rendered = render_copper_layer(layer, dpi=dpi)
        mask = to_binary_mask(rendered, threshold=threshold)
        if do_flip:
            mask = flip_horizontal(mask)

        save_mask(mask, str(opath))

        # 통계: 전체 픽셀, copper 픽셀, 비율
        import numpy as np
        total = int(mask.size)
        copper = int((mask > 127).sum())
        ratio = copper / total if total else 0.0
        stats[f"{side}_shape"] = f"{mask.shape[1]}x{mask.shape[0]}"
        stats[f"{side}_copper_ratio"] = round(ratio, 4)
        stats[f"{side}_copper_px"] = copper

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="6보드 일괄 GT mask 생성")
    parser.add_argument("--dpi", type=int, default=500,
                        help="렌더링 DPI (기본 500)")
    parser.add_argument("--threshold", type=int, default=130,
                        help="이진화 임계값 (THRESH_BINARY_INV)")
    parser.add_argument("--board", default=None,
                        help="단일 보드만 처리 (생략 시 6개 모두)")
    parser.add_argument("--gerber-root",
                        default=str(_TRACE_SEG / "data" / "gerber"),
                        help="Gerber 입력 root")
    parser.add_argument("--mask-root",
                        default=str(_TRACE_SEG / "data" / "masks"),
                        help="Mask 출력 root")
    args = parser.parse_args()

    gerber_root = Path(args.gerber_root)
    mask_root = Path(args.mask_root)

    targets = [args.board] if args.board else BOARDS
    all_stats = []
    for b in targets:
        try:
            s = process_board(b, gerber_root, mask_root,
                              dpi=args.dpi, threshold=args.threshold)
            all_stats.append(s)
        except Exception as e:
            print(f"\n[ERROR] {b}: {type(e).__name__}: {e}")
            all_stats.append({"board": b, "error": f"{type(e).__name__}: {e}"})

    # 요약
    print("\n" + "=" * 70)
    print("요약")
    print("=" * 70)
    for s in all_stats:
        if "error" in s:
            print(f"  {s['board']:42s}  ERROR: {s['error']}")
            continue
        print(f"  {s['board']:42s}  "
              f"F={s['front_shape']} cu={s['front_copper_ratio']*100:.1f}%  "
              f"B={s['back_shape']} cu={s['back_copper_ratio']*100:.1f}%")


if __name__ == "__main__":
    main()
