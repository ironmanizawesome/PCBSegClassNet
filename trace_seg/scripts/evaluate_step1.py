"""
Step 1 (HSV trace segmentation) 평가 batch.

사진 1장 + 짝지어진 GT mask 1장으로 Step 1을 돌려 예측 mask 생성하고
IoU/DICE/F1/error map 산출.

data/images/{BOARD}/{side}_{photo,web}.* 를 자동 탐색해 GT와 페어링한다.
사진을 organize_photos.py로 정리/추가하면 코드 수정 없이 자동 반영.

사용:
    python scripts/evaluate_step1.py
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_TRACE_SEG = _THIS.parent.parent
sys.path.insert(0, str(_TRACE_SEG / "src"))

from segmentation.trace_segment import load_image, segment_traces, save_mask
from evaluation.evaluate import (
    load_mask, align_masks, compute_metrics, build_error_map,
    print_metrics, save_results,
)
import cv2


SIDES = ("front", "back")
SOURCES = ("photo", "web")   # photo = 직접촬영, web = roboflow


def discover_pairs():
    """data/images/{BOARD}/{side}_{source}.* 와 data/masks/{BOARD}/{side}.png 자동 페어링.

    사진을 추가/정리(organize_photos.py)하면 PAIRS 수정 없이 자동 반영.
    반환: (photo, gt_mask, output_prefix, label) 튜플 리스트.
    """
    images = _TRACE_SEG / "data/images"
    masks = _TRACE_SEG / "data/masks"
    pairs = []
    if not images.exists():
        return pairs
    for board_dir in sorted(p for p in images.iterdir() if p.is_dir()):
        board = board_dir.name
        for side in SIDES:
            gt = masks / board / f"{side}.png"
            if not gt.exists():
                continue
            for source in SOURCES:
                hits = sorted(board_dir.glob(f"{side}_{source}.*"))
                if not hits:
                    continue
                prefix = _TRACE_SEG / f"data/step1_eval/{board}/{side}_{source}"
                pairs.append((hits[0], gt, prefix, f"{board} {side}/{source}"))
    return pairs


def run_one(photo: Path, gt_path: Path, out_prefix: Path, label: str) -> dict:
    print(f"\n{'#' * 60}\n# {label}\n{'#' * 60}")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    pred_path  = out_prefix.with_name(out_prefix.name + "_pred.png")
    error_path = out_prefix.with_name(out_prefix.name + "_error.png")
    json_path  = out_prefix.with_name(out_prefix.name + ".json")

    # 1) Step 1: photo -> predicted mask
    print(f"[Step 1] {photo.name} -> pred mask")
    img = load_image(str(photo))
    pred_mask = segment_traces(img, use_subsurface=True)
    save_mask(pred_mask, str(pred_path))

    # 2) Eval: pred vs GT (auto-align gt to pred shape)
    pred_arr = load_mask(str(pred_path))
    gt_arr   = load_mask(str(gt_path))
    pred_arr, gt_arr = align_masks(pred_arr, gt_arr)
    metrics = compute_metrics(pred_arr, gt_arr)
    print_metrics(metrics, str(pred_path), str(gt_path))

    # 3) Save error map + JSON
    err = build_error_map(pred_arr, gt_arr)
    cv2.imwrite(str(error_path), err)
    save_results(metrics, str(json_path), str(pred_path), str(gt_path))

    return {"label": label, "metrics": metrics,
            "pred": str(pred_path), "gt": str(gt_path),
            "error_map": str(error_path), "json": str(json_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1 evaluation batch")
    args = parser.parse_args()

    pairs = discover_pairs()
    if not pairs:
        print("평가할 (사진, GT) 쌍 없음. data/images/{BOARD}/{side}_{photo,web}.* 확인.")
        return

    results = []
    for photo, gt_path, prefix, label in pairs:
        if not photo.exists():
            print(f"SKIP {label}: photo not found ({photo})")
            continue
        if not gt_path.exists():
            print(f"SKIP {label}: gt not found ({gt_path})")
            continue
        try:
            r = run_one(photo, gt_path, prefix, label)
            results.append(r)
        except Exception as e:
            print(f"ERROR {label}: {type(e).__name__}: {e}")
            results.append({"label": label, "error": str(e)})

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in results:
        if "error" in r:
            print(f"  {r['label']:25s}  ERROR: {r['error']}")
            continue
        m = r["metrics"]
        print(f"  {r['label']:25s}  IoU={m['iou']*100:5.2f}%  "
              f"DICE={m['dice']*100:5.2f}%  P={m['precision']*100:5.2f}%  "
              f"R={m['recall']*100:5.2f}%  F1={m['f1']*100:5.2f}%")

    # Save combined summary
    summary_path = _TRACE_SEG / "data/step1_eval/summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
