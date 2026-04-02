#!/usr/bin/env python3
"""
세그멘테이션 결과 정량 평가 — Stage 2

예측 마스크(Step 1 출력)와 Ground Truth 마스크(Step 2 Gerber 변환 결과)를
픽셀 단위로 비교하여 IoU, DICE, Precision, Recall, F1을 계산한다.

실행 예시:
    python src/evaluation/evaluate.py \
        --pred data/masks/leonardo_front_trace.png \
        --gt   data/masks/gt_front.png \
        --output data/evaluation/result_front.json
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# 마스크 로드
# ---------------------------------------------------------------------------

def load_mask(path: str) -> np.ndarray:
    """
    이진 마스크 로드.
    그레이스케일로 읽은 뒤 threshold로 이진화 (0 또는 255).
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"마스크 파일을 열 수 없음: {path}")
    _, binary = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
    return binary.astype(np.uint8)


def align_masks(pred: np.ndarray, gt: np.ndarray) -> tuple:
    """
    두 마스크의 크기가 다르면 gt를 pred 크기로 리사이즈.
    크기가 많이 다를 경우 경고 출력.
    """
    if pred.shape == gt.shape:
        return pred, gt

    ph, pw = pred.shape
    gh, gw = gt.shape
    print(f"[경고] 마스크 크기 불일치: pred={pw}x{ph}, gt={gw}x{gh} → gt를 pred 크기로 리사이즈")
    gt_resized = cv2.resize(gt.astype(np.float32), (pw, ph),
                             interpolation=cv2.INTER_NEAREST)
    return pred, (gt_resized > 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------
# 평가 지표 계산
# ---------------------------------------------------------------------------

def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """
    픽셀 단위 이진 세그멘테이션 지표 계산.

    Args:
        pred: 예측 마스크 (0 또는 1)
        gt:   GT 마스크   (0 또는 1)

    Returns:
        {
          "iou":        Intersection over Union (Jaccard Index)
          "dice":       DICE Coefficient (F1 of pixel-level)
          "precision":  TP / (TP + FP)
          "recall":     TP / (TP + FN)
          "f1":         2 * precision * recall / (precision + recall)
          "pixel_acc":  (TP + TN) / 전체 픽셀
          "tp": int, "fp": int, "fn": int, "tn": int
        }
    """
    pred_flat = pred.flatten().astype(bool)
    gt_flat   = gt.flatten().astype(bool)

    tp = int(np.logical_and(pred_flat,  gt_flat).sum())
    fp = int(np.logical_and(pred_flat,  ~gt_flat).sum())
    fn = int(np.logical_and(~pred_flat, gt_flat).sum())
    tn = int(np.logical_and(~pred_flat, ~gt_flat).sum())
    total = tp + fp + fn + tn

    iou       = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    dice      = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    pixel_acc = (tp + tn) / total if total > 0 else 0.0

    return {
        "iou":       round(iou,       4),
        "dice":      round(dice,      4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "pixel_acc": round(pixel_acc, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "total_pixels": total,
    }


# ---------------------------------------------------------------------------
# 오류 시각화
# ---------------------------------------------------------------------------

def build_error_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    TP/FP/FN/TN을 색상으로 시각화한 BGR 이미지 생성.
      - 흰색 (255,255,255): TP  (정확히 검출된 트레이스)
      - 빨간색 (0,0,255):   FP  (잘못 검출된 영역)
      - 파란색 (255,0,0):   FN  (놓친 트레이스)
      - 검정  (0,0,0):      TN  (배경)
    """
    h, w = pred.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)

    tp_mask = (pred == 1) & (gt == 1)
    fp_mask = (pred == 1) & (gt == 0)
    fn_mask = (pred == 0) & (gt == 1)

    vis[tp_mask] = (255, 255, 255)  # TP: 흰색
    vis[fp_mask] = (0,   0,   255)  # FP: 빨간색
    vis[fn_mask] = (255, 0,   0  )  # FN: 파란색

    return vis


# ---------------------------------------------------------------------------
# 출력
# ---------------------------------------------------------------------------

def print_metrics(metrics: dict, pred_path: str, gt_path: str) -> None:
    """평가 결과를 콘솔에 보기 좋게 출력."""
    print("\n" + "=" * 50)
    print(f"  예측:  {pred_path}")
    print(f"  GT:    {gt_path}")
    print("=" * 50)
    print(f"  IoU        : {metrics['iou']:.4f}  ({metrics['iou']*100:.2f}%)")
    print(f"  DICE       : {metrics['dice']:.4f}  ({metrics['dice']*100:.2f}%)")
    print(f"  Precision  : {metrics['precision']:.4f}")
    print(f"  Recall     : {metrics['recall']:.4f}")
    print(f"  F1         : {metrics['f1']:.4f}")
    print(f"  Pixel Acc  : {metrics['pixel_acc']:.4f}  ({metrics['pixel_acc']*100:.2f}%)")
    print("-" * 50)
    print(f"  TP={metrics['tp']:,}  FP={metrics['fp']:,}  FN={metrics['fn']:,}  TN={metrics['tn']:,}")
    print("=" * 50 + "\n")


def save_results(metrics: dict, output_path: str,
                 pred_path: str, gt_path: str) -> None:
    """평가 결과를 JSON으로 저장."""
    result = {
        "pred": pred_path,
        "gt":   gt_path,
        "metrics": metrics,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"결과 저장: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="세그멘테이션 결과 정량 평가 (IoU, DICE, Precision, Recall)"
    )
    parser.add_argument("--pred",   required=True,
                        help="예측 이진 마스크 경로")
    parser.add_argument("--gt",     required=True,
                        help="Ground Truth 이진 마스크 경로")
    parser.add_argument("--output", default=None,
                        help="결과 JSON 저장 경로 (생략 시 콘솔만 출력)")
    parser.add_argument("--error-map", default=None,
                        help="TP/FP/FN 오류 맵 이미지 저장 경로 (생략 시 저장 안 함)")
    args = parser.parse_args()

    pred = load_mask(args.pred)
    gt   = load_mask(args.gt)
    pred, gt = align_masks(pred, gt)

    metrics = compute_metrics(pred, gt)
    print_metrics(metrics, args.pred, args.gt)

    if args.output:
        save_results(metrics, args.output, args.pred, args.gt)

    if args.error_map:
        error_img = build_error_map(pred, gt)
        Path(args.error_map).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.error_map, error_img)
        print(f"오류 맵 저장: {args.error_map}")
        print("  흰색=TP  빨간=FP  파란=FN  검정=TN")


if __name__ == "__main__":
    main()
