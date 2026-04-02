#!/usr/bin/env python3
"""
PCB 배선(trace) 세그멘테이션 — Stage 2, Step 1 (개선)
전통적 이미지 처리: CLAHE + HSV 색상 필터링 + morphological operation

대상 보드: Arduino 계열 (녹색 솔더마스크, 금색/은색 구리 패드)

개선 이력:
  v1: 기본 HSV 필터링 + morphological
  v2: CLAHE 전처리, 보드 마스크, 솔더마스크 아래 트레이스 검출, 면적 필터 추가

실행 예시:
    python src/segmentation/trace_segment.py \
        --input data/images/leonardo_front.webp \
        --output data/masks/leonardo_front_trace.png \
        --overlay data/masks/leonardo_front_overlay.png \
        --debug-dir data/masks/debug_front
"""

import argparse
import cv2
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# HSV 범위 상수 (OpenCV 기준: H=[0,180], S=[0,255], V=[0,255])
# ---------------------------------------------------------------------------

# 구리/금색 노출 패드 (ENIG 마감, 채도 높은 경우): 노란색-주황색
COPPER_GOLD_LOWER = np.array([10,  60, 120])
COPPER_GOLD_UPPER = np.array([30, 255, 255])

# 무광 금속 패드 (ENIG/HASL, 사진에서 채도 낮고 밝게 찍히는 경우)
# 실측 샘플: S≈23, V≈234. 실크스크린(S≈5, V>215)과 S 값으로 구분
COPPER_METAL_LOWER = np.array([ 0,  12, 130])
COPPER_METAL_UPPER = np.array([180, 70, 240])
# 흰색 실크스크린 제외 임계값 (S < 12 AND V > 215)
SILKSCREEN_S_MAX = 12
SILKSCREEN_V_MIN = 215

# 녹색 솔더마스크 (주요 배경)
SOLDERMASK_LOWER = np.array([35, 40, 30])
SOLDERMASK_UPPER = np.array([85, 255, 255])

# 구리 위 솔더마스크 (올리브/따뜻한 녹색) — 일반 솔더마스크보다 H 낮음
SUBSURFACE_TRACE_LOWER = np.array([28, 30,  40])
SUBSURFACE_TRACE_UPPER = np.array([40, 130, 170])

# 사진 배경 (균일한 회색): 채도 낮고 적당한 밝기
BACKGROUND_LOWER = np.array([ 0,  0, 100])
BACKGROUND_UPPER = np.array([180, 25, 230])


# ---------------------------------------------------------------------------
# 이미지 로드
# ---------------------------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    """
    이미지 로드. cv2.imread 우선, webp 실패 시 PIL fallback.
    반환: BGR 형식 np.ndarray
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        from PIL import Image
        pil_img = Image.open(path).convert("RGB")
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    if img is None:
        raise FileNotFoundError(f"이미지를 열 수 없음: {path}")
    return img


# ---------------------------------------------------------------------------
# 전처리
# ---------------------------------------------------------------------------

def enhance_contrast(img_bgr: np.ndarray,
                     clip_limit: float = 2.0,
                     tile_size: int = 8) -> np.ndarray:
    """
    CLAHE(Contrast Limited Adaptive Histogram Equalization)로 로컬 대비 향상.
    LAB 색공간의 L(밝기) 채널에만 적용해 색조 왜곡을 최소화한다.
    솔더마스크 아래 구리 트레이스의 미세한 색상 차이를 증폭하는 데 효과적.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                             tileGridSize=(tile_size, tile_size))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# 마스크 생성
# ---------------------------------------------------------------------------

def build_board_mask(hsv: np.ndarray) -> np.ndarray:
    """
    PCB 보드 영역 마스크 생성 (사진 배경 제외).
    배경: 채도 낮고 중간~밝은 균일한 회색 → 이를 제외한 영역 = 보드.
    """
    bg_mask = cv2.inRange(hsv, BACKGROUND_LOWER, BACKGROUND_UPPER)

    # 반전해서 보드 후보 영역 얻기
    board_mask = cv2.bitwise_not(bg_mask)

    # MORPH_CLOSE로 보드 내부 작은 구멍(부품 실장홀 등) 채움
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    board_mask = cv2.morphologyEx(board_mask, cv2.MORPH_CLOSE, kernel)

    # 가장 큰 연결 요소만 유지 (= 보드 본체, 보드 바깥 잡음 제거)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(board_mask, connectivity=8)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        board_mask = np.where(labels == largest, 255, 0).astype(np.uint8)

    return board_mask


def build_exposed_copper_mask(hsv: np.ndarray) -> np.ndarray:
    """
    노출된 구리 패드 검출.

    전략:
    1) 채도 높은 금색 패드 (ENIG 이상적 조건): H=[10-30], S 높음
    2) 무광 금속 패드 (사진에서 낮은 채도로 찍히는 경우): S=[12-70], V=[130-240]
       → 흰색 실크스크린(S<12, V>215)은 S가 너무 낮으므로 제외
    """
    mask_gold = cv2.inRange(hsv, COPPER_GOLD_LOWER, COPPER_GOLD_UPPER)

    # 무광 금속 패드 후보
    mask_metal = cv2.inRange(hsv, COPPER_METAL_LOWER, COPPER_METAL_UPPER)
    # 순수 흰색 실크스크린 제거 (S가 극히 낮고 V가 매우 높은 영역)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    silk_mask = ((s < SILKSCREEN_S_MAX) & (v > SILKSCREEN_V_MIN)).astype(np.uint8) * 255
    mask_metal = cv2.bitwise_and(mask_metal, cv2.bitwise_not(silk_mask))

    # 녹색 솔더마스크 영역도 제거 (H=[75-110], S>50 인 픽셀)
    h = hsv[:, :, 0]
    soldermask_pix = ((h >= 75) & (h <= 110) & (s > 50)).astype(np.uint8) * 255
    mask_metal = cv2.bitwise_and(mask_metal, cv2.bitwise_not(soldermask_pix))

    return cv2.bitwise_or(mask_gold, mask_metal)


def build_subsurface_trace_mask(hsv: np.ndarray,
                                board_mask: np.ndarray) -> np.ndarray:
    """
    솔더마스크 아래 구리 트레이스 검출.

    원리: 구리 위의 솔더마스크는 FR4 위의 솔더마스크보다 H가 낮고(올리브/따뜻한 녹색)
    채도(S)가 낮다. CLAHE 후 이 미세한 차이를 임계값으로 분리한다.

    보드 마스크를 AND 해 배경 픽셀을 원천 차단한다.
    """
    mask_subsurface = cv2.inRange(hsv, SUBSURFACE_TRACE_LOWER, SUBSURFACE_TRACE_UPPER)
    # 일반 솔더마스크 범위(H가 더 높은 쪽)와 겹치는 부분 제외
    mask_pure_soldermask = cv2.inRange(hsv, np.array([42, 80, 50]),
                                            np.array([85, 255, 255]))
    mask_subsurface = cv2.bitwise_and(mask_subsurface,
                                      cv2.bitwise_not(mask_pure_soldermask))
    # 보드 영역 내부로 제한
    return cv2.bitwise_and(mask_subsurface, board_mask)


def apply_morphology(mask: np.ndarray,
                     open_ksize: int = 3,
                     close_ksize: int = 9) -> np.ndarray:
    """
    Morphological 후처리.
    - Opening:  미세 노이즈 제거
    - Closing:  인접 트레이스 연결, 작은 구멍 채움
    """
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (open_ksize,  open_ksize))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (close_ksize, close_ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    return mask


def filter_by_area(mask: np.ndarray, min_area: int = 80) -> np.ndarray:
    """
    연결 요소(Connected Component) 면적 필터.
    min_area 픽셀 미만의 작은 요소는 노이즈로 간주해 제거한다.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == i] = 255
    return filtered


# ---------------------------------------------------------------------------
# 메인 세그멘테이션 파이프라인
# ---------------------------------------------------------------------------

def segment_traces(img_bgr: np.ndarray,
                   open_ksize: int = 3,
                   close_ksize: int = 9,
                   min_area: int = 80,
                   use_subsurface: bool = True,
                   debug_dir: str = None) -> np.ndarray:
    """
    PCB 이미지에서 배선(trace) 이진 마스크를 생성한다.

    파이프라인:
      A. CLAHE 대비 향상
      B. Gaussian blur + HSV 변환
      C. 보드 마스크 생성 (배경 제거)
      D. 노출 구리 패드 검출
      E. (선택) 솔더마스크 아래 트레이스 검출
      F. 합산 후 보드 마스크 적용
      G. Morphological 후처리
      H. 면적 필터 (노이즈 점 제거)

    Args:
        img_bgr:        BGR 입력 이미지
        open_ksize:     opening 커널 크기 (기본 3)
        close_ksize:    closing 커널 크기 (기본 9)
        min_area:       연결 요소 최소 면적 픽셀 수 (기본 80)
        use_subsurface: 솔더마스크 아래 트레이스 검출 포함 여부
        debug_dir:      중간 결과 저장 디렉토리 (None이면 저장 안 함)

    Returns:
        이진 마스크 (배경=0, 트레이스=255)
    """
    # A. CLAHE 대비 향상
    enhanced = enhance_contrast(img_bgr)

    # B. Gaussian blur + HSV 변환
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # C. 보드 마스크 (배경 회색 제거)
    board_mask = build_board_mask(hsv)

    # D. 노출 구리 패드 검출
    mask_copper = build_exposed_copper_mask(hsv)
    mask_copper = cv2.bitwise_and(mask_copper, board_mask)

    # E. 솔더마스크 아래 트레이스 검출 (선택)
    if use_subsurface:
        mask_sub = build_subsurface_trace_mask(hsv, board_mask)
    else:
        mask_sub = np.zeros_like(mask_copper)

    # F. 합산
    mask_combined = cv2.bitwise_or(mask_copper, mask_sub)

    # G. Morphological 후처리
    mask_morph = apply_morphology(mask_combined, open_ksize, close_ksize)

    # H. 면적 필터
    mask_final = filter_by_area(mask_morph, min_area)

    if debug_dir:
        _save_debug_images(debug_dir, img_bgr, enhanced, hsv,
                           board_mask, mask_copper, mask_sub,
                           mask_combined, mask_morph, mask_final)

    return mask_final


# ---------------------------------------------------------------------------
# 저장 유틸리티
# ---------------------------------------------------------------------------

def save_mask(mask: np.ndarray, output_path: str) -> None:
    """이진 마스크를 PNG로 저장 (배경=0, 전경=255)."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, mask)
    print(f"마스크 저장: {output_path}")


def save_overlay(img_bgr: np.ndarray, mask: np.ndarray,
                 output_path: str) -> None:
    """원본 이미지 위에 빨간색 마스크를 반투명 오버레이해서 저장."""
    red_layer = np.zeros_like(img_bgr)
    red_layer[mask == 255] = (0, 0, 255)
    blended = cv2.addWeighted(img_bgr, 0.6, red_layer, 0.4, 0)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, blended)
    print(f"오버레이 저장: {output_path}")


def _save_debug_images(debug_dir: str,
                       img_orig: np.ndarray,
                       img_enhanced: np.ndarray,
                       hsv: np.ndarray,
                       board_mask: np.ndarray,
                       mask_copper: np.ndarray,
                       mask_sub: np.ndarray,
                       mask_combined: np.ndarray,
                       mask_morph: np.ndarray,
                       mask_final: np.ndarray) -> None:
    """중간 단계 이미지를 debug_dir에 모두 저장."""
    d = Path(debug_dir)
    d.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(d / "dbg_0_original.png"),   img_orig)
    cv2.imwrite(str(d / "dbg_0_enhanced.png"),   img_enhanced)
    cv2.imwrite(str(d / "dbg_1_board_mask.png"), board_mask)
    cv2.imwrite(str(d / "dbg_2_copper.png"),     mask_copper)
    cv2.imwrite(str(d / "dbg_3_subsurface.png"), mask_sub)
    cv2.imwrite(str(d / "dbg_4_combined.png"),   mask_combined)
    cv2.imwrite(str(d / "dbg_5_morph.png"),      mask_morph)
    cv2.imwrite(str(d / "dbg_6_final.png"),      mask_final)

    h, s, v = cv2.split(hsv)
    cv2.imwrite(str(d / "dbg_hsv_h.png"), h * 2)
    cv2.imwrite(str(d / "dbg_hsv_s.png"), s)
    cv2.imwrite(str(d / "dbg_hsv_v.png"), v)

    print(f"디버그 이미지 저장: {debug_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PCB 배선 세그멘테이션 — CLAHE + HSV 기반 (Stage 2, Step 1 개선)"
    )
    parser.add_argument("--input",     required=True,
                        help="입력 PCB 이미지 경로 (PNG/JPG/WEBP)")
    parser.add_argument("--output",    required=True,
                        help="출력 이진 마스크 경로 (PNG)")
    parser.add_argument("--overlay",   default=None,
                        help="오버레이 디버그 이미지 경로")
    parser.add_argument("--debug-dir", default=None,
                        help="중간 단계 이미지 저장 디렉토리")
    parser.add_argument("--open-ksize",  type=int, default=3,
                        help="Opening 커널 크기 (기본: 3)")
    parser.add_argument("--close-ksize", type=int, default=9,
                        help="Closing 커널 크기 (기본: 9)")
    parser.add_argument("--min-area",    type=int, default=80,
                        help="연결 요소 최소 면적 픽셀 수 (기본: 80)")
    parser.add_argument("--no-subsurface", action="store_true",
                        help="솔더마스크 아래 트레이스 검출 비활성화")
    args = parser.parse_args()

    img = load_image(args.input)
    mask = segment_traces(
        img,
        open_ksize=args.open_ksize,
        close_ksize=args.close_ksize,
        min_area=args.min_area,
        use_subsurface=not args.no_subsurface,
        debug_dir=args.debug_dir,
    )
    save_mask(mask, args.output)

    if args.overlay:
        save_overlay(img, mask, args.overlay)


if __name__ == "__main__":
    main()
