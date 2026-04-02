#!/usr/bin/env python3
"""
Gerber 구리 레이어 → 이진 마스크 PNG 변환 — Stage 2, Step 2

Arduino Leonardo Gerber 파일에서 copper layer를 추출하여
Ground Truth 이진 마스크를 생성한다.

Gerber 파일 확보 방법:
  Arduino Leonardo 공식 설계 파일은 오픈소스로 공개되어 있다.
  arduino/ArduinoCore-avr GitHub 저소에서 hardware/arduino/avr/variants/
  또는 별도 보드 리포지토리의 /extras/gerber/ 디렉토리에서 다운로드.

  대표 레이어 파일명 (Eagle CAM 출력 기준):
    - Leonardo.GTL   : Top Copper Layer     ← 앞면 배선 GT
    - Leonardo.GBL   : Bottom Copper Layer  ← 뒷면 배선 GT
    - Leonardo.GBO   : Board Outline

  KiCad Gerber 기준:
    - F_Cu.gbr, B_Cu.gbr

실행 예시:
    # 앞면 GT 생성
    python src/segmentation/gerber_to_mask.py \
        --input eagle/Leonardo.GTL \
        --output data/masks/gt_front.png \
        --dpi 500

    # 뒷면 GT 생성
    python src/segmentation/gerber_to_mask.py \
        --input eagle/Leonardo.GBL \
        --output data/masks/gt_back.png \
        --dpi 500 --flip-horizontal
"""

import argparse
import cv2
import numpy as np
from pathlib import Path

import gerber
from gerber.render.cairo_backend import GerberCairoContext, RenderSettings, THEMES


# ---------------------------------------------------------------------------
# Gerber 로드 및 렌더링
# ---------------------------------------------------------------------------

def load_gerber(path: str):
    """
    Gerber 파일 로드.
    pcb-tools가 Python 3.11의 'rU' 모드와 충돌하므로
    직접 파일을 읽어 gerber.loads()에 전달한다.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    return gerber.loads(content)


def render_copper_layer(layer,
                        dpi: int = 500,
                        tmp_path: str = "_tmp_gerber_render.png") -> np.ndarray:
    """
    Gerber 구리 레이어를 PNG로 렌더링한 뒤 BGR 배열로 반환.

    Args:
        layer:    load_gerber()로 읽은 레이어 객체
        dpi:      출력 해상도. Gerber 단위에 따라 자동 변환.
                  Arduino Leonardo(67.5mm x 52.2mm) 기준:
                    300 dpi → ~800 x 620 px
                    500 dpi → ~1330 x 1030 px
        tmp_path: 중간 임시 파일 경로 (자동 삭제됨)

    Returns:
        BGR 형식 np.ndarray
    """
    units = getattr(layer, 'units', 'inch')
    # scale: GerberCairoContext에 전달하는 pixels-per-unit 값
    scale = dpi / 25.4 if units == 'metric' else float(dpi)

    # 검은 배경 + 밝은 구리 설정
    copper_settings = THEMES['default'].top          # RenderSettings
    bg_settings     = THEMES['default'].background   # RenderSettings

    ctx = GerberCairoContext(scale=scale)
    ctx.render_layer(layer, settings=copper_settings, bgsettings=bg_settings)

    tmp = Path(tmp_path)
    ctx.dump(str(tmp))
    img = cv2.imread(str(tmp), cv2.IMREAD_COLOR)
    tmp.unlink(missing_ok=True)

    if img is None:
        raise RuntimeError(f"렌더링 결과 읽기 실패: {tmp_path}")
    return img


def to_binary_mask(img_bgr: np.ndarray,
                   threshold: int = 130) -> np.ndarray:
    """
    렌더링된 컬러 이미지에서 구리 영역을 이진 마스크로 변환.

    pcb-tools의 기본 테마는 구리=어둡고(~74-140) 배경=밝다(~160-222).
    따라서 THRESH_BINARY_INV: 픽셀값 < threshold → 구리(255), 나머지 → 배경(0)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return mask


# ---------------------------------------------------------------------------
# 이미지 보정
# ---------------------------------------------------------------------------

def flip_horizontal(mask: np.ndarray) -> np.ndarray:
    """하단(뒷면) Gerber는 미러링이 필요할 수 있음."""
    return cv2.flip(mask, 1)


def resize_to_match(mask: np.ndarray,
                    target_w: int,
                    target_h: int) -> np.ndarray:
    """
    GT 마스크를 PCB 사진 크기에 맞게 리사이즈.
    세밀한 정합(homography)이 필요하면 align_to_photo() 사용.
    """
    return cv2.resize(mask, (target_w, target_h),
                      interpolation=cv2.INTER_NEAREST)


def print_layer_info(layer) -> None:
    """Gerber 레이어 메타데이터 출력 (디버깅용)."""
    bounds = layer.bounds  # ((min_x, max_x), (min_y, max_y))
    units = getattr(layer, 'units', '?')
    w = bounds[0][1] - bounds[0][0]
    h = bounds[1][1] - bounds[1][0]
    unit_str = "mm" if units == "metric" else "inch"
    print(f"  단위: {units}")
    print(f"  크기: {w:.3f} x {h:.3f} {unit_str}")
    if units == "metric":
        print(f"  크기(inch): {w/25.4:.3f} x {h/25.4:.3f} inch")
    else:
        print(f"  크기(mm): {w*25.4:.1f} x {h*25.4:.1f} mm")


# ---------------------------------------------------------------------------
# 저장
# ---------------------------------------------------------------------------

def save_mask(mask: np.ndarray, output_path: str) -> None:
    """이진 마스크를 PNG로 저장."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, mask)
    print(f"GT 마스크 저장: {output_path}  크기: {mask.shape[1]}x{mask.shape[0]}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Gerber 구리 레이어 → Ground Truth 이진 마스크 (Stage 2, Step 2)"
    )
    parser.add_argument(
        "--input", required=True,
        help="입력 Gerber 파일 경로 (.GTL / .GBL / F_Cu.gbr 등)"
    )
    parser.add_argument(
        "--output", required=True,
        help="출력 이진 마스크 PNG 경로"
    )
    parser.add_argument(
        "--dpi", type=int, default=500,
        help="렌더링 해상도 (pixels per inch, 기본: 500)"
    )
    parser.add_argument(
        "--flip-horizontal", action="store_true",
        help="좌우 반전 — 뒷면(Bottom) Gerber를 사진 방향에 맞출 때 사용"
    )
    parser.add_argument(
        "--target-width", type=int, default=None,
        help="출력 마스크 너비 픽셀 (PCB 사진 크기에 맞출 때)"
    )
    parser.add_argument(
        "--target-height", type=int, default=None,
        help="출력 마스크 높이 픽셀 (PCB 사진 크기에 맞출 때)"
    )
    parser.add_argument(
        "--threshold", type=int, default=130,
        help="구리 영역 이진화 임계값 (THRESH_BINARY_INV, 기본: 130)"
    )
    args = parser.parse_args()

    print(f"Gerber 로드: {args.input}")
    layer = load_gerber(args.input)
    print_layer_info(layer)

    print(f"렌더링 중... (DPI={args.dpi})")
    rendered = render_copper_layer(layer, dpi=args.dpi)

    mask = to_binary_mask(rendered, threshold=args.threshold)

    if args.flip_horizontal:
        mask = flip_horizontal(mask)
        print("좌우 반전 적용")

    if args.target_width and args.target_height:
        mask = resize_to_match(mask, args.target_width, args.target_height)
        print(f"리사이즈: {args.target_width}x{args.target_height}")

    save_mask(mask, args.output)


if __name__ == "__main__":
    main()
