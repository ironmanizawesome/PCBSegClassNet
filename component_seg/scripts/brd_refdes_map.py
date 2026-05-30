# ------------------------------------------------------------------------
# EAGLE .brd -> ref-des 위치 맵(PNG) + 부품 표(CSV) 생성기.
#
# Arduino 보드 annotation 시 GT(어떤 부품이 어디에 / 무슨 종류인지)를 확인하는
# 용도. Altium 365 Web Viewer는 Altium 네이티브(.PcbDoc)용이라 EAGLE .brd는
# 위치 특정이 어렵다. EAGLE .brd는 XML이고 <element>에 ref-des(name) / package /
# value / library / 좌표(x,y) / rot(회전·mirror)가 그대로 들어있으므로, viewer
# 없이 여기서 직접 뽑아 사진 옆에 두고 대조한다.
#
# 출력(보드별):
#   {board}_refdes_top.png     - top 면 부품 위치 + ref-des 라벨
#   {board}_refdes_bottom.png  - bottom 면 (사진 뒤집힘에 맞춰 좌우 미러), 부품 있을 때만
#   {board}_refdes.csv         - ref_des, side, type_guess, package, value, library, x_mm, y_mm, rot
#
# type_guess는 prefix/library/package 기반 "추정"이다. 사진으로 구별 불가한
# 부품(0402 R vs C, R vs L, D vs LED 등)은 annotation_rules.md section 9에 따라
# 반드시 BOM/value로 최종 확인할 것. PNG 색상은 시각적 그룹핑 보조일 뿐이다.
#
# Usage (repo root, pscn env):
#   pip install pillow            # 보통 이미 설치됨 (convert_heic.py가 PIL 사용)
#   # 단일 보드
#   python component_seg/scripts/brd_refdes_map.py \
#       -i trace_seg/data/eagle_raw/UNO_R3_BOARD_A000066/UNO_R3_BOARD_A000066.brd \
#       -o component_seg/data/refdes_maps
#   # 폴더 일괄 (.brd 재귀 스캔)
#   python component_seg/scripts/brd_refdes_map.py \
#       -i trace_seg/data/eagle_raw -o component_seg/data/refdes_maps
#
# 참고: 입력 .brd는 trace_seg/data/eagle_raw 아래의 공용 원본 설계 데이터다.
# 본 스크립트는 그 경로를 인자로 읽기만 하며(read-only), TF/PyTorch 의존성 없음.
# ------------------------------------------------------------------------

import argparse
import csv
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# 추정 종류 -> 색상 (시각적 그룹핑용, 정답 아님)
TYPE_COLORS = {
    "R": (220, 50, 50),     # resistor
    "C": (50, 120, 220),    # capacitor
    "L": (180, 120, 40),    # inductor
    "D": (160, 50, 200),    # diode
    "LED": (0, 170, 90),    # led
    "Q": (230, 140, 0),     # transistor
    "U": (40, 40, 40),      # IC
    "J": (0, 160, 180),     # connector / header
    "Y": (120, 80, 200),    # crystal / resonator
    "JP": (150, 150, 0),    # jumper / solder jumper
    "F": (200, 0, 120),     # fuse
    "SW": (90, 90, 90),     # switch / button
    "TP": (200, 200, 200),  # test point (기본 비표시)
    "FID": (215, 200, 185), # fiducial mark (기본 비표시)
    "other": (120, 120, 120),
}

# annotation 대상이 아닌 비부품 마크 (기본 제외)
NON_COMPONENT = {"TP", "FID"}


def parse_rot(rot: str):
    """EAGLE rot 문자열 -> (side, deg). 'MR90'->('bottom',90), 'R270'->('top',270)."""
    s = rot or "R0"
    letters = re.match(r"[A-Za-z]*", s).group()
    mirror = "M" in letters.upper()          # M = mirror = bottom 면
    m = re.search(r"\d+", s)
    deg = int(m.group()) if m else 0
    return ("bottom" if mirror else "top"), deg


def is_testpoint(package: str, value: str) -> bool:
    """테스트포인트(스프링 패드) 판별. package 'TP-SP' / value 'TP_SP' 등."""
    return (package or "").upper().startswith("TP") or \
           (value or "").upper().startswith("TP_SP")


def is_fiducial(package: str, value: str) -> bool:
    """피듀셜 정렬 마크 판별 (annotation 대상 아님)."""
    return "FIDUCIA" in (package or "").upper() or \
           "FIDUCIAL" in (value or "").upper()


def guess_type(name: str, library: str, package: str, value: str) -> str:
    """종류 추정 (best-effort). 사진 구별 불가 부품은 BOM/value로 최종 확인."""
    if is_testpoint(package, value):
        return "TP"
    if is_fiducial(package, value):
        return "FID"

    lib = (library or "").lower()
    pkg = (package or "").upper()

    # library/package 가 이름보다 신뢰도 높은 경우 먼저 (예: RX/TX = LED, ICSP = 헤더)
    if "led" in lib or "LED" in pkg:
        return "LED"
    if "diode" in lib:
        return "D"
    if "jumper" in lib or pkg == "SJ":        # 솔더 점퍼 -> JP (annotation_rules)
        return "JP"
    if re.match(r"\d+X\d+", pkg):             # 1X08, 2X03 = 핀 헤더
        return "J"
    if pkg.startswith("SOT23") or pkg.startswith("SOT-23"):
        return "Q"
    if any(pkg.startswith(p) for p in ("DIL", "SOIC", "QFP", "TQFP", "QFN", "DFN", "BGA")):
        return "U"

    # 표준 designator(문자+숫자, 예: R12/ZU4)일 때만 prefix 매핑.
    # RESET/POWER_5V 같은 단어형 이름엔 적용 안 함 (오분류 방지 -> other).
    m = re.match(r"^([A-Za-z]{1,3})\d+$", name or "")
    if not m:
        return "other"
    prefix = m.group(1).upper()

    prefix_map = {
        "R": "R", "RN": "R",
        "C": "C", "PC": "C",
        "L": "L",
        "D": "D",
        "Q": "Q", "T": "Q",
        "U": "U", "IC": "U", "ZU": "U",
        "J": "J", "CN": "J", "CON": "J", "P": "J",
        "Y": "Y", "X": "Y", "XTAL": "Y", "CR": "Y",
        "JP": "JP", "SJ": "JP",
        "F": "F", "FD": "F",
        "SW": "SW", "BTN": "SW", "PB": "SW",
    }
    return prefix_map.get(prefix, prefix_map.get(prefix[0], "other"))


def parse_brd(path: Path):
    """.brd(XML) -> (outline_segments, elements). outline은 layer 20 wire 직선 근사."""
    root = ET.parse(path).getroot()
    board = root.find("drawing/board")
    if board is None:
        raise ValueError("'<board>' 섹션 없음 (.brd 가 아닐 수 있음)")

    outline = []
    plain = board.find("plain")
    if plain is not None:
        for w in plain.findall("wire"):
            if w.get("layer") == "20":        # 20 = Dimension = 보드 외곽선
                outline.append((
                    float(w.get("x1")), float(w.get("y1")),
                    float(w.get("x2")), float(w.get("y2")),
                ))

    elements = []
    elems_node = board.find("elements")
    if elems_node is not None:
        for el in elems_node.findall("element"):
            name = el.get("name", "?")
            library = el.get("library", "")
            package = el.get("package", "")
            value = el.get("value", "")
            x = float(el.get("x", "0"))
            y = float(el.get("y", "0"))
            rot = el.get("rot", "R0")
            side, deg = parse_rot(rot)
            elements.append({
                "name": name, "library": library, "package": package,
                "value": value, "x": x, "y": y, "rot": rot,
                "side": side, "deg": deg,
                "type": guess_type(name, library, package, value),
            })
    return outline, elements


def compute_bounds(outline, elements):
    """렌더 범위. 외곽선 우선, 없으면 부품 좌표 + 패딩."""
    xs, ys = [], []
    for x1, y1, x2, y2 in outline:
        xs += [x1, x2]
        ys += [y1, y2]
    if not xs:
        xs = [e["x"] for e in elements] or [0.0]
        ys = [e["y"] for e in elements] or [0.0]
        pad = 5.0
        return min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad
    return min(xs), max(xs), min(ys), max(ys)


def render_side(board_name, side, outline, elems, bounds, scale, margin, mirror,
                font, title_font, fontsize):
    xmin, xmax, ymin, ymax = bounds
    header_h = title_font.size + fontsize + 28
    W = int((xmax - xmin) * scale) + 2 * margin
    H = int((ymax - ymin) * scale) + 2 * margin + header_h
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    def tx(x):
        # bottom 면: 보드를 좌우로 뒤집은 사진에 맞춰 x 미러
        u = (xmax - x) if mirror else (x - xmin)
        return margin + u * scale

    def ty(y):
        return header_h + margin + (ymax - y) * scale   # EAGLE y는 위쪽 -> 픽셀 y 뒤집기

    # 보드 외곽선 (직선 근사; 라운드 코너의 curve는 무시)
    ow = max(1, int(scale / 12))
    for x1, y1, x2, y2 in outline:
        draw.line([(tx(x1), ty(y1)), (tx(x2), ty(y2))], fill=(60, 60, 60), width=ow)

    # 부품: 점 + ref-des 라벨
    r = max(2, int(scale / 9))
    for e in elems:
        px, py = tx(e["x"]), ty(e["y"])
        col = TYPE_COLORS.get(e["type"], TYPE_COLORS["other"])
        draw.ellipse([px - r, py - r, px + r, py + r], fill=col, outline=(255, 255, 255))
        draw.text((px + r + 1, py - fontsize // 2), e["name"], fill=col, font=font)

    # 제목 + 범례
    title = f"{board_name}  —  {side.upper()}"
    if mirror:
        title += " (mirrored / 사진 뒷면 기준)"
    title += f"   [{len(elems)} parts]"
    draw.text((margin, margin // 2), title, fill=(0, 0, 0), font=title_font)

    lx = margin
    ly = margin // 2 + title_font.size + 6
    for t in sorted({e["type"] for e in elems}):
        col = TYPE_COLORS.get(t, TYPE_COLORS["other"])
        draw.rectangle([lx, ly, lx + 12, ly + 12], fill=col)
        draw.text((lx + 16, ly - 1), t, fill=(0, 0, 0), font=font)
        lx += 16 + 12 + int(draw.textlength(t, font=font)) + 14
    return img


def load_fonts(fontsize: int):
    """Windows arial 우선, 실패 시 PIL 기본 폰트."""
    def _try(size):
        for cand in ("arial.ttf", "C:/Windows/Fonts/arial.ttf"):
            try:
                return ImageFont.truetype(cand, size)
            except OSError:
                continue
        return ImageFont.load_default()
    return _try(fontsize), _try(fontsize + 5)


def process_brd(brd_path: Path, out_dir: Path, scale: int, mirror_bottom: bool,
                include_tp: bool) -> None:
    board_name = brd_path.stem
    try:
        outline, elements = parse_brd(brd_path)
    except (ET.ParseError, ValueError) as e:
        print(f"[ERR] {board_name}: {e.__class__.__name__}: {e}")
        return

    # 테스트포인트/피듀셜은 annotation 대상이 아니므로 기본 제외 (BOM sanity check와 일치)
    n_tp = sum(1 for e in elements if e["type"] in NON_COMPONENT)
    if not include_tp:
        elements = [e for e in elements if e["type"] not in NON_COMPONENT]

    out_dir.mkdir(parents=True, exist_ok=True)
    fontsize = max(10, int(scale * 0.55))
    margin = 40
    font, title_font = load_fonts(fontsize)
    bounds = compute_bounds(outline, elements)

    # CSV
    csv_path = out_dir / f"{board_name}_refdes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ref_des", "side", "type_guess", "package", "value",
                    "library", "x_mm", "y_mm", "rot"])
        for e in sorted(elements, key=lambda d: (d["side"], d["name"])):
            w.writerow([e["name"], e["side"], e["type"], e["package"], e["value"],
                        e["library"], e["x"], e["y"], e["rot"]])

    # 면별 PNG
    n_png = 0
    for side, mirror in (("top", False), ("bottom", mirror_bottom)):
        elems = [e for e in elements if e["side"] == side]
        if not elems:
            continue
        img = render_side(board_name, side, outline, elems, bounds, scale, margin,
                          mirror, font, title_font, fontsize)
        png_path = out_dir / f"{board_name}_refdes_{side}.png"
        img.save(png_path, "PNG")
        n_png += 1
        print(f"[ok]  {png_path.name}  ({len(elems)} parts)")

    tp_note = f", {n_tp} TP/FID excluded" if (n_tp and not include_tp) else ""
    print(f"[ok]  {csv_path.name}  ({len(elements)} parts{tp_note}) "
          f"-> {n_png} PNG")


def main():
    parser = argparse.ArgumentParser(
        prog="brd_refdes_map",
        description="EAGLE .brd -> ref-des 위치 맵 PNG + 부품 CSV",
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="입력 .brd 파일 또는 디렉터리 (디렉터리면 *.brd 재귀 스캔)",
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="출력 디렉터리 (보드 stem 기준 파일명으로 저장)",
    )
    parser.add_argument(
        "--scale", type=int, default=26,
        help="px/mm 해상도 (기본 26; 부품 밀집 보드는 키워서 라벨 겹침 완화)",
    )
    parser.add_argument(
        "--no-mirror-bottom", action="store_true",
        help="bottom 면을 좌우 미러하지 않음 (기본은 사진 뒷면에 맞춰 미러)",
    )
    parser.add_argument(
        "--include-testpoints", action="store_true",
        help="테스트포인트·피듀셜 등 비부품 마크도 포함 (기본 제외 — annotation 대상 아님)",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output)
    mirror_bottom = not args.no_mirror_bottom

    if in_path.is_dir():
        brds = sorted(in_path.rglob("*.brd"))
        if not brds:
            print(f"No .brd files found under {in_path}")
            return
        print(f"Found {len(brds)} .brd file(s) under {in_path}\n")
        for brd in brds:
            process_brd(brd, out_dir, args.scale, mirror_bottom,
                        args.include_testpoints)
    elif in_path.is_file():
        process_brd(in_path, out_dir, args.scale, mirror_bottom,
                    args.include_testpoints)
    else:
        print(f"[ERR] input not found: {in_path}")


if __name__ == "__main__":
    main()
