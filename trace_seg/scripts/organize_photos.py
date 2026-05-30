#!/usr/bin/env python3
"""
trace_seg 사진 정리: Dataset2(직접촬영) + roboflow_dataset(웹) 원본 더미를
사용 가능 6보드용 깔끔한 구조로 재배치.

규칙:
- 사용 가능 = GT mask 있는 6보드(USABLE_BOARDS). 그 보드의 FRONT/BACK 뷰만 사용.
  -> data/images/{BOARD_ID}/{front,back}_{photo,web}.{ext}
     (photo = Dataset2 직접촬영, web = roboflow 웹)
- HEIC(확장자 없는 것 포함, 매직바이트로 판별)는 PNG로 변환(pillow-heif).
  JPG/webp는 그대로 이동. 원본 HEIC는 archive로 보관(삭제 X).
- SIDE 촬영 + 비대상 보드(멀티레이어/실드/어댑터)는 data/images_unused/ 로 이동.
- roboflow 지저분한 이름('의 사본', 이중 .webp, _WEB, 공백/이중밑줄)은 BOARD_ID로 정규화.

dry-run 기본. 실제 이동/변환은 --execute.

사용:
    python scripts/organize_photos.py            # 계획만 출력 (파일 변경 없음)
    python scripts/organize_photos.py --execute  # 실제 수행
"""
import argparse
import re
import shutil
from pathlib import Path

_TRACE_SEG = Path(__file__).resolve().parent.parent
DATA = _TRACE_SEG / "data"
DATASET2 = DATA / "Dataset2"
ROBOFLOW = DATA / "roboflow_dataset"
IMAGES = DATA / "images"
UNUSED = DATA / "images_unused"

# GT mask 존재하는 6보드 (trace_seg 사용 대상)
USABLE_BOARDS = {
    "LEONARDO_BOARD_A000057",
    "LEONARDO_BOARD_NH_A000052",
    "MEGA_R3_BOARD_2560_A000067",
    "NANO_BOARD_A000005",
    "UNO_R3_BOARD_A000066",
    "UNO_R3_BOARD_SMD_A000073",
}

# roboflow 서술형 이름(정규화 stem) -> BOARD_ID
ROBOFLOW_MAP = {
    "arduino_uno_rev3": "UNO_R3_BOARD_A000066",
    "arduino_uno_rev3_smd": "UNO_R3_BOARD_SMD_A000073",
    "arduino_mega_2560_rev3": "MEGA_R3_BOARD_2560_A000067",
    "arduino_leonardo_with_headers": "LEONARDO_BOARD_A000057",
    "arduino_leonardo_without_headers": "LEONARDO_BOARD_NH_A000052",
    "nano_board_a000005": "NANO_BOARD_A000005",
}

HEIC_BRANDS = {b"heic", b"heix", b"hevc", b"heim", b"heis",
               b"hevm", b"hevs", b"mif1", b"msf1"}


def sniff_format(path: Path) -> str | None:
    """매직바이트로 이미지 포맷 판별 (확장자 무시)."""
    with open(path, "rb") as f:
        head = f.read(32)
    if head[:3] == b"\xff\xd8\xff":
        return "jpg"
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    if head[4:8] == b"ftyp" and head[8:12] in HEIC_BRANDS:
        return "heic"
    return None


def dataset2_view(board_id: str, stem: str):
    """Dataset2 파일 stem -> (side, usable_view). 정확히 FRONT/BACK만 사용 뷰."""
    suf = stem[len(board_id):].lstrip("_").upper()
    if suf == "FRONT":
        return "front", True
    if suf == "BACK":
        return "back", True
    if suf.startswith("SIDE"):
        return "side", False
    if suf.startswith("FRONT"):       # FRONT1/FRONT2 같은 여분
        return "front_extra", False
    if suf.startswith("BACK"):
        return "back_extra", False
    return (suf.lower() or "other"), False


def normalize_roboflow(name: str) -> str:
    """roboflow 파일명 -> 정규화 stem ('의 사본'/이중확장자/공백/_WEB 제거)."""
    low = name.lower()
    idx = low.find(".webp")
    s = name[:idx] if idx != -1 else name      # 첫 .webp 이후 전부 제거
    s = s.lower().replace(" ", "_")
    s = re.sub(r"_+", "_", s).strip("_")
    s = re.sub(r"_web$", "", s)
    return s


def roboflow_parse(name: str):
    """roboflow 파일명 -> (board_id|None, side|None, usable)."""
    s = normalize_roboflow(name)
    m = re.search(r"_(front|back|side)$", s)
    side = m.group(1) if m else None
    desc = s[:m.start()] if m else s
    board = ROBOFLOW_MAP.get(desc)
    if board is None and desc.upper() in USABLE_BOARDS:
        board = desc.upper()
    usable = (board in USABLE_BOARDS) and (side in ("front", "back"))
    return board, side, usable


def build_plan():
    """(action, src, dst) 리스트 생성. action: 'convert'(heic->png) | 'move'."""
    ops = []

    # --- Dataset2 (직접촬영) ---
    if DATASET2.exists():
        for f in sorted(p for p in DATASET2.rglob("*") if p.is_file()):
            board_id = f.parent.name
            side, usable_view = dataset2_view(board_id, f.stem)
            if board_id in USABLE_BOARDS and usable_view:
                fmt = sniff_format(f)
                if fmt == "heic":
                    ops.append(("convert", f, IMAGES / board_id / f"{side}_photo.png"))
                    ops.append(("move", f, UNUSED / "Dataset2" / board_id / f.name))  # 원본 보관
                else:
                    ext = fmt or (f.suffix.lstrip(".").lower() or "bin")
                    ops.append(("move", f, IMAGES / board_id / f"{side}_photo.{ext}"))
            else:
                ops.append(("move", f, UNUSED / "Dataset2" / board_id / f.name))

    # --- roboflow (웹) ---
    if ROBOFLOW.exists():
        for f in sorted(p for p in ROBOFLOW.rglob("*.webp") if p.is_file()):
            board, side, usable = roboflow_parse(f.name)
            if usable:
                ops.append(("move", f, IMAGES / board / f"{side}_web.webp"))
            else:
                ops.append(("move", f, UNUSED / "roboflow" / f.name))

    # --- 기존 loose leonardo webp (구 평가용, web 소스로 대체됨) ---
    for legacy in ("leonardo_front.webp", "leonardo_back.webp"):
        p = IMAGES / legacy
        if p.exists():
            ops.append(("move", p, UNUSED / "legacy" / legacy))

    return ops


def print_plan(ops):
    usable = [o for o in ops if str(o[2]).startswith(str(IMAGES))]
    unused = [o for o in ops if str(o[2]).startswith(str(UNUSED))]
    print(f"\n=== 사용 가능 -> data/images/ ({len(usable)}) ===")
    for action, src, dst in usable:
        tag = "[CONVERT heic->png]" if action == "convert" else "[move]"
        print(f"  {dst.relative_to(DATA)}  <-  {src.relative_to(DATA)}  {tag}")
    print(f"\n=== 비대상 -> data/images_unused/ ({len(unused)}) ===")
    for action, src, dst in unused:
        print(f"  {dst.relative_to(DATA)}  <-  {src.relative_to(DATA)}")
    print(f"\n총 {len(ops)} 작업 (사용 {len(usable)} / 비대상 {len(unused)})")


def execute(ops):
    if any(a == "convert" for a, _, _ in ops):
        import pillow_heif
        from PIL import Image
        pillow_heif.register_heif_opener()
    else:
        Image = None

    done = 0
    for action, src, dst in ops:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if action == "convert":
            Image.open(src).convert("RGB").save(dst, "PNG")
            print(f"[convert] {dst.relative_to(DATA)}")
            done += 1
        else:  # move
            if not src.exists():
                continue                       # 이미 이동됨 (재실행)
            if dst.exists():
                dst.unlink()
            shutil.move(str(src), str(dst))
            print(f"[move]    {dst.relative_to(DATA)}")
            done += 1

    # 비어버린 원본 더미 디렉터리 정리
    for root in (DATASET2, ROBOFLOW):
        if root.exists():
            for d in sorted((p for p in root.rglob("*") if p.is_dir()),
                            key=lambda p: len(p.parts), reverse=True):
                try:
                    d.rmdir()
                except OSError:
                    pass
            try:
                root.rmdir()
            except OSError:
                pass
    print(f"\n완료: {done} 작업 수행.")


def main():
    parser = argparse.ArgumentParser(description="trace_seg 사진 정리 (Dataset2 + roboflow)")
    parser.add_argument("--execute", action="store_true",
                        help="실제 이동/변환 수행 (미지정 시 계획만 출력)")
    args = parser.parse_args()

    ops = build_plan()
    if not ops:
        print("처리할 파일이 없습니다 (Dataset2/roboflow_dataset 비었거나 이미 정리됨).")
        return

    print_plan(ops)
    if args.execute:
        print("\n--- 실행 ---")
        execute(ops)
    else:
        print("\n(dry-run) 실제 수행하려면 --execute 추가.")


if __name__ == "__main__":
    main()
