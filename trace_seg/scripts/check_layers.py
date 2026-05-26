"""
EAGLE .brd 파일들의 구리 layer 수 확인.
다층 PCB(>2층)는 우리 작업 대상 아님 — 미리 솎아내기 위함.

사용:
    python scripts/check_layers.py --input data/eagle_raw
"""
import argparse
import os
import re
import sys
import glob

sys.stdout.reconfigure(encoding="utf-8")


def detect_layers(brd_path: str) -> tuple[list[int], list[int]]:
    """
    .brd 파일에서 구리 layer (1~16) 정의/사용 현황 추출.
    Returns (정의된 layer 번호 리스트, 실제 사용된 layer 번호 리스트).
    """
    with open(brd_path, encoding="utf-8") as f:
        text = f.read()

    defined = set()
    for m in re.finditer(r'<layer\s+number="(\d+)"', text):
        n = int(m.group(1))
        if 1 <= n <= 16:
            defined.add(n)

    used = set()
    for tag in ("wire", "polygon", "via", "smd", "pad"):
        pat = r"<" + tag + r"[^>]*\blayer=\"(\d+)\""
        for m in re.finditer(pat, text):
            n = int(m.group(1))
            if 1 <= n <= 16:
                used.add(n)

    return sorted(defined), sorted(used)


def main() -> None:
    parser = argparse.ArgumentParser(description="EAGLE .brd 구리 layer 검사")
    parser.add_argument("--input", required=True, help="eagle_raw 디렉토리")
    args = parser.parse_args()

    rows = []
    for name in sorted(os.listdir(args.input)):
        sub = os.path.join(args.input, name)
        if not os.path.isdir(sub):
            continue
        brd_files = glob.glob(os.path.join(sub, "*.brd"))
        if not brd_files:
            continue
        defined, used = detect_layers(brd_files[0])
        rows.append((name, defined, used))

    print(f"{'폴더':50s} {'정의':25s} {'실사용':25s} 층수  판정")
    print("-" * 130)
    for name, defined, used in rows:
        n_layers = len(used) if used else len(defined)
        verdict = "OK (단/양면)" if n_layers <= 2 else "[!] 다층 - 대상 외"
        print(f"{name:50s} {str(defined):25s} {str(used):25s} {n_layers}층  {verdict}")


if __name__ == "__main__":
    main()
