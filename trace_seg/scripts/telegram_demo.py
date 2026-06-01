#!/usr/bin/env python3
"""
Telegram 라이브 데모 봇 — 보드 사진을 받아 trace 추출 + net 분리 결과를 회신.

폰으로 봇에게 PCB(Arduino류, top-down) 사진 전송 → 이 봇이 도는 머신(GPU+모델)이
U-Net 추론 + connected-components net 색칠 → [추출 오버레이 | net 맵] 패널 회신.

토큰: @BotFather로 봇 생성 후 TELEGRAM_BOT_TOKEN 환경변수 또는 --token.
(토큰은 비밀 — 커밋/공유 금지.)

사용 (trace_seg 디렉터리, all-data 모델 학습 후):
  $env:TELEGRAM_BOT_TOKEN="123:ABC"
  python scripts/telegram_demo.py
  # 또는
  python scripts/telegram_demo.py --token 123:ABC --ckpt data/unet/all/unet_best.pt

주의: Arduino류 + 반듯한(top-down, 보드 꽉 찬) 사진일 때만 잘 됨(학습 분포).
임의 PCB·비스듬 사진은 결과 거칢.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests
import torch

_TRACE_SEG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TRACE_SEG / "src"))
sys.path.insert(0, str(_TRACE_SEG / "scripts"))
from models.unet import UNet                  # noqa: E402
from netlist_poc import components, palette    # noqa: E402  (net 분리 로직 재사용)


def letterbox(img, size, pad_value, interp):
    """dataset.TraceDataset._letterbox 와 동일해야 함 (학습 입력과 일치)."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    r = cv2.resize(img, (nw, nh), interpolation=interp)
    top, left = (size - nh) // 2, (size - nw) // 2
    return cv2.copyMakeBorder(r, top, size - nh - top, left, size - nw - left,
                              cv2.BORDER_CONSTANT, value=pad_value)


def load_model(ckpt, device):
    ck = torch.load(ckpt, map_location=device)
    model = UNet(base=ck.get("base", 32)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck.get("size", 512)


def infer_panel(photo_bgr, model, size, device, min_area):
    """사진 -> [trace 오버레이 | net 색칠맵] 패널(BGR), net 개수."""
    rgb = cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2RGB)
    lb = letterbox(rgb, size, (0, 0, 0), cv2.INTER_AREA)
    t = torch.from_numpy(np.ascontiguousarray(lb, np.float32).transpose(2, 0, 1) / 255.0)
    with torch.no_grad():
        pred = (torch.sigmoid(model(t[None].to(device)))[0, 0] > 0.5).cpu().numpy().astype(np.uint8)
    pred = cv2.morphologyEx(pred, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    labels, keep, _ = components(pred, min_area)
    cols = palette(max(keep, default=0) + 1)
    netmap = np.zeros((size, size, 3), np.uint8)
    for lab in keep:
        netmap[labels == lab] = cols[lab]

    lb_bgr = cv2.cvtColor(lb, cv2.COLOR_RGB2BGR)
    over = lb_bgr.copy()
    over[pred > 0] = (0, 0, 255)                      # trace = red
    over = cv2.addWeighted(lb_bgr, 0.55, over, 0.45, 0)
    return np.hstack([over, netmap]), len(keep)


def main():
    ap = argparse.ArgumentParser(description="Telegram 라이브 데모 봇")
    ap.add_argument("--token", default=os.environ.get("TELEGRAM_BOT_TOKEN"))
    ap.add_argument("--ckpt", default=str(_TRACE_SEG / "data/unet/all/unet_best.pt"))
    ap.add_argument("--min-area", type=int, default=30)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if not args.token:
        print("토큰 없음. TELEGRAM_BOT_TOKEN env 또는 --token 필요 (@BotFather 발급).")
        return
    if not Path(args.ckpt).exists():
        print(f"체크포인트 없음: {args.ckpt} — all-data 모델 먼저 학습.")
        return

    device = torch.device(args.device)
    model, size = load_model(args.ckpt, device)
    api = f"https://api.telegram.org/bot{args.token}"
    print(f"봇 시작. device={device}, ckpt={Path(args.ckpt).name}, size={size}. Ctrl+C 종료.")

    offset = None
    while True:
        try:
            r = requests.get(f"{api}/getUpdates",
                             params={"timeout": 30, "offset": offset}, timeout=40)
            for upd in r.json().get("result", []):
                offset = upd["update_id"] + 1
                msg = upd.get("message") or {}
                chat = msg.get("chat", {}).get("id")
                if not chat:
                    continue
                if "photo" in msg:
                    fid = msg["photo"][-1]["file_id"]            # 최대 해상도
                    meta = requests.get(f"{api}/getFile", params={"file_id": fid}, timeout=20).json()
                    fp = meta["result"]["file_path"]
                    raw = requests.get(f"https://api.telegram.org/file/bot{args.token}/{fp}", timeout=30).content
                    arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                    if arr is None:
                        requests.post(f"{api}/sendMessage", data={"chat_id": chat, "text": "이미지 디코드 실패"})
                        continue
                    panel, n = infer_panel(arr, model, size, device, args.min_area)
                    sd = _TRACE_SEG / "data/unet/bot_live"     # 진단용 입력/출력 저장
                    sd.mkdir(parents=True, exist_ok=True)
                    ts = time.strftime("%H%M%S")
                    cv2.imwrite(str(sd / f"{ts}_in.png"), arr)
                    cv2.imwrite(str(sd / f"{ts}_out.png"), panel)
                    buf = cv2.imencode(".png", panel)[1].tobytes()
                    requests.post(f"{api}/sendPhoto",
                                  data={"chat_id": chat,
                                        "caption": f"trace 추출 + net {n}개  (좌: 추출 오버레이, 우: net 맵)"},
                                  files={"photo": ("result.png", buf, "image/png")}, timeout=30)
                    print(f"[ok] chat={chat} nets={n}")
                elif msg.get("text", "").startswith("/start"):
                    requests.post(f"{api}/sendMessage",
                                  data={"chat_id": chat,
                                        "text": "Arduino류 PCB를 top-down으로 찍어 보내면 배선(trace)과 net을 추출해 돌려줍니다."})
        except requests.RequestException as e:
            print(f"[net] {e}; 5s 후 재시도")
            time.sleep(5)
        except KeyboardInterrupt:
            print("종료")
            break


if __name__ == "__main__":
    main()
