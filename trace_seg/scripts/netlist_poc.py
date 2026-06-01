"""
Stage 3 미니 PoC — trace mask에서 net(전기적 연결) 복원 가능성 데모.

아이디어: 한 구리 레이어에서 **서로 이어진 구리 = 하나의 net**.
예측 trace mask를 connected-components로 net 단위로 분리하고, CAD 기반
GT mask의 net과 대조해 "trace에서 netlist 위상을 복원할 수 있는가"를 본다.

평가(예측 net ↔ GT net 매칭):
  - clean   : GT net 1개 ↔ 예측 net 1개 (정상 복원). matched IoU 보고.
  - short   : 예측 net 1개가 GT net 2개+ 를 다리놓음 (병합) — netlist에 가장 치명적.
  - open    : GT net 1개가 예측에서 여러 조각 (분할).
  - missed  : GT net인데 예측에 대응 없음.
  - spurious: 예측 net인데 GT에 대응 없음.

정직 범위(발표 명시): **단일 레이어**만. via/뒷면 연결 미포함이라, 실제로 양면을
오가는 net은 한 면에서 여러 섬으로 보일 수 있음(= open으로 과다 집계될 수 있음).
1보드 데모. "완성된 netlist 추출"이 아니라 핵심 조각(연결성 복원)의 실증.

사용 (trace_seg 디렉터리):
  python scripts/netlist_poc.py            # 기본 UNO TH, holdout 모델로 예측
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_TRACE_SEG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TRACE_SEG / "src"))
from training.dataset import TraceDataset, list_samples   # noqa: E402
from models.unet import UNet                                # noqa: E402


def predict_mask(model, img_t, device):
    with torch.no_grad():
        logit = model(img_t[None].to(device))
        return (torch.sigmoid(logit)[0, 0] > 0.5).cpu().numpy().astype(np.uint8)


def components(mask_bin, min_area):
    """이진 mask -> (label 이미지, 유지할 label 리스트, area dict). 작은 잡음 제거."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    keep, area = [], {}
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a >= min_area:
            keep.append(i)
            area[i] = a
    return labels, keep, area


def overlaps(gt_labels, pred_labels, gk, pk, sig):
    """유의미 겹침(>= sig px) 간선. min-area로 살아남은 label만. 반환: g->[p], p->[g], ov."""
    gkset, pkset = set(gk), set(pk)
    both = (gt_labels > 0) & (pred_labels > 0)
    g = gt_labels[both].astype(np.int64)
    p = pred_labels[both].astype(np.int64)
    if g.size == 0:
        return {}, {}, {}
    key = g * 1_000_000 + p
    uk, cnt = np.unique(key, return_counts=True)
    g2p, p2g, ov = {}, {}, {}
    for k, c in zip(uk, cnt):
        if c < sig:
            continue
        gi, pi = int(k // 1_000_000), int(k % 1_000_000)
        if gi not in gkset or pi not in pkset:     # 걸러진 작은 net 제외
            continue
        ov[(gi, pi)] = int(c)
        g2p.setdefault(gi, []).append(pi)
        p2g.setdefault(pi, []).append(gi)
    return g2p, p2g, ov


def evaluate_nets(gt_labels, gk, ga, pred_labels, pk, pa, sig):
    g2p, p2g, ov = overlaps(gt_labels, pred_labels, gk, pk, sig)
    clean, ious = [], []
    for g in gk:
        ps = g2p.get(g, [])
        if len(ps) == 1:
            p = ps[0]
            if len(p2g.get(p, [])) == 1:           # 1:1
                inter = ov[(g, p)]
                union = ga[g] + pa[p] - inter
                ious.append(inter / union if union else 0.0)
                clean.append((g, p))
    opens = [g for g in gk if len(g2p.get(g, [])) >= 2]
    shorts = [p for p in pk if len(p2g.get(p, [])) >= 2]
    missed = [g for g in gk if len(g2p.get(g, [])) == 0]
    spurious = [p for p in pk if len(p2g.get(p, [])) == 0]
    return {
        "n_gt": len(gk), "n_pred": len(pk),
        "clean": len(clean), "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "short": len(shorts), "open": len(opens),
        "missed": len(missed), "spurious": len(spurious),
        "_g2p": g2p, "_p2g": p2g, "_ov": ov,
    }


def palette(n, seed=0):
    rng = np.random.default_rng(seed)
    cols = rng.integers(50, 256, size=(n + 1, 3), dtype=np.uint8)
    return cols


def color_nets(gt_labels, gk, pred_labels, pk, res):
    """매칭 색칠: GT net별 색 부여, 예측 net은 최대 겹침 GT 색(없으면 흰=spurious)."""
    cols = palette(max(gk, default=0) + 1)
    gt_vis = np.zeros((*gt_labels.shape, 3), np.uint8)
    for g in gk:
        gt_vis[gt_labels == g] = cols[g]
    p2g, ov = res["_p2g"], res["_ov"]
    pred_vis = np.zeros((*pred_labels.shape, 3), np.uint8)
    for p in pk:
        gs = p2g.get(p, [])
        if gs:
            best = max(gs, key=lambda g: ov[(g, p)])
            pred_vis[pred_labels == p] = cols[best]
        else:
            pred_vis[pred_labels == p] = (255, 255, 255)   # spurious
    return gt_vis, pred_vis


def label_strip(img, text):
    bar = np.zeros((28, img.shape[1], 3), np.uint8)
    cv2.putText(bar, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return np.vstack([bar, img])


def main():
    ap = argparse.ArgumentParser(description="Stage 3 netlist PoC (trace -> nets)")
    ap.add_argument("--board", default="UNO_R3_BOARD_A000066")
    ap.add_argument("--ckpt", default=str(_TRACE_SEG / "data/unet/ho_UNO_TH/unet_best.pt"))
    ap.add_argument("--images", default=str(_TRACE_SEG / "data/images_registered"))
    ap.add_argument("--masks", default=str(_TRACE_SEG / "data/masks"))
    ap.add_argument("--out", default=str(_TRACE_SEG / "data/unet/netlist_poc"))
    ap.add_argument("--min-area", type=int, default=30, help="net로 셀 최소 구리 픽셀")
    ap.add_argument("--sig", type=int, default=20, help="net 간 유의미 겹침 픽셀")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    samples = list_samples(args.images, args.masks, board=args.board)
    if not samples:
        print(f"샘플 없음: {args.board}")
        return
    device = torch.device(args.device)
    ck = torch.load(args.ckpt, map_location=device)
    size = ck.get("size", 512)
    model = UNet(base=ck.get("base", 32)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    ds = TraceDataset(samples, size=size, augment=False)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    print(f"board={args.board}  ckpt={Path(args.ckpt).name}  size={size}\n")
    print(f"{'sample':32s} {'GTnet':>5} {'Pnet':>5} {'clean':>5} {'IoU':>6} "
          f"{'short':>5} {'open':>5} {'miss':>4} {'spur':>4}")
    for i in range(len(ds)):
        img_t, mask_t, label = ds[i]
        gt = (mask_t[0].numpy() > 0.5).astype(np.uint8)
        pred = predict_mask(model, img_t, device)
        pred = cv2.morphologyEx(pred, cv2.MORPH_OPEN, kernel)   # 잡음 제거

        gtl, gk, ga = components(gt, args.min_area)
        prl, pk, pa = components(pred, args.min_area)
        res = evaluate_nets(gtl, gk, ga, prl, pk, pa, args.sig)

        gt_vis, pred_vis = color_nets(gtl, gk, prl, pk, res)
        gap = np.full((gt_vis.shape[0], 8, 3), 40, np.uint8)
        panel = np.hstack([label_strip(gt_vis, "GT nets"),
                           np.vstack([np.zeros((28, 8, 3), np.uint8), gap]),
                           label_strip(pred_vis, "Predicted nets")])
        safe = label.replace("/", "__")
        cv2.imwrite(str(out / f"{safe}_nets.png"), panel)

        print(f"{label:32s} {res['n_gt']:5d} {res['n_pred']:5d} {res['clean']:5d} "
              f"{res['mean_iou']*100:5.1f}% {res['short']:5d} {res['open']:5d} "
              f"{res['missed']:4d} {res['spurious']:4d}")

    print(f"\nnet map 저장: {out}")
    print("주의: 단일 레이어 기준. 양면 net은 한 면에서 분리돼 open 과다집계 가능.")


if __name__ == "__main__":
    main()
