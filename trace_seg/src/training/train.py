"""
trace_seg U-Net 학습/검증 루프 — Stage 2, Step 3.

손실: BCEWithLogits + Dice. 지표: IoU. device 자동(cuda 우선).

가장 먼저 할 일은 **overfit 검증**: 1보드 직접촬영 사진(front+back)만으로
loss가 0 근처, IoU가 1.0 근처까지 가는지 확인해 모델/손실/루프가 올바른지 본다.
(데이터-마스크 정합 전이라 일반화 학습은 아직 의미 없음 — dataset.py 주석 참조)

사용 (trace_seg 디렉터리에서):
    # overfit 검증 (Leonardo 직접촬영 2장)
    python src/training/train.py --board LEONARDO_BOARD_A000057 --source photo \
        --epochs 300 --save-pred
    # 전체 (정합 후 본 학습용 — 지금은 비권장)
    python src/training/train.py --epochs 100
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_SRC = Path(__file__).resolve().parents[1]      # .../trace_seg/src
_TRACE_SEG = _SRC.parent
sys.path.insert(0, str(_SRC))

from models.unet import UNet                      # noqa: E402
from training.dataset import TraceDataset, list_samples  # noqa: E402
import cv2                                         # noqa: E402
import numpy as np                                 # noqa: E402


def dice_loss(logits, target, eps: float = 1.0):
    p = torch.sigmoid(logits).flatten(1)
    t = target.flatten(1)
    num = 2 * (p * t).sum(1) + eps
    den = p.sum(1) + t.sum(1) + eps
    return (1 - num / den).mean()


@torch.no_grad()
def iou_score(logits, target, thr: float = 0.5):
    p = (torch.sigmoid(logits) > thr).float()
    inter = (p * target).sum((1, 2, 3))
    union = ((p + target) > 0).float().sum((1, 2, 3))
    return (inter / (union + 1e-6)).mean().item()


def save_predictions(model, dataset, device, out_dir, size):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            img, mask, label = dataset[i]
            logits = model(img[None].to(device))
            pred = (torch.sigmoid(logits)[0, 0] > 0.5).cpu().numpy().astype(np.uint8) * 255
            safe = label.replace("/", "__")
            cv2.imwrite(str(out_dir / f"{safe}_pred.png"), pred)
            cv2.imwrite(str(out_dir / f"{safe}_gt.png"),
                        (mask[0].numpy() * 255).astype(np.uint8))
    print(f"예측/GT 저장: {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="trace_seg U-Net 학습 (Step 3)")
    ap.add_argument("--images", default=str(_TRACE_SEG / "data/images"))
    ap.add_argument("--masks", default=str(_TRACE_SEG / "data/masks"))
    ap.add_argument("--out", default=str(_TRACE_SEG / "data/unet"))
    ap.add_argument("--board", default=None, help="특정 BOARD_ID만 사용 (overfit 검증용)")
    ap.add_argument("--side", choices=["front", "back"], default=None)
    ap.add_argument("--source", choices=["photo", "web", "both"], default="both")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-pred", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    sources = ("photo", "web") if args.source == "both" else (args.source,)
    samples = list_samples(args.images, args.masks, sources=sources, board=args.board)
    if args.side:
        samples = [s for s in samples if f"/{args.side}/" in s[2] + "/"]
    if not samples:
        print("샘플 없음. --images/--masks/--board/--source/--side 확인.")
        return

    print(f"device={args.device}  samples={len(samples)}")
    for _, _, label in samples:
        print(f"  - {label}")

    dataset = TraceDataset(samples, size=args.size, augment=args.augment)
    loader = DataLoader(dataset, batch_size=min(args.batch, len(samples)),
                        shuffle=len(samples) > 1, num_workers=0)

    device = torch.device(args.device)
    model = UNet(base=args.base).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"UNet base={args.base}  params={n_params/1e6:.2f}M  "
          f"size={args.size}  batch={loader.batch_size}  epochs={args.epochs}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss = ep_iou = 0.0
        n = 0
        for img, mask, _ in loader:
            img, mask = img.to(device), mask.to(device)
            opt.zero_grad()
            logits = model(img)
            loss = bce(logits, mask) + dice_loss(logits, mask)
            loss.backward()
            opt.step()
            bs = img.size(0)
            ep_loss += loss.item() * bs
            ep_iou += iou_score(logits, mask) * bs
            n += bs
        if epoch % args.log_every == 0 or epoch == 1 or epoch == args.epochs:
            print(f"epoch {epoch:4d}/{args.epochs}  "
                  f"loss={ep_loss/n:.4f}  IoU={ep_iou/n*100:5.2f}%")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "unet_last.pt"
    torch.save({"model": model.state_dict(), "base": args.base,
                "size": args.size}, ckpt)
    print(f"체크포인트 저장: {ckpt}")

    if args.save_pred:
        save_predictions(model, dataset, device, out / "preds", args.size)


if __name__ == "__main__":
    main()
