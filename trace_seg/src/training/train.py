"""
trace_seg U-Net 학습/검증 루프 — Stage 2, Step 3.

손실: BCEWithLogits + Dice. 지표: IoU. device 자동(cuda 우선).
augmentation(flips/rot90/color jitter) 기본 ON, best-checkpoint, cosine lr,
grad clip 포함.

검증 분할:
  --val-board BOARD_ID : 그 보드를 통째로 val (누수 없는 cross-board 일반화 테스트)
  --val-frac F         : 무작위 F 비율 val (같은 보드가 train/val에 섞일 수 있음=낙관적)
  (둘 다 없으면) 전체 학습, val 없음

사용 (trace_seg 디렉터리에서, 정합된 사진 사용):
  # 보드 단위 hold-out
  python src/training/train.py --images data/images_registered \
      --val-board NANO_BOARD_A000005 --epochs 200 --save-pred
  # overfit 검증(옛 용법)
  python src/training/train.py --board LEONARDO_BOARD_A000057 --source photo --epochs 300
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_SRC = Path(__file__).resolve().parents[1]
_TRACE_SEG = _SRC.parent
sys.path.insert(0, str(_SRC))

from models.unet import UNet                              # noqa: E402
from training.dataset import TraceDataset, list_samples   # noqa: E402
import cv2                                                 # noqa: E402


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


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tot, n = 0.0, 0
    for img, mask, _ in loader:
        img, mask = img.to(device), mask.to(device)
        tot += iou_score(model(img), mask) * img.size(0)
        n += img.size(0)
    return tot / max(n, 1)


def save_predictions(model, dataset, device, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            img, mask, label = dataset[i]
            logits = model(img[None].to(device))
            pred = (torch.sigmoid(logits)[0, 0] > 0.5).cpu().numpy().astype(np.uint8) * 255
            safe = label.replace("/", "__")
            cv2.imwrite(str(out_dir / f"{safe}_pred.png"), pred)
    print(f"예측 저장: {out_dir}")


def split_samples(samples, args):
    if args.val_board:
        val = [s for s in samples if s[2].split("/")[0] == args.val_board]
        train = [s for s in samples if s[2].split("/")[0] != args.val_board]
    elif args.val_frac > 0:
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(len(samples))
        n_val = max(1, int(len(samples) * args.val_frac))
        val = [samples[i] for i in idx[:n_val]]
        train = [samples[i] for i in idx[n_val:]]
    else:
        train, val = samples, []
    return train, val


def main():
    ap = argparse.ArgumentParser(description="trace_seg U-Net 학습 (Step 3)")
    ap.add_argument("--images", default=str(_TRACE_SEG / "data/images_registered"))
    ap.add_argument("--masks", default=str(_TRACE_SEG / "data/masks"))
    ap.add_argument("--out", default=str(_TRACE_SEG / "data/unet"))
    ap.add_argument("--board", default=None, help="특정 BOARD_ID만 사용 (overfit용)")
    ap.add_argument("--side", choices=["front", "back"], default=None)
    ap.add_argument("--source", choices=["photo", "web", "both"], default="both")
    ap.add_argument("--val-board", default=None, help="이 보드를 통째로 val (cross-board)")
    ap.add_argument("--val-frac", type=float, default=0.0, help="무작위 val 비율")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clip", type=float, default=1.0, help="grad norm clip")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--log-every", type=int, default=10)
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

    train_s, val_s = split_samples(samples, args)
    print(f"device={args.device}  train={len(train_s)}  val={len(val_s)}")
    if val_s:
        print("  [val] " + ", ".join(s[2] for s in val_s))

    augment = not args.no_augment
    train_ds = TraceDataset(train_s, size=args.size, augment=augment)
    train_loader = DataLoader(train_ds, batch_size=min(args.batch, len(train_s)),
                              shuffle=len(train_s) > 1, num_workers=0)
    val_loader = None
    if val_s:
        val_ds = TraceDataset(val_s, size=args.size, augment=False)
        val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)

    device = torch.device(args.device)
    model = UNet(base=args.base).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    bce = nn.BCEWithLogitsLoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"UNet base={args.base}  params={n_params/1e6:.2f}M  size={args.size}  "
          f"batch={train_loader.batch_size}  augment={augment}  epochs={args.epochs}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best_iou, best_tag = -1.0, "train"

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss = ep_iou = 0.0
        n = 0
        for img, mask, _ in train_loader:
            img, mask = img.to(device), mask.to(device)
            opt.zero_grad()
            logits = model(img)
            loss = bce(logits, mask) + dice_loss(logits, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            bs = img.size(0)
            ep_loss += loss.item() * bs
            ep_iou += iou_score(logits, mask) * bs
            n += bs
        sched.step()

        train_iou = ep_iou / n
        val_iou = evaluate(model, val_loader, device) if val_loader else None
        monitor = val_iou if val_iou is not None else train_iou
        if monitor > best_iou:
            best_iou, best_tag = monitor, ("val" if val_iou is not None else "train")
            torch.save({"model": model.state_dict(), "base": args.base,
                        "size": args.size, "epoch": epoch, f"best_{best_tag}_iou": best_iou},
                       out / "unet_best.pt")

        if epoch % args.log_every == 0 or epoch == 1 or epoch == args.epochs:
            msg = (f"epoch {epoch:4d}/{args.epochs}  loss={ep_loss/n:.4f}  "
                   f"train_IoU={train_iou*100:5.2f}%")
            if val_iou is not None:
                msg += f"  val_IoU={val_iou*100:5.2f}%"
            print(msg)

    torch.save({"model": model.state_dict(), "base": args.base, "size": args.size},
               out / "unet_last.pt")
    print(f"best {best_tag}_IoU = {best_iou*100:.2f}%  -> {out/'unet_best.pt'}")

    if args.save_pred:
        save_predictions(model, train_ds, device, out / "preds")
        if val_loader:
            save_predictions(model, val_ds, device, out / "preds_val")


if __name__ == "__main__":
    main()
