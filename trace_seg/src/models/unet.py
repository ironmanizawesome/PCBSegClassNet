"""
PCB 배선(trace) segmentation용 U-Net — Stage 2, Step 3.

표준 U-Net (encoder-decoder + skip connection). 입력 RGB 3채널, 출력 1채널 logit
(이진 trace mask). 손실 계산은 BCEWithLogitsLoss/Dice 쪽에서 sigmoid 처리.

- pool 4회 → 입력 H,W는 16의 배수여야 함 (예: 512).
- base 채널로 모델 크기 조절 (base=32 기본; 4060 Ti 8GB에서 512 입력·작은 batch OK).
"""
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(conv 3x3 -> BN -> ReLU) x2."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 1, base: int = 32):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        # encoder
        self.enc1 = DoubleConv(in_ch, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.enc4 = DoubleConv(base * 4, base * 8)
        self.bottleneck = DoubleConv(base * 8, base * 16)

        # decoder (transpose conv upsample + skip concat)
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.head = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        c1 = self.enc1(x)
        c2 = self.enc2(self.pool(c1))
        c3 = self.enc3(self.pool(c2))
        c4 = self.enc4(self.pool(c3))
        b = self.bottleneck(self.pool(c4))

        d4 = self.dec4(torch.cat([self.up4(b), c4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), c3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), c2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), c1], dim=1))
        return self.head(d1)   # logits (B,1,H,W)
