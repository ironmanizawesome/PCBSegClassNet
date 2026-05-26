# Eagle / Gerber 파일 디렉토리

## Arduino Leonardo Gerber 파일 확보 방법

Arduino Leonardo는 오픈소스 하드웨어다. 공식 설계 파일은 아래 경로에서 받을 수 있다.

### GitHub에서 직접 다운로드

1. https://github.com/arduino/ArduinoCore-avr 접속
2. `variants/leonardo/` 디렉토리 → Eagle `.brd` 파일 확인
3. 또는 릴리즈 페이지에서 `ArduinoCore-avr-x.x.x.zip` 다운로드

### Gerber 파일이 이미 있는 경우

Arduino 공식 리포지토리 일부 릴리즈는 `/extras/` 하위에 Gerber 파일을 포함한다.

### Eagle .brd → Gerber 변환 (직접 생성)

1. Eagle CAD (무료 버전 가능) 또는 KiCad에서 `.brd` 파일 열기
2. File → CAM Processor → `gerb274x.cam` 실행
3. 출력 파일:
   - `*.GTL` — Top Copper Layer    ← 앞면 배선 GT
   - `*.GBL` — Bottom Copper Layer ← 뒷면 배선 GT
   - `*.GBO` — Board Outline

### 이 디렉토리에 놓을 파일

```
eagle/
├── README.md            ← 이 파일
├── Leonardo.brd         ← Eagle 보드 파일 (옵션)
├── Leonardo.sch         ← Eagle 회로도 파일 (옵션)
└── gerber/
    ├── Leonardo.GTL     ← 앞면 구리 레이어 Gerber
    ├── Leonardo.GBL     ← 뒷면 구리 레이어 Gerber
    └── Leonardo.GBO     ← 보드 외곽선
```

## GT 마스크 생성 실행 방법

Gerber 파일을 위 경로에 배치한 후:

```bash
# 앞면 GT 마스크 생성
python src/segmentation/gerber_to_mask.py \
    --input eagle/gerber/Leonardo.GTL \
    --output data/masks/gt_front.png \
    --dpi 500

# 뒷면 GT 마스크 생성 (좌우 반전 적용)
python src/segmentation/gerber_to_mask.py \
    --input eagle/gerber/Leonardo.GBL \
    --output data/masks/gt_back.png \
    --dpi 500 --flip-horizontal

# PCB 사진 크기에 맞춰 리사이즈 (1000x750 사진 기준)
python src/segmentation/gerber_to_mask.py \
    --input eagle/gerber/Leonardo.GTL \
    --output data/masks/gt_front.png \
    --dpi 500 --target-width 1000 --target-height 750
```

## 주의사항

- Gerber → 사진 정합(image registration)은 별도 작업 필요
  - DPI 기반 단순 리사이즈: 보드 크기가 동일하면 근사 가능
  - 정확한 정합: 패드 위치 기준 homography 변환 사용
- Arduino Leonardo 보드 크기: 약 68.6mm × 53.4mm
