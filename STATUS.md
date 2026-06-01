# 프로젝트 현황 — PCB 사진 → 회로도(netlist) 복원

> 최종 업데이트: 2026-06-02 · 사람이 읽는 **통합 현황 스냅샷**.
> (코드 규범은 각 폴더 `CLAUDE.md`, trace_seg 작업지시서는 `trace_seg/stage2_trace_segmentation_brief.md`.)

## 목표
PCB 기판 **사진**에서 **회로도(netlist)**를 복원한다. monorepo 구성: `component_seg/`(부품 검출) + `trace_seg/`(배선 추출).

## 파이프라인 3단계 현황

| Stage | 내용 | 상태 |
|---|---|---|
| 1. 부품 검출 | 사진 → 부품 위치/종류 (`component_seg`) | 부분 · 별개 진행 · **미통합** |
| 2. 배선 추출 | 사진 → 구리 trace (`trace_seg`) | ✅ **됨** (U-Net) |
| 3. 연결 → netlist | 부품+배선 합쳐 netlist | ⚠️ **절반** (배선→net 연결성만, 부품 미통합) |

## Stage 1 — 부품 검출 (`component_seg`)
- TF 기반 PCBSegClassNet + FPIC 25클래스. FPIC 데이터셋 접근 **대기 중**.
- 자체 Arduino 데이터 transfer learning: segmentation IoU ~0.71 수준.
- **trace_seg와 통합 안 됨**(데이터·보드·스택 별개). EAGLE `.brd`→ref-des 맵 도구(`component_seg/scripts/brd_refdes_map.py`)로 annotation 보조.

## Stage 2 — 배선 추출 (`trace_seg`) ✅
- **데이터**: Arduino 6설계 × {front,back} × {직접촬영, 공식웹} = **정합(homography) 21쌍**. 정답(GT)은 `.brd`/Gerber에서 추출.
- **결과**:
  - 전통 CV(HSV): IoU ~10% → 구조적 한계 확인.
  - **U-Net**(PyTorch, BCE+Dice, augmentation, letterbox): **처음 보는 설계 IoU 70~91%** (5-fold leave-one-design-out 교차검증). HSV 대비 대폭 향상.
- 핵심 코드: `trace_seg/src/models/unet.py`, `src/training/{dataset,train}.py`, `scripts/{generate_gt_masks,register_photo,evaluate_step1,eval_unet}.py`.

## Stage 3 — netlist 복원 (부분)
- **`trace_seg/scripts/netlist_poc.py`**: 예측 trace → connected-components로 **net(연결 덩어리) 분리** → CAD 정답 net과 대조.
- 다보드(각 보드 holdout 모델) net 복원율: **NANO 93% / UNO_TH 86% / UNO_SMD 77% / Leonardo 66%** — 전부 **short≈0**(잘못된 연결 거의 없음). **MEGA만 22%**(가장 dense, thin trace 뭉개짐).
- **한계(중요)**: 이건 Stage 3의 **배선 쪽 절반**. **부품 핀↔net 매핑(진짜 netlist) 미구현**, Stage 1 부품검출 통합도 안 됨.

## 정직한 범위·한계
- **좁은 도메인**: Arduino 녹색 보드 ~5설계. 임의 PCB 일반화 불가.
- **정합 입력 의존**: 모델은 homography 정합된 입력에만 동작. **생 폰 사진은 안 됨** → 라이브 Telegram 봇(`trace_seg/scripts/telegram_demo.py`)은 비viable, scaffolding으로만 보관.
- **단일 레이어** net 분리(via/양면 연결 미포함). netlist→회로도 작도 미구현.

## 발표용 메시지 (정직)
- ✅ 말할 수 있음: "전통 CV로 안 되던 trace 추출을 학습으로 해냈고(처음 보는 보드 70~91%), 그 배선에서 net 위상까지 대부분 복원(short≈0) — 좁은 도메인에서 netlist 복원 **가능성 실증**."
- ❌ 말하면 안 됨: "임의 사진에서 회로도 자동 복원", "부품+배선 통합 netlist 완성".

## 다음 후보
1. **Stage 3 완성도↑**: CAD pad 위치 → pad→net 매핑 → CAD netlist 대조("부품 핀이 net으로 묶임" 실증). 부품은 우선 CAD로 대체.
2. **데이터 확장**(범용화의 유일한 레버): 합성 데이터(CAD 렌더 + domain randomization) / 단면 보드 설계 추가.
3. 발표 자료 정리(CV 표 + net맵 비교 + caveat 1장).

## 재현 (요약)
```bash
# Stage 2 평가 / 학습
python trace_seg/scripts/evaluate_step1.py
python trace_seg/src/training/train.py --images data/images_registered --val-board NANO_BOARD_A000005
# Stage 3 net 복원 + baseline
python trace_seg/scripts/netlist_poc.py --board UNO_R3_BOARD_A000066 --ckpt data/unet/ho_UNO_TH/unet_best.pt
```
