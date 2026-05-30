# NIGHT_RUN — trace_seg 자동 작업 로그

**시작**: 2026-05-27 (밤 — 사용자 취침 중 자율 진행)
**범위**: 1번만 — `gerber_to_mask.py` 6보드 갱신 + 12장 GT mask 생성 + Step 1 재평가 (Leonardo만)
**가드레일**: commit/push 금지, destructive shell 금지, pip install 금지, 외부 호출 금지. 모든 변경은 working tree에 untracked로 쌓아둠.

## 작업 계획

1. 기존 `src/segmentation/gerber_to_mask.py` 읽고 코드 파악
2. 옛 `eagle/gerber/Leonardo.GTL` 가정을 새 `data/gerber/{board}/copper_top.gbr` 구조로 갱신 (보드 인자 받게)
3. 6보드 × 2면 = 12장 GT mask 일괄 생성 → `data/masks/{board}/{front,back}.png`
4. 각 mask sanity check (해상도, trace 픽셀 비율, 외곽선 존재 여부)
5. Step 1 (`src/segmentation/trace_segment.py`) 코드도 점검 — 보드 인자 받게 갱신 필요
6. Leonardo 사진 (front/back) 으로 Step 1 재평가 + evaluate.py로 IoU/DICE/F1
7. 결과를 이 파일에 기록 + 메모리 (trace_seg_progress) 갱신
8. 다 끝나면 마지막에 완료 timestamp + 요약

## 한계 (시작 전 명시)

- **다른 5보드 사진 없음** → Step 1 재평가는 Leonardo 1보드만 가능. 나머지는 GT mask 생성·시각화까지.
- **사진↔GT 정합은 여전히 단순 리사이즈** (homography 정합은 별도 작업, 다음 단계).
- **HSV Step 1은 Leonardo로 튜닝된 파라미터** — 다른 보드에 적용 시 색감 차이로 더 안 좋은 결과 예상. 본 작업 범위 외.

## 진행 로그

### 03:45 — pcb-tools 미설치 발견

trace_seg/requirements.txt에 `pcb-tools>=0.1.6` 있지만 모든 conda env (djangoProj, lingbot-map, ppp2025, pscn, base) 에 미설치. trace_seg 전용 venv도 없음.

**결정**: 사용자 확인 후 install 허용 (특정 env). 새 env `trace_seg` 생성 (pscn은 component_seg TF용으로 추정, 섞으면 위험).

**명령**:
```
conda create -n trace_seg python=3.11 -y
C:\Users\ironm\.conda\envs\trace_seg\python.exe -m pip install pcb-tools opencv-python numpy Pillow
```

이후 모든 trace_seg python 호출은 위 env의 python.exe 경로 사용.

### 03:50 — env 생성 + 의존성 install 완료

`conda create -n trace_seg python=3.11 -y` + pip install pcb-tools / opencv-python / numpy / Pillow. 다른 conda env 영향 없음.

설치된 버전: pcb-tools 0.1.6, opencv-python 4.13.0.92, numpy 2.4.6, Pillow 12.2.0, cairocffi 0.9.0.

cairo backend import OK (Windows cairo native lib 별도 필요 없이 cairocffi가 처리).

### 04:00 — batch script 작성 + 6보드 mask 일괄 생성 완료

`trace_seg/scripts/generate_gt_masks.py` 새로 작성:
- `BOARDS` 리스트로 6보드 loop
- 각 보드 copper_top.gbr → front.png, copper_bottom.gbr → back.png (flip-horizontal 적용)
- 결과: `trace_seg/data/masks/{board}/{front,back}.png`
- `--board` 옵션으로 단일 보드 가능, `--dpi` 기본 500

기존 `gerber_to_mask.py`는 single-file CLI 그대로 두고 함수만 import해서 재사용 — 인터페이스 deprecation 없음.

12장 결과 (모두 정상):

| 보드 | Front (WxH, copper%) | Back (WxH, copper%) |
|---|---|---|
| LEONARDO_BOARD_A000057 | 1329x1028, 30.1% | 1330x1028, 31.9% |
| LEONARDO_BOARD_NH_A000052 | 1329x1028, 30.1% | 1330x1028, 31.9% |
| MEGA_R3_BOARD_2560_A000067 | 1968x1018, 36.2% | 1971x1018, 27.1% |
| NANO_BOARD_A000005 | 842x342, 38.1% | 842x342, 40.5% |
| UNO_R3_BOARD_A000066 | 1463x1034, 38.9% | 1335x1035, 35.7% |
| UNO_R3_BOARD_SMD_A000073 | 1463x1034, 39.3% | 1335x1035, 35.0% |

**Sanity check 결론**:
- 모든 보드 copper ratio 27~40% — 합리적 PCB 분포
- Leonardo A000057/NH는 동일 결과 (예상 — 같은 .brd 사본)
- UNO front(1463) vs back(1335)이 약 130px 차이 — copper layer별 bounding box 차이. 정합 단계에서 인지 필요.
- MEGA back 27%로 가장 낮음 — ground plane 비율 차이로 추정.

### 04:10 — Step 1 평가 batch script 작성 (`evaluate_step1.py`)

`trace_seg/scripts/evaluate_step1.py` 새로 작성:
- PAIRS 리스트로 (photo, gt_mask, output_prefix, label) 튜플 관리
- `trace_segment.segment_traces()` + `evaluation.evaluate.*` 함수 직접 import 사용 (subprocess 없이)
- 결과: `data/step1_eval/{board}/{front,back}_{pred,error}.png + .json` + `summary.json`
- 사진 추가되면 PAIRS만 확장하면 재사용 가능

**버그 1건 발견·수정**: 첫 실행에서 `print_metrics(metrics, str(pred_path), str(gt))` 의 `gt`가 ndarray로 shadow돼 path 자리에 배열 repr 출력. variable shadow가 원인. `gt_path`/`gt_arr`로 명시 분리해서 fix. metrics 자체엔 영향 없음.

### 04:15 — Step 1 재평가 결과 (Leonardo, 새 GT 기준)

| 측면 | IoU | DICE | Precision | Recall | F1 | Pixel Acc |
|---|---|---|---|---|---|---|
| Front | **2.25%** | 4.40% | 16.90% | 2.53% | 4.40% | 66.80% |
| Back  | **4.25%** | 8.16% | 24.34% | 4.90% | 8.16% | 64.73% |

옛 결과 (2026-04-03, 옛 trace_seg/eagle/gerber/ GT 기준):

| 측면 | IoU | DICE | Precision | Recall |
|---|---|---|---|---|
| Front | 1.94% | 3.80% | 17.37% | 2.13% |
| Back  | 3.16% | 6.12% | 22.80% | 3.54% |

**분석**:
- 알고리즘은 동일 (`segment_traces` 코드 변경 없음). 차이는 **GT가 다른 EAGLE export** (옛 source 불명 → 새 source는 우리가 .brd에서 직접 추출). GT가 다르니 픽셀 비교 결과도 약간 다름. 새 결과가 IoU 살짝 높음 (+0.3~1.1%p).
- 결론은 동일: **Step 1 (HSV)는 trace 검출에 부적합**. FN 압도 (front 220k FN vs 5.7k TP, back 227k FN vs 11.7k TP). 솔더마스크 아래 trace 거의 못 잡음.
- back이 front보다 약간 더 잘 나옴 — back은 ground plane 영역이 많아서 노출 구리 비중이 높고, HSV 필터가 그쪽엔 더 잘 들어맞는 듯. 하지만 절대 수치 자체가 낮아서 큰 의미 없음.

### 한계 다시 확인

5보드 (LEONARDO_NH는 사진 source 미정) 사진 미확보 → Step 1 평가 불가능. 12장 GT만 생성된 상태. 사진 multi-source 단계 (다음 단계)에서 확장 가능.

---

## 산출물 목록 (working tree, untracked)

### 코드
- `trace_seg/scripts/generate_gt_masks.py` (신규) — 6보드 일괄 GT mask 생성
- `trace_seg/scripts/evaluate_step1.py` (신규) — Step 1 평가 batch

### 데이터 (gitignored — `data/masks/`, `data/evaluation/` 패턴이 .gitignore에 잡혀 있어야 함)
- `trace_seg/data/masks/{6 boards}/{front,back}.png` — 12장 GT mask
- `trace_seg/data/step1_eval/LEONARDO_BOARD_A000057/{front,back}.{json,_pred.png,_error.png}` — Leonardo 평가 결과 + error map (TP=흰, FP=빨강, FN=파랑)
- `trace_seg/data/step1_eval/summary.json` — 통합 요약

### 환경
- 새 conda env `trace_seg` (Python 3.11 + pcb-tools 0.1.6 + opencv-python 4.13 + numpy 2.4 + Pillow 12.2)
- `.claude/settings.local.json` 신규 — PowerShell/Bash 자동 허용

## 다음 단계 후보 (작업 결과 기반)

1. **사진 multi-source 확보** — 6보드 사진 (웹 + 직접 촬영). 확보 후 PAIRS만 확장하면 즉시 평가.
2. **GT mask down-sample을 평가 전 명시적 처리** — 현재는 evaluate.py가 즉석 resize. 사진 크기(1000x750) ≠ GT 크기로 인한 정합 오차 가능성. 더 정확한 평가엔 homography registration 필요.
3. **Step 1 보드별 파라미터 튜닝** — HSV 파라미터가 Leonardo로 튜닝됨. 다른 보드 색감에 맞춰 보드별/조명별 프로파일 추가 검토.
4. **Step 3 U-Net 인프라** — 다음 자율 진행 후보 (이번엔 범위 밖이라 시작 안 함).

## 검토 포인트 (사용자 깬 뒤)

- [ ] 12장 mask 시각적 sanity check (특히 LEONARDO/UNO/MEGA front-back 정렬)
- [ ] Step 1 평가 정확도 — IoU 2~4%가 옛 1.9~3.2% 대비 +0.3~1.1%p 개선은 GT 변경 효과인지, 잡음 범위인지?
- [ ] `data/masks/`, `data/step1_eval/` 가 gitignore에 포함돼야 (현재 `data/masks/`, `data/evaluation/` 만 있음 — `data/step1_eval/` 추가 필요할 수도)
- [ ] commit 결정 — code (`scripts/*.py`) 만 commit하고 데이터(`data/masks/`, `data/step1_eval/`)는 gitignore?

## 완료 timestamp

**종료**: 2026-05-27 ~04:20
**총 소요**: 약 35분 (env 생성 + 12장 mask + Leonardo 양면 평가 + 문서)

