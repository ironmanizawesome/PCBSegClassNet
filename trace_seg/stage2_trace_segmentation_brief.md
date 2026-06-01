# Stage 2: PCB 배선(Trace) 세그멘테이션 작업 지시서

## 프로젝트 배경
PCB 기판 사진으로부터 회로도를 복원하는 전자공학종합설계 프로젝트.
전체 파이프라인은 3단계로 구성:
- Stage 1: 부품 검출/세그멘테이션 (FPIC 데이터셋 확보 대기 중)
- **Stage 2: 배선(trace) 세그멘테이션 ← 현재 작업**
- Stage 3: 부품-배선 연결관계 추론 → netlist 복원

## 현재 상황
- Stage 1은 FPIC 데이터셋 저자(Navid Asadizanjani, UF)에게 접근 요청 메일 발송 후 대기 중
- Stage 2를 먼저 진행하기로 결정
- Arduino 보드 사진은 확보된 상태
- 배선 세그멘테이션용 공개 모델/데이터셋은 사실상 없음 (기존 연구는 대부분 X-ray CT 기반)

## 작업 목표
Arduino 보드의 광학 사진에서 구리 배선(trace)을 픽셀 단위로 세그멘테이션하는 것

## 접근 전략 (순서대로)

### Step 1: 전통적 이미지 처리 기반 프로토타입
- 보드 사진을 HSV 색 공간으로 변환
- 솔더 마스크(녹색)와 구리(금색/은색) 색상 차이를 이용한 색상 기반 필터링
- threshold 세그멘테이션으로 트레이스 영역 분리
- morphological operation으로 노이즈 제거 및 연결성 개선
- 결과를 이진 마스크로 출력

### Step 2: Ground Truth 생성 (가능하면)
- Arduino 보드의 Gerber 파일(오픈소스)에서 copper layer 정보 추출
- Gerber → 이미지 변환으로 트레이스 마스크 자동 생성
- 이를 정답지로 사용하여 Step 1 결과의 정확도 측정

### Step 3: 딥러닝 모델 학습 (Step 1,2 안정화 이후)
- U-Net 기반 세그멘테이션 모델
- Step 2에서 만든 ground truth로 학습
- Step 1의 전통적 방법과 성능 비교

## 작업 원칙
- 다층 PCB는 대상에서 제외 (외부 사진으로 내부 레이어 파악 불가)
- Arduino 계열 단순 보드부터 진행
- 파싱이 불안정한 상태에서 후속 단계를 과하게 확장하지 말 것
- Step 1이 충분히 동작하는지 먼저 확인 후 Step 2,3으로 진행

## 먼저 해야 할 일
1. 업로드된 Arduino 보드 사진을 확인
2. HSV 변환 + 색상 기반 필터링으로 트레이스 영역 분리 시도
3. 결과 이미지를 보면서 파라미터 조정
4. 이진 마스크 출력 및 품질 평가

## 참고 논문/자료
- PCBSegClassNet (Makwana et al., 2023) - 부품 세그멘테이션, FPIC 데이터셋 사용
- DCNN-GC (Qiao et al., 2018) - PCB wire segmentation, DCNN + Graph Cut
- "Towards PCB Netlist Extraction from Multimodal Imagery" (Balint et al., 2022) - FPN 기반 세그멘테이션 → netlist
- Botero et al. (2021) - X-ray CT 기반 trace/copper plane 추출, 그래프 이론 활용
- Stanford EE368 PCB Reverse Engineering - 광학 이미지 이진화 기반 접근

---

## Stage 2 진행 결과 (2026-06-01)

위 지시서 이후 실제 수행 결과. 상세 진행 이력은 auto-memory `trace_seg_progress.md` 참조.

### Step 1 (HSV 전통 CV) — 완료, 한계 정량화
- `src/segmentation/trace_segment.py` (CLAHE + HSV 색 필터 + morphology).
- 6보드 평가 IoU 1~15% (recall 압도적 미달 = 솔더마스크 아래 trace 미검출). 전통 색 필터의 구조적 한계 확인.

### Step 2 (Gerber → GT mask) — 완료
- `src/segmentation/gerber_to_mask.py` + `scripts/generate_gt_masks.py`. 6보드 × 양면 이진 GT mask.

### 데이터셋 (사진 ↔ GT 정합 포함)
- 직접촬영 + 공식 web(roboflow) 사진을 `scripts/organize_photos.py`로 정리 (HEIC→PNG, side/비대상 보드 제외).
- 사진↔GT **수동 homography 정합** `scripts/register_photo.py` — 자동 보드검출은 검은 헤더가 배경과 섞여 실패 → 수동 대응점 채택.
- 최종 **21쌍 / 6설계** (photo 9 + web 12). UNO SMD 직접촬영은 CAD 리비전 불일치 → 공식 web으로 대체. 다층/실드/어댑터 보드 제외.

### Step 3 (U-Net) — PoC 검증 완료
- `src/models/unet.py`, `src/training/{dataset,train}.py` (PyTorch, BCE+Dice, augmentation, letterbox, best-ckpt, cosine lr, grad clip).
- **5-fold leave-one-design-out 교차검증** (`scripts/eval_unet.py`로 trivial baseline 대조):

| held-out 설계 | model IoU | all-fg | mean-mask |
|---|---|---|---|
| NANO | 89.1% | 16.0% | 15.2% |
| MEGA | 70.2% | 16.3% | 15.2% |
| UNO R3 (TH) | 90.6% | 27.5% | 14.8% |
| UNO R3 (SMD) | 90.8% | 27.4% | 17.3% |
| Leonardo (A57+NH) | 87.6% | 23.9% | 10.3% |

- **결론**: 처음 보는 Arduino 설계에서 IoU 70~91% (baseline 3~6×) → **좁은 도메인 내 일반화 확증**(순수 과적합 아님). HSV(~10%) 대비 대폭 향상.
- **범위/한계**: 독립 설계 ~5개뿐 → 임의 PCB 전이 불가, **"Arduino 스타일 trace 추출기" PoC**로 규정. UNO TH/SMD는 형제보드라 상호 near-leakage(낙관), MEGA는 가장 복잡·distinct해 최약(70%).
- **확장 레버**: ① 단면/양면 + CAD 보드 설계 추가 ② 합성 데이터(CAD 렌더링 + domain randomization).
