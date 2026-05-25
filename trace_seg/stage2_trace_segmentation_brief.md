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
