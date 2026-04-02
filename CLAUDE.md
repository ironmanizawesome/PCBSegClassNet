# CLAUDE.md

PCB 기판 사진으로부터 회로도(schematic)를 복원하는 전자공학종합설계 프로젝트

## 기술 스택

- **언어**: Python 3.8+
- **이미지 처리**: OpenCV, scikit-image, Pillow
- **딥러닝**: PyTorch (세그멘테이션 모델 학습 시)
- **데이터 처리**: NumPy, pandas
- **시각화**: matplotlib
- **회로도 파싱**: xml.etree.ElementTree (EAGLE .sch 파일용)
- **패키지 매니저**: pip (requirements.txt)
- **데이터 포맷**: JSON (파싱 결과 저장), PNG (이미지/마스크)

## 디렉토리 구조

```
project/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── 01_project_summary.md      # 프로젝트 요약
├── 02_working_rules.md        # 작업 규칙 및 금지사항
├── 03_recent_context.md       # 최근 작업 맥락
├── 04_decision_log.md         # 의사결정 로그
├── 05_current_hypotheses.md   # 현재 가설
├── eagle/                     # 원본 EAGLE .sch 파일
├── eagle_parsed/              # .sch 파싱 결과 JSON
├── assets/
│   ├── papers/                # 참고 논문 PDF
│   ├── code/                  # 유틸리티 코드
│   ├── datasets/              # 데이터셋 설명
│   └── examples/              # 예시 입출력
├── data/
│   ├── images/                # PCB 보드 사진
│   └── masks/                 # 세그멘테이션 마스크 (생성 예정)
├── src/
│   ├── parsing/               # EAGLE .sch 파서
│   ├── segmentation/          # 배선/부품 세그멘테이션 코드
│   └── evaluation/            # 복원 결과 평가 코드
└── presentation/              # 발표 자료
```

## 빌드/실행

```bash
# 환경 설정
pip install -r requirements.txt

# EAGLE .sch 파일 파싱 (단일)
python src/parsing/parse_sch.py --input eagle/board.sch --output eagle_parsed/board_parsed.json

# EAGLE .sch 일괄 파싱
python src/parsing/parse_all.py --input_dir eagle/ --output_dir eagle_parsed/

# 배선 세그멘테이션 (전통적 방법)
python src/segmentation/trace_segment.py --input data/images/board.png --output data/masks/board_trace.png

# 세그멘테이션 결과 평가
python src/evaluation/evaluate.py --pred data/masks/board_trace.png --gt data/masks/board_gt.png
```

## 코드 스타일

- 함수명/변수명: snake_case
- 클래스명: PascalCase
- 한 파일에 하나의 주요 기능만 담을 것
- 파싱 결과는 항상 JSON으로 저장하며, 파일명은 `{원본명}_parsed.json`
- 이미지 마스크는 PNG 포맷, 배경 0 / 전경 255의 이진 마스크
- 주석은 한국어 가능, 코드와 변수명은 영어
- 각 스크립트는 `argparse`로 CLI 인터페이스 제공

## 현재 작업 우선순위

1. **Stage 2 — 배선 세그멘테이션 Step 3** (다음 단계)
   - Step 1(HSV 색상 기반): 완료. IoU 앞면 1.94% / 뒷면 3.16% — 구조적 한계 확인
   - Step 2(Gerber GT 생성): 완료. gt_front.png, gt_back.png (1000×750) 생성
   - 평가 파이프라인: 완료. evaluate.py — IoU/DICE/Precision/Recall/F1/오류맵
   - **Step 3: U-Net 기반 딥러닝 세그멘테이션 구현 예정**
     - 학습 데이터 부족 문제 선검토 필요 (단일 보드 이미지)
     - 이미지 정합(homography) 고도화 병행 검토

2. **Stage 1 — 부품 세그멘테이션** (데이터셋 대기 중)
   - FPIC 데이터셋 접근 요청 발송 완료 (UF Navid Asadizanjani 교수)
   - PCBSegClassNet 모델 코드 확보 완료

3. **EAGLE .sch 파서 안정화**
   - 단일 파일 파싱 → 일괄 파싱 자동화
   - 출력 JSON 구조 표준화

4. **비교 평가 파이프라인**
   - 복원 결과와 ground truth netlist 비교
   - IoU, DICE 등 평가 지표 구현 완료

## 금지사항

- 다층 PCB를 기본 대상으로 가정하지 말 것
- 회로도 이미지를 정답지(ground truth)로 사용하지 말 것 — 반드시 원본 설계 파일(.sch 등)에서 추출
- 이미지 복원보다 정답지 생성 파이프라인을 먼저 안정화할 것
- 확인되지 않은 파일 형식이나 필드를 단정하지 말 것
- 파싱이 불안정한 상태에서 후속 단계를 과하게 확장하지 말 것
- 프로젝트 범위를 갑자기 "일반 PCB 전체"로 확장하지 말 것
- 기존 문서(01~05, decision_log 등)를 임의로 삭제하거나 덮어쓰지 말 것
- requirements.txt에 불필요한 대형 패키지를 추가하지 말 것

## 핵심 맥락

- 대상 보드: Arduino UNO, Leonardo 등 단순 구조 오픈소스 보드
- EAGLE `.sch`는 XML 기반이라 Python xml 파싱으로 처리 가능
- `.brd`보다 `.sch`가 논리 연결관계 추출에 우선
- 부품 세그멘테이션과 배선 세그멘테이션은 분리하여 각각 수행 후 합치는 방식
- 초기에는 딥러닝보다 규칙 기반/전통적 이미지 처리 접근이 더 안정적
- 최종 목표: 이미지 기반 복원 결과를 .sch 파싱 기반 ground truth netlist와 비교
- pcb-tools는 Python 3.11에서 rs274x.read() 미사용 — gerber.loads(content) 방식으로 우회
- Python 실행 환경: C:\Users\ironm\AppData\Local\Programs\Python\Python311\python.exe (cv2 설치됨)
