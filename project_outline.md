# PCB 사진 기반 회로도 복원 — 프로젝트 Outline

> Claude Code용 참고 문서.
> 진행 상태/대기 사항은 제외하고, **입력 데이터 → 사용 기술 → 산출물 → 목표 달성 경로**만 정리.

---

## 1. 최종 목표

PCB 기판 사진을 입력으로 받아 회로도(논리 연결관계)를 복원한다.
복원 결과는 동일 보드의 원본 설계 파일에서 추출한 ground truth와 비교해 정확도를 측정한다.

```
[PCB 사진]  ──►  [복원 회로도]  ──┐
                                  ├──►  [비교/평가]
[원본 설계 파일]  ──►  [정답 회로도]  ──┘
```

---

## 2. 전체 파이프라인

본 프로젝트는 두 개의 독립 파이프라인이 마지막 단계에서 합쳐지는 구조다.

- **Track A (정답지 생성)**: 회로도 원본 파일 → 구조화된 netlist/graph
- **Track B (이미지 복원)**: PCB 사진 → 추정 netlist/graph
- **Track C (평가)**: A와 B의 결과를 동일한 표현 위에서 비교

현재 우선순위는 **Track A**이며, 이는 평가 기준이 없으면 Track B의 결과 해석이 불가능하기 때문이다.

---

## 3. Track A — Ground Truth 생성

### 3.1 입력 정보
- 보드 모델명 (예: `arduino-uno`, `arduino-leonardo`)
- 해당 모델의 회로도 원본 파일
  - 1차 대상: EAGLE `.sch` (XML 기반)
  - 확장 대상: KiCad `.kicad_sch` (S-expression 기반)

### 3.2 사용 기술
| 단계 | 기술 / 도구 |
|---|---|
| EAGLE `.sch` 파싱 | Python `xml.etree.ElementTree` (또는 `lxml`) |
| KiCad 파싱 (추후) | S-expression 파서 (`sexpdata` 등) |
| 데이터 구조화 | Python dict → JSON 직렬화 |
| 그래프 표현 (선택) | `networkx` |

### 3.3 추출 대상
- **Part**: 부품 ID, 이름, 라이브러리, value, package
- **Pin**: 부품-핀 매핑 (어느 부품의 몇 번 핀인지)
- **Net**: 전기적으로 동일한 연결선의 이름과 멤버
- **Connection**: `(part, pin) ↔ net` 관계 집합

### 3.4 출력 포맷 (표준 JSON 스키마)
```json
{
  "board": "arduino-uno",
  "source": "eagle/arduino-uno.sch",
  "parts": [
    {"id": "U1", "name": "ATMEGA328P", "package": "TQFP32", "value": "ATMEGA328P-AU"}
  ],
  "nets": [
    {"name": "VCC", "pins": [{"part": "U1", "pin": "4"}, {"part": "C1", "pin": "1"}]}
  ],
  "graph": {
    "nodes": ["U1.4", "C1.1", ...],
    "edges": [["U1.4", "C1.1", {"net": "VCC"}], ...]
  }
}
```
- 모든 보드/툴(EAGLE/KiCad)에서 동일 스키마로 통일
- 파일명 규칙: `{원본파일명}_parsed.json`

### 3.5 폴더 구조
```
eagle/                 # 원본 EAGLE .sch 파일
eagle_parsed/          # 파싱 결과 JSON
kicad/                 # 원본 KiCad .kicad_sch (확장 시)
kicad_parsed/          # 파싱 결과 JSON
```

---

## 4. Track B — 이미지 기반 복원 (후속)

### 4.1 입력 정보
- PCB 기판 사진 (단면/양면 단순 보드 우선, 다층 PCB 제외)
- 가능 시 보드 모델 식별 정보 (OCR로 실크스크린 인식)

### 4.2 예상 처리 단계
1. **부품 검출**: PCB 상의 부품 위치/종류 탐지 (object detection)
2. **패드/핀 검출**: 부품 핀의 PCB 상 좌표 추출
3. **트레이스 추출**: 동박 배선 segmentation
4. **연결 추론**: 트레이스 연결성을 따라 핀들을 net 단위로 그룹화
5. **JSON 출력**: Track A와 동일한 스키마로 변환

### 4.3 사용 기술 후보
- 부품 검출: YOLO 계열, 또는 템플릿 매칭(초기)
- 트레이스 segmentation: 고전 이미지 처리(이진화 + 연결성분) 또는 U-Net 계열
- 보드 식별: OCR (Tesseract / PaddleOCR) + 실크스크린 문자열 매칭

> 데이터셋 규모가 작을 가능성이 높으므로 초기에는 **규칙 기반 / 구조 기반** 접근을 우선하고,
> 학습 기반은 검증 가능한 평가 루프가 생긴 이후에 도입한다.

---

## 5. Track C — 비교 및 평가

### 5.1 비교 대상
- Track A 결과 (정답 graph/netlist)
- Track B 결과 (복원 graph/netlist)

### 5.2 평가 지표 (정의 예정)
- **Net-level**: net 단위 precision / recall / F1
- **Component-level**: 부품 식별 정확도
- **Pin-mapping**: `(part, pin) ↔ net` 매칭 정확도
- **Graph similarity**: graph edit distance, 또는 subgraph isomorphism 기반 매칭

### 5.3 비교 가능 조건
- 두 결과가 **동일한 JSON 스키마**로 표현되어야 함
- 부품 ID 정규화(reference designator 기준)가 선행되어야 함

---

## 6. 데이터셋 범위

| 항목 | 내용 |
|---|---|
| 우선 대상 | Arduino UNO, Arduino Leonardo |
| 확장 대상 | Arduino Nano, Mega |
| 제외 대상 | 다층 PCB (내부 레이어 정보를 사진으로 확인 불가) |
| 자료 출처 | 각 보드의 공식 EAGLE / KiCad 설계 파일 |
| 파일 종류 | `.sch` 우선, `.brd`는 보조 (물리 배치용) |

---

## 7. 작업 우선순위 (현재 기준)

1. **EAGLE `.sch` 파서 안정화** — `eagle/` → `eagle_parsed/` 일괄 처리
2. **JSON 스키마 표준화** — 위 3.4 형식으로 고정
3. **Arduino 계열 데이터셋 수집/정리** — 보드별 출처 표 작성
4. **평가 지표 구현** — 두 JSON 간 비교 함수
5. **Track B 착수** — 가장 단순한 부품 검출부터

---

## 8. 설계 원칙 (위반 금지)

- 다층 PCB를 기본 가정으로 두지 않는다.
- 회로도 **이미지**를 정답지로 쓰지 않는다 (원본 설계 파일만 사용).
- Track B보다 Track A 안정화가 우선이다.
- 출력은 모든 단계에서 동일한 JSON 스키마를 유지한다.
- 파싱이 불안정한 상태에서 학습 모델 설계를 확장하지 않는다.
- 확인되지 않은 EAGLE/KiCad 필드는 단정해서 사용하지 않는다.
