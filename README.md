# PCB 회로도 복원 (Monorepo)

PCB 기판 사진으로부터 회로도(netlist)를 복원하는 프로젝트.
전체 목표·트랙 구조는 [project_outline.md](project_outline.md) 참조.

## 구조

```
.
├── component_seg/    Track B 1단계: 부품 검출/세그멘테이션 (TF, PCBSegClassNet 기반)
├── trace_seg/        Track B 3단계: 배선(trace) 세그멘테이션 (PyTorch, 자체 개발)
├── project_outline.md  프로젝트 전체 outline (Track A/B/C)
└── README.md
```

각 sub-project는 자체적인 `README.md`, `CLAUDE.md`, `requirements.txt`를
가짐. 두 프로젝트는 의존성·기술 스택이 분리됨 (TF vs PyTorch) — 한쪽에서
작업할 때는 해당 폴더로 `cd` 후 진행.

## 빠른 시작

```bash
# 부품 검출 작업
cd component_seg
# (component_seg/README.md 참조)

# 배선 검출 작업
cd trace_seg
# (trace_seg/README.md 참조)
```

## History

- `component_seg/`: 원래 [CandleLabAI/PCBSegClassNet](https://github.com/CandleLabAI/PCBSegClassNet) fork에서 출발, Arduino transfer learning 작업 추가
- `trace_seg/`: 자체 개발 (subtree merge로 통합, history 보존)
