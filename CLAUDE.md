# CLAUDE.md

이 레포는 **monorepo**다. 두 개의 독립 sub-project가 공존하며, 각자
자체 코드/데이터/CLAUDE.md를 가진다.

## 구조

```
component_seg/   Track B 1단계: 부품 검출/세그멘테이션
                 TensorFlow 2.x. PCBSegClassNet 기반 (FPIC 25 클래스).
                 자체 CLAUDE.md, requirements.txt, src/, data/, ...

trace_seg/       Track B 3단계: 배선(trace) 세그멘테이션
                 PyTorch. 자체 개발. 자체 CLAUDE.md, requirements.txt, src/, data/, ...

project_outline.md  프로젝트 전체 outline (Track A 회로도 파싱 / Track B 이미지 복원 /
                    Track C 평가).
```

## 작업 원칙

- **한 번에 한 sub-project**: 두 폴더는 의존성·기술 스택이 다름. 작업 시
  해당 폴더로 `cd` 후 진행.
- **각 sub-project의 CLAUDE.md 우선**: 명령어, 환경 설정, 아키텍처 정보는
  해당 폴더의 CLAUDE.md를 참조 — 본 root CLAUDE.md는 monorepo 레벨 정보만.
- **공통 문서는 root에**: `project_outline.md`, `LICENSE`, `.gitignore`는
  두 프로젝트에 모두 적용되는 cross-cutting 문서.

## Sub-project 요약

### component_seg/
- 입력: PCB 사진
- 출력: 부품별 segmentation mask (25 클래스)
- 환경: TF 2.10.1 (Win native) 또는 TF 2.15 (Colab)
- 상세: [component_seg/CLAUDE.md](component_seg/CLAUDE.md)

### trace_seg/
- 입력: PCB 사진
- 출력: 구리 배선(trace) binary mask
- 환경: PyTorch + OpenCV
- 상세: [trace_seg/CLAUDE.md](trace_seg/CLAUDE.md)

## 데이터/체크포인트는 ignored

각 sub-project 안의 `data/`, `dataset/`, `*.h5`, `*.zip` 등은 모두
`.gitignore` 대상. 트래킹되는 건 코드, config, 작은 annotation JSON 정도.
