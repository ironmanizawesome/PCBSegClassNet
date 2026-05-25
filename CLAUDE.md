# CLAUDE.md

이 레포는 monorepo다. `component_seg/`와 `trace_seg/` 두 sub-project가 공존한다.
구조 상세와 sub-project 요약은 auto-memory의 `monorepo_structure.md` 참조.

## 작업 원칙 (양쪽 공통)

- **한 번에 한 sub-project**: 두 폴더는 의존성·기술 스택이 다름 (TF vs PyTorch). 작업 시 해당 폴더로 `cd` 후 진행.
- **각 sub-project의 CLAUDE.md 우선**: 명령어, 영역 전용 규범은 해당 폴더의 CLAUDE.md를 참조. 본 root는 monorepo-level 공통 규범만.
- **Cross-cutting 변경은 양쪽 영향 확인 필수**: 공통 문서/JSON 스키마/`.gitignore` 변경은 두 sub-project에 영향. review 후 진행.

## 공통 금지사항

핵심 5개. 더 자세한 근거는 `project_outline.md` section 8 참조.

- **다층 PCB를 기본 가정으로 두지 X**: 대상은 단면/양면 단순 보드. 다층은 사진으로 trace 추출 불가.
- **회로도 이미지를 정답지(ground truth)로 사용 X**: GT는 항상 원본 설계 파일(`.sch`, `.brd`)에서 추출.
- **Track A(정답지 생성) 안정화 우선**: Track B(이미지 복원)보다 앞서 안정화. 평가 기준선 없이 복원 성능 해석 불가.
- **모든 단계 동일 JSON 스키마 유지**: Track A/B 출력은 같은 스키마라야 비교 가능.
- **확인 안 된 EAGLE/KiCad 필드 단정 X**: 파싱 시 미확인 필드는 명시적 검증 후 사용.

## 공통 코드 스타일

- Python: 함수/변수는 `snake_case`, 클래스는 `PascalCase`
- 모든 스크립트는 `argparse` CLI 인터페이스 의무
- 주석 한국어 OK, 코드/변수명은 영어
- 결과물 명명 일관성: 파싱 JSON은 `{원본명}_parsed.json`, 마스크는 `*_mask.png` 등
- 한 파일에 하나의 주요 기능만 담을 것

## 사용자/환경 정보

상세는 auto-memory의 `user_profile.md`, `env_setup.md`, `colab_setup.md` 참조.
