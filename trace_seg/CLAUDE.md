# CLAUDE.md (trace_seg)

PyTorch 기반 PCB 배선(trace) segmentation. 단면/단순 보드 대상.
현재 진행 상태·결과 수치는 auto-memory의 `trace_seg_progress.md` 참조.
Gerber→GT 변환과 평가 메트릭은 `shared_gerber_pipeline.md`, `shared_eval_metrics.md` 참조.

## 작업 원칙 (영역 전용)

- **Stage 순차 진행 (Step 1 CV → Step 2 GT → Step 3 U-Net)**: 건너뛰기 X. 단계별 평가 수치 확보 후 다음 단계 착수.
- **결과 mask는 binary PNG**: 배경 0, 전경 255. 다른 인코딩 X.
- **파싱 결과 JSON 명명**: `{원본명}_parsed.json` 규칙 의무 (root CLAUDE.md 공통 코드 스타일).
- **이미지 정합 정밀도 인지**: 현재 단순 리사이즈로 PCB 사진과 GT mask 매칭. 정밀 정합(homography)은 다음 단계 작업 — 이전엔 픽셀 어긋남 존재함을 인지하고 결과 해석.

## 명령어

`cd trace_seg` 후 실행:

```bash
# EAGLE .sch 단일 파싱
python src/parsing/parse_sch.py --input eagle/board.sch --output eagle_parsed/board_parsed.json

# EAGLE .sch 일괄 파싱
python src/parsing/parse_all.py --input_dir eagle/ --output_dir eagle_parsed/

# 배선 segmentation (Step 1, 전통 CV)
python src/segmentation/trace_segment.py --input data/images/board.png --output data/masks/board_trace.png

# Gerber → GT mask 변환 (Step 2)
# 상세 사용법은 shared_gerber_pipeline.md 참조

# 결과 평가 (binary mask 두 장 비교, Step 1/2/3 모두 동일)
python src/evaluation/evaluate.py --pred data/masks/board_trace.png --gt data/masks/board_gt.png
```

## 영역 전용 금지사항

- **기존 trace_seg 진행 문서 임의 삭제/덮어쓰기 X**: 본 폴더의 README, brief 등은 작업 history. 정리할 때도 별도 검토.
- **`requirements.txt`에 대형 패키지 추가 X**: 현재 PyTorch + OpenCV + scikit-image 등 가벼운 구성 유지.
- **단일 보드 (UNO) 한정 학습으로 over-claim X**: 현재 Gerber 1세트만 검증됨. 다보드 일반화 주장은 별도 검증 후.
