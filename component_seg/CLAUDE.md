# CLAUDE.md (component_seg)

TF 2.x 기반 PCB 부품 segmentation/classification. PCBSegClassNet 코드베이스 + FPIC 25 클래스.
프로젝트 맥락은 auto-memory의 `project_goal.md`, `monorepo_structure.md`, `arduino_dataset.md` 참조.
아키텍처 상세는 `component_seg_architecture.md` 참조.

## 작업 원칙 (영역 전용)

- **학습/평가 명령은 `component_seg/src/`에서 실행**: 모든 yml의 상대경로가 이 cwd 기준 (`../data/`, `../checkpoints/` 등).
- **데이터 경로는 yml 내 상대경로 유지, 절대경로 박지 X**: Win native ↔ Colab 호환성 깨짐.
- **mask 색상 정확히 유지**: `src/data/dataloader.py::color_values`와 mask 생성 코드의 RGB tuple이 1:1 일치 필수. 불일치 시 dataloader one-hot 인코딩 깨짐 (`tf.reduce_all(tf.equal(...))` 매칭 실패).
- **Transfer learning은 `-resume` + 작은 lr (1e-5) 권장**: ModelCheckpoint reset 함정 인지 (상세 `training_baseline.md`).
- **Classification crop 생성은 `super_resolution.h5` 필요**: 보유 X면 segmentation only로 진행. Arduino transfer learning은 segmentation only 시나리오에 해당.

## 명령어

학습 (100 epoch 기준):
```bash
cd component_seg/src
python train_segmentation.py -opt cfs/pscn_seg.yml -epoch 100
python train_classification.py -opt cfs/pscn_class.yml -epoch 100
```

평가 (best checkpoint 로드, 학습 skip):
```bash
python train_segmentation.py -opt cfs/pscn_seg.yml -epoch 0
python train_classification.py -opt cfs/pscn_class.yml -epoch 0
```

Arduino fine-tune (transfer learning):
```bash
python train_segmentation.py -opt cfs/pscn_seg_arduino.yml -resume -epoch N
```

FPIC 원본 데이터 처리 (`component_seg/src/data/`에서):
```bash
python create_mask.py -i ../../data/pcb_image/ -a ../../data/smd_annotation/ \
       -id ../../data/segmentation/images -ad ../../data/segmentation/masks \
       -cd ../../data/classification/images/
python create_patches.py -i ../../data/segmentation/images/ \
       -m ../../data/segmentation/masks -cd ../../data/classification/images/ -ps 768
```

Arduino annotation 파이프라인 (repo root에서):
```bash
# 1. CVAT COCO export → FPIC CSV (폴더 재귀 스캔)
python component_seg/scripts/coco_to_fpic_csv.py \
       -i component_seg/PCBannotations -o component_seg/data/annotations
# 2. CSV + cropped 이미지 → 학습 데이터셋
python component_seg/scripts/build_seg_dataset.py \
       -a component_seg/data/annotations -i component_seg/dataset_cropped \
       -o component_seg/data/segmentation_arduino -ps 768
# 3. fine-tune (위 명령 참조)
```

## Configuration

YAML 위치: `component_seg/src/cfs/`
- `pscn_seg.yml` — FPIC 25-class segmentation 기본 (Adam lr=1e-4, batch=16, input 512×512)
- `pscn_seg_finetune.yml` — FPIC 추가 fine-tune용 (작은 lr)
- `pscn_seg_arduino.yml` — Arduino transfer learning용 (batch=4)
- `pscn_class.yml` — classification 기본

Checkpoint: `component_seg/checkpoints/best_seg.h5`, `best_class.h5`. 로그: `component_seg/logs/app.log`.

환경 셋업은 `env_setup.md` (Windows native) / `colab_setup.md` (Colab) 참조.

## 영역 전용 금지사항

- **`color_values` 임의 변경 X**: mask 생성 코드와 dataloader 간 동기화 자동화 안 됨. 변경 시 모든 mask 재생성 + 양쪽 갱신 필요.
- **`mixed_float16` 적용 X**: 학습 첫 epoch부터 NaN 발생 (`mixed_precision_nan.md`).
- **yml에 절대경로 박지 X**: Win native ↔ Colab 호환성 깨짐.
- **로컬 GPU (4060 Ti 8GB)에서 fp32 학습 시도 X**: SSIM gradient backward가 VRAM 초과. Colab L4 24GB 이상 권장 (`vram_limit.md`).
