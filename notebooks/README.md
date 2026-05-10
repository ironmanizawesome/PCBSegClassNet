# Colab Training

`colab_train.ipynb` is a self-contained notebook that runs the **full pipeline** end-to-end on a Colab GPU runtime: data preprocessing (mask generation + patches + train/val split) → segmentation training → classification training.

## Quickstart

1. **Get the raw FPIC dataset** (request access codes from the dataset authors — see top-level [README.md](../README.md)).
2. **Zip raw inputs** and upload to Drive:
    ```powershell
    Compress-Archive -Path data\pcb_image, data\smd_annotation -DestinationPath data_raw.zip -Force
    ```
    Place at `MyDrive/PCBSegClassNet/data_raw.zip` (~7 GB).
3. **Open the notebook in Colab**:
    ```
    https://colab.research.google.com/github/<your-fork>/PCBSegClassNet/blob/colab/notebooks/colab_train.ipynb
    ```
4. **Runtime → Change runtime type → GPU**, then run cells top to bottom.

## What the notebook does

| Section | Purpose |
|---|---|
| 1 | `nvidia-smi` GPU sanity |
| 2 | Clone this repo (`colab` branch) |
| 3 | Swap kernel to Python 3.10 base via `condacolab`, then pin TF 2.10.1 + matching keras / protobuf / numpy. Colab's default Python 3.12 has no TF 2.10 wheels, and this codebase isn't compatible with Keras 3 (TF 2.16+) |
| 4 | Mount Drive, unzip `data_raw.zip` to local Colab disk |
| 5 | `create_mask.py` — polygon masks + classification crops (EDSR super-resolution, GPU) |
| 6 | `create_patches.py` — 768 px patches + 80/20 train/val split (CPU) |
| 7 | Set up Drive checkpoint directory for persistence across sessions |
| 8 | Segmentation training (5 epochs sanity → 100 epochs full → mirror checkpoint to Drive) |
| 9 | Classification training (same pattern) |
| 10 | Optional: re-evaluate from Drive checkpoints in a fresh session |

## Why preprocess on Colab?

- Raw inputs (~7 GB) are smaller than the processed dataset (~18 GB) — easier to transfer to Drive.
- Reproducibility: anyone with raw data + this notebook can recreate the exact training set without trusting an opaque processed zip.
- Easy to iterate on preprocessing knobs (e.g. patch size) without re-uploading.

If you already have a processed dataset zip, you can skip cells 5–6 and unzip it directly into `data/` instead.

## Why TF 2.10 specifically?

- This repo uses `tf.keras.activations.softmax(tensor)` and `tf.keras.backend.{dot,transpose}` patterns that broke in Keras 3.
- TF 2.10 was the last release with native Windows GPU; verified to work end-to-end.
- Colab's bundled TF (2.15+ with Keras 3) can produce `AttributeError`s on import without changes to the codebase.

## VRAM notes

| GPU | Comfortable batch size at 512×512 input |
|---|---|
| T4 (16 GB) | 16 |
| A100 (40 GB) | 32+ |
| L4 (24 GB) | 16-24 |
| RTX 4060 Ti (8 GB) | 4-8 (and even 8 OOMs in this codebase due to SSIM gradient) |

The default `batch_size: 16` in `cfs/pscn_seg.yml` works on all Colab GPUs.

## Session persistence

Colab wipes `/content` on disconnect but Drive persists. The notebook copies the best checkpoint to Drive after each training run; section 10 shows how to restore it in a new session for evaluation.
