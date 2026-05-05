# Colab Training

`colab_train.ipynb` is a self-contained notebook that runs the full training pipeline (segmentation + classification) on a Colab GPU runtime.

## Quickstart

1. **Prepare the dataset locally** (see top-level [README.md](../README.md) for `create_mask.py` → `create_patches.py`).
2. **Zip the prepared `data/` directory** (segmentation/ + classification/ subfolders) and upload to Drive at `MyDrive/PCBSegClassNet/data.zip`.
3. **Open the notebook in Colab**: from GitHub the easiest path is the `Open in Colab` Chrome extension, or use the URL form:
    ```
    https://colab.research.google.com/github/<your-fork>/PCBSegClassNet/blob/colab/notebooks/colab_train.ipynb
    ```
4. **Runtime → Change runtime type → GPU**, then run cells top to bottom.

## What the notebook does

| Cell | Purpose |
|---|---|
| 1 | `nvidia-smi` GPU sanity |
| 2 | Clone this repo (`colab` branch) |
| 3 | Pin TF 2.10.1 + matching keras / protobuf / numpy (the codebase isn't compatible with Keras 3) |
| 4 | Mount Drive, unzip `data.zip` to local Colab disk (≪ Drive in IO speed) |
| 5 | Set up Drive checkpoint directory for persistence across sessions |
| 6 | Segmentation training (5 epochs sanity → 100 epochs full → mirror checkpoint to Drive) |
| 7 | Classification training (same pattern) |
| 8 | Optional: re-evaluate from Drive checkpoints in a fresh session |

## Why TF 2.10 specifically?

- This repo uses `tf.keras.activations.softmax(tensor)` and other patterns that broke in Keras 3.
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

Colab wipes `/content` on disconnect but Drive persists. The notebook copies the best checkpoint to Drive after each run; cell 8 shows how to restore it in a new session for evaluation.
