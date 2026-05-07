# GalaxEye Binary Change Detection

Lightweight, reproducible PyTorch repository for binary semantic change detection on the official GalaxEye assignment dataset.

## Overview

- Task: binary change detection after remapping raw labels `{0,1,2,3}` to `{0,0,1,1}`
- Baseline: `U-Net + ResNet encoder` from `segmentation_models_pytorch`
- Supported models: `unet`, `unetplusplus`, `deeplabv3plus`
- Dataset source: official Hugging Face dataset [`doron333/change-detection-dataset`](https://huggingface.co/datasets/doron333/change-detection-dataset?utm_source=chatgpt.com)

The repository is designed so code and configs stay in Git, while datasets, checkpoints, logs, and outputs stay out of the repo.

For the current official Hugging Face snapshot, the resolved training inputs are:

- `pre-event`: RGB EO image
- `post-event`: single-channel post-event image
- `target`: raw segmentation mask

So the default multimodal baseline in this repository is `eo_pre + sar_post`, which yields a 4-channel input tensor.

## Dataset Workflow

The project now supports two modes:

- Hugging Face mode: download or reuse the official dataset cache automatically
- Local mode: point `paths.dataset_root` or `data.local_dataset_root` to an existing dataset copy

Expected dataset root after download or resolution:

```text
<dataset_root>/
├── train/
├── val/
└── test/
```

The runtime resolves dataset roots in this order:

1. `paths.dataset_root`
2. `data.local_dataset_root`
3. cached Hugging Face copy from `data.hf_repo_id`

No ZIP extraction workflow is required anymore.

## Setup

Use Python 3.10+.

```bash
pip install -r requirements.txt
```

## Configs

- [configs/config.yaml](/C:/Users/sanus/Desktop/GalaxEye%20Space%20Assignment/configs/config.yaml): general experimentation config
- [configs/stabilization_config.yaml](/C:/Users/sanus/Desktop/GalaxEye%20Space%20Assignment/configs/stabilization_config.yaml): subset stabilization run on GPU
- [configs/final_config.yaml](/C:/Users/sanus/Desktop/GalaxEye%20Space%20Assignment/configs/final_config.yaml): recommended full training config

Key defaults:

- Hugging Face dataset source enabled
- cache under `~/.cache/huggingface`
- CUDA required for stabilization and final configs
- AMP enabled for training, evaluation, and inference

## Google Colab

Set the Colab runtime to `GPU` first.

```bash
!git clone <your-repo-url>
%cd <your-repo-folder>
!pip install -r requirements.txt
```

Prepare or reuse the official dataset cache:

```bash
!python scripts/prepare_hf_dataset.py --config configs/stabilization_config.yaml
```

Run the stabilization subset experiment:

```bash
!python train.py --config configs/stabilization_config.yaml --device cuda --require-cuda
```

Evaluate the best checkpoint:

```bash
!python eval.py --config configs/stabilization_config.yaml --checkpoint outputs/subset_stabilization_eo_only/checkpoints/best.pt --splits val --device cuda --require-cuda
```

Generate qualitative inference outputs:

```bash
!python inference.py --config configs/stabilization_config.yaml --checkpoint outputs/subset_stabilization_eo_only/checkpoints/best.pt --split val --sample-limit 24 --device cuda --require-cuda
```

Run the full baseline after stabilization is healthy:

```bash
!python train.py --config configs/final_config.yaml --device cuda --require-cuda
```

## Local Cached Dataset

If you already have a local cached copy, set one of these in the config:

```yaml
paths:
  dataset_root: /absolute/path/to/change-detection-dataset
```

or

```yaml
data:
  local_dataset_root: /absolute/path/to/change-detection-dataset
```

Then run the same training, evaluation, or inference commands. No code changes are needed.

## Useful Commands

Inspect dataset structure:

```bash
python scripts/explore_dataset.py --config configs/config.yaml --save-json outputs/dataset_report.json
```

Sanity check a few samples:

```bash
python scripts/sanity_check.py --config configs/config.yaml --split train --num-samples 4
```

Validate the dataloader on a tiny subset:

```bash
python scripts/check_dataloader.py --config configs/config.yaml --split train --sample-limit 4
```

## Repository Structure

```text
.
├── configs/
├── datasets/
├── models/
├── scripts/
├── utils/
├── eval.py
├── inference.py
├── train.py
├── requirements.txt
└── README.md
```

Generated artifacts are written under `outputs/` and ignored by Git.

## Reproducibility

- deterministic seeds are set in train, eval, and inference
- checkpoints store model settings, active modalities, thresholds, and subset indices
- Hugging Face dataset resolution is config-driven and cache-aware
- CUDA-only configs fail fast if no GPU is available

## Notes

- The repository intentionally does not track datasets, outputs, logs, or checkpoints.
- `scripts/prepare_hf_dataset.py` is the recommended entry point for Colab dataset setup.
- Best checkpoint download link: `TBD`

## References

- Ronneberger et al., U-Net: Convolutional Networks for Biomedical Image Segmentation
- Zhou et al., UNet++: A Nested U-Net Architecture for Medical Image Segmentation
- Chen et al., Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation
- `segmentation_models_pytorch`
