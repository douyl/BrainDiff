# BraDiff — Inference

BraDiff generates 3D brain volumes in a fixed latent space (8× downsampled VAE) from text prompts, optionally conditioned on a source volume. This repository provides three inference scripts covering VAE reconstruction, unconditional text-to-volume synthesis, and source+text-to-target translation.

All scripts assume NIfTI (`.nii.gz`) inputs/outputs and write results under `Results/`.

## Requirements

| Item | Specification |
|------|---------------|
| Python | 3.10 (tested) |
| GPU | CUDA-capable GPU recommended; CPU fallback is supported but slow |
| Disk | ~5.5 GB for bundled checkpoints + ~440 MB for BERT weights |

Dependencies are listed in [`requirements.txt`](requirements.txt), pinned to versions verified in the `hugface` conda environment.

### Environment setup

```bash
cd BraDiff

# 1. Create and activate a virtual environment (conda or venv)
conda create -n bradiff python=3.10 -y
conda activate bradiff

# 2. Install PyTorch with CUDA support matching your driver
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu118

# 3. Install remaining dependencies
pip install -r requirements.txt
```

For CPU-only machines, omit the `--index-url` line and install the CPU build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/).

Place model checkpoints under `checkpoints/` and ensure `pretrained_models/bert-base-uncased/` is present (used as the text encoder).

## Project layout

```text
BraDiff/
├── infer_autoencoder.py          # VAE encode / decode
├── infer_text2volume.py          # text → volume
├── infer_textvolume2volume.py    # source volume + text → target volume
├── configs/
│   ├── Text2Volume.yaml
│   └── TextVolume2Volume.yaml
├── checkpoints/
│   ├── Autoencoder_8x.ckpt
│   ├── Text2Volume_0090500.pt
│   └── TextVolume2Volume_0229000.pt
├── pretrained_models/bert-base-uncased/
├── TestCase/
│   ├── Huashan/                  # example source volumes (T1w, T2w, …)
│   ├── TargetText/               # curated target-modality prompts
│   └── [All]Prompt/              # full prompt collections per cohort/modality
└── Results/
    ├── AutoEncoder/
    ├── Text2Volume/
    └── TextVolume2Volume/
```

## Inference modes

| Script | Task | Input | Output directory |
|--------|------|-------|------------------|
| `infer_autoencoder.py` | VAE encode / decode | source `.nii.gz` | `Results/AutoEncoder/` |
| `infer_text2volume.py` | text → volume | target text `.txt` | `Results/Text2Volume/` |
| `infer_textvolume2volume.py` | source + text → target | source `.nii.gz` + target text `.txt` | `Results/TextVolume2Volume/` |

### Preprocessing (AutoEncoder & TextVolume2Volume)

Source volumes are normalized per scan: values are clipped at the 99.9th percentile, min–max scaled to `[0, 1]`, then center-cropped or zero-padded to `(H, W, D) = (192, 256, 256)` before VAE encoding. Reconstructions are mapped back to the original field of view using the saved affine.

Diffusion models operate in VAE latent space with shape `(C, D, H, W) = (8, 32, 24, 32)`, corresponding to an effective output volume of `(256, 192, 256)` after 8× upsampling.

### Text prompts

Target text files should describe the desired modality and acquisition parameters (scanner, TR/TE/TI/FA for MRI; kVp/mAs for CT; isotope/dose for PET, etc.). Example prompts are in `TestCase/TargetText/`; larger collections are in `TestCase/[All]Prompt/`.

---

## 1. AutoEncoder

Encode a source volume to latent space and reconstruct it.

```bash
python infer_autoencoder.py TestCase/Huashan/T2w/AFM0002_Zhanghubian.nii.gz
```

**Outputs** (`Results/AutoEncoder/`):

| File | Description |
|------|-------------|
| `{name}_original.nii.gz` | Normalized source after preprocessing |
| `{name}_latent.npy` | VAE latent tensor |
| `{name}_reconstructed.nii.gz` | VAE reconstruction |

---

## 2. Text2Volume

Generate volumes from a text prompt alone. By default three random seeds are used (`42`, `1024`, `0`); pass `--seeds` to override.

```bash
python infer_text2volume.py TestCase/TargetText/OAS30001_to-FLAIR.txt
```

**Outputs** (`Results/Text2Volume/`):

| File | Description |
|------|-------------|
| `{prompt}_seed{seed}.nii.gz` | Generated volume for each seed |

Diffusion runs 1000 denoising steps (`timesteps` in config). Expect ~1–2 min per seed on a mid-range GPU (tested: RTX 4060 Ti, 16 GB).

---

## 3. TextVolume2Volume

Generate a target modality given a source volume and a target text prompt. The source latent serves as a structural condition via ControlNet.

```bash
python infer_textvolume2volume.py \
  TestCase/Huashan/T1w/AFM0002_Zhanghubian.nii.gz \
  TestCase/TargetText/AFM0002_to-AV45CT.txt
```

**Outputs** (`Results/TextVolume2Volume/`):

| File | Description |
|------|-------------|
| `{name}_source.nii.gz` | Normalized source used as condition |
| `{name}_target.txt` | Copy of the target prompt |
| `{name}_target.nii.gz` | Generated target volume |

---

## Checkpoints & configs

| Model | Checkpoint | Config |
|-------|------------|--------|
| VAE (8×) | `checkpoints/Autoencoder_8x.ckpt` | — |
| Text2Volume (BiFlowNet) | `checkpoints/Text2Volume_0090500.pt` | `configs/Text2Volume.yaml` |
| TextVolume2Volume (ControlNet) | `checkpoints/TextVolume2Volume_0229000.pt` | `configs/TextVolume2Volume.yaml` |

TextVolume2Volume additionally loads the Text2Volume base weights from `checkpoints/Text2Volume_0090500.pt` (`--basemodel_load_from`).

YAML configs contain inference-relevant hyperparameters only (architecture dimensions, diffusion timesteps, BERT path). Training-only fields are commented out.

## Sample data

`TestCase/TargetText/` provides six curated prompts:

| Subject | Prompts |
|---------|---------|
| AFM0001 (Huashan) | T2w, AV45 PET |
| AFM0002 (Huashan) | T1w, AV45 CT |
| OAS30001 (OASIS3) | T2w, FLAIR |

Source volumes for Huashan subjects are under `TestCase/Huashan/{T1w,T2w,FLAIR,CT,AV45-PET}/`.

## Common options

All diffusion scripts accept:

```text
--device auto|cuda|cuda:0|cpu   # default: auto
--config PATH                   # YAML config (defaults under configs/)
--checkpoint_path PATH          # diffusion checkpoint
--results_dir PATH              # output directory
```

`infer_text2volume.py` additionally accepts `--seeds INT [INT ...]`.

`infer_autoencoder.py` and `infer_textvolume2volume.py` accept `--target_shape H W D` (default `192 256 256`).
