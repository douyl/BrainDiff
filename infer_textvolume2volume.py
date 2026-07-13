import argparse
import contextlib
import copy
import io
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torchio as tio
import yaml
from transformers import BertModel, BertTokenizer

from AutoEncoder.model.Volume8x import volumeAE
from ddpm.controlnet import ControlNet
from ddpm.diffusion import GaussianDiffusion
from ddpm.network import BiFlowNet


class Config(SimpleNamespace):
    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)


def strip_nii_gz(path):
    name = Path(path).name
    return name[:-7] if name.endswith(".nii.gz") else Path(name).stem


def sanitize_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def resolve_project_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_text(text_path):
    text_path = resolve_project_path(text_path)
    if not text_path.exists():
        raise FileNotFoundError(text_path)
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        raise ValueError(f"Text file is empty: {text_path}")
    return text, text_path


def preprocess_volume(image_path, target_shape):
    img = tio.ScalarImage(str(image_path))
    img_data = img.data.numpy()
    upper = np.percentile(img_data, 99.9)
    img_data = np.clip(img_data, 0, upper)
    denom = img_data.max() - img_data.min()
    if denom > 0:
        img_data = (img_data - img_data.min()) / denom
    else:
        img_data = np.zeros_like(img_data)
    img.set_data(torch.from_numpy(img_data).float())

    original_shape = tuple(img.data.shape[1:])
    processed_img = copy.deepcopy(img)
    pad_params = []
    crop_slices = []
    restore_info = []

    for orig, target in zip(original_shape, target_shape):
        if orig < target:
            pad_total = target - orig
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            pad_params.extend([pad_before, pad_after])
            crop_slices.append(slice(0, target))
            restore_info.append(("pad", pad_before, orig))
        else:
            crop_start = (orig - target) // 2
            crop_slices.append(slice(crop_start, crop_start + target))
            pad_params.extend([0, 0])
            restore_info.append(("crop", crop_start, target))

    if any(orig < target for orig, target in zip(original_shape, target_shape)):
        processed_img = tio.transforms.Pad(padding=pad_params, padding_mode=0)(processed_img)
    if any(orig > target for orig, target in zip(original_shape, target_shape)):
        processed_img_data = processed_img.data[:, crop_slices[0], crop_slices[1], crop_slices[2]]
        processed_img.set_data(processed_img_data)

    if tuple(processed_img.data.shape[1:]) != tuple(target_shape):
        raise ValueError(f"Preprocessed shape {processed_img.data.shape[1:]} does not match {target_shape}.")

    sample = processed_img.data * 2 - 1
    sample = sample.transpose(1, 3).transpose(2, 3).type(torch.float32).unsqueeze(0)
    return sample, {
        "original_shape": original_shape,
        "restore_info": restore_info,
        "affine": img.affine,
        "normalized": img.data.clone(),
    }


def restore_processed_volume(processed_tensor, meta):
    proc_slices = []
    orig_slices = []
    for dim, (mode, offset, length) in enumerate(meta["restore_info"]):
        orig = meta["original_shape"][dim]
        if mode == "pad":
            proc_slices.append(slice(offset, offset + orig))
            orig_slices.append(slice(0, orig))
        else:
            proc_slices.append(slice(0, length))
            orig_slices.append(slice(offset, offset + length))

    restored = torch.zeros((processed_tensor.shape[0], *meta["original_shape"]), dtype=processed_tensor.dtype)
    restored[(slice(None), *orig_slices)] = processed_tensor[(slice(None), *proc_slices)]
    return restored


@torch.inference_mode()
def encode_source_latent(vae, source_path, target_shape):
    sample, meta = preprocess_volume(source_path, target_shape)
    latent = vae.encode(sample)
    return latent, meta


@torch.inference_mode()
def decode_target_latent(vae, latent, meta):
    volume = vae.decode(latent, quantize=True)
    volume = (volume[0].detach().cpu().clamp(-1, 1) + 1) / 2
    volume = volume.transpose(2, 3).transpose(1, 3).float()
    return restore_processed_volume(volume, meta)


@torch.inference_mode()
def encode_text(text, tokenizer, text_encoder, max_length, device):
    prompt_token = tokenizer(
        text,
        max_length=int(max_length),
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    prompt_token = {key: value.to(device) for key, value in prompt_token.items()}
    return text_encoder(prompt_token["input_ids"], attention_mask=prompt_token["attention_mask"])[0]


def load_vae_silently(checkpoint_path):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return volumeAE.load_from_checkpoint(checkpoint_path, strict=False)


def load_models(args, device):
    vae = load_vae_silently(resolve_project_path(args.AE_load_from)).to("cpu").eval()
    ae_min = vae.codebook.embeddings.min()
    ae_max = vae.codebook.embeddings.max()

    basemodel = BiFlowNet(
        dim=args.model_dim,
        prompt_dim=args.prompt_dim,
        dim_mults=args.dim_mults,
        channels=args.volume_channels,
        init_kernel_size=3,
        learn_sigma=False,
        use_sparse_linear_attn=args.use_attn,
        vq_size=args.vq_size,
        num_mid_DiT=args.num_dit,
        patch_size=args.patch_size,
        res_condition=False,
    ).to(device)
    checkpoint = torch.load(resolve_project_path(args.basemodel_load_from), map_location="cpu")
    basemodel.load_state_dict(checkpoint["model"], strict=True)

    model = ControlNet(basemodel, condition_channels=args.condition_channels).to(device)
    checkpoint = torch.load(resolve_project_path(args.checkpoint_path), map_location="cpu")
    state_dict = checkpoint["ema"] if "ema" in checkpoint else checkpoint["model"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    diffusion = GaussianDiffusion(timesteps=args.timesteps, loss_type=args.loss_type).to(device)

    tokenizer = BertTokenizer.from_pretrained(args.prompt_load_from)
    text_encoder = BertModel.from_pretrained(args.prompt_load_from).to(device).eval()
    return vae, ae_min, ae_max, model, diffusion, tokenizer, text_encoder


def empty_cuda_cache(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()


def build_output_paths(args, source_path, target_text_path):
    subject = sanitize_name(strip_nii_gz(source_path))
    result_dir = resolve_project_path(args.results_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    source_out = result_dir / f"{subject}_source.nii.gz"
    target_out = result_dir / f"{subject}_target.nii.gz"
    text_out = result_dir / f"{subject}_target.txt"

    shutil.copy2(target_text_path, text_out)
    return result_dir, source_out, target_out, text_out


@torch.inference_mode()
def run_inference(args):
    source_path = resolve_project_path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    target_text, target_text_path = load_text(args.text)
    result_dir, source_out, target_out, text_out = build_output_paths(args, source_path, target_text_path)

    vae, ae_min, ae_max, model, diffusion, tokenizer, text_encoder = load_models(args, device)
    condition_latent, source_meta = encode_source_latent(vae, source_path, tuple(args.target_shape))
    tio.ScalarImage(tensor=source_meta["normalized"], affine=source_meta["affine"]).save(source_out)
    condition = ((condition_latent - ae_min) / (ae_max - ae_min)) * 2.0 - 1.0
    condition = condition.to(device)

    prompt_embed = encode_text(target_text, tokenizer, text_encoder, args.prompt_max_length, device)
    z = torch.randn(
        1,
        args.volume_channels,
        args.image_size[0],
        args.image_size[1],
        args.image_size[2],
        device=device,
    )

    samples = diffusion.p_sample_loop(model, z, y=prompt_embed, condition=condition, progress_bar=True)

    ae_min = ae_min.to(samples.device)
    ae_max = ae_max.to(samples.device)
    latent = (((samples + 1.0) / 2.0) * (ae_max - ae_min)) + ae_min
    latent = latent.cpu()
    del model, diffusion, text_encoder, condition_latent, condition, prompt_embed, z, samples
    empty_cuda_cache(device)

    target_volume = decode_target_latent(vae, latent, source_meta)
    tio.ScalarImage(tensor=target_volume, affine=source_meta["affine"]).save(target_out)

    print(f"Saved result directory: {result_dir}")
    print(f"Source copy: {source_out}")
    print(f"Target text: {text_out}")
    print(f"Generated target: {target_out}")


def parse_args():
    parser = argparse.ArgumentParser(description="Infer target modality NIfTI from source NIfTI and target text.")
    parser.add_argument("source", type=str, help="Relative or absolute path to a single source modality .nii.gz file.")
    parser.add_argument("text", type=str, help="Relative or absolute path to a txt file containing the target modality prompt.")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs/TextVolume2Volume.yaml"))
    parser.add_argument("--checkpoint_path", type=str, default=str(PROJECT_ROOT / "checkpoints/TextVolume2Volume_0229000.pt"))
    parser.add_argument("--basemodel_load_from", type=str, default=str(PROJECT_ROOT / "checkpoints/Text2Volume_0090500.pt"))
    parser.add_argument("--results_dir", type=str, default=str(PROJECT_ROOT / "Results/TextVolume2Volume"))
    parser.add_argument("--device", type=str, default="auto", help="auto, cuda, cuda:0, or cpu.")
    parser.add_argument("--target_shape", type=int, nargs=3, default=[192, 256, 256], help="VAE input shape in H W D order.")

    cli_args = parser.parse_args()
    with open(cli_args.config, "r", encoding="utf-8") as f:
        config = Config(**yaml.safe_load(f))

    merged = SimpleNamespace()
    merged.__dict__.update(vars(config))
    merged.__dict__.update(vars(cli_args))
    return merged


if __name__ == "__main__":
    run_inference(parse_args())
