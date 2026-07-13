import argparse
import contextlib
import copy
import io
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torchio as tio

from AutoEncoder.model.Volume8x import volumeAE


def strip_nii_gz(path):
    name = Path(path).name
    return name[:-7] if name.endswith(".nii.gz") else Path(name).stem


def sanitize_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def resolve_project_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_vae_silently(checkpoint_path):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return volumeAE.load_from_checkpoint(checkpoint_path, strict=False)


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
def run(args):
    source_path = resolve_project_path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    vae = load_vae_silently(resolve_project_path(args.ae_checkpoint)).to("cpu").eval()
    sample, meta = preprocess_volume(source_path, tuple(args.target_shape))
    latent = vae.encode(sample)

    decoded = vae.decode(latent, quantize=True)
    decoded = (decoded[0].detach().cpu().clamp(-1, 1) + 1) / 2
    decoded = decoded.transpose(2, 3).transpose(1, 3).float()
    decoded = restore_processed_volume(decoded, meta)

    subject = sanitize_name(strip_nii_gz(source_path))
    result_dir = resolve_project_path(args.results_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    original_path = result_dir / f"{subject}_original.nii.gz"
    latent_path = result_dir / f"{subject}_latent.npy"
    reconstructed_path = result_dir / f"{subject}_reconstructed.nii.gz"

    tio.ScalarImage(tensor=meta["normalized"], affine=meta["affine"]).save(original_path)
    np.save(latent_path, latent.squeeze(0).cpu().numpy())
    tio.ScalarImage(tensor=decoded, affine=meta["affine"]).save(reconstructed_path)

    print(f"Saved original: {original_path}")
    print(f"Saved latent: {latent_path}")
    print(f"Saved reconstruction: {reconstructed_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Encode a source NIfTI with VAE and reconstruct it.")
    parser.add_argument("source", type=str, help="Relative or absolute path to source .nii.gz.")
    parser.add_argument("--ae_checkpoint", type=str, default=str(PROJECT_ROOT / "checkpoints/Autoencoder_8x.ckpt"))
    parser.add_argument("--results_dir", type=str, default=str(PROJECT_ROOT / "Results/AutoEncoder"))
    parser.add_argument("--target_shape", type=int, nargs=3, default=[192, 256, 256], help="VAE input shape in H W D order.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
