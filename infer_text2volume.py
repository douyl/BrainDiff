import argparse
import contextlib
import io
import re
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torchio as tio
import yaml
from transformers import BertModel, BertTokenizer

from AutoEncoder.model.Volume8x import volumeAE
from ddpm.diffusion import GaussianDiffusion
from ddpm.network import BiFlowNet


class Config(SimpleNamespace):
    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)


def resolve_project_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sanitize_filename(text, max_length=180):
    name = re.sub(r"[^A-Za-z0-9_.()=,+;-]+", "_", text).strip("_")
    return name[:max_length].rstrip("_.") or "prompt"


def load_text(text_path):
    text_path = resolve_project_path(text_path)
    if not text_path.exists():
        raise FileNotFoundError(text_path)
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        raise ValueError(f"Text file is empty: {text_path}")
    return text


def load_vae_silently(checkpoint_path):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return volumeAE.load_from_checkpoint(checkpoint_path, strict=False)


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


def set_seed(seed, device):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def load_models(args, device):
    vae = load_vae_silently(resolve_project_path(args.AE_load_from)).to("cpu").eval()
    ae_min = vae.codebook.embeddings.min()
    ae_max = vae.codebook.embeddings.max()

    model = BiFlowNet(
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

    checkpoint = torch.load(resolve_project_path(args.checkpoint_path), map_location="cpu")
    state_dict = checkpoint["ema"] if "ema" in checkpoint else checkpoint["model"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    diffusion = GaussianDiffusion(timesteps=args.timesteps, loss_type=args.loss_type).to(device)
    tokenizer = BertTokenizer.from_pretrained(args.prompt_load_from)
    text_encoder = BertModel.from_pretrained(args.prompt_load_from).to(device).eval()
    return vae, ae_min, ae_max, model, diffusion, tokenizer, text_encoder


@torch.inference_mode()
def decode_and_save(vae, latent, output_path):
    output = vae.decode(latent, quantize=True)
    output = (output[0].detach().cpu().clamp(-1, 1) + 1) / 2
    output = output.transpose(2, 3).transpose(1, 3).float()
    tio.ScalarImage(tensor=output).save(output_path)


@torch.inference_mode()
def run(args):
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    text = load_text(args.text)
    output_dir = resolve_project_path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vae, ae_min, ae_max, model, diffusion, tokenizer, text_encoder = load_models(args, device)
    prompt_embed = encode_text(text, tokenizer, text_encoder, args.prompt_max_length, device)
    filename_base = sanitize_filename(text)

    for seed in args.seeds:
        set_seed(seed, device)
        z = torch.randn(
            1,
            args.volume_channels,
            args.image_size[0],
            args.image_size[1],
            args.image_size[2],
            device=device,
        )
        sample = diffusion.p_sample_loop(model, z, y=prompt_embed, progress_bar=True)
        latent = (((sample + 1.0) / 2.0) * (ae_max.to(sample.device) - ae_min.to(sample.device))) + ae_min.to(sample.device)
        output_path = output_dir / f"{filename_base}_seed{seed}.nii.gz"
        decode_and_save(vae, latent.cpu(), output_path)
        print(f"Saved seed {seed}: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate volumes from a text prompt.")
    parser.add_argument("text", type=str, help="Relative or absolute path to text prompt file.")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs/Text2Volume.yaml"))
    parser.add_argument("--checkpoint_path", type=str, default=str(PROJECT_ROOT / "checkpoints/Text2Volume_0090500.pt"))
    parser.add_argument("--results_dir", type=str, default=str(PROJECT_ROOT / "Results/Text2Volume"))
    parser.add_argument("--device", type=str, default="auto", help="auto, cuda, cuda:0, or cpu.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1024, 0])

    cli_args = parser.parse_args()
    with open(cli_args.config, "r", encoding="utf-8") as f:
        config = Config(**yaml.safe_load(f))

    merged = SimpleNamespace()
    merged.__dict__.update(vars(config))
    merged.__dict__.update(vars(cli_args))
    return merged


if __name__ == "__main__":
    run(parse_args())
