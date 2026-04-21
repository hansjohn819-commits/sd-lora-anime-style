"""Single-image inference for trained LoRAs and the custom autoencoder.

Examples:
    python inference.py --model sdxl --prompt "a photo of a woman walking" --trigger --output out.png
    python inference.py --model sd15 --prompt "a little girl playing in the park" --output out.png
    python inference.py --model ae --input_image input.jpg --output recon.png
"""
import argparse
import os

import torch
from PIL import Image

from sd_lora_anime import models, utils


DEFAULT_SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_SD15_MODEL_ID = "benjamin-paine/stable-diffusion-v1-5"
DEFAULT_TRIGGER = "xkz"
DEFAULT_SDXL_WEIGHTS = "weights/sdxl"
DEFAULT_SD15_WEIGHTS = "weights/sd15"
DEFAULT_AE_CKPT = "weights/custom_ae/vae_epoch035.pt"
DEFAULT_AE_SIZE = 256


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--model", choices=["sdxl", "sd15", "ae"], required=True)
    p.add_argument("--prompt", type=str, default=None,
                   help="Prompt text (required for sdxl/sd15)")
    p.add_argument("--input_image", type=str, default=None,
                   help="Input image path (required for ae)")
    p.add_argument("--output", type=str, required=True,
                   help="Output image path")
    p.add_argument("--seed", type=int, default=22333)
    p.add_argument("--steps", type=int, default=40,
                   help="Diffusion inference steps (sdxl/sd15 only)")
    p.add_argument("--trigger", action="store_true",
                   help="Prepend the trigger word to the prompt")
    p.add_argument("--trigger_word", type=str, default=DEFAULT_TRIGGER)
    p.add_argument("--weights_dir", type=str, default=None,
                   help="LoRA weights directory; defaults to weights/{sdxl|sd15}")
    p.add_argument("--ae_ckpt", type=str, default=DEFAULT_AE_CKPT,
                   help="Custom AE checkpoint path")
    p.add_argument("--sdxl_model_id", type=str, default=DEFAULT_SDXL_MODEL_ID)
    p.add_argument("--sd15_model_id", type=str, default=DEFAULT_SD15_MODEL_ID)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def run_diffusion(args):
    if not args.prompt:
        raise SystemExit("--prompt is required for --model sdxl/sd15")

    base = args.weights_dir or (
        DEFAULT_SDXL_WEIGHTS if args.model == "sdxl" else DEFAULT_SD15_WEIGHTS
    )
    unet_dir = os.path.join(base, "unet")
    te_dir = os.path.join(base, "text_encoder")
    if not (os.path.isdir(unet_dir) and os.path.isdir(te_dir)):
        raise SystemExit(f"LoRA weights not found under {base} (expected unet/ and text_encoder/)")

    prompt = args.prompt
    if args.trigger and not prompt.lstrip().startswith(args.trigger_word):
        prompt = f"{args.trigger_word}, {prompt}"

    if args.model == "sdxl":
        pipe = models.sdxl_load_lora(args.sdxl_model_id, unet_dir, te_dir, args.device)
    else:
        pipe = models.sd15_load_lora(args.sd15_model_id, unet_dir, te_dir, args.device)

    image = utils.create_image(pipe, prompt, args.device, args.seed, args.steps)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    image.save(args.output)
    print(f"Saved: {args.output}")


def run_ae(args):
    if not args.input_image:
        raise SystemExit("--input_image is required for --model ae")
    if not os.path.isfile(args.ae_ckpt):
        raise SystemExit(f"AE checkpoint not found: {args.ae_ckpt}")

    model = models.SimpleAutoencoderKL()
    state = torch.load(args.ae_ckpt, map_location=args.device)
    model.load_state_dict(state)
    img = Image.open(args.input_image).convert("RGB")
    out = utils.create_image_vae(model, img, args.device, DEFAULT_AE_SIZE)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out.save(args.output)
    print(f"Saved: {args.output}")


def main():
    args = parse_args()
    if args.model in ("sdxl", "sd15"):
        run_diffusion(args)
    else:
        run_ae(args)


if __name__ == "__main__":
    main()
