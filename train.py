"""Train LoRA for SDXL 1.0 and SD 1.5, and a custom autoencoder.

Usage:
    python train.py --stage all
    python train.py --stage sdxl_train
    python train.py --stage sd15_train --lr 1e-5 --epochs 5
    python train.py --stage vae_train
"""
import argparse
import os
import random
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import logging as transformers_logging
from diffusers.utils import logging as diffusers_logging

from sd_lora_anime import data, engine, models, utils


# ======================================================================
# Config (edit defaults here; CLI flags override per-stage values)
# ======================================================================

# -- Shared --
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_NAME = "none-yet/anime-captions"
DATASET_SEED = 321
N_SAMPLES = 1240
TRAIN_NUM = 1200
MAX_TOKENS = 65                 # CLIP limit is 77; leave headroom for trigger + special tokens
TRIGGER_WORD = "xkz"            # rare token that tokenizes into 2 subword pieces (avoids vocab collisions)
BATCH_SIZE = 2
NUM_WORKERS = 0
GRAD_CLIP = 1.0
LR_T_MAX = 10000                # cosine schedule length > total steps (~6000) to avoid LR collapsing to eta_min before training ends
LR_ETA_MIN = 1e-7
RUNS_DIR = "runs"
TEST_SEED = 22333
INFERENCE_STEPS = 40

# -- SDXL 1.0 --
SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_RESOLUTION = 1024
SDXL_UNET_RANK = 32
SDXL_UNET_ALPHA = 32
SDXL_TE_RANK = 8
SDXL_TE_ALPHA = 8
SDXL_LR = 1e-5
SDXL_WEIGHT_DECAY = 1e-4
SDXL_EPOCHS = 10
SDXL_SAVE_EVERY = 200
SDXL_VAE_SCALING = 0.13025      # SDXL VAE latent scaling factor (fixed by the pretrained model)
SDXL_LORA_DIR = "lora_weights"
SDXL_CKPT_DIR = "checkpoints"
SDXL_TEST_PROMPT = "a photo of a woman walking on the street"

# -- SD 1.5 --
SD15_MODEL_ID = "benjamin-paine/stable-diffusion-v1-5"
SD15_RESOLUTION = 512
SD15_UNET_RANK = 24
SD15_UNET_ALPHA = 24
SD15_TE_RANK = 8
SD15_TE_ALPHA = 8
SD15_LR = 5e-5
SD15_WEIGHT_DECAY = 1e-4
SD15_EPOCHS = 10
SD15_SAVE_EVERY = 200
SD15_VAE_SCALING = 0.18215      # SD1.5 VAE latent scaling factor (fixed by the pretrained model)
SD15_LORA_DIR = "sd15/lora_weights"
SD15_CKPT_DIR = "sd15/checkpoints"
SD15_TEST_PROMPT = "two young women having dinner on the table"

# -- Custom Autoencoder --
AE_LR = 1e-4
AE_EPOCHS = 50
AE_BATCH_SIZE = 4
AE_SAVE_EVERY = 5
AE_RESOLUTION = 256
AE_CKPT_DIR = "vaecheckpoints"


# ======================================================================
# Stages
# ======================================================================

def _quiet():
    warnings.filterwarnings("ignore")
    transformers_logging.set_verbosity_error()
    diffusers_logging.set_verbosity_error()


def run_sdxl_train(lr, epochs, batch_size, save_every):
    pipe = models.create_sdxl(SDXL_MODEL_ID, DEVICE)
    train_subset, val_subset = data.get_dataset(
        pipe.tokenizer, DATASET_NAME, N_SAMPLES, TRAIN_NUM,
        DATASET_SEED, MAX_TOKENS, TRIGGER_WORD,
    )
    train_loader, val_loader = data.create_dataloader(
        train_subset, val_subset, SDXL_RESOLUTION, batch_size, NUM_WORKERS,
    )

    unet = pipe.unet
    vae = pipe.vae.to(torch.float32)
    tokenizer = pipe.tokenizer
    tokenizer2 = pipe.tokenizer_2
    text_encoder = pipe.text_encoder
    text_encoder2 = pipe.text_encoder_2
    noise_scheduler = pipe.scheduler

    text_encoder = models.lora_te(text_encoder, SDXL_TE_RANK, SDXL_TE_ALPHA)
    unet = models.lora_unet(unet, SDXL_UNET_RANK, SDXL_UNET_ALPHA)

    trainable = (
        list(filter(lambda p: p.requires_grad, unet.parameters())) +
        list(filter(lambda p: p.requires_grad, text_encoder.parameters()))
    )
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=SDXL_WEIGHT_DECAY)
    lr_sched = CosineAnnealingLR(optimizer, T_max=LR_T_MAX, eta_min=LR_ETA_MIN)

    writer = utils.create_writer("SDXL", RUNS_DIR)
    for i in range(epochs):
        engine.train_one_epoch_sdxl(
            i, unet, vae, tokenizer, tokenizer2, text_encoder, text_encoder2,
            noise_scheduler, train_loader, val_loader, optimizer, lr_sched,
            DEVICE, writer,
            save_every=save_every,
            vae_scaling=SDXL_VAE_SCALING,
            lora_dir=SDXL_LORA_DIR,
            ckpt_dir=SDXL_CKPT_DIR,
            grad_clip=GRAD_CLIP,
        )
    writer.close()

    del pipe, unet, vae, tokenizer, tokenizer2, text_encoder, text_encoder2, noise_scheduler
    torch.cuda.empty_cache()


def run_sdxl_infer(save_every):
    writer = utils.create_writer("SDXL-image", RUNS_DIR)
    prompt_no = SDXL_TEST_PROMPT
    prompt_with = f"{TRIGGER_WORD}, {SDXL_TEST_PROMPT}"
    step_num = save_every
    while True:
        unet_dir = os.path.join(SDXL_LORA_DIR, str(step_num), "unet")
        te_dir = os.path.join(SDXL_LORA_DIR, str(step_num), "text_encoder")
        if not (os.path.isdir(unet_dir) and os.path.isdir(te_dir)):
            break
        pipe = models.sdxl_load_lora(SDXL_MODEL_ID, unet_dir, te_dir, DEVICE)
        img1 = utils.create_image(pipe, prompt_no, DEVICE, TEST_SEED, INFERENCE_STEPS)
        img2 = utils.create_image(pipe, prompt_with, DEVICE, TEST_SEED, INFERENCE_STEPS)
        writer.add_image(f"no_trigger/step_{step_num}", TF.to_tensor(img1), global_step=step_num)
        writer.add_image(f"with_trigger/step_{step_num}", TF.to_tensor(img2), global_step=step_num)
        del pipe
        torch.cuda.empty_cache()
        print(f"SDXL infer step: {step_num} done.")
        step_num += save_every
    writer.close()


def run_sd15_train(lr, epochs, batch_size, save_every):
    pipe = models.create_sd15(SD15_MODEL_ID, DEVICE)
    train_subset, val_subset = data.get_dataset(
        pipe.tokenizer, DATASET_NAME, N_SAMPLES, TRAIN_NUM,
        DATASET_SEED, MAX_TOKENS, TRIGGER_WORD,
    )
    train_loader, val_loader = data.create_dataloader(
        train_subset, val_subset, SD15_RESOLUTION, batch_size, NUM_WORKERS,
    )

    unet = pipe.unet
    vae = pipe.vae.to(torch.float32)
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    noise_scheduler = pipe.scheduler

    text_encoder = models.lora_te(text_encoder, SD15_TE_RANK, SD15_TE_ALPHA)
    unet = models.lora_unet(unet, SD15_UNET_RANK, SD15_UNET_ALPHA)

    trainable = (
        list(filter(lambda p: p.requires_grad, unet.parameters())) +
        list(filter(lambda p: p.requires_grad, text_encoder.parameters()))
    )
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=SD15_WEIGHT_DECAY)
    lr_sched = CosineAnnealingLR(optimizer, T_max=LR_T_MAX, eta_min=LR_ETA_MIN)

    writer = utils.create_writer("SD1.5", RUNS_DIR)
    for i in range(epochs):
        engine.train_one_epoch_sd15(
            i, unet, vae, tokenizer, text_encoder, noise_scheduler,
            train_loader, val_loader, optimizer, lr_sched,
            DEVICE, writer,
            save_every=save_every,
            vae_scaling=SD15_VAE_SCALING,
            lora_dir=SD15_LORA_DIR,
            ckpt_dir=SD15_CKPT_DIR,
            grad_clip=GRAD_CLIP,
        )
    writer.close()

    del pipe, unet, vae, tokenizer, text_encoder, noise_scheduler
    torch.cuda.empty_cache()


def run_sd15_infer(save_every):
    writer = utils.create_writer("SD1.5-image", RUNS_DIR)
    prompt_no = SD15_TEST_PROMPT
    prompt_with = f"{TRIGGER_WORD}, {SD15_TEST_PROMPT}"
    step_num = save_every
    while True:
        unet_dir = os.path.join(SD15_LORA_DIR, str(step_num), "unet")
        te_dir = os.path.join(SD15_LORA_DIR, str(step_num), "text_encoder")
        if not (os.path.isdir(unet_dir) and os.path.isdir(te_dir)):
            break
        pipe = models.sd15_load_lora(SD15_MODEL_ID, unet_dir, te_dir, DEVICE)
        img1 = utils.create_image(pipe, prompt_no, DEVICE, TEST_SEED, INFERENCE_STEPS)
        img2 = utils.create_image(pipe, prompt_with, DEVICE, TEST_SEED, INFERENCE_STEPS)
        writer.add_image(f"no_trigger/step_{step_num}", TF.to_tensor(img1), global_step=step_num)
        writer.add_image(f"with_trigger/step_{step_num}", TF.to_tensor(img2), global_step=step_num)
        del pipe
        torch.cuda.empty_cache()
        print(f"SD1.5 infer step: {step_num} done.")
        step_num += save_every
    writer.close()


def run_vae_train(lr, epochs, batch_size, save_every):
    # VAE training needs the same HF subset; reuse the SD1.5 tokenizer to filter by length.
    pipe = models.create_sd15(SD15_MODEL_ID, DEVICE)
    train_subset, val_subset = data.get_dataset(
        pipe.tokenizer, DATASET_NAME, N_SAMPLES, TRAIN_NUM,
        DATASET_SEED, MAX_TOKENS, TRIGGER_WORD,
    )
    del pipe
    torch.cuda.empty_cache()

    train_dataset = data.VAEDataset(train_subset, AE_RESOLUTION)
    test_dataset = data.VAEDataset(val_subset, AE_RESOLUTION)

    vae = models.SimpleAutoencoderKL()
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    writer = utils.create_writer("vae", RUNS_DIR)
    engine.train_vae(
        vae, train_dataset, test_dataset, writer, epochs,
        optimizer, loss_fn, DEVICE,
        batch_size=batch_size,
        save_every=save_every,
        ckpt_dir=AE_CKPT_DIR,
        num_workers=NUM_WORKERS,
    )
    writer.close()
    return train_subset, val_subset


def run_vae_infer(save_every, epochs, train_subset=None, val_subset=None):
    if val_subset is None:
        pipe = models.create_sd15(SD15_MODEL_ID, DEVICE)
        _, val_subset = data.get_dataset(
            pipe.tokenizer, DATASET_NAME, N_SAMPLES, TRAIN_NUM,
            DATASET_SEED, MAX_TOKENS, TRIGGER_WORD,
        )
        del pipe
        torch.cuda.empty_cache()

    writer = utils.create_writer("vae-image", RUNS_DIR)
    random.seed(42)
    n_val = len(val_subset)
    idx1, idx2 = random.sample(range(n_val), 2)
    img1 = val_subset[idx1]["image"].convert("RGB")
    img2 = val_subset[idx2]["image"].convert("RGB")

    vae = models.SimpleAutoencoderKL()
    for i in range(epochs // save_every):
        epoch = (i + 1) * save_every
        path = os.path.join(AE_CKPT_DIR, f"vae_epoch{epoch:03d}.pt")
        if not os.path.isfile(path):
            continue
        vae.load_state_dict(torch.load(path, map_location=DEVICE))
        vae = vae.to(DEVICE)
        result1 = utils.create_image_vae(vae, img1, DEVICE, AE_RESOLUTION)
        result2 = utils.create_image_vae(vae, img2, DEVICE, AE_RESOLUTION)
        writer.add_image(f"img1/step_{epoch}", TF.to_tensor(result1), global_step=epoch)
        writer.add_image(f"img2/step_{epoch}", TF.to_tensor(result2), global_step=epoch)
        print(f"VAE infer epoch {epoch} done.")
    writer.close()


# ======================================================================
# CLI
# ======================================================================

STAGES = [
    "sdxl_train", "sdxl_infer",
    "sd15_train", "sd15_infer",
    "vae_train", "vae_infer",
    "all",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--stage", choices=STAGES, default="all",
                   help="Which stage(s) to run (default: all)")
    p.add_argument("--lr", type=float, default=None,
                   help="Override learning rate for the selected training stage")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override number of epochs for the selected training stage")
    p.add_argument("--batch_size", type=int, default=None,
                   help="Override batch size for the selected training stage")
    p.add_argument("--save_every", type=int, default=None,
                   help="Override save-every interval for the selected stage")
    p.add_argument("--seed", type=int, default=None,
                   help="Override inference seed")
    return p.parse_args()


def main():
    args = parse_args()
    _quiet()

    global TEST_SEED
    if args.seed is not None:
        TEST_SEED = args.seed

    stage = args.stage

    if stage in ("sdxl_train", "all"):
        run_sdxl_train(
            lr=args.lr if args.lr is not None else SDXL_LR,
            epochs=args.epochs if args.epochs is not None else SDXL_EPOCHS,
            batch_size=args.batch_size if args.batch_size is not None else BATCH_SIZE,
            save_every=args.save_every if args.save_every is not None else SDXL_SAVE_EVERY,
        )

    if stage in ("sdxl_infer", "all"):
        run_sdxl_infer(
            save_every=args.save_every if args.save_every is not None else SDXL_SAVE_EVERY,
        )

    if stage in ("sd15_train", "all"):
        run_sd15_train(
            lr=args.lr if args.lr is not None else SD15_LR,
            epochs=args.epochs if args.epochs is not None else SD15_EPOCHS,
            batch_size=args.batch_size if args.batch_size is not None else BATCH_SIZE,
            save_every=args.save_every if args.save_every is not None else SD15_SAVE_EVERY,
        )

    if stage in ("sd15_infer", "all"):
        run_sd15_infer(
            save_every=args.save_every if args.save_every is not None else SD15_SAVE_EVERY,
        )

    vae_train_subset = vae_val_subset = None
    if stage in ("vae_train", "all"):
        vae_train_subset, vae_val_subset = run_vae_train(
            lr=args.lr if args.lr is not None else AE_LR,
            epochs=args.epochs if args.epochs is not None else AE_EPOCHS,
            batch_size=args.batch_size if args.batch_size is not None else AE_BATCH_SIZE,
            save_every=args.save_every if args.save_every is not None else AE_SAVE_EVERY,
        )

    if stage in ("vae_infer", "all"):
        run_vae_infer(
            save_every=args.save_every if args.save_every is not None else AE_SAVE_EVERY,
            epochs=args.epochs if args.epochs is not None else AE_EPOCHS,
            train_subset=vae_train_subset,
            val_subset=vae_val_subset,
        )


if __name__ == "__main__":
    main()
