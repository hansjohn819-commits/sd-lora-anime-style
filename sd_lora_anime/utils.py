import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


def create_writer(experiment_name: str, runs_dir: str = "runs"):
    log_dir = os.path.join(runs_dir, experiment_name)
    return SummaryWriter(log_dir=log_dir)


def create_image(model, prompt: str, device: str, seed: int, n_steps: int = 40):
    generator = torch.Generator(device=device).manual_seed(seed)
    image = model(
        prompt=prompt,
        num_inference_steps=n_steps,
        generator=generator,
    ).images[0]
    return image


def create_image_vae(model, img, device: str, size: int = 256):
    model.eval().to(device)
    pre = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    x = pre(img).unsqueeze(0).to(device)

    with torch.no_grad():
        z = model.encode(x).latent_dist.sample()
        recon = model.decode(z).sample
    recon = (recon.squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1)
    return transforms.ToPILImage()(recon)
