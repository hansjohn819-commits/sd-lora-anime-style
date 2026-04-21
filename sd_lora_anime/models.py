import torch
import torch.nn as nn
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from peft import LoraConfig, get_peft_model, PeftModel
from types import SimpleNamespace
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution


def create_sdxl(model_id: str, device: str):
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16",
    )
    return pipe.to(device)


def create_sd15(model_id: str, device: str):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    )
    return pipe.to(device)


def lora_te(text_encoder, rank: int, alpha: int):
    te_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0,
        bias="none",
    )
    return get_peft_model(text_encoder, te_config)


def lora_unet(unet, rank: int, alpha: int):
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0,
    )
    return get_peft_model(unet, config)


def sdxl_load_lora(model_id: str, unet_lora_dir: str, te_lora_dir: str, device: str):
    pipe = create_sdxl(model_id, device)
    pipe.unet = PeftModel.from_pretrained(pipe.unet, unet_lora_dir)
    pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, te_lora_dir)
    return pipe


def sd15_load_lora(model_id: str, unet_lora_dir: str, te_lora_dir: str, device: str):
    pipe = create_sd15(model_id, device)
    pipe.unet = PeftModel.from_pretrained(pipe.unet, unet_lora_dir)
    pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, te_lora_dir)
    return pipe


# ---------------- Custom Autoencoder ----------------

class ResnetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act = nn.SiLU()
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.shortcut(x)


class Attention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(32, ch)
        self.to_q = nn.Linear(ch, ch)
        self.to_k = nn.Linear(ch, ch)
        self.to_v = nn.Linear(ch, ch)
        self.to_out = nn.Linear(ch, ch)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.reshape(B, C, H * W).permute(0, 2, 1)
        q, k, v = self.to_q(h), self.to_k(h), self.to_v(h)
        attn = torch.softmax(q @ k.transpose(-1, -2) / C ** 0.5, dim=-1)
        h = self.to_out(attn @ v)
        h = h.permute(0, 2, 1).reshape(B, C, H, W)
        return x + h


class MidBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.res1 = ResnetBlock(ch, ch)
        self.attn = Attention(ch)
        self.res2 = ResnetBlock(ch, ch)

    def forward(self, x):
        return self.res2(self.attn(self.res1(x)))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(3, 128, 3, padding=1)
        self.down_blocks = nn.ModuleList([
            nn.Sequential(ResnetBlock(128, 128), nn.Conv2d(128, 128, 3, stride=2, padding=1)),
            nn.Sequential(ResnetBlock(128, 256), nn.Conv2d(256, 256, 3, stride=2, padding=1)),
            nn.Sequential(ResnetBlock(256, 512), nn.Conv2d(512, 512, 3, stride=2, padding=1)),
        ])
        self.mid_block = MidBlock(512)
        self.conv_norm_out = nn.GroupNorm(32, 512)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(512, 8, 3, padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        for block in self.down_blocks:
            h = block(h)
        h = self.mid_block(h)
        h = self.conv_act(self.conv_norm_out(h))
        return self.conv_out(h)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(4, 512, 3, padding=1)
        self.mid_block = MidBlock(512)
        self.up_blocks = nn.ModuleList([
            nn.Sequential(ResnetBlock(512, 512), nn.Upsample(scale_factor=2), nn.Conv2d(512, 512, 3, padding=1)),
            nn.Sequential(ResnetBlock(512, 256), nn.Upsample(scale_factor=2), nn.Conv2d(256, 256, 3, padding=1)),
            nn.Sequential(ResnetBlock(256, 128), nn.Upsample(scale_factor=2), nn.Conv2d(128, 128, 3, padding=1)),
        ])
        self.conv_norm_out = nn.GroupNorm(32, 128)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(128, 3, 3, padding=1)

    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid_block(h)
        for block in self.up_blocks:
            h = block(h)
        h = self.conv_act(self.conv_norm_out(h))
        return self.conv_out(h)


class SimpleAutoencoderKL(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.quant_conv = nn.Conv2d(8, 8, 1)
        self.post_quant_conv = nn.Conv2d(4, 4, 1)
        # Mimic diffusers.AutoencoderKL.config so this class is drop-in compatible with StableDiffusionPipeline.
        self.config = SimpleNamespace(
            scaling_factor=0.18215,
            use_quant_conv=True,
            use_post_quant_conv=True,
        )

    def encode(self, x: torch.Tensor, return_dict: bool = True):
        h = self.encoder(x)
        h = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(h)
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(self, z: torch.Tensor, return_dict: bool = True, generator=None):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator=None,
    ):
        posterior = self.encode(sample).latent_dist
        z = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
        dec = self.decode(z).sample
        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)
