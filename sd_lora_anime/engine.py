import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _get_trainable(*modules):
    params = []
    for m in modules:
        params += list(filter(lambda p: p.requires_grad, m.parameters()))
    return params


def train_one_epoch_sdxl(
    epoch_idx,
    unet,
    vae,
    tokenizer,
    tokenizer2,
    text_encoder,
    text_encoder2,
    scheduler,
    train_loader,
    val_loader,
    optimizer,
    scheduler_lr,
    device,
    writer,
    save_every: int,
    vae_scaling: float,
    lora_dir: str,
    ckpt_dir: str,
    grad_clip: float,
):
    unet.train()
    vae.eval()
    unet_dtype = torch.float16

    total_loss = 0.0
    train_accum_steps = 0
    step_num = epoch_idx * len(train_loader)

    for batch in train_loader:
        pixel_values = batch["image"].to(device, dtype=torch.float32)
        H, W = pixel_values.shape[-2], pixel_values.shape[-1]

        with torch.no_grad():
            latents = vae.encode(pixel_values.to(vae.dtype)).latent_dist.sample()
            latents = (latents * vae_scaling).to(dtype=torch.float16)

        prompts = batch["prompt"]
        with torch.no_grad():
            tokens_1 = tokenizer(
                prompts, padding="max_length", max_length=tokenizer.model_max_length,
                truncation=True, return_tensors="pt",
            ).input_ids.to(device)
            tokens_2 = tokenizer2(
                prompts, padding="max_length", max_length=tokenizer2.model_max_length,
                truncation=True, return_tensors="pt",
            ).input_ids.to(device)
            encoder_out_2 = text_encoder2(tokens_2, output_hidden_states=True)
            prompt_embeds_2 = encoder_out_2.hidden_states[-2].detach()
            pooled_prompt_embeds = encoder_out_2[0].detach().to(unet_dtype)

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps, (bsz,), device=device
        ).long()
        noisy_latents = scheduler.add_noise(latents, noise, timesteps).to(unet_dtype)

        add_time_ids = torch.tensor(
            [[H, W, 0, 0, H, W]], device=device, dtype=unet_dtype
        ).repeat(bsz, 1)
        added_cond_kwargs = {
            "text_embeds": pooled_prompt_embeds,
            "time_ids": add_time_ids,
        }

        with torch.amp.autocast("cuda", dtype=torch.float16):
            encoder_out_1 = text_encoder(tokens_1, output_hidden_states=True)
            prompt_embeds_1 = encoder_out_1.hidden_states[-2]
            prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1).to(unet_dtype)
            model_pred = unet(
                noisy_latents, timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
            ).sample
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            _get_trainable(unet, text_encoder),
            max_norm=grad_clip,
        )
        optimizer.step()
        optimizer.zero_grad()
        scheduler_lr.step()

        total_loss += loss.item()
        train_accum_steps += 1
        step_num += 1

        if step_num % save_every == 0:
            unet.eval()
            text_encoder.eval()
            val_loss = 0.0
            val_steps = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    val_pixel_values = val_batch["image"].to(device, dtype=torch.float32)
                    val_H, val_W = val_pixel_values.shape[-2], val_pixel_values.shape[-1]
                    val_latents = vae.encode(val_pixel_values.to(vae.dtype)).latent_dist.sample()
                    val_latents = (val_latents * vae_scaling).to(dtype=torch.float16)

                    val_prompts = val_batch["prompt"]
                    val_tokens_1 = tokenizer(
                        val_prompts, padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True, return_tensors="pt",
                    ).input_ids.to(device)
                    val_tokens_2 = tokenizer2(
                        val_prompts, padding="max_length",
                        max_length=tokenizer2.model_max_length,
                        truncation=True, return_tensors="pt",
                    ).input_ids.to(device)
                    encoder_out_2 = text_encoder2(val_tokens_2, output_hidden_states=True)
                    val_prompt_embeds_2 = encoder_out_2.hidden_states[-2]
                    val_pooled = encoder_out_2[0].to(unet_dtype)

                    val_noise = torch.randn_like(val_latents)
                    val_bsz = val_latents.shape[0]
                    val_timesteps = torch.randint(
                        0, scheduler.config.num_train_timesteps, (val_bsz,), device=device,
                    ).long()
                    val_noisy_latents = scheduler.add_noise(val_latents, val_noise, val_timesteps).to(unet_dtype)

                    val_time_ids = torch.tensor(
                        [[val_H, val_W, 0, 0, val_H, val_W]], device=device, dtype=unet_dtype
                    ).repeat(val_bsz, 1)
                    val_added_cond_kwargs = {"text_embeds": val_pooled, "time_ids": val_time_ids}

                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        encoder_out_1 = text_encoder(val_tokens_1, output_hidden_states=True)
                        val_prompt_embeds_1 = encoder_out_1.hidden_states[-2]
                        val_prompt_embeds = torch.cat(
                            [val_prompt_embeds_1, val_prompt_embeds_2], dim=-1
                        ).to(unet_dtype)
                        val_pred = unet(
                            val_noisy_latents, val_timesteps,
                            encoder_hidden_states=val_prompt_embeds,
                            added_cond_kwargs=val_added_cond_kwargs,
                        ).sample
                        val_loss += F.mse_loss(val_pred.float(), val_noise.float(), reduction="mean").item()
                        val_steps += 1

            unet.train()
            text_encoder.train()

            avg_train = total_loss / max(train_accum_steps, 1)
            avg_val = val_loss / max(val_steps, 1)
            writer.add_scalars(
                main_tag="Loss",
                tag_scalar_dict={"train_loss": avg_train, "val_loss": avg_val},
                global_step=step_num,
            )
            print(f"Step {step_num}: Train Loss: {avg_train:.4f}, Valid Loss: {avg_val:.4f}")
            total_loss = 0.0
            train_accum_steps = 0

            unet_out = os.path.join(lora_dir, str(step_num), "unet")
            te_out = os.path.join(lora_dir, str(step_num), "text_encoder")
            _ensure_dir(unet_out)
            _ensure_dir(te_out)
            _ensure_dir(ckpt_dir)
            unet.save_pretrained(unet_out)
            text_encoder.save_pretrained(te_out)
            torch.save(
                {
                    "step": step_num,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler_lr.state_dict(),
                },
                os.path.join(ckpt_dir, f"ckpt_{step_num}.pt"),
            )


def train_one_epoch_sd15(
    epoch_idx,
    unet,
    vae,
    tokenizer,
    text_encoder,
    scheduler,
    train_loader,
    val_loader,
    optimizer,
    scheduler_lr,
    device,
    writer,
    save_every: int,
    vae_scaling: float,
    lora_dir: str,
    ckpt_dir: str,
    grad_clip: float,
):
    unet.train()
    text_encoder.train()
    vae.eval()
    unet_dtype = torch.float16

    total_loss = 0.0
    train_accum_steps = 0
    step_num = epoch_idx * len(train_loader)

    for batch in train_loader:
        pixel_values = batch["image"].to(device, dtype=torch.float32)

        with torch.no_grad():
            latents = vae.encode(pixel_values.to(vae.dtype)).latent_dist.sample()
            latents = (latents * vae_scaling).to(dtype=torch.float16)

        prompts = batch["prompt"]
        with torch.no_grad():
            tokens_1 = tokenizer(
                prompts, padding="max_length", max_length=tokenizer.model_max_length,
                truncation=True, return_tensors="pt",
            ).input_ids.to(device)

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps, (bsz,), device=device,
        ).long()
        noisy_latents = scheduler.add_noise(latents, noise, timesteps).to(unet_dtype)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            encoder_out_1 = text_encoder(tokens_1)
            prompt_embeds_1 = encoder_out_1.last_hidden_state
            model_pred = unet(
                noisy_latents, timesteps,
                encoder_hidden_states=prompt_embeds_1,
            ).sample
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            _get_trainable(unet, text_encoder),
            max_norm=grad_clip,
        )
        optimizer.step()
        optimizer.zero_grad()
        scheduler_lr.step()

        total_loss += loss.item()
        train_accum_steps += 1
        step_num += 1

        if step_num % save_every == 0:
            unet.eval()
            text_encoder.eval()
            val_loss = 0.0
            val_steps = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    val_pixel_values = val_batch["image"].to(device, dtype=torch.float32)
                    val_latents = vae.encode(val_pixel_values.to(vae.dtype)).latent_dist.sample()
                    val_latents = (val_latents * vae_scaling).to(dtype=torch.float16)

                    val_prompts = val_batch["prompt"]
                    val_tokens_1 = tokenizer(
                        val_prompts, padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True, return_tensors="pt",
                    ).input_ids.to(device)

                    val_noise = torch.randn_like(val_latents)
                    val_bsz = val_latents.shape[0]
                    val_timesteps = torch.randint(
                        0, scheduler.config.num_train_timesteps, (val_bsz,), device=device,
                    ).long()
                    val_noisy_latents = scheduler.add_noise(val_latents, val_noise, val_timesteps).to(unet_dtype)

                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        encoder_out_1 = text_encoder(val_tokens_1)
                        val_prompt_embeds_1 = encoder_out_1.last_hidden_state
                        val_pred = unet(
                            val_noisy_latents, val_timesteps,
                            encoder_hidden_states=val_prompt_embeds_1,
                        ).sample
                        val_loss += F.mse_loss(val_pred.float(), val_noise.float(), reduction="mean").item()
                        val_steps += 1

            unet.train()
            text_encoder.train()

            avg_train = total_loss / max(train_accum_steps, 1)
            avg_val = val_loss / max(val_steps, 1)
            writer.add_scalars(
                main_tag="Loss",
                tag_scalar_dict={"train_loss": avg_train, "val_loss": avg_val},
                global_step=step_num,
            )
            print(f"Step {step_num}: Train Loss: {avg_train:.4f}, Valid Loss: {avg_val:.4f}")
            total_loss = 0.0
            train_accum_steps = 0

            unet_out = os.path.join(lora_dir, str(step_num), "unet")
            te_out = os.path.join(lora_dir, str(step_num), "text_encoder")
            _ensure_dir(unet_out)
            _ensure_dir(te_out)
            _ensure_dir(ckpt_dir)
            unet.save_pretrained(unet_out)
            text_encoder.save_pretrained(te_out)
            torch.save(
                {
                    "step": step_num,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler_lr.state_dict(),
                },
                os.path.join(ckpt_dir, f"ckpt_{step_num}.pt"),
            )


def train_vae(
    model,
    train_dataset,
    test_dataset,
    writer,
    epochs: int,
    optimizer,
    loss_fn,
    device,
    batch_size: int,
    save_every: int,
    ckpt_dir: str,
    num_workers: int = 0,
):
    _ensure_dir(ckpt_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = model.to(device).float()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs in train_loader:
            imgs = imgs.to(device)
            z = model.encode(imgs).latent_dist.sample()
            recon = model.decode(z).sample
            loss = loss_fn(recon, imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for imgs in test_loader:
                imgs = imgs.to(device)
                z = model.encode(imgs).latent_dist.sample()
                recon = model.decode(z).sample
                test_loss += loss_fn(recon, imgs).item()

        test_loss /= max(len(test_loader), 1)

        print(f"Epoch {epoch:03d} | train {train_loss:.4f} | test {test_loss:.4f}")
        writer.add_scalars(
            main_tag="Loss",
            tag_scalar_dict={"train_loss": train_loss, "val_loss": test_loss},
            global_step=epoch,
        )

        if epoch % save_every == 0:
            path = os.path.join(ckpt_dir, f"vae_epoch{epoch:03d}.pt")
            torch.save(model.state_dict(), path)
            print(f"Saved: {path}")

    print("Train finished.")
