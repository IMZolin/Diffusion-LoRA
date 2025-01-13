import os
import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
from lora import apply_lora_replacement
from utils import collect_lora_parameters, freeze_all_but_lora

import os
from datetime import datetime

import os
from datetime import datetime

def train_lora(
    vae, unet, text_encoder, tokenizer, noise_scheduler, dataloader, device, 
    lora_rank=4, lora_alpha=1.0, num_epochs=10, lr=5e-5, save_dir="experiments"
):
    # Create a unique experiment folder with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(save_dir, f"experiment_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)

    # Save experiment details to a text file
    experiment_info = {
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "num_epochs": num_epochs,
        "learning_rate": lr,
        "device": device,
        "timestamp": timestamp,
    }
    with open(os.path.join(experiment_dir, "experiment_info.txt"), "w") as f:
        for key, value in experiment_info.items():
            f.write(f"{key}: {value}\n")

    # Move models to the correct device
    vae = vae.to(device)
    unet = unet.to(device)
    text_encoder = text_encoder.to(device)

    # Apply LoRA replacements
    apply_lora_replacement(models=[unet, vae, text_encoder], lora_alpha=lora_alpha, lora_rank=lora_rank)
    
    # Freeze all but LoRA layers
    freeze_all_but_lora(unet)
    freeze_all_but_lora(vae)
    freeze_all_but_lora(text_encoder)

    # Collect LoRA parameters
    lora_parameters = collect_lora_parameters([unet, vae, text_encoder])

    # Optimizer and loss
    optimizer = AdamW(lora_parameters, lr=lr)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(num_epochs):
        for images, prompts in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Move images to the correct device
            images = images.to(device, dtype=torch.float16)
            images = images * 2.0 - 1.0  # Normalize to [-1, 1]

            # Encode images into latent space
            latents = vae.encode(images).latent_dist.sample()
            latents = latents * 0.18215  # Scale for consistency with SD1.x
            latents = latents.to(device)  # Ensure latents are on the correct device

            # Tokenize prompts
            inputs = tokenizer(list(prompts), return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            text_embeds = text_encoder(inputs["input_ids"])[0]

            # Add noise to latents
            noise = torch.randn_like(latents, device=device)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=device, dtype=torch.long
            )
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Predict noise with UNet
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeds).sample

            # Compute loss and update weights
            loss = criterion(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} completed. Loss: {loss.item()}")

    # Save LoRA weights only for the last epoch
    weights_path = os.path.join(experiment_dir, "lora_weights_last_epoch.pth")
    torch.save(
        {model_name: model.state_dict() for model_name, model in zip(["vae", "unet", "text_encoder"], [vae, unet, text_encoder])},
        weights_path
    )
    print(f"Final weights saved to {weights_path}.")

