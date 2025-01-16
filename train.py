import os
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
from lora import apply_lora_replacement
from utils import collect_lora_parameters, freeze_all_but_lora
from datetime import datetime
from torch.amp import GradScaler, autocast
from torchvision import transforms
from torchvision.transforms import RandomErasing

def train_lora(
    vae, unet, text_encoder, tokenizer, noise_scheduler, dataloader, device, 
    lora_rank=4, lora_alpha=1.0, num_epochs=10, lr=5e-5, save_dir="experiments", 
    accumulation_steps=4
):
    # Initialize gradient scaler for mixed precision
    scaler = GradScaler()

    # Create a unique experiment folder with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(save_dir, f"experiment_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)

    # Save experiment details
    experiment_info = {
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "num_epochs": num_epochs,
        "learning_rate": lr,
        "device": device,
        "accumulation_steps": accumulation_steps,
        "timestamp": timestamp,
    }
    with open(os.path.join(experiment_dir, "experiment_info.txt"), "w") as f:
        for key, value in experiment_info.items():
            f.write(f"{key}: {value}\n")

    # Move models to device
    vae = vae.to(device)
    unet = unet.to(device)
    text_encoder = text_encoder.to(device)

    # Apply LoRA
    apply_lora_replacement(models=[unet, vae, text_encoder], lora_alpha=lora_alpha, lora_rank=lora_rank)
    freeze_all_but_lora(unet)
    freeze_all_but_lora(vae)
    freeze_all_but_lora(text_encoder)
    
    # Collect LoRA parameters and setup optimizer
    lora_parameters = collect_lora_parameters([unet, vae, text_encoder])
    optimizer = AdamW(lora_parameters, lr=lr)
    criterion = nn.MSELoss()

    # Initialize loss list for visualization
    epoch_losses = []

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        optimizer.zero_grad()
        
        for step, (images, prompts) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            torch.cuda.empty_cache()
            
            # Apply data augmentation
            images = augment_data(images)  

            images = images.to(device, dtype=torch.float16)
            images = images * 2.0 - 1.0  

            inputs = tokenizer(list(prompts), return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            text_embeds = text_encoder(inputs["input_ids"])[0]

            latents = vae.encode(images).latent_dist.sample()
            latents = latents * 0.18215
            latents = latents.to(device)

            noise = torch.randn_like(latents, device=device)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, 
                (latents.shape[0],), device=device, dtype=torch.long
            )
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with autocast(device_type=device.type):
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeds).sample
                loss = criterion(noise_pred, noise) / accumulation_steps
            
            # Backward pass
            scaler.scale(loss).backward()
            running_loss += loss.item()
            if (step + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        
        # Log average loss for the epoch
        avg_loss = running_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1} completed. Avg Loss: {avg_loss}")

    # Save final weights
    weights_path = os.path.join(experiment_dir, "lora_weights_last_epoch.pth")
    torch.save(
        {model_name: model.state_dict() for model_name, model in zip(["vae", "unet", "text_encoder"], [vae, unet, text_encoder])},
        weights_path
    )
    draw_loss(num_epochs, epoch_losses, experiment_dir)
    print(f"Final weights saved to {weights_path}.")



def augment_data(images):
    """Apply aggressive data augmentation including random erasing."""
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        # transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        # RandomErasing(p=0.5),  # Randomly erase parts of the image
    ])
    augmented_images = torch.stack([transform(image) for image in images])
    return augmented_images


def draw_loss(num_epochs, epoch_losses, experiment_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o', label="Training Loss")
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(experiment_dir, "loss_curve.png"))
    plt.close()
    print(f"Loss curve saved to {os.path.join(experiment_dir, 'loss_curve.png')}")

