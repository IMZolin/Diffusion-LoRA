from diffusers import StableDiffusionPipeline
import torch
from metrics import (
    calculate_fid, calculate_ssim, calculate_lpips, 
    calculate_clip_score, measure_inference_time, measure_gpu_memory
)
from PIL import Image
import os

# Select device
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# Set paths for saving results
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# Define experimental configurations
experiments = {
    "baseline": {"lora": False, "rank": None},
    "lora_rank_4": {"lora": True, "rank": 4},
    "lora_rank_8": {"lora": True, "rank": 8},
    "lora_rank_16": {"lora": True, "rank": 16},
}

# Define prompts
prompts = [
    "A beautiful landscape with mountains and rivers.",
    "A futuristic cityscape with glowing neon lights.",
    "An anime character with blue hair."
]

# Define inference steps
steps_list = [20, 30, 50]
base_model_path = "CompVis/stable-diffusion-v1-4"
lora_model_base_path = "sayakpaul/sd-model-finetuned-lora-t4"

# Run experiments
for exp_name, config in experiments.items():
    print(f"Running experiment: {exp_name}")
    
    # Load the pipeline
    if config["lora"]:
        model_path = f"lora_rank_{config['rank']}_model"  # Ensure this path corresponds to the correct LoRA model
        pipe = StableDiffusionPipeline.from_pretrained(base_model_path, torch_dtype=torch.float16)
        pipe.unet.load_attn_procs(model_path)  # Load LoRA attention processors
    else:
        pipe = StableDiffusionPipeline.from_pretrained(base_model_path, torch_dtype=torch.float16)
    
    pipe.to(device)
    
    for prompt in prompts:
        for steps in steps_list:
            # Generate image
            print(f"Prompt: '{prompt}' | Steps: {steps}")
            image = pipe(prompt, num_inference_steps=steps, guidance_scale=7.5).images[0]
            image_path = os.path.join(output_dir, f"{exp_name}_{steps}_steps.png")
            image.save(image_path)
            print(f"Saved image: {image_path}")
            
            # Evaluate metrics
            clip_score = calculate_clip_score(image_path, prompt, device)
            print(f"CLIP score: {clip_score}")
            # Additional metrics (FID, SSIM, LPIPS) could be computed here

# Save results and metrics
print("Experiments completed.")
