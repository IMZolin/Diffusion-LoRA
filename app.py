import streamlit as st
import os
import torch
from PIL import Image
from dataset import create_dataloader
from utils import load_config
from metrics import calculate_clip_score
from model import load_model
from train import train_lora 

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

st.title("Stable Diffusion Image Generation with LoRA")

# Sidebar configuration
with st.sidebar:
    st.header("Experiment Configuration")
    lora_option = st.checkbox("Enable LoRA")
    
    if lora_option:
        rank = st.selectbox("LoRA Rank", [4, 8, 16])
        alpha = st.slider("LoRA Alpha", 0.1, 2.0, 1.0)
    else:
        rank = alpha = None
    
    st.header("Prompt and Inference Settings")
    prompt = st.text_input("Enter prompt:", "A beautiful landscape with mountains and rivers.")
    steps = st.slider("Select number of inference steps", 20, 50, 30)
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

config = load_config()  

pipe = load_model(lora=lora_option, config=config, rank=rank, alpha=alpha, device=device)

if st.button("Generate Image"):
    st.write(f"Generating image with prompt: '{prompt}' and {steps} steps.")
    with st.spinner("Generating..."):
        image = pipe(prompt, num_inference_steps=steps, guidance_scale=7.5).images[0]

    st.image(image, caption=f"Generated Image ({steps} steps)", use_container_width=True)

    prompt_dir = os.path.join(output_dir, prompt.replace(" ", "_"))
    os.makedirs(prompt_dir, exist_ok=True)
    image_path = os.path.join(prompt_dir, f"{steps}_steps.png")
    image.save(image_path)
    st.success(f"Image saved to: {image_path}")

    clip_score = calculate_clip_score(image_path, prompt, device)
    st.write(f"CLIP Score: {clip_score}")

    if st.button("Start LoRA Training"):
        st.write(f"Starting LoRA training with rank {rank} and alpha {alpha}.")
        train_lora(
            vae=pipe.vae, 
            unet=pipe.unet, 
            text_encoder=pipe.text_encoder, 
            tokenizer=pipe.tokenizer,
            noise_scheduler=pipe.scheduler,
            dataloader=create_dataloader(folder_path=config.get("DATASET_PATH"), prompt=prompt, batch_size=1),  # Provide the dataloader here
            device=device,
            lora_rank=rank,
            lora_alpha=alpha,
            num_epochs=20,  
            lr=5e-5,
            save_dir=config.get("TRAINING_FOLDER_NAME")
        )

