import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
import os
from PIL import Image
from metrics import calculate_clip_score
from utils import load_config, load_model

st.title("Stable Diffusion Image Generation with LoRA")


with st.sidebar:
    st.header("Experiment Configuration")
    lora_option = st.checkbox("Enable LoRA")
    
    if lora_option:
        rank = st.selectbox("LoRA Rank", [4, 8, 16])
    else:
        rank = None
    
    st.header("Prompt and Inference Settings")
    prompt = st.text_input("Enter prompt:", "A beautiful landscape with mountains and rivers.")
    steps = st.slider("Select number of inference steps", 20, 50, 30)
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

config = load_config()
pipe = load_model(lora=lora_option, config=config)

if st.button("Generate Image"):
    st.write(f"Generating image with prompt: '{prompt}' and {steps} steps.")
    with st.spinner("Generating..."):
        image = pipe(prompt, num_inference_steps=steps, guidance_scale=7.5).images[0]

    st.image(image, caption=f"Generated Image ({steps} steps)", use_column_width=True)

    prompt_dir = os.path.join(output_dir, prompt.replace(" ", "_"))
    os.makedirs(prompt_dir, exist_ok=True)
    image_path = os.path.join(prompt_dir, f"{steps}_steps.png")
    image.save(image_path)
    st.success(f"Image saved to: {image_path}")
    clip_score = calculate_clip_score(image_path, prompt, device="cuda" if torch.cuda.is_available() else "cpu")
    st.write(f"CLIP Score: {clip_score}")