import streamlit as st
import os
import torch
from utils import load_config
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from model import (
    initialize_pipeline,
    load_models
)

from lora import (
    apply_lora_replacement,
    enable_lora,
    disable_lora
)
# Function to initialize models and pipeline
def setup_pipeline(device, lora_rank, lora_alpha):
    config = load_config()
    vae, unet, text_encoder, tokenizer, noise_scheduler = load_models()
    apply_lora_replacement(models=[unet, vae, text_encoder], lora_rank=lora_rank, lora_alpha=lora_alpha)
    pipe = initialize_pipeline(vae, unet, text_encoder, tokenizer, noise_scheduler, device)
    return pipe, vae, unet, text_encoder

# Streamlit App
st.title("Image Generation with LoRA Customization")

# Sidebar for settings
st.sidebar.header("Settings")
device = "cuda" if torch.cuda.is_available() else "cpu"
lora_rank = st.sidebar.slider("LoRA Rank", min_value=4, max_value=256, value=128)
lora_alpha = st.sidebar.slider("LoRA Alpha", min_value=1.0, max_value=128.0, value=64.0)
num_inference_steps = st.sidebar.number_input("Number of Inference Steps", min_value=1, max_value=100, value=50)

# Enable or disable specific model components
st.sidebar.subheader("Enable/Disable Model Components")
enable_vae = st.sidebar.checkbox("Enable VAE LoRA", value=True)
enable_unet = st.sidebar.checkbox("Enable UNet LoRA", value=True)
enable_text_encoder = st.sidebar.checkbox("Enable Text Encoder LoRA", value=True)

# Prompt input
prompt = st.text_input("Enter your prompt:", "The wolf from the cartoon 'Well, Wait!'")

# Load and setup pipeline
st.write("Initializing models...")
pipe, vae, unet, text_encoder = setup_pipeline(device, lora_rank, lora_alpha)

# Apply enable/disable settings
if enable_vae:
    enable_lora(vae, output=False)
else:
    disable_lora(vae, output=False)

if enable_unet:
    enable_lora(unet, output=False)
else:
    disable_lora(unet, output=False)

if enable_text_encoder:
    enable_lora(text_encoder, output=False)
else:
    disable_lora(text_encoder, output=False)

st.write("Models initialized successfully!")

# Generate image
if st.button("Generate Image"):
    with st.spinner("Generating image..."):
        pipe.unet.eval()
        pipe.vae.eval()
        pipe.text_encoder.eval()
        with torch.inference_mode():
            image = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=7.5).images[0]
        
        # Display the generated image
        st.image(image, caption="Generated Image", use_column_width=True)
        save_path = os.path.join("outputs", f"{prompt.replace(' ', '_')}.png")
        image.save(save_path)
        st.success(f"Image saved to {save_path}")
