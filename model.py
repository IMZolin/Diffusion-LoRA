from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline
from lora import apply_lora_replacement
from utils import load_config

def load_models(config):
    text_encoder = CLIPTextModel.from_pretrained(config.get("MODEL_PATH"), subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(config.get("MODEL_PATH"), subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(config.get("MODEL_PATH"), subfolder="unet")
    tokenizer = CLIPTokenizer.from_pretrained(config.get("MODEL_PATH"), subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(config.get("MODEL_PATH"), subfolder="scheduler")
    return vae, unet, text_encoder, tokenizer, noise_scheduler

def initialize_pipeline(vae, unet, text_encoder, tokenizer, noise_scheduler, device):
    return StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=noise_scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False
    ).to(device, dtype=torch.float16)


def generate_image(pipe, prompt, save_path="test_lora_all_sd.png"):
    pipe.unet.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()
    with torch.inference_mode():
        image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    image.save(save_path)
    print(f"Generated picture save in : {save_path}")


def load_model(lora=False, config=None, rank=None, alpha=None, device=None):
    vae, unet, text_encoder, tokenizer, noise_scheduler = load_models(config)
    if lora:
        apply_lora_replacement(models=[vae, unet, text_encoder], lora_rank=rank, lora_alpha=alpha)

    pipe = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=noise_scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False
    ).to(device, dtype=torch.float16)
    return pipe
