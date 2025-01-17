from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from utils import load_config

def load_models():
    config = load_config()
    text_encoder = CLIPTextModel.from_pretrained(config.get("MODEL_PATH"), subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(config.get("MODEL_PATH"), subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(config.get("MODEL_PATH"), subfolder="unet")
    tokenizer = CLIPTokenizer.from_pretrained(config.get("MODEL_PATH"), subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(config.get("MODEL_PATH"), subfolder="scheduler")
    return vae, unet, text_encoder, tokenizer, noise_scheduler

def initialize_pipeline(vae, unet, text_encoder, tokenizer, noise_scheduler, device):
    vae.to(device, dtype=torch.float16)
    unet.to(device, dtype=torch.float16)
    # text_encoder.to(device, dtype=torch.float16)
    # text_encoder.to_empty(device, True)
    text_encoder.to_empty(device=device).to(dtype=torch.float16)
    from diffusers import StableDiffusionPipeline
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
        image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
    image.save(save_path)
    print(f"Generated picture save in : {save_path}")



def load_lora_model(model, weights_path: str):
    state_dict = torch.load(weights_path)
    model.load_state_dict(state_dict)