import torch
import yaml
from diffusers import StableDiffusionPipeline

def load_config(config_path="config.yml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_model(lora=False, config=None):
    base_model_path = config.get("BASE_MODEL_PATH")
    lora_model_path = config.get("LORA_MODEL_PATH")

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    pipe = StableDiffusionPipeline.from_pretrained(base_model_path, torch_dtype=torch.float16)
    if lora:
        pipe.unet.load_attn_procs(lora_model_path)  
    pipe.to(device)
    return pipe

if __name__ == '__main__':
    config = load_config()
    lora_option = True  
    pipe = load_model(lora=lora_option, config=config)
