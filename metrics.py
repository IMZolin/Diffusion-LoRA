import time
from pytorch_fid import fid_score
import torch
import torchvision.transforms as T
from skimage.metrics import structural_similarity as ssim
import numpy as np
import lpips
from transformers import CLIPProcessor, CLIPModel
from PIL import Image


def calculate_fid(real_images_dir, generated_images_dir):
    """
    Calculate FID between real and generated images.
    Args:
        real_images_dir (str): Path to the directory with real images.
        generated_images_dir (str): Path to the directory with generated images.
    Returns:
        float: FID score.
    """
    return fid_score.calculate_fid_given_paths([real_images_dir, generated_images_dir], batch_size=32, device='cuda', dims=2048)


def calculate_ssim(img1, img2):
    """
    Calculate SSIM between two images.
    Args:
        img1 (PIL.Image or torch.Tensor): First image.
        img2 (PIL.Image or torch.Tensor): Second image.
    Returns:
        float: SSIM score.
    """
    transform = T.ToTensor()
    img1 = transform(img1).permute(1, 2, 0).cpu().numpy() 
    img2 = transform(img2).permute(1, 2, 0).cpu().numpy()  
    return ssim(img1, img2, multichannel=True, data_range=img1.max() - img1.min())


def calculate_lpips(img1, img2):
    """
    Calculate LPIPS between two images.
    Args:
        img1 (torch.Tensor): First image (CHW, normalized to [-1, 1]).
        img2 (torch.Tensor): Second image (CHW, normalized to [-1, 1]).
    Returns:
        float: LPIPS score.
    """
    loss_fn = lpips.LPIPS(net='vgg').to('cuda')
    img1 = img1.unsqueeze(0).to('cuda')  # Add batch dimension
    img2 = img2.unsqueeze(0).to('cuda')
    return loss_fn(img1, img2).item()


def calculate_clip_score(image_path, prompt, device):
    """
    Calculate CLIP score between an image and a text prompt.
    Args:
        image_path (str): Path to the image.
        prompt (str): Text prompt.
    Returns:
        float: CLIP score (cosine similarity).
    """
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=prompt, images=image, return_tensors="pt", padding=True).to(device)
    
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # Image-text similarity score
    return logits_per_image.softmax(dim=1).max().item()


def measure_inference_time(pipe, prompt):
    """
    Measure inference time for generating an image.
    Args:
        pipe (StableDiffusionPipeline): Diffusion pipeline.
        prompt (str): Text prompt.
    Returns:
        float: Time taken to generate an image (in seconds).
    """
    start_time = time.time()
    _ = pipe(prompt)
    end_time = time.time()
    return end_time - start_time


def measure_gpu_memory():
    """
    Measure GPU memory usage.
    Returns:
        float: GPU memory usage in MB.
    """
    return torch.cuda.memory_allocated() / (1024 * 1024)