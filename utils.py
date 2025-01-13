import os
import yaml
from PIL import Image


def freeze_all_but_lora(model):
    # Freeze all weights except LoRA
    for module in model.modules():
        # If the module has lora_A / lora_B attributes, these are our LoRA line layers
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True
        # Everything else is frozen
        else:
            for param in module.parameters():
                param.requires_grad = False

def collect_lora_parameters(models: list):
    lora_parameters = []
    for model in models:
        for module in model.modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                lora_parameters.append(module.lora_A)
                lora_parameters.append(module.lora_B)

    if not lora_parameters:
        raise ValueError("No LoRA parameters found. Ensure the layers were correctly replaced.")
    return lora_parameters

def load_config(config_path="config.yml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def compress_to_square(input_folder, output_folder, square_size=256, fill_color=(255, 255, 255)):
    """
    Compress all images in a folder to a square format.
    
    Args:
        input_folder (str): Path to the input folder containing images.
        output_folder (str): Path to the output folder to save compressed images.
        square_size (int): Size of the square output images (default: 512x512).
        fill_color (tuple): RGB color to fill the empty space (default: white).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    supported_formats = (".jpeg", ".jpg", ".png", ".webp")
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(supported_formats):
            try:
                image_path = os.path.join(input_folder, filename)
                with Image.open(image_path) as img:
                    img = img.convert("RGB") if img.mode != "RGB" else img
                    img.thumbnail((square_size, square_size), Image.Resampling.LANCZOS)
                    square_img = Image.new("RGB", (square_size, square_size), fill_color)
                    offset = ((square_size - img.size[0]) // 2, (square_size - img.size[1]) // 2)
                    square_img.paste(img, offset)
                    output_path = os.path.join(output_folder, filename)
                    square_img.save(output_path, quality=95)
                    print(f"Compressed and saved: {output_path}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    input_folder = "source_nupogodi_dataset"  # Change to your input folder path
    output_folder = "nupogodi_dataset"  # Change to your output folder path
    compress_to_square(input_folder, output_folder)