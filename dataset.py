import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class AnorLondoDataset(Dataset):
    def __init__(self, folder_path, prompt, transform):
        super().__init__()
        self.prompt = prompt
        self.transform = transform
        exts = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"]
        self.image_paths = [p for ext in exts for p in glob.glob(os.path.join(folder_path, ext))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, self.prompt

def create_dataloader(folder_path, prompt, batch_size):
    transform = T.Compose([T.ToTensor()])
    dataset = AnorLondoDataset(folder_path, prompt, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
