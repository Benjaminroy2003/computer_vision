import torch 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader , Dataset
from pathlib import Path
from PIL import Image

class ImageClassificationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        self.samples = []
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        self.load_dataset()

    def load_dataset(self):
        for cls in self.classes:
            cls_folder = self.root_dir / cls
            for img_path in cls_folder.glob("*.*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                    self.samples.append((img_path, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
    
    def get_loader(self, batch_size=32, shuffle=True):
        """Return DataLoader for this dataset"""
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=4)