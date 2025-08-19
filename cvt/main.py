import torch 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader , Dataset
from pathlib import Path
from PIL import Image
from data_loader import ImageClassificationDataset

def main():
    train = "path/to/your/dataset"  
    test = "path/to/your/testset"
    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])

    train_dataset = ImageClassificationDataset("dataset", transform)
    train_dataloader = train_dataset.get_loader(batch_size=16, shuffle=True)

    test_dataset = ImageClassificationDataset("testset", transform)
    test_dataloader = test_dataset.get_loader(batch_size=16, shuffle=False)

    print("Classes:", train_dataloader.classes)
    for images, labels in train_dataloader:
        print(images.shape, labels.shape)
        break