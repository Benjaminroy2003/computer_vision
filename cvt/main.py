import torch 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader , Dataset
from pathlib import Path
from PIL import Image
from data_loader import ImageClassificationDataset
import torch.nn as nn
from model import create_cvt_small
from train import Train_model


def main(train_path,test_path,transform,criterion,optimizer,scheduler,device,epochs,model_save_path):

    train_dataset = ImageClassificationDataset(train_path, transform)
    train_dataloader = train_dataset.get_loader(batch_size=16, shuffle=True)

    test_dataset = ImageClassificationDataset(test_path, transform)
    test_dataloader = test_dataset.get_loader(batch_size=16, shuffle=False)

    for images, labels in train_dataloader:
        print(images.shape,"images_shape")
        print(labels.shape,"labels_shape")
        break
    model = create_cvt_small(num_classes=9)
    model.to(device)
    optimizer = optimizer(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = scheduler(optimizer, step_size=30, gamma=0.1)
    model_train = Train_model(model, train_dataloader, test_dataloader, criterion, optimizer, device, epochs, scheduler, model_save_path)
    model_train.train()
    model_train.validation()

if __name__ == "__main__":
    model_save_path = "C:/learning/vision_transformers/cvt/Model_output/cvt"
    train_path = "C:/learning/WBC_DC_dataset/train"
    test_path = "C:/learning/WBC_DC_dataset/test"
    # transforms.Resize((224, 224)),
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam
    scheduler = torch.optim.lr_scheduler.StepLR
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    epochs=50
    main(train_path,test_path,transform,criterion,optimizer,scheduler,device,epochs,model_save_path)