import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import os

TRAIN_PATH = os.path.join(os.getcwd(), "dataset", "train")
VAL_PATH = os.path.join(os.getcwd(), "dataset", "val")
TEST_PATH = os.path.join(os.getcwd(), "dataset", "test")

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()  # important!
        self.create_datasets()
        self.setup_model(self.num_classes)

    def create_datasets(self):
        transform = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.train_dataset = datasets.ImageFolder(TRAIN_PATH, transform=transform)
        self.val_dataset   = datasets.ImageFolder(VAL_PATH, transform=transform)
        self.test_dataset  = datasets.ImageFolder(TEST_PATH, transform=transform)

        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.val_loader   = DataLoader(self.val_dataset, batch_size=32)
        self.test_loader  = DataLoader(self.test_dataset, batch_size=32)

        self.num_classes = len(self.train_dataset.classes)

    def setup_model(self, num_classes):
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.AdaptiveAvgPool2d((1, 1))  # works with any input size
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
