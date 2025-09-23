import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

from collections import deque
from model import Model
from typing import List

class ModelRunner():
    def __init__(self, device, train_accuracies, val_accuracies, num_epochs):
        self.model: Model = Model()
        self.device: str = device
        self.train_accuracies: List = train_accuracies
        self.val_accuracies: List = val_accuracies
        self.num_epochs: int = num_epochs

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        self.model = self.model.to(self.device)
        self.saved_models = deque(maxlen=10)

    def train(self):
        self.model.train()
        running_loss = 0.0
        running_corrects = 0

        for images, labels in self.model.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            running_corrects += (preds == labels).sum().item()

        epoch_loss = running_loss / len(self.model.train_loader.dataset)
        train_acc = running_corrects / len(self.model.train_loader.dataset)
        self.train_accuracies.append(train_acc)

        return epoch_loss

    def validation(self):
        self.model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in self.model.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
        val_acc = correct / len(self.model.val_loader.dataset)
        self.val_accuracies.append(val_acc)
        return val_acc

    def plot(self):
        plt.plot(range(1, self.num_epochs + 1), self.train_accuracies, label='Training Accuracy', marker='o')
        plt.plot(range(1, self.num_epochs + 1), self.val_accuracies, label='Validation Accuracy', marker='o')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Training vs Validation Accuracy")
        plt.legend()
        plt.grid(True)
        plt.show()