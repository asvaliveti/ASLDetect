import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from collections import deque
import signal
import sys
import os

from model import Model
from typing import List, Deque
from SavedModel import SavedModel

MODELS_PATH = os.path.join(os.getcwd(), "models")

class ModelRunner():
    def __init__(self, id, num_epochs, learning_rate=0.001, lr_decay=0.01, batch_size=32):
        """
        Initialize the ModelRunner with model ID, number of epochs, learning rate, and learning rate decay.
        Sets up model, device, optimizer, scheduler, and signal handling.
        """
        self.id = id
        self.model: Model = Model(batch_size=batch_size)
        self.device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_accuracies: List = []
        self.val_accuracies: List = []
        self.num_epochs: int = num_epochs

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=lr_decay, patience=10)
        
        self.model = self.model.to(self.device)
        self.saved_models: Deque[SavedModel]  = deque(maxlen=10)

        signal.signal(signal.SIGTERM, self.handle_sigterm)

    def train(self):
        """
        Train the model for one epoch using the training data loader.
        Returns the average loss for the epoch.
        """
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
        """
        Evaluate the model on the validation set and record accuracy and loss.
        Returns the validation accuracy and average validation loss for the current epoch.
        """
        self.model.eval()
        correct = 0
        total_loss = 0.0
        with torch.no_grad():
            for images, labels in self.model.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
        val_acc = correct / len(self.model.val_loader.dataset)
        val_loss = total_loss / len(self.model.val_loader.dataset)
        self.val_accuracies.append(val_acc)
        return val_acc, val_loss
    
    def test(self):
        """
        Evaluate the model on the test set and return the test accuracy.
        """
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.model.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        test_acc = correct / total
        return test_acc


    def plot(self):
        """
        Plot training and validation accuracy over epochs using matplotlib.
        Updates the plot interactively after each epoch.
        """
        if not hasattr(self, 'fig'):
            plt.ion()
            self.fig, self.ax = plt.subplots()
            self.train_line, = self.ax.plot([], [], label='Training Accuracy', marker='o')
            self.val_line, = self.ax.plot([], [], label='Validation Accuracy', marker='o')

            self.ax.set_xlabel("Epochs")
            self.ax.set_ylabel("Accuracy")
            self.ax.set_title("Training vs Validation Accuracy")
            self.ax.legend()
            self.ax.grid(True)

        epochs = range(1, len(self.train_accuracies) + 1)
        self.train_line.set_data(epochs, self.train_accuracies)
        self.val_line.set_data(epochs, self.val_accuracies)

        self.ax.relim()
        self.ax.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def handle_sigterm(self):
        """
        Handle SIGTERM signal for graceful shutdown.
        """
        print("RECEIVED A SIGTERM")
        sys.exit(0)

    def save_models(self):
        """
        Save the last few trained model states to disk.
        """
        best_val = 0
        best_model = None
        while self.saved_models:
            current_model = self.saved_models.pop()
            if best_val < current_model.val_acc:
                best_val = current_model.val_acc
                best_model = current_model

        torch.save(best_model.get_state_dict(), os.path.join(MODELS_PATH, f"{self.id}.pth"))

    def should_stop_early(self, patience=5):
        """
        Determine if training should stop early based on validation accuracy.
        Stops if no improvement for 'patience' epochs.
        """
        if len(self.val_accuracies) < patience + 1:
            return False
        best_acc = max(self.val_accuracies[:-patience])
        recent_accs = self.val_accuracies[-patience:]
        return all(acc <= best_acc for acc in recent_accs)

    def kill_gracefully(self, should_exit):
        """
        Save models and exit the program gracefully.
        """
        test_accuracy = self.test()
        print(f"Test Accuracy: {test_accuracy}")
        self.save_models()
        plt.savefig(os.path.join("model_graphs", f"{self.id}_training_plot.png"))
        if hasattr(self, 'fig'):
            plt.close(self.fig)

        if should_exit:
            sys.exit(0)

    def main_loop(self, num_epochs):
        """
        Main training loop for the model.
        Trains for the specified number of epochs, validates, plots, saves models, and handles early stopping.
        """
        print("starting training")
        for epoch in range(num_epochs):
            try:
                print(f"epoch {epoch}")
                epoch_loss = self.train()
                val_acc, val_loss = self.validation()
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Val Acc: {val_acc:.4f}")
                self.plot()
                current_model = SavedModel(self.model.state_dict(), val_acc)
                self.saved_models.append(current_model)
                self.scheduler.step(val_loss)
                if self.should_stop_early():
                    self.kill_gracefully(False)
            except KeyboardInterrupt:
                self.kill_gracefully(True)
        
        self.kill_gracefully(False)