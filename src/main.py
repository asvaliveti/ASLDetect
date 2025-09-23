from ModelRunner import ModelRunner
import torch
import os

MODELS_PATH = os.path.join(os.getcwd(), "models")
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_accuracies = []
    val_accuracies = []
    num_epochs = 1000

    runner = ModelRunner(device, train_accuracies, val_accuracies, num_epochs)
    for epoch in range(num_epochs):
        print(f"epoch {epoch}")
        epoch_loss = runner.train()
        val_acc = runner.validation()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Val Acc: {val_acc:.4f}")
        runner.saved_models.append(runner.model.state_dict())

    count = 10
    while runner.saved_models:
        torch.save(runner.saved_models.pop(), os.path.join(MODELS_PATH, f"{count}.pth"))
        count -= 1
    
    runner.plot()