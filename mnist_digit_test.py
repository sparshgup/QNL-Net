from model import create_qsa_nn, HybridCNNQSA

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import manual_seed, no_grad
from torch.nn import NLLLoss
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# -----------------------------------------------------------------------------
# Load Model
# -----------------------------------------------------------------------------

qsa_nn = create_qsa_nn()
model = HybridCNNQSA(qsa_nn)

# Load desired model
n_samples = 1000
num_epochs = 5
model.load_state_dict(
    torch.load(f"model/model_{n_samples}samples_{num_epochs}epochs.pt")
)

# -----------------------------------------------------------------------------
# Dataset (Test)
# -----------------------------------------------------------------------------

# Set test shuffle seed (for reproducibility)
manual_seed(239)

batch_size = 1
n_samples = 100

# Use pre-defined torchvision function to load MNIST test data
X_test = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()])
)

# Define torch dataloader with filtered data
test_loader = DataLoader(X_test, batch_size=batch_size, shuffle=True)

# -----------------------------------------------------------------------------
# Testing - Model
# -----------------------------------------------------------------------------

loss_func = NLLLoss()

model.eval()  # set model to evaluation mode
with no_grad():

    correct = 0
    total_loss = []

    for batch_idx, (data, target) in enumerate(test_loader):
        output = model(data)
        if len(output.shape) == 1:
            output = output.reshape(1, *output.shape)

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        loss = loss_func(output, target)
        total_loss.append(loss.item())

    print(
        "Performance on test data:\n"
        "Loss: {:.4f}\n"
        "Accuracy: {:.1f}%".format(
            sum(total_loss) / len(total_loss),
            correct / len(test_loader) / batch_size * 100
        )
    )
