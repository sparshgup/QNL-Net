import numpy as np
import torch
from torch import manual_seed, no_grad
from torch.nn import NLLLoss
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mnist_fashion_binaryclass_model import create_qsa_nn, HybridCNNQSA

# -----------------------------------------------------------------------------
# Load Model
# -----------------------------------------------------------------------------

num_qubits = 4

qsa_nn = create_qsa_nn()
model = HybridCNNQSA(qsa_nn)

# Load desired model
n_samples = 12000
num_epochs = 5
lr = 1.75e-04
op = "adam"
loss_str = "nll"
model.load_state_dict(
    torch.load(f"model/model_{n_samples}samples_{num_epochs}epochs_{op}_lr{lr}_{loss_str}.pt")
)

# -----------------------------------------------------------------------------
# Dataset (Test)
# -----------------------------------------------------------------------------

# Set test shuffle seed (for reproducibility)
manual_seed(239)

batch_size = 1
n_samples = 10000

# Use pre-defined torchvision function to load MNIST test data
X_test = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()])
)

# Filter out labels, leaving only labels 5 (sandal) and 7 (sneaker)
idx = np.append(
    np.where(X_test.targets == 5)[0][:n_samples],
    np.where(X_test.targets == 7)[0][:n_samples]
)
X_test.data = X_test.data[idx]
X_test.targets = X_test.targets[idx]

# Encode 5 (sandal) as 0 and 7 (sneaker) as 1 in the targets
X_test.targets[X_test.targets == 5] = 0
X_test.targets[X_test.targets == 7] = 1

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

    print("------------------------------------------------------")
    print(f"Performance on Test data - {len(X_test)} test samples")
    print("------------------------------------------------------")

    print(
        "Loss: {:.4f}\n"
        "Accuracy: {:.1f}%".format(
            sum(total_loss) / len(total_loss),
            correct / len(test_loader) / batch_size * 100
        )
    )
