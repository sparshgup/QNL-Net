import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from torch import manual_seed
from torch.nn import NLLLoss
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim

from cifar10_binaryclass_model import create_qsa_nn, HybridCNNQSA

start_time = time.time()  # Start measuring runtime

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

qsa_nn = create_qsa_nn()
model = HybridCNNQSA(qsa_nn)

print("----------------------------------------------")
print("Hybrid CNN-QSA model Instantiated Successfully")
print("----------------------------------------------")

# -----------------------------------------------------------------------------
# Dataset (Train)
# -----------------------------------------------------------------------------

# Set train shuffle seed (for reproducibility)
manual_seed(239)

batch_size = 1
n_samples = 100
num_epochs = 10  # Set number of epochs for training

# Use pre-defined torchvision function to load CIFAR10 data
X_train = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5,), (0.5,), (0.5,))])
)

# Filter out labels, leaving only labels "bird" (2) and "cat" (3)
idx = np.append(
    np.where(np.array(X_train.targets) == 2)[0][:n_samples],
    np.where(np.array(X_train.targets) == 3)[0][:n_samples]
)

X_train.data = X_train.data[idx]
X_train.targets = np.array(X_train.targets)[idx]

# Encode "bird" (2) as 0 and "cat" (3) as 1 in the targets
X_train.targets[X_train.targets == 2] = 0
X_train.targets[X_train.targets == 3] = 1

# Define torch dataloader
train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True)

print(X_train)

print("----------------------------------------------")
print("Training Data Loaded Successfully")
print("----------------------------------------------")

# -----------------------------------------------------------------------------
# Training - Model
# -----------------------------------------------------------------------------

print("----------------------------------------------")
print("Training Model ...")
print("----------------------------------------------")

use_cuda = True
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# Define optimizer, scheduler, and loss function
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = ExponentialLR(optimizer, gamma=0.9)
loss_func = NLLLoss()

# Start training
loss_list = []  # Store loss history
model.train()  # Set model to training mode

for epoch in range(num_epochs):

    total_loss = []

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad(set_to_none=True)  # Initialize gradient
        output = model(data)  # Forward pass
        loss = loss_func(output, target)  # Calculate loss
        loss.backward()  # Backward pass
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient Clipping
        optimizer.step()  # Optimize weights
        total_loss.append(loss.item())  # Store loss

    loss_list.append(sum(total_loss) / len(total_loss))

    scheduler.step()

    print("Epoch [{}/{}]\tTraining Loss: {:.4f}".format(
        epoch + 1, num_epochs, loss_list[-1]))

# -----------------------------------------------------------------------------
# Training Accuracy
# -----------------------------------------------------------------------------

model.eval()  # Set model to evaluation mode

correct_train, total_train = 0, len(X_train)

with torch.no_grad():
    # Training accuracy
    for batch_idx, (data, target) in enumerate(train_loader):
        output = model(data)
        if len(output.shape) == 1:
            output = output.reshape(1, *output.shape)

        pred = output.argmax(dim=1, keepdim=True)
        correct_train += pred.eq(target.view_as(pred)).sum().item()

    training_accuracy = correct_train / total_train

    print("----------------------------------------------")
    print(f"Training Accuracy: {100 * training_accuracy:.2f}%")
    print("----------------------------------------------")

# -----------------------------------------------------------------------------
# Save Model and Runtime
# -----------------------------------------------------------------------------

# Save Model
torch.save(
    model.state_dict(),
    f"model/model_{len(X_train)}samples_{num_epochs}epochs.pt"
)

print("----------------------------------------------")
print("Model Saved Successfully")
print("----------------------------------------------")

# Runtime
end_time = time.time()  # Stop measuring runtime
runtime = end_time - start_time  # Calculate runtime
print("----------------------------------------------")
print(f"Runtime: {runtime:.2f} seconds")
print("----------------------------------------------")
