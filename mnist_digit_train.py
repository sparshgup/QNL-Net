import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from torch import manual_seed
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim

from model import create_qsa_nn, HybridCNNQSA

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
manual_seed(42)

batch_size = 1
n_samples = 100

# Use pre-defined torchvision function to load MNIST train data
X_train = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()])
)

# Define torch dataloader
train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True)

print("----------------------------------------------")
print("Training Data Loaded Successfully")
print("----------------------------------------------")

# -----------------------------------------------------------------------------
# Training - Model
# -----------------------------------------------------------------------------

print("----------------------------------------------")
print("Training Model ...")
print("----------------------------------------------")

# Define model, optimizer, and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_func = CrossEntropyLoss()

# Start training
num_epochs = 5  # Set number of epochs
loss_list = []  # Store loss history
model.train()  # Set model to training mode

for epoch in range(num_epochs):

    total_loss = []

    for batch_idx, (data, target) in enumerate(train_loader):

        optimizer.zero_grad(set_to_none=True)  # Initialize gradient
        output = model(data)  # Forward pass
        # target = target.float() (only use if using sum in model forward pass)
        loss = loss_func(output, target)  # Calculate loss
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize weights
        total_loss.append(loss.item())  # Store loss

    loss_list.append(sum(total_loss) / len(total_loss))

    print("Training [{:.0f}%]\tLoss: {:.4f}".format(100.0 * (epoch + 1) /
                                                    num_epochs, loss_list[-1]))


# Save Model
torch.save(
    model.state_dict(),
    f"model/model_{n_samples}samples_{num_epochs}epochs.pt"
)

print("----------------------------------------------")
print("Model Saved Successfully")
print("----------------------------------------------")


# Runtime
end_time = time.time()  # Stop measuring runtime
runtime = end_time - start_time  # Calculate runtime
print("----------------------------------------------")
print("Runtime: {:.2f} seconds", runtime)
print("----------------------------------------------")
