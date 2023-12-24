import numpy as np
import time
import torch
from torch import manual_seed
from torch.nn import NLLLoss
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim

from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap

from mnist_digit_binaryclass_model import create_qsa_nn, HybridCNNQSA

start_time = time.time()  # Start measuring runtime

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
num_qubits = 4
feature_map = ZZFeatureMap(num_qubits)  # Choose feature map (Z or ZZ)
feature_map_str = "ZZFeatureMap"  # to save model (same as feature map)
qsa_nn = create_qsa_nn(feature_map)
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
n_samples = 10
num_epochs = 10  # Set number of epochs for training

# Use pre-defined torchvision function to load MNIST data
X_train = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5,), (0.5,))])
)

# Filter out labels (originally 0-9), leaving only labels 0 and 1
idx = np.append(
    np.where(X_train.targets == 0)[0][:n_samples],
    np.where(X_train.targets == 1)[0][:n_samples]
)
X_train.data = X_train.data[idx]
X_train.targets = X_train.targets[idx]

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
optimizer = optim.Adam(model.parameters(), lr=1e-4)
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
    f"model/model_{feature_map_str}_{len(X_train)}samples_{num_epochs}epochs.pt"
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
