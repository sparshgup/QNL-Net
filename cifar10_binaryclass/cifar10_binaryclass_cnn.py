import numpy as np
import torch.optim as optim
from torch import manual_seed, no_grad
from torch.nn import NLLLoss
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchsummary import summary
import csv

from cifar10_binaryclass_model_cnn import create_qnlnet, HybridCNNQNLNet

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

ansatz = 0
feature_map_reps = 1
ansatz_reps = 1
num_epochs = 100
lr = 3e-4

qnlnn = create_qnlnet(feature_map_reps, ansatz, ansatz_reps)
model = HybridCNNQNLNet(qnlnn)

print("================================================================")
print("Hybrid CNN-QNL-Net model Instantiated")
print("================================================================")
print("Model Architecture")
summary(model, input_size=(3, 32, 32))
print("================================================================")

# -----------------------------------------------------------------------------
# Dataset (Train & Test)
# -----------------------------------------------------------------------------

# Set train shuffle seed (for reproducibility)
manual_seed(239)

batch_size = 1
n_train_samples = 50000
n_test_samples = 10000

# Use pre-defined torchvision function to load CIFAR10 data
train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(
                                      (0.4915, 0.4823, .4468),
                                      (0.2470, 0.2435, 0.2616)
                                  )])
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(
                                      (0.4915, 0.4823, .4468),
                                      (0.2470, 0.2435, 0.2616)
                                  )])
)

# Filter out labels
train_idx = np.append(
    np.where(np.array(train_dataset.targets) == 2)[0][:n_train_samples],
    np.where(np.array(train_dataset.targets) == 8)[0][:n_train_samples]
)

test_idx = np.append(
    np.where(np.array(test_dataset.targets) == 2)[0][:n_test_samples],
    np.where(np.array(test_dataset.targets) == 8)[0][:n_test_samples]
)

train_dataset.data = train_dataset.data[train_idx]
train_dataset.targets = np.array(train_dataset.targets)[train_idx]

test_dataset.data = test_dataset.data[test_idx]
test_dataset.targets = np.array(test_dataset.targets)[test_idx]

# Encode desired classes as targets
train_dataset.targets[train_dataset.targets == 2] = 0
train_dataset.targets[train_dataset.targets == 8] = 1

test_dataset.targets[test_dataset.targets == 2] = 0
test_dataset.targets[test_dataset.targets == 8] = 1

# Define torch dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Print training and testing dataset info
print(train_dataset)
print(test_dataset)
print("================================================================")

# -----------------------------------------------------------------------------
# Training and Testing - Model
# -----------------------------------------------------------------------------

loss_func = NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = ExponentialLR(optimizer, gamma=0.9)

model.train()  # Set model to training mode

epoch_data = []

for epoch in range(num_epochs):
    total_loss = 0
    correct_train = 0
    total_train = len(train_dataset)

    for data, target in train_loader:
        optimizer.zero_grad()  # Initialize gradient
        output = model(data)  # Forward pass
        loss = loss_func(output, target)  # Calculate loss
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize weights
        total_loss += loss.item() * data.size(0)  # Accumulate loss
        pred = output.argmax(dim=1, keepdim=True)
        correct_train += pred.eq(target.view_as(pred)).sum().item()

    epoch_loss = total_loss / total_train
    epoch_accuracy_train = correct_train / total_train

    # Testing
    model.eval()  # Set model to evaluation mode
    correct_test = 0
    total_test = len(test_dataset)

    with no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct_test += pred.eq(target.view_as(pred)).sum().item()

    test_accuracy = correct_test / total_test

    epoch_data.append((epoch + 1, epoch_loss, epoch_accuracy_train, test_accuracy))

    print("Epoch {}: Train Loss: {:.4f}; Train Accuracy: {:.4f}; Test Accuracy: {:.4f}".format(
        epoch + 1, epoch_loss, epoch_accuracy_train, test_accuracy))

    model.train()  # Set model back to training mode
    scheduler.step()  # Adjust learning rate for next epoch
print("================================================================")


def save_to_csv():
    # Write metrics to CSV file
    csv_file = f"epoch_data/cifar10_binaryclass_cnn_28_{ansatz}{feature_map_reps}{ansatz_reps}.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train Loss", "Train Accuracy", "Test Accuracy"])
        writer.writerows(epoch_data)

    print(f"Epoch metrics saved to {csv_file}.")
    print("================================================================")


save_to_csv()
