import numpy as np
import torch
import torch.optim as optim
from torch import manual_seed, no_grad
from torch.nn import NLLLoss
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchsummary import summary
import csv
from sklearn.decomposition import PCA

from cifar10_binaryclass_model_pca import create_qnlnn, HybridClassicalQNLNN

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

feature_map_reps = 1
ansatz = 0
ansatz_reps = 1
num_epochs = 50
lr = 4e-4
qnlnn = create_qnlnn(feature_map_reps, ansatz, ansatz_reps)
model = HybridClassicalQNLNN(qnlnn)

print("================================================================")
print("Hybrid PCA-QNLNN model Instantiated")
print("================================================================")
print("Model Architecture")
summary(model, input_size=(1, 1, 4))  # Adjust input size to match PCA output
print("================================================================")

# -----------------------------------------------------------------------------
# Dataset (Train & Test)
# -----------------------------------------------------------------------------

# Set train shuffle seed (for reproducibility)
manual_seed(239)

batch_size = 1
n_train_samples = 60000
n_test_samples = 20000

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
# PCA Transformation
# -----------------------------------------------------------------------------

# Convert data to tensor and reshape for PCA
train_data_tensor = torch.tensor(train_dataset.data).reshape(len(train_dataset), -1)
test_data_tensor = torch.tensor(test_dataset.data).reshape(len(test_dataset), -1)

# Convert tensors to numpy for PCA
train_data = train_data_tensor.numpy()
test_data = test_data_tensor.numpy()

# Apply PCA
pca = PCA(n_components=4)
train_pca = pca.fit_transform(train_data)
test_pca = pca.transform(test_data)

# Normalize PCA output
train_pca = (train_pca - np.mean(train_pca)) / np.std(train_pca)
test_pca = (test_pca - np.mean(test_pca)) / np.std(test_pca)

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

    for data, target in zip(train_pca, train_dataset.targets):
        data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        target = torch.tensor(target, dtype=torch.long)
        optimizer.zero_grad()  # Initialize gradient
        output = model(data)  # Forward pass
        loss = loss_func(output, target.unsqueeze(0))  # Calculate loss
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize weights
        total_loss += loss.item()  # Accumulate loss
        pred = output.argmax(dim=1, keepdim=True)
        correct_train += pred.eq(target.view_as(pred)).sum().item()

    epoch_loss = total_loss / total_train
    epoch_accuracy_train = correct_train / total_train

    # Testing
    model.eval()  # Set model to evaluation mode
    correct_test = 0
    total_test = len(test_dataset)

    with no_grad():
        for data, target in zip(test_pca, test_dataset.targets):
            data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
            target = torch.tensor(target, dtype=torch.long)
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
    csv_file = f"epoch_data/cifar10_binaryclass_pca_28_z{feature_map_reps}_a{ansatz}{ansatz_reps}.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train Loss", "Train Accuracy", "Test Accuracy"])
        writer.writerows(epoch_data)

    print(f"Epoch metrics saved to {csv_file}.")
    print("================================================================")
