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

from mnist_digit_multiclass_model_cnn import create_quan_sam, HybridCNNQSA


def main(num_classes, feature_map_reps, ansatz, ansatz_reps):
    # Model Setup
    num_qubits = 4
    output_shape = 2  # Number of classes

    models = []

    for class_label in range(num_classes):
        # Instantiate Model
        quan_sam = create_quan_sam(feature_map_reps, ansatz, ansatz_reps)
        model = HybridCNNQSA(quan_sam)
        models.append(model)

    # Print Model Architecture
    for idx, model in enumerate(models):
        print("================================================================")
        print("Hybrid CNN-Quan-SAM model Instantiated")
        print("================================================================")
        print(f"Model for class {idx} Instantiated")
        print("================================================================")
        print("Model Architecture")
        summary(model, input_size=(1, 28, 28))
        print("================================================================")

    # Dataset (Train & Test) Setup
    batch_size = 32
    n_train_samples = 60000
    n_test_samples = 10000

    # Use pre-defined torchvision function to load MNIST data
    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))])
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))])
    )

    # Filter out labels
    train_idx = np.array([])
    test_idx = np.array([])
    for class_label in range(num_classes):
        train_idx = np.append(train_idx, np.where(np.array(train_dataset.targets) == class_label)[0][:n_train_samples])
        test_idx = np.append(test_idx, np.where(np.array(test_dataset.targets) == class_label)[0][:n_test_samples])

    train_dataset.data = train_dataset.data[train_idx.astype(int)]
    train_dataset.targets = np.array(train_dataset.targets)[train_idx.astype(int)]

    test_dataset.data = test_dataset.data[test_idx.astype(int)]
    test_dataset.targets = np.array(test_dataset.targets)[test_idx.astype(int)]

    # Define torch dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Print training and testing dataset info
    print(train_dataset)
    print(test_dataset)
    print("================================================================")

    # Training and Testing - Model
    loss_func = NLLLoss()
    optimizers = [optim.Adam(model.parameters(), lr=lr) for model in models]
    schedulers = [ExponentialLR(optimizer, gamma=0.9) for optimizer in optimizers]

    for epoch in range(num_epochs):
        epoch_data = []
        for idx, model in enumerate(models):
            model.train()  # Set model to training mode

            total_loss = 0
            correct_train = 0
            total_train = len(train_dataset)

            for data, target in train_loader:
                target_binary = torch.where(target == idx, torch.tensor(1), torch.tensor(0))  # One-vs-Rest

                optimizer = optimizers[idx]
                optimizer.zero_grad()  # Initialize gradient
                output = model(data)  # Forward pass
                loss = loss_func(output, target_binary)  # Calculate loss
                loss.backward()  # Backward pass
                optimizer.step()  # Optimize weights
                total_loss += loss.item() * data.size(0)  # Accumulate loss
                pred = output.argmax(dim=1, keepdim=True)
                correct_train += pred.eq(target_binary.view_as(pred)).sum().item()

            epoch_loss = total_loss / total_train
            epoch_accuracy_train = correct_train / total_train

            # Testing
            model.eval()  # Set model to evaluation mode
            correct_test = 0
            total_test = len(test_dataset)

            with no_grad():
                for data, target in test_loader:
                    target_binary = torch.where(target == idx, torch.tensor(1), torch.tensor(0))  # One-vs-Rest

                    output = model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct_test += pred.eq(target_binary.view_as(pred)).sum().item()

            test_accuracy = correct_test / total_test

            epoch_data.append((epoch + 1, idx, epoch_loss, epoch_accuracy_train, test_accuracy))

            print("Epoch {}: Class {}: Train Loss: {:.4f}; Train Accuracy: {:.4f}; Test Accuracy: {:.4f}".format(
                epoch + 1, idx, epoch_loss, epoch_accuracy_train, test_accuracy))

            schedulers[idx].step()  # Adjust learning rate for next epoch

        # Save metrics to CSV file
        csv_file = f"epoch_data/mnist_multiclass_cnn_01_z{feature_map_reps}_a{ansatz}{ansatz_reps}.csv"
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if epoch == 0:
                writer.writerow(["Epoch", "Class", "Train Loss", "Train Accuracy", "Test Accuracy"])
            writer.writerows(epoch_data)

        print(f"Epoch {epoch + 1} metrics saved to {csv_file}.")

        print("================================================================")


if __name__ == "__main__":
    # Parameters
    num_classes = 10  # Number of classes to include
    feature_map_reps = 1
    ansatz = 0
    ansatz_reps = 1
    num_epochs = 25
    lr = 7.5e-4

    main(num_classes, feature_map_reps, ansatz, ansatz_reps)
