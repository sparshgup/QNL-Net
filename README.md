# QNL-Net

This repository houses the code for "A Scalable Quantum Non-local Neural Network for Image Classification"

## QNL-Net Module

The core QNL-Net module is present in the `qnlnet_circuit.py` script, located within the relevant dataset directories.

## Models

The directories `mnist_digit_binaryclass` and `cifar10_binaryclass` consists of the code for the hybrid CNN-QNL-Net 
and PCA-QNL-Net models for the respective datasets. These model files are named as `{dataset}_model_{cnn/pca}.py`.

## Dependencies

| Package                 | Version    |
|-------------------------|------------|
| numpy                   | \>= 1.26.4 |
| qiskit                  | \>= 1.1.0  |
| qiskit_machine_learning | \>= 0.7.2  |
| scikit_learn            | \>= 1.4.2  |
| scipy                   | \>= 1.13.1 |
| torch                   | \>= 2.3.0  |
| torchsummary            | \>= 1.5.1  |
| torchvision             | \>= 0.18.0 |
| pandas                  | \>= 2.1.1  |
| matplotlib              | \>= 3.7.2  |

The dependencies for this project can be found within `requirements.txt` and can be installed by using `pip` in the command line 
(make sure that your pwd is the base of this repo):
```commandline
pip install -r requirements.txt
```

Alternatively, you can simply build a virtual environment using an IDE based on this file. (recommended)

## Execution

- The code files to run the models are also present within the desired dataset directories and are simply named as `{dataset}_{cnn/pca}.py`.
- The datasets are automatically downloaded by PyTorch when the scripts are run.
- The ansatz, feature map reps, ansatz reps, number of epochs, learning rate, and the binary classes to classify can all be configured within these scripts.
- The training data for the model runs is stored in the `epoch_data` subdirectory within the relevant directories `mnist_digit_binaryclass` and `cifar10_binaryclass`.

## Figures & Plots

The `docs` directory contains figures generated programmatically for the QNL-Net ansatz and the training & accuracy plots generated using `plots.ipynb`.
