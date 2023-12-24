import sys
import os
import torch
from torch.nn import (
    Module,
    Conv2d,
    Linear,
    Dropout2d,
    Flatten,
)
from torch import cat
import torch.nn.functional as F
from qiskit_machine_learning.connectors import TorchConnector
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZFeatureMap
from qiskit_machine_learning.neural_networks import EstimatorQNN

parent_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(parent_dir)

from quantum_self_attention import QuantumSelfAttention

num_qubits = 4
output_shape = 2  # Number of classes


# Interpret for EstimatorQNN
def parity(x):
    return f"{bin(x)}".count("1") % 2


# Compose Quantum Self-Attention Neural Network with Feature Map
def create_qsa_nn():
    """
    Compose Quantum Self-Attention Neural Network with Feature Map
    utilizing EstimatorQNN.

    Returns:
        Quantum neural network with self-attention.
    """
    # Feature Map for Encoding
    feature_map = ZFeatureMap(num_qubits)

    # Quantum Self Attention circuit
    qsa = QuantumSelfAttention(num_qubits=num_qubits)
    qsa_circuit = qsa.get_circuit()

    # QSA NN circuit
    qc = QuantumCircuit(num_qubits)
    qc.compose(feature_map, inplace=True)
    qc.compose(qsa_circuit, inplace=True)

    # REMEMBER TO SET input_gradients=True FOR ENABLING HYBRID GRADIENT BACKPROP
    qsa_nn = EstimatorQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=qsa.circuit_parameters(),
        input_gradients=True,
    )

    return qsa_nn


# Define torch Module for Hybrid CNN-QSA
class HybridCNNQSA(Module):
    """
    HybridCNNQSA is a hybrid quantum-classical convolutional neural network
    with Quantum Self Attention.

    Args:
        qsa_nn: Quantum neural network with self-attention.
    """
    def __init__(self, qsa_nn):
        super().__init__()
        self.conv1 = Conv2d(3, 6, kernel_size=5)
        self.conv2 = Conv2d(6, 12, kernel_size=5)
        self.dropout = Dropout2d()
        self.flatten = Flatten()
        self.fc1 = Linear(300, 128)
        self.fc2 = Linear(128, num_qubits)  # 4-dimensional input to QSA-NN

        # Apply torch connector, weights chosen
        # uniformly at random from interval [-1,1].
        self.qsa_nn = TorchConnector(qsa_nn)

        # output from QSA-NN
        self.output_layer = Linear(1, 1)

    def forward(self, x):
        """
        Forward pass of the HybridCNNQSA.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            x (torch.Tensor): Output tensor.
        """
        # CNN
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # QSA-NN
        x = self.qsa_nn.forward(x)

        # Post-QSA Classical Linear layer
        x = self.output_layer(x)

        x = cat((x, 1 - x), -1)

        return x
