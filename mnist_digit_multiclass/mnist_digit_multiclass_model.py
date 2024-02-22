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
import torch.nn.functional as F
from qiskit_machine_learning.connectors import TorchConnector
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZFeatureMap
from qiskit_machine_learning.neural_networks import SamplerQNN

parent_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(parent_dir)

from quantum_self_attention import QuantumSelfAttention

num_qubits = 4
output_shape = 4  # Number of classes


# Interpret for SamplerQNN
def interpretation(x):
    if x % output_shape == 0:
        return 0
    if x % output_shape == 1:
        return 1
    if x % output_shape == 2:
        return 2
    else:
        return 3


# Compose Quantum Self-Attention Neural Network with Feature Map
def create_qsa_nn():
    """
    Compose Quantum Self-Attention Neural Network with Feature Map
    utilizing SamplerQNN.

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
    qsa_nn = SamplerQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=qsa.circuit_parameters(),
        interpret=interpretation,
        output_shape=output_shape,
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
        self.conv1 = Conv2d(1, 2, kernel_size=5)
        self.conv2 = Conv2d(2, 16, kernel_size=5)
        self.dropout = Dropout2d()
        self.flatten = Flatten()
        self.fc1 = Linear(256, 128)
        self.fc2 = Linear(128, num_qubits)  # 4 inputs to Quan-SANN

        # Apply torch connector, weights chosen
        # uniformly at random from interval [-1,1].
        self.qsa_nn = TorchConnector(qsa_nn)

        # output from QSA-NN
        self.output_layer = Linear(output_shape, output_shape)

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

        return x
