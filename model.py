from quantum_self_attention import QuantumSelfAttention

from torch.nn import (
    Module,
    Conv2d,
    Linear,
    Dropout2d,
)
import torch.nn.functional as F
from qiskit_machine_learning.connectors import TorchConnector

from qiskit import QuantumCircuit
from qiskit.circuit.library import ZFeatureMap

from qiskit_machine_learning.neural_networks import SamplerQNN


num_qubits = 4
output_shape = 10  # Number of classes


# Interpret for SamplerQNN
def interpretation(x):
    """
    Interpret function for SamplerQNN

    Args:
        x: QNN output

    Returns:
        Class label
    """
    binary_x = f"{x:b}"  # Convert to binary representation
    decimal_x = int(binary_x, 2)  # Convert binary to decimal
    return decimal_x % output_shape


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
        self.fc1 = Linear(256, 64)
        self.fc2 = Linear(64, num_qubits)  # 4-dimensional input to QSA-NN

        # Apply torch connector, weights chosen
        # uniformly at random from interval [-1,1].
        self.qsa_nn = TorchConnector(qsa_nn)

        # output from QSA-NN
        # self.output_layer = Linear(output_shape, 1)
        # set to (output_shape, batch_size) if batch_size > 1

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
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # QSA-NN
        x = self.qsa_nn.forward(x)

        # # Post-QSA Classical computation (only use if batch_size > 1)
        # x = self.output_layer(x)
        # x = sum(x, dim=0)  # Sum the tensors

        # Post-QSA Softmax layer for multi-class probabilities
        x = F.softmax(x, dim=1)

        return x
