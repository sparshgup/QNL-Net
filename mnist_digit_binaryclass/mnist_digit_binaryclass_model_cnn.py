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
from qiskit.quantum_info import SparsePauliOp, Pauli
from qnlnet_circuit import QNLNetCircuit

num_qubits = 4
output_shape = 2  # Number of classes


# Compose QNL-Net with Feature Map
def create_qnlnet(feature_map_reps, ansatz, ansatz_reps):
    """
    Compose QNL-Net with Feature Map utilizing EstimatorQNN.

    Returns:
        Quantum non-local neural network.
    """
    # Feature Map for Encoding
    feature_map = ZFeatureMap(num_qubits, reps=feature_map_reps)

    # QNL-Net circuit
    qnlnet_instance = QNLNetCircuit(num_qubits=num_qubits, ansatz=ansatz, ansatz_reps=ansatz_reps)
    qnlnet_circuit = qnlnet_instance.get_circuit()

    # QNL-Net circuit
    qc = QuantumCircuit(num_qubits)
    qc.compose(feature_map, inplace=True)
    qc.compose(qnlnet_circuit, inplace=True)

    # EstimatorQNN Observable
    pauli_z_qubit0 = Pauli('Z' + 'I' * (num_qubits - 1))
    observable = SparsePauliOp(pauli_z_qubit0)

    # REMEMBER TO SET input_gradients=True FOR ENABLING HYBRID GRADIENT BACKPROP
    qnlnet = EstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=qnlnet_instance.circuit_parameters(),
        input_gradients=True,
    )

    return qnlnet


# Define torch Module for Hybrid CNN-QNL-Net
class HybridCNNQNLNet(Module):
    """
    HybridCNNQNLNet is a hybrid quantum-classical convolutional neural network
    with QNL-Net.

    Args:
        qnlnet: Quantum non-local neural network.
    """

    def __init__(self, qnlnet):
        super().__init__()
        self.conv1 = Conv2d(1, 2, kernel_size=5)
        self.conv2 = Conv2d(2, 16, kernel_size=5)
        self.dropout = Dropout2d()
        self.flatten = Flatten()
        self.fc1 = Linear(256, 128)
        self.fc2 = Linear(128, num_qubits)  # 4 inputs to QNL-Net

        # Apply torch connector, weights chosen
        # uniformly at random from interval [-1,1].
        self.qnlnet = TorchConnector(qnlnet)

        # output from QNLNN
        self.output_layer = Linear(1, 1)

    def forward(self, x):
        """
        Forward pass of the HybridCNNQNLNet.

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

        # QNLNN
        x = self.qnlnet.forward(x)

        # Post-QNLNN Classical Linear layer
        x = self.output_layer(x)

        x = cat((x, 1 - x), -1)

        return x
