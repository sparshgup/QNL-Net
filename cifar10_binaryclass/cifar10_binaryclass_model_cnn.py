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

from qnlnn_circuit import QNLNNCircuit

num_qubits = 4
output_shape = 2  # Number of classes


# Interpret for EstimatorQNN
def parity(x):
    return f"{bin(x)}".count("1") % 2


# Compose QNLNN Mechanism with Feature Map
def create_qnlnn(feature_map_reps, ansatz, ansatz_reps):
    """
    Compose QNLNN Mechanism with Feature Map utilizing EstimatorQNN.

    Returns:
        Quantum non-local neural network.
    """
    # Feature Map for Encoding
    feature_map = ZFeatureMap(num_qubits, reps=feature_map_reps)

    # QNLNN circuit
    qnlnn_instance = QNLNNCircuit(num_qubits=num_qubits, ansatz=ansatz, ansatz_reps=ansatz_reps)
    qnlnn_circuit = qnlnn_instance.get_circuit()

    qc = QuantumCircuit(num_qubits)
    qc.compose(feature_map, inplace=True)
    qc.compose(qnlnn_circuit, inplace=True)

    # EstimatorQNN Observable
    pauli_z_qubit0 = Pauli('Z' + 'I' * (num_qubits - 1))
    observable = SparsePauliOp(pauli_z_qubit0)

    # REMEMBER TO SET input_gradients=True FOR ENABLING HYBRID GRADIENT BACKPROP
    qnlnn = EstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=qnlnn_instance.circuit_parameters(),
        input_gradients=True,
    )

    return qnlnn


# Define torch Module for Hybrid CNN-QSA
class HybridCNNQNLNN(Module):
    """
    HybridCNNQNLNN is a hybrid quantum-classical convolutional neural network
    with QNLNN.

    Args:
        qnlnn: Quantum non-local neural network.
    """

    def __init__(self, qnlnn):
        super().__init__()
        self.conv1 = Conv2d(3, 6, kernel_size=5)
        self.conv2 = Conv2d(6, 12, kernel_size=5)
        self.dropout = Dropout2d()
        self.flatten = Flatten()
        self.fc1 = Linear(300, 128)
        self.fc2 = Linear(128, num_qubits)  # 4 inputs to QNLNN

        # Apply torch connector, weights chosen
        # uniformly at random from interval [-1,1].
        self.qnlnn = TorchConnector(qnlnn)

        # output from QNLNN
        self.output_layer = Linear(1, 1)

    def forward(self, x):
        """
        Forward pass of the HybridCNNQNLNN.

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
        x = self.qnlnn.forward(x)

        # Post-QNLNN Classical Linear layer
        x = self.output_layer(x)

        x = cat((x, 1 - x), -1)

        return x
