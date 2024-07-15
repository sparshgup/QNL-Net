import torch
from torch.nn import (
    Module,
    Linear,
)
from torch import cat
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


# Define torch Module for Hybrid QNL-Net
class HybridClassicalQNLNet(Module):
    """
    HybridCNNQNLNN is a hybrid quantum-classical PCA with QNL-Net.

    Args:
        qnlnet: Quantum non-local neural network.
    """
    def __init__(self, qnlnet):
        super().__init__()

        self.fc2 = Linear(4, num_qubits)  # 4 inputs to QNLNN

        # Apply torch connector, weights chosen
        # uniformly at random from interval [-1,1].
        self.qnlnet = TorchConnector(qnlnet)

        # output from QNL-Net
        self.output_layer = Linear(1, 1)

    def forward(self, x):
        """
        Forward pass of the HybridClassicalQNLNet.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            x (torch.Tensor): Output tensor.
        """
        x = self.fc2(x)

        # QNL-Net
        x = self.qnlnet.forward(x)

        # Post-QNL-Net Classical Linear layer
        x = self.output_layer(x)

        x = cat((x, 1 - x), -1)

        return x
