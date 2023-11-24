from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.extensions import UnitaryGate
import numpy as np


class QuantumSelfAttention:
    def __init__(self, num_qubits=4):
        self.num_qubits = num_qubits
        self.circuit = QuantumCircuit(num_qubits)

        # Parameters to be optimized
        self.x = Parameter('x')
        self.theta = Parameter('theta')
        self.phi = Parameter('phi')

        # Weighted matrix for g linear embedding
        self.weighted_matrix_g = np.eye(2)  # Identity matrix as default

        # Parameters
        self.parameters = None

        # Create the circuit
        self.build_circuit()

    def build_circuit(self):

        # X data embedding
        self.circuit.ry(self.x, 0)
        # theta embedding
        self.circuit.ry(self.theta, 1)
        # phi embedding
        self.circuit.ry(self.phi, 2)

        unitary_g = UnitaryGate(self.weighted_matrix_g, label="Unitary (W_g)")
        self.circuit.append(unitary_g, [3])

        # Entanglement using CNOT gate -Tensor Product (theta and phi)
        self.circuit.cx(1, 2)

        # Entanglement using CNOT gate - Tensor Product (gaussian and linear)
        self.circuit.cx(2, 3)

        # Entanglement using CNOT gate - (to replicate Element-wise Sum (X and embedding output))
        self.circuit.cx(3, 0)

    def circuit_parameters(self):
        # Set parameters
        self.parameters = {self.x, self.theta, self.phi}

        return self.parameters

    def get_circuit(self):
        return self.circuit
