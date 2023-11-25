from qiskit import QuantumCircuit
from qiskit.circuit import Parameter


class QuantumSelfAttention:
    def __init__(self, num_qubits=4):
        self.num_qubits = num_qubits
        self.circuit = QuantumCircuit(num_qubits)

        # Parameters to be optimized
        self.x_0 = Parameter('x_0')
        self.x_1 = Parameter('x_1')
        self.theta = Parameter('theta')
        self.phi = Parameter('phi')
        self.g = Parameter('g')

        # Parameters
        self.parameters = None

        # Create the circuit
        self.build_circuit()

    def build_circuit(self):

        # X embedding
        self.circuit.ry(self.x_0, 0)
        # theta embedding
        self.circuit.ry(self.theta, 1)
        # phi embedding
        self.circuit.ry(self.phi, 2)
        # g embedding
        self.circuit.rx(self.g, 3)

        # Entanglement using CNOT gate - theta and phi
        self.circuit.cx(1, 2)

        # Entanglement using CNOT gate - gaussian and linear
        self.circuit.cx(2, 3)

        # Entanglement using CNOT gate - X and embedding output
        self.circuit.cx(3, 0)

        # Rotation gate on q0
        self.circuit.ry(self.x_1, 0)

    def circuit_parameters(self):
        # Set parameters
        self.parameters = {self.x_0, self.x_1, self.theta, self.phi, self.g}

        return self.parameters

    def get_circuit(self):
        return self.circuit
