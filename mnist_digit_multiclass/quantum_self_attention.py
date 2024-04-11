from numpy import sin, pi

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter


class QuantumSelfAttention:
    def __init__(self, num_qubits=4):
        """
        QuantumSelfAttention class implements a quantum circuit
        for self-attention.

        Args:
             num_qubits: The number of qubit used in the circuit. It is fixed
                to be 4 qubits for this circuit implementation.
        """
        self.num_qubits = num_qubits
        self.circuit = QuantumCircuit(num_qubits)

        # Parameters to be optimized
        self.x_0 = Parameter('x_0')
        self.x_1 = Parameter('x_1')
        self.theta = Parameter('theta')
        self.phi = Parameter('phi')
        self.g = Parameter('g')

        self.x_2 = Parameter('x_2')
        self.x_3 = Parameter('x_3')
        self.x_4 = Parameter('x_4')

        # Parameters
        self.parameters = None

        # Create the circuit
        self.build_circuit()

    def build_circuit(self):
        """
        Builds the quantum self-attention circuit
        """

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

        self.circuit.ry(self.x_2, 1)
        self.circuit.ry(self.x_3, 2)
        self.circuit.ry(self.x_4, 3)

    def circuit_parameters(self):
        """
        Sets the parameters to be optimized for the circuit.

        Returns:
             A set containing all parameters.
        """
        # Set parameters
        self.parameters = {self.x_0, self.x_1, self.theta, self.phi, self.g,
                           self.x_2, self.x_3, self.x_4}

        return self.parameters

    def get_circuit(self):
        """
        Returns the circuit.
        """
        return self.circuit
