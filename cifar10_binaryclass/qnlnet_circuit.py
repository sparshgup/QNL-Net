from qiskit import QuantumCircuit
from qiskit.circuit import Parameter


class QNLNetCircuit:
    def __init__(self, num_qubits=4, ansatz=0, ansatz_reps=1):
        """
        QNLNNCircuit class implements a quantum circuit
        for a non-local neural network.

        Args:
             num_qubits: The number of qubit used in the circuit. It is fixed
                to be 4 qubits for this circuit implementation.
        """
        self.ansatz = ansatz
        self.ansatz_reps = ansatz_reps
        self.num_qubits = num_qubits
        self.circuit = QuantumCircuit(num_qubits)

        # Parameters to be optimized
        self.x_0 = Parameter('x_0')
        self.x_1 = Parameter('x_1')
        self.theta_0 = Parameter('theta_0')
        self.phi_0 = Parameter('phi_0')
        self.g_0 = Parameter('g_0')

        self.x_2 = Parameter('x_2')
        self.x_3 = Parameter('x_3')
        self.theta_1 = Parameter('theta_1')
        self.phi_1 = Parameter('phi_1')
        self.g_1 = Parameter('g_1')

        self.x_4 = Parameter('x_4')
        self.x_5 = Parameter('x_5')
        self.theta_2 = Parameter('theta_2')
        self.phi_2 = Parameter('phi_2')
        self.g_2 = Parameter('g_2')

        # Parameters
        self.parameters = None

        # Create the circuit
        self.build_circuit()

    def build_circuit(self):
        """
        Builds the QNLNN circuit with the desired ansatz
        and number of repetitions
        """
        if self.ansatz == 0:
            if self.ansatz_reps == 1:
                self.ansatz_0_1()
            if self.ansatz_reps == 2:
                self.ansatz_0_2()
            if self.ansatz_reps == 3:
                self.ansatz_0_3()
        elif self.ansatz == 1:
            if self.ansatz_reps == 1:
                self.ansatz_1_1()
            if self.ansatz_reps == 2:
                self.ansatz_1_2()
            if self.ansatz_reps == 3:
                self.ansatz_1_3()
        elif self.ansatz == 2:
            if self.ansatz_reps == 1:
                self.ansatz_2_1()
            if self.ansatz_reps == 2:
                self.ansatz_2_2()
            if self.ansatz_reps == 3:
                self.ansatz_2_3()
        else:
            print("Invalid Ansatz")

    def ansatz_0_1(self):
        # rep 1
        self.circuit.rz(self.x_0, 0)
        self.circuit.ry(self.theta_0, 1)
        self.circuit.ry(self.phi_0, 2)
        self.circuit.rx(self.g_0, 3)

        self.circuit.cx(1, 2)
        self.circuit.cx(2, 3)
        self.circuit.cx(3, 0)

        self.circuit.rz(self.x_1, 0)

    def ansatz_0_2(self):
        # rep 1
        self.circuit.rz(self.x_0, 0)
        self.circuit.ry(self.theta_0, 1)
        self.circuit.ry(self.phi_0, 2)
        self.circuit.rx(self.g_0, 3)

        self.circuit.cx(1, 2)
        self.circuit.cx(2, 3)
        self.circuit.cx(3, 0)

        self.circuit.rz(self.x_1, 0)

        # rep 2
        self.circuit.rz(self.x_2, 0)
        self.circuit.ry(self.theta_1, 1)
        self.circuit.ry(self.phi_1, 2)
        self.circuit.rx(self.g_1, 3)

        self.circuit.cx(1, 2)
        self.circuit.cx(2, 3)
        self.circuit.cx(3, 0)

        self.circuit.rz(self.x_3, 0)

    def ansatz_0_3(self):
        # rep 1
        self.circuit.rz(self.x_0, 0)
        self.circuit.ry(self.theta_0, 1)
        self.circuit.ry(self.phi_0, 2)
        self.circuit.rx(self.g_0, 3)

        self.circuit.cx(1, 2)
        self.circuit.cx(2, 3)
        self.circuit.cx(3, 0)

        self.circuit.rz(self.x_1, 0)

        # rep 2
        self.circuit.rz(self.x_2, 0)
        self.circuit.ry(self.theta_1, 1)
        self.circuit.ry(self.phi_1, 2)
        self.circuit.rx(self.g_1, 3)

        self.circuit.cx(1, 2)
        self.circuit.cx(2, 3)
        self.circuit.cx(3, 0)

        self.circuit.rz(self.x_3, 0)

        # rep 3
        self.circuit.rz(self.x_4, 0)
        self.circuit.ry(self.theta_2, 1)
        self.circuit.ry(self.phi_2, 2)
        self.circuit.rx(self.g_2, 3)

        self.circuit.cx(1, 2)
        self.circuit.cx(2, 3)
        self.circuit.cx(3, 0)

        self.circuit.rz(self.x_5, 0)

    def ansatz_1_1(self):
        # rep 1
        self.circuit.rz(self.x_0, 0)
        self.circuit.ry(self.theta_0, 1)
        self.circuit.ry(self.phi_0, 2)
        self.circuit.rx(self.g_0, 3)

        self.circuit.cx(3, 2)
        self.circuit.cx(2, 1)
        self.circuit.cx(1, 0)

        self.circuit.rz(self.x_1, 0)

    def ansatz_1_2(self):
        # rep 1
        self.circuit.rz(self.x_0, 0)
        self.circuit.ry(self.theta_0, 1)
        self.circuit.ry(self.phi_0, 2)
        self.circuit.rx(self.g_0, 3)

        self.circuit.cx(3, 2)
        self.circuit.cx(2, 1)
        self.circuit.cx(1, 0)

        self.circuit.rz(self.x_1, 0)

        # rep 2
        self.circuit.rz(self.x_2, 0)
        self.circuit.ry(self.theta_1, 1)
        self.circuit.ry(self.phi_1, 2)
        self.circuit.rx(self.g_1, 3)

        self.circuit.cx(3, 2)
        self.circuit.cx(2, 1)
        self.circuit.cx(1, 0)

        self.circuit.rz(self.x_3, 0)

    def ansatz_1_3(self):
        # rep 1
        self.circuit.rz(self.x_0, 0)
        self.circuit.ry(self.theta_0, 1)
        self.circuit.ry(self.phi_0, 2)
        self.circuit.rx(self.g_0, 3)

        self.circuit.cx(3, 2)
        self.circuit.cx(2, 1)
        self.circuit.cx(1, 0)

        self.circuit.rz(self.x_1, 0)

        # rep 2
        self.circuit.rz(self.x_2, 0)
        self.circuit.ry(self.theta_1, 1)
        self.circuit.ry(self.phi_1, 2)
        self.circuit.rx(self.g_1, 3)

        self.circuit.cx(3, 2)
        self.circuit.cx(2, 1)
        self.circuit.cx(1, 0)

        self.circuit.rz(self.x_3, 0)

        # rep 3
        self.circuit.rz(self.x_4, 0)
        self.circuit.ry(self.theta_2, 1)
        self.circuit.ry(self.phi_2, 2)
        self.circuit.rx(self.g_2, 3)

        self.circuit.cx(3, 2)
        self.circuit.cx(2, 1)
        self.circuit.cx(1, 0)

        self.circuit.rz(self.x_5, 0)

    def ansatz_2_1(self):
        # rep 1
        self.circuit.rz(self.x_0, 0)
        self.circuit.ry(self.theta_0, 1)
        self.circuit.ry(self.phi_0, 2)
        self.circuit.rx(self.g_0, 3)

        self.circuit.cx(1, 3)
        self.circuit.cx(3, 2)
        self.circuit.cx(2, 0)

        self.circuit.rz(self.x_1, 0)

    def ansatz_2_2(self):
        # rep 1
        self.circuit.rz(self.x_0, 0)
        self.circuit.ry(self.theta_0, 1)
        self.circuit.ry(self.phi_0, 2)
        self.circuit.rx(self.g_0, 3)

        self.circuit.cx(1, 3)
        self.circuit.cx(3, 2)
        self.circuit.cx(2, 0)

        self.circuit.rz(self.x_1, 0)

        # rep 2
        self.circuit.rz(self.x_2, 0)
        self.circuit.ry(self.theta_1, 1)
        self.circuit.ry(self.phi_1, 2)
        self.circuit.rx(self.g_1, 3)

        self.circuit.cx(1, 3)
        self.circuit.cx(3, 2)
        self.circuit.cx(2, 0)

        self.circuit.rz(self.x_3, 0)

    def ansatz_2_3(self):
        # rep 1
        self.circuit.rz(self.x_0, 0)
        self.circuit.ry(self.theta_0, 1)
        self.circuit.ry(self.phi_0, 2)
        self.circuit.rx(self.g_0, 3)

        self.circuit.cx(1, 3)
        self.circuit.cx(3, 2)
        self.circuit.cx(2, 0)

        self.circuit.rz(self.x_1, 0)

        # rep 2
        self.circuit.rz(self.x_2, 0)
        self.circuit.ry(self.theta_1, 1)
        self.circuit.ry(self.phi_1, 2)
        self.circuit.rx(self.g_1, 3)

        self.circuit.cx(1, 3)
        self.circuit.cx(3, 2)
        self.circuit.cx(2, 0)

        self.circuit.rz(self.x_3, 0)

        # rep 3
        self.circuit.rz(self.x_4, 0)
        self.circuit.ry(self.theta_2, 1)
        self.circuit.ry(self.phi_2, 2)
        self.circuit.rx(self.g_2, 3)

        self.circuit.cx(1, 3)
        self.circuit.cx(3, 2)
        self.circuit.cx(2, 0)

        self.circuit.rz(self.x_5, 0)

    def circuit_parameters(self):
        """
        Sets the parameters to be optimized for the circuit.

        Returns:
             A set containing all parameters.
        """
        # Set parameters
        if self.ansatz_reps == 1:
            self.parameters = {self.x_0, self.x_1, self.theta_0, self.phi_0, self.g_0}
        elif self.ansatz_reps == 2:
            self.parameters = {self.x_0, self.x_1, self.theta_0, self.phi_0, self.g_0,
                               self.x_2, self.x_3, self.theta_1, self.phi_1, self.g_1}
        elif self.ansatz_reps == 3:
            self.parameters = {self.x_0, self.x_1, self.theta_0, self.phi_0, self.g_0,
                               self.x_2, self.x_3, self.theta_1, self.phi_1, self.g_1,
                               self.x_4, self.x_5, self.theta_2, self.phi_2, self.g_2}
        else:
            print("Invalid Number of Ansatz Repetitions")

        return self.parameters

    def get_circuit(self):
        """
        Returns the circuit.
        """
        return self.circuit
