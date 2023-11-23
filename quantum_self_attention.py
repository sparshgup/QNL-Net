from qiskit import QuantumCircuit, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.extensions import UnitaryGate
import numpy as np

# Use AerSimulator
simulator = AerSimulator()

# Create a Quantum Circuit
circuit = QuantumCircuit(4)

# Add classical register to measure output
cr = ClassicalRegister(2, name='Z')
circuit.add_register(cr)

# X data encoding
x_theta = 0.3
circuit.ry(0.3, 0)

# Theta embedding
theta = 0.3
circuit.ry(0.3, 1)

# Unitary Gate - theta embedding
weighted_matrix_theta = np.array([[1, 0], [0, 1]])
unitary_theta = UnitaryGate(weighted_matrix_theta, label="W_theta")
circuit.append(unitary_theta, [1])

# Phi embedding
phi = 0.3
circuit.ry(0.5, 2)

# Unitary Gate - phi
weighted_matrix_phi = np.array([[1, 0], [0, 1]])
unitary_phi = UnitaryGate(weighted_matrix_phi, label="W_phi")
circuit.append(unitary_phi, [2])

# Tensor Product (theta and phi) - using CNOT gate
circuit.cx(1, 2)

# Linear (g) embedding
# Unitary Gate - g
weighted_matrix_g = np.array([[1, 0], [0, 1]])
unitary_g = UnitaryGate(weighted_matrix_g, label="W_g")
circuit.append(unitary_g, [3])


# Tensor Product (gaussian and linear) - using CNOT gate
circuit.cx(2, 3)

# Measure qubit 0 (X) and embedding (qubit 3)
circuit.measure([0, 3], cr)
