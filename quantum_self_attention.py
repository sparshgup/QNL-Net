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
circuit.ry(x_theta, 0)

# Theta embedding
theta = 0.3
circuit.ry(theta, 1)

# Phi embedding
phi = 0.3
circuit.ry(phi, 2)

# Linear (g) embedding - Unitary Gate
weighted_matrix_g = np.array([[1, 0], [0, 1]])
unitary_g = UnitaryGate(weighted_matrix_g, label="Unitary (W_g)")
circuit.append(unitary_g, [3])

# Tensor Product (theta and phi) - using CNOT gate
circuit.cx(1, 2)

# Tensor Product (gaussian and linear) - using CNOT gate
circuit.cx(2, 3)

# Measure qubit 0 (X) and embedding (qubit 3)
circuit.measure([0, 3], cr)
