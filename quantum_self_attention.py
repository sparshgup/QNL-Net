from qiskit import QuantumCircuit, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.extensions import UnitaryGate
import numpy as np

# Use Aer's AerSimulator
simulator = AerSimulator()

# Weighted matrix (sample for testing)
weighted_matrix = np.array([[1, 0], [0, 1]])
wm_gate = UnitaryGate(weighted_matrix)

# Create a Quantum Circuit
circuit = QuantumCircuit(4)

# Add classical register to measure output
cr = ClassicalRegister(2, name='Z')
circuit.add_register(cr)

# data encoding
theta = 0.3
circuit.ry(0.3, 0)

# Theta embedding
theta = 0.3
circuit.ry(0.3, 1)
circuit.append(wm_gate, [1])

# Phi embedding
phi = 0.3
circuit.ry(0.5, 2)
circuit.append(wm_gate, [2])

# Tensor Product (theta and phi)
circuit.cnot(1, 2)

# Linear embedding
circuit.append(wm_gate, [3])

# Tensor Product (gaussian and linear)
circuit.cnot(2, 3)

# Measure qubit 0 (X) and embedding (qubit 3)
circuit.measure([0, 3], cr)
