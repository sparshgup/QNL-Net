"""
Script to draw and generate quantum circuit images
"""
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.circuit import Parameter
from scipy.stats import unitary_group

x = Parameter('x')
x_0 = Parameter('x_0')
theta = Parameter('θ')
phi = Parameter('φ')
g = Parameter('g')
x_1 = Parameter('x_1')

# Create a Quantum Circuit
circuit = QuantumCircuit(4)

# Add classical register to measure output
cr = ClassicalRegister(1, name='Z')
circuit.add_register(cr)

###########################################
# Z Feature Map
circuit.h(0)
circuit.h(1)
circuit.h(2)
circuit.h(3)

circuit.p(2.0 * x, 0)
circuit.p(2.0 * theta, 1)
circuit.p(2.0 * phi, 2)
circuit.p(2.0 * g, 3)

###########################################
circuit.barrier()

my_unitary_matrix = unitary_group.rvs(16)
circuit.unitary(my_unitary_matrix, [0, 1, 2, 3], label="Variational\nQuantum\nCircuit\n(VQC)")

###########################################
circuit.barrier()

# Measure qubit 0 (X) and embedding (qubit 3)
circuit.measure([0], cr)

###########################################
# Draw the circuit
style = {
    'backgroundcolor': 'white',
    'barrierfacecolor': '#f6f6f6',
    'dpi': 1000,
    'creglinecolor': "#292c2e",
    "displaycolor": {
        "h": [
            "#ff474c",
            "#000000"
        ],
        "cx": [
            "#000000",
            "#FFFFFF"
        ],
        "rx": [
            "#000066",
            "#FFFFFF"
        ],
        "ry": [
            "#000066",
            "#FFFFFF"
        ],
        "p": [
            "#87CEEB",
            "#000000"
        ],
        "unitary": [
            "#2a2e2e",
            "#FFFFFF"
        ]
    }
}
circuit.draw(output='mpl', style=style, initial_state=True, filename="./docs/model_circuit.png")
