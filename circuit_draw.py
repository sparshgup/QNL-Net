"""
Script to draw and generate quantum circuit images
"""
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.circuit import Parameter
from scipy.stats import unitary_group

####################################################
# Style
#####################################################
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
            "#9a1543",
            "#FFFFFF"
        ],
        "ry": [
            "#003366",
            "#FFFFFF"
        ],
        "rz": [
            "#05697c",
            "#FFFFFF"
        ],
        "p": [
            "#87CEEB",
            "#000000"
        ],
        "unitary": [
            "#0a3463",
            "#FFFFFF"
        ],
        "measure": [
            "#000000",
            "#FFFFFF"
        ]
    }
}
#####################################################

#####################################################
# Circuit
#####################################################
x = Parameter('x')
theta = Parameter('θ')
phi = Parameter('ϕ')
g = Parameter('g')

circuit = QuantumCircuit(4)
cr = ClassicalRegister(1, name='Z')
circuit.add_register(cr)
# Z Feature Map
circuit.h(0)
circuit.h(1)
circuit.h(2)
circuit.h(3)
circuit.p(2.0 * x, 0)
circuit.p(2.0 * theta, 1)
circuit.p(2.0 * phi, 2)
circuit.p(2.0 * g, 3)

circuit.barrier()

my_unitary_matrix = unitary_group.rvs(16)
circuit.unitary(my_unitary_matrix, [0, 1, 2, 3], label="Variational\nQuantum\nCircuit\n(VQC)")

circuit.barrier()

circuit.measure([0], cr)

circuit.draw(output='mpl', style=style, initial_state=True, filename="./docs/model_circuit.png")

#####################################################
# Ansatz 0
#####################################################
ansatz0 = QuantumCircuit(4)
x_0 = Parameter('x\u2080')
x_1 = Parameter('x\u2081')
theta = Parameter('θ\u2080')
phi = Parameter('ϕ\u2080')
g = Parameter('g\u2080')

ansatz0.rz(x_0, 0)
ansatz0.ry(theta, 1)
ansatz0.ry(phi, 2)
ansatz0.rx(g, 3)

ansatz0.cx(1, 2)
ansatz0.cx(2, 3)
ansatz0.cx(3, 0)

ansatz0.rz(x_1, 0)

ansatz0.draw(output='mpl', style=style, initial_state=True, filename="./docs/ansatz0.png")

#####################################################
# Ansatz 1
#####################################################
ansatz1 = QuantumCircuit(4)

ansatz1.rz(x_0, 0)
ansatz1.ry(theta, 1)
ansatz1.ry(phi, 2)
ansatz1.rx(g, 3)

ansatz1.cx(3, 2)
ansatz1.cx(2, 1)
ansatz1.cx(1, 0)

ansatz1.rz(x_1, 0)

ansatz1.draw(output='mpl', style=style, initial_state=True, filename="./docs/ansatz1.png")

#####################################################
# Ansatz 2
#####################################################
ansatz2 = QuantumCircuit(4)

ansatz2.rz(x_0, 0)
ansatz2.ry(theta, 1)
ansatz2.ry(phi, 2)
ansatz2.rx(g, 3)

ansatz2.cx(1, 3)
ansatz2.cx(3, 2)
ansatz2.cx(2, 0)

ansatz2.rz(x_1, 0)

ansatz2.draw(output='mpl', style=style, initial_state=True, filename="./docs/ansatz2.png")
