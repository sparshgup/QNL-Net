"""
Script to generate quantum circuit images
"""
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.circuit import Parameter

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

# X embedding
circuit.ry(x_0, 0)

# theta embedding
circuit.ry(theta, 1)

# phi embedding
circuit.ry(phi, 2)

# g embedding
circuit.rx(g, 3)

# Entanglement using CNOT gate - theta and phi
circuit.cx(1, 2)

# Entanglement using CNOT gate - gaussian and linear
circuit.cx(2, 3)

# Entanglement using CNOT gate - X and embedding output
circuit.cx(3, 0)

# Rotation gate on q0
circuit.ry(x_1, 0)

###########################################
circuit.barrier()

# Measure qubit 0 (X) and embedding (qubit 3)
circuit.measure([0], cr)

###########################################
# Draw the circuit
style = {'backgroundcolor': 'white',
         "displaycolor": {
             "h": [
                 "#DC143C",
                 "#FFFFFF"
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
                 "#008080",
                 "#FFFFFF"
             ],
         }
         }
circuit.draw(output='mpl', style=style, filename="./docs/model_circuit_zfeaturemap.png")
###########################################

######################################################################################
######################################################################################
######################################################################################
x = Parameter('x')
x_0 = Parameter('x_0')
theta = Parameter('θ')
phi = Parameter('φ')
g = Parameter('g')
x_1 = Parameter('x_1')
pi_x = Parameter('(pi - x)')
pi_theta = Parameter('(pi - θ)')
pi_phi = Parameter('(pi - φ)')
pi_g = Parameter('(pi - g)')

# Create a Quantum Circuit
circuit = QuantumCircuit(4)

# Add classical register to measure output
cr = ClassicalRegister(1, name='Z')
circuit.add_register(cr)

###########################################
# ZZ Feature Map
circuit.h(0)
circuit.h(1)
circuit.h(2)
circuit.h(3)

circuit.p(2.0 * x, 0)
circuit.p(2.0 * theta, 1)
circuit.p(2.0 * phi, 2)
circuit.p(2.0 * g, 3)

circuit.cx(0, 1)
circuit.p(2.0 * pi_x * pi_theta, 1)
circuit.cx(0, 1)

circuit.cx(1, 2)
circuit.p(2.0 * pi_theta * pi_phi, 2)
circuit.cx(1, 2)

circuit.cx(2, 3)
circuit.p(2.0 * pi_phi * pi_g, 3)
circuit.cx(2, 3)

###########################################
circuit.barrier()

# X embedding
circuit.ry(x_0, 0)

# theta embedding
circuit.ry(theta, 1)

# phi embedding
circuit.ry(phi, 2)

# g embedding
circuit.rx(g, 3)

# Entanglement using CNOT gate - theta and phi
circuit.cx(1, 2)

# Entanglement using CNOT gate - gaussian and linear
circuit.cx(2, 3)

# Entanglement using CNOT gate - X and embedding output
circuit.cx(3, 0)

# Rotation gate on q0
circuit.ry(x_1, 0)

###########################################
circuit.barrier()

# Measure qubit 0 (X) and embedding (qubit 3)
circuit.measure([0], cr)

###########################################
# Draw the circuit
style = {'backgroundcolor': 'white',
         "displaycolor": {
             "h": [
                 "#DC143C",
                 "#FFFFFF"
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
                 "#008080",
                 "#FFFFFF"
             ],
         }
         }
circuit.draw(output='mpl', style=style, filename="./docs/model_circuit_zzfeaturemap.png")
###########################################