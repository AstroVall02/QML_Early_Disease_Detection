import pennylane as qml
from pennylane import numpy as pnp
import numpy as np

N_QUBITS = 4
N_LAYERS = 3
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev)
def circuit(weights, x):
    qml.AngleEmbedding(x, wires=range(N_QUBITS), rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(0))

np.random.seed(0)
weight_shape = qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=N_QUBITS)
weights = pnp.array(np.random.uniform(0, 2*np.pi, weight_shape), requires_grad=True)
dummy_x = pnp.array([0.5, 1.0, 1.5, 2.0])

fig, ax = qml.draw_mpl(circuit, style="sketch", decimals=None, level="device")(weights, dummy_x)
fig.set_size_inches(18, 5)
fig.suptitle("Hybrid Quantum-Classical Circuit: Angle Encoding + Strongly Entangling Ansatz", fontsize=13, y=1.05)
fig.savefig("Hybird_circuit_diagram.png", dpi=220, bbox_inches="tight")
print("Saved Hybrid_circuit_diagram.png")
