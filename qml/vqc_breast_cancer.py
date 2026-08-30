"""
SIH26139: Hybrid Quantum Machine Learning Platform for Early Disease Detection

Standalone Variational Quantum Classifier (VQC) on the Wisconsin Breast
Cancer dataset, using PennyLane.

Goal for V1 prototype: A concerete accuracy number on a 4-qubit VQC, reduced via PCA.
"""

import pennylane as qml
from pennylane import numpy as pnp  # Pennylane provides it's own version of Numpy which is autograd wrapped
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA


# 1] Load and split data into training and testing sets
data = load_breast_cancer()
X, y = data.data, data.target  # y: 0 = malignant, 1 = benign

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 2] Standardise
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 3] PCA - 4 Qubits (Ran tests for 2, 6, 8 qubits as well)
N_QUBITS = 4
pca = PCA(n_components=N_QUBITS)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"\nExplained variance by {N_QUBITS} PCA components: "
      f"{pca.explained_variance_ratio_.sum():.3f}\n")


# 4] Feature Encoding - Angle Embedding via RY Gate (Y-rotation)
# qml.AngleEmbedding() is used in step 5
angle_scaler = MinMaxScaler(feature_range=(0, np.pi))
X_train_angles = angle_scaler.fit_transform(X_train_pca)
X_test_angles = angle_scaler.transform(X_test_pca)

# Labels: convert {0,1} -> {-1,+1} for square-loss training
y_train_pm = pnp.array(y_train * 2 - 1, requires_grad=False)
y_test_pm = pnp.array(y_test * 2 - 1, requires_grad=False)


# 5] Quantum Ciruit
dev = qml.device("default.qubit", wires=N_QUBITS)

N_LAYERS = 3  # depth of the entangling ansatz (currently shallow, planned increase for future prototypes)

@qml.qnode(dev)
def circuit(weights, x):
    qml.AngleEmbedding(x, wires=range(N_QUBITS), rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS)) 
    # Reason for choosing Strong Entanglement: Paper citation {https://arxiv.org/pdf/1803.11173}
    return qml.expval(qml.PauliZ(0))

def variational_classifier(weights, bias, x):
    return circuit(weights, x) + bias


# 6] Loss & Acc funcs
def square_loss(labels, predictions):
    return pnp.mean((labels - qml.math.stack(predictions)) ** 2)

def accuracy(labels, predictions):
    preds_sign = pnp.sign(qml.math.stack(predictions))
    return pnp.mean(preds_sign == labels)

def cost(weights, bias, X, y):
    predictions = [variational_classifier(weights, bias, x) for x in X]
    return square_loss(y, predictions)


# 7] TRAINING
np.random.seed(0)
weight_shape = qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=N_QUBITS)
weights = pnp.array(np.random.uniform(0, 2 * np.pi, weight_shape), requires_grad=True)
bias = pnp.array(0.0, requires_grad=True)

opt = qml.NesterovMomentumOptimizer(stepsize=0.05)
BATCH_SIZE = 16
N_EPOCHS = 15

n_train = len(X_train_angles)

for epoch in range(N_EPOCHS):
    # shuffle each epoch
    perm = np.random.permutation(n_train)
    X_shuffled = X_train_angles[perm]
    y_shuffled = y_train_pm[perm]

    for start in range(0, n_train, BATCH_SIZE):
        X_batch = X_shuffled[start:start + BATCH_SIZE]
        y_batch = y_shuffled[start:start + BATCH_SIZE]

        weights, bias, _, _ = opt.step(
            cost, weights, bias, X_batch, y_batch
        )

    # evaluate every epoch on a subset for speed, full sets at the end
    train_preds = [variational_classifier(weights, bias, x) for x in X_train_angles]
    train_acc = accuracy(y_train_pm, train_preds)
    train_cost = square_loss(y_train_pm, train_preds)

    print(f"Epoch {epoch+1:2d}/{N_EPOCHS} | "
          f"Cost: {train_cost:.4f} | Train Acc: {train_acc:.4f}")

# TESTING
test_preds = [variational_classifier(weights, bias, x) for x in X_test_angles]
test_acc = accuracy(y_test_pm, test_preds)
print(f"\nFinal Test Accuracy: {test_acc:.4f}")
