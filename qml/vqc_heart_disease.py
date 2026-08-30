"""
VQC on the UCI Heart Disease dataset - SAME architecture and pipeline as the Breast Cancer model 
4 qubits, AngleEmbedding + StronglyEntanglingLayers
Trained as its own separate model

This demonstrates that th e platform generalizes across diseases, which was a core requirement of the ps

0 = negative for heart disease, 1 = positibe for heart disease
"""

import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report

from evaluate import evaluate_and_report

# 1] Load Data
df = pd.read_csv("heart.csv")

X = df.drop(columns=["target"]).values
y = df["target"].values

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
print(f"Explained variance by {N_QUBITS} PCA components: "
      f"{pca.explained_variance_ratio_.sum():.3f}")

angle_scaler = MinMaxScaler(feature_range=(0, np.pi))
X_train_angles = angle_scaler.fit_transform(X_train_pca)
X_test_angles = angle_scaler.transform(X_test_pca)

y_train_pm = pnp.array(y_train * 2 - 1, requires_grad=False)
y_test_pm = pnp.array(y_test * 2 - 1, requires_grad=False)


# 4] Quantum Circuit (identical to the Breast Cancer circuiy)
dev = qml.device("default.qubit", wires=N_QUBITS)
N_LAYERS = 3


@qml.qnode(dev, diff_method="backprop")
def circuit(weights, x):
    qml.AngleEmbedding(x, wires=range(N_QUBITS), rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(0))

def variational_classifier(weights, bias, x):
    return circuit(weights, x) + bias

def square_loss(labels, predictions):
    return pnp.mean((labels - qml.math.stack(predictions)) ** 2)

def cost(weights, bias, X, y):
    predictions = [variational_classifier(weights, bias, x) for x in X]
    return square_loss(y, predictions)


# 5] TRAINING
np.random.seed(0)
weight_shape = qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=N_QUBITS)
weights = pnp.array(np.random.uniform(0, 2 * np.pi, weight_shape), requires_grad=True)
bias = pnp.array(0.0, requires_grad=True)

opt = qml.NesterovMomentumOptimizer(stepsize=0.05)
BATCH_SIZE = 16
N_EPOCHS = 15
n_train = len(X_train_angles)

for epoch in range(N_EPOCHS):
    perm = np.random.permutation(n_train)
    X_shuffled = X_train_angles[perm]
    y_shuffled = y_train_pm[perm]

    for start in range(0, n_train, BATCH_SIZE):
        X_batch = X_shuffled[start:start + BATCH_SIZE]
        y_batch = y_shuffled[start:start + BATCH_SIZE]
        weights, bias, _, _ = opt.step(cost, weights, bias, X_batch, y_batch)

    train_preds = [variational_classifier(weights, bias, x) for x in X_train_angles]
    train_acc = pnp.mean(pnp.sign(qml.math.stack(train_preds)) == y_train_pm)
    print(f"Epoch {epoch+1:2d}/{N_EPOCHS} | Train Acc: {train_acc:.4f}")


# 6] TESTING
test_preds = [variational_classifier(weights, bias, x) for x in X_test_angles]
preds_sign = np.sign(qml.math.stack(test_preds))
# Convert the raw VQC output from approximately [-1, +1]
# into a probability-like score where higher means more likely disease.
y_prob_disease = np.clip((np.asarray(test_preds) + 1) / 2, 0, 1)
y_true_01 = ((np.asarray(y_test_pm) + 1) // 2).astype(int)
y_pred_01 = ((np.asarray(preds_sign) + 1) // 2).astype(int)


# 7] Evaluation
evaluate_and_report(
    y_true_01,
    y_pred_01,
    y_prob=y_prob_disease,
    model_name="VQC Heart Disease (4 qubits)",
    save_path="VQC_Heart_confusion_matrix.png",
    disease_label=1,
    class_names=("Disease", "No Disease")
)
