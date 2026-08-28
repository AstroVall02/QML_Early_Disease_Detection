"""
Member 3 -- Hybrid integration: wraps the quantum circuit (from Member 1)
into a PyTorch model, so it can eventually sit downstream of Member 2's
classical preprocessing pipeline.

IMPORTANT: This file is split into two clearly marked parts.

PART A -- fully independent of Member 2. Builds and trains the hybrid
          PyTorch+quantum model using placeholder data generated locally
          (same PCA+angle-scaling Member 1 already validated). Can be
          built, run, and tested today.

PART B -- the ONLY part that needs Member 2's finished pipeline. It's a
          single function stub with a documented contract. Once Member 2
          delivers `preprocess()` matching this contract, swap it in --
          nothing else in this file changes.
"""

import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import time
from evaluation.evaluate import evaluate_and_report
# ============================================================
# PART A: Everything here works TODAY, independent of Member 2
# ============================================================

N_QUBITS = 4
N_LAYERS = 3

# ---- A1. Quantum circuit (Member 1's design, unchanged) ----
dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev, diff_method="backprop", interface="torch")
def circuit(inputs, weights):
    # NOTE: TorchLayer requires the trainable-input argument to be named
    # exactly "inputs" -- this is a PennyLane convention, not optional.
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(w)) for w in range(N_QUBITS)]


weight_shapes = {"weights": qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=N_QUBITS)}


# ---- A2. Hybrid PyTorch model ----
class HybridQuantumClassifier(nn.Module):
    """
    classical input (4 features, scaled to [0, pi])
        -> quantum layer (4 qubits, StronglyEntanglingLayers)
        -> classical linear layer (4 -> 1)
        -> sigmoid -> probability of "benign"
    """
    def __init__(self):
        super().__init__()
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
        self.classical_out = nn.Linear(N_QUBITS, 1)

    def forward(self, x):
        q_out = self.qlayer(x)
        logits = self.classical_out(q_out).squeeze(-1)
        return logits

# ---- A3. Placeholder data generator (stand-in for Member 2's pipeline) ----
def get_placeholder_data():
    """
    Reproduces the exact PCA + angle-scaling contract Member 2's real
    pipeline must eventually satisfy: N_QUBITS features, scaled to [0, pi].
    Used ONLY so Member 3 can build/test without waiting on Member 2.
    """
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=N_QUBITS)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    angle_scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train_angles = angle_scaler.fit_transform(X_train_pca)
    X_test_angles = angle_scaler.transform(X_test_pca)

    # ========================================================
    # CONTRACT VALIDATION
    # ========================================================

    assert X_train_angles.shape[1] == N_QUBITS
    assert X_test_angles.shape[1] == N_QUBITS

    assert np.all(X_train_angles >= 0)
    assert np.all(X_train_angles <= np.pi)

    # Test data may legitimately fall slightly outside the training range.
    # This is expected because MinMaxScaler is fitted only on training data.
    print(
        f"Test angle range: "
        f"[{X_test_angles.min():.4f}, {X_test_angles.max():.4f}]"
    )

    assert len(X_train_angles) == len(y_train)
    assert len(X_test_angles) == len(y_test)

    assert set(np.unique(y_train)).issubset({0, 1})
    assert set(np.unique(y_test)).issubset({0, 1})

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(
        X_train_angles, dtype=torch.float32
    )
    X_test_tensor = torch.tensor(
        X_test_angles, dtype=torch.float32
    )
    y_train_tensor = torch.tensor(
        y_train, dtype=torch.float32
    )
    y_test_tensor = torch.tensor(
        y_test, dtype=torch.float32
    )

    return (
        X_train_tensor,
        X_test_tensor,
        y_train_tensor,
        y_test_tensor,
    )

# ---- A4. Training loop (standard PyTorch, nothing quantum-specific here) ----
def train_hybrid_model(model, X_train, y_train, epochs=15, batch_size=16, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    n = len(X_train)

    for epoch in range(epochs):
        perm = torch.randperm(n)
        X_shuffled, y_shuffled = X_train[perm], y_train[perm]
        epoch_loss = 0.0

        for start in range(0, n, batch_size):
            X_batch = X_shuffled[start:start + batch_size]
            y_batch = y_shuffled[start:start + batch_size]

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(X_batch)

        avg_loss = epoch_loss / n

        with torch.no_grad():
            train_logits = model(X_train)
            train_probabilities = torch.sigmoid(train_logits)
            train_preds = (train_probabilities >= 0.5).float()

        train_acc = (train_preds == y_train).float().mean().item()

        print(
            f"Epoch {epoch+1:2d}/{epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Train Acc: {train_acc:.4f}"
        )

    return model


# ============================================================
# PART B: The ONLY section that needs Member 2's actual pipeline
# ============================================================

def get_real_data_from_member2():
    """
    *** REPLACE THIS once Member 2's pipeline is ready. ***

    CONTRACT Member 2's `preprocess()` function must satisfy:
      - Input: raw dataset (WDBC and/or Heart Disease)
      - Output: (X_train, X_test, y_train, y_test) where
          X_train, X_test are torch.float32 tensors, shape (N, N_QUBITS)
          all values scaled to the range [0, pi]
          y_train, y_test are torch.float32 tensors of 0/1 labels

    Everything in PART A (model, training loop, evaluation) is written
    against this contract and does NOT need to change once this
    function is filled in -- only this function gets swapped out.
    """
    raise NotImplementedError("Waiting on Member 2's preprocessing pipeline")


# ============================================================
# Demo run using Part A only (placeholder data)
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    X_train, X_test, y_train, y_test = get_placeholder_data()

    model = HybridQuantumClassifier()

    training_start = time.perf_counter()

    model = train_hybrid_model(
        model,
        X_train,
        y_train,
        epochs=15
    )

    training_time = time.perf_counter() - training_start

    print(f"\nTraining Time: {training_time:.4f} seconds")

    inference_start = time.perf_counter()

    with torch.no_grad():
        test_logits = model(X_test)
        test_probabilities = torch.sigmoid(test_logits)
        test_preds = (test_probabilities >= 0.5).float()

    inference_time = time.perf_counter() - inference_start

    test_acc = (test_preds == y_test).float().mean().item()

    print(f"Inference Time: {inference_time:.4f} seconds")
    print(f"Test Accuracy:  {test_acc:.4f}")
    print(
        f"Inference Time/Sample: "
        f"{(inference_time / len(X_test)) * 1000:.4f} ms"
    )
    evaluate_and_report(
    y_test.numpy(),
    test_preds.numpy(),
    test_probabilities.numpy(),
    model_name="Hybrid TorchLayer VQC",
    save_path="hybrid_confusion_matrix.png"
    )
# Quick sanity check: reuse the Part A evaluation contract with evaluate.py
# (uncomment once evaluate.py is in the same folder)
# from evaluate import evaluate_and_report
# evaluate_and_report(y_test.numpy(), test_preds.numpy(), model_name="Hybrid TorchLayer VQC", save_path="hybrid_confusion_matrix.png")
