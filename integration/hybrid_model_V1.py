"""
Reusable hybrid quantum-classical model.

This module defines the quantum circuit, hybrid classifier,
and training function. Dataset loading, preprocessing,
evaluation, and experiment execution are handled separately
by run_hybrid.py.
"""

import torch
import torch.nn as nn
import pennylane as qml


# ============================================================
# CONFIGURATION
# ============================================================

N_QUBITS = 4
N_LAYERS = 3

DEFAULT_EPOCHS = 15
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 0.01


# ============================================================
# QUANTUM CIRCUIT
# ============================================================

dev = qml.device(
    "default.qubit",
    wires=N_QUBITS
)


@qml.qnode(
    dev,
    diff_method="backprop",
    interface="torch"
)
def circuit(inputs, weights):

    # Encode the four classical features as qubit rotation angles.
    qml.AngleEmbedding(
        inputs,
        wires=range(N_QUBITS),
        rotation="Y"
    )

    # Trainable variational quantum circuit.
    qml.StronglyEntanglingLayers(
        weights,
        wires=range(N_QUBITS)
    )

    # Return one expectation value from each qubit.
    return [
        qml.expval(qml.PauliZ(w))
        for w in range(N_QUBITS)
    ]


weight_shapes = {
    "weights": qml.StronglyEntanglingLayers.shape(
        n_layers=N_LAYERS,
        n_wires=N_QUBITS
    )
}


# ============================================================
# HYBRID MODEL
# ============================================================

class HybridQuantumClassifier(nn.Module):
    """
    Hybrid binary classifier:

        4 input features
            ↓
        4-qubit quantum layer
            ↓
        4 quantum expectation values
            ↓
        Linear(4 → 1)
            ↓
        output logit

    Sigmoid is applied during inference to convert the logit
    into a probability for the positive class.
    """

    def __init__(self):

        super().__init__()

        self.qlayer = qml.qnn.TorchLayer(
            circuit,
            weight_shapes
        )

        self.classical_out = nn.Linear(
            N_QUBITS,
            1
        )

    def forward(self, x):

        q_out = self.qlayer(x)

        logits = self.classical_out(
            q_out
        ).squeeze(-1)

        return logits


# ============================================================
# TRAINING
# ============================================================

def train_hybrid_model(
    model,
    X_train,
    y_train,
    epochs=DEFAULT_EPOCHS,
    batch_size=DEFAULT_BATCH_SIZE,
    lr=DEFAULT_LEARNING_RATE
):
    """
    Train the hybrid quantum-classical model.

    Parameters
    ----------
    model : HybridQuantumClassifier
        Model to train.

    X_train : torch.Tensor
        Four-feature quantum-ready training data.

    y_train : torch.Tensor
        Binary training labels.

    epochs : int
        Number of training epochs.

    batch_size : int
        Training batch size.

    lr : float
        Adam learning rate.

    Returns
    -------
    HybridQuantumClassifier
        Trained model.
    """

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    # BCEWithLogitsLoss combines sigmoid activation and binary
    # cross-entropy in a numerically stable implementation.
    loss_fn = nn.BCEWithLogitsLoss()

    n = len(X_train)

    for epoch in range(epochs):

        perm = torch.randperm(n)

        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]

        epoch_loss = 0.0

        for start in range(
            0,
            n,
            batch_size
        ):

            X_batch = X_shuffled[
                start:start + batch_size
            ]

            y_batch = y_shuffled[
                start:start + batch_size
            ]

            optimizer.zero_grad()

            logits = model(X_batch)

            loss = loss_fn(
                logits,
                y_batch
            )

            loss.backward()

            optimizer.step()

            epoch_loss += (
                loss.item() *
                len(X_batch)
            )

        avg_loss = epoch_loss / n

        with torch.no_grad():

            train_logits = model(
                X_train
            )

            train_probabilities = torch.sigmoid(
                train_logits
            )

            train_preds = (
                train_probabilities >= 0.5
            ).float()

            train_acc = (
                train_preds == y_train
            ).float().mean().item()

        print(
            f"Epoch {epoch + 1:2d}/{epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Train Acc: {train_acc:.4f}"
        )

    return model