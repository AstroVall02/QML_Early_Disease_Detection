"""
Evaluation module for the VQC breast cancer classifier.

This is deliberately separate from the training script
Reuse - `evaluate_and_report()` on any trained model 
[Note: pass in plain numpy arrays of true labels and predicted labels.]

Labels convention: 0 = malignant, 1 = benign
"""

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    recall_score,
)
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_and_report(y_true, y_pred, model_name="Model", save_path=None):
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()

    acc = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred, pos_label=0)
    specificity = recall_score(y_true, y_pred, pos_label=1)

    print(f"{model_name}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Sensitivity (malignant recall):  {sensitivity:.4f}")
    print(f"Specificity (benign recall):     {specificity:.4f}\n")
    print(classification_report(
        y_true, y_pred, labels=[0, 1], target_names=["Malignant", "Benign"]
    ))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Malignant", "Benign"],
        yticklabels=["Malignant", "Benign"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved confusion matrix plot to {save_path}")
    plt.close(fig)

    return {
        "accuracy": acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "confusion_matrix": cm,
    }


if __name__ == "__main__":
    # Quick demo using the same StronglyEntanglingLayers VQC setup
    # already validated, so this can be run byt itself for the pitch.
    import pennylane as qml
    from pennylane import numpy as pnp
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.decomposition import PCA

    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    N_QUBITS = 4
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    pca = PCA(n_components=N_QUBITS)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    angle_scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train_angles = angle_scaler.fit_transform(X_train_pca)
    X_test_angles = angle_scaler.transform(X_test_pca)
    y_train_pm = pnp.array(y_train * 2 - 1, requires_grad=False)
    y_test_pm = pnp.array(y_test * 2 - 1, requires_grad=False)

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

    np.random.seed(0)
    weight_shape = qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=N_QUBITS)
    weights = pnp.array(np.random.uniform(0, 2 * np.pi, weight_shape), requires_grad=True)
    bias = pnp.array(0.0, requires_grad=True)

    opt = qml.NesterovMomentumOptimizer(stepsize=0.05)
    for epoch in range(15):
        perm = np.random.permutation(len(X_train_angles))
        Xs, ys = X_train_angles[perm], y_train_pm[perm]
        for start in range(0, len(Xs), 16):
            weights, bias, _, _ = opt.step(cost, weights, bias, Xs[start:start + 16], ys[start:start + 16])

    test_preds = [variational_classifier(weights, bias, x) for x in X_test_angles]
    preds_sign = np.sign(qml.math.stack(test_preds))

    # convert back from {-1,+1} to {0,1} == {malignant, benign}
    y_true_01 = ((np.asarray(y_test_pm) + 1) // 2).astype(int)
    y_pred_01 = ((np.asarray(preds_sign) + 1) // 2).astype(int)

    evaluate_and_report(
        y_true_01, y_pred_01,
        model_name="VQC (4 qubits)",
        save_path="vqc_confusion_matrix.png",
    )
