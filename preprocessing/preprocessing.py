"""
Shared preprocessing pipeline for biomedical datasets.

Supported datasets:
    - WDBC (Wisconsin Diagnostic Breast Cancer)
    - UCI Heart Disease

The pipeline performs:
    1. Dataset loading
    2. Dataset-specific target encoding
    3. Train/test splitting
    4. Standardization
    5. PCA dimensionality reduction to 4 features
    6. MinMax scaling to the quantum angle range [0, pi]
    7. Conversion to PyTorch float32 tensors

The returned interface is shared by the hybrid quantum-classical model:

    X_train, X_test, y_train, y_test

X_train and X_test contain 4 quantum-ready features.
"""

import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from .data_loader import load_dataset

# ============================================================
# CONFIGURATION
# ============================================================

N_QUBITS = 4
TEST_SIZE = 0.20
RANDOM_STATE = 42




# ============================================================
# PREPROCESSING
# ============================================================

def preprocess(dataset="wdbc"):
    """
    Prepare a dataset for the hybrid quantum-classical model.

    Pipeline:
        raw features
            ↓
        train/test split
            ↓
        StandardScaler
            ↓
        PCA → 4 components
            ↓
        MinMaxScaler → [0, pi]
            ↓
        PyTorch float32 tensors

    Returns
    -------
    X_train : torch.Tensor
        Shape (N, 4), float32.

    X_test : torch.Tensor
        Shape (N, 4), float32.

    y_train : torch.Tensor
        Binary labels, float32.

    y_test : torch.Tensor
        Binary labels, float32.
    """

    df = load_dataset(dataset)

    # Separate features and target.
    if dataset.lower() == "wdbc":
        X = df.drop(columns=["diagnosis"])
        y = df["diagnosis"]

    elif dataset.lower() == "heart":
        X = df.drop(columns=["target"])
        y = df["target"]

    else:
        raise ValueError(
            f"Unsupported dataset '{dataset}'."
        )

    # Ensure the target contains only binary labels.
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError(
            f"{dataset} target must contain only binary labels 0 and 1."
        )

    print("=" * 60)
    print(f"DATASET: {dataset.upper()}")
    print("=" * 60)

    print("Samples :", len(df))
    print("Features:", X.shape[1])

    print("\nClass distribution:")
    print(y.value_counts())

    # --------------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\nTrain/Test Split:")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test :", y_test.shape)

    # --------------------------------------------------------
    # STANDARDIZATION
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    if X_train_scaled.shape[1] < N_QUBITS:
        raise ValueError(
            f"{dataset} has fewer than {N_QUBITS} features; "
            "cannot perform the required PCA reduction."
        )

    pca = PCA(n_components=N_QUBITS)

    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    explained_variance = pca.explained_variance_ratio_.sum()

    print(
        f"\nPCA components: {N_QUBITS}"
    )

    print(
        f"Explained variance: "
        f"{explained_variance:.4f}"
    )

    # --------------------------------------------------------
    # QUANTUM ANGLE SCALING
    # --------------------------------------------------------

    angle_scaler = MinMaxScaler(
        feature_range=(0, np.pi)
    )

    X_train_angles = angle_scaler.fit_transform(
        X_train_pca
    )

    X_test_angles = angle_scaler.transform(
        X_test_pca
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    assert X_train_angles.shape[1] == N_QUBITS
    assert X_test_angles.shape[1] == N_QUBITS

    assert len(X_train_angles) == len(y_train)
    assert len(X_test_angles) == len(y_test)

    # Training data must lie inside the scaler's target range.
    assert np.all(X_train_angles >= 0)
    assert np.all(X_train_angles <= np.pi)

    # Test data can fall slightly outside [0, pi] because the
    # scaler was fitted only on the training data.
    print(
        f"Test angle range: "
        f"[{X_test_angles.min():.4f}, "
        f"{X_test_angles.max():.4f}]"
    )

    # --------------------------------------------------------
    # CONVERT TO PYTORCH TENSORS
    # --------------------------------------------------------

    X_train_tensor = torch.tensor(
        X_train_angles,
        dtype=torch.float32
    )

    X_test_tensor = torch.tensor(
        X_test_angles,
        dtype=torch.float32
    )

    y_train_tensor = torch.tensor(
        y_train.to_numpy(),
        dtype=torch.float32
    )

    y_test_tensor = torch.tensor(
        y_test.to_numpy(),
        dtype=torch.float32
    )

    # Final contract validation.
    assert X_train_tensor.dtype == torch.float32
    assert X_test_tensor.dtype == torch.float32
    assert y_train_tensor.dtype == torch.float32
    assert y_test_tensor.dtype == torch.float32

    assert X_train_tensor.shape[1] == N_QUBITS
    assert X_test_tensor.shape[1] == N_QUBITS

    assert set(
        y_train_tensor.numpy().astype(int)
    ).issubset({0, 1})

    assert set(
        y_test_tensor.numpy().astype(int)
    ).issubset({0, 1})

    print("\nQuantum-ready data:")
    print("X_train:", X_train_tensor.shape)
    print("X_test :", X_test_tensor.shape)

    return (
        X_train_tensor,
        X_test_tensor,
        y_train_tensor,
        y_test_tensor
    )


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    print("\nTesting WDBC preprocessing...")
    preprocess("wdbc")

    print("\n\nTesting Heart Disease preprocessing...")
    preprocess("heart")

