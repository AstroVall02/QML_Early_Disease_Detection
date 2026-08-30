"""
Experiment runner for the hybrid quantum-classical classifier.

Select a supported dataset, preprocess it, train the shared
hybrid model, evaluate its predictions, and save the results.

Usage:

    python -m integration.run_hybrid

The dataset is selected interactively at runtime.
"""

import time

import numpy as np
import torch

from preprocessing.preprocessing import preprocess
from evaluation.evaluate import evaluate_and_report

from integration.hybrid_model_V1 import (
    HybridQuantumClassifier,
    train_hybrid_model,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_CONFIG = {
    "wdbc": {
        "disease_label": 1,
        "class_names": ("Malignant", "Benign"),
    },

    "heart": {
        "disease_label": 1,
        "class_names": ("Disease", "No Disease"),
    },
}


# ============================================================
# DATASET SELECTION
# ============================================================

def select_dataset():

    dataset = input(
        "Enter dataset (wdbc/heart): "
    ).strip().lower()

    if dataset not in DATASET_CONFIG:

        raise ValueError(
            f"Invalid dataset '{dataset}'. "
            "Please choose 'wdbc' or 'heart'."
        )

    return dataset


# ============================================================
# EXPERIMENT
# ============================================================

def run_experiment(dataset):

    config = DATASET_CONFIG[dataset]

    # Reproducible model initialization and data shuffling.
    torch.manual_seed(0)
    np.random.seed(0)

    # --------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = preprocess(
        dataset
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = HybridQuantumClassifier()

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    training_start = time.perf_counter()

    model = train_hybrid_model(
        model,
        X_train,
        y_train
    )

    training_time = (
        time.perf_counter() -
        training_start
    )

    print(
        f"\nTraining Time: "
        f"{training_time:.4f} seconds"
    )

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    inference_start = time.perf_counter()

    model.eval()

    with torch.no_grad():

        test_logits = model(
            X_test
        )

        test_probabilities = torch.sigmoid(
            test_logits
        )

        test_preds = (
            test_probabilities >= 0.5
        ).float()

    inference_time = (
        time.perf_counter() -
        inference_start
    )

    test_acc = (
        test_preds == y_test
    ).float().mean().item()

    print(
        f"Inference Time: "
        f"{inference_time:.4f} seconds"
    )

    print(
        f"Test Accuracy:  "
        f"{test_acc:.4f}"
    )

    print(
        f"Inference Time/Sample: "
        f"{(inference_time / len(X_test)) * 1000:.4f} ms"
    )

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    results_path = (
        f"{dataset}_hybrid_confusion_matrix.png"
    )

    evaluate_and_report(
        y_test.numpy(),
        test_preds.numpy(),
        test_probabilities.numpy(),
        model_name=f"Hybrid TorchLayer VQC ({dataset})",
        save_path=results_path,
        disease_label=config["disease_label"],
        class_names=config["class_names"],
    )

    return model


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    dataset = select_dataset()

    run_experiment(dataset)