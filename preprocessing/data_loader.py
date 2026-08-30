"""
Dataset loading and dataset-specific cleaning.

This module provides a common interface for loading the datasets
used by the quantum and hybrid models.
"""

import os
import pandas as pd


WDBC_PATH = "data/wdbc.data"
HEART_PATH = "data/heart.csv"


def load_wdbc():
    """
    Load and clean the Wisconsin Diagnostic Breast Cancer dataset.

    Labels:
        0 = Benign
        1 = Malignant
    """

    features = [
        "radius_mean",
        "texture_mean",
        "perimeter_mean",
        "area_mean",
        "smoothness_mean",
        "compactness_mean",
        "concavity_mean",
        "concave_points_mean",
        "symmetry_mean",
        "fractal_dimension_mean",

        "radius_se",
        "texture_se",
        "perimeter_se",
        "area_se",
        "smoothness_se",
        "compactness_se",
        "concavity_se",
        "concave_points_se",
        "symmetry_se",
        "fractal_dimension_se",

        "radius_worst",
        "texture_worst",
        "perimeter_worst",
        "area_worst",
        "smoothness_worst",
        "compactness_worst",
        "concavity_worst",
        "concave_points_worst",
        "symmetry_worst",
        "fractal_dimension_worst",
    ]

    columns = ["id", "diagnosis"] + features

    if not os.path.exists(WDBC_PATH):
        raise FileNotFoundError(
            f"WDBC dataset not found at: {WDBC_PATH}"
        )

    df = pd.read_csv(
        WDBC_PATH,
        header=None,
        names=columns,
    )

    # Remove the patient identifier; it is not a predictive feature.
    df = df.drop(columns=["id"])

    # Convert diagnosis to binary labels.
    df["diagnosis"] = df["diagnosis"].map({
        "B": 0,
        "M": 1,
    })

    return df


def load_heart():
    """
    Load the UCI Heart Disease dataset.

    Labels:
        0 = No disease
        1 = Disease
    """

    if not os.path.exists(HEART_PATH):
        raise FileNotFoundError(
            f"Heart Disease dataset not found at: {HEART_PATH}"
        )

    df = pd.read_csv(HEART_PATH)

    if "target" not in df.columns:
        raise ValueError(
            "Heart Disease dataset must contain a 'target' column."
        )

    return df


def load_dataset(dataset):
    """
    Load one of the supported datasets.

    Parameters
    ----------
    dataset : str
        "wdbc" or "heart"

    Returns
    -------
    pandas.DataFrame
        Cleaned dataset.
    """

    dataset = dataset.lower()

    if dataset == "wdbc":
        return load_wdbc()

    if dataset == "heart":
        return load_heart()

    raise ValueError(
        f"Unsupported dataset '{dataset}'. "
        "Choose 'wdbc' or 'heart'."
    )