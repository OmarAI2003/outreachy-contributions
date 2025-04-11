import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    # PrecisionRecallDisplay,          # visualizations
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,  # Performence Metreics
    recall_score,
    roc_auc_score,
)

# Set the path to the data directory.
data_path = Path(__file__).parent.parent / "data"
# Read all CSVs into a dictionary of DataFrames
dfs = {
    "comp_train": pd.read_csv(data_path / "ComPEmbed_train_features.csv").drop(
        columns=["key", "input"]
    ),
    "comp_valid": pd.read_csv(data_path / "CompEmbed_valid_features.csv").drop(
        columns=["key", "input"]
    ),
    "comp_test": pd.read_csv(data_path / "CompEmbed_test_features.csv").drop(
        columns=["key", "input"]
    ),
    "morgan_train": pd.read_csv(data_path / "morgan_train_features.csv").drop(
        columns=["key", "input"]
    ),
    "morgan_valid": pd.read_csv(data_path / "morgan_valid_features.csv").drop(
        columns=["key", "input"]
    ),
    "morgan_test": pd.read_csv(data_path / "morgan_test_features.csv").drop(
        columns=["key", "input"]
    ),
    "test_labels": pd.read_csv(data_path / "test_labels.csv"),
    "train_labels": pd.read_csv(data_path / "train_labels.csv"),
    "valid_labels": pd.read_csv(data_path / "valid_labels.csv"),
}


def plot_metric(
    clf, testX, testY, name, style="classic", show_mat=False, show_roc=False
):
    """
    Plots the confusion matrix and/or ROC curve for a classifier using a given matplotlib style.

    Parameters
    ----------
    clf : classifier
        The trained classifier object implementing `predict` and `predict_proba`.
    testX : array-like
        The input features for the test set.
    testY : array-like
        The true labels for the test set.
    name : str
        The name of the model (used in plot titles).
    style : str, optional
        The matplotlib style to use for plotting. Defaults to 'classic'.
    show_mat : bool, optional
        Whether to display the confusion matrix. Defaults to False.
    show_roc : bool, optional
        Whether to display the ROC curve. Defaults to False.

    Notes
    -----
    If the specified style is not recognized, a random style from a predefined list is chosen.
    Both plots are displayed using `matplotlib.pyplot.show()`.
    """
    styles = ["bmh", "classic", "fivethirtyeight", "ggplot"]

    # Use the given style or pick a random valid style
    plt.style.use(style if style in styles else random.choice(styles))

    # Plot confusion matrix if requested
    if show_mat:
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_estimator(clf, testX, testY, ax=ax)
        ax.set_title(f"Confusion Matrix For {name}")
        plt.grid(False)
        plt.tight_layout()
        plt.show()

    # Plot ROC curve if requested
    if show_roc:
        RocCurveDisplay.from_estimator(clf, testX, testY)
        plt.title(f"ROC Curve For {name}")
        plt.grid(False)
        plt.tight_layout()
        plt.show()


def print_scores(y_true, y_pred, y_proba, label=""):
    """
    Prints classification evaluation metrics including accuracy, precision, recall,
    F1 score, and ROC AUC for a binary classifier.

    Parameters
    ----------
    y_true : array-like
        The true labels for the test set.
    y_pred : array-like
        The predicted labels generated by the classifier.
    y_proba : array-like
        The predicted probabilities for the positive class.
    label : str, optional
        An optional label name used in the printed output. Defaults to an empty string.
    """
    label = label.capitalize()  # Capitalize the label for display formatting

    print("_________________________________________")
    print(f"{label} accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%")
    print(f"{label} Precision: {precision_score(y_true, y_pred) * 100:.2f}%")
    print(f"{label} Recall: {recall_score(y_true, y_pred) * 100:.2f}%")
    print(f"{label} F1 Score: {f1_score(y_true, y_pred) * 100:.2f}%")
    print(f"{label} ROC AUC: {roc_auc_score(y_true, y_proba) * 100:.2f}%")
