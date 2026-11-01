"""Evaluation metrics computation for disease classification models."""

from typing import Dict, Any, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Computes comprehensive classification metrics.
    
    Args:
        y_true: True binary target labels.
        y_pred: Predicted binary labels.
        y_proba: Predicted probabilities for the positive class (or 2D array [P(0), P(1)]).
        
    Returns:
        Dictionary with accuracy, precision, recall, f1_score, roc_auc, and confusion_matrix.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))
    cm = confusion_matrix(y_true_arr, y_pred_arr).tolist()

    # Clinical specificity / sensitivity:
    # Class 0: Malignant, Class 1: Benign
    malignant_recall = float(recall_score(y_true_arr, y_pred_arr, pos_label=0, zero_division=0))
    benign_recall = float(recall_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0))

    roc_auc = 0.5
    if y_proba is not None:
        proba_arr = np.asarray(y_proba)
        if proba_arr.ndim == 2 and proba_arr.shape[1] >= 2:
            scores = proba_arr[:, 1]
        else:
            scores = proba_arr.ravel()
        try:
            roc_auc = float(roc_auc_score(y_true_arr, scores))
        except ValueError:
            # Handle edge case where only one class is present in y_true
            roc_auc = 0.5

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "malignant_sensitivity": malignant_recall,
        "benign_sensitivity": benign_recall,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
    }
