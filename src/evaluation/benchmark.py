"""Benchmarking runner and comparative visualization suite."""

import os
import time
from typing import Dict, List, Optional, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from .metrics import compute_classification_metrics


class BenchmarkRunner:
    """Orchestrates comparative benchmarking between quantum and classical classifiers."""

    def __init__(
        self,
        models: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        class_names: Optional[List[str]] = None,
    ) -> None:
        self.models = models
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.class_names = class_names or ["Class 0", "Class 1"]

        self.results: Dict[str, Dict[str, Any]] = {}
        self.predictions: Dict[str, Dict[str, np.ndarray]] = {}

    def run(self) -> Dict[str, Dict[str, Any]]:
        """Trains and evaluates each registered classifier, logging time and metrics."""
        for name, model in self.models.items():
            # Training phase
            start_train = time.perf_counter()
            model.fit(self.X_train, self.y_train)
            train_duration = time.perf_counter() - start_train

            # Inference phase
            start_infer = time.perf_counter()
            y_pred = model.predict(self.X_test)
            infer_duration = time.perf_counter() - start_infer

            y_proba = None
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(self.X_test)
            elif hasattr(model, "decision_function"):
                scores = model.decision_function(self.X_test)
                p1 = 1.0 / (1.0 + np.exp(-scores))
                y_proba = np.column_stack([1.0 - p1, p1])

            metrics = compute_classification_metrics(self.y_test, y_pred, y_proba)
            metrics["train_time_sec"] = round(train_duration, 4)
            metrics["infer_time_sec"] = round(infer_duration, 4)

            self.results[name] = metrics
            self.predictions[name] = {
                "y_pred": y_pred,
                "y_proba": y_proba,
            }

        return self.results

    def generate_summary_table(self) -> str:
        """Returns a Markdown formatted table comparing model performance."""
        headers = [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
            "ROC-AUC",
            "Train Time (s)",
            "Infer Time (s)",
        ]
        divider = ["---"] * len(headers)
        rows = [" | ".join(headers), " | ".join(divider)]

        for name, m in self.results.items():
            row = [
                name,
                f"{m['accuracy']:.4f}",
                f"{m['precision']:.4f}",
                f"{m['recall']:.4f}",
                f"{m['f1_score']:.4f}",
                f"{m['roc_auc']:.4f}",
                f"{m['train_time_sec']:.3f}",
                f"{m['infer_time_sec']:.3f}",
            ]
            rows.append(" | ".join(row))

        return "\n".join(rows)

    def plot_confusion_matrices(self, save_path: str) -> None:
        """Plots and saves confusion matrices for all evaluated models."""
        n_models = len(self.models)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
        axes_flat = axes.flatten()

        for idx, (name, m) in enumerate(self.results.items()):
            ax = axes_flat[idx]
            cm = np.array(m["confusion_matrix"])
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=self.class_names,
                yticklabels=self.class_names,
                ax=ax,
            )
            ax.set_title(f"{name}\nAcc: {m['accuracy']:.3f} | F1: {m['f1_score']:.3f}")
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")

        for idx in range(n_models, len(axes_flat)):
            fig.delaxes(axes_flat[idx])

        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close(fig)

    def plot_roc_curves(self, save_path: str) -> None:
        """Plots and saves comparative ROC curves with AUC annotations."""
        plt.figure(figsize=(8, 6))

        for name, preds in self.predictions.items():
            y_proba = preds["y_proba"]
            if y_proba is not None:
                if y_proba.ndim == 2 and y_proba.shape[1] >= 2:
                    scores = y_proba[:, 1]
                else:
                    scores = y_proba.ravel()
                fpr, tpr, _ = roc_curve(self.y_test, scores)
                roc_score = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_score:.3f})")

        # Chance line
        plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random Chance")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12)
        plt.title("Receiver Operating Characteristic (ROC) Comparison", fontsize=14)
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)

        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
