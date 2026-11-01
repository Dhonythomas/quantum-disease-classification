import pytest
import numpy as np
import os
from src.evaluation.metrics import compute_classification_metrics
from src.models.classical import get_classical_baselines
from src.evaluation.benchmark import BenchmarkRunner

def test_metrics_computation():
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 0, 0, 0])
    y_proba = np.array([0.1, 0.8, 0.9, 0.2, 0.4, 0.3])
    
    metrics = compute_classification_metrics(y_true, y_pred, y_proba)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "roc_auc" in metrics
    assert 0.0 <= metrics["roc_auc"] <= 1.0

def test_classical_baselines_fit():
    baselines = get_classical_baselines(random_state=42)
    assert "Classical_SVM" in baselines
    assert "Logistic_Regression" in baselines
    assert "Random_Forest" in baselines
    
    X_toy = np.array([[0.1, 0.2], [0.2, 0.1], [2.8, 2.9], [3.0, 2.7]])
    y_toy = np.array([0, 0, 1, 1])
    for name, model in baselines.items():
        model.fit(X_toy, y_toy)
        preds = model.predict(X_toy)
        assert len(preds) == 4

def test_benchmark_runner_execution(tmp_path):
    X_train = np.array([[0.1, 0.2], [0.2, 0.1], [2.8, 2.9], [3.0, 2.7]])
    y_train = np.array([0, 0, 1, 1])
    X_test = np.array([[0.15, 0.22], [2.95, 2.85]])
    y_test = np.array([0, 1])
    
    baselines = get_classical_baselines(random_state=42)
    runner = BenchmarkRunner(
        models=baselines,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        class_names=["Malignant", "Benign"],
    )
    results = runner.run()
    assert "Classical_SVM" in results
    assert "accuracy" in results["Classical_SVM"]
    
    cm_path = tmp_path / "cm.png"
    roc_path = tmp_path / "roc.png"
    runner.plot_confusion_matrices(str(cm_path))
    runner.plot_roc_curves(str(roc_path))
    
    assert os.path.exists(cm_path)
    assert os.path.exists(roc_path)
