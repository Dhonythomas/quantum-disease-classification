import pytest
import os
import json
from main import run_full_pipeline

def test_full_pipeline_execution(tmp_path):
    report_dir = tmp_path / "reports"
    results = run_full_pipeline(
        n_components=4,
        vqc_epochs=2,
        report_dir=str(report_dir),
        fast_mode=True,
        random_state=42,
    )
    
    # Check all 5 models are present in results
    expected_models = [
        "QSVM",
        "VQC",
        "Classical_SVM",
        "Logistic_Regression",
        "Random_Forest",
    ]
    for model_name in expected_models:
        assert model_name in results
        assert "accuracy" in results[model_name]
        assert "roc_auc" in results[model_name]
        assert "f1_score" in results[model_name]
        assert results[model_name]["accuracy"] >= 0.5  # Non-trivial sanity check
    
    # Verify report files were created
    assert os.path.exists(report_dir / "benchmark_summary.json")
    assert os.path.exists(report_dir / "benchmark_summary.md")
    assert os.path.exists(report_dir / "roc_comparison.png")
    assert os.path.exists(report_dir / "confusion_matrices.png")
    assert os.path.exists(report_dir / "quantum_circuit_zz.txt")
    
    # Verify JSON content is valid
    with open(report_dir / "benchmark_summary.json", "r") as f:
        data = json.load(f)
        assert "QSVM" in data
