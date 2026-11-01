"""Quantum Machine Learning for Disease Classification Pipeline Orchestrator.

Executes an end-to-end hybrid quantum-classical machine learning pipeline on the
Breast Cancer Wisconsin dataset, comparing Quantum Kernel SVM (QSVM) and
Variational Quantum Classifiers (VQC) against Classical SVM, Logistic Regression,
and Random Forest.
"""

import argparse
import json
import os
from typing import Any, Dict
import numpy as np

from src.data.preprocessor import DataPipeline
from src.evaluation.benchmark import BenchmarkRunner
from src.models.classical import get_classical_baselines
from src.models.qsvm import QuantumKernelClassifier
from src.models.vqc import VariationalQuantumClassifier
from src.quantum.feature_maps import QuantumFeatureEncoder


def run_full_pipeline(
    n_components: int = 4,
    vqc_epochs: int = 15,
    report_dir: str = "reports",
    fast_mode: bool = False,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Runs the full disease classification workflow and outputs benchmark artifacts.
    
    Args:
        n_components: Number of PCA dimensions mapped to qubits (default 4).
        vqc_epochs: Number of training epochs for the Variational Quantum Classifier.
        report_dir: Output folder for benchmark reports and figures.
        fast_mode: If True, uses stratified subsample for rapid validation.
        random_state: Seed for deterministic reproducibility.
        
    Returns:
        Dictionary mapping model names to evaluation metric dictionaries.
    """
    os.makedirs(report_dir, exist_ok=True)
    print("=" * 70)
    print("   QUANTUM MACHINE LEARNING FOR DISEASE CLASSIFICATION")
    print("   Dataset: Breast Cancer Wisconsin (Diagnostic)")
    print(f"   Qubits (PCA features): {n_components} | Random Seed: {random_state}")
    print("=" * 70)

    # 1. Dataset Loading and Preprocessing
    print("\n[Phase 1] Preprocessing dataset...")
    pipeline = DataPipeline(
        n_components=n_components,
        test_size=0.2,
        random_state=random_state,
    )
    data = pipeline.fit_transform()
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    explained_var = pipeline.explained_variance_ratio_
    var_sum = float(np.sum(explained_var)) if explained_var is not None else 0.0
    print(f"  - Total clinical samples: 569 (Train: {len(X_train)}, Test: {len(X_test)})")
    print(f"  - PCA reduction to {n_components} features preserves {var_sum * 100:.2f}% of total variance.")

    if fast_mode:
        print("  - [Fast Mode] Using subsampled stratified slice for swift test verification.")
        idx_train = np.hstack([
            np.where(y_train == 0)[0][:30],
            np.where(y_train == 1)[0][:50],
        ])
        idx_test = np.hstack([
            np.where(y_test == 0)[0][:10],
            np.where(y_test == 1)[0][:15],
        ])
        X_train, y_train = X_train[idx_train], y_train[idx_train]
        X_test, y_test = X_test[idx_test], y_test[idx_test]

    # 2. Quantum Circuit Architecture Visualization
    print("\n[Phase 2] Designing quantum feature maps and parameterized ansatz...")
    encoder = QuantumFeatureEncoder(n_qubits=n_components, encoding_type="zz", reps=1)
    circuit_str = encoder.get_circuit_diagram()
    circuit_path = os.path.join(report_dir, "quantum_circuit_zz.txt")
    with open(circuit_path, "w", encoding="utf-8") as f:
        f.write("=== Havlicek ZZFeatureMap Circuit Diagram ===\n")
        f.write(circuit_str)
    print(f"  - Saved quantum circuit diagram to: {circuit_path}")

    # 3. Model Assembly
    print("\n[Phase 3] Assembling hybrid quantum and classical models...")
    models: Dict[str, Any] = {
        "QSVM": QuantumKernelClassifier(
            n_qubits=n_components,
            encoding_type="zz",
            reps=1,
            C=1.0,
            probability=True,
            random_state=random_state,
        ),
        "VQC": VariationalQuantumClassifier(
            n_qubits=n_components,
            n_layers=2,
            epochs=vqc_epochs,
            lr=0.05,
            random_state=random_state,
        ),
    }
    classical_baselines = get_classical_baselines(random_state=random_state)
    models.update(classical_baselines)

    for m_name in models.keys():
        print(f"  - Registered: {m_name}")

    # 4. Benchmarking and Metric Evaluation
    print("\n[Phase 4] Training and benchmarking all models...")
    target_names = pipeline.get_target_names()
    # Scikit-learn target names: ['malignant', 'benign']
    class_labels = [name.capitalize() for name in target_names]

    runner = BenchmarkRunner(
        models=models,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        class_names=class_labels,
    )
    results = runner.run()

    # 5. Output Generation and Reporting
    print("\n[Phase 5] Generating visualization figures and comparative reports...")
    roc_path = os.path.join(report_dir, "roc_comparison.png")
    cm_path = os.path.join(report_dir, "confusion_matrices.png")
    runner.plot_roc_curves(roc_path)
    runner.plot_confusion_matrices(cm_path)
    print(f"  - Saved ROC curves to: {roc_path}")
    print(f"  - Saved Confusion matrices to: {cm_path}")

    # Export JSON metrics
    json_path = os.path.join(report_dir, "benchmark_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"  - Saved metrics JSON to: {json_path}")

    # Export Markdown summary
    md_table = runner.generate_summary_table()
    summary_md = f"""# Quantum vs Classical Disease Classification Benchmark

**Dataset:** Breast Cancer Wisconsin (Diagnostic)
**Features (Qubits):** {n_components} (Standardized + PCA, scaled to $[0, \\pi]$)
**Total Samples:** {len(X_train) + len(X_test)} (Train: {len(X_train)}, Test: {len(X_test)})

## Performance Summary Table

{md_table}

## Architecture Overview
- **QSVM (Quantum Support Vector Machine):** Computes quantum state overlap $|\\langle \\phi(x_1) | \\phi(x_2) \\rangle|^2$ via a second-order Havlíček ZZFeatureMap.
- **VQC (Variational Quantum Classifier):** Parameterized single-qubit rotations ($R_Y, R_Z$) with circular CNOT entanglement ring, measured with Pauli-Z readout and optimized via Adam.
- **Classical Baselines:** Radial Basis Function SVM (RBF), Logistic Regression ($L_2$), and Random Forest (100 estimators).
"""
    md_path = os.path.join(report_dir, "benchmark_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(summary_md)
    print(f"  - Saved summary report to: {md_path}")

    print("\n" + "=" * 70)
    print("                    BENCHMARK RESULTS")
    print("=" * 70)
    print(md_table)
    print("=" * 70)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantum Machine Learning for Disease Classification"
    )
    parser.add_argument(
        "--n-qubits",
        type=int,
        default=4,
        help="Number of qubits / PCA reduced features (default: 4)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Training epochs for the Variational Quantum Classifier (default: 15)",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="reports",
        help="Output directory for generated reports and charts (default: reports)",
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Run fast mode on stratified subset for quick demonstration",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()
    run_full_pipeline(
        n_components=args.n_qubits,
        vqc_epochs=args.epochs,
        report_dir=args.report_dir,
        fast_mode=args.fast_mode,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
