# Quantum Machine Learning for Disease Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an end-to-end hybrid quantum-classical machine learning project that classifies patients using the Breast Cancer Wisconsin dataset, exploring quantum feature encoding, Quantum Support Vector Machine (QSVM with Quantum Kernels), Variational Quantum Classifiers (VQC), and benchmarking them against classical baselines with complete evaluation metrics.

**Architecture:** A modular Python pipeline consisting of:
1. Data preprocessing and dimensionality reduction (PCA to 4–6 qubits, scaled to $[0, \pi]$).
2. Quantum feature encoding and parameterized ansatz circuits (ZZFeatureMap, Angle Encoding, RealAmplitudes/TwoLocal).
3. Quantum model implementations (Quantum Kernel QSVM and PennyLane/Qiskit-based VQC).
4. Classical baseline classifiers (RBF SVM, Logistic Regression, Random Forest).
5. Standardized evaluation, metric generation (ROC-AUC, F1, Accuracy, Precision, Recall), confusion matrix plotting, and comparative benchmarking.

**Tech Stack:** Python 3.13, Qiskit 2.x, PennyLane, Scikit-Learn, NumPy, Pandas, Matplotlib, Seaborn, Pytest.

**Spec:** Disease Classification using Hybrid Quantum-Classical ML on the Breast Cancer Wisconsin dataset, adhering to ethical standards (public de-identified data), demonstrating feature encoding, quantum circuit design, pipeline integration, evaluation metrics, and classical comparison.

## Global Constraints

- Must run smoothly and efficiently in `qiskit-env` on Windows Python 3.13 without native compilation failures.
- Dataset must be public and de-identified: `sklearn.datasets.load_breast_cancer`.
- Dimensionality reduction must be parameterized (default 4 features/qubits) to prevent exponential simulation slowdown.
- Models must produce standardized prediction outputs (`predict` returning binary array `{0, 1}`, `predict_proba` or decision scores for ROC-AUC).
- All code must have automated unit tests run via `pytest`.

## Review Focus

1. **Class Imbalance & Stratification:** Breast Cancer Wisconsin has 212 Malignant and 357 Benign samples (~37% / 63%); train/test splits must be stratified to prevent skewed evaluation.
2. **Kernel Matrix Properties:** The quantum kernel matrix $K_{ij} = |\langle\phi(x_i)|\phi(x_j)\rangle|^2$ must be symmetric, positive semi-definite with diagonal elements identically equal to 1.0.
3. **Feature Scaling for Angle Encoding:** Raw features have vastly different ranges (e.g. area ~2500 vs smoothness ~0.1); features must be scaled to $[0, \pi]$ before feeding into trigonometric quantum rotations ($R_X, R_Y, R_Z$) to avoid phase periodicity aliasing.
4. **VQC Convergence & Numerical Stability:** Variational circuit optimization must handle gradients stably, bounding probabilities to $[10^{-7}, 1 - 10^{-7}]$ to avoid $\log(0)$ in binary cross-entropy.
5. **Reproducibility:** All train-test splits, PCA reductions, and parameter initializations must accept explicit random seed control.

---

### Task 1: Environment Setup and Dataset Preprocessing Module

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `src/data/__init__.py`
- Create: `src/data/preprocessor.py`
- Test: `tests/test_preprocessor.py`

**Interfaces:**
- Produces: `DataPipeline` class with methods:
  - `load_data() -> Tuple[np.ndarray, np.ndarray, List[str]]`
  - `preprocess(n_components: int = 4, test_size: float = 0.2, random_state: int = 42) -> Dict[str, np.ndarray]`

- [ ] **Step 1: Write the failing test for data loading and preprocessing**

```python
# tests/test_preprocessor.py
import pytest
import numpy as np
from src.data.preprocessor import DataPipeline

def test_data_pipeline_shapes_and_scaling():
    pipeline = DataPipeline(n_components=4, test_size=0.2, random_state=42)
    data = pipeline.fit_transform()
    
    assert "X_train" in data and "X_test" in data
    assert "y_train" in data and "y_test" in data
    assert data["X_train"].shape[1] == 4
    assert data["X_test"].shape[1] == 4
    assert len(data["X_train"]) + len(data["X_test"]) == 569
    
    # Verify values are bounded in [0, pi] for angle encoding
    assert np.all(data["X_train"] >= -1e-6)
    assert np.all(data["X_train"] <= np.pi + 1e-6)
    
    # Verify stratification
    train_prop = np.mean(data["y_train"])
    test_prop = np.mean(data["y_test"])
    assert abs(train_prop - test_prop) < 0.05
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\qiskit-env\Scripts\pytest tests/test_preprocessor.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'src')

- [ ] **Step 3: Install dependencies and implement DataPipeline in `src/data/preprocessor.py`**

Install `requirements.txt` (`scikit-learn`, `numpy`, `pandas`, `pytest`, `matplotlib`, `seaborn`, `qiskit`, `pennylane`).
Implement `DataPipeline` using `load_breast_cancer`, `StandardScaler`, `PCA(n_components=n_components)`, and `MinMaxScaler(feature_range=(0, np.pi))` with stratified train-test split.

- [ ] **Step 4: Run test to verify it passes**

Run: `.\qiskit-env\Scripts\pytest tests/test_preprocessor.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add requirements.txt src/ tests/
git commit -m "feat: implement dataset loading, PCA dimensionality reduction, and scaling pipeline"
```

---

### Task 2: Quantum Feature Maps and Circuit Design

**Files:**
- Create: `src/quantum/__init__.py`
- Create: `src/quantum/feature_maps.py`
- Test: `tests/test_feature_maps.py`

**Interfaces:**
- Consumes: Preprocessed feature vectors $x \in [0, \pi]^n$
- Produces: 
  - `create_angle_feature_map(n_qubits: int, rotation: str = 'Y')`
  - `create_zz_feature_map(n_qubits: int, reps: int = 2)`
  - `compute_quantum_state(feature_vector: np.ndarray, feature_map_type: str = 'zz') -> np.ndarray`

- [ ] **Step 1: Write the failing test for quantum feature maps**

```python
# tests/test_feature_maps.py
import pytest
import numpy as np
from src.quantum.feature_maps import QuantumFeatureEncoder

def test_feature_map_statevector_generation():
    encoder = QuantumFeatureEncoder(n_qubits=4, encoding_type="zz", reps=1)
    sample_x = np.array([0.5, 1.2, 2.1, 0.8])
    state = encoder.encode_to_statevector(sample_x)
    
    assert state.shape == (16,)  # 2^4
    # Statevector must be normalized
    norm = np.linalg.norm(state)
    assert np.isclose(norm, 1.0, atol=1e-5)

def test_kernel_value_identity():
    encoder = QuantumFeatureEncoder(n_qubits=4, encoding_type="angle")
    sample_x = np.array([0.5, 1.2, 2.1, 0.8])
    # Overlap of a state with itself is 1.0
    val = encoder.compute_overlap(sample_x, sample_x)
    assert np.isclose(val, 1.0, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\qiskit-env\Scripts\pytest tests/test_feature_maps.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `QuantumFeatureEncoder` in `src/quantum/feature_maps.py`**

Implement angle encoding and ZZ feature map with $H$, $R_Z$, and entangling $CNOT - R_Z(\theta_i \theta_j) - CNOT$ circuits. Provide statevector generation and fidelity computation $|\langle \phi(x_1) | \phi(x_2) \rangle|^2$.

- [ ] **Step 4: Run test to verify it passes**

Run: `.\qiskit-env\Scripts\pytest tests/test_feature_maps.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/quantum/ tests/test_feature_maps.py
git commit -m "feat: add quantum feature encoding and fidelity kernel computation"
```

---

### Task 3: Quantum Support Vector Classifier (QSVM / Quantum Kernel)

**Files:**
- Create: `src/models/__init__.py`
- Create: `src/models/qsvm.py`
- Test: `tests/test_qsvm.py`

**Interfaces:**
- Consumes: `QuantumFeatureEncoder` from `src.quantum.feature_maps`
- Produces: `QuantumKernelClassifier` with scikit-learn compatible interface:
  - `fit(X: np.ndarray, y: np.ndarray) -> QuantumKernelClassifier`
  - `predict(X: np.ndarray) -> np.ndarray`
  - `predict_proba(X: np.ndarray) -> np.ndarray`
  - `compute_kernel_matrix(X1: np.ndarray, X2: np.ndarray) -> np.ndarray`

- [ ] **Step 1: Write the failing test for QSVM**

```python
# tests/test_qsvm.py
import pytest
import numpy as np
from src.models.qsvm import QuantumKernelClassifier

def test_qsvm_kernel_matrix_properties():
    qsvm = QuantumKernelClassifier(n_qubits=2, encoding_type="zz", C=1.0)
    X_toy = np.array([[0.1, 0.2], [1.0, 0.8], [2.1, 1.5]])
    y_toy = np.array([0, 1, 1])
    
    K = qsvm.compute_kernel_matrix(X_toy, X_toy)
    # Check shape
    assert K.shape == (3, 3)
    # Check symmetry
    assert np.allclose(K, K.T, atol=1e-5)
    # Check diagonal is 1.0
    assert np.allclose(np.diag(K), 1.0, atol=1e-5)
    # Check positive semi-definite (all eigenvalues >= -1e-6)
    eigvals = np.linalg.eigvalsh(K)
    assert np.all(eigvals >= -1e-6)

def test_qsvm_fit_predict():
    qsvm = QuantumKernelClassifier(n_qubits=2, encoding_type="angle", C=1.0)
    X_train = np.array([[0.1, 0.2], [0.2, 0.1], [2.8, 2.9], [3.0, 2.7]])
    y_train = np.array([0, 0, 1, 1])
    qsvm.fit(X_train, y_train)
    
    preds = qsvm.predict(X_train)
    assert len(preds) == 4
    assert np.array_equal(preds, y_train)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\qiskit-env\Scripts\pytest tests/test_qsvm.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `QuantumKernelClassifier` in `src/models/qsvm.py`**

Implement vectorized/cached Gram matrix computation using `QuantumFeatureEncoder`, plugged into `sklearn.svm.SVC(kernel='precomputed', probability=True)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.\qiskit-env\Scripts\pytest tests/test_qsvm.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/qsvm.py tests/test_qsvm.py
git commit -m "feat: implement Quantum Support Vector Machine with quantum kernel computation"
```

---

### Task 4: Variational Quantum Classifier (VQC) with Parameterized Quantum Circuit

**Files:**
- Create: `src/models/vqc.py`
- Test: `tests/test_vqc.py`

**Interfaces:**
- Consumes: Preprocessed features $x \in \mathbb{R}^n$
- Produces: `VariationalQuantumClassifier` with interface:
  - `fit(X: np.ndarray, y: np.ndarray, epochs: int = 30, lr: float = 0.05) -> Dict[str, List[float]]`
  - `predict(X: np.ndarray) -> np.ndarray`
  - `predict_proba(X: np.ndarray) -> np.ndarray`

- [ ] **Step 1: Write the failing test for VQC**

```python
# tests/test_vqc.py
import pytest
import numpy as np
from src.models.vqc import VariationalQuantumClassifier

def test_vqc_forward_pass_and_training():
    vqc = VariationalQuantumClassifier(n_qubits=2, n_layers=2, random_state=42)
    X_toy = np.array([[0.2, 0.3], [2.8, 2.9]])
    y_toy = np.array([0, 1])
    
    # Forward pass outputs valid probabilities in [0, 1]
    probs = vqc.predict_proba(X_toy)
    assert probs.shape == (2, 2)
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)
    
    # Train for a few steps and ensure loss decreases
    history = vqc.fit(X_toy, y_toy, epochs=10, lr=0.1)
    assert "loss" in history
    assert len(history["loss"]) == 10
    assert history["loss"][-1] <= history["loss"][0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\qiskit-env\Scripts\pytest tests/test_vqc.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `VariationalQuantumClassifier` in `src/models/vqc.py`**

Implement parameterized variational circuit with angle encoding and hardware-efficient entangling ansatz ($R_Y, R_Z$, CNOT ring). Compute Pauli-Z expectation value, map via sigmoid to binary probabilities, and optimize with gradient-based or Adam optimizer.

- [ ] **Step 4: Run test to verify it passes**

Run: `.\qiskit-env\Scripts\pytest tests/test_vqc.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/vqc.py tests/test_vqc.py
git commit -m "feat: implement Variational Quantum Classifier with parameterized ansatz"
```

---

### Task 5: Classical Baselines and Benchmarking Pipeline

**Files:**
- Create: `src/models/classical.py`
- Create: `src/evaluation/__init__.py`
- Create: `src/evaluation/metrics.py`
- Create: `src/evaluation/benchmark.py`
- Test: `tests/test_benchmark.py`

**Interfaces:**
- Consumes: Models implementing `fit`, `predict`, `predict_proba`
- Produces: 
  - `train_classical_baselines(X_train, y_train) -> Dict[str, Any]`
  - `evaluate_model(model, X_test, y_test) -> Dict[str, float]` (Accuracy, Precision, Recall, F1, ROC-AUC)
  - `BenchmarkRunner` to run all models (QSVM, VQC, Classical SVM, Logistic Regression, Random Forest), generate comparison table, and save ROC curves and Confusion Matrices to `reports/`.

- [ ] **Step 1: Write the failing test for benchmarking and metric calculation**

```python
# tests/test_benchmark.py
import pytest
import numpy as np
from src.evaluation.metrics import compute_classification_metrics
from src.models.classical import get_classical_baselines

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
    baselines = get_classical_baselines()
    assert "Classical_SVM" in baselines
    assert "Logistic_Regression" in baselines
    assert "Random_Forest" in baselines
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\qiskit-env\Scripts\pytest tests/test_benchmark.py -v`
Expected: FAIL

- [ ] **Step 3: Implement metrics, classical baselines, and `BenchmarkRunner`**

Implement `src/models/classical.py`, `src/evaluation/metrics.py`, and `src/evaluation/benchmark.py` with plotting for ROC curves and confusion matrices.

- [ ] **Step 4: Run test to verify it passes**

Run: `.\qiskit-env\Scripts\pytest tests/test_benchmark.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/classical.py src/evaluation/ tests/test_benchmark.py
git commit -m "feat: implement classical baselines, evaluation metrics, and benchmark runner"
```

---

### Task 6: CLI Runner, Pipeline Orchestrator, and Comprehensive Reporting

**Files:**
- Create: `main.py`
- Create: `reports/` directory
- Create: `README.md`
- Test: `tests/test_pipeline_e2e.py`

**Interfaces:**
- Unified CLI: `python main.py --n-qubits 4 --epochs 25 --save-plots`
- Generates:
  - `reports/benchmark_summary.md` and `reports/benchmark_summary.json`
  - `reports/roc_comparison.png`
  - `reports/confusion_matrices.png`
  - Complete project README with scientific explanation, architecture diagrams, and usage commands.

- [ ] **Step 1: Write the failing E2E test for CLI / Pipeline runner**

```python
# tests/test_pipeline_e2e.py
import pytest
import os
from main import run_full_pipeline

def test_full_pipeline_execution(tmp_path):
    report_dir = tmp_path / "reports"
    results = run_full_pipeline(n_components=4, vqc_epochs=5, report_dir=str(report_dir), fast_mode=True)
    
    assert "QSVM" in results
    assert "VQC" in results
    assert "Classical_SVM" in results
    assert os.path.exists(report_dir / "benchmark_summary.json")
    assert os.path.exists(report_dir / "roc_comparison.png")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\qiskit-env\Scripts\pytest tests/test_pipeline_e2e.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `main.py` and `README.md`**

Implement CLI argument parsing, execution orchestration, plot saving, formatted markdown summary generation, and a comprehensive `README.md`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.\qiskit-env\Scripts\pytest tests/test_pipeline_e2e.py -v`
Expected: PASS

- [ ] **Step 5: Run full test suite and verify end-to-end execution**

Run: `.\qiskit-env\Scripts\pytest tests/ -v`
Run: `.\qiskit-env\Scripts\python.exe main.py --n-qubits 4 --epochs 15`
Expected: All tests pass; benchmark report and comparison plots created.

- [ ] **Step 6: Commit**

```bash
git add main.py README.md tests/test_pipeline_e2e.py reports/
git commit -m "feat: complete end-to-end CLI orchestrator, benchmarking, and documentation"
```
