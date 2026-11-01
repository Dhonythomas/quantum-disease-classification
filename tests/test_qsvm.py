import pytest
import numpy as np
from src.models.qsvm import QuantumKernelClassifier

def test_qsvm_kernel_matrix_properties():
    qsvm = QuantumKernelClassifier(n_qubits=2, encoding_type="zz", C=1.0)
    X_toy = np.array([[0.1, 0.2], [1.0, 0.8], [2.1, 1.5]])
    
    K = qsvm.compute_kernel_matrix(X_toy, X_toy)
    # Check shape
    assert K.shape == (3, 3)
    # Check symmetry: K = K.T
    assert np.allclose(K, K.T, atol=1e-5)
    # Check diagonal elements are identically 1.0
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
    
    # Check probabilities
    probs = qsvm.predict_proba(X_train)
    assert probs.shape == (4, 2)
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-4)

def test_qsvm_predict_unseen():
    qsvm = QuantumKernelClassifier(n_qubits=2, encoding_type="angle", C=1.0)
    X_train = np.array([[0.1, 0.2], [2.9, 3.0]])
    y_train = np.array([0, 1])
    qsvm.fit(X_train, y_train)
    
    X_test = np.array([[0.15, 0.25], [2.85, 2.95]])
    preds = qsvm.predict(X_test)
    assert np.array_equal(preds, np.array([0, 1]))
