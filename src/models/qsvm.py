"""Quantum Support Vector Classifier using Quantum Kernel Methods."""

from typing import Optional
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from src.quantum.feature_maps import QuantumFeatureEncoder


class QuantumKernelClassifier(BaseEstimator, ClassifierMixin):
    """Quantum Support Vector Machine (QSVM) using quantum state fidelity kernels.
    
    The quantum kernel computes the transition amplitude between quantum states:
        K(x_i, x_j) = |<phi(x_i) | phi(x_j)>|^2
    
    Optimization:
    Evaluates quantum statevectors once per sample (O(N)), then calculates the full
    Gram matrix via vectorized Hermitian matrix multiplication in O(N^2) BLAS operations.
    Probability calibration is performed via standard Platt scaling (logistic calibration).
    """

    def __init__(
        self,
        n_qubits: int = 4,
        encoding_type: str = "zz",
        reps: int = 1,
        C: float = 1.0,
        probability: bool = True,
        random_state: int = 42,
    ) -> None:
        self.n_qubits = n_qubits
        self.encoding_type = encoding_type
        self.reps = reps
        self.C = C
        self.probability = probability
        self.random_state = random_state

        self.encoder = QuantumFeatureEncoder(
            n_qubits=self.n_qubits,
            encoding_type=self.encoding_type,
            reps=self.reps,
        )
        self.svc_ = SVC(
            kernel="precomputed",
            C=self.C,
            random_state=self.random_state,
        )
        self.calibrator_: Optional[LogisticRegression] = None

        self.X_train_: Optional[np.ndarray] = None
        self.train_statevectors_: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None

    def _compute_statevectors(self, X: np.ndarray) -> np.ndarray:
        """Computes statevectors for a batch of input samples."""
        num_samples = len(X)
        dim = 2 ** self.n_qubits
        statevectors = np.empty((num_samples, dim), dtype=complex)
        for i in range(num_samples):
            statevectors[i] = self.encoder.encode_to_statevector(X[i])
        return statevectors

    def compute_kernel_matrix(
        self,
        X1: np.ndarray,
        X2: Optional[np.ndarray] = None,
        statevectors1: Optional[np.ndarray] = None,
        statevectors2: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Computes the quantum kernel Gram matrix between X1 and X2.
        
        Args:
            X1: Array of shape (n_samples_1, n_qubits).
            X2: Array of shape (n_samples_2, n_qubits). If None, computes self-kernel on X1.
            statevectors1: Optional cached statevectors for X1.
            statevectors2: Optional cached statevectors for X2.
            
        Returns:
            Kernel matrix of shape (n_samples_1, n_samples_2).
        """
        is_self_kernel = X2 is None or (X1 is X2)

        psi1 = statevectors1 if statevectors1 is not None else self._compute_statevectors(X1)
        if is_self_kernel:
            psi2 = psi1
        else:
            psi2 = statevectors2 if statevectors2 is not None else self._compute_statevectors(X2)

        # Transition amplitude inner products: psi1 @ psi2.conj().T
        inner_products = np.matmul(psi1, psi2.conj().T)
        K = np.abs(inner_products) ** 2

        if is_self_kernel:
            # Enforce exact symmetry and diagonal identity for numerical precision
            K = 0.5 * (K + K.T)
            np.fill_diagonal(K, 1.0)

        return np.clip(K, 0.0, 1.0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantumKernelClassifier":
        """Fits the Quantum Support Vector Classifier.
        
        Args:
            X: Training features of shape (n_samples, n_qubits).
            y: Target binary labels of shape (n_samples,).
        """
        self.X_train_ = np.asarray(X, dtype=float)
        self.train_statevectors_ = self._compute_statevectors(self.X_train_)
        self.classes_ = np.unique(y)

        K_train = self.compute_kernel_matrix(
            self.X_train_,
            self.X_train_,
            statevectors1=self.train_statevectors_,
            statevectors2=self.train_statevectors_,
        )
        self.svc_.fit(K_train, y)

        if self.probability:
            scores = self.svc_.decision_function(K_train).reshape(-1, 1)
            self.calibrator_ = LogisticRegression(random_state=self.random_state)
            self.calibrator_.fit(scores, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class labels for samples in X."""
        if self.X_train_ is None or self.train_statevectors_ is None:
            raise ValueError("The model is not fitted yet. Call fit() first.")
        X = np.asarray(X, dtype=float)
        test_statevectors = self._compute_statevectors(X)
        K_test = self.compute_kernel_matrix(
            X,
            self.X_train_,
            statevectors1=test_statevectors,
            statevectors2=self.train_statevectors_,
        )
        return self.svc_.predict(K_test)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts class probabilities for samples in X."""
        if self.X_train_ is None or self.train_statevectors_ is None:
            raise ValueError("The model is not fitted yet. Call fit() first.")
        X = np.asarray(X, dtype=float)
        test_statevectors = self._compute_statevectors(X)
        K_test = self.compute_kernel_matrix(
            X,
            self.X_train_,
            statevectors1=test_statevectors,
            statevectors2=self.train_statevectors_,
        )
        scores = self.svc_.decision_function(K_test).reshape(-1, 1)
        if self.calibrator_ is not None:
            return self.calibrator_.predict_proba(scores)
        else:
            # Default Platt logistic sigmoid mapping
            p1 = 1.0 / (1.0 + np.exp(-scores.ravel()))
            return np.column_stack([1.0 - p1, p1])

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Evaluates decision function values for samples in X."""
        if self.X_train_ is None or self.train_statevectors_ is None:
            raise ValueError("The model is not fitted yet. Call fit() first.")
        X = np.asarray(X, dtype=float)
        test_statevectors = self._compute_statevectors(X)
        K_test = self.compute_kernel_matrix(
            X,
            self.X_train_,
            statevectors1=test_statevectors,
            statevectors2=self.train_statevectors_,
        )
        return self.svc_.decision_function(K_test)
