"""Variational Quantum Classifier (VQC) with parameterized quantum circuits."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.base import BaseEstimator, ClassifierMixin


class VariationalQuantumClassifier(BaseEstimator, ClassifierMixin):
    """Variational Quantum Classifier (VQC) using a parameterized ansatz and Pauli-Z readout.
    
    Architecture:
    1. Feature Encoding: Angle encoding using RY gates.
    2. Variational Layers: Parameterized single-qubit rotations (RY, RZ) coupled with
       entangling CNOT circular ring topology.
    3. Measurement: Pauli-Z expectation value mapped through a logistic sigmoid.
    4. Optimization: Classical gradient-based optimization using Adam optimizer on BCE loss.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        random_state: int = 42,
    ) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.random_state = random_state

        np.random.seed(self.random_state)
        # Initialize variational weights: (n_layers, n_qubits, 2)
        initial_weights = np.random.normal(
            loc=0.0,
            scale=0.1,
            size=(self.n_layers, self.n_qubits, 2),
        )
        self.weights_ = pnp.array(initial_weights, requires_grad=True)
        self.bias_ = pnp.array(0.0, requires_grad=True)
        self.classes_ = np.array([0, 1])

        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self._qnode = self._build_qnode()

    def _build_qnode(self):
        n_qubits = self.n_qubits

        @qml.qnode(self.dev)
        def circuit(weights, x):
            # Feature encoding layer
            for i in range(n_qubits):
                qml.RY(x[i], wires=i)

            # Parameterized variational layers
            for l in range(weights.shape[0]):
                for i in range(n_qubits):
                    qml.RY(weights[l, i, 0], wires=i)
                    qml.RZ(weights[l, i, 1], wires=i)
                # Entanglement layer (circular CNOT chain)
                if n_qubits > 1:
                    for i in range(n_qubits):
                        qml.CNOT(wires=[i, (i + 1) % n_qubits])

            return qml.expval(qml.PauliZ(0))

        return circuit

    def _forward_sample(self, weights, bias, x):
        """Computes probability P(y=1) for a single input vector x."""
        expectation = self._qnode(weights, x)
        # Scale expectation and add bias, then map via sigmoid
        z = 2.0 * expectation + bias
        prob = 1.0 / (1.0 + pnp.exp(-z))
        return pnp.clip(prob, 1e-7, 1.0 - 1e-7)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 20,
        lr: float = 0.05,
        batch_size: int = 16,
    ) -> Dict[str, List[float]]:
        """Trains the variational quantum parameters using mini-batch Adam optimization.
        
        Args:
            X: Training feature matrix (n_samples, n_qubits).
            y: Binary target labels (n_samples,).
            epochs: Number of full passes over the training data.
            lr: Learning rate for the Adam optimizer.
            batch_size: Mini-batch size.
            
        Returns:
            Dictionary with training history ('loss' and 'accuracy').
        """
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=int)
        n_samples = len(X_arr)
        effective_batch_size = min(batch_size, n_samples)

        optimizer = qml.AdamOptimizer(stepsize=lr)
        history: Dict[str, List[float]] = {"loss": [], "accuracy": []}

        weights = self.weights_
        bias = self.bias_

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            epoch_losses = []

            for start_idx in range(0, n_samples, effective_batch_size):
                batch_indices = indices[start_idx : start_idx + effective_batch_size]
                X_batch = pnp.array(X_arr[batch_indices], requires_grad=False)
                y_batch = pnp.array(y_arr[batch_indices], requires_grad=False)

                def cost_fn(w, b):
                    loss = 0.0
                    for xi, yi in zip(X_batch, y_batch):
                        p1 = self._forward_sample(w, b, xi)
                        loss = loss - (yi * pnp.log(p1) + (1.0 - yi) * pnp.log(1.0 - p1))
                    return loss / len(X_batch)

                (weights, bias), batch_loss = optimizer.step_and_cost(cost_fn, weights, bias)
                epoch_losses.append(float(batch_loss))

            self.weights_ = weights
            self.bias_ = bias
            mean_loss = float(np.mean(epoch_losses))
            history["loss"].append(mean_loss)

            # Compute current training accuracy
            preds = self.predict(X_arr)
            acc = float(np.mean(preds == y_arr))
            history["accuracy"].append(acc)

        return history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Computes class probabilities for a batch of samples."""
        X_arr = np.asarray(X, dtype=float)
        probs = np.empty((len(X_arr), 2), dtype=float)
        for idx, x in enumerate(X_arr):
            p1 = float(self._forward_sample(self.weights_, self.bias_, x))
            probs[idx, 0] = 1.0 - p1
            probs[idx, 1] = p1
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts binary class labels {0, 1}."""
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)
