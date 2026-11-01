import pytest
import numpy as np
from src.models.vqc import VariationalQuantumClassifier

def test_vqc_forward_pass_and_probabilities():
    vqc = VariationalQuantumClassifier(n_qubits=2, n_layers=2, random_state=42)
    X_toy = np.array([[0.2, 0.3], [2.8, 2.9]])
    
    # Forward pass outputs valid probabilities in [0, 1] that sum to 1.0
    probs = vqc.predict_proba(X_toy)
    assert probs.shape == (2, 2)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)

def test_vqc_training_decreases_loss():
    vqc = VariationalQuantumClassifier(n_qubits=2, n_layers=1, random_state=42)
    X_toy = np.array([[0.1, 0.2], [3.0, 3.1]])
    y_toy = np.array([0, 1])
    
    history = vqc.fit(X_toy, y_toy, epochs=15, lr=0.2, batch_size=2)
    assert "loss" in history
    assert len(history["loss"]) == 15
    # Loss should decrease as parameters optimize
    assert history["loss"][-1] < history["loss"][0]

def test_vqc_predictions_shape_and_values():
    vqc = VariationalQuantumClassifier(n_qubits=2, n_layers=1, random_state=42)
    X_toy = np.array([[0.2, 0.3], [2.8, 2.9]])
    preds = vqc.predict(X_toy)
    assert preds.shape == (2,)
    assert set(preds).issubset({0, 1})
