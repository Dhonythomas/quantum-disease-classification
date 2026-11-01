import pytest
import numpy as np
from src.quantum.feature_maps import QuantumFeatureEncoder

def test_feature_map_statevector_generation():
    encoder = QuantumFeatureEncoder(n_qubits=4, encoding_type="zz", reps=1)
    sample_x = np.array([0.5, 1.2, 2.1, 0.8])
    state = encoder.encode_to_statevector(sample_x)
    
    assert state.shape == (16,)  # 2^4 = 16 amplitudes
    # Statevector must be normalized: sum of squared magnitudes = 1.0
    norm = np.linalg.norm(state)
    assert np.isclose(norm, 1.0, atol=1e-5)

def test_kernel_value_identity():
    encoder = QuantumFeatureEncoder(n_qubits=4, encoding_type="angle")
    sample_x = np.array([0.5, 1.2, 2.1, 0.8])
    # Overlap of a state with itself is 1.0
    val = encoder.compute_overlap(sample_x, sample_x)
    assert np.isclose(val, 1.0, atol=1e-5)

def test_circuit_diagram_export():
    encoder = QuantumFeatureEncoder(n_qubits=2, encoding_type="zz", reps=1)
    sample_x = np.array([0.5, 1.2])
    diagram = encoder.get_circuit_diagram(sample_x)
    assert isinstance(diagram, str)
    assert len(diagram) > 0

def test_different_vectors_have_proper_overlap_bounds():
    encoder = QuantumFeatureEncoder(n_qubits=4, encoding_type="zz")
    x1 = np.array([0.1, 0.2, 0.3, 0.4])
    x2 = np.array([2.5, 2.8, 3.0, 1.9])
    overlap = encoder.compute_overlap(x1, x2)
    assert 0.0 <= overlap <= 1.000001
