"""Quantum feature encoding and circuit generation for disease classification."""

from typing import Optional
import numpy as np
import pennylane as qml


class QuantumFeatureEncoder:
    """Encodes classical medical feature vectors into quantum statevectors.
    
    Supports:
    - 'angle': Simple single-qubit rotations (RY gates).
    - 'zz': Havlicek et al. (2019) second-order Pauli-Z expansion with
      entangling CNOT gates and pairwise phase rotations.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        encoding_type: str = "zz",
        reps: int = 1,
    ) -> None:
        self.n_qubits = n_qubits
        self.encoding_type = encoding_type.lower()
        self.reps = reps

        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self._qnode = qml.QNode(self._circuit, self.dev)

    def _circuit(self, x: np.ndarray):
        """Constructs the quantum feature mapping circuit."""
        if self.encoding_type == "angle":
            for i in range(self.n_qubits):
                qml.RY(x[i], wires=i)
        elif self.encoding_type == "zz":
            for _ in range(self.reps):
                # Layer of Hadamards
                for i in range(self.n_qubits):
                    qml.Hadamard(wires=i)
                # Single-qubit phase rotations
                for i in range(self.n_qubits):
                    qml.RZ(2.0 * x[i], wires=i)
                # Entangling two-qubit ZZ interactions
                for i in range(self.n_qubits):
                    for j in range(i + 1, self.n_qubits):
                        qml.CNOT(wires=[i, j])
                        # Havlicek non-linear kernel phase: 2 * (pi - x_i) * (pi - x_j)
                        phase = 2.0 * (np.pi - x[i]) * (np.pi - x[j])
                        qml.RZ(phase, wires=j)
                        qml.CNOT(wires=[i, j])
        else:
            raise ValueError(f"Unsupported encoding_type: {self.encoding_type}. Choose 'angle' or 'zz'.")

        return qml.state()

    def encode_to_statevector(self, feature_vector: np.ndarray) -> np.ndarray:
        """Simulates the feature map circuit and returns the full quantum statevector.
        
        Args:
            feature_vector: 1D array of shape (n_qubits,) with values in [0, pi].
            
        Returns:
            Complex statevector of shape (2^n_qubits,).
        """
        if len(feature_vector) != self.n_qubits:
            raise ValueError(
                f"Feature vector length ({len(feature_vector)}) must match n_qubits ({self.n_qubits})."
            )
        state = np.array(self._qnode(feature_vector), dtype=complex)
        return state

    def compute_overlap(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Computes the transition fidelity (quantum kernel value): |<phi(x1)|phi(x2)>|^2."""
        state1 = self.encode_to_statevector(x1)
        state2 = self.encode_to_statevector(x2)
        inner_product = np.vdot(state1, state2)
        fidelity = float(np.abs(inner_product) ** 2)
        return np.clip(fidelity, 0.0, 1.0)

    def get_circuit_diagram(self, sample_x: Optional[np.ndarray] = None) -> str:
        """Returns ASCII circuit diagram for the specified feature map."""
        if sample_x is None:
            sample_x = np.full(self.n_qubits, np.pi / 2.0)
        draw_fn = qml.draw(self._qnode)
        return str(draw_fn(sample_x))
