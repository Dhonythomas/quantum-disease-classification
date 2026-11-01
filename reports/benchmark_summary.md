# Quantum vs Classical Disease Classification Benchmark

**Dataset:** Breast Cancer Wisconsin (Diagnostic)
**Features (Qubits):** 4 (Standardized + PCA, scaled to $[0, \pi]$)
**Total Samples:** 105 (Train: 80, Test: 25)

## Performance Summary Table

Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Train Time (s) | Infer Time (s)
--- | --- | --- | --- | --- | --- | --- | ---
QSVM | 0.8800 | 0.8750 | 0.9333 | 0.9032 | 0.9533 | 0.150 | 0.044
VQC | 0.8800 | 0.8333 | 1.0000 | 0.9091 | 1.0000 | 20.707 | 0.095
Classical_SVM | 0.9600 | 1.0000 | 0.9333 | 0.9655 | 1.0000 | 0.003 | 0.000
Logistic_Regression | 0.9600 | 1.0000 | 0.9333 | 0.9655 | 1.0000 | 0.003 | 0.000
Random_Forest | 0.8800 | 1.0000 | 0.8000 | 0.8889 | 1.0000 | 0.069 | 0.004

## Architecture Overview
- **QSVM (Quantum Support Vector Machine):** Computes quantum state overlap $|\langle \phi(x_1) | \phi(x_2) \rangle|^2$ via a second-order Havlíček ZZFeatureMap.
- **VQC (Variational Quantum Classifier):** Parameterized single-qubit rotations ($R_Y, R_Z$) with circular CNOT entanglement ring, measured with Pauli-Z readout and optimized via Adam.
- **Classical Baselines:** Radial Basis Function SVM (RBF), Logistic Regression ($L_2$), and Random Forest (100 estimators).
