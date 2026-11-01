# Quantum vs Classical Disease Classification Benchmark

**Dataset:** Breast Cancer Wisconsin (Diagnostic)
**Features (Qubits):** 4 (Standardized + PCA, scaled to $[0, \pi]$)
**Total Samples:** 569 (Train: 455, Test: 114)

## Performance Summary Table

Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Train Time (s) | Infer Time (s)
--- | --- | --- | --- | --- | --- | --- | ---
QSVM | 0.9298 | 0.9571 | 0.9306 | 0.9437 | 0.9686 | 0.915 | 0.241
VQC | 0.9386 | 0.9333 | 0.9722 | 0.9524 | 0.9792 | 71.709 | 0.475
Classical_SVM | 0.9561 | 0.9589 | 0.9722 | 0.9655 | 0.9924 | 0.006 | 0.001
Logistic_Regression | 0.9561 | 0.9467 | 0.9861 | 0.9660 | 0.9931 | 0.003 | 0.000
Random_Forest | 0.9211 | 0.9437 | 0.9306 | 0.9371 | 0.9869 | 0.092 | 0.005

## Architecture Overview
- **QSVM (Quantum Support Vector Machine):** Computes quantum state overlap $|\langle \phi(x_1) | \phi(x_2) \rangle|^2$ via a second-order Havlíček ZZFeatureMap.
- **VQC (Variational Quantum Classifier):** Parameterized single-qubit rotations ($R_Y, R_Z$) with circular CNOT entanglement ring, measured with Pauli-Z readout and optimized via Adam.
- **Classical Baselines:** Radial Basis Function SVM (RBF), Logistic Regression ($L_2$), and Random Forest (100 estimators).
