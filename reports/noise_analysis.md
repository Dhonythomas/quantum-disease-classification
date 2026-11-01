# NISQ Hardware Noise & Depolarizing Channel Study

## 1. Experimental Overview
To simulate real-world physical quantum processors (such as IBM Quantum transmon qubits), we evaluate the **Quantum Support Vector Machine (QSVM)** under varying levels of physical depolarizing noise:
$$\mathcal{E}(\rho) = (1 - p)\rho + \frac{p}{2^n} I$$
where $p$ represents the per-gate error probability and $n = 4$ qubits.

## 2. Experimental Results

| Depolarizing Rate (p) | Test Accuracy | ROC-AUC |
| :---: | :---: | :---: |
| 0.000 | 0.8750 | 0.9760 |
| 0.010 | 0.8750 | 0.9733 |
| 0.020 | 0.8750 | 0.9733 |
| 0.040 | 0.7250 | 0.9520 |
| 0.060 | 0.6250 | 0.9360 |
| 0.080 | 0.6250 | 0.8800 |
| 0.100 | 0.6250 | 0.8133 |

## 3. Findings & Mitigation Strategies
1. **Exponential Kernel Concentration:** As the depolarizing rate $p$ increases, quantum states decay toward the maximally mixed state $\rho \to \frac{I}{2^n}$. Consequently, off-diagonal kernel elements contract toward $\frac{1}{2^n}$, reducing separability.
2. **Resilience Threshold:** The quantum kernel retains competitive classification accuracy (>80%) up to an error rate of $p \approx 0.03$, which aligns with the error threshold of current commercial superconducting NISQ processors.
3. **Error Mitigation (ZNE):** On real hardware, **Zero-Noise Extrapolation (ZNE)** can be applied by artificially scaling the noise (via unitary folding $U \to U U^\dagger U$) and extrapolating back to the $p=0$ zero-noise limit via Richardson extrapolation.
