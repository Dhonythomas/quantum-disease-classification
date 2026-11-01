# ⚛️ QuantumCare AI™ — Hybrid Quantum Machine Learning for Disease Classification

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-2.0+-613394.svg)](https://qiskit.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.45+-yellowgreen.svg)](https://pennylane.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42+-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9+-F7931E.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/Tests-18%20Passed%20(100%25)-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end hybrid quantum-classical machine learning platform designed for clinical breast cancer discrimination. Built on the **Breast Cancer Wisconsin (Diagnostic)** dataset (569 patients, 30 morphological features), this project incorporates entangled quantum feature maps (Havlíček `ZZFeatureMap`), fast vectorized Quantum Support Vector Machines (QSVM), and Variational Quantum Classifiers (VQC) alongside classical machine learning baselines into a real-time clinical dashboard.

---

## 📑 Table of Contents
- [1. Executive Summary & Benchmark Highlights](#1-executive-summary--benchmark-highlights)
- [2. The Clinical & Computational Challenge](#2-the-clinical--computational-challenge)
- [3. Theoretical Foundations](#3-theoretical-foundations)
- [4. Dual Hybrid Quantum Architectures](#4-dual-hybrid-quantum-architectures)
- [5. Interactive Streamlit Web Dashboard](#5-interactive-streamlit-web-dashboard)
- [6. NISQ Hardware Noise Study](#6-nisq-hardware-noise-study)
- [7. Benchmark Results vs. Classical Baselines](#7-benchmark-results-vs-classical-baselines)
- [8. Project Directory Structure](#8-project-directory-structure)
- [9. Quick Start Guide](#9-quick-start-guide)
- [10. Testing & Verification](#10-testing--verification)
- [11. Pitch Deck & Presentation Guide](#11-pitch-deck--presentation-guide)
- [12. Ethics & Data Privacy](#12-ethics--data-privacy)

---

## 1. Executive Summary & Benchmark Highlights

| Metric | QSVM (Quantum Kernel) | VQC (Quantum NN) | Classical SVM (RBF) | Logistic Regression | Random Forest |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Test Accuracy** | **92.98%** | 90.35% | 95.61% | 95.61% | 92.11% |
| **Precision** | **95.71%** | 86.75% | 95.89% | 94.67% | 94.37% |
| **Benign Recall** | 93.06% | **100.00%** | 97.22% | 98.61% | 93.06% |
| **Malignancy Sensitivity** | **92.86%** | 73.81% | 92.86% | 90.48% | 90.48% |
| **ROC-AUC** | **0.9686** | 0.9775 | 0.9924 | 0.9931 | 0.9869 |
| **Training Time** | **2.22s** | 249.8s | 0.012s | 0.007s | 0.211s |
| **Verdict** | 🏆 **Top QML Choice** | ⭐ **Zero False Alarms** | Classical Gold Standard | Linear Baseline | Classical Ensemble |

* **Key Takeaway:** QSVM achieved **92.98% accuracy** and a **92.86% cancer detection rate**, training in **under 2.3 seconds** via vectorized Hermitian matrix multiplication and outperforming classical Random Forest (92.11%).

---

## 2. The Clinical & Computational Challenge

Fine-needle aspirate (FNA) biopsies describe cell nucleus morphology (radius, perimeter, texture, concavity, symmetry). In clinical oncology, missing an invasive cancer case (false negative) is catastrophic. 

1. **High Dimensional Non-Linearities:** Subtle biological cross-talk between multiple nuclear features defines early malignancy.
2. **Computational Tractability:** Simulating 30 qubits directly would require storing $2^{30} \approx 10^9$ complex amplitudes (~16 GB RAM per state).
3. **Engineering Solution:** We standardized clinical features and used Principal Component Analysis (PCA) to project 30 features into **4 salient principal components** capturing **79.63% of total variance**, scaling values onto $[0, \pi]$ to eliminate trigonometric phase wrapping in quantum gates.

---

## 3. Theoretical Foundations

### 3.1 Havlíček Second-Order Pauli-Z Expansion (`ZZFeatureMap`)
Following Havlíček et al. (*Nature* 567, 209–212, 2019), classical vectors $\mathbf{x} \in [0, \pi]^n$ are mapped into an entangled state $|\phi(\mathbf{x})\rangle$:

$$|\phi(\mathbf{x})\rangle = \mathcal{U}_{\Phi(\mathbf{x})} |0\rangle^{\otimes n} = \left[ U_{\Phi(\mathbf{x})} H^{\otimes n} \right]^d |0\rangle^{\otimes n}$$

where the entangling unitary incorporates single-qubit phases and pairwise interactions:

$$U_{\Phi(\mathbf{x})} = \exp\left( i \sum_{i=1}^n x_i Z_i + i \sum_{i=1}^n \sum_{j > i}^n 2(\pi - x_i)(\pi - x_j) Z_i Z_j \right)$$

Synthesized circuit topology for two qubits:
```
       ┌───┐ ┌──────────────┐
q_0: ──┤ H ├─┤ Rz(2*x[0])   ├──■────────────────────────■──
        ├───┤ ├──────────────┤┌─┴─┐┌──────────────────┐┌─┴─┐
q_1: ──┤ H ├─┤ Rz(2*x[1])   ├┤ X ├┤ Rz(2*(π-x0)(π-x1))├┤ X ├──
        └───┘ └──────────────┘└───┘└──────────────────┘└───┘
```

---

## 4. Dual Hybrid Quantum Architectures

### 4.1 Quantum Support Vector Machine (QSVM)
Computes transition fidelity directly in 16-dimensional quantum Hilbert space:

$$K(\mathbf{x}_i, \mathbf{x}_j) = |\langle \phi(\mathbf{x}_i) | \phi(\mathbf{x}_j) \rangle|^2$$

* **$O(N)$ Vectorized Acceleration:** Statevectors are simulated once per patient ($\mathbf{\Psi} \in \mathbb{C}^{N \times 2^n}$), and the Gram matrix is computed via BLAS Hermitian multiplication $\mathbf{K} = |\mathbf{\Psi}\mathbf{\Psi}^\dagger|^2$ in **2.2 seconds** (avoiding $O(N^2)$ circuit runs).
* **Platt Probability Calibration:** Monotonic 1D logistic calibration provides calibrated confidence scores without deprecated Scikit-learn warnings.

### 4.2 Variational Quantum Classifier (VQC)
A trainable Quantum Neural Network:
1. **Encoding:** Single-qubit $R_Y(x_i)$ angle encoding.
2. **Ansatz:** Alternating layers of parameterized $R_Y(\theta), R_Z(\phi)$ gates and circular CNOT entanglement rings.
3. **Readout & Loss:** Pauli-Z measurement $\langle Z_0 \rangle \in [-1, 1]$ mapped through a sigmoid logit and optimized via Adam on Binary Cross-Entropy loss.

---

## 5. Interactive Streamlit Web Dashboard

The web application (`app.py`) provides three tailored views:

```bash
streamlit run app.py --server.port 8501
```

1. **🩺 Clinician Diagnostic Hub:**
   * **Actionable Risk Assessment:** Large verdict banner (`🔴 High Risk: Malignant` vs `🟢 Low Risk: Benign`) with immediate next-step guidance (biopsy orders vs 12-month routine follow-up).
   * **Digital Tumor Board Consensus:** Real-time agreement meters across 2 Quantum and 3 Classical models.
   * **Morphology Explainability:** Interactive bar chart comparing patient cellular radius, perimeter, and concavity against healthy population averages.
   * **1-Click Clinical Presets:** Pre-configured cases (Confirmed Malignant, Confirmed Benign, and Borderline Ambiguous) for instant presentation demos.

2. **📊 Executive & Judge Scorecard:**
   * Head-to-head metrics table comparing Accuracy, Precision, Benign Recall, Malignant Sensitivity, ROC-AUC, and Latency.
   * Embedded high-resolution ROC curves and Confusion Matrix grids.

3. **🔬 Quantum Architecture & NISQ Lab:**
   * Live circuit schematic showing the exact synthesized gates.
   * Qubit rotation angle inspector ($\theta_1, \theta_2, \theta_3, \theta_4$) with $\pi$-bound reference line.
   * Hardware noise degradation curve under physical depolarizing gate noise.

---

## 6. NISQ Hardware Noise Study

To assess real-world quantum hardware readiness, we simulated physical depolarizing gate infidelities using PennyLane's `default.mixed` density matrix simulator:

$$\mathcal{E}(\rho) = (1 - p)\rho + \frac{p}{2^n} I$$

* **Findings:**
  * Up to $p = 0.02$ (2% error rate per gate), classification accuracy remains stable at **87.5%**.
  * At $p = 0.04$, ROC-AUC remains strong at **0.9520**.
  * At high noise levels ($p > 0.08$), state contraction toward the maximally mixed state motivates Zero-Noise Extrapolation (ZNE) error mitigation on physical QPUs.
* Visual artifacts available in [`reports/noise_robustness.png`](reports/noise_robustness.png) and [`reports/noise_analysis.md`](reports/noise_analysis.md).

---

## 7. Benchmark Results vs. Classical Baselines

```
======================================================================
                    BENCHMARK RESULTS (569 Patients)
======================================================================
Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Train Time (s)
--- | --- | --- | --- | --- | --- | ---
QSVM | 0.9298 | 0.9571 | 0.9306 | 0.9437 | 0.9686 | 2.224s
VQC | 0.9035 | 0.8675 | 1.0000 | 0.9290 | 0.9775 | 249.85s
Classical_SVM | 0.9561 | 0.9589 | 0.9722 | 0.9655 | 0.9924 | 0.012s
Logistic_Regression | 0.9561 | 0.9467 | 0.9861 | 0.9660 | 0.9931 | 0.007s
Random_Forest | 0.9211 | 0.9437 | 0.9306 | 0.9371 | 0.9869 | 0.211s
======================================================================
```

* **ROC Curves:** [`reports/roc_comparison.png`](reports/roc_comparison.png)
* **Confusion Matrices:** [`reports/confusion_matrices.png`](reports/confusion_matrices.png)
* **Full Report:** [`reports/benchmark_summary.md`](reports/benchmark_summary.md)

---

## 8. Project Directory Structure

```
d:\quantum-hackday -sample\
├── main.py                     # Unified CLI pipeline runner & benchmark orchestrator
├── app.py                      # Interactive Streamlit Web Dashboard (Port 8501)
├── demo_walkthrough.ipynb      # Fully pre-rendered Jupyter notebook with figures
├── requirements.txt            # Project dependencies
├── README.md                   # Scientific documentation and benchmark overview
├── pytest.ini                  # Pytest runner configuration
├── reports/                    # Output artifacts, figures, and presentation pitch deck
│   ├── benchmark_summary.md
│   ├── benchmark_summary.json
│   ├── roc_comparison.png
│   ├── confusion_matrices.png
│   ├── noise_robustness.png
│   ├── noise_analysis.md
│   ├── quantum_circuit_zz.txt
│   ├── vqc_weights_4q.npz      # Cached pre-trained VQC weights
│   ├── PITCH_DECK.md           # 3-minute pitch script and judge Q&A guide
│   └── GITHUB_SUBMISSION_GUIDE.md
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocessor.py     # Stratified split, PCA, [0, pi] scaling, transform()
│   ├── quantum/
│   │   ├── __init__.py
│   │   ├── feature_maps.py     # Havlicek ZZFeatureMap, Angle encoding, fidelity
│   │   └── noise_study.py      # NISQ depolarizing noise simulator
│   ├── models/
│   │   ├── __init__.py
│   │   ├── qsvm.py             # Vectorized Quantum Kernel Classifier with Platt scaling
│   │   ├── vqc.py              # Variational Quantum Classifier with Adam optimization
│   │   └── classical.py        # Classical SVM, Logistic Regression, Random Forest
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py          # Clinical metrics (Sensitivity, Precision, ROC-AUC)
│       └── benchmark.py        # Comparative benchmark runner & visualizer
└── tests/                      # 18 unit & integration test suites (100% passing)
```

---

## 9. Quick Start Guide

### 9.1 Environment Setup
```bash
# Clone the repository
git clone <YOUR_REPO_URL>
cd "quantum-hackday-sample"

# Activate virtual environment
.\qiskit-env\Scripts\activate   # Windows
# source qiskit-env/bin/activate # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 9.2 Launching the Interactive Web Dashboard
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

### 9.3 Running the Unified CLI Pipeline
```bash
python main.py --n-qubits 4 --epochs 15
```

---

## 10. Testing & Verification

All components are verified via comprehensive `pytest` test suites:

```bash
pytest tests/ -v
```

Output:
```
============================= 18 passed in 9.07s =============================
```

---

## 11. Pitch Deck & Presentation Guide

For hackathon and conference judging sessions, refer to:
* **[`reports/PITCH_DECK.md`](reports/PITCH_DECK.md)**: 8 presentation slides, 3-minute verbatim speaker script, and anticipated judge Q&A answers.
* **[`reports/GITHUB_SUBMISSION_GUIDE.md`](reports/GITHUB_SUBMISSION_GUIDE.md)**: GitHub setup and remote push instructions.

---

## 12. Ethics & Data Privacy

* **De-identified Data:** This project uses the open-access **Breast Cancer Wisconsin (Diagnostic)** dataset collected by Dr. William H. Wolberg, W. Nick Street, and Olvi L. Mangasarian (University of Wisconsin).
* **Zero Personally Identifiable Information (PII):** The dataset consists solely of anonymized numeric cell nuclear morphology measurements. No patient names, dates of birth, medical record numbers, or locations are included or processed.
