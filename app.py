"""Interactive Streamlit Web Dashboard for Hybrid Quantum Disease Classification.

Run with:
    streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

from src.data.preprocessor import DataPipeline
from src.quantum.feature_maps import QuantumFeatureEncoder
from src.models.qsvm import QuantumKernelClassifier
from src.models.vqc import VariationalQuantumClassifier
from src.models.classical import get_classical_baselines


st.set_page_config(
    page_title="Quantum Disease Classifier",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #3B82F6;
    }
    .malignant-badge {
        background-color: #FEE2E2;
        color: #991B1B;
        font-weight: bold;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
    }
    .benign-badge {
        background-color: #D1FAE5;
        color: #065F46;
        font-weight: bold;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_and_train_models(n_qubits: int = 4):
    """Loads dataset and trains quantum and classical models once."""
    pipeline = DataPipeline(n_components=n_qubits, test_size=0.2, random_state=42)
    data = pipeline.fit_transform()
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    # 1. QSVM
    qsvm = QuantumKernelClassifier(
        n_qubits=n_qubits,
        encoding_type="zz",
        reps=1,
        C=1.0,
        random_state=42,
    )
    qsvm.fit(X_train, y_train)

    # 2. VQC
    vqc = VariationalQuantumClassifier(
        n_qubits=n_qubits,
        n_layers=2,
        epochs=8,
        lr=0.05,
        random_state=42,
    )
    vqc.fit(X_train, y_train)

    # 3. Classical Baselines
    classical = get_classical_baselines(random_state=42)
    for model in classical.values():
        model.fit(X_train, y_train)

    return pipeline, data, qsvm, vqc, classical


# Header
st.markdown('<div class="main-header">⚛️ Quantum Machine Learning for Disease Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Hybrid Quantum-Classical Diagnostics on the Breast Cancer Wisconsin Dataset</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("🔬 Configuration")
n_qubits = st.sidebar.selectbox("Quantum Registers (Qubits / PCA Features)", [4, 6], index=0)

with st.spinner("Initializing quantum circuits and pre-training models..."):
    pipeline, data, qsvm, vqc, classical = load_and_train_models(n_qubits)

raw_bunch = load_breast_cancer()
raw_features = raw_bunch.data
raw_targets = raw_bunch.target
feature_names = raw_bunch.feature_names

st.sidebar.markdown("---")
st.sidebar.subheader("👤 Patient Case Selection")
input_mode = st.sidebar.radio("Input Mode", ["Sample Patient from Test Set", "Interactive Feature Sliders"])

sample_idx = 0
if input_mode == "Sample Patient from Test Set":
    test_case_options = [
        f"Case {i}: Ground Truth {'Benign' if data['y_test'][i] == 1 else 'Malignant'}"
        for i in range(min(20, len(data["y_test"])))
    ]
    selected_option = st.sidebar.selectbox("Select Patient Case", test_case_options)
    sample_idx = int(selected_option.split()[1].replace(":", ""))
    patient_q_features = data["X_test"][sample_idx]
    ground_truth = data["y_test"][sample_idx]
else:
    st.sidebar.markdown("**Key Nuclear Characteristics:**")
    mean_radius = st.sidebar.slider("Mean Radius", 6.0, 30.0, 14.1)
    mean_texture = st.sidebar.slider("Mean Texture", 9.0, 40.0, 19.3)
    mean_perimeter = st.sidebar.slider("Mean Perimeter", 43.0, 190.0, 92.0)
    mean_area = st.sidebar.slider("Mean Area", 140.0, 2500.0, 650.0)
    mean_concavity = st.sidebar.slider("Mean Concavity", 0.0, 0.45, 0.08)

    # Synthesize realistic 30-feature vector based on dataset mean
    custom_vector = np.mean(raw_features, axis=0)
    custom_vector[0] = mean_radius
    custom_vector[1] = mean_texture
    custom_vector[2] = mean_perimeter
    custom_vector[3] = mean_area
    custom_vector[6] = mean_concavity

    patient_q_features = pipeline.transform(custom_vector.reshape(1, -1))[0]
    ground_truth = None

# Top Section: Clinical Diagnostic Verdict
st.subheader("📋 Diagnostic Predictions")

# Evaluate all models on patient_q_features
q_input = patient_q_features.reshape(1, -1)

# Predictions
qsvm_pred = int(qsvm.predict(q_input)[0])
qsvm_prob = qsvm.predict_proba(q_input)[0]

vqc_pred = int(vqc.predict(q_input)[0])
vqc_prob = vqc.predict_proba(q_input)[0]

c_svm_pred = int(classical["Classical_SVM"].predict(q_input)[0])
c_svm_prob = classical["Classical_SVM"].predict_proba(q_input)[0]

lr_pred = int(classical["Logistic_Regression"].predict(q_input)[0])
lr_prob = classical["Logistic_Regression"].predict_proba(q_input)[0]

rf_pred = int(classical["Random_Forest"].predict(q_input)[0])
rf_prob = classical["Random_Forest"].predict_proba(q_input)[0]

labels = ["Malignant (Class 0)", "Benign (Class 1)"]

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### ⚛️ QSVM (Quantum Kernel)")
    verdict = "Benign" if qsvm_pred == 1 else "Malignant"
    badge = "benign-badge" if qsvm_pred == 1 else "malignant-badge"
    st.markdown(f'<span class="{badge}">{verdict}</span>', unsafe_allow_html=True)
    st.progress(float(qsvm_prob[qsvm_pred]))
    st.caption(f"Confidence: {qsvm_prob[qsvm_pred]*100:.1f}%")

with col2:
    st.markdown("### 🧬 VQC (Quantum Neural Net)")
    verdict = "Benign" if vqc_pred == 1 else "Malignant"
    badge = "benign-badge" if vqc_pred == 1 else "malignant-badge"
    st.markdown(f'<span class="{badge}">{verdict}</span>', unsafe_allow_html=True)
    st.progress(float(vqc_prob[vqc_pred]))
    st.caption(f"Confidence: {vqc_prob[vqc_pred]*100:.1f}%")

with col3:
    st.markdown("### 🖥️ Classical Baselines Consensus")
    consensus = [c_svm_pred, lr_pred, rf_pred]
    cons_pred = 1 if sum(consensus) >= 2 else 0
    verdict = "Benign" if cons_pred == 1 else "Malignant"
    badge = "benign-badge" if cons_pred == 1 else "malignant-badge"
    st.markdown(f'<span class="{badge}">{verdict}</span>', unsafe_allow_html=True)
    st.caption(f"Classical Agreement: {consensus.count(cons_pred)} / 3 models")

if ground_truth is not None:
    gt_str = "Benign" if ground_truth == 1 else "Malignant"
    st.info(f"🎯 **Ground Truth Pathology Outcome:** {gt_str}")

st.markdown("---")

# Middle Section: Comparative Breakdown & Quantum Encoding
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📊 Comparative Probabilities Across Models")
    comparison_df = pd.DataFrame({
        "Model": ["QSVM (Quantum Kernel)", "VQC (Variational)", "Classical SVM", "Logistic Regression", "Random Forest"],
        "P(Malignant)": [qsvm_prob[0], vqc_prob[0], c_svm_prob[0], lr_prob[0], rf_prob[0]],
        "P(Benign)": [qsvm_prob[1], vqc_prob[1], c_svm_prob[1], lr_prob[1], rf_prob[1]],
        "Decision": [
            "Benign" if qsvm_pred == 1 else "Malignant",
            "Benign" if vqc_pred == 1 else "Malignant",
            "Benign" if c_svm_pred == 1 else "Malignant",
            "Benign" if lr_pred == 1 else "Malignant",
            "Benign" if rf_pred == 1 else "Malignant",
        ]
    })
    st.dataframe(comparison_df.style.format({"P(Malignant)": "{:.3f}", "P(Benign)": "{:.3f}"}), use_container_width=True)

with col_right:
    st.subheader("📐 Quantum State Rotation Angles (Radians)")
    fig, ax = plt.subplots(figsize=(6, 3.2))
    qubit_labels = [f"Qubit {i} (PC{i+1})" for i in range(n_qubits)]
    bars = ax.bar(qubit_labels, patient_q_features, color="#6366F1")
    ax.axhline(np.pi, color="red", linestyle="--", label="π Boundary")
    ax.set_ylabel("Rotation Angle θ (rad)")
    ax.set_ylim(0, np.pi + 0.5)
    ax.legend(loc="upper right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# Bottom Section: Quantum Circuit & Benchmark Reports
st.markdown("---")
tab1, tab2, tab3 = st.tabs(["🔬 Quantum Circuit Architecture", "📈 ROC & Performance Benchmarks", "🗂️ Confusion Matrices"])

with tab1:
    st.subheader("Havlíček Second-Order Pauli-Z Expansion (ZZFeatureMap)")
    st.markdown("""
    Maps classical data vector $x \\in [0, \\pi]^n$ into a $2^n$-dimensional Hilbert space:
    $$U_{\\Phi(x)} = \\exp\\left( i \\sum_i x_i Z_i + i \\sum_{j > i} 2(\\pi - x_i)(\\pi - x_j) Z_i Z_j \\right)$$
    """)
    encoder = QuantumFeatureEncoder(n_qubits=n_qubits, encoding_type="zz", reps=1)
    circuit_ascii = encoder.get_circuit_diagram(patient_q_features)
    st.code(circuit_ascii, language="text")

with tab2:
    st.subheader("Receiver Operating Characteristic (ROC) Comparison")
    if os.path.exists("reports/roc_comparison.png"):
        st.image("reports/roc_comparison.png", use_container_width=True)
    if os.path.exists("reports/benchmark_summary.md"):
        with open("reports/benchmark_summary.md", "r") as f:
            st.markdown(f.read())

with tab3:
    st.subheader("Confusion Matrices (Test Set)")
    if os.path.exists("reports/confusion_matrices.png"):
        st.image("reports/confusion_matrices.png", use_container_width=True)
