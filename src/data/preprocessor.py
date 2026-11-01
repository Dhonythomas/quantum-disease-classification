"""Data preprocessing and dimensionality reduction for quantum disease classification."""

from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class DataPipeline:
    """Preprocesses the Breast Cancer Wisconsin dataset for quantum classification.
    
    The pipeline executes:
    1. Stratified train/test splitting.
    2. Feature standardization (Z-score).
    3. Dimensionality reduction via PCA to map 30 clinical features to n_components (qubits).
    4. Rescaling to [0, pi] for angle-based quantum feature mapping.
    """

    def __init__(
        self,
        n_components: int = 4,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        self.n_components = n_components
        self.test_size = test_size
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self.minmax_scaler = MinMaxScaler(feature_range=(0.0, float(np.pi)))

        self.raw_data = load_breast_cancer()
        self.feature_names: List[str] = list(self.raw_data.feature_names)
        self.target_names: List[str] = list(self.raw_data.target_names)
        self.explained_variance_ratio_: Optional[np.ndarray] = None

    def get_feature_names(self) -> List[str]:
        """Returns the original 30 clinical feature names."""
        return self.feature_names

    def get_target_names(self) -> List[str]:
        """Returns the target class names (malignant, benign)."""
        return self.target_names

    def fit_transform(self) -> Dict[str, np.ndarray]:
        """Executes full preprocessing pipeline and returns train/test splits.
        
        Returns:
            Dictionary containing 'X_train', 'X_test', 'y_train', 'y_test'.
        """
        X = self.raw_data.data
        y = self.raw_data.target

        # Stratified train/test split to preserve class proportions (Malignant ~37%, Benign ~63%)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        # Standardize train and test features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Reduce dimensionality to n_components for qubit encoding
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        X_test_pca = self.pca.transform(X_test_scaled)
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_

        # Map to [0, pi] range for quantum rotation angles
        X_train_final = self.minmax_scaler.fit_transform(X_train_pca)
        X_test_final = self.minmax_scaler.transform(X_test_pca)

        # Clip values slightly to guard against floating-point boundary issues
        X_train_final = np.clip(X_train_final, 0.0, float(np.pi))
        X_test_final = np.clip(X_test_final, 0.0, float(np.pi))

        return {
            "X_train": X_train_final,
            "X_test": X_test_final,
            "y_train": y_train,
            "y_test": y_test,
        }
