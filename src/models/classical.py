"""Classical machine learning baseline models."""

from typing import Dict, Any, Optional
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


class ClassicalSVMClassifier(BaseEstimator, ClassifierMixin):
    """Classical Support Vector Machine with RBF kernel and logistic calibration."""

    def __init__(self, C: float = 1.0, random_state: int = 42) -> None:
        self.C = C
        self.random_state = random_state
        self.svc_ = SVC(kernel="rbf", C=self.C, random_state=self.random_state)
        self.calibrator_: Optional[LogisticRegression] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClassicalSVMClassifier":
        self.classes_ = np.unique(y)
        self.svc_.fit(X, y)
        scores = self.svc_.decision_function(X).reshape(-1, 1)
        self.calibrator_ = LogisticRegression(random_state=self.random_state)
        self.calibrator_.fit(scores, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.svc_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self.svc_.decision_function(X).reshape(-1, 1)
        if self.calibrator_ is not None:
            return self.calibrator_.predict_proba(scores)
        p1 = 1.0 / (1.0 + np.exp(-scores.ravel()))
        return np.column_stack([1.0 - p1, p1])

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.svc_.decision_function(X)


def get_classical_baselines(random_state: int = 42) -> Dict[str, Any]:
    """Instantiates classical benchmark models for disease classification comparison.
    
    Includes:
    1. Support Vector Classifier with radial basis function (RBF) kernel.
    2. Logistic Regression (L2 regularized).
    3. Random Forest Classifier (ensemble of 100 decision trees).
    """
    return {
        "Classical_SVM": ClassicalSVMClassifier(
            C=1.0,
            random_state=random_state,
        ),
        "Logistic_Regression": LogisticRegression(
            random_state=random_state,
            max_iter=1000,
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
        ),
    }
