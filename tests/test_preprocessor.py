import pytest
import numpy as np
from src.data.preprocessor import DataPipeline

def test_data_pipeline_shapes_and_scaling():
    pipeline = DataPipeline(n_components=4, test_size=0.2, random_state=42)
    data = pipeline.fit_transform()
    
    assert "X_train" in data and "X_test" in data
    assert "y_train" in data and "y_test" in data
    assert data["X_train"].shape[1] == 4
    assert data["X_test"].shape[1] == 4
    assert len(data["X_train"]) + len(data["X_test"]) == 569
    
    # Verify values are bounded in [0, pi] for angle encoding
    assert np.all(data["X_train"] >= -1e-6)
    assert np.all(data["X_train"] <= np.pi + 1e-6)
    
    # Verify stratification
    train_prop = np.mean(data["y_train"])
    test_prop = np.mean(data["y_test"])
    assert abs(train_prop - test_prop) < 0.05

def test_data_pipeline_feature_names():
    pipeline = DataPipeline(n_components=4)
    feature_names = pipeline.get_feature_names()
    assert len(feature_names) == 30
    assert "mean radius" in feature_names

def test_data_pipeline_transform_unseen():
    pipeline = DataPipeline(n_components=4, random_state=42)
    pipeline.fit_transform()
    
    # 2 raw samples with 30 features
    X_raw_new = np.random.normal(10.0, 2.0, size=(2, 30))
    X_proc = pipeline.transform(X_raw_new)
    assert X_proc.shape == (2, 4)
    assert np.all(X_proc >= -1e-6)
    assert np.all(X_proc <= np.pi + 1e-6)
