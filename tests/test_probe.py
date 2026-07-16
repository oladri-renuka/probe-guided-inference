"""
Unit tests for src/probe.py -- probe artifact loading/validation and the
layer-activation forward hook, in isolation from the scheduler.
"""

import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.probe import LayerActivationCapture, ProbeArtifact, ProbeNotTrainedError, load_probe

HIDDEN_DIM = 8


def _fitted_pipeline() -> Pipeline:
    rng = np.random.RandomState(0)
    X = rng.randn(40, HIDDEN_DIM)
    y = (X[:, 0] > 0).astype(int)
    pipeline = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    pipeline.fit(X, y)
    return pipeline


def make_artifact(layer=16, checkpoint=150) -> ProbeArtifact:
    return ProbeArtifact(
        pipeline=_fitted_pipeline(), trained_layer=layer, trained_checkpoint=checkpoint,
        expected_hidden_dim=HIDDEN_DIM, n_train_samples=40, train_auc=0.61,
    )


def test_predict_returns_label_and_confidence_above_half():
    artifact = make_artifact()
    label, confidence = artifact.predict(np.array([5.0] * HIDDEN_DIM))
    assert label in ("convergent", "divergent")
    assert 0.5 <= confidence <= 1.0


def test_predict_batch_matches_single_predict_row_by_row():
    artifact = make_artifact()
    rng = np.random.RandomState(1)
    mat = rng.randn(5, HIDDEN_DIM)
    labels, confidences = artifact.predict_batch(mat)
    for i in range(5):
        expected_label, expected_conf = artifact.predict(mat[i])
        assert labels[i] == expected_label
        assert confidences[i] == pytest.approx(expected_conf)


def test_load_probe_missing_file_raises_probe_not_trained_error(tmp_path):
    missing = tmp_path / "does_not_exist.pkl"
    with pytest.raises(ProbeNotTrainedError):
        load_probe(missing, expected_layer=16, expected_checkpoint=150)


def test_load_probe_rejects_layer_mismatch(tmp_path):
    import joblib
    artifact = make_artifact(layer=16, checkpoint=150)
    path = tmp_path / "probe.pkl"
    joblib.dump(artifact.__dict__, path)

    with pytest.raises(ValueError, match="layer"):
        load_probe(path, expected_layer=20, expected_checkpoint=150)


def test_load_probe_rejects_checkpoint_mismatch(tmp_path):
    import joblib
    artifact = make_artifact(layer=16, checkpoint=150)
    path = tmp_path / "probe.pkl"
    joblib.dump(artifact.__dict__, path)

    with pytest.raises(ValueError, match="checkpoint"):
        load_probe(path, expected_layer=16, expected_checkpoint=200)


def test_load_probe_accepts_matching_config(tmp_path):
    import joblib
    artifact = make_artifact(layer=16, checkpoint=150)
    path = tmp_path / "probe.pkl"
    joblib.dump(artifact.__dict__, path)

    loaded = load_probe(path, expected_layer=16, expected_checkpoint=150)
    assert loaded.trained_layer == 16
    assert loaded.trained_checkpoint == 150


def test_layer_activation_capture_fires_and_shape_matches(tiny_model):
    capture = LayerActivationCapture(tiny_model, layer_idx=1)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        tiny_model(input_ids=input_ids)
    assert capture.last is not None
    assert capture.last.shape == (1, 4, tiny_model.config.hidden_size)
    capture.remove()


def test_layer_activation_capture_resolve_layer_error_message(tiny_model):
    with pytest.raises(AttributeError, match="Llama/Qwen2-style"):
        LayerActivationCapture._resolve_layer(torch.nn.Linear(2, 2), layer_idx=0)
