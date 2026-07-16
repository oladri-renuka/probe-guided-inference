"""
Unit tests for src/gate.py -- the classification gate, using a stub probe
so these tests exercise gate logic only (readiness predicate, dimension
validation, batching) without any dependency on a real fitted model.
"""

import dataclasses

import numpy as np
import pytest

from src.gate import ClassificationGate, RoutingDecision
from src.probe import ProbeArtifact

HIDDEN_DIM = 4


class _StubPipeline:
    """predict_proba returns a fixed [P(divergent), P(convergent)] pair
    regardless of input -- lets tests assert exact label/confidence
    without needing a real fitted classifier."""

    def __init__(self, prob_convergent: float):
        self.prob_convergent = prob_convergent

    def predict_proba(self, X):
        n = X.shape[0]
        return np.tile([1 - self.prob_convergent, self.prob_convergent], (n, 1))


def make_gate(prob_convergent: float, checkpoint: int = 150) -> ClassificationGate:
    artifact = ProbeArtifact(
        pipeline=_StubPipeline(prob_convergent), trained_layer=16, trained_checkpoint=checkpoint,
        expected_hidden_dim=HIDDEN_DIM, n_train_samples=200, train_auc=0.61,
    )
    return ClassificationGate(artifact)


def test_ready_at_false_before_checkpoint():
    gate = make_gate(0.9, checkpoint=150)
    assert gate.ready_at(149) is False


def test_ready_at_true_at_and_after_checkpoint():
    gate = make_gate(0.9, checkpoint=150)
    assert gate.ready_at(150) is True
    assert gate.ready_at(151) is True


def test_classify_convergent_label_and_confidence():
    gate = make_gate(prob_convergent=0.9)
    decision = gate.classify(np.zeros(HIDDEN_DIM))
    assert decision.label == "convergent"
    assert decision.confidence == pytest.approx(0.9)
    assert decision.is_divergent is False
    assert decision.checkpoint_position == 150


def test_classify_divergent_label_and_confidence():
    gate = make_gate(prob_convergent=0.2)
    decision = gate.classify(np.zeros(HIDDEN_DIM))
    assert decision.label == "divergent"
    assert decision.confidence == pytest.approx(0.8)
    assert decision.is_divergent is True


def test_classify_rejects_wrong_hidden_dim():
    gate = make_gate(prob_convergent=0.9)
    with pytest.raises(ValueError, match="hidden_vec has dim"):
        gate.classify(np.zeros(HIDDEN_DIM + 1))


def test_classify_batch_matches_classify_row_by_row():
    gate = make_gate(prob_convergent=0.7)
    mat = np.random.RandomState(0).randn(6, HIDDEN_DIM)
    decisions = gate.classify_batch(mat)
    assert len(decisions) == 6
    for d in decisions:
        assert isinstance(d, RoutingDecision)
        assert d.label == "convergent"
        assert d.confidence == pytest.approx(0.7)


def test_classify_batch_rejects_wrong_hidden_dim():
    gate = make_gate(prob_convergent=0.9)
    with pytest.raises(ValueError, match="hidden_mat has dim"):
        gate.classify_batch(np.zeros((3, HIDDEN_DIM + 1)))


def test_routing_decision_is_frozen():
    decision = RoutingDecision(label="divergent", confidence=0.8, checkpoint_position=150)
    with pytest.raises(dataclasses.FrozenInstanceError):
        decision.label = "convergent"
