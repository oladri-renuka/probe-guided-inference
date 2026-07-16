"""
The classification gate: turns a layer-16 hidden state at token 150 into a
routing decision the scheduler can act on.

This is deliberately a thin, stateless wrapper around `ProbeArtifact.predict`
(src/probe.py) -- all of the "when do we have enough tokens to check" and
"what do we DO with convergent vs divergent" logic lives in
src/model_runner.py and src/scheduler.py respectively. Keeping this module
to just "vector in, label+confidence out" is what makes it unit-testable
without a GPU or a real model (see tests/test_gate.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.probe import ProbeArtifact

Label = Literal["convergent", "divergent"]


@dataclass(frozen=True)
class RoutingDecision:
    label: Label
    confidence: float
    checkpoint_position: int

    @property
    def is_divergent(self) -> bool:
        return self.label == "divergent"


class ClassificationGate:
    """Wraps a ProbeArtifact with the fixed checkpoint position it was
    trained for, so callers can't accidentally invoke it at the wrong
    token position (a request at position 90 has no business being
    classified against a probe trained at position 150 -- its hidden
    state at that point is from a different point in the generation and
    the probe was never shown vectors like it)."""

    def __init__(self, probe: ProbeArtifact):
        self.probe = probe
        self.checkpoint_position = probe.trained_checkpoint

    def ready_at(self, n_generated: int) -> bool:
        """True the instant a request has generated `checkpoint_position`
        tokens. `n_generated` is `len(request.generated_ids)` -- a count of
        GENERATED tokens only, deliberately independent of prompt length or
        KV-cache absolute position, since "150 tokens into the thinking
        chain" (early_detection's framing) means 150 generated tokens
        regardless of how long the prompt was.

        Mirrors the docx build-order's `classify_at_checkpoint`
        `hidden_state.shape[1] < token_position` early-return, but phrased
        as a predicate the caller checks BEFORE running the probe, since
        in the batched serving path there's no "too early" hidden state to
        pass in the first place -- the gate is simply not invoked yet."""
        return n_generated >= self.checkpoint_position

    def classify(self, hidden_vec: np.ndarray) -> RoutingDecision:
        """hidden_vec: 1D array, shape (hidden_dim,) -- the layer-16
        hidden state of the token generated at `checkpoint_position`."""
        if hidden_vec.shape[-1] != self.probe.expected_hidden_dim:
            raise ValueError(
                f"hidden_vec has dim {hidden_vec.shape[-1]}, probe expects "
                f"{self.probe.expected_hidden_dim}. Wrong model or wrong layer hooked?"
            )
        label, confidence = self.probe.predict(hidden_vec)
        return RoutingDecision(label=label, confidence=confidence,
                                checkpoint_position=self.checkpoint_position)

    def classify_batch(self, hidden_mat: np.ndarray) -> list[RoutingDecision]:
        """hidden_mat: 2D array, shape (N, hidden_dim). Returns one
        RoutingDecision per row, same order. Batched purely for
        throughput when several requests cross the checkpoint on the
        same decode tick -- classification logic is identical to
        `classify` called N times."""
        if hidden_mat.shape[-1] != self.probe.expected_hidden_dim:
            raise ValueError(
                f"hidden_mat has dim {hidden_mat.shape[-1]}, probe expects "
                f"{self.probe.expected_hidden_dim}."
            )
        labels, confidences = self.probe.predict_batch(hidden_mat)
        return [
            RoutingDecision(label=str(labels[i]), confidence=float(confidences[i]),
                             checkpoint_position=self.checkpoint_position)
            for i in range(hidden_mat.shape[0])
        ]
