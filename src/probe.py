"""
Loads the trained early-convergence probe and attaches it to a model's
forward pass.

The probe is a `sklearn.pipeline.Pipeline(StandardScaler, LogisticRegression)`
fit by `scripts/train_probe.py` on activations produced by
`early_detection/generate.py`. It is loaded here, not retrained -- this
module's only job is getting the exact same layer-16, position-150 hidden
state vector early_detection validated (AUC 0.612 vs 0.445 behavioral
baseline, p=0.001) in front of that pretrained classifier at serving time.

IMPORTANT -- single-position vector, not mean-pooled:
early_detection's actual probe (early_detection/generate.py
`GenerationInstrumenter._make_layer_hook` /`_sweep_layer_hook`) captures
`hidden_state[:, -1, :]` at the exact checkpoint position -- i.e. the
layer-16 hidden state of the single token generated AT position 150, not a
mean over positions 0..150. The projB build-order pseudocode's
`classify_at_checkpoint` sketches a mean-pool (`hidden_state[0,
:token_position, :].mean(dim=0)`) as a simplified illustration; this
project follows the methodology that was actually validated end-to-end in
early_detection, because the probe's fitted coefficients are meaningless
against a differently-aggregated feature vector, and there is no
early_detection result establishing what the mean-pooled variant's AUC
would even be. See docs/ARCHITECTURE.md ("Deviation: single-position vs.
mean-pooled features") for the full reasoning, and CHECKPOINT_POSITION in
src/config.py for where this couples to the rest of the pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.pipeline import Pipeline

logger = logging.getLogger("probe")


class ProbeNotTrainedError(FileNotFoundError):
    """Raised when probe_weights_path doesn't exist. Points at
    scripts/train_probe.py rather than failing with a bare FileNotFoundError,
    since the fix is a specific command, not a path typo in most cases."""


@dataclass
class ProbeArtifact:
    """A fitted probe plus the metadata needed to trust it at serving time.

    `expected_hidden_dim` and `trained_layer`/`trained_checkpoint` are
    stamped in by train_probe.py and checked against the live model /
    config at load time (see `load_probe`) -- silently running a probe
    against the wrong layer or checkpoint doesn't error, it just returns
    confidently wrong routing decisions, which is worse than a crash.
    """

    pipeline: Pipeline
    trained_layer: int
    trained_checkpoint: int
    expected_hidden_dim: int
    n_train_samples: int
    train_auc: float

    def predict(self, hidden_vec: np.ndarray) -> tuple[str, float]:
        """hidden_vec: 1D array, shape (expected_hidden_dim,).
        Returns (label, confidence) where label in {"convergent", "divergent"}."""
        prob_convergent = float(self.pipeline.predict_proba(hidden_vec.reshape(1, -1))[0, 1])
        label = "convergent" if prob_convergent > 0.5 else "divergent"
        confidence = max(prob_convergent, 1.0 - prob_convergent)
        return label, confidence

    def predict_batch(self, hidden_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """hidden_mat: 2D array, shape (N, expected_hidden_dim).
        Returns (labels, confidences), each shape (N,)."""
        prob_convergent = self.pipeline.predict_proba(hidden_mat)[:, 1]
        labels = np.where(prob_convergent > 0.5, "convergent", "divergent")
        confidences = np.maximum(prob_convergent, 1.0 - prob_convergent)
        return labels, confidences


def load_probe(path: Path, expected_layer: int, expected_checkpoint: int) -> ProbeArtifact:
    if not path.exists():
        raise ProbeNotTrainedError(
            f"No probe weights at {path}. Run `python scripts/train_probe.py` first -- it fits "
            f"and exports the probe from early_detection's checkpoint activations. This project "
            f"does not ship pretrained weights (they're derived from a private ~10-hour GPU run's "
            f"checkpoints, not something to check into git)."
        )
    bundle = joblib.load(path)
    artifact = ProbeArtifact(**bundle)

    if artifact.trained_layer != expected_layer:
        raise ValueError(
            f"Probe at {path} was trained on layer {artifact.trained_layer}, "
            f"but this server is configured for layer {expected_layer} (src/config.py "
            f"PGI_PROBE_LAYER). Routing decisions from a layer-mismatched probe are not "
            f"meaningful -- refusing to load."
        )
    if artifact.trained_checkpoint != expected_checkpoint:
        raise ValueError(
            f"Probe at {path} was trained at checkpoint {artifact.trained_checkpoint}, "
            f"but this server is configured for checkpoint {expected_checkpoint}. Refusing to load."
        )
    logger.info(
        "Loaded probe: layer=%d checkpoint=%d train_auc=%.3f n_train=%d",
        artifact.trained_layer, artifact.trained_checkpoint,
        artifact.train_auc, artifact.n_train_samples,
    )
    return artifact


class LayerActivationCapture:
    """Registers a forward hook on `model's` transformer block at
    `layer_idx` and stashes the most recent output on every call.

    Works for both prefill (output shape [batch, seq_len, hidden]) and
    single-token decode steps (output shape [batch, 1, hidden]) -- callers
    read `.last` and slice whatever position they need immediately after
    the forward pass that produced it; this class does not try to
    interpret WHICH position matters; src/model_runner.py owns that
    (it knows each request's absolute position, this hook does not).
    """

    def __init__(self, model: torch.nn.Module, layer_idx: int):
        self.layer_idx = layer_idx
        self.last: torch.Tensor | None = None
        self._handle = None
        target_layer = self._resolve_layer(model, layer_idx)
        self._handle = target_layer.register_forward_hook(self._hook_fn)

    @staticmethod
    def _resolve_layer(model: torch.nn.Module, layer_idx: int) -> torch.nn.Module:
        # HF causal LMs expose decoder blocks at model.model.layers[i]
        # (LlamaModel/Qwen2Model convention -- DeepSeek-R1-Distill-Qwen-7B
        # is a Qwen2 architecture and follows it). early_detection's
        # generate.py and verify_hooks.py hook the identical path.
        try:
            return model.model.layers[layer_idx]
        except AttributeError as exc:
            raise AttributeError(
                f"Could not resolve model.model.layers[{layer_idx}] on {type(model).__name__}. "
                f"This hook path assumes a Llama/Qwen2-style decoder stack, matching "
                f"early_detection's verify_hooks.py. If you swap in a different model family, "
                f"update LayerActivationCapture._resolve_layer to match its module tree."
            ) from exc

    def _hook_fn(self, module: torch.nn.Module, inputs, output) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        self.last = hidden.detach()

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self) -> LayerActivationCapture:
        return self

    def __exit__(self, *exc_info) -> None:
        self.remove()


def make_capture(model: torch.nn.Module, layer_idx: int) -> LayerActivationCapture:
    """Thin factory so callers don't import the class name directly --
    matches the project's `probe.py` "load weights, run forward hook at
    layer 16" framing from the build-order doc."""
    return LayerActivationCapture(model, layer_idx)
