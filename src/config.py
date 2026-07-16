"""
Single source of truth for every tunable in this project. Every other
module imports `settings` from here rather than hardcoding a magic number,
so `python -m src.server` and `benchmark/routing_eval.py` are guaranteed to
agree on which layer, which checkpoint, and which model is in play -- a
mismatch between the probe's training config and the serving config would
silently degrade the AUC to noise without raising an error anywhere.

Overridable via environment variables (prefix `PGI_`), e.g.:
    PGI_MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
    PGI_MAX_BATCH_SIZE=8
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PGI_")

    # ── Model ──────────────────────────────────────────────────────────────
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    torch_dtype: str = "float16"

    # ── Probe (must match early_detection's winning configuration) ─────────
    # early_detection found the activation-vs-behavioral advantage peaks at
    # layer 16 (60% depth of 28 layers) and token 150 -- see
    # early_detection/README.md "Sweep Across Checkpoint Positions". Changing
    # either without retraining the probe (scripts/train_probe.py) silently
    # runs a probe against activations it was never fit on.
    probe_layer: int = 16
    checkpoint_position: int = 150
    probe_weights_path: Path = Path("probe_weights/probe_layer16_cp150.pkl")

    # ── Routing thresholds (docx "Step 3: Three Routing Strategies") ───────
    terminate_confidence_threshold: float = 0.7

    # ── Continuous batching scheduler ───────────────────────────────────────
    max_batch_size: int = 8
    max_new_tokens: int = 10_000  # matches early_detection's 10K non-convergence cap
    tick_sleep_s: float = 0.0

    # ── Server ───────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    # ── Benchmark ────────────────────────────────────────────────────────────
    aime_dataset: str = "gneubig/aime-1983-2024"
    aime_seed: int = 42
    aime_n_samples: int = 200
    think_end_token_id: int = 151649
    think_start_token_id: int = 151648


settings = Settings()
