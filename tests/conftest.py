"""
Shared fixtures for the CPU-only test suite.

`tiny_model` builds a REAL `Qwen2ForCausalLM` (the same architecture as
DeepSeek-R1-Distill-Qwen-7B -- GQA + RoPE + RMSNorm) at a tiny, randomly
initialized size, rather than downloading a "tiny-random" checkpoint from
the HF Hub. This means the entire test suite runs offline, deterministic,
and in seconds, while still exercising the EXACT `transformers` code path
(Cache API, attention, rotary embeddings) that runs in production --
unlike a hand-rolled fake model, a bug in how src/hf_cache_bridge.py talks
to transformers' Cache objects would show up here too.
"""

import pytest
import torch
from transformers.models.qwen2 import Qwen2Config, Qwen2ForCausalLM

TINY_HIDDEN_DIM = 32
TINY_N_LAYERS = 3
TINY_VOCAB = 100
TINY_EOS_TOKEN_ID = 5


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=TINY_VOCAB,
        hidden_size=TINY_HIDDEN_DIM,
        num_hidden_layers=TINY_N_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,  # GQA: fewer KV heads than attention heads, like the real model
        intermediate_size=64,
        max_position_embeddings=256,
        eos_token_id=TINY_EOS_TOKEN_ID,
    )
    model = Qwen2ForCausalLM(config)
    model.eval()
    return model
