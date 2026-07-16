"""
Batched KV-cache bridge for heterogeneous-length continuous batching on a
real HuggingFace model.

inference-server's `paged_cache.py` / `contiguous_cache.py` implement their
own attention math by hand (`gpt2_model.py`) precisely so they can define a
`write(seq_id, layer_idx, pos, k, v)` / `read(seq_id, layer_idx)` interface
and use it interchangeably under a hand-rolled GPT-2 forward pass. That
approach does not transfer to this project -- see docs/ARCHITECTURE.md
("Why HF's forward pass, not a hand-rolled Qwen2") for the full reasoning;
in short, DeepSeek-R1-Distill-Qwen-7B's GQA + RoPE + RMSNorm stack is
materially riskier to reimplement correctly than GPT-2's vanilla MHA, and
any numerical drift from HF's own forward pass would shift the layer-16
hidden states away from the exact distribution the probe was trained
against, silently invalidating its AUC.

Given that constraint, THIS module is the piece inference-server didn't
need to build: a way to run one batched decode step across N sequences
that each have their own, independently-grown HuggingFace `DynamicCache`,
using only its public per-layer `.keys` / `.values` tensors (transformers'
`Cache` object model as of the `DynamicLayer` refactor -- see
"Version note" below) rather than any hand-parsed internal format.

Per-tick algorithm:
  1. Each active sequence's cache is stored, between ticks, as its own
     per-layer (key, value) tensor pair -- shape (1, n_kv_heads,
     seq_len_i, head_dim) -- exactly what a batch-size-1 prefill or decode
     call produces.
  2. To decode a tick, LEFT-pad every active sequence's cache up to the
     tick's max active length and concatenate along the batch dim, so the
     newest real token is always the rightmost column (required for
     causal masking to line up across rows of different history length).
  3. Run ONE forward pass over the padded batch.
  4. Split the returned (now one-token-longer) batched cache back into
     per-sequence (key, value) pairs, RIGHT-trimming back down to that
     sequence's true (now +1) length -- undoing the padding immediately so
     next tick's padding amount is computed fresh against each sequence's
     actual length, not an accumulated pad.

This is the naive/contiguous end of the paging spectrum inference-server's
README describes (full re-pad-and-copy every tick, not incremental block
allocation) -- appropriate here because AIME generations run one sequence
at a time to convergence rather than under bursty multi-tenant arrival, so
the paged allocator's main benefit (not wasting memory reserved for tokens
never generated) matters far less than it does for inference-server's
short/long mixed-prompt HTTP workload.

Version note: transformers' Cache object model changed twice in ways that
matter here. Older releases exposed `Cache.to_legacy_cache()` /
`DynamicCache.from_legacy_cache(tuple_of_(k,v))`; transformers 5.x removed
both in favor of `cache.layers[i].keys` / `.values` (a `DynamicLayer` per
layer) and a `DynamicCache(ddp_cache_data=[(k0,v0), (k1,v1), ...])`
constructor for building one from raw tensors. This module targets the
transformers>=5.0 API (confirmed against 5.5.4, the version this project
was built and tested against) -- see requirements.txt's floor. If pinned to
an older transformers for some other reason, `_layer_kv` / `_build_cache`
below are the only two functions that need a compatibility branch.
"""

from __future__ import annotations

import torch
from transformers import DynamicCache

LayerKV = tuple[torch.Tensor, torch.Tensor]  # (key, value), each (batch, n_kv_heads, seq_len, head_dim)
SeqCache = tuple[LayerKV, ...]  # one entry per layer


def _layer_kv(cache: DynamicCache, layer_idx: int) -> LayerKV:
    layer = cache.layers[layer_idx]
    return layer.keys, layer.values


def _num_layers(cache: DynamicCache) -> int:
    return len(cache.layers)


def _build_cache(per_layer_kv: list[LayerKV]) -> DynamicCache:
    return DynamicCache(ddp_cache_data=per_layer_kv)


class BatchedSequenceCache:
    """Owns one HF-native KV cache per active request_id and knows how to
    fuse/split them for batched decode steps. Does NOT know about
    GenerationRequest, routing, or the probe -- src/model_runner.py owns
    that; this class is pure cache-plumbing, unit-tested in isolation
    (tests/test_hf_cache_bridge.py) the same way inference-server's
    test_paged_cache.py tests its allocator without a model.
    """

    def __init__(self):
        self._cache: dict[int, SeqCache] = {}
        self._seq_len: dict[int, int] = {}

    def store_prefill(self, seq_id: int, past_key_values: DynamicCache, seq_len: int) -> None:
        n_layers = _num_layers(past_key_values)
        self._cache[seq_id] = tuple(_layer_kv(past_key_values, i) for i in range(n_layers))
        self._seq_len[seq_id] = seq_len

    def seq_len(self, seq_id: int) -> int:
        return self._seq_len[seq_id]

    def evict(self, seq_id: int) -> None:
        self._cache.pop(seq_id, None)
        self._seq_len.pop(seq_id, None)

    @property
    def active_ids(self) -> list[int]:
        return list(self._cache.keys())

    def build_batch(self, seq_ids: list[int]) -> tuple[DynamicCache, torch.Tensor, torch.Tensor, int]:
        """Returns (batched_cache, attention_mask, position_ids, max_len)
        for exactly these seq_ids, in this order. attention_mask covers the
        padded history PLUS the one new token about to be produced this
        step (shape (N, max_len + 1)); position_ids is (N, 1), each row's
        TRUE absolute position of the new token (i.e. that sequence's
        current seq_len, unaffected by how much left-padding other rows in
        the batch needed)."""
        lengths = [self._seq_len[s] for s in seq_ids]
        max_len = max(lengths)
        n_layers = len(self._cache[seq_ids[0]])
        sample_k = self._cache[seq_ids[0]][0][0]
        device, dtype = sample_k.device, sample_k.dtype

        batched_layers: list[LayerKV] = []
        for layer_idx in range(n_layers):
            k_rows, v_rows = [], []
            for seq_id, length in zip(seq_ids, lengths, strict=False):
                k, v = self._cache[seq_id][layer_idx]
                pad_amount = max_len - length
                if pad_amount > 0:
                    n_kv_heads, head_dim = k.shape[1], k.shape[3]
                    pad_shape = (1, n_kv_heads, pad_amount, head_dim)
                    k = torch.cat([torch.zeros(pad_shape, dtype=dtype, device=device), k], dim=2)
                    v = torch.cat([torch.zeros(pad_shape, dtype=dtype, device=device), v], dim=2)
                k_rows.append(k)
                v_rows.append(v)
            batched_layers.append((torch.cat(k_rows, dim=0), torch.cat(v_rows, dim=0)))

        attention_mask = torch.zeros(len(seq_ids), max_len + 1, dtype=torch.long, device=device)
        for i, length in enumerate(lengths):
            attention_mask[i, max_len - length:] = 1  # real history + the new token slot

        position_ids = torch.tensor(lengths, dtype=torch.long, device=device).unsqueeze(1)  # (N, 1)

        return _build_cache(batched_layers), attention_mask, position_ids, max_len

    def scatter_updated_batch(self, seq_ids: list[int], updated_past_key_values: DynamicCache, max_len: int) -> None:
        """After a decode step, split the batched (max_len + 1)-length cache
        back into per-sequence (key, value) pairs, trimming each row's left
        padding back off so its stored length is exactly its true new
        length (original_length + 1)."""
        n_layers = _num_layers(updated_past_key_values)
        for i, seq_id in enumerate(seq_ids):
            new_len = self._seq_len[seq_id] + 1
            per_layer = []
            for layer_idx in range(n_layers):
                k, v = _layer_kv(updated_past_key_values, layer_idx)
                k_i = k[i:i + 1, :, -new_len:, :].contiguous()
                v_i = v[i:i + 1, :, -new_len:, :].contiguous()
                per_layer.append((k_i, v_i))
            self._cache[seq_id] = tuple(per_layer)
            self._seq_len[seq_id] = new_len
