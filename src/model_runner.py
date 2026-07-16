"""
Bridges GenerationRequest objects (src/queue.py) to the HuggingFace model
forward pass, via the batched cache bridge (src/hf_cache_bridge.py) and the
probe gate (src/gate.py). src/scheduler.py only ever calls these two
functions -- matching inference-server's model_runner.py convention of
being the sole seam between the scheduler and model internals.

This is also the ONLY place the probe's forward hook fires and is read:
`run_decode_step` checks, for every active request, whether this tick just
produced its `checkpoint_position`-th generated token, and if so classifies
that token's layer-16 hidden state and stamps `req.routing_decision`.
src/scheduler.py reacts to that field; this module never decides what a
routing strategy DOES with it.

Decoding is greedy throughout (do_sample=False), matching both
early_detection's methodology (so probe-serving activations are drawn from
the same decoding distribution the probe was trained on) and
inference-server's rationale (deterministic, comparable runs).
"""

from __future__ import annotations

import torch

from src.gate import ClassificationGate
from src.hf_cache_bridge import BatchedSequenceCache
from src.probe import LayerActivationCapture
from src.queue import GenerationRequest


@torch.no_grad()
def run_prefill(
    model,
    cache: BatchedSequenceCache,
    capture: LayerActivationCapture,
    req: GenerationRequest,
    eos_token_id: int,
    max_tokens_cap: int,
) -> bool:
    """Runs a brand-new request's full prompt through the model in one
    forward pass, writing its KV cache and producing its first generated
    token. Returns True if the request is ALREADY finished (immediate EOS,
    or max_tokens <= 1) -- the scheduler should not add it to the active
    decode batch in that case.

    checkpoint_position is never reached during prefill (it counts
    GENERATED tokens, and prefill produces zero of those), so the gate is
    never invoked here -- only in run_decode_step.
    """
    device = model.device
    input_ids = torch.tensor([req.prompt_ids], dtype=torch.long, device=device)
    outputs = model(input_ids=input_ids, use_cache=True)
    seq_len = input_ids.shape[1]
    cache.store_prefill(req.request_id, outputs.past_key_values, seq_len)

    next_id = int(torch.argmax(outputs.logits[0, -1, :]).item())
    req.position = seq_len
    req.generated_ids.append(next_id)
    if req.stream and req.token_queue is not None:
        req.token_queue.put_nowait(next_id)

    finished = next_id == eos_token_id or req.n_generated >= min(req.max_tokens, max_tokens_cap)
    if finished:
        req.finish_reason = "eos" if next_id == eos_token_id else "max_tokens"
    return finished


@torch.no_grad()
def run_decode_step(
    model,
    cache: BatchedSequenceCache,
    capture: LayerActivationCapture,
    gate: ClassificationGate | None,
    active_reqs: list[GenerationRequest],
    eos_token_id: int,
    max_tokens_cap: int,
) -> list[int]:
    """Advances every currently-active request by exactly one token in a
    single batched forward pass. Returns the request_ids that just
    finished (hit EOS or their own max_tokens) so the caller can evict
    them. Requests crossing `checkpoint_position` this tick get
    `req.routing_decision` populated in place; the caller (scheduler)
    decides what to do about it.
    """
    seq_ids = [r.request_id for r in active_reqs]
    next_tokens = torch.tensor(
        [[r.generated_ids[-1]] for r in active_reqs], dtype=torch.long, device=model.device
    )
    batched_cache, attention_mask, position_ids, max_len = cache.build_batch(seq_ids)

    outputs = model(
        input_ids=next_tokens,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=batched_cache,
        use_cache=True,
    )
    cache.scatter_updated_batch(seq_ids, outputs.past_key_values, max_len)

    next_ids = torch.argmax(outputs.logits[:, -1, :], dim=-1)

    # capture.last is (N, 1, hidden_dim) for THIS decode step, same row
    # order as active_reqs -- LayerActivationCapture just stashed whatever
    # the hooked layer produced during the forward pass above. capture is
    # None for the baseline strategy (no gate, so no hook was ever
    # registered -- see ContinuousBatcher.__init__), in which case it's
    # never dereferenced below.
    hidden_this_step = capture.last if capture is not None else None

    finished_ids = []
    for i, req in enumerate(active_reqs):
        nid = int(next_ids[i].item())
        req.generated_ids.append(nid)
        req.position += 1
        if req.stream and req.token_queue is not None:
            req.token_queue.put_nowait(nid)

        if (
            gate is not None
            and req.routing_decision is None
            and gate.ready_at(req.n_generated)
        ):
            hidden_vec = hidden_this_step[i, -1, :].float().cpu().numpy()
            req.routing_decision = gate.classify(hidden_vec)
            import time as _time
            req.checkpoint_hit_time = _time.time()

        if nid == eos_token_id or req.n_generated >= min(req.max_tokens, max_tokens_cap):
            req.finish_reason = "eos" if nid == eos_token_id else "max_tokens"
            finished_ids.append(req.request_id)

    return finished_ids
