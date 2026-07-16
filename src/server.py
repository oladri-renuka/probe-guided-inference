"""
FastAPI wrapper around the probe-guided continuous batching scheduler.

Structurally the same shape as inference-server's src/server.py:
POST /generate enqueues a request and awaits the scheduler's result future
-- it never runs the model inline. Generation happens in
ContinuousBatcher.run_forever(), started once as a background asyncio
task at startup.

Routing strategy is fixed at process startup (env var PGI_STRATEGY), not
per-request -- comparing strategies means running three separate server
processes against identical traffic (see benchmark/routing_eval.py), the
same way inference-server benchmarks naive/static/continuous as three
separate processes rather than a per-request flag.

Run:
    PGI_STRATEGY=probe_deprioritize uvicorn src.server:app --host 0.0.0.0 --port 8000
"""

import asyncio
import json
import os
import time

import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import settings
from src.gate import ClassificationGate
from src.probe import load_probe
from src.queue import RequestQueue
from src.scheduler import ContinuousBatcher, RoutingStrategy

STRATEGY = RoutingStrategy(os.environ.get("PGI_STRATEGY", RoutingStrategy.BASELINE.value))

app = FastAPI(title="Probe-Guided Inference Scheduler", description=f"strategy={STRATEGY.value}")


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = settings.max_new_tokens


class GenerateResponse(BaseModel):
    text: str
    latency_s: float
    finish_reason: str | None = None
    routing_label: str | None = None
    routing_confidence: float | None = None


@app.on_event("startup")
async def startup() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    app.state.device = device
    app.state.strategy = STRATEGY

    app.state.tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
    app.state.model = AutoModelForCausalLM.from_pretrained(
        settings.model_name,
        torch_dtype=getattr(torch, settings.torch_dtype),
        device_map=device,
    )
    app.state.model.eval()

    gate = None
    if STRATEGY != RoutingStrategy.BASELINE:
        probe = load_probe(settings.probe_weights_path, settings.probe_layer, settings.checkpoint_position)
        gate = ClassificationGate(probe)
    app.state.gate = gate

    app.state.queue = RequestQueue()
    app.state.batcher = ContinuousBatcher(
        model=app.state.model,
        tokenizer=app.state.tokenizer,
        queue=app.state.queue,
        strategy=STRATEGY,
        gate=gate,
        max_batch_size=settings.max_batch_size,
        max_tokens_cap=settings.max_new_tokens,
        terminate_confidence_threshold=settings.terminate_confidence_threshold,
        tick_sleep=settings.tick_sleep_s,
    )
    app.state.batcher_task = asyncio.create_task(app.state.batcher.run_forever())


@app.on_event("shutdown")
async def shutdown() -> None:
    app.state.batcher.stop()
    app.state.batcher.close()
    app.state.batcher_task.cancel()


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    start = time.time()
    tokenizer = app.state.tokenizer
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": req.prompt}], add_generation_prompt=True, tokenize=False,
    )
    prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"][0].tolist()
    gen_req = app.state.queue.submit(prompt_text, req.max_tokens, prompt_ids)
    text = await gen_req.result
    return GenerateResponse(
        text=text,
        latency_s=time.time() - start,
        finish_reason=gen_req.finish_reason,
        routing_label=gen_req.routing_decision.label if gen_req.routing_decision else None,
        routing_confidence=gen_req.routing_decision.confidence if gen_req.routing_decision else None,
    )


@app.post("/stream")
async def stream_generate(req: GenerateRequest):
    tokenizer = app.state.tokenizer
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": req.prompt}], add_generation_prompt=True, tokenize=False,
    )
    prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"][0].tolist()
    gen_req = app.state.queue.submit(prompt_text, req.max_tokens, prompt_ids, stream=True)

    async def token_generator():
        start_time = time.perf_counter()
        first_token_sent = False
        while True:
            token_id = await gen_req.token_queue.get()
            if token_id is None:
                break
            if not first_token_sent:
                ttft = time.perf_counter() - start_time
                yield f"data: {json.dumps({'ttft_ms': round(ttft * 1000, 2)})}\n\n"
                first_token_sent = True
            token_text = tokenizer.decode([token_id])
            yield f"data: {json.dumps({'token': token_text})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        token_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
async def health():
    batcher = app.state.batcher
    return {
        "status": "ok",
        "strategy": app.state.strategy.value,
        "active_sequences": len(batcher.active),
        "queued": len(app.state.queue),
        "total_admitted": batcher.total_admitted,
        "total_finished": batcher.total_finished,
        "total_probe_terminated": batcher.total_probe_terminated,
        "total_preemptions": batcher.total_preemptions,
    }


@app.get("/metrics")
async def metrics():
    """Plain-JSON metrics snapshot (not Prometheus exposition format --
    the benchmark harness reads this directly; if this server is ever
    pointed at an actual Prometheus scraper, wrap this in
    prometheus_client instead of hand-rolling text format here)."""
    batcher = app.state.batcher
    active_priorities = [r.priority for r in batcher.active.values()]
    return {
        "strategy": app.state.strategy.value,
        "active_sequences": len(batcher.active),
        "queued": len(app.state.queue),
        "active_deprioritized": sum(1 for p in active_priorities if p == 1),
        "total_admitted": batcher.total_admitted,
        "total_finished": batcher.total_finished,
        "total_probe_terminated": batcher.total_probe_terminated,
        "total_preemptions": batcher.total_preemptions,
        "gpu_current_bytes_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else None,
        "gpu_peak_bytes_allocated": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
    }
