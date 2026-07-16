"""
Request queue shared between the HTTP/benchmark layer and the continuous
batching scheduler.

Structurally identical to inference-server's src/queue.py (plain deque, not
asyncio.Queue, so the scheduler can peek/push-front for fair retry
ordering under single-threaded asyncio semantics -- see that project's
docstring for the full reasoning, unchanged here). `GenerationRequest`
grows three fields inference-server's never needed: `routing_decision`,
`priority`, and `metadata`, all specific to probe-guided routing.
"""

import asyncio
import itertools
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

from src.gate import RoutingDecision

_request_id_counter = itertools.count(1)


@dataclass
class GenerationRequest:
    prompt: str
    max_tokens: int
    prompt_ids: list[int]
    arrival_time: float = field(default_factory=time.time)
    request_id: int = field(default_factory=lambda: next(_request_id_counter))
    result: "asyncio.Future" = field(
        default_factory=lambda: asyncio.get_running_loop().create_future()
    )
    stream: bool = False
    token_queue: Optional["asyncio.Queue"] = field(default=None, repr=False)

    # mutated by model_runner.py as generation proceeds
    generated_ids: list[int] = field(default_factory=list)
    position: int = 0  # next absolute position to write into the KV cache (prompt_len + n_generated)

    # probe-guided routing state (all None/default until the gate fires --
    # see src/gate.py ClassificationGate.ready_at)
    routing_decision: RoutingDecision | None = None
    routed: bool = False  # True once the scheduler has reacted to routing_decision (see scheduler.py _apply_routing)
    priority: int = 0  # 0 = normal, 1 = deprioritized (probe_deprioritize strategy only)
    terminated_by_probe: bool = False
    finish_reason: str | None = None  # "eos" | "max_tokens" | "probe_terminated"
    checkpoint_hit_time: float | None = None  # wall-clock time the gate fired, for latency analysis
    completion_time: float | None = None  # wall-clock time the scheduler actually finished this request

    # free-form context the caller (e.g. benchmark/routing_eval.py) attaches
    # and reads back off the finished request -- e.g. {"aime_idx": 12,
    # "expected_answer": "204"}. Keeps AIME-specific fields out of the
    # general-purpose request/scheduler path.
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_generated(self) -> int:
        return len(self.generated_ids)


class RequestQueue:
    def __init__(self):
        self._pending: deque[GenerationRequest] = deque()

    def submit(self, prompt: str, max_tokens: int, prompt_ids: list[int],
               stream: bool = False, metadata: dict | None = None) -> GenerationRequest:
        req = GenerationRequest(prompt=prompt, max_tokens=max_tokens, prompt_ids=prompt_ids,
                                 stream=stream, metadata=metadata or {})
        if stream:
            req.token_queue = asyncio.Queue()
        self._pending.append(req)
        return req

    def peek(self) -> GenerationRequest | None:
        return self._pending[0] if self._pending else None

    def popleft(self) -> GenerationRequest:
        return self._pending.popleft()

    def push_front(self, req: GenerationRequest) -> None:
        """Used when a request couldn't be admitted this tick -- goes back
        to the FRONT so it's retried next tick ahead of later arrivals."""
        self._pending.appendleft(req)

    def push_front_priority_aware(self, req: GenerationRequest) -> None:
        """Used by the probe_deprioritize strategy: a re-queued deprioritized
        (priority=1) request is inserted behind every priority=0 request
        already waiting, instead of unconditionally at the front -- so a
        steady stream of normal-priority arrivals can keep jumping ahead of
        it, which is the entire mechanism the strategy routes on (see
        src/scheduler.py `ProbeDeprioritizeStrategy`)."""
        if req.priority == 0:
            self._pending.appendleft(req)
            return
        # Find the position right after the LAST normal-priority (0) item
        # currently in the queue -- this is the earliest index a
        # deprioritized item may occupy without cutting in front of a
        # normal-priority waiter. Then skip forward past any deprioritized
        # items already sitting at that position, so multiple preempted
        # (priority=1) requests stay in FIFO order relative to each other
        # instead of the newest always landing first.
        last_priority0_idx = -1
        for i, other in enumerate(self._pending):
            if other.priority == 0:
                last_priority0_idx = i
        idx = last_priority0_idx + 1
        while idx < len(self._pending) and self._pending[idx].priority == 1:
            idx += 1
        self._pending.insert(idx, req)

    def __len__(self) -> int:
        return len(self._pending)
