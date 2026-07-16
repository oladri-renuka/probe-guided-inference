"""
Unit tests for src/queue.py -- RequestQueue ordering, including the
priority-aware requeue path probe_deprioritize's preemption relies on
(src/scheduler.py `_preempt`).
"""

import pytest

from src.queue import GenerationRequest, RequestQueue


def make_req(name: str, priority: int = 0) -> GenerationRequest:
    req = GenerationRequest(prompt=name, max_tokens=10, prompt_ids=[1])
    req.priority = priority
    return req


@pytest.mark.asyncio
async def test_submit_then_popleft_is_fifo():
    q = RequestQueue()
    r1 = q.submit("a", 10, [1])
    r2 = q.submit("b", 10, [2])
    assert q.popleft() is r1
    assert q.popleft() is r2
    assert len(q) == 0


@pytest.mark.asyncio
async def test_peek_does_not_remove():
    q = RequestQueue()
    r1 = q.submit("a", 10, [1])
    assert q.peek() is r1
    assert len(q) == 1
    assert q.popleft() is r1


@pytest.mark.asyncio
async def test_push_front_puts_request_ahead_of_existing_queue():
    q = RequestQueue()
    r1 = q.submit("a", 10, [1])
    r2 = q.submit("b", 10, [2])
    q.popleft()  # remove r1, simulating a failed admission attempt
    q.push_front(r1)
    assert q.popleft() is r1
    assert q.popleft() is r2


@pytest.mark.asyncio
async def test_priority_aware_requeue_normal_priority_goes_to_front():
    q = RequestQueue()
    victim = make_req("victim", priority=0)
    q.push_front_priority_aware(victim)
    assert q.popleft() is victim


@pytest.mark.asyncio
async def test_priority_aware_requeue_deprioritized_goes_behind_normal_priority_waiters():
    q = RequestQueue()
    normal_waiter = make_req("normal", priority=0)
    q._pending.append(normal_waiter)  # already waiting in line, priority 0

    victim = make_req("victim", priority=1)
    q.push_front_priority_aware(victim)  # preempted, deprioritized -- must NOT jump ahead

    assert q.popleft() is normal_waiter
    assert q.popleft() is victim


@pytest.mark.asyncio
async def test_priority_aware_requeue_deprioritized_preserves_fifo_among_equals():
    """A second deprioritized preemption lands BEHIND the first one (both
    priority=1), preserving FIFO order among equal-priority requests
    rather than always jumping to the very front."""
    q = RequestQueue()
    first_victim = make_req("first", priority=1)
    q.push_front_priority_aware(first_victim)

    second_victim = make_req("second", priority=1)
    q.push_front_priority_aware(second_victim)

    assert q.popleft() is first_victim
    assert q.popleft() is second_victim


@pytest.mark.asyncio
async def test_priority_aware_requeue_normal_priority_jumps_ahead_of_deprioritized():
    q = RequestQueue()
    deprioritized = make_req("deprioritized", priority=1)
    q.push_front_priority_aware(deprioritized)

    normal = make_req("normal", priority=0)
    q.push_front_priority_aware(normal)

    assert q.popleft() is normal
    assert q.popleft() is deprioritized
