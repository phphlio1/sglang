"""Decode time-slicing for first-token-starved waiters (--boost-preempt-wait-s).

Under KV-memory pressure the queue-wait tail is set by memory, not by order:
the waiting head needs device KV that long-running decodes hold, and output
lengths are heavy-tailed, so the request that has decoded the most is expected
to hold its KV the longest. When a request still waiting for its first token
has queued for ``wait_s`` and the pool cannot fit it, the scheduler releases
the longest-attained decodes into the radix tree -- the finished-request KV
lifecycle, so each resumes from its own cached KV (device, or host after a
write-back eviction) -- and puts the waiter at the head of the next admission
pass. Measurement basis: ph2lio/experiments/e2/docs/boost_tail_memory_bound.md.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, List, Optional, Sequence

import msgspec

from sglang.srt.managers.schedule_policy import CLIP_MAX_NEW_TOKENS

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)

# Arbitrary: one plan per second keeps a released decode's KV from being
# counted twice before the next admission pass has consumed it.
PREEMPT_COOLDOWN_S = 1.0
# Arbitrary bound on the waiter's head-of-queue slot (passes are tens of ms)
# when the released memory still does not fit it.
RESERVE_MAX_PASSES = 256


class DecodePreemptionPlan(msgspec.Struct, frozen=True, kw_only=True):
    target_rid: str
    target_wait_s: float
    need_tokens: int
    deficit_tokens: int
    # Indices into the running batch's reqs, in release order.
    victim_indices: tuple[int, ...]
    victim_output_lens: tuple[int, ...]
    freed_tokens: int


def decode_reserve_tokens(req: Req, new_token_ratio: float) -> float:
    """Future-decode tokens PrefillAdder reserves for a running request."""
    return (
        min(
            req.sampling_params.max_new_tokens - len(req.output_ids),
            CLIP_MAX_NEW_TOKENS,
        )
        * new_token_ratio
    )


def admission_need_tokens(req: Req, page_size: int) -> int:
    """KV tokens PrefillAdder.add_one_req charges a waiting request: the
    uncached tail plus host hits (they load into fresh device slots), the
    clipped decode reservation and one page of alignment overhead."""
    max_new = min(
        max(req.sampling_params.max_new_tokens - len(req.output_ids), 0),
        CLIP_MAX_NEW_TOKENS,
    )
    seq_len = len(req.origin_input_ids) + len(req.output_ids)
    return seq_len - req.prefix_len + max_new + page_size


def is_preemptible(req: Req, *, quantum_tokens: int, max_per_req: int) -> bool:
    # Beam groups cannot be retracted (retract_decode aborts them instead).
    if req.finished() or req.beam_group is not None:
        return False
    if req.num_preemptions >= max_per_req:
        return False
    # The quantum doubles per preemption, so every request keeps progressing.
    decoded = len(req.output_ids) - req.preempt_resume_output_len
    return decoded >= quantum_tokens << req.num_preemptions


def _starving_target(
    waiting_queue: Sequence[Req],
    *,
    now_s: float,
    wait_s: float,
    preferred_rid: Optional[str],
) -> Optional[Req]:
    def starving(req: Req) -> bool:
        # A non-empty output means the first token was already delivered (a
        # preempted or retracted decode); only first-token waiters trigger.
        return (
            not req.output_ids
            and req.first_queue_entry_time is not None
            and now_s - req.first_queue_entry_time >= wait_s
        )

    if preferred_rid is not None:
        for req in waiting_queue:
            if req.rid == preferred_rid and starving(req):
                return req
    for req in waiting_queue:
        if starving(req):
            return req
    return None


def _token_prefix(req: Req, n: int):
    ids = req.origin_input_ids
    return ids[:n] if n <= len(ids) else [*ids, *req.output_ids][:n]


def _common_prefix_len(a: Req, b: Req, limit: int) -> int:
    pa, pb = _token_prefix(a, limit), _token_prefix(b, limit)
    for lo in range(0, limit, 4096):
        hi = min(lo + 4096, limit)
        if list(pa[lo:hi]) != list(pb[lo:hi]):
            return lo + next(
                i for i, (x, y) in enumerate(zip(pa[lo:hi], pb[lo:hi])) if x != y
            )
    return limit


def released_tokens(victim: Req, lockers, page_size: int) -> int:
    """Upstream #48079 r2: own KV plus the tree prefix no staying locker shares."""
    protected = victim.kv.cache_protected_len
    still_locked = 0
    for other in lockers:
        limit = min(protected, other.kv.cache_protected_len)
        if other is not victim and limit > still_locked:
            still_locked = max(still_locked, _common_prefix_len(victim, other, limit))
    return victim.kv.kv_allocated_len - (still_locked - still_locked % page_size)


def plan_decode_preemption(
    *,
    now_s: float,
    waiting_queue: Sequence[Req],
    running_reqs: Sequence[Req],
    have_tokens: int,
    new_token_ratio: float,
    page_size: int,
    wait_s: float,
    quantum_tokens: int,
    max_per_req: int,
    preferred_rid: Optional[str] = None,
    chunked_req: Optional[Req] = None,
) -> Optional[DecodePreemptionPlan]:
    """The smallest set of longest-attained decodes whose release lets the
    first starving waiter fit, or None when it already fits (its wait has
    another cause) or no eligible set covers the deficit.

    ``have_tokens`` is the pool's available + evictable KV tokens."""
    target = _starving_target(
        waiting_queue, now_s=now_s, wait_s=wait_s, preferred_rid=preferred_rid
    )
    if target is None:
        return None
    reserves = [decode_reserve_tokens(r, new_token_ratio) for r in running_reqs]
    need = admission_need_tokens(target, page_size)
    deficit = need - (have_tokens - sum(reserves))
    if deficit <= 0:
        return None

    # Least-attained-service order: the most decoded output first; heavy-tailed
    # outputs make it the longest expected remaining hold.
    candidates = sorted(
        (
            i
            for i, r in enumerate(running_reqs)
            if is_preemptible(r, quantum_tokens=quantum_tokens, max_per_req=max_per_req)
        ),
        key=lambda i: (
            -len(running_reqs[i].output_ids),
            -len(running_reqs[i].origin_input_ids),
        ),
    )
    other_lockers = [] if chunked_req is None else [chunked_req]
    chosen: List[int] = []
    freed = 0.0
    for i in candidates:
        chosen.append(i)
        stays = [r for j, r in enumerate(running_reqs) if j not in chosen]
        freed += reserves[i] + released_tokens(
            running_reqs[i], [*stays, *other_lockers], page_size
        )
        if freed >= deficit:
            return DecodePreemptionPlan(
                target_rid=target.rid,
                target_wait_s=now_s - target.first_queue_entry_time,
                need_tokens=need,
                deficit_tokens=int(math.ceil(deficit)),
                victim_indices=tuple(chosen),
                victim_output_lens=tuple(
                    len(running_reqs[j].output_ids) for j in chosen
                ),
                freed_tokens=int(freed),
            )
    return None


class DecodePreemptor:
    """Per-scheduler state around plan_decode_preemption: the cooldown, the
    waiter's head-of-queue reservation, and the requests' preemption
    bookkeeping (count, decode progress since admission, re-queued stall)."""

    def __init__(
        self,
        *,
        wait_s: float,
        quantum_tokens: int,
        max_per_req: int,
        dp_rank: int,
    ) -> None:
        self.wait_s = wait_s
        self.quantum_tokens = quantum_tokens
        self.max_per_req = max_per_req
        self.dp_rank = dp_rank
        self._last_plan_s = -math.inf
        self._target_rid: Optional[str] = None
        self._target_passes_left = 0
        self.num_events = 0
        self.num_victims = 0

    @classmethod
    def maybe_create(
        cls,
        *,
        wait_s: float,
        quantum_tokens: int,
        max_per_req: int,
        dp_rank: int,
        tree_supports_mamba: bool,
        tree_supports_swa: bool,
        disaggregation_mode: str,
    ) -> Optional[DecodePreemptor]:
        if wait_s <= 0:
            return None
        # Only the FULL-component release path is a no-op cleanup when the KV is
        # inserted unfinished; SWA/mamba trees keep per-request state outside it.
        if tree_supports_mamba or tree_supports_swa:
            raise ValueError(
                "--boost-preempt-wait-s requires a full-attention radix cache "
                "(no SWA or mamba components)"
            )
        if disaggregation_mode != "null":
            raise ValueError(
                "--boost-preempt-wait-s is not supported with PD disaggregation"
            )
        return cls(
            wait_s=wait_s,
            quantum_tokens=quantum_tokens,
            max_per_req=max_per_req,
            dp_rank=dp_rank,
        )

    def maybe_plan(
        self,
        *,
        now_s: float,
        waiting_queue: Sequence[Req],
        running_reqs: Sequence[Req],
        have_tokens: int,
        new_token_ratio: float,
        page_size: int,
        chunked_req: Optional[Req] = None,
    ) -> Optional[DecodePreemptionPlan]:
        if now_s - self._last_plan_s < PREEMPT_COOLDOWN_S:
            return None
        return plan_decode_preemption(
            now_s=now_s,
            waiting_queue=waiting_queue,
            running_reqs=running_reqs,
            have_tokens=have_tokens,
            new_token_ratio=new_token_ratio,
            page_size=page_size,
            wait_s=self.wait_s,
            quantum_tokens=self.quantum_tokens,
            max_per_req=self.max_per_req,
            preferred_rid=self._target_rid,
            chunked_req=chunked_req,
        )

    def on_preempted(
        self, plan: DecodePreemptionPlan, preempted: Sequence[Req], now_s: float
    ) -> None:
        self._last_plan_s = now_s
        self._target_rid = plan.target_rid
        self._target_passes_left = RESERVE_MAX_PASSES
        for req in preempted:
            req.num_preemptions += 1
            req.preempt_stall_start_s = now_s
        self.num_events += 1
        self.num_victims += len(preempted)
        logger.info(
            "decode preemption [dp %d]: waiter %s queued %.1f s needs %d tok "
            "(deficit %d); released %d decode(s) with output %s tok, ~%d tok "
            "freed; totals events=%d victims=%d",
            self.dp_rank,
            plan.target_rid,
            plan.target_wait_s,
            plan.need_tokens,
            plan.deficit_tokens,
            len(preempted),
            list(plan.victim_output_lens),
            plan.freed_tokens,
            self.num_events,
            self.num_victims,
        )

    def promote_target(self, waiting_queue: List[Req]) -> None:
        """Move the reserved waiter to the head of the sorted queue, so the
        released memory goes to it and the admission scan stops at it (the
        misfit break) instead of letting later requests take that memory."""
        if self._target_rid is None:
            return
        idx = next(
            (i for i, r in enumerate(waiting_queue) if r.rid == self._target_rid),
            None,
        )
        self._target_passes_left -= 1
        if idx is None or self._target_passes_left < 0:
            self._target_rid = None
            return
        if idx:
            waiting_queue.insert(0, waiting_queue.pop(idx))

    def on_admitted(self, reqs: Sequence[Req], now_s: float) -> None:
        for req in reqs:
            if req.preempt_stall_start_s is not None:
                req.preempt_stall_s += now_s - req.preempt_stall_start_s
                req.preempt_stall_start_s = None
            req.preempt_resume_output_len = len(req.output_ids)
            if req.rid == self._target_rid:
                self._target_rid = None
