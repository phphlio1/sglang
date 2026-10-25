"""Opt-in per-request schedule evaluation log (``--log-schedule-eval-path``).

One JSON line per finished request, written from the scheduler process and
flushed per result batch. The file stands alone for analysis: it carries its
own cache classification (matched prefix at first/last admission) and queue
wait, so no join with client-side records is needed.

Timestamps: every ``*_s`` duration is a ``time.perf_counter`` delta (the clock
``SchedulerReqTimeStats`` uses); ``arrival_wall_s`` is epoch seconds
(``convert_time_to_realtime`` over ``scheduler_recv_time``) so records can be
aligned with server-metrics scrapes.

``queue_wait_s`` is first admission minus FIRST queue entry: both stamps are
taken once and survive retraction, unlike ``wait_queue_entry_time`` which
re-stamps on every re-entry. ``retraction_count`` on the same record says
whether the wait spans a retraction cycle.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Iterable, Optional, Set, TextIO

from sglang.srt.observability.req_time_stats import convert_time_to_realtime

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


def stamp_admission_prefix_matches(reqs: Iterable[Req]) -> None:
    """Record the admission-time prefix match on each admitted req.

    First admission is stamp-once (a first-attempt hit rate; the client-side
    cache-read metric conflates re-processed hits after retraction); the last
    admission is refreshed on every one. The device/host split and the
    in-batch-dedup follower verdict freeze with the first admission too:
    ``prefix_indices``/``host_hit_length`` at this point come from
    ``init_next_round_input``'s re-match (policy-independent), so the split
    shows the real reuse even under policies that leave
    ``num_matched_prefix_tokens`` at zero.
    """
    for req in reqs:
        if req.matched_prefix_tokens_first_admission is None:
            req.matched_prefix_tokens_first_admission = req.num_matched_prefix_tokens
            req.device_matched_prefix_tokens_first_admission = len(req.prefix_indices)
            req.host_hit_prefix_tokens_first_admission = req.host_hit_length
            req.in_batch_dedup_follower_first_admission = req.in_batch_dedup_pending
        req.matched_prefix_tokens_last_admission = req.num_matched_prefix_tokens


class ScheduleEvalLogger:
    """Appends one JSON line per finished request; never crashes the server.

    An unwritable path logs one warning and disables the logger. The rid set
    keeps a finished req from being written twice if it survives one result
    batch (overlap mode); its size is one entry per finished request.
    """

    def __init__(
        self,
        path: str,
        policy: str,
        boost_gamma: float,
        boost_bin_size: int,
        boost_clock: str,
        boost_cache_alpha: float,
    ) -> None:
        self.path = path
        self.policy = policy
        self.boost_gamma = boost_gamma
        self.boost_bin_size = boost_bin_size
        self.boost_clock = boost_clock
        self.boost_cache_alpha = boost_cache_alpha
        self._seen_rids: Set[str] = set()
        self._file: Optional[TextIO] = None
        try:
            self._file = open(path, "a", encoding="utf-8")
        except OSError as e:
            logger.warning(
                "cannot open --log-schedule-eval-path %s (%s); schedule eval "
                "logging disabled",
                path,
                e,
            )

    def log_finished_batch(self, reqs: Iterable[Req]) -> None:
        if self._file is None:
            return
        try:
            for req in reqs:
                if req.rid in self._seen_rids or not req.finished():
                    continue
                self._seen_rids.add(req.rid)
                self._file.write(json.dumps(self._record(req)) + "\n")
            self._file.flush()
        except OSError as e:
            logger.warning(
                "schedule eval log %s became unwritable (%s); disabling",
                self.path,
                e,
            )
            self.close()

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except OSError:
                pass
            self._file = None

    def _record(self, req: Req) -> dict:
        ts = req.time_stats
        # 0.0 means "never stamped" for the stamp-once perf_counter fields.
        queue_wait_s = (
            ts.forward_entry_time - req.first_queue_entry_time
            if ts.forward_entry_time > 0.0 and req.first_queue_entry_time is not None
            else None
        )
        ttft_s = (
            ts.prefill_finished_time - ts.scheduler_recv_time
            if ts.prefill_finished_time > 0.0 and ts.scheduler_recv_time > 0.0
            else None
        )
        e2e_s = (
            ts.completion_time - ts.scheduler_recv_time
            if ts.completion_time > 0.0 and ts.scheduler_recv_time > 0.0
            else None
        )
        arrival_wall_s = (
            convert_time_to_realtime(ts.scheduler_recv_time)
            if ts.scheduler_recv_time > 0.0
            else None
        )
        return {
            "rid": req.rid,
            "policy": self.policy,
            "gamma": self.boost_gamma,
            # A gamma value alone is unit-ambiguous across clock modes.
            "boost_clock": self.boost_clock,
            "bin_size": self.boost_bin_size,
            # boost-cache arm identity (inert under every other policy).
            "cache_alpha": self.boost_cache_alpha,
            "arrival_wall_s": arrival_wall_s,
            "queue_wait_s": queue_wait_s,
            "ttft_s": ttft_s,
            "e2e_s": e2e_s,
            "input_len": len(req.origin_input_ids),
            "output_len": len(req.output_ids),
            "matched_prefix_tokens_first_admission": (
                req.matched_prefix_tokens_first_admission
            ),
            "matched_prefix_tokens_last_admission": (
                req.matched_prefix_tokens_last_admission
            ),
            "device_matched_prefix_tokens_first_admission": (
                req.device_matched_prefix_tokens_first_admission
            ),
            "host_hit_prefix_tokens_first_admission": (
                req.host_hit_prefix_tokens_first_admission
            ),
            "in_batch_dedup_follower_first_admission": (
                req.in_batch_dedup_follower_first_admission
            ),
            "num_retractions": req.retraction_count,
            "boost_arrival_tokens": req.boost_arrival_tokens,
            "boost_commit_hold_passes": req.boost_commit_hold_passes,
        }
