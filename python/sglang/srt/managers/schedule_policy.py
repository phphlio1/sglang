from __future__ import annotations

import logging
import math
import time
from array import array

from sglang.srt.environ import envs
from sglang.srt.managers.prefill_delayer import PrefillDelayerSinglePassExecutor
from sglang.srt.runtime_context import (
    get_disagg,
    get_exec,
    get_schedule,
)
from sglang.srt.utils import get_bool_env_var, is_gfx95_supported, is_hip

_ROUTING_KEY_POLICY_DEBUG_LOG = get_bool_env_var("SGLANG_ROUTING_KEY_POLICY_DEBUG_LOG")
logger = logging.getLogger(__name__)

# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Request scheduler policy"""

import os
import random
from collections import Counter, deque
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from enum import Enum, auto
from functools import lru_cache
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Set, Union

import torch

from sglang.srt.dllm.config import DllmConfig
from sglang.srt.managers.schedule_batch import (
    Req,
    ScheduleBatch,
    split_cached_prefix_by_tier,
)
from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.page_interleave import page_interleave_shard_size
from sglang.srt.mem_cache.allocator.swa import (
    SWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.unified_hybrid_swa import (
    UnifiedMambaSWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.unified_mamba import (
    UnifiedMambaTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    InitLoadBackParams,
    InsertParams,
    MatchPrefixParams,
    zero_match_result,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode
from sglang.srt.mem_cache.unified_cache.components import (
    CacheTransferPhase,
    ComponentType,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator

# Clip the estimation of max_new_tokens for the request whose max_new_tokens is very large.
# This can prevent the server from being too conservative.
# Note that this only clips the estimation in the scheduler but does not change the stop
# condition. The request can still generate tokens until it hits the unclipped max_new_tokens.
CLIP_MAX_NEW_TOKENS = int(
    os.environ.get("SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION", "4096")
)


@lru_cache(maxsize=1)
def _use_exact_chunk_fill() -> bool:
    """Whether to charge the chunked-prefill compute budget in tokens (gfx95 only).

    Gated on gfx95 because that is where the win is: the aiter absorb bmm picks
    its EVEN_MN specialization from M % BLOCK_SIZE_M, so a short batch costs 2x
    on that kernel, against 1% for the hipBLASLt GEMMs that just run one extra
    partial tile.
    """
    return envs.SGLANG_EXACT_CHUNK_FILL.get() and is_gfx95_supported()


# Threshold for in-batch prefix cache.
# If a request has a matched prefix length (against existing cache) less than this value,
# the scheduler runs the in-batch prefix caching check for this request.
# If we set it to -1, it means we disable in-batch prefix caching.
IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD = int(
    os.environ.get("IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD", "32")
)

# Threshold for in-batch prefix cache.
# If a request has a matched prefix length (within the waiting queue) larger than this value,
# the scheduler deprioritizes this request
IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD = int(
    os.environ.get("IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD", "32")
)


IGNORE_EOS_RESERVE_TOKENS = 1
# AMD/HIP-only: the prefill tile-budget admission control is part of the AMD
# compact extend-attention work and is gated on HIP (see _check_prefill_tile_budget),
# so non-AMD vendors keep the exact legacy scheduler behavior.
_IS_HIP = is_hip()
PREFILL_TILE_BUDGET = envs.SGLANG_PREFILL_TILE_BUDGET.get()
PREFILL_TILE_BUDGET_MODE = envs.SGLANG_PREFILL_TILE_BUDGET_MODE.get().strip().lower()
if PREFILL_TILE_BUDGET_MODE not in {"legacy", "compact"}:
    logger.warning(
        "Unsupported SGLANG_PREFILL_TILE_BUDGET_MODE=%s. Falling back to compact.",
        PREFILL_TILE_BUDGET_MODE,
    )
    PREFILL_TILE_BUDGET_MODE = "compact"


def _ceil_div(value: int, divisor: int) -> int:
    return -(-value // divisor)


def estimate_prefill_extend_tile_metrics(
    extend_lens: List[int], block_m: int
) -> Dict[str, Union[int, float, List[int], None]]:
    """Estimate extend-attention query tiles per head for a prefill batch."""
    normalized_lens = [max(0, int(length)) for length in extend_lens]
    q_tiles = [
        _ceil_div(length, block_m) if length > 0 else 0 for length in normalized_lens
    ]
    legacy_tiles = len(q_tiles) * max(q_tiles) if q_tiles else 0
    compact_tiles = sum(q_tiles)
    saved_tiles = legacy_tiles - compact_tiles
    saved_ratio = saved_tiles / legacy_tiles if legacy_tiles else None
    return {
        "block_m": int(block_m),
        "request_count": len(normalized_lens),
        "extend_lens": normalized_lens,
        "q_tiles_per_request": q_tiles,
        "max_extend_len": max(normalized_lens) if normalized_lens else 0,
        "sum_extend_len": sum(normalized_lens),
        "legacy_q_tiles_per_head": legacy_tiles,
        "compact_q_tiles_per_head": compact_tiles,
        "saved_q_tiles_per_head": saved_tiles,
        "saved_q_tile_ratio": saved_ratio,
    }


def match_prefix_for_req(
    tree_cache: BasePrefixCache,
    req: Req,
    token_ids: Optional[array[int]] = None,
    *,
    cow_mamba: bool = False,
    include_req: bool = False,
    max_prefix_len: Optional[int] = None,
):
    if token_ids is None:
        token_ids = req.origin_input_ids + req.output_ids

    # unified_kv SWA lives in a per-request ring that's not content-stable and is
    # never stored in the radix tree, so a reused prefix carries stale SWA. Cap
    # the match by the trailing sliding window so it gets re-prefilled, rewriting
    # this request's SWA ring. No-op for other layouts.
    reprefill_tail = tree_cache.swa_reprefill_tail_tokens()
    key_limit = max(0, len(token_ids) - reprefill_tail) if reprefill_tail else None
    if max_prefix_len is not None:
        key_limit = (
            max_prefix_len if key_limit is None else min(key_limit, max_prefix_len)
        )

    match_result = tree_cache.match_prefix(
        MatchPrefixParams(
            key=RadixKey(
                token_ids=token_ids,
                extra_key=req.extra_key,
                limit=key_limit,
                cache_salt=req.cache_salt,
            ),
            cow_mamba=cow_mamba,
            req=req if include_req else None,
        )
    )
    if envs.SGLANG_RADIX_FORCE_MISS.get():
        match_result = zero_match_result(
            tree_cache, match_result, extra_key=req.extra_key
        )
    (
        req.prefix_indices,
        req.last_node,
        req.last_host_node,
        req.best_match_node,
        req.host_hit_length,
        req.swa_host_hit_length,
        req.mamba_host_hit_length,
    ) = (
        match_result.device_indices,
        match_result.last_device_node,
        match_result.last_host_node,
        match_result.best_match_node,
        match_result.host_hit_length,
        match_result.swa_host_hit_length,
        match_result.mamba_host_hit_length,
    )
    max_len = req._compute_max_prefix_len(len(token_ids))
    req.num_matched_prefix_tokens = min(
        len(req.prefix_indices) + req.host_hit_length, max_len
    )
    req.swa_branching_seqlen = match_result.swa_branching_seqlen
    if match_result.mamba_branching_seqlen is not None:
        req.mamba_branching_seqlen = match_result.mamba_branching_seqlen
    if match_result.cache_protected_len is not None:
        req.kv.cache_protected_len = match_result.cache_protected_len
    return match_result


def gamma_boost(work: float, gamma: float) -> float:
    """Gamma-boost priority boost (UniBoost, Li et al., ICML 2026, Eq. 2).

    b(w) = -ln(1 - exp(-gamma * w)) / gamma, strictly decreasing in w: small
    remaining work gets a large boost, large work gets none. Combined with
    geometric_bin, b(bin_size) is the maximum boost any request can receive,
    which bounds how far a newcomer may jump ahead of an already-waiting
    request (the anti-starvation property).
    """
    x = gamma * work
    if x <= 0:
        # Unreachable from _sort_by_boost (work >= 1 is binned to >= bin_size
        # and gamma is validated > 0); kept for direct callers.
        return float("inf")
    v = -math.expm1(-x)  # 1 - exp(-x), in (0, 1]
    if v >= 1.0:
        # x large: 1 - exp(-x) rounds to 1.0 and the boost underflows to 0.
        return 0.0
    return -math.log(v) / gamma


def geometric_bin(work: int, bin_size: int) -> int:
    """Quantize work to the geometric ladder {k, 2k, 4k, ...} (Eq. 8).

    q(u) = k * 2^floor(log2(max(u, k) / k)); u <= k floors to k. This caps the
    boost at b(k), makes priority change only at bin boundaries, and leaves
    all requests with u <= k tied at the max boost (FCFS among them via the
    arrival stamp).
    """
    return bin_size * 2 ** math.floor(math.log2(max(work, bin_size) / bin_size))


def _percentile(sorted_vals: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile; same definition as the analysis-side
    gamma-hat estimator, so runtime and offline gammas agree."""
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * p / 100.0
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])


class BoostClock(Enum):
    """Arrival/work clock for the boost policy; see --boost-clock."""

    TOKENS = "tokens"  # counted-token clock (paper-exact)
    WALL = "wall"  # wall seconds; work via measured token-clock rate


class TokenClockRate:
    """EMA of the token-clock rate (counted prefill tokens per wall second).

    Wall-clock boost converts uncached work to seconds-of-service with this
    rate. The counted stream includes linear replay of matched prefixes (and
    mixed-chunk decode tokens), so the rate absorbs the cache-state swing in
    the counter; the conversion is exact where the counter is prefill-pure.
    """

    BETA = 0.2

    def __init__(self) -> None:
        self._last_tokens: Optional[int] = None
        self._last_t: Optional[float] = None
        self._tokens_per_s: Optional[float] = None

    @property
    def tokens_per_s(self) -> Optional[float]:
        return self._tokens_per_s

    def observe(self, processed_tokens: int, now_s: float) -> Optional[float]:
        """Fold one (counter, wall) sample into the EMA; returns the rate
        (None until the first positive-delta interval is seen)."""
        if self._last_tokens is None:
            self._last_tokens = processed_tokens
            self._last_t = now_s
            return self._tokens_per_s
        delta_tokens = processed_tokens - self._last_tokens
        delta_s = now_s - self._last_t
        if delta_tokens > 0 and delta_s > 0:
            # No sample and no baseline advance on nonpositive deltas: an
            # idle or decode-only stretch folds into the next sample's
            # window, keeping the rate load-inclusive.
            sample = delta_tokens / delta_s
            self._last_tokens = processed_tokens
            self._last_t = now_s
            if self._tokens_per_s is None:
                self._tokens_per_s = sample
            else:
                self._tokens_per_s = (
                    1.0 - self.BETA
                ) * self._tokens_per_s + self.BETA * sample
        return self._tokens_per_s


class AdaptiveBoostGamma:
    """Adaptive gamma for the boost policy (gamma-Ada, UniBoost Eq. 9).

    gamma_hat = ln(0.05/0.01) / (t99 - t95) = ln 5 / tail_width: the
    log-linear tail-slope fit of the completed-request E2E latency CCDF in
    the [t95, t99] band, in wall seconds. The fitted variable is response
    time, not queue wait: waits are endogenous to the policy's own ordering
    (fat wait tail -> smaller gamma -> longer giant waits -> fatter tail),
    while response time keeps the exogenous service component in the fit.
    Wall mode only -- E2E latency is wall seconds and gamma is then in 1/s
    (--boost-adaptive-gamma requires --boost-clock wall).

    The low clip is a max-jump bound, not a raw gamma: gamma is floored so
    the largest boost any newcomer can receive, b_gamma(bin_size / rate),
    stays <= max_jump_s wall seconds. "How far a newcomer may cut ahead" is
    workload- and CCU-independent; a raw gamma floor is not (its starvation
    effect scales with the service rate). Measured basis (E2 c16 sweep,
    2026-10-13): jump ~50 s (gamma 0.09, k 256, ~2k tok/s) holds
    cold-request stretch at the lpm level; jump ~101 s (gamma 0.042)
    doubles it.
    """

    WINDOW = 128
    MIN_SAMPLES = 32
    BETA = 0.2
    # Outer clip in 1/s; the jump floor usually binds first on the low side.
    GAMMA_MIN_S = 1e-3
    GAMMA_MAX_S = 15.0
    LN5 = math.log(5.0)
    # epsilon floor on the band width: a compressed tail (t99 == t95) is the
    # degenerate thin tail (FCFS-ward, gamma_hat -> GAMMA_MAX_S), not an error.
    TAIL_EPS_S = 1e-6
    # Trajectory log lines: at most one per LOG_MIN_UPDATES updates, or on a
    # >= LOG_REL_CHANGE move since the last logged gamma.
    LOG_MIN_UPDATES = 64
    LOG_REL_CHANGE = 0.25

    def __init__(
        self,
        gamma0: float,
        bin_size: int,
        max_jump_s: float,
        window: int = WINDOW,
        min_samples: int = MIN_SAMPLES,
    ) -> None:
        self._samples: deque = deque(maxlen=window)
        self._min_samples = min_samples
        self._bin_size = bin_size
        self._max_jump_s = max_jump_s
        self._gamma = min(max(gamma0, self.GAMMA_MIN_S), self.GAMMA_MAX_S)
        self._updates_since_log = 0
        self._last_logged_gamma = self._gamma

    @property
    def gamma(self) -> float:
        return self._gamma

    def record_latencies(
        self,
        latencies_s: Iterable[float],
        rate_tokens_per_s: Optional[float],
        works: Optional[Iterable[int]] = None,
    ) -> Optional[float]:
        """Feed completed-request E2E latencies (wall seconds); returns the
        updated gamma when this call applied an update, None while the
        window is still filling below min_samples.

        When works is provided, uses a size-aware tail fit: computes gamma_hat
        separately for short and long requests (split at median work). If long
        requests have a fatter tail (giant starvation), uses the short-request
        gamma_hat instead of the overall -- the paper's tail fit assumes the
        fat tail comes from short-request starvation, which is inverted under
        overload where giants are starved by the policy itself."""
        lat_list = [max(0.0, float(t)) for t in latencies_s]
        if works is not None:
            work_list = list(works)
            self._samples.extend(zip(lat_list, work_list))
        else:
            self._samples.extend((lat, 0) for lat in lat_list)
        if len(self._samples) < self._min_samples:
            return None

        gamma_floor = self._gamma_floor(rate_tokens_per_s)
        gamma_hat, raw_fit, fit_src = self._size_aware_gamma_hat(gamma_floor)
        gamma = min(
            max((1.0 - self.BETA) * self._gamma + self.BETA * gamma_hat, gamma_floor),
            self.GAMMA_MAX_S,
        )
        self._maybe_log(gamma, raw_fit, fit_src, gamma_floor, len(self._samples))
        self._gamma = gamma
        return gamma

    def _size_aware_gamma_hat(self, gamma_floor: float) -> tuple[float, float, str]:
        """Overall tail fit (paper Eq. 9), corrected for giant starvation.

        Under overload, giants starved by the policy inflate the overall tail,
        driving gamma_hat toward zero (positive feedback). Splitting at median
        work isolates the short-request tail, which reflects the exogenous
        service variance the paper's fit targets -- not the policy-induced
        giant starvation.

        Returns the floored estimate, the raw pre-floor fit, and which tail
        produced it ("overall" or "short") for the trajectory log.
        """
        lats = [s[0] for s in self._samples]
        works = [s[1] for s in self._samples]
        gamma_overall = self._tail_gamma_hat(lats)
        fallback = min(max(gamma_overall, gamma_floor), self.GAMMA_MAX_S)

        if len(works) < 2 * self._min_samples or all(w == works[0] for w in works):
            return fallback, gamma_overall, "overall"

        sorted_works = sorted(works)
        median_w = sorted_works[(len(sorted_works) - 1) // 2]
        short_lats = [l for l, w in self._samples if w <= median_w]
        long_lats = [l for l, w in self._samples if w > median_w]
        if len(short_lats) < self._min_samples // 2 or len(long_lats) < self._min_samples // 2:
            return fallback, gamma_overall, "overall"

        gamma_short = self._tail_gamma_hat(short_lats)
        gamma_long = self._tail_gamma_hat(long_lats)

        # Giant starvation: long tail fatter than short (gamma_long < gamma_short).
        # Use gamma_short to ignore the policy-induced giant tail.
        if gamma_long < gamma_short:
            raw_fit, fit_src = gamma_short, "short"
        else:
            raw_fit, fit_src = gamma_overall, "overall"
        return min(max(raw_fit, gamma_floor), self.GAMMA_MAX_S), raw_fit, fit_src

    @staticmethod
    def _tail_gamma_hat(lats: list[float]) -> float:
        """Paper Eq. 9: ln(5) / (t99 - t95), floored at TAIL_EPS_S."""
        ordered = sorted(lats)
        delta = max(
            _percentile(ordered, 99) - _percentile(ordered, 95),
            AdaptiveBoostGamma.TAIL_EPS_S,
        )
        return AdaptiveBoostGamma.LN5 / delta

    def _gamma_floor(self, rate_tokens_per_s: Optional[float]) -> float:
        """Smallest gamma whose max boost b_gamma(bin_size / rate) fits
        within max_jump_s. b_gamma(tau) = tau * g(gamma * tau) with
        g(u) = -ln(1 - exp(-u)) / u strictly decreasing, so the floor is the
        root of g(u) = max_jump_s / tau at u = gamma * tau, found by
        bisection on u. Rate unmeasured -> the outer floor: the sort gives
        boost 0 (pure arrival order) while the rate is None, so no jump
        exists to bound."""
        if rate_tokens_per_s is None or rate_tokens_per_s <= 0.0:
            return self.GAMMA_MIN_S
        tau_s = self._bin_size / rate_tokens_per_s
        budget = self._max_jump_s / tau_s
        lo, hi = 1e-9, 1e3  # g(1e-9) ~ 2e10 > any budget; g(1e3) ~ 0
        for _ in range(64):
            mid = math.sqrt(lo * hi)
            if -math.log(-math.expm1(-mid)) / mid > budget:
                lo = mid
            else:
                hi = mid
        return hi / tau_s

    def _maybe_log(
        self,
        gamma: float,
        raw_fit: float,
        fit_src: str,
        gamma_floor: float,
        n: int,
    ) -> None:
        self._updates_since_log += 1
        rel = abs(gamma - self._last_logged_gamma) / self._last_logged_gamma
        if (
            self._updates_since_log >= self.LOG_MIN_UPDATES
            or rel >= self.LOG_REL_CHANGE
        ):
            logger.info(
                "boost adaptive gamma: gamma=%.3e (fit=%.3e from %s tail, "
                "width=%.3g s, floor=%.3e, n=%d)",
                gamma,
                raw_fit,
                fit_src,
                self.LN5 / raw_fit,
                gamma_floor,
                n,
            )
            self._updates_since_log = 0
            self._last_logged_gamma = gamma


class CacheAwarePolicy(Enum):
    """Scheduling policies that are aware of the tree cache."""

    LPM = "lpm"  # longest prefix match
    DFS_WEIGHT = "dfs-weight"  # depth-first search weighting
    HRRN = "hrrn"  # highest response ratio next, token-based aging
    SHORTEST_PREFILL_FIRST = "shortest-prefill-first"
    BOOST = "boost"  # gamma-boosted arrival time (anti-starvation)


class CacheAgnosticPolicy(Enum):
    """Scheduling policies that are not aware of the tree cache."""

    FCFS = "fcfs"  # first come first serve
    # FCFS ordering with the sort-time prefix match forced on: populates
    # num_matched_prefix_tokens/prefix_indices for caches whose
    # supports_fast_match_prefix() is False (every current implementation),
    # where plain fcfs leaves them zero. Ordering is identical to fcfs.
    FCFS_MATCH = "fcfs-match"
    LOF = "lof"  # longest output first
    RANDOM = "random"
    ROUTING_KEY = "routing-key"  # prioritize by routing key frequency in running batch


class SchedulePolicy:
    Policy = Union[CacheAwarePolicy, CacheAgnosticPolicy]

    def __init__(
        self,
        policy: str,
        tree_cache: BasePrefixCache,
        enable_hierarchical_cache: bool,
        enable_priority_scheduling: bool,
        schedule_low_priority_values_first: bool,
        boost_gamma: float = 1e-4,
        boost_bin_size: int = 256,
        boost_adaptive_gamma: bool = False,
        boost_adaptive_max_jump_s: float = 60.0,
        boost_clock: str = "tokens",
        in_batch_prefix_caching_max_pin_s: float = 60.0,
    ):
        self.policy = self._validate_and_adjust_policy(policy, tree_cache)
        self.tree_cache = tree_cache
        self.enable_hierarchical_cache = enable_hierarchical_cache
        self.enable_priority_scheduling = enable_priority_scheduling
        self.schedule_low_priority_values_first = schedule_low_priority_values_first
        self.priority_sign = 1 if schedule_low_priority_values_first else -1
        self.boost_gamma = boost_gamma
        self.boost_bin_size = boost_bin_size
        self.boost_clock = BoostClock(boost_clock)
        self.in_batch_prefix_caching_max_pin_s = in_batch_prefix_caching_max_pin_s
        # Wall-mode token-clock rate EMA; None in tokens mode and for
        # non-boost policies (fixed-gamma tokens path stays bit-identical).
        self.prefill_token_rate: Optional[TokenClockRate] = (
            TokenClockRate()
            if self.boost_clock is BoostClock.WALL
            and self.policy == CacheAwarePolicy.BOOST
            else None
        )
        # gamma-Ada (Eq. 9) fits wall-second E2E latencies, so it is
        # wall-mode only; None unless boost + --boost-adaptive-gamma (off by
        # default, so the fixed-gamma path is bit-identical to before).
        if (
            boost_adaptive_gamma
            and self.policy == CacheAwarePolicy.BOOST
            and self.boost_clock is not BoostClock.WALL
        ):
            raise ValueError(
                "--boost-adaptive-gamma requires --boost-clock wall "
                "(the E2E-latency tail fit is wall seconds)"
            )
        self.adaptive_gamma: Optional[AdaptiveBoostGamma] = (
            AdaptiveBoostGamma(
                boost_gamma,
                boost_bin_size,
                boost_adaptive_max_jump_s,
            )
            if boost_adaptive_gamma and self.policy == CacheAwarePolicy.BOOST
            else None
        )
        self._shortest_prefill_calls = 0

        # It is used to find the matching prefix for in-batch prefix caching.
        self.waiting_queue_radix_tree = RadixCache.create_simulated()

    def calc_priority(
        self,
        waiting_queue: List[Req],
        running_batch: Optional[ScheduleBatch] = None,
        processed_tokens: int = 0,
    ) -> None:
        # Before _determine_active_policy so the >128-queue FCFS fallback
        # still refreshes the rate between wall-mode sorts.
        if self.prefill_token_rate is not None:
            self.prefill_token_rate.observe(processed_tokens, time.perf_counter())
        policy = self._determine_active_policy(waiting_queue)

        # Populate req.num_matched_prefix_tokens at schedule time. Cache-aware policies
        # set it in _compute_prefix_matches; do the same full match for
        # cache-agnostic policies when the radix supports it, so the load
        # snapshot has it. Skip on decode (never prefills). fcfs-match forces
        # the match on regardless: it is the sort-time-match reference arm for
        # caches where supports_fast_match_prefix() is False (there the plain
        # cache-agnostic branch never fires; admission reports its own match
        # via init_next_round_input).
        if (
            not isinstance(policy, CacheAwarePolicy)
            and (
                policy == CacheAgnosticPolicy.FCFS_MATCH
                or self.tree_cache.supports_fast_match_prefix()
            )
            and get_disagg().disaggregation_mode != "decode"
        ):
            for r in waiting_queue:
                match_prefix_for_req(self.tree_cache, r, include_req=True)

        if self.policy in (CacheAgnosticPolicy.FCFS, CacheAgnosticPolicy.FCFS_MATCH):
            if self.enable_priority_scheduling:
                SchedulePolicy._sort_by_priority_and_fcfs(
                    waiting_queue, self.priority_sign
                )
            return

        if isinstance(policy, CacheAwarePolicy):
            temporary_deprioritized = self._compute_prefix_matches(
                waiting_queue, policy
            )
            for r in waiting_queue:
                r.in_batch_dedup_pending = r.rid in temporary_deprioritized
            if policy == CacheAwarePolicy.LPM:
                SchedulePolicy._sort_by_longest_prefix(
                    waiting_queue, temporary_deprioritized
                )
            elif policy == CacheAwarePolicy.DFS_WEIGHT:
                SchedulePolicy._sort_by_dfs_weight(waiting_queue, self.tree_cache)
            elif policy == CacheAwarePolicy.HRRN:
                SchedulePolicy._sort_by_hrrn(
                    waiting_queue, temporary_deprioritized, processed_tokens
                )
            elif policy == CacheAwarePolicy.BOOST:
                SchedulePolicy._sort_by_boost(
                    waiting_queue,
                    temporary_deprioritized,
                    self.boost_gamma,
                    self.boost_bin_size,
                    self.enable_priority_scheduling,
                    self.priority_sign,
                    clock=self.boost_clock,
                    rate_tokens_per_s=(
                        self.prefill_token_rate.tokens_per_s
                        if self.prefill_token_rate is not None
                        else None
                    ),
                )
            elif policy == CacheAwarePolicy.SHORTEST_PREFILL_FIRST:
                SchedulePolicy._sort_by_shortest_prefill(
                    waiting_queue, temporary_deprioritized
                )
                self._shortest_prefill_calls += 1
                if waiting_queue and self._shortest_prefill_calls % 128 == 1:
                    logger.info(
                        "Experimental shortest-prefill-first: queue=%d shortest_uncached=%d",
                        len(waiting_queue),
                        self._shortest_prefill_work(waiting_queue[0]),
                    )
            else:
                raise ValueError(f"Unknown CacheAware Policy: {policy=}")
        else:
            # No in-batch dedup classification on this path (agnostic policy or
            # the >128-queue fallback); clear stale flags so a first-admission
            # stamp never reads an earlier sort's verdict.
            for r in waiting_queue:
                r.in_batch_dedup_pending = False
            if policy == CacheAgnosticPolicy.FCFS:
                pass
            elif policy == CacheAgnosticPolicy.LOF:
                SchedulePolicy._sort_by_longest_output(
                    waiting_queue,
                    self.enable_priority_scheduling,
                    self.priority_sign,
                )
            elif policy == CacheAgnosticPolicy.RANDOM:
                SchedulePolicy._sort_randomly(waiting_queue)
            elif policy == CacheAgnosticPolicy.ROUTING_KEY:
                if running_batch is not None:
                    SchedulePolicy._sort_by_routing_key(waiting_queue, running_batch)
            else:
                raise ValueError(f"Unknown CacheAgnostic Policy: {policy=}")

    def _determine_active_policy(self, waiting_queue: List[Req]) -> Policy:
        if (
            self.policy
            in (
                CacheAwarePolicy.LPM,
                CacheAwarePolicy.HRRN,
            )
            and len(waiting_queue) > 128
        ):
            # Turn off the expensive prefix matching and sorting when the #queue is large.
            return CacheAgnosticPolicy.FCFS
        return self.policy

    def record_completed_latencies(self, reqs: Sequence[Req]) -> Optional[float]:
        """Feed the E2E latencies (scheduler_recv -> completion, wall
        seconds) of the requests finished among reqs to gamma-Ada; returns
        the updated gamma when it moved (the caller mirrors it into the
        eval log), None on every other path. No-op unless boost + adaptive
        gamma."""
        if self.adaptive_gamma is None:
            return None
        latencies = []
        works = []
        for r in reqs:
            ts = r.time_stats
            # Unstamped (0.0) sides mean the req never reached the scheduler
            # or the completion stamp -- the eval log's e2e_s guard, so the
            # runtime fit and the offline gamma agree on the population.
            if (
                not r.finished()
                or ts.completion_time <= 0.0
                or ts.scheduler_recv_time <= 0.0
            ):
                continue
            latencies.append(ts.completion_time - ts.scheduler_recv_time)
            works.append(len(r.origin_input_ids) + len(r.output_ids))
        if not latencies:
            return None
        gamma = self.adaptive_gamma.record_latencies(
            latencies,
            rate_tokens_per_s=(
                self.prefill_token_rate.tokens_per_s
                if self.prefill_token_rate is not None
                else None
            ),
            works=works,
        )
        if gamma is not None:
            self.boost_gamma = gamma
        return gamma

    def waiting_queue_prefix_matched(self, waiting_queue: List[Req]) -> bool:
        policy = self._determine_active_policy(waiting_queue)
        return (
            isinstance(policy, CacheAwarePolicy)
            or policy == CacheAgnosticPolicy.FCFS_MATCH
            or self.tree_cache.supports_fast_match_prefix()
        )

    def _validate_and_adjust_policy(
        self, policy: str, tree_cache: BasePrefixCache
    ) -> Policy:
        """
        Validates the policy and adjusts it if necessary based on tree cache settings.
        """
        try:
            policy_enum = CacheAwarePolicy(policy)
            if getattr(tree_cache, "disable", True):
                if policy_enum == CacheAwarePolicy.SHORTEST_PREFILL_FIRST:
                    raise ValueError(
                        "Experimental shortest-prefill-first requires prefix caching"
                    )
                if policy_enum == CacheAwarePolicy.BOOST:
                    logger.warning(
                        "schedule_policy=boost requires prefix caching; "
                        "falling back to fcfs"
                    )
                # If tree_cache is disabled, using CacheAgnosticPolicy policy
                return CacheAgnosticPolicy.FCFS
            return policy_enum
        except ValueError:
            try:
                agnostic_enum = CacheAgnosticPolicy(policy)
            except ValueError:
                raise ValueError(f"Unknown schedule_policy: {policy=}")
            if agnostic_enum == CacheAgnosticPolicy.FCFS_MATCH and tree_cache.disable:
                logger.warning(
                    "schedule_policy=fcfs-match requires prefix caching; "
                    "falling back to fcfs"
                )
                return CacheAgnosticPolicy.FCFS
            return agnostic_enum

    def _compute_prefix_matches(
        self, waiting_queue: List[Req], policy: CacheAwarePolicy
    ) -> Set[int]:
        """
        Computes and caches the matching prefixes for requests in the waiting queue,
            and handles in-batch prefix caching logic.
        """
        temporary_deprioritized: Set[int] = set()
        self.waiting_queue_radix_tree.reset()
        now_s = time.perf_counter()

        for r in waiting_queue:
            prefix_ids = r.origin_input_ids + r.output_ids
            extra_key = r.extra_key
            cache_salt = r.cache_salt
            match_result = match_prefix_for_req(
                self.tree_cache, r, prefix_ids, include_req=True
            )

            # NOTE(sang): This logic is for in-batch prefix caching;
            # If there are more than 1 request that have small matching prefix from
            # existing cache, but all those requests share the same prefix, we prefer
            # to schedule only one of them so that we can increase the cache hit rate.
            # We prefer to set IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD > 0 because too small
            # threshold means we cannot use in-batch prefix caching for short prefixes.
            # It is kind of common when the engine is long running (e.g., imagine the prefix "the").
            if len(r.prefix_indices) <= IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD:
                match_result = self.waiting_queue_radix_tree.match_prefix(
                    MatchPrefixParams(
                        key=RadixKey(
                            token_ids=prefix_ids,
                            extra_key=extra_key,
                            cache_salt=cache_salt,
                        )
                    )
                )
                if envs.SGLANG_RADIX_FORCE_MISS.get():
                    match_result = zero_match_result(
                        self.waiting_queue_radix_tree, match_result, extra_key=extra_key
                    )
                in_batch_matching_prefixes = match_result.device_indices
                if len(in_batch_matching_prefixes) >= (
                    IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD
                ) and not self._pin_hold_expired(r, now_s):
                    temporary_deprioritized.add(r.rid)
                else:
                    # Insert with a dummy key
                    self.waiting_queue_radix_tree.insert(
                        InsertParams(
                            key=RadixKey(
                                token_ids=prefix_ids,
                                extra_key=extra_key,
                                cache_salt=cache_salt,
                            ),
                            value=torch.empty(len(prefix_ids), dtype=torch.bool),
                        )
                    )
        return temporary_deprioritized

    def _pin_hold_expired(self, r: Req, now_s: float) -> bool:
        """Whether the follower pin's wall-second hold bound has released r
        (total queue wait since first entry; --in-batch-prefix-caching-max-pin-s)."""
        if self.in_batch_prefix_caching_max_pin_s <= 0:
            return False
        # None = never stamped (disagg/synthetic); keep the pin, old behavior.
        if r.first_queue_entry_time is None:
            return False
        return now_s - r.first_queue_entry_time > self.in_batch_prefix_caching_max_pin_s

    @staticmethod
    def _shortest_prefill_work(r: Req) -> int:
        return max(
            1,
            len(r.origin_input_ids) + len(r.output_ids) - r.num_matched_prefix_tokens,
        )

    @staticmethod
    def _sort_by_shortest_prefill(
        waiting_queue: List[Req], temporary_deprioritized: Set[int]
    ) -> None:
        # Prioritize short uncached prefills while deferring duplicate prefixes.
        waiting_queue.sort(
            key=lambda r: (
                r.rid in temporary_deprioritized,
                SchedulePolicy._shortest_prefill_work(r),
                r.time_stats.wait_queue_entry_time,
            )
        )

    def shortest_prefill_chunk_limit(
        self, chunked_req: Req, waiting_queue: List[Req], budget: int, page_size: int
    ) -> Optional[int]:
        """Cap the active prefill chunk to reserve tokens for shorter waiting requests."""
        if (
            self.policy
            not in (CacheAwarePolicy.SHORTEST_PREFILL_FIRST, CacheAwarePolicy.BOOST)
            or budget < 2 * page_size
        ):
            return None
        remaining = len(chunked_req.full_untruncated_fill_ids) - len(
            chunked_req.prefix_indices
        )
        reserved = 0
        for req in waiting_queue:
            work = self._shortest_prefill_work(req)
            charge = _ceil_div(work, page_size) * page_size
            if work >= remaining or reserved + charge > budget - page_size:
                break
            reserved += charge
        if not reserved:
            return None
        # Page alignment keeps continuation boundaries allocator-compatible.
        return (budget - reserved) // page_size * page_size

    @staticmethod
    def _sort_by_longest_prefix(
        waiting_queue: List[Req], temporary_deprioritized: Set[int]
    ) -> None:
        """Sorts the waiting queue based on the longest prefix match."""
        waiting_queue.sort(
            key=lambda r: (
                -r.num_matched_prefix_tokens
                if r.rid not in temporary_deprioritized
                else float("inf")
            )
        )

    @staticmethod
    def _uncached_len(r: Req) -> int:
        """Number of tokens that must actually be prefilled for this req
        (all cache levels — device + host via hicache — counted as cached)."""
        return max(0, len(r.origin_input_ids) - r.num_matched_prefix_tokens)

    @staticmethod
    def _sort_by_hrrn(
        waiting_queue: List[Req],
        temporary_deprioritized: Set[int],
        processed_tokens: int,
    ) -> None:
        """Highest Response Ratio Next, with token-based aging.

        Equivalence with classic HRRN when throughput is constant:
            ratio = 1 + wait_sec / est_prefill_time
                  = 1 + (processed_tokens - arrival_processed_tokens) / uncached

        Caller (Scheduler) contract:
          - Maintain a monotonically increasing counter of prefill tokens processed so far
            (accumulate batch.extend_num_tokens per forward). Pass it in as `processed_tokens`.
          - Snapshot `req.arrival_processed_tokens = counter` when the req enters waiting_queue
            (pop_bootstrapped for disagg prefill, _add_request_to_queue for unified).

        Call sites that omit `processed_tokens` (dllm, disagg decode)
        degrade to rid-lexicographic order; those queues carry no prefill work.
        """

        def _key(r: Req):
            rid = r.rid
            if rid in temporary_deprioritized:
                return (float("inf"), rid)
            uncached = SchedulePolicy._uncached_len(r)
            if uncached <= 0:
                # No prefill work; drain immediately.
                return (-float("inf"), rid)
            waited_tokens = max(0, processed_tokens - r.arrival_processed_tokens)
            ratio_delta = waited_tokens / uncached
            return (-ratio_delta, rid)

        waiting_queue.sort(key=_key)

    @staticmethod
    def _sort_by_boost(
        waiting_queue: List[Req],
        temporary_deprioritized: Set[int],
        gamma: float,
        bin_size: int,
        enable_priority_scheduling: bool = False,
        priority_sign: int = -1,
        *,
        clock: BoostClock = BoostClock.TOKENS,
        rate_tokens_per_s: Optional[float] = None,
    ) -> None:
        """Gamma-boosted arrival ordering (UniBoost, Li et al., ICML 2026, Sec. 2).

        phi(r) = A(r) - b_gamma(q(u(r))), served in ascending phi.
        A(r): token-clock stamp at first queue entry (survives retraction), or
              the wall first-entry stamp under clock=WALL.
        u(r): uncached work = input + generated - matched prefix tokens
              (device + host hits; retracted requests charged for re-prefill).
        q():  geometric binning at bin_size; caps max boost at b_gamma(bin_size).
        Wall mode converts work to seconds-of-service via rate_tokens_per_s
        (gamma in 1/s); unmeasured rate -> boost 0 (pure arrival order).

        Anti-starvation: a newcomer can only jump ahead by its boost, which is
        bounded by b_gamma(bin_size) clock-tokens (or b_gamma(bin_size / rate)
        wall seconds); requests older than that are unreachable — bounded
        displacement, converging to FCFS.
        """

        def key(r: Req):
            u = SchedulePolicy._shortest_prefill_work(r)
            w = geometric_bin(u, bin_size)
            if clock is BoostClock.WALL:
                # None = never stamped (synthetic reqs); treat as oldest.
                arrival_s = (
                    r.first_queue_entry_time
                    if r.first_queue_entry_time is not None
                    else 0.0
                )
                boost_s = (
                    gamma_boost(w / rate_tokens_per_s, gamma)
                    if rate_tokens_per_s is not None
                    else 0.0
                )
                rank = arrival_s - boost_s
            else:
                # None = never stamped (synthetic reqs); treat as oldest.
                arrival = (
                    r.boost_arrival_tokens if r.boost_arrival_tokens is not None else 0
                )
                rank = arrival - gamma_boost(w, gamma)
            if r.rid in temporary_deprioritized:
                # In-batch dedup followers still go last, same as lpm.
                group, phi = 1, 0.0
            else:
                group, phi = 0, rank
            if enable_priority_scheduling:
                return (r.priority * priority_sign, group, phi, r.rid)
            return (group, phi, r.rid)

        waiting_queue.sort(key=key)

    @staticmethod
    def _sort_by_dfs_weight(
        waiting_queue: List[Req], tree_cache: BasePrefixCache
    ) -> None:
        """Sorts the waiting queue based on a depth-first search weighting."""
        order = tree_cache.dfs_weight_order([req.last_node for req in waiting_queue])
        waiting_queue[:] = [waiting_queue[index] for index in order]

    @staticmethod
    def _sort_by_longest_output(
        waiting_queue: List[Req],
        enable_priority_scheduling: bool,
        priority_sign: int,
    ) -> None:
        """Sorts the waiting queue based on the longest output (max_new_tokens). If using priority scheduling, sort by priority first."""
        if enable_priority_scheduling:
            waiting_queue.sort(
                key=lambda x: (
                    x.priority * priority_sign,
                    -x.sampling_params.max_new_tokens,
                )
            )
        else:
            waiting_queue.sort(key=lambda x: -x.sampling_params.max_new_tokens)

    @staticmethod
    def _sort_randomly(waiting_queue: List[Req]) -> None:
        """Shuffles the waiting queue randomly."""
        random.shuffle(waiting_queue)

    @staticmethod
    def _sort_by_priority_and_fcfs(
        waiting_queue: List[Req], priority_sign: int
    ) -> None:
        """Sorts the waiting queue based on the request priority then received titmestamp."""
        waiting_queue.sort(
            key=lambda x: (
                x.priority * priority_sign,
                x.time_stats.wait_queue_entry_time,
            )
        )

    @staticmethod
    def _sort_by_routing_key(
        waiting_queue: List[Req], running_batch: ScheduleBatch
    ) -> None:
        """Sorts waiting queue by routing key frequency in running batch."""
        routing_key_counts = Counter(
            r.routing_key for r in running_batch.reqs if r.routing_key
        )

        if _ROUTING_KEY_POLICY_DEBUG_LOG:
            waiting_keys_before = [r.routing_key for r in waiting_queue]
            logger.info(
                f"routing_key_counts={dict(routing_key_counts)}, "
                f"waiting_keys_before={waiting_keys_before}"
            )

        if not routing_key_counts:
            return

        def sort_key(req: Req):
            key = req.routing_key
            if key and key in routing_key_counts:
                count = routing_key_counts[key]
                return (0, -count, key)
            else:
                return (1, 0, key or "")

        waiting_queue.sort(key=sort_key)

        if _ROUTING_KEY_POLICY_DEBUG_LOG:
            waiting_keys_after = [r.routing_key for r in waiting_queue]
            logger.info(f"waiting_keys_after={waiting_keys_after}")


class AddReqResult(Enum):
    CONTINUE = auto()  # Continue to add requests
    NO_TOKEN = auto()  # No token left
    OTHER = auto()  # Other reasons to stop adding requests


@dataclass(frozen=True, slots=True)
class _PrefillAdmission:
    prefix_len: int
    extend_len: int
    max_new_tokens: int
    is_chunked: bool


class PrefillAdder:
    def __init__(
        self,
        page_size: int,
        tree_cache: BasePrefixCache,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        running_batch: ScheduleBatch,
        new_token_ratio: float,
        rem_input_tokens: int,
        rem_chunk_tokens: Optional[int],
        num_mixed_decode_tokens: int = 0,
        priority_scheduling_preemption_threshold: int = 0,
        max_prefill_bs: int = 0,
        max_running_requests: Optional[int] = None,
        prefill_max_requests: Optional[int] = None,
        prefill_delayer_single_pass: Optional[PrefillDelayerSinglePassExecutor] = None,
        dllm_config: Optional[DllmConfig] = None,
        waiting_queue_len: int = 0,
        prefill_tile_block_m: int = 64,
    ):
        self.page_size = page_size
        self.prefill_tile_block_m = prefill_tile_block_m
        self.tree_cache = tree_cache
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.running_batch = running_batch
        self.new_token_ratio = new_token_ratio
        self.rem_input_tokens = rem_input_tokens - num_mixed_decode_tokens
        self.rem_chunk_tokens = rem_chunk_tokens
        self.chunked_req_limit: Optional[int] = None
        self.dllm_config = dllm_config
        # Mamba checkpoints only land on page-aligned chunk ends.
        self.exact_chunk_fill = (
            _use_exact_chunk_fill()
            and dllm_config is None
            and not (
                tree_cache.supports_mamba() and tree_cache.supports_prefix_sharing()
            )
        )

        if self.dllm_config is not None:
            self._init_dllm_meta(dllm_config)

        if self.rem_chunk_tokens is not None:
            self.rem_chunk_tokens -= num_mixed_decode_tokens
        self.memory_budget = token_to_kv_pool_allocator.create_prefill_budget(
            tree_cache, num_mixed_decode_tokens=num_mixed_decode_tokens
        )

        self.req_states = None
        self.can_run_list = []
        self.preempt_list = []
        self.new_chunked_req = None
        self.log_hit_tokens = 0
        self.reprocessed_log_hit_tokens = 0
        self.log_device_hit_tokens = 0
        self.log_host_hit_tokens = 0
        self.log_storage_hit_tokens = 0
        self.log_input_tokens = 0
        self.log_replay_tokens = 0
        self.reprocessed_log_input_tokens = 0

        if running_batch is not None:
            # Estimate the offset in the remaining token space
            self.memory_budget.total_offset += sum(
                [
                    self._get_running_request_total_token_offset(r)
                    for r in running_batch.reqs
                ]
            )

        # DeepSeek V4 HiSparse wraps an SWATokenToKVPoolAllocator internally and
        # exposes the full SWA allocator interface.
        self.is_hybrid_swa = isinstance(
            self.token_to_kv_pool_allocator,
            (SWATokenToKVPoolAllocator, DeepSeekV4HiSparseTokenToKVPoolAllocator),
        )
        self.is_hybrid_ssm_cache = self.tree_cache.supports_mamba()
        # A new state slot eats shared-gap bytes that `rem_total_tokens` counts
        # as free, so reserve per slot or admission over-commits. Gate on the
        # ALLOCATOR: the reservation holds with the radix cache disabled too.
        self._mamba_slot_cost = 0
        if isinstance(
            self.token_to_kv_pool_allocator,
            (
                UnifiedMambaTokenToKVPoolAllocator,
                UnifiedMambaSWATokenToKVPoolAllocator,
            ),
        ):
            self._mamba_slot_cost = (
                self.token_to_kv_pool_allocator.mamba_slot_full_token_cost()
            )

        # `mamba_gap_reserve` is charged to `rem_total_tokens`, which INCLUDES
        # `full_evictable_size()` — but `alloc_req_slots` can only recover
        # MAMBA-recoverable bytes for a mamba slot (shared gap + peer holes +
        # mamba-evictable radix), NOT full-evictable. Gate new mamba slots on
        # that mamba-recoverable budget separately or an over-admit hits the
        # fail-loud `RuntimeError`. `None` outside the unified Mamba pool.
        self.rem_mamba_slots = None
        if self._mamba_slot_cost:
            self.rem_mamba_slots = self.token_to_kv_pool_allocator.mamba_allocator.schedulable_available_size()
            if self.is_hybrid_ssm_cache:
                self.rem_mamba_slots += self.tree_cache.mamba_evictable_size()

        self.priority_scheduling_preemption_threshold = (
            priority_scheduling_preemption_threshold
        )
        self.max_running_requests = max_running_requests
        # Align sharded chunk boundaries to physical pages; 0 disables alignment.
        kv_shard_size = page_interleave_shard_size(self.token_to_kv_pool_allocator)
        self.kv_shard_granule = (
            self.token_to_kv_pool_allocator.page_size if kv_shard_size > 1 else 0
        )
        self.kv_shard_size = kv_shard_size
        # Each admission must fit the batch's prefix and chunk scratch regions.
        self.kv_shard_scratch_spec = (
            self.token_to_kv_pool_allocator.shard_spec if kv_shard_size > 1 else None
        )
        self.kv_shard_block_bound_pages = 0
        self.kv_shard_chunk_pages = 0
        # Reserve one extra page per shard class for allocation rounding.
        self.per_req_token_overhead = (
            kv_shard_size * self.kv_shard_granule
            if self.kv_shard_granule
            else self.page_size
        )
        self.prefill_max_requests = prefill_max_requests
        self.prefill_delayer_single_pass = prefill_delayer_single_pass
        self.max_prefill_bs = max_prefill_bs
        # Snapshot of scheduler waiting_queue length at the start of this
        # prefill pass. Used by PrefillDelayer's queue-based trigger.
        self.waiting_queue_len = waiting_queue_len

    def _admitted_extend_lens(self) -> List[int]:
        return [int(getattr(req, "extend_input_len", 0)) for req in self.can_run_list]

    def _tile_admission_metric_key(self) -> str:
        return f"{PREFILL_TILE_BUDGET_MODE}_q_tiles_per_head"

    def _candidate_tile_metrics(self, candidate_extend_len: int) -> Dict[str, object]:
        return estimate_prefill_extend_tile_metrics(
            [*self._admitted_extend_lens(), int(candidate_extend_len)],
            block_m=self.prefill_tile_block_m,
        )

    def _check_prefill_tile_budget(
        self, candidate_extend_len: int
    ) -> Optional[AddReqResult]:
        # AMD-only: leave non-AMD scheduler admission unchanged even if the env
        # budget is set.
        if not _IS_HIP or PREFILL_TILE_BUDGET <= 0:
            return None

        if not self.can_run_list:
            return None

        metrics = self._candidate_tile_metrics(candidate_extend_len)
        candidate_metric = int(metrics.get(self._tile_admission_metric_key()) or 0)
        if candidate_metric <= PREFILL_TILE_BUDGET:
            return None

        return AddReqResult.OTHER

    def _init_dllm_meta(self, dllm_config: DllmConfig):
        self.dllm_block_size = dllm_config.block_size
        max_running_reqs = dllm_config.max_running_requests

        self.rem_dllm_tokens = (
            self.rem_input_tokens
            if dllm_config.requires_separate_context_encoding
            else max_running_reqs * self.dllm_block_size
        )

    def _get_running_request_total_token_offset(self, req: Req) -> int:
        return (
            min(
                (req.sampling_params.max_new_tokens - len(req.output_ids)),
                CLIP_MAX_NEW_TOKENS,
            )
            * self.new_token_ratio
        )

    @property
    def rem_total_tokens(self):
        return self.memory_budget.remaining_total

    @property
    def cur_rem_tokens(self):
        return self.memory_budget.remaining_current

    def _swa_new_tokens(self, req: Req) -> int:
        """Tokens a request may still decode, for SWA headroom sizing. Mirrors
        the remaining-then-clip order of add_one_req's max_new: clip-then-subtract
        would zero out a request that has already generated >= CLIP tokens but
        still has a long decode ahead, under-reserving its window."""
        return min(
            max(req.sampling_params.max_new_tokens - len(req.output_ids), 0),
            CLIP_MAX_NEW_TOKENS,
        )

    def _check_prefill_budget(
        self,
        req: Req,
        *,
        extend_input_len: int,
        total_tokens: int,
        swa_host_hit_length: int,
    ) -> tuple[bool, Optional[int]]:
        return self.memory_budget.check_prefill(
            extend_input_len=extend_input_len,
            total_tokens=total_tokens,
            max_new_tokens=self._swa_new_tokens(req),
            input_tokens=len(req.full_untruncated_fill_ids),
            swa_host_hit_length=swa_host_hit_length,
            chunk_limit=self.rem_chunk_tokens,
        )

    def _mamba_gap_budget_for_req(self, req: Req) -> int:
        """Shared-gap reservation (full-token-equivalents) for a request's new
        mamba state. Charged only on the SHARED Mamba pool (`_mamba_slot_cost > 0`)
        and only when the req has no state yet (`mamba_pool_idx is None`, mirroring
        `HybridReqToTokenPool.alloc`); 0 keeps baseline / SWA / non-Mamba unchanged.

        Conservative by design (`_mamba_slot_cost` rounds UP). Does NOT reserve
        radix COW headroom or locked-but-evictable bytes — that residual is
        backstopped by the fail-loud RuntimeError in `alloc_req_slots`. FIXME: if
        over-admission crashes under pressure, make this more conservative (e.g.
        multiply by `MAMBA_STATE_PER_REQ_PREFIX_CACHE`)."""
        if self._mamba_slot_cost and not req.kv.holds_mamba:
            return self._mamba_slot_cost
        return 0

    def ceil_paged_tokens(self, tokens: int) -> int:
        return -(-tokens // self.page_size) * self.page_size

    def budget_state(self):
        no_token = not self.memory_budget.has_capacity()
        # Gate new mamba slots separately: rem_total_tokens' full_evictable can't
        # cover a mamba slot, which needs mamba-recoverable bytes (see __init__).
        if not no_token and self.rem_mamba_slots is not None:
            no_token = self.rem_mamba_slots <= 0
        if no_token:
            return AddReqResult.NO_TOKEN

        if self.rem_input_tokens <= 0:
            return AddReqResult.OTHER

        if self.dllm_config is not None:
            if self.rem_dllm_tokens <= 0:
                return AddReqResult.OTHER
        else:
            if self.rem_chunk_tokens is not None and self.rem_chunk_tokens <= 0:
                return AddReqResult.OTHER

        return AddReqResult.CONTINUE

    def _update_prefill_budget(
        self,
        prefix_len: int,
        extend_input_len: int,
        max_new_tokens: int,
        retracted_stain: bool,
        mamba_gap_reserve: int = 0,
        is_chunked_continuation: bool = False,
        compute_charge: Optional[int] = None,
    ):
        """Charge one admitted request against the prefill budgets.

        `compute_charge` is what the compute budgets (`rem_chunk_tokens`,
        `rem_input_tokens`, `rem_dllm_tokens`) are billed, counted in
        forward-pass tokens; the KV budgets are always billed page-ceiled
        tokens. It defaults to the ceiled count, so only the exact-chunk-fill
        path parts from upstream behaviour. Both compute budgets have to take
        it: they are usually configured to the same value, so leaving either one
        ceiled makes it hit zero first and stop admission with the rounding
        slack unspent.
        """
        # TODO(lsyin): check this workaround logic, which only ensures the prefill will not out of memory, and may be too conservative
        raw_extend_input_len = extend_input_len
        extend_input_len = self.ceil_paged_tokens(extend_input_len)
        if compute_charge is None:
            compute_charge = extend_input_len

        self.memory_budget.reserve(
            extend_input_len,
            max_new_tokens,
            # reserve() already charges one page; add the remaining shard pages.
            extra_tokens=(
                mamba_gap_reserve + self.per_req_token_overhead - self.page_size
            ),
            chunk_limit=self.rem_chunk_tokens,
            is_chunked_continuation=is_chunked_continuation,
        )
        # The new mamba slot also consumes one mamba-recoverable slot (gated
        # separately so full_evictable can't cover it — see __init__).
        if mamba_gap_reserve and self.rem_mamba_slots is not None:
            self.rem_mamba_slots -= 1
        self.rem_input_tokens -= compute_charge

        if self.dllm_config is not None:
            self.rem_dllm_tokens -= compute_charge
        elif self.rem_chunk_tokens is not None:
            self.rem_chunk_tokens -= compute_charge

        # reprocessed_log_* is a subset of log_*; metrics_reporter subtracts it
        # when computing the first-attempt prefix cache hit rate.
        self.log_hit_tokens += prefix_len
        self.log_input_tokens += raw_extend_input_len
        if retracted_stain:
            self.reprocessed_log_hit_tokens += prefix_len
            self.reprocessed_log_input_tokens += raw_extend_input_len

    def _account_prefill_cache_admission(self, req: Req, prefix_len: int) -> None:
        if get_exec().features.enable_encoder_swa_bounded_replay and (
            req.kv.req_pool_idx is None or req.is_retracted
        ):
            replay_tokens = min(prefix_len, 128)
            self.log_replay_tokens += replay_tokens
            self.rem_input_tokens -= replay_tokens
            if self.rem_chunk_tokens is not None:
                self.rem_chunk_tokens -= replay_tokens
        if req.retracted_stain:
            # Retraction attribution is intentionally omitted for now; discard
            # its lifecycle state so a later abort cannot report it as a drop.
            self.tree_cache.discard_storage_prefetch_accounting(
                req.cache_request_handle
            )
            return

        if prefix_len > 0:
            device_hit, host_hit, storage_hit = split_cached_prefix_by_tier(
                prefix_len=prefix_len,
                host_hit_len=req.materialized_host_hit_len(),
                storage_hit_len=req.storage_hit_length,
                storage_hit_start=req.storage_hit_start,
                host_hit_is_storage=req.host_hit_is_storage,
            )
            self.log_device_hit_tokens += device_hit
            self.log_host_hit_tokens += host_hit
            self.log_storage_hit_tokens += storage_hit

        fulfilled_storage_hit = req.fulfilled_storage_hit_len(prefix_len)
        reason = None
        if fulfilled_storage_hit < req.storage_hit_length:
            reason = (
                "device_capacity"
                if req.needs_host_load_back()
                and req.host_loaded_length < req.host_hit_length
                else "cache_admission_shortfall"
            )
        self.tree_cache.finish_storage_prefetch_admission(
            req.cache_request_handle,
            fulfilled_tokens=fulfilled_storage_hit,
            reason=reason,
        )

    def _get_dllm_remain_tokens(self, req: Optional[Req] = None) -> int:
        _rem_tokens = min(self.rem_dllm_tokens, int(self.rem_total_tokens))
        if not (
            self.dllm_config.requires_separate_context_encoding
            and req is not None
            and req.is_dllm_prefill()
        ):
            _rem_tokens = min(_rem_tokens, self.dllm_block_size)
        if _rem_tokens <= 0:
            _rem_tokens = self.rem_dllm_tokens

        if self.dllm_config.requires_separate_context_encoding:
            _rem_tokens = min(_rem_tokens, int(self.cur_rem_tokens) - self.page_size)
            if self.is_hybrid_swa:
                _rem_tokens = min(
                    _rem_tokens, int(self.memory_budget.remaining_swa) - self.page_size
                )
        return _rem_tokens

    def _get_dllm_extend_len(self, req: Req, prefix_len: int) -> int:
        if self.dllm_config.requires_separate_context_encoding:
            remaining = (
                req.dllm_block_offset - prefix_len
                if req.is_dllm_prefill()
                else len(req.full_untruncated_fill_ids) - prefix_len
            )
            limit = self.rem_dllm_tokens
            if not req.is_dllm_prefill():
                limit = min(limit, self.dllm_block_size)
            trunc_len = min(limit, remaining)
        else:
            trunc_len = (
                min(self.rem_dllm_tokens, self.dllm_block_size)
                // self.page_size
                * self.page_size
            )

        return trunc_len

    def _add_dllm_req(self, req: Req, prefix_len: int):
        trunc_len = self._get_dllm_extend_len(req, prefix_len)
        req.set_extend_range(prefix_len, prefix_len + trunc_len)

        self.can_run_list.append(req)

        self._update_prefill_budget(
            prefix_len,
            trunc_len,
            0,
            req.retracted_stain,
            mamba_gap_reserve=self._mamba_gap_budget_for_req(req),
        )
        self._account_prefill_cache_admission(req, prefix_len)

    def _req_inc_lock_ref(self, req: Req):
        # Persist the release receipt.
        req.lock_receipt = self.tree_cache.inc_lock_ref(req.last_node).to_dec_params()

    def _kv_shard_reserve_scratch(self, prefix_len: int, extend_len: int) -> bool:
        """Reserve scratch or return False to defer; no-op when sharding is off.

        Prefixes use N * sum_i ceil(prefix_pages_i / N) pages; extends use
        sum_i ceil(extend_i / ps). A request must fit an empty batch or raise.
        """
        if self.kv_shard_scratch_spec is None:
            return True
        ps = self.kv_shard_granule
        shard_size = self.kv_shard_size
        block = self.kv_shard_block_bound_pages + -(-(prefix_len // ps) // shard_size)
        chunk = self.kv_shard_chunk_pages + -(-extend_len // ps)
        if (
            shard_size * block * ps > self.kv_shard_scratch_spec.max_prefix_tokens
            or chunk * ps > self.kv_shard_scratch_spec.chunk_tokens
        ):
            if self.kv_shard_block_bound_pages == 0 and self.kv_shard_chunk_pages == 0:
                # Deferring a request that cannot fit an empty batch would livelock.
                raise RuntimeError(
                    "request cannot fit the sharded assembly scratch even in "
                    f"an empty batch (prefix_len={prefix_len}, extend_len="
                    f"{extend_len}, max_prefix_tokens="
                    f"{self.kv_shard_scratch_spec.max_prefix_tokens}, "
                    f"chunk_tokens={self.kv_shard_scratch_spec.chunk_tokens}); "
                    "KV sharding requires chunked prefill sized within the "
                    "assembly scratch"
                )
            return False
        self.kv_shard_block_bound_pages = block
        self.kv_shard_chunk_pages = chunk
        return True

    def add_dllm_staging_req(self, req: Req):
        assert self.dllm_config is not None
        _rem_tokens = self._get_dllm_remain_tokens(req)

        if _rem_tokens <= 0:
            return AddReqResult.NO_TOKEN

        # Truncate input length to available tokens and update request metadata
        cand_extend_input_len = len(req.full_untruncated_fill_ids) - len(
            req.prefix_indices
        )
        if (
            self.dllm_config.requires_separate_context_encoding
            and req.is_dllm_prefill()
        ):
            cand_extend_input_len = min(
                cand_extend_input_len,
                req.dllm_block_offset - len(req.prefix_indices),
            )
        if (
            req.dllm_incomplete_ids
            or (
                self.dllm_config.requires_separate_context_encoding
                and not req.is_dllm_prefill()
            )
        ) and cand_extend_input_len > _rem_tokens:
            return AddReqResult.NO_TOKEN
        truncated = cand_extend_input_len > _rem_tokens
        new_len = min(cand_extend_input_len, _rem_tokens)
        req.set_extend_range(len(req.prefix_indices), len(req.prefix_indices) + new_len)
        self.can_run_list.append(req)

        # Update budget: reserve max_new_tokens only if not truncated
        max_new_tokens = (
            min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS)
            if not truncated
            else 0
        )
        self._update_prefill_budget(
            0,
            req.extend_range.length,
            max_new_tokens,
            req.retracted_stain,
            mamba_gap_reserve=self._mamba_gap_budget_for_req(req),
        )

        # Return based on remaining token availability
        return (
            AddReqResult.NO_TOKEN
            if self._get_dllm_remain_tokens(req) <= 0
            else AddReqResult.CONTINUE
        )

    def add_chunked_req(self, req: Req):
        if self.dllm_config is not None:
            _rem_tokens = self._get_dllm_remain_tokens(req)
        else:
            _rem_tokens = self.memory_budget.available_chunk_tokens(
                self.rem_chunk_tokens
            )
            if _rem_tokens is None:
                return req
            if self.kv_shard_granule:
                # Align the absolute boundary to keep pages within one scratch region.
                # If no budget remains, use a full chunk to make progress.
                prefix_len = len(req.prefix_indices)
                floored = (
                    prefix_len + _rem_tokens
                ) // self.kv_shard_granule * self.kv_shard_granule - prefix_len
                _rem_tokens = floored if floored > 0 else self.rem_chunk_tokens

        # A mid-chunk rank prefills this pass regardless of the delayer
        # verdict, so report prefillable=True and ignore the result.
        if self.prefill_delayer_single_pass is not None:
            self.prefill_delayer_single_pass.negotiate_should_allow_prefill(
                local_prefillable=True,
                running_batch=self.running_batch.batch_size(),
                max_prefill_bs=self.max_prefill_bs,
                max_running_requests=self.max_running_requests,
                waiting_queue_len=self.waiting_queue_len,
            )

        if self.chunked_req_limit is not None:
            assert self.chunked_req_limit > 0
            _rem_tokens = min(_rem_tokens, self.chunked_req_limit)

        cand_extend_input_len = len(req.full_untruncated_fill_ids) - len(
            req.prefix_indices
        )
        _rem_tokens = self.memory_budget.fit_chunk(
            extend_input_len=cand_extend_input_len,
            max_new_tokens=self._swa_new_tokens(req),
            chunk_limit=_rem_tokens,
        )
        if _rem_tokens is None:
            return req
        truncated = cand_extend_input_len > _rem_tokens
        new_len = min(cand_extend_input_len, _rem_tokens)
        # The continuing chunk must fit. Keep reservation outside assert for -O.
        reserved = self._kv_shard_reserve_scratch(
            prefix_len=len(req.prefix_indices), extend_len=new_len
        )
        if not reserved:
            raise RuntimeError("chunked request exceeds the sharded assembly scratch")
        req.set_extend_range(len(req.prefix_indices), len(req.prefix_indices) + new_len)
        self.can_run_list.append(req)
        self._update_prefill_budget(
            0,
            req.extend_range.length,
            (
                min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS)
                if not truncated
                else 0
            ),
            req.retracted_stain,
            mamba_gap_reserve=self._mamba_gap_budget_for_req(req),
            is_chunked_continuation=True,
            compute_charge=req.extend_range.length if self.exact_chunk_fill else None,
        )

        # Return if chunked prefill not finished
        return req if truncated else None

    @contextmanager
    def _lock_node(self, last_node: TreeNode, *, lock_host: bool = False):
        host_lock_params = (
            self.tree_cache.inc_host_lock_ref(last_node).to_dec_params()
            if lock_host
            else None
        )
        try:
            # Replay the acquire's receipt (SWA boundary uuid, mamba flag) so the
            # release takes back exactly what this temporary lock took.
            dec_lock_params = self.tree_cache.inc_lock_ref(last_node).to_dec_params()
            try:
                yield None
            finally:
                self.tree_cache.dec_lock_ref(last_node, dec_lock_params)
        finally:
            if host_lock_params is not None:
                self.tree_cache.dec_host_lock_ref(last_node, host_lock_params)

    def add_one_req_ignore_eos(self, req: Req):
        cand_extend_input_len = len(req.full_untruncated_fill_ids) - len(
            req.prefix_indices
        )
        paged_input = (
            self.ceil_paged_tokens(cand_extend_input_len) + self.per_req_token_overhead
        )
        # Shared Mamba pool: fold the new mamba state's shared-gap cost into the
        # budget gate so admission can't over-commit (0 for baseline / non-Mamba).
        paged_input += self._mamba_gap_budget_for_req(req)
        fits = self.memory_budget.can_allocate_prefill(
            paged_input=paged_input,
            extend_input_len=cand_extend_input_len,
            max_new_tokens=self._swa_new_tokens(req),
            chunk_limit=self.rem_chunk_tokens,
        )
        if not fits:
            return AddReqResult.NO_TOKEN

        def add_req_state(r, insert_sort=False):
            new_token_ratio = (
                1.0 if r.sampling_params.ignore_eos else self.new_token_ratio
            )
            tokens_left = r.sampling_params.max_new_tokens * new_token_ratio - len(
                r.output_ids
            )
            tokens_occupied = len(r.origin_input_ids) + len(r.output_ids)

            if tokens_left <= 0:
                return

            if not insert_sort:
                self.req_states.append((tokens_left, tokens_occupied))
            else:
                i = 0
                for i in range(len(self.req_states)):
                    if tokens_left <= self.req_states[i][0]:
                        break
                self.req_states.insert(i, (tokens_left, tokens_occupied))

        if self.req_states is None:
            self.req_states = []
            add_req_state(req)
            if self.running_batch is not None:
                for r in self.running_batch.reqs:
                    add_req_state(r)
            for r in self.can_run_list:
                add_req_state(r)
            self.req_states.sort(key=lambda x: x[0])
        else:
            add_req_state(req, insert_sort=True)

        if not self.is_hybrid_swa:
            # Skip this logic for swa. The SWA has different memory management, and
            # this mechanism is underestimating the memory usage.
            cur_rem_tokens = self.cur_rem_tokens - paged_input
            tokens_freed = 0
            for i, (tokens_left, tokens_occupied) in enumerate(self.req_states):
                # tokens_left gives a reservative calculation as the last token is not stored
                bs = len(self.req_states) - i
                min_free_tokens = cur_rem_tokens + tokens_freed - tokens_left * bs
                # reserve tokens for corner cases
                if min_free_tokens <= IGNORE_EOS_RESERVE_TOKENS * bs:
                    return AddReqResult.NO_TOKEN
                tokens_freed += tokens_occupied

        if (self.prefill_delayer_single_pass is not None) and (
            not self.prefill_delayer_single_pass.negotiate_should_allow_prefill(
                local_prefillable=True,
                running_batch=self.running_batch.batch_size(),
                max_prefill_bs=self.max_prefill_bs,
                max_running_requests=self.max_running_requests,
                waiting_queue_len=self.waiting_queue_len,
            )
        ):
            return AddReqResult.OTHER

        if self.dllm_config is not None:
            if self.rem_dllm_tokens <= 0:
                return AddReqResult.OTHER

            if (
                tile_stop := self._check_prefill_tile_budget(cand_extend_input_len)
            ) is not None:
                return tile_stop

            self._add_dllm_req(req, 0)
        elif (
            self.rem_chunk_tokens is None  # chunked prefill is disabled
            or cand_extend_input_len <= self.rem_chunk_tokens  # it is the last chunk
        ):
            if (
                tile_stop := self._check_prefill_tile_budget(cand_extend_input_len)
            ) is not None:
                return tile_stop

            # Non-chunked prefill — the whole sequence is committed this iter.
            if not self._kv_shard_reserve_scratch(
                prefix_len=len(req.prefix_indices),
                extend_len=cand_extend_input_len,
            ):
                return AddReqResult.OTHER
            req.set_extend_range(
                len(req.prefix_indices), len(req.full_untruncated_fill_ids)
            )
            self.can_run_list.append(req)
            self._update_prefill_budget(
                0,
                req.extend_range.length,
                min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS),
                req.retracted_stain,
                mamba_gap_reserve=self._mamba_gap_budget_for_req(req),
                compute_charge=(
                    req.extend_range.length if self.exact_chunk_fill else None
                ),
            )
        else:
            if self.rem_chunk_tokens <= 0:
                return AddReqResult.OTHER

            # Chunked prefill
            trunc_len = self.rem_chunk_tokens

            if (tile_stop := self._check_prefill_tile_budget(trunc_len)) is not None:
                return tile_stop

            # Reserve after the tile gate so rejected requests consume no scratch.
            if not self._kv_shard_reserve_scratch(prefix_len=0, extend_len=trunc_len):
                return AddReqResult.OTHER

            assert len(req.prefix_indices) == 0
            req.set_extend_range(
                len(req.prefix_indices), len(req.prefix_indices) + trunc_len
            )
            self.can_run_list.append(req)
            self.new_chunked_req = req
            self._update_prefill_budget(
                0,
                trunc_len,
                0,
                req.retracted_stain,
                mamba_gap_reserve=self._mamba_gap_budget_for_req(req),
                compute_charge=trunc_len if self.exact_chunk_fill else None,
            )

        return self.budget_state()

    def add_one_req(
        self, req: Req, has_chunked_req: bool, truncation_align_size: Optional[int]
    ):
        if (x := self.prefill_max_requests) is not None and len(self.can_run_list) >= x:
            return AddReqResult.OTHER

        if req.sampling_params.ignore_eos and getattr(self.tree_cache, "disable", True):
            return self.add_one_req_ignore_eos(req)

        # Reserve page_size for page-alignment overhead: the paged allocator may
        # consume one extra page per request (see alloc_extend), which
        # _update_prefill_budget also deducts.
        max_new = min(
            max(req.sampling_params.max_new_tokens - len(req.output_ids), 0),
            CLIP_MAX_NEW_TOKENS,
        )
        cand_extend_input_len = len(req.full_untruncated_fill_ids) - len(
            req.prefix_indices
        )
        total_tokens = cand_extend_input_len + max_new + self.per_req_token_overhead
        # Shared Mamba pool: fold the new mamba state's shared-gap cost into
        # `total_tokens` so both `rem_total_tokens` gates reflect the joint budget.
        # Read before `init_load_back` binds `req.mamba_pool_idx` — after that
        # this returns 0, so the debit sites below reuse the value.
        mamba_gap_reserve = self._mamba_gap_budget_for_req(req)
        total_tokens += mamba_gap_reserve

        # The temporary pin excludes this prefix from the evictable budget.
        # Selection itself neither allocates slots nor materializes host hits.
        with self._lock_node(req.last_node):
            admission = self._select_prefill_admission(
                req,
                total_tokens=total_tokens,
                host_hit_length=req.host_hit_length,
                swa_host_hit_length=req.swa_host_hit_length,
                truncation_align_size=truncation_align_size,
                has_chunked_req=has_chunked_req,
            )
            if isinstance(admission, AddReqResult):
                return admission

            # A rejected candidate must not report prefillable or queue H2D.
            if (self.prefill_delayer_single_pass is not None) and (
                not self.prefill_delayer_single_pass.negotiate_should_allow_prefill(
                    local_prefillable=True,
                    running_batch=self.running_batch.batch_size(),
                    max_prefill_bs=self.max_prefill_bs,
                    max_running_requests=self.max_running_requests,
                    waiting_queue_len=self.waiting_queue_len,
                )
            ):
                return AddReqResult.OTHER

            if req.needs_host_load_back():
                load_max_new = min(max_new, admission.max_new_tokens)
                # Reclaim can write back device victims and evict host leaves.
                # Pin the selected host/aux match until load-back owns its locks.
                with (
                    self._lock_node(req.best_match_node, lock_host=True)
                    if isinstance(self.tree_cache, UnifiedRadixCache)
                    else nullcontext()
                ):
                    full_load_tokens = req.host_hit_length
                    if (
                        isinstance(self.tree_cache, UnifiedRadixCache)
                        and self.tree_cache.buffer_pipeline is None
                        and not (
                            self.tree_cache.linker is not None
                            and self.tree_cache.linker.has_hit(req.rid)
                        )
                    ):
                        # Host hits can include resident FULL behind host-only aux.
                        # Reuse the FULL transfer spec to count only new slots.
                        full_transfer = (
                            self.tree_cache.tree_core.build_hicache_transfers(
                                ComponentType.FULL,
                                req.best_match_node,
                                CacheTransferPhase.LOAD_BACK,
                            )[0]
                        )
                        full_load_tokens = len(full_transfer.host_indices)
                    if not self.memory_budget.prepare_load_back(
                        full_tokens=(
                            full_load_tokens
                            + admission.extend_len
                            + load_max_new
                            + self.page_size
                            + mamba_gap_reserve
                        ),
                        extend_input_len=admission.extend_len,
                        max_new_tokens=load_max_new,
                        swa_host_hit_length=req.swa_host_hit_length,
                        chunk_limit=self.rem_chunk_tokens,
                    ):
                        return AddReqResult.NO_TOKEN
                    promised_host_hit = req.host_hit_length
                    loaded = self.tree_cache.init_load_back(
                        InitLoadBackParams(
                            best_match_node=req.best_match_node,
                            host_hit_length=req.host_hit_length,
                            req=req,
                        )
                    )
                if loaded is None:
                    return AddReqResult.OTHER
                new_indices, req.last_node = loaded
                req.host_loaded_length = len(new_indices)
                if 0 < req.host_loaded_length < promised_host_hit:
                    raise RuntimeError(
                        "HiCache load-back must commit all promised FULL tokens or none: "
                        f"req={req.rid} promised={promised_host_hit} "
                        f"loaded={req.host_loaded_length}"
                    )
                if req.host_loaded_length > promised_host_hit:
                    # A load can expose resident FULL behind host-only aux state; its
                    # H2D is queued, so keep the approved budget and shrink the work.
                    prefix_len = len(req.prefix_indices) + req.host_loaded_length
                    extend_len = admission.extend_len
                    if self.dllm_config is None:
                        extend_len = min(
                            extend_len, len(req.full_untruncated_fill_ids) - prefix_len
                        )
                    is_chunked = admission.is_chunked and (
                        prefix_len + extend_len < len(req.full_untruncated_fill_ids)
                    )
                    max_new_tokens = admission.max_new_tokens
                    if admission.is_chunked and not is_chunked:
                        max_new_tokens = min(
                            req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS
                        )
                    admission = _PrefillAdmission(
                        prefix_len, extend_len, max_new_tokens, is_chunked
                    )
                elif req.host_loaded_length < promised_host_hit:
                    # No FULL was loaded; recomputation may no longer fit.
                    admission = self._select_prefill_admission(
                        req,
                        total_tokens=total_tokens,
                        host_hit_length=0,
                        swa_host_hit_length=0,
                        truncation_align_size=truncation_align_size,
                        has_chunked_req=has_chunked_req,
                    )
                    if isinstance(admission, AddReqResult):
                        return admission
                req.prefix_indices = torch.cat([req.prefix_indices, new_indices])
                req.kv.cache_protected_len = len(req.prefix_indices)

            # Sharded pools cannot load host KV; reserve scratch after all other gates.
            if not self._kv_shard_reserve_scratch(
                prefix_len=admission.prefix_len, extend_len=admission.extend_len
            ):
                return AddReqResult.OTHER

            self._commit_prefill_admission(req, admission, mamba_gap_reserve)

        # This verdict controls the next candidate, not the committed request.
        return self.budget_state()

    def _select_prefill_admission(
        self,
        req: Req,
        *,
        total_tokens: int,
        host_hit_length: int,
        swa_host_hit_length: int,
        truncation_align_size: Optional[int],
        has_chunked_req: bool = False,
    ) -> _PrefillAdmission | AddReqResult:
        """Select a prefill shape without allocating or publishing cached KV."""
        prefix_len = len(req.prefix_indices) + host_hit_length
        extend_len = len(req.full_untruncated_fill_ids) - prefix_len
        input_tokens = self.ceil_paged_tokens(extend_len)
        # Whether the request fits whole. Against the raw length under
        # exact-chunk-fill, so a request whose ceiled length would spill is
        # not needlessly split into a second chunk.
        chunk_fit_tokens = extend_len if self.exact_chunk_fill else input_tokens
        can_admit, chunk_tokens_limit = self._check_prefill_budget(
            req,
            extend_input_len=extend_len,
            total_tokens=total_tokens,
            swa_host_hit_length=swa_host_hit_length,
        )
        if not can_admit:
            return AddReqResult.NO_TOKEN

        # Without chunking, allow the first request even above the input cap.
        if (
            self.rem_chunk_tokens is None
            and self.can_run_list
            and input_tokens >= self.rem_input_tokens
        ):
            return AddReqResult.OTHER

        is_chunked = False
        max_new_tokens = min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS)
        tile_tokens = input_tokens
        if self.dllm_config is not None:
            assert truncation_align_size is None, (
                "truncation_align_size is not supported for dllm prefill"
            )
            extend_len = self._get_dllm_extend_len(req, prefix_len)
            if extend_len <= 0:
                return AddReqResult.OTHER
            max_new_tokens = 0
        elif chunk_tokens_limit is not None and chunk_fit_tokens > chunk_tokens_limit:
            if (
                has_chunked_req
                and get_schedule().schedule_policy == "shortest-prefill-first"
            ):
                # Only one unfinished chunked request can be tracked.
                return AddReqResult.OTHER
            if self.exact_chunk_fill:
                # Take the remainder verbatim so the batch hits exactly
                # chunked_prefill_size. `chunk_fit_tokens > chunk_tokens_limit`
                # here, so this never runs past the end of the prompt. Uses the
                # limit rather than rem_chunk_tokens so an SWA-capped chunk stays
                # capped.
                extend_len = chunk_tokens_limit
                if truncation_align_size is not None:
                    extend_len = (
                        extend_len // truncation_align_size * truncation_align_size
                    )
            else:
                extend_len = chunk_tokens_limit // self.page_size * self.page_size
                if truncation_align_size is not None:
                    extend_len = (
                        extend_len // truncation_align_size * truncation_align_size
                    )
                end = (prefix_len + extend_len) // self.page_size * self.page_size
                extend_len = end - prefix_len
            if extend_len <= 0:
                return AddReqResult.OTHER
            is_chunked = True
            max_new_tokens = 0
            tile_tokens = extend_len

        if (verdict := self._check_prefill_tile_budget(tile_tokens)) is not None:
            return verdict

        return _PrefillAdmission(prefix_len, extend_len, max_new_tokens, is_chunked)

    def _commit_prefill_admission(
        self, req: Req, admission: _PrefillAdmission, mamba_gap_reserve: int
    ) -> None:
        assert len(req.prefix_indices) == admission.prefix_len
        req.set_extend_range(
            admission.prefix_len, admission.prefix_len + admission.extend_len
        )
        self._req_inc_lock_ref(req)
        self.can_run_list.append(req)
        if admission.is_chunked:
            self.new_chunked_req = req
        self._update_prefill_budget(
            admission.prefix_len,
            admission.extend_len,
            admission.max_new_tokens,
            req.retracted_stain,
            mamba_gap_reserve=mamba_gap_reserve,
            # Compute budgets are billed forward-pass tokens under exact-chunk-fill.
            compute_charge=admission.extend_len if self.exact_chunk_fill else None,
        )
        self._account_prefill_cache_admission(req, admission.prefix_len)

    def preempt_to_schedule(self, req: Req) -> bool:
        """
        Preempt running requests to serve the new request if the priority threshold is met and token count sum is verified.
        Returns True if preemption was committed, and the new request can be scheduled.
        """
        # Iterate running requests to find preemptible requests
        priority_sign = 1 if get_schedule().schedule_low_priority_values_first else -1

        # NOTE: A request finishes in two phases:
        #   1) update_finish_state + release_kv_cache  (in process_batch_result)
        #   2) filter out of batch                (in get_next_batch_to_run / update_running_batch)
        # Preemption runs between these two phases (inside get_new_batch_prefill),
        # so running_batch may still contain requests whose KV cache is already freed.
        # We must skip them here to avoid a double-free on release_req.
        valid_running_reqs = (
            r
            for r in self.running_batch.reqs
            if r not in self.preempt_list and not r.finished()
        )

        sorted_valid_running_reqs = sorted(
            valid_running_reqs,
            key=lambda x: (
                x.priority * (-priority_sign),
                -x.time_stats.wait_queue_entry_time,
            ),
        )

        preemptible_reqs = []
        min_tokens_to_remove = (
            len(req.full_untruncated_fill_ids)
            - len(req.prefix_indices)
            + min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS)
            - self.rem_total_tokens
        )
        for running_req in sorted_valid_running_reqs:
            # Priority difference needs to meet the threshold to be preemptible.
            priority_diff = (req.priority - running_req.priority) * (-priority_sign)

            if priority_diff > self.priority_scheduling_preemption_threshold:
                preemptible_reqs.append(running_req)
                min_tokens_to_remove -= self._get_running_request_total_token_offset(
                    running_req
                )
                if min_tokens_to_remove <= 0:
                    break
            else:
                break

        # Check max token count limit can be met
        if len(preemptible_reqs) == 0 or min_tokens_to_remove > 0:
            return False

        # Preempt running requests. Release allocated resources for immediate usage.
        preemptible_reqs = set(preemptible_reqs)
        keep_indices = []
        release_counter = 0
        for i, running_req in enumerate(self.running_batch.reqs):
            if running_req in preemptible_reqs:
                self.memory_budget.total_offset -= (
                    self._get_running_request_total_token_offset(running_req)
                )
                release_counter += 1
                self.running_batch.release_req(
                    i, len(self.running_batch.reqs) - release_counter
                )
            else:
                keep_indices.append(i)
        self.running_batch.filter_batch(keep_indices=keep_indices)
        self.preempt_list.extend(preemptible_reqs)
        return True
