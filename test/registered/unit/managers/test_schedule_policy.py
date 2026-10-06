import math
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import (
    AdaptiveBoostGamma,
    CacheAgnosticPolicy,
    CacheAwarePolicy,
    SchedulePolicy,
    gamma_boost,
    geometric_bin,
)
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_req(rid, origin_input_text, origin_input_ids, sampling_params=None, **kwargs):
    if sampling_params is None:
        sampling_params = SamplingParams()
    return Req(
        rid,
        origin_input_text,
        array("q", origin_input_ids),
        sampling_params,
        **kwargs,
    )


def _make_boost_req(rid, uncached, *, arrival, cached=0, output=0):
    """Req with u(r) = uncached (cached prefix + generated tokens charged in)."""
    req = _make_req(rid, "", list(range(uncached + cached)))
    req.num_matched_prefix_tokens = cached
    if output:
        req.output_ids.extend([0] * output)
    req.boost_arrival_tokens = arrival
    return req


class TestSchedulePolicyHRRN(CustomTestCase):
    def test_calc_priority_hrrn(self):
        """HRRN sorts by response ratio (waited_tokens / uncached_tokens).

        With three fresh reqs (waited_tokens = 0 for all), the ratio is 0 / uncached for each,
        and stable sort keeps original order among equal keys -- effectively pure SUF via the
        rid tie-breaker is avoided, so we set a non-zero arrival snapshot on some reqs to exercise
        the aging half of the formula.
        """
        tree_cache = RadixCache.create_simulated()

        # r_short: small uncached, just arrived (waited=0)
        # r_long:  large uncached, just arrived (waited=0)
        # r_aged:  medium uncached, arrived long ago (waited>>0)
        r_short = _make_req("short", "a", [1])
        r_long = _make_req("long", "a" * 10, list(range(10)))
        r_aged = _make_req("aged", "a" * 3, [1, 2, 3])

        # Fresh reqs arrived with the counter still at 0.
        r_short.arrival_processed_tokens = 0
        r_long.arrival_processed_tokens = 0
        r_aged.arrival_processed_tokens = 0

        waiting_queue = [r_long, r_aged, r_short]

        policy = SchedulePolicy(
            policy="hrrn",
            tree_cache=tree_cache,
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        # processed_tokens=1000 -> waited_tokens is 1000 for every req here.
        # Ratios: short = 1000/1 = 1000; aged = 1000/3 ~= 333; long = 1000/10 = 100.
        # Highest ratio first -> short, aged, long.
        policy.calc_priority(waiting_queue, processed_tokens=1000)

        self.assertEqual(waiting_queue[0].rid, "short")
        self.assertEqual(waiting_queue[1].rid, "aged")
        self.assertEqual(waiting_queue[2].rid, "long")

    def test_calc_priority_hrrn_aging_overtakes_short(self):
        """A long request that has waited enough should overtake a
        just-arrived short request."""
        tree_cache = RadixCache.create_simulated()

        r_long_old = _make_req("long_old", "a" * 100, list(range(100)))
        r_short_new = _make_req("short_new", "a", [1])

        # long_old arrived at counter=0 and has been waiting; short_new
        # just arrived (its snapshot equals the current counter).
        r_long_old.arrival_processed_tokens = 0
        r_short_new.arrival_processed_tokens = 100000

        waiting_queue = [r_short_new, r_long_old]

        policy = SchedulePolicy(
            policy="hrrn",
            tree_cache=tree_cache,
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        # processed_tokens=100000 -> long_old.waited = 100000, ratio = 100000/100 = 1000.
        #                            short_new.waited = 0, ratio = 0.
        # long_old should now be first.
        policy.calc_priority(waiting_queue, processed_tokens=100000)

        self.assertEqual(waiting_queue[0].rid, "long_old")
        self.assertEqual(waiting_queue[1].rid, "short_new")

    def test_calc_priority_hrrn_cached_length_affects_order(self):
        """Cached prefix length shortens uncached, so reqs with the same input
        length can sort differently by HRRN. Also verifies rid tie-break for
        equal-ratio reqs.

        Uses _sort_by_hrrn directly to bypass the prefix-match pass inside
        calc_priority, which would overwrite num_matched_prefix_tokens.
        """
        r_more_cached = _make_req("a", "x" * 100, list(range(100)))
        r_less_cached = _make_req("b", "x" * 100, list(range(100)))
        r_tie = _make_req("c", "x" * 100, list(range(100)))

        # r_more_cached: 90 cached -> uncached = 10 -> ratio = 1000 / 10 = 100.
        # r_less_cached:  0 cached -> uncached = 100 -> ratio = 1000 / 100 = 10.
        # r_tie:          0 cached -> uncached = 100 -> ratio = 1000 / 100 = 10 (ties with r_less_cached; rid "b" < "c").
        r_more_cached.num_matched_prefix_tokens = 90
        r_less_cached.num_matched_prefix_tokens = 0
        r_tie.num_matched_prefix_tokens = 0
        r_more_cached.arrival_processed_tokens = 0
        r_less_cached.arrival_processed_tokens = 0
        r_tie.arrival_processed_tokens = 0

        waiting_queue = [r_tie, r_less_cached, r_more_cached]
        SchedulePolicy._sort_by_hrrn(waiting_queue, set(), processed_tokens=1000)

        self.assertEqual(waiting_queue[0].rid, "a")
        self.assertEqual(waiting_queue[1].rid, "b")
        self.assertEqual(waiting_queue[2].rid, "c")


class TestShortestPrefillFirst(CustomTestCase):
    def setUp(self):
        self.policy = SchedulePolicy(
            policy="shortest-prefill-first",
            tree_cache=RadixCache.create_simulated(),
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )

    def make_req(self, rid, uncached, *, cached=0, arrived=0):
        req = _make_req(rid, "", list(range(uncached + cached)))
        req.full_untruncated_fill_ids = req.origin_input_ids[:]
        req.num_matched_prefix_tokens = cached
        req.prefix_indices = list(range(cached))
        req.time_stats.wait_queue_entry_time = arrived
        return req

    def test_calc_priority_uses_uncached_work(self):
        cached = self.make_req("cached", 16, cached=4096)
        short = self.make_req("short", 32)
        long = self.make_req("long", 1024)
        queue = [long, short, cached]
        with patch.object(self.policy, "_compute_prefix_matches", return_value=set()):
            self.policy.calc_priority(queue)
        self.assertEqual([req.rid for req in queue], ["cached", "short", "long"])

    def test_equal_work_uses_arrival_time(self):
        older = self.make_req("z", 32, arrived=1)
        newer = self.make_req("a", 32, arrived=2)
        queue = [newer, older]
        self.policy._sort_by_shortest_prefill(queue, set())
        self.assertEqual(queue, [older, newer])

    def test_duplicate_prefix_is_deprioritized(self):
        duplicate = self.make_req("duplicate", 1)
        other = self.make_req("other", 1024)
        queue = [duplicate, other]
        with patch.object(
            self.policy, "_compute_prefix_matches", return_value={duplicate.rid}
        ):
            self.policy.calc_priority(queue)
        self.assertEqual(queue, [other, duplicate])

    def test_retracted_output_is_part_of_uncached_work(self):
        replay = self.make_req("replay", 16, cached=1024)
        replay.output_ids.extend([0] * 64)
        short = self.make_req("short", 32)
        queue = [replay, short]
        self.policy._sort_by_shortest_prefill(queue, set())
        self.assertEqual(queue, [short, replay])

    def test_chunk_limit_reserves_complete_short_prefills(self):
        continuation = self.make_req("continuation", 16384)
        waiting = [self.make_req("a", 512), self.make_req("b", 1024)]
        self.assertEqual(
            self.policy.shortest_prefill_chunk_limit(continuation, waiting, 4096, 256),
            2560,
        )

    def test_reservation_rounds_to_pages_and_keeps_continuation_progress(self):
        continuation = self.make_req("continuation", 16384)
        self.assertEqual(
            self.policy.shortest_prefill_chunk_limit(
                continuation, [self.make_req("short", 257)], 4096, 256
            ),
            3584,
        )
        self.assertEqual(
            self.policy.shortest_prefill_chunk_limit(
                continuation, [self.make_req("short", 3840)], 4096, 256
            ),
            256,
        )

    def test_no_reservation_when_request_cannot_fit_or_is_not_shorter(self):
        continuation = self.make_req("continuation", 8192)
        for waiting, budget in [
            ([], 4096),
            ([self.make_req("same", 8192)], 4096),
            ([self.make_req("too-large", 4096)], 4096),
            ([self.make_req("short", 1)], 256),
        ]:
            with self.subTest(budget=budget, waiting=[req.rid for req in waiting]):
                self.assertIsNone(
                    self.policy.shortest_prefill_chunk_limit(
                        continuation, waiting, budget, 256
                    )
                )

    def test_other_policy_keeps_normal_chunk_limit(self):
        self.policy.policy = CacheAwarePolicy.HRRN
        self.assertIsNone(
            self.policy.shortest_prefill_chunk_limit(
                self.make_req("continuation", 8192),
                [self.make_req("short", 512)],
                4096,
                256,
            )
        )


class TestGammaBoostMath(CustomTestCase):
    """gamma_boost numerics (UniBoost Eq. 2): the expm1-stable form must stay
    finite and monotone across the whole gamma*work domain the scheduler can
    reach, including the underflow end where the boost is exactly 0."""

    def test_strictly_decreasing_in_work(self):
        # Monotone non-increasing across the whole ladder (the tail underflows
        # to exactly 0.0); strictly decreasing wherever the boost is nonzero.
        for gamma in (1e-5, 1e-4, 1e-3, 1.0):
            works = [2.0**i for i in range(0, 30)]
            boosts = [gamma_boost(w, gamma) for w in works]
            for larger, smaller in zip(boosts, boosts[1:]):
                self.assertGreaterEqual(larger, smaller)
                if larger > 0.0:
                    self.assertGreater(larger, smaller)

    def test_finite_and_nonnegative_across_gamma_work_domain(self):
        # gamma*work spans 1e-8 .. 800+ (float64: 1 - exp(-40) already rounds
        # to 1.0, so anything past x=40 exercises the underflow guard).
        for gamma in (1e-8, 1e-5, 1e-4, 1e-2, 1.0):
            for work in (1e-4, 1e-1, 1.0, 10.0, 1e3, 1e6, 8e6, 1e9):
                value = gamma_boost(work, gamma)
                self.assertTrue(math.isfinite(value), (gamma, work, value))
                self.assertGreaterEqual(value, 0.0)

    def test_large_gamma_work_underflows_to_exact_zero(self):
        self.assertEqual(gamma_boost(8_000_000.0, 1e-4), 0.0)
        self.assertEqual(gamma_boost(100.0, 1.0), 0.0)

    def test_smaller_gamma_gives_larger_boost(self):
        # gamma -> 0+ diverges: -ln(1 - exp(-gamma*w))/gamma ~ -ln(gamma*w)/gamma.
        self.assertGreater(gamma_boost(256.0, 1e-8), gamma_boost(256.0, 1e-4))
        self.assertGreater(gamma_boost(256.0, 1e-4), gamma_boost(256.0, 1e-2))

    def test_nonpositive_gamma_work_is_infinite(self):
        # Unreachable from the sort (u >= 1, gamma validated > 0) but pinned
        # so direct callers cannot silently get a finite boost for free work.
        self.assertEqual(gamma_boost(0.0, 1e-4), float("inf"))
        self.assertEqual(gamma_boost(5.0, -1.0), float("inf"))

    def test_max_boost_under_quantization_is_b_of_bin_size(self):
        # q(u) >= k always, and b is decreasing, so b(q(u)) <= b(k): the bin
        # size caps the boost. Equality exactly on the u <= k floor.
        k, gamma = 256, 1e-4
        ceiling = gamma_boost(k, gamma)
        for u in (1, k, k + 1, 2 * k, 3 * k, 100_000, 250_000, 262_144):
            self.assertLessEqual(gamma_boost(geometric_bin(u, k), gamma), ceiling)
        self.assertEqual(gamma_boost(geometric_bin(k, k), gamma), ceiling)
        self.assertEqual(gamma_boost(geometric_bin(1, k), gamma), ceiling)


class TestGeometricBin(CustomTestCase):
    """geometric_bin (UniBoost Eq. 8): exact {k, 2k, 4k, ...} ladder with the
    u <= k floor, for power-of-two and arbitrary bin sizes."""

    def test_exact_ladder_power_of_two_k(self):
        k = 256
        cases = {
            0: 256,
            1: 256,
            255: 256,
            256: 256,
            257: 256,
            511: 256,
            512: 512,
            513: 512,
            1023: 512,
            1024: 1024,
            1025: 1024,
            100_000: 65_536,
            250_000: 131_072,
        }
        for u, expected in cases.items():
            self.assertEqual(geometric_bin(u, k), expected, (k, u))

    def test_exact_ladder_arbitrary_k(self):
        k = 100
        cases = {1: 100, 100: 100, 101: 100, 199: 100, 200: 200, 399: 200, 400: 400}
        for u, expected in cases.items():
            self.assertEqual(geometric_bin(u, k), expected, (k, u))

    def test_every_output_is_on_the_ladder(self):
        for k in (1, 3, 256, 1000):
            ladder = {k * 2**j for j in range(0, 20)}
            for u in list(range(0, 4 * k + 3)) + [100_000, 262_144]:
                self.assertIn(geometric_bin(u, k), ladder, (k, u))


class TestBoostSort(CustomTestCase):
    """_sort_by_boost: phi(r) = A(r) - b_gamma(q(u(r))), ascending.

    The two headline properties, both derived from the formula rather than
    mirrored from the implementation:
      - locality: a small-work newcomer still jumps ahead of a large-work
        older request (the point of the boost);
      - starvation bound: it can only do so while its arrival stamp is below
        b_gamma(k) - b_gamma(q(u_cold)) -- after that the old request is
        unreachable, so wait is bounded by b_gamma(k) clock-tokens.
    """

    GAMMA_K = (1e-4, 256)

    def sort_rids(self, reqs, deprioritized=frozenset(), gamma=1e-4, bin_size=256):
        queue = list(reqs)
        SchedulePolicy._sort_by_boost(queue, set(deprioritized), gamma, bin_size)
        return [r.rid for r in queue]

    def test_locality_hot_jumps_ahead_of_older_cold(self):
        # Hand-checked anchors, gamma=1e-4 k=256: b(256) = 36779,
        # b(q(100_000)) = b(65536) ~= 14.3. phi_hot = 500 - 36779 = -36279
        # < phi_cold = -14.3 despite arriving 500 clock-tokens later.
        cold = _make_boost_req("cold", 100_000, arrival=0)
        hot = _make_boost_req("hot", 256, arrival=500)
        self.assertEqual(
            self.sort_rids([cold, hot], gamma=1e-4, bin_size=256), ["hot", "cold"]
        )

    def test_starvation_bound_cold_becomes_unreachable(self):
        for gamma in (5e-5, 1e-4, 2e-4):
            for u_cold in (100_000, 250_000):
                with self.subTest(gamma=gamma, u_cold=u_cold):
                    k = 256
                    # Newest arrival that can still beat the cold req.
                    bound = gamma_boost(k, gamma) - gamma_boost(
                        geometric_bin(u_cold, k), gamma
                    )
                    self.assertGreater(bound, 0)
                    arrivals = [
                        0,
                        int(bound * 0.5),
                        int(bound * 0.9),
                        int(bound * 1.1),
                        int(bound * 2),
                        int(bound * 10),
                    ]
                    reqs = [_make_boost_req("cold", u_cold, arrival=0)]
                    reqs += [
                        _make_boost_req(f"hot{a}", 256, arrival=a) for a in arrivals
                    ]
                    order = self.sort_rids(reqs, gamma=gamma, bin_size=k)
                    cold_at = order.index("cold")
                    # Every hot below the bound is ahead of the cold; every
                    # hot above it is behind; each class is FCFS (equal boost).
                    ahead = [f"hot{a}" for a in arrivals if a < bound]
                    behind = [f"hot{a}" for a in arrivals if a > bound]
                    self.assertEqual(order[:cold_at], ahead)
                    self.assertEqual(order[cold_at + 1 :], behind)

    def test_large_gamma_converges_to_fcfs_by_arrival(self):
        # gamma=1.0 with w >= 256: 1 - exp(-256) rounds to 1.0, so every boost
        # is exactly 0 and phi = A -- pure arrival order regardless of work.
        reqs = [
            _make_boost_req("new_short", 256, arrival=900),
            _make_boost_req("old_huge", 250_000, arrival=100),
            _make_boost_req("mid", 5_000, arrival=500),
        ]
        self.assertEqual(
            self.sort_rids(reqs, gamma=1.0), ["old_huge", "mid", "new_short"]
        )

    def test_small_gamma_converges_to_shortest_uncached_work(self):
        # gamma=1e-9: boost differences (~ln(2)/gamma ~ 7e8 tokens) dominate
        # any arrival differences, so ordering is ascending uncached work.
        reqs = [
            _make_boost_req("big_new", 4_096, arrival=1_000_000),
            _make_boost_req("small_old", 256, arrival=0),
            _make_boost_req("mid", 1_024, arrival=500_000),
        ]
        self.assertEqual(
            self.sort_rids(reqs, gamma=1e-9), ["small_old", "mid", "big_new"]
        )

    def test_retracted_output_tokens_are_charged_as_work(self):
        # A retracted req re-queues with matched=0 and its generated tokens
        # must count into u(r) (they are re-prefilled), or it would keep a
        # small-work boost it no longer deserves.
        replay = _make_boost_req("replay", 100, arrival=0, output=900)
        fresh = _make_boost_req("fresh", 500, arrival=0)
        self.assertEqual(self.sort_rids([replay, fresh]), ["fresh", "replay"])

    def test_in_batch_dedup_followers_sort_last(self):
        follower = _make_boost_req("follower", 256, arrival=0)
        other = _make_boost_req("other", 250_000, arrival=1_000_000)
        self.assertEqual(
            self.sort_rids([follower, other], deprioritized={"follower"}),
            ["other", "follower"],
        )

    def test_priority_scheduling_is_the_primary_key(self):
        # Same composition contract as _sort_by_priority_and_fcfs: priority
        # first, then the boost key. Default sign: higher value first.
        low_boost_star = _make_boost_req("low_boost_star", 256, arrival=0)
        low_boost_star.priority = 1
        high_boost_dog = _make_boost_req("high_boost_dog", 250_000, arrival=10**9)
        high_boost_dog.priority = 5
        queue = [low_boost_star, high_boost_dog]
        SchedulePolicy._sort_by_boost(
            queue, set(), 1e-4, 256, enable_priority_scheduling=True, priority_sign=-1
        )
        self.assertEqual([r.rid for r in queue], ["high_boost_dog", "low_boost_star"])

    def test_retraction_reset_keeps_the_arrival_stamp(self):
        # reset_for_retract clears the cache-match state but must keep the
        # first-entry stamp, or retracted reqs lose their seniority on requeue.
        req = _make_boost_req("retracted", 100, arrival=1234)
        req.reset_for_retract()
        self.assertEqual(req.boost_arrival_tokens, 1234)
        self.assertEqual(req.retraction_count, 1)
        self.assertEqual(req.num_matched_prefix_tokens, 0)

    def test_calc_priority_dispatches_boost(self):
        policy = SchedulePolicy(
            policy="boost",
            tree_cache=RadixCache.create_simulated(),
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
            boost_gamma=1e-4,
            boost_bin_size=256,
        )
        cold = _make_boost_req("cold", 100_000, arrival=0)
        hot = _make_boost_req("hot", 256, arrival=500)
        queue = [cold, hot]
        with patch.object(policy, "_compute_prefix_matches", return_value=set()):
            policy.calc_priority(queue)
        self.assertEqual([r.rid for r in queue], ["hot", "cold"])

    def test_large_queue_falls_back_to_fcfs_like_lpm(self):
        # Cost parity with lpm/hrrn: past 128 waiting reqs the active policy
        # is FCFS and the expensive match+sort pass is skipped.
        policy = SchedulePolicy(
            policy="boost",
            tree_cache=RadixCache.create_simulated(),
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        small = [_make_boost_req(f"r{i}", 256, arrival=0) for i in range(128)]
        large = small + [_make_boost_req("r129", 256, arrival=0)]
        self.assertIs(policy._determine_active_policy(small), CacheAwarePolicy.BOOST)
        self.assertIs(policy._determine_active_policy(large), CacheAgnosticPolicy.FCFS)

    def test_boost_without_prefix_cache_degrades_to_fcfs_with_warning(self):
        with self.assertLogs("sglang.srt.managers.schedule_policy", level="WARNING"):
            policy = SchedulePolicy(
                policy="boost",
                tree_cache=RadixCache.create_simulated(disable=True),
                enable_hierarchical_cache=True,
                enable_priority_scheduling=False,
                schedule_low_priority_values_first=False,
            )
        self.assertIs(policy.policy, CacheAgnosticPolicy.FCFS)


class TestFcfsMatch(CustomTestCase):
    """fcfs-match: FCFS ordering with the sort-time prefix match forced on.

    The arm exists because every current cache returns False from
    supports_fast_match_prefix(), so plain fcfs never populates
    num_matched_prefix_tokens and the eval log reports zero matched tokens
    even though admission re-matches. The contract under test: identical
    ordering to fcfs, with the match populated.
    """

    def _policy(self, name, tree_cache):
        return SchedulePolicy(
            policy=name,
            tree_cache=tree_cache,
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )

    def _cache_with_shared_prefix(self):
        cache = RadixCache.create_simulated()
        cache.insert(InsertParams(key=RadixKey(token_ids=array("q", list(range(64))))))
        return cache

    def test_populates_match_without_reordering(self):
        # cold arrives first and shares nothing; warm arrives second and
        # extends the cached 64-token prefix. FCFS order must keep cold first
        # while warm's match becomes visible -- the exact ordering-vs-metric
        # separation the arm exists to measure.
        for name in ("fcfs", "fcfs-match"):
            with self.subTest(policy=name):
                cache = self._cache_with_shared_prefix()
                cold = _make_req("cold", "x" * 80, [100 + i for i in range(80)])
                warm = _make_req("warm", "y", list(range(64)) + [7, 8, 9])
                queue = [cold, warm]
                with patch(
                    "sglang.srt.managers.schedule_policy.get_disagg",
                    return_value=SimpleNamespace(disaggregation_mode="none"),
                ):
                    self._policy(name, cache).calc_priority(queue)

                self.assertEqual([r.rid for r in queue], ["cold", "warm"])
                self.assertEqual(cold.num_matched_prefix_tokens, 0)
                expected = 64 if name == "fcfs-match" else 0
                self.assertEqual(warm.num_matched_prefix_tokens, expected)

    def test_falls_back_to_fcfs_without_prefix_cache(self):
        # Same degradation contract as boost: no prefix cache, no match arm.
        with self.assertLogs("sglang.srt.managers.schedule_policy", level="WARNING"):
            policy = self._policy(
                "fcfs-match", RadixCache.create_simulated(disable=True)
            )
        self.assertIs(policy.policy, CacheAgnosticPolicy.FCFS)


class TestInBatchDedupStamping(CustomTestCase):
    """calc_priority's in_batch_dedup_pending stamp, read at first admission
    by stamp_admission_prefix_matches to exempt dedup followers in the
    overtaking-invariant check. Per-sort last-write-wins, cleared off the
    cache-aware path so a first-admission stamp never reads a stale verdict.
    """

    def test_cache_aware_sort_stamps_the_verdict(self):
        policy = SchedulePolicy(
            policy="boost",
            tree_cache=RadixCache.create_simulated(),
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        follower = _make_boost_req("follower", 256, arrival=0)
        other = _make_boost_req("other", 4_096, arrival=10)
        queue = [follower, other]
        with patch.object(policy, "_compute_prefix_matches", return_value={"follower"}):
            policy.calc_priority(queue)
        self.assertIs(follower.in_batch_dedup_pending, True)
        self.assertIs(other.in_batch_dedup_pending, False)

    def test_agnostic_path_clears_stale_verdicts(self):
        # Past the 128-queue fallback the active policy is FCFS and the sort
        # does not classify dedup; a flag left from an earlier sort would
        # exempt a non-follower from the invariant bound.
        policy = SchedulePolicy(
            policy="boost",
            tree_cache=RadixCache.create_simulated(),
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        queue = [_make_boost_req(f"r{i}", 256, arrival=0) for i in range(129)]
        queue[0].in_batch_dedup_pending = True  # stale from an earlier sort
        policy.calc_priority(queue)
        self.assertFalse(any(r.in_batch_dedup_pending for r in queue))


class TestAdaptiveBoostGamma(CustomTestCase):
    # window/min_samples shrunk per instance so the estimator math stays
    # testable without 2048 admissions; the formulas are window-agnostic.
    LN5 = math.log(5.0)

    def test_estimator_math_and_ema(self):
        # waits = {0, 1000, ..., 99000}: linear-interpolated p95 = 94050,
        # p99 = 98010, so tail width = 3960 and gamma_hat = ln5/3960. One
        # update from gamma0 applies the 0.2 EMA weight to that estimate.
        # Guards the percentile definition (nearest-rank would give 95000 /
        # 98000 -> width 3000) and the beta placement (weighting the old
        # value instead of the estimate changes the result).
        adaptive = AdaptiveBoostGamma(gamma0=1e-4, window=128, min_samples=64)
        waits = list(range(0, 100_000, 1000))
        gamma = adaptive.record_admissions(waits)
        expected = 0.8 * 1e-4 + 0.2 * (self.LN5 / 3960.0)
        self.assertIsNotNone(gamma)
        self.assertAlmostEqual(gamma, expected, delta=expected * 1e-12)
        self.assertAlmostEqual(adaptive.gamma, expected, delta=expected * 1e-12)

    def test_clip_bounds(self):
        # Degenerate tails clip, never saturate past the bounds: an all-equal
        # window (width 0) is the maximally thin tail -> GAMMA_MAX; a huge
        # spread is the fat tail -> GAMMA_MIN. The floor 3e-7 is the task's
        # corrected value (1e-5 would clip off the c16 equilibrium gamma).
        thin = AdaptiveBoostGamma(gamma0=1e-4, window=16, min_samples=8)
        for _ in range(200):
            thin_gamma = thin.record_admissions([1000] * 16)
        self.assertLessEqual(thin_gamma, AdaptiveBoostGamma.GAMMA_MAX)
        self.assertGreater(thin_gamma, 9.9e-4)

        fat = AdaptiveBoostGamma(gamma0=1e-4, window=16, min_samples=8)
        for _ in range(200):
            fat_gamma = fat.record_admissions([0] * 15 + [10**12])
        self.assertGreaterEqual(fat_gamma, AdaptiveBoostGamma.GAMMA_MIN)
        self.assertLess(fat_gamma, 3.05e-7)

    def test_direction_load_step_burst(self):
        # The corrected direction: a fat wait tail -> smaller gamma (stronger
        # boosts, locality/LAS-ward); a thin tail after the load step ends ->
        # gamma recovers (FCFS-ward). An inverted estimator goes red here.
        adaptive = AdaptiveBoostGamma(gamma0=1e-4, window=16, min_samples=8)
        for _ in range(40):
            fat_gamma = adaptive.record_admissions([0] * 15 + [50_000_000])
        self.assertLess(fat_gamma, 1e-6)
        for _ in range(40):
            thin_gamma = adaptive.record_admissions([1000] * 16)
        self.assertGreater(thin_gamma, 5e-4)
        self.assertLess(fat_gamma, 1e-4, "fat tail must push gamma below gamma0")
        self.assertGreater(thin_gamma, fat_gamma, "recovery must raise gamma")

    def test_warmup_returns_none_below_min_samples(self):
        # Below min_samples percentiles are noise; the caller must be able to
        # skip (no gamma move, no eval-log mutation) until the window fills.
        adaptive = AdaptiveBoostGamma(gamma0=1e-4, window=16, min_samples=8)
        self.assertIsNone(adaptive.record_admissions([5] * 7))
        self.assertEqual(adaptive.gamma, 1e-4)
        self.assertIsNotNone(adaptive.record_admissions([5]))

    def test_schedule_policy_wiring(self):
        # boost + flag -> the window feeds rebind the live sort's gamma;
        # without the flag, or on another policy, gamma-Ada stays off (the
        # fixed-gamma path must be bit-identical to before).
        def make(policy="boost", **kw):
            return SchedulePolicy(
                policy=policy,
                tree_cache=RadixCache.create_simulated(),
                enable_hierarchical_cache=True,
                enable_priority_scheduling=False,
                schedule_low_priority_values_first=False,
                boost_gamma=1e-4,
                boost_bin_size=256,
                **kw,
            )

        on = make(boost_adaptive_gamma=True)
        self.assertIsNotNone(on.adaptive_gamma)
        gamma = on.record_admission_waits(list(range(0, 70_000, 700)))
        self.assertIsNotNone(gamma)
        self.assertNotEqual(gamma, 1e-4)
        self.assertEqual(on.boost_gamma, gamma)

        off = make()
        self.assertIsNone(off.adaptive_gamma)
        self.assertIsNone(off.record_admission_waits(list(range(0, 70_000, 700))))
        self.assertEqual(off.boost_gamma, 1e-4)

        other = make(policy="fcfs", boost_adaptive_gamma=True)
        self.assertIsNone(other.adaptive_gamma)


if __name__ == "__main__":
    unittest.main()
