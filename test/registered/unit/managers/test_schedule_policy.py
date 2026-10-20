import math
import re
import time
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.managers.schedule_batch import FINISH_LENGTH, Req
from sglang.srt.managers.schedule_policy import (
    AdaptiveBoostGamma,
    BoostClock,
    CacheAgnosticPolicy,
    CacheAwarePolicy,
    SchedulePolicy,
    TokenClockRate,
    gamma_boost,
    geometric_bin,
)
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.observability.schedule_eval_log import stamp_admission_prefix_matches
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


def _make_wall_req(rid, uncached, *, entry_s, cached=0, output=0):
    """Wall-clock twin of _make_boost_req: first_queue_entry_time in seconds."""
    req = _make_req(rid, "", list(range(uncached + cached)))
    req.num_matched_prefix_tokens = cached
    if output:
        req.output_ids.extend([0] * output)
    req.first_queue_entry_time = entry_s
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
        with patch.object(
            self.policy, "_compute_prefix_matches", return_value=(set(), {})
        ):
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
            self.policy,
            "_compute_prefix_matches",
            return_value=({duplicate.rid}, {}),
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

    def test_boost_policy_gets_chunk_limit(self):
        # Boost shares the chunk-limit cap so a giant chunked prefill reserves
        # tokens for shorter high-boost waiting requests (the c16 giant-prefill
        # blocking fix). A regression that removes BOOST from the policy guard
        # turns this red.
        self.policy.policy = CacheAwarePolicy.BOOST
        continuation = self.make_req("continuation", 16384)
        waiting = [self.make_req("a", 512), self.make_req("b", 1024)]
        self.assertEqual(
            self.policy.shortest_prefill_chunk_limit(continuation, waiting, 4096, 256),
            2560,
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

    def sort_rids(
        self, reqs, deprioritized=frozenset(), gamma=1e-4, bin_size=256, leader_of=None
    ):
        queue = list(reqs)
        SchedulePolicy._sort_by_boost(
            queue, set(deprioritized), gamma, bin_size, leader_of=leader_of
        )
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

    def test_in_batch_dedup_follower_sorts_behind_its_leader(self):
        # Regression (e2 fair sweep seed 77188, 2026-10-20): boost demoted
        # in-batch dedup followers to a global group-1 tail, which sits
        # behind every group-0 request -- under a giant-logged overloaded
        # queue that is behind every future arrival, so later session turns
        # starved 1500-1950 s while fcfs served the same reqs in <= 453 s.
        # A follower must sort just behind its leader (group 0), ahead of
        # every later arrival that the boost itself would not put ahead of
        # the leader (late arrives 30k tokens after leader but only saves
        # ~25.9k via its boost, so it still trails the leader).
        leader = _make_boost_req("leader", 4_096, arrival=0)
        follower = _make_boost_req("follower", 256, arrival=500)
        late = _make_boost_req("late", 256, arrival=30_000)
        self.assertEqual(
            self.sort_rids(
                [leader, follower, late],
                deprioritized={"follower"},
                leader_of={"follower": "leader"},
            ),
            ["leader", "follower", "late"],
        )

    def test_in_batch_dedup_follower_tie_breaks_by_arrival_not_rid(self):
        # Regression (e2 g5.0 seed 77188, 2026-10-20 night): followers of
        # one leader tie at phi(leader)+eps; a rid tie-break ordered the tie
        # by rid, and the next pass makes the first family member in this
        # order the next leader once the current one drains -- so a later
        # arrival with a smaller rid kept winning headship and an early
        # rid-unlucky request starved 1870 s under continuous same-prefix
        # arrivals (the cell's final drain admitted the family in exact rid
        # order, arrival ignored). The tie must break by arrival stamp.
        leader = _make_boost_req("leader", 4_096, arrival=0)
        early = _make_boost_req("zz_early", 256, arrival=500)
        late = _make_boost_req("aa_late", 256, arrival=30_000)
        self.assertEqual(
            self.sort_rids(
                [leader, early, late],
                deprioritized={"zz_early", "aa_late"},
                leader_of={"zz_early": "leader", "aa_late": "leader"},
            ),
            ["leader", "zz_early", "aa_late"],
        )

    def test_in_batch_dedup_follower_without_leader_sorts_last(self):
        # Leader left the queue between classification and sort (admitted
        # or retracted): the follower keeps the group-1 tail. Also the
        # contract for callers that omit leader_of entirely.
        follower = _make_boost_req("follower", 256, arrival=0)
        other = _make_boost_req("other", 250_000, arrival=1_000_000)
        for leader_of in ({"follower": "gone"}, None):
            with self.subTest(leader_of=leader_of):
                self.assertEqual(
                    self.sort_rids(
                        [follower, other],
                        deprioritized={"follower"},
                        leader_of=leader_of,
                    ),
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
        with patch.object(policy, "_compute_prefix_matches", return_value=(set(), {})):
            policy.calc_priority(queue)
        self.assertEqual([r.rid for r in queue], ["hot", "cold"])

    def test_large_queue_stays_boost_not_fcfs(self):
        # Unlike lpm/hrrn whose prefix matching is O(n * tree_depth), boost's
        # sort is O(n log n) and safe at scale. A regression that re-adds
        # BOOST to the _determine_active_policy fallback list turns this red.
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
        self.assertIs(policy._determine_active_policy(large), CacheAwarePolicy.BOOST)

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


class TestTokenClockRate(CustomTestCase):
    """TokenClockRate: the wall-mode work-to-seconds converter."""

    def test_first_pair_is_baseline_no_ema(self):
        # The first observe only stores baselines: no interval has elapsed,
        # so there is no sample. The first positive-delta interval assigns
        # its exact ratio (no EMA blend from an undefined prior).
        rate = TokenClockRate()
        self.assertIsNone(rate.observe(0, 0.0))
        self.assertIsNone(rate.tokens_per_s)
        self.assertEqual(rate.observe(30_000, 1.0), 30_000.0)

    def test_zero_delta_folds_time_not_a_sample(self):
        # An idle or decode-only stretch (no counted tokens) must not pollute
        # the EMA, and must not advance the time baseline either: the next
        # positive sample's window spans the gap, so the rate stays
        # load-inclusive (the quantity measured as ~3.3-6k tok/s fresh vs
        # ~36.7k warm).
        rate = TokenClockRate()
        rate.observe(0, 0.0)
        self.assertIsNone(rate.observe(0, 10.0))
        self.assertAlmostEqual(rate.observe(100, 11.0), 100.0 / 11.0)

    def test_ema_tracks_rate_step(self):
        # The fresh->warm rate step must actually move the estimate: a frozen
        # or mis-blended EMA would pin the wall bound to the boot-time rate
        # forever, recreating the cache-state drift in wall units.
        rate = TokenClockRate()
        rate.observe(0, 0.0)
        tokens, now = 0, 0.0
        for _ in range(200):
            tokens, now = tokens + 5_000, now + 1.0
            rate.observe(tokens, now)
        self.assertAlmostEqual(rate.tokens_per_s, 5_000.0, delta=1.0)
        for _ in range(200):
            tokens, now = tokens + 36_700, now + 1.0
            rate.observe(tokens, now)
        self.assertGreater(rate.tokens_per_s, 30_000.0)
        self.assertLess(rate.tokens_per_s, 36_700.0)

    def test_nonpositive_delta_ignored(self):
        # A counter regression (e.g. a caller omitting processed_tokens,
        # defaulting to 0) must neither crash nor sample a negative rate.
        rate = TokenClockRate()
        rate.observe(30_000, 0.0)
        self.assertIsNone(rate.observe(0, 1.0))
        self.assertIsNone(rate.observe(30_000, 2.0))
        self.assertAlmostEqual(rate.observe(40_000, 3.0), 10_000.0 / 3.0)


class TestBoostWallSort(CustomTestCase):
    """_sort_by_boost wall mode: phi = A_s - b(q(u)/R, gamma_s), seconds.

    Properties derived from the formula: the unmeasured-rate fallback, the
    starvation bound in seconds, and the cross-mode identity
    phi_wall = phi_tokens / R at gamma_s = gamma_t * R (wall mode is tokens
    mode retuned per cache state).
    """

    def sort_rids(self, reqs, gamma_s, rate, bin_size=256, deprioritized=frozenset()):
        queue = list(reqs)
        SchedulePolicy._sort_by_boost(
            queue,
            set(deprioritized),
            gamma_s,
            bin_size,
            clock=BoostClock.WALL,
            rate_tokens_per_s=rate,
        )
        return [r.rid for r in queue]

    def test_unmeasured_rate_is_pure_arrival_order(self):
        # No rate yet (first prefill batch not run): the defined fallback is
        # boost 0, so ordering is entry time alone -- in tokens mode the same
        # reqs reorder by work (test_locality_hot_jumps_ahead_of_older_cold).
        reqs = [
            _make_wall_req("new_short", 256, entry_s=900.0),
            _make_wall_req("old_huge", 250_000, entry_s=100.0),
            _make_wall_req("mid", 5_000, entry_s=500.0),
        ]
        self.assertEqual(
            self.sort_rids(reqs, gamma_s=0.1, rate=None),
            ["old_huge", "mid", "new_short"],
        )

    def test_wall_starvation_bound_in_seconds(self):
        for gamma_s in (0.02, 0.1, 0.5):
            for u_cold in (100_000, 250_000):
                with self.subTest(gamma_s=gamma_s, u_cold=u_cold):
                    k, rate = 256, 36_700.0
                    bound_s = gamma_boost(k / rate, gamma_s) - gamma_boost(
                        geometric_bin(u_cold, k) / rate, gamma_s
                    )
                    self.assertGreater(bound_s, 0)
                    lags = [
                        bound_s * 0.5,
                        bound_s * 0.9,
                        bound_s * 1.1,
                        bound_s * 2.0,
                        bound_s * 10.0,
                    ]
                    reqs = [_make_wall_req("cold", u_cold, entry_s=0.0)]
                    reqs += [
                        _make_wall_req(f"hot{lag:.3f}", 256, entry_s=lag)
                        for lag in lags
                    ]
                    order = self.sort_rids(reqs, gamma_s=gamma_s, rate=rate)
                    cold_at = order.index("cold")
                    ahead = [f"hot{lag:.3f}" for lag in lags if lag < bound_s]
                    behind = [f"hot{lag:.3f}" for lag in lags if lag > bound_s]
                    self.assertEqual(order[:cold_at], ahead)
                    self.assertEqual(order[cold_at + 1 :], behind)

    def test_wall_equivalent_to_tokens_at_scaled_gamma(self):
        # b and the arrival axis both scale by 1/R at gamma_s = gamma_t * R,
        # so the two clock modes must order the same reqs identically; a
        # tokens-unit leak into the wall branch (or vice versa) breaks it.
        rate, gamma_t = 36_700.0, 1e-4
        spec = [
            ("cold", 100_000, 0.0),
            ("hot1", 256, 0.0005),
            ("hot2", 256, 0.001),
            ("mid", 5_000, 0.002),
        ]
        wall_order = self.sort_rids(
            [_make_wall_req(rid, u, entry_s=e) for rid, u, e in spec],
            gamma_s=gamma_t * rate,
            rate=rate,
        )
        queue = [_make_boost_req(rid, u, arrival=int(e * rate)) for rid, u, e in spec]
        SchedulePolicy._sort_by_boost(queue, set(), gamma_t, 256)
        self.assertEqual(wall_order, [r.rid for r in queue])
        # The shared order is genuinely boost-driven, not arrival-ordered.
        self.assertEqual(wall_order, ["hot1", "hot2", "mid", "cold"])

    def test_in_batch_dedup_follower_sorts_behind_its_leader(self):
        # Wall-clock contract of the starvation fix (tokens-clock twin in
        # TestBoostSort): a demoted follower ranks at phi(leader) + 1e-3 s,
        # ahead of every later arrival the boost would not put ahead of the
        # leader (late's 43.6 s boost does not cover its 50 s arrival lag).
        leader = _make_wall_req("leader", 4_096, entry_s=0.0)
        follower = _make_wall_req("follower", 256, entry_s=5.0)
        late = _make_wall_req("late", 256, entry_s=50.0)
        queue = [leader, follower, late]
        SchedulePolicy._sort_by_boost(
            queue,
            {"follower"},
            0.1,
            256,
            clock=BoostClock.WALL,
            rate_tokens_per_s=2000.0,
            leader_of={"follower": "leader"},
        )
        self.assertEqual([r.rid for r in queue], ["leader", "follower", "late"])

    def test_calc_priority_measures_rate_from_processed_tokens(self):
        policy = SchedulePolicy(
            policy="boost",
            tree_cache=RadixCache.create_simulated(),
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
            boost_gamma=0.01,
            boost_clock="wall",
        )
        cold = _make_wall_req("cold", 100_000, entry_s=0.0)
        hot = _make_wall_req("hot", 256, entry_s=100.0)
        queue = [cold, hot]
        with (
            patch.object(policy, "_compute_prefix_matches", return_value=(set(), {})),
            patch(
                "sglang.srt.managers.schedule_policy.time.perf_counter",
                side_effect=[0.0, 1.0, 2.0],
            ),
        ):
            policy.calc_priority(queue, processed_tokens=0)
            # Unmeasured rate: pure arrival order (cold entered first).
            self.assertEqual([r.rid for r in queue], ["cold", "hot"])
            policy.calc_priority(queue, processed_tokens=30_000)
            self.assertEqual(policy.prefill_token_rate.tokens_per_s, 30_000.0)
            # Unchanged counter: no new sample, rate survives.
            policy.calc_priority(queue, processed_tokens=30_000)
            self.assertEqual(policy.prefill_token_rate.tokens_per_s, 30_000.0)
        # Measured 30k tok/s, gamma_s=0.01: hot's boost is ~937 s against a
        # 100 s arrival lag, so hot now leads.
        self.assertEqual([r.rid for r in queue], ["hot", "cold"])


class TestFcfsMatch(CustomTestCase):
    """fcfs-match: FCFS ordering with the sort-time prefix match forced on.

    The arm predates the admission-time bookkeeping fix: it forced the
    sort-time match because plain fcfs never populated
    num_matched_prefix_tokens. Since init_next_round_input now records the
    admission re-match, plain fcfs reports real reuse too; fcfs-match stays
    as the sort-time-match reference arm. The contract under test: identical
    ordering to fcfs, with the sort-time match populated.
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

    def test_admission_rematch_records_match_length(self):
        # The fcfs hit-metric bug: init_next_round_input's admission re-match
        # never recorded its length, so plain fcfs (no sort-time match)
        # reported zero matched tokens in the eval log while enjoying real
        # reuse -- every fcfs cache-hit row in the campaign tables read
        # 0.000. The re-match is read-only (inserts nothing); fcfs ordering
        # never reads num_matched_prefix_tokens, so order is unaffected.
        warm = _make_req("warm", "y", list(range(64)) + [7, 8, 9])
        warm.init_next_round_input(self._cache_with_shared_prefix())
        self.assertEqual(warm.num_matched_prefix_tokens, 64)
        stamp_admission_prefix_matches([warm])
        self.assertEqual(warm.matched_prefix_tokens_first_admission, 64)


class TestInBatchPrefixCaching(CustomTestCase):
    """_compute_prefix_matches classification: the in-batch dedup follower pin
    shared by every cache-aware policy, and its wall-second hold bound
    (--in-batch-prefix-caching-max-pin-s). The pin had no direct coverage
    before the bound; these cases pin the classification itself.
    """

    def _policy(self, max_pin_s=60.0):
        return SchedulePolicy(
            policy="lpm",
            tree_cache=RadixCache.create_simulated(),
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
            in_batch_prefix_caching_max_pin_s=max_pin_s,
        )

    @staticmethod
    def _cold_req(rid, token_ids, *, entry_offset_s=None):
        req = _make_req(rid, "", token_ids)
        if entry_offset_s is not None:
            req.first_queue_entry_time = time.perf_counter() - entry_offset_s
        return req

    def test_classification_leader_follower_unrelated(self):
        # Cold queue, empty radix: the first scanned req is inserted into the
        # simulated tree (leader); a later req sharing >= 32 leading tokens
        # with it is deprioritized (follower) and mapped to that leader; an
        # unrelated req is neither.
        leader = self._cold_req("leader", list(range(64)))
        follower = self._cold_req("follower", list(range(64)))
        unrelated = self._cold_req("unrelated", list(range(1000, 1064)))
        queue = [leader, follower, unrelated]

        pinned, leader_of = self._policy()._compute_prefix_matches(
            queue, CacheAwarePolicy.LPM
        )
        self.assertEqual(pinned, {"follower"})
        self.assertEqual(leader_of, {"follower": "leader"})

    def test_pin_bound_promotes_after_60s(self):
        # The observed 112-298 s unbounded holds, unit form: a follower whose
        # total queue wait exceeds the bound is promoted back to group 0
        # (sorts by its own key again) instead of waiting behind everyone.
        leader = self._cold_req("leader", list(range(64)))
        follower_late = self._cold_req(
            "follower_late", list(range(64)) + [7, 8, 9], entry_offset_s=61.0
        )
        pinned, _ = self._policy()._compute_prefix_matches(
            [leader, follower_late], CacheAwarePolicy.LPM
        )
        self.assertNotIn("follower_late", pinned)

        # Below the bound the old behavior holds: the follower stays pinned.
        follower_recent = self._cold_req(
            "follower_recent", list(range(64)) + [7, 8, 9], entry_offset_s=30.0
        )
        pinned, leader_of = self._policy()._compute_prefix_matches(
            [leader, follower_recent], CacheAwarePolicy.LPM
        )
        self.assertEqual(pinned, {"follower_recent"})
        self.assertEqual(leader_of, {"follower_recent": "leader"})

    def test_pin_bound_zero_disables(self):
        # 0 restores the unbounded pin: a follower waiting an hour stays
        # pinned (the negative-branch contract of the flag). The leader map
        # is still built -- leader-relative placement under boost must not
        # depend on the pin bound.
        leader = self._cold_req("leader", list(range(64)))
        follower = self._cold_req("follower", list(range(64)), entry_offset_s=3600.0)
        pinned, leader_of = self._policy(max_pin_s=0.0)._compute_prefix_matches(
            [leader, follower], CacheAwarePolicy.LPM
        )
        self.assertEqual(pinned, {"follower"})
        self.assertEqual(leader_of, {"follower": "leader"})

    def test_unstamped_req_never_promotes(self):
        # first_queue_entry_time is None on disagg/synthetic reqs: the wait is
        # unmeasurable, so the pin keeps the pre-bound behavior instead of
        # crashing or promoting.
        leader = self._cold_req("leader", list(range(64)))
        follower = self._cold_req("follower", list(range(64)))
        self.assertIsNone(follower.first_queue_entry_time)
        pinned, _ = self._policy()._compute_prefix_matches(
            [leader, follower], CacheAwarePolicy.LPM
        )
        self.assertEqual(pinned, {"follower"})


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
        with patch.object(
            policy,
            "_compute_prefix_matches",
            return_value=({"follower"}, {"follower": "other"}),
        ):
            policy.calc_priority(queue)
        self.assertIs(follower.in_batch_dedup_pending, True)
        self.assertIs(other.in_batch_dedup_pending, False)

    def test_agnostic_path_clears_stale_verdicts(self):
        # Past the 128-queue fallback the active policy is FCFS and the sort
        # does not classify dedup; a flag left from an earlier sort would
        # exempt a non-follower from the invariant bound.
        policy = SchedulePolicy(
            policy="lpm",
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
    # testable without 128 completions; the formulas are window-agnostic.
    LN5 = math.log(5.0)

    def test_estimator_math_and_ema(self):
        # latencies = {0, 1, ..., 99} s: linear-interpolated p95 = 94.05,
        # p99 = 98.01, so tail width = 3.96 s and gamma_hat = ln5/3.96. One
        # update from gamma0 applies the 0.2 EMA weight to that estimate.
        # Guards the percentile definition (nearest-rank would give 95/98
        # -> width 3) and the beta placement (weighting the old value
        # instead of the estimate changes the result).
        adaptive = AdaptiveBoostGamma(
            gamma0=0.1, bin_size=256, max_jump_s=60.0, window=128, min_samples=64
        )
        gamma = adaptive.record_latencies(
            [float(t) for t in range(100)], rate_tokens_per_s=None
        )
        expected = 0.8 * 0.1 + 0.2 * (self.LN5 / 3.96)
        self.assertIsNotNone(gamma)
        self.assertAlmostEqual(gamma, expected, delta=expected * 1e-12)
        self.assertAlmostEqual(adaptive.gamma, expected, delta=expected * 1e-12)

    def test_jump_floor_bounds_max_boost(self):
        # The low clip is the max-jump invariant, not a raw gamma: with
        # k=256, rate=2000 tok/s (tau=0.128 s) and J=60 s, a degenerate fat
        # tail (gamma_hat ~ 0) must still converge to gamma ~ 0.077, where
        # b_gamma(k/rate) = 60 s. A raw gamma floor (1e-3) would let gamma
        # collapse and the newcomer jump grow unbounded with the tail.
        adaptive = AdaptiveBoostGamma(
            gamma0=0.09, bin_size=256, max_jump_s=60.0, window=16, min_samples=8
        )
        for _ in range(200):
            gamma = adaptive.record_latencies(
                [0.0] * 15 + [1e9], rate_tokens_per_s=2000.0
            )
        self.assertGreater(gamma, 0.07)
        self.assertLess(gamma, 0.08)
        jump_s = gamma_boost(256 / 2000.0, gamma)
        self.assertLessEqual(jump_s, 60.0)
        self.assertGreaterEqual(jump_s, 59.9, "floor must be tight, not slack")

    def test_jump_floor_tracks_rate_not_gamma(self):
        # Same jump budget at 10x the service rate: tau shrinks 10x, the
        # floor gamma moves (0.077 -> ~0.11) so the *jump* stays 60 s. A
        # fixed-gamma floor regression (one gamma for every rate) fails
        # here: at 0.077 the 20k-tok/s jump would be ~2.4x the budget.
        for rate, lo, hi in ((2000.0, 0.07, 0.08), (20000.0, 0.10, 0.11)):
            adaptive = AdaptiveBoostGamma(
                gamma0=0.09, bin_size=256, max_jump_s=60.0, window=16, min_samples=8
            )
            for _ in range(200):
                gamma = adaptive.record_latencies(
                    [0.0] * 15 + [1e9], rate_tokens_per_s=rate
                )
            self.assertGreater(gamma, lo)
            self.assertLess(gamma, hi)
            jump_s = gamma_boost(256 / rate, gamma)
            self.assertLessEqual(jump_s, 60.0)
            self.assertGreaterEqual(jump_s, 59.9)

    def test_thin_tail_hits_gamma_max_despite_floor(self):
        # An all-equal window is the degenerate thin tail: width 0 -> the
        # eps floor -> gamma_hat = ln5/eps clipped to GAMMA_MAX_S, above the
        # jump floor (0.077 here). The floor must not cap the FCFS-ward
        # direction, only the locality-ward collapse.
        adaptive = AdaptiveBoostGamma(
            gamma0=0.09, bin_size=256, max_jump_s=60.0, window=16, min_samples=8
        )
        for _ in range(200):
            gamma = adaptive.record_latencies(
                [30.0] * 16, rate_tokens_per_s=2000.0
            )
        # EMA approaches the clip asymptotically (0.8-weighted), so it
        # stalls an ulp or two below GAMMA_MAX_S.
        self.assertAlmostEqual(gamma, AdaptiveBoostGamma.GAMMA_MAX_S, places=10)

    def test_rate_unmeasured_uses_outer_floor(self):
        # Rate None (early boot, no positive token-delta interval yet): the
        # jump bound is not computable and the outer floor applies. The
        # sort gives boost 0 while the rate is None, so nothing is unbounded.
        adaptive = AdaptiveBoostGamma(
            gamma0=0.09, bin_size=256, max_jump_s=60.0, window=16, min_samples=8
        )
        for _ in range(200):
            gamma = adaptive.record_latencies(
                [0.0] * 15 + [1e9], rate_tokens_per_s=None
            )
        self.assertAlmostEqual(gamma, AdaptiveBoostGamma.GAMMA_MIN_S)

    def test_trajectory_log_reports_raw_fit_and_floor(self):
        # Regression (e2 c16 investigation, 2026-10-19): the log line printed
        # the post-floor gamma_hat and a width derived from it, so a
        # floor-pinned overload run logged a 4-20 s "tail" while the raw tail
        # was 86-560 s -- the floor-bound state was invisible and the
        # trajectory read as a converged fit. The line must carry the raw
        # pre-floor fit, which tail it came from, the floor, and the share
        # of updates clamped to that floor (100% = the J bound, not the fit,
        # set gamma the whole run -- the ada@77188 state).
        adaptive = AdaptiveBoostGamma(
            gamma0=0.09, bin_size=256, max_jump_s=60.0, window=16, min_samples=8
        )
        with self.assertLogs(
            "sglang.srt.managers.schedule_policy", level="INFO"
        ) as cm:
            # Degenerate fat tail: raw fit ~ ln5/(6e8) far below the floor.
            for _ in range(200):
                adaptive.record_latencies(
                    [0.0] * 15 + [1e9], rate_tokens_per_s=2000.0
                )
        fit_m = re.search(r"fit=([\d.eE+-]+)", cm.output[-1])
        floor_m = re.search(r"floor=([\d.eE+-]+)", cm.output[-1])
        pinned_m = re.search(r"pinned=(\d+)%", cm.output[-1])
        self.assertIsNotNone(fit_m, "log must report the raw fit")
        self.assertIsNotNone(floor_m, "log must report the floor")
        self.assertIsNotNone(pinned_m, "log must report the floor-pinned share")
        fit = float(fit_m.group(1))
        floor = float(floor_m.group(1))
        self.assertLess(fit, 1e-6, "fit must be the pre-floor raw estimate")
        self.assertGreater(floor, 0.07, "floor is the J=60 jump bound at 2k tok/s")
        self.assertGreaterEqual(
            int(pinned_m.group(1)), 99, "every update must clamp to the floor"
        )
        self.assertIn("tail", cm.output[-1])

    def test_warmup_returns_none_below_min_samples(self):
        # Below min_samples percentiles are noise; the caller must be able to
        # skip (no gamma move, no eval-log mutation) until the window fills.
        adaptive = AdaptiveBoostGamma(
            gamma0=0.09, bin_size=256, max_jump_s=60.0, window=16, min_samples=8
        )
        self.assertIsNone(
            adaptive.record_latencies([5.0] * 7, rate_tokens_per_s=None)
        )
        self.assertEqual(adaptive.gamma, 0.09)
        self.assertIsNotNone(
            adaptive.record_latencies([5.0], rate_tokens_per_s=None)
        )

    def test_gamma0_outside_outer_bounds_clips_at_init(self):
        # A tokens-scale gamma0 (1e-4 per-token) is below the 1/s outer
        # floor; it must clip up at construction, not run unclipped until
        # the first completion arrives (mode-mismatch guard for the
        # default-resolution path).
        adaptive = AdaptiveBoostGamma(
            gamma0=1e-4, bin_size=256, max_jump_s=60.0
        )
        self.assertEqual(adaptive.gamma, AdaptiveBoostGamma.GAMMA_MIN_S)


class TestCompletedLatencyFeed(CustomTestCase):
    """record_completed_latencies: the scheduler's gamma-Ada feed. Only
    finished reqs with both stamps count, in wall seconds; a moved gamma
    rebinds the live sort's gamma."""

    def _policy(self, **kw):
        return SchedulePolicy(
            policy="boost",
            tree_cache=RadixCache.create_simulated(),
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
            boost_gamma=0.09,
            boost_bin_size=256,
            boost_adaptive_gamma=True,
            boost_clock="wall",
            **kw,
        )

    @staticmethod
    def _finished_req(rid, *, recv_s, done_s):
        req = _make_wall_req(rid, 256, entry_s=recv_s)
        req.time_stats.set_scheduler_recv_time(recv_s)
        req.time_stats.set_completion_time(done_s)
        req.finished_reason = FINISH_LENGTH(length=8)
        return req

    def test_feed_filters_unfinished_and_unstamped(self):
        # The tail fit's population contract: an unfinished req (still
        # streaming) and a finished-but-unstamped one (completion_time 0 --
        # the eval log's e2e_s guard) must stay out of the window, else the
        # fit mixes in-flight and phantom latencies.
        policy = self._policy()
        streaming = _make_wall_req("streaming", 256, entry_s=0.0)
        # completion_time left at its 0.0 default (set_completion_time
        # treats 0.0 as "unset" and stamps perf_counter, so poke the field).
        unstamped = self._finished_req("unstamped", recv_s=100.0, done_s=100.0)
        unstamped.time_stats.completion_time = 0.0
        valid = self._finished_req("valid", recv_s=100.0, done_s=160.0)
        policy.record_completed_latencies([streaming, unstamped, valid])
        self.assertEqual(list(policy.adaptive_gamma._samples), [(60.0, 256)])

    def test_feed_moves_gamma_and_rebinds_sort_gamma(self):
        # boost + flag -> a full window of completions moves gamma and the
        # moved value rebinds the live sort's gamma (policy.boost_gamma),
        # which is the whole point of the feed.
        policy = self._policy()
        reqs = [
            self._finished_req(f"r{i}", recv_s=100.0, done_s=100.0 + t)
            for i, t in enumerate(range(1, 35))
        ]
        gamma = policy.record_completed_latencies(reqs)
        self.assertIsNotNone(gamma)
        self.assertNotEqual(gamma, 0.09)
        self.assertEqual(policy.boost_gamma, gamma)
        self.assertEqual(len(policy.adaptive_gamma._samples), 34)

    def test_off_and_mismatched_paths(self):
        # Without the flag gamma-Ada stays off (the fixed-gamma path must
        # be bit-identical to before); on another policy it stays off; and
        # a tokens-clock construction is a unit mismatch (the feed is wall
        # seconds, gamma would be per-token) -> rejected at construction.
        def make(**kw):
            return SchedulePolicy(
                policy=kw.pop("policy", "boost"),
                tree_cache=RadixCache.create_simulated(),
                enable_hierarchical_cache=True,
                enable_priority_scheduling=False,
                schedule_low_priority_values_first=False,
                boost_gamma=0.09,
                boost_bin_size=256,
                **kw,
            )

        off = make()
        self.assertIsNone(off.adaptive_gamma)
        self.assertIsNone(off.record_completed_latencies([]))
        self.assertEqual(off.boost_gamma, 0.09)

        other = make(policy="fcfs", boost_adaptive_gamma=True)
        self.assertIsNone(other.adaptive_gamma)

        with self.assertRaises(ValueError):
            make(boost_adaptive_gamma=True, boost_clock="tokens")


if __name__ == "__main__":
    unittest.main()
