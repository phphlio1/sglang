"""Decode time-slicing for first-token-starved waiters (--boost-preempt-wait-s)."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers import schedule_batch  # noqa: E402
from sglang.srt.managers.decode_preemption import (  # noqa: E402
    PREEMPT_COOLDOWN_S,
    DecodePreemptor,
    plan_decode_preemption,
)
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch  # noqa: E402
from sglang.srt.managers.schedule_policy import CLIP_MAX_NEW_TOKENS  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.sampling.sampling_params import SamplingParams  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE = 64
MAX_NEW = 8192  # above the clip, so every reserve is CLIP * ratio


def _req(rid, *, input_len, output_len=0, device_prefix=0, queued_at=None):
    req = Req(
        rid,
        "",
        array("q", [1] * input_len),
        SamplingParams(max_new_tokens=MAX_NEW),
    )
    req.output_ids = array("q", [2] * output_len)
    req.prefix_indices = torch.zeros(device_prefix, dtype=torch.int64)
    req.first_queue_entry_time = queued_at
    return req


def _running(rid, *, input_len, output_len):
    return _req(rid, input_len=input_len, output_len=output_len)


def _plan(waiting, running, have, **kw):
    args = dict(
        now_s=100.0,
        waiting_queue=waiting,
        running_reqs=running,
        have_tokens=have,
        new_token_ratio=0.5,
        page_size=PAGE,
        wait_s=30.0,
        quantum_tokens=512,
        max_per_req=2,
    )
    args.update(kw)
    return plan_decode_preemption(**args)


class TestPlanDecodePreemption(CustomTestCase):
    def test_trigger_requires_a_memory_deficit_on_the_adder_budget(self):
        # The deficit must use PrefillAdder's budget: available + evictable
        # minus every running request's clipped decode reserve. A waiter that
        # fits is starving for another reason (e.g. the prefill delayer), so
        # releasing decodes would only add stalls.
        waiter = _req("w", input_len=100_000, device_prefix=40_000, queued_at=50.0)
        need = 100_000 - 40_000 + CLIP_MAX_NEW_TOKENS + PAGE
        running = [_running("r", input_len=90_000, output_len=4_000)]
        reserve = min(MAX_NEW - 4_000, CLIP_MAX_NEW_TOKENS) * 0.5
        self.assertIsNone(_plan([waiter], running, have=int(need + reserve)))
        plan = _plan([waiter], running, have=int(need + reserve) - 1)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.target_rid, "w")
        self.assertEqual(plan.deficit_tokens, 1)

    def test_victims_are_the_minimal_longest_attained_prefix(self):
        # Heavy-tailed outputs: the most decoded request has the longest
        # expected remaining hold, so release in that order and stop as soon
        # as the waiter fits -- not every eligible decode.
        waiter = _req("w", input_len=150_000, queued_at=0.0)
        running = [
            _running("young", input_len=100_000, output_len=100),  # < quantum
            _running("mid", input_len=60_000, output_len=3_000),
            _running("short", input_len=60_000, output_len=800),
            _running("long", input_len=60_000, output_len=5_000),
        ]
        plan = _plan([waiter], running, have=0)
        self.assertEqual(
            [running[i].rid for i in plan.victim_indices], ["long", "mid", "short"]
        )
        # Room for most of the need already: one release suffices.
        plan = _plan([waiter], running, have=120_000)
        self.assertEqual([running[i].rid for i in plan.victim_indices], ["long"])

    def test_no_plan_when_eligible_releases_cannot_cover_the_deficit(self):
        waiter = _req("w", input_len=250_000, queued_at=0.0)
        running = [_running("r", input_len=20_000, output_len=1_000)]
        self.assertIsNone(_plan([waiter], running, have=0))

    def test_quantum_doubles_per_release_and_the_cap_ends_releases(self):
        # Progress guarantee: after k releases a request must decode
        # quantum * 2**k tokens since its latest admission before the next,
        # and max_per_req releases end it.
        waiter = _req("w", input_len=100_000, queued_at=0.0)
        r = _running("r", input_len=200_000, output_len=3_000)
        r.num_preemptions = 1
        r.preempt_resume_output_len = 3_000 - 1_023  # 1023 < 512 << 1
        self.assertIsNone(_plan([waiter], [r], have=0))
        r.preempt_resume_output_len = 3_000 - 1_024
        self.assertIsNotNone(_plan([waiter], [r], have=0))
        r.num_preemptions = 2  # == max_per_req
        r.preempt_resume_output_len = 0
        self.assertIsNone(_plan([waiter], [r], have=0))

    def test_only_first_token_waiters_trigger(self):
        # A released decode re-queues with output already streamed; letting it
        # trigger releases of its own would cascade stalls between decodes.
        released = _req("rel", input_len=100_000, output_len=2_000, queued_at=0.0)
        running = [_running("r", input_len=200_000, output_len=6_000)]
        self.assertIsNone(_plan([released], running, have=0))
        fresh = _req("w", input_len=100_000, queued_at=0.0)
        plan = _plan([released, fresh], running, have=0)
        self.assertEqual(plan.target_rid, "w")
        # Not yet starving: no trigger.
        recent = _req("w2", input_len=100_000, queued_at=80.0)
        self.assertIsNone(_plan([recent], running, have=0))


class TestDecodePreemptor(CustomTestCase):
    def _preemptor(self):
        return DecodePreemptor(
            wait_s=30.0, quantum_tokens=512, max_per_req=2, dp_rank=0
        )

    def test_reserved_waiter_heads_the_queue_until_admitted(self):
        # The released memory must reach the waiter: it heads the sorted queue
        # (the misfit break then stops later arrivals from taking the memory)
        # until admitted; admission ends the reservation and closes the
        # released decode's stall.
        pre = self._preemptor()
        waiter = _req("w", input_len=100_000, queued_at=0.0)
        victim = _running("v", input_len=200_000, output_len=6_000)
        plan = pre.maybe_plan(
            now_s=100.0,
            waiting_queue=[waiter],
            running_reqs=[victim],
            have_tokens=0,
            new_token_ratio=0.5,
            page_size=PAGE,
        )
        pre.on_preempted(plan, [victim], now_s=100.0)
        self.assertEqual(victim.num_preemptions, 1)

        a, b = _req("a", input_len=10), _req("b", input_len=10)
        queue = [a, b, waiter, victim]
        pre.promote_target(queue)
        self.assertEqual([r.rid for r in queue], ["w", "a", "b", "v"])

        # Within the cooldown no second plan, even with a starving waiter.
        self.assertIsNone(
            pre.maybe_plan(
                now_s=100.0 + PREEMPT_COOLDOWN_S / 2,
                waiting_queue=queue,
                running_reqs=[_running("x", input_len=200_000, output_len=9_000)],
                have_tokens=0,
                new_token_ratio=0.5,
                page_size=PAGE,
            )
        )

        pre.on_admitted([waiter, victim], now_s=112.5)
        self.assertAlmostEqual(victim.preempt_stall_s, 12.5)
        self.assertIsNone(victim.preempt_stall_start_s)
        self.assertEqual(victim.preempt_resume_output_len, 6_000)
        queue = [a, b]
        pre.promote_target(queue)
        self.assertEqual([r.rid for r in queue], ["a", "b"])

    def test_off_and_unsupported_configurations(self):
        common = dict(quantum_tokens=512, max_per_req=2, dp_rank=0)
        ok = dict(
            tree_supports_mamba=False,
            tree_supports_swa=False,
            disaggregation_mode="null",
        )
        # Off must construct nothing (the scheduler path stays bit-identical).
        self.assertIsNone(DecodePreemptor.maybe_create(wait_s=0.0, **common, **ok))
        self.assertIsNotNone(DecodePreemptor.maybe_create(wait_s=30.0, **common, **ok))
        for bad in (
            dict(ok, tree_supports_swa=True),
            dict(ok, tree_supports_mamba=True),
            dict(ok, disaggregation_mode="decode"),
        ):
            with self.assertRaises(ValueError):
                DecodePreemptor.maybe_create(wait_s=30.0, **common, **bad)


class TestPreemptDecodeRelease(CustomTestCase):
    def test_release_inserts_into_the_tree_and_keeps_the_output(self):
        # The point of preempting instead of retracting: the KV goes into the
        # radix tree (release_kv_cache is_insert=True), so the request resumes
        # from its own cache; OOM retraction drops the uncached tail.
        keep = _running("keep", input_len=10, output_len=5)
        victim = _running("victim", input_len=10, output_len=7)
        batch = ScheduleBatch.__new__(ScheduleBatch)
        batch.reqs = [keep, victim]
        batch.tree_cache = object()
        batch.hisparse_coordinator = None
        batch.filter_batch = Mock()

        victim.kv.req_pool_idx = 3  # decoding: holds a KV row

        def _release(req, tree_cache, is_insert=True):
            self.assertTrue(is_insert)
            # Mirrors req_to_token_pool.free + mark_kv_released.
            req.kv.req_pool_idx = None
            req.kv.mark_kv_released()

        with patch.object(
            schedule_batch, "release_kv_cache", side_effect=_release
        ) as rel:
            preempted = batch.preempt_decode([1])
        rel.assert_called_once()
        self.assertEqual(preempted, [victim])
        batch.filter_batch.assert_called_once_with(keep_indices=[0])
        self.assertTrue(victim.is_retracted)
        self.assertEqual(victim.retraction_count, 1)
        self.assertEqual(len(victim.output_ids), 7)


class TestSchedulerPreemptGlue(CustomTestCase):
    def test_released_decodes_rejoin_the_waiting_queue(self):
        # A released request is no longer in any batch; if it does not
        # re-enter the waiting queue it never finishes.
        sched = Scheduler.__new__(Scheduler)
        waiter = _req("w", input_len=100_000, queued_at=0.0)
        victim = _running("v", input_len=200_000, output_len=6_000)
        sched.waiting_queue = [waiter]
        sched.decode_preemptor = DecodePreemptor(
            wait_s=30.0, quantum_tokens=512, max_per_req=2, dp_rank=0
        )
        sched.tree_cache = SimpleNamespace(available_and_evictable_size=lambda: 0)
        sched.new_token_ratio_tracker = SimpleNamespace(current=0.5)
        sched.page_size = PAGE
        sched.decode_offload_manager = None
        requeued = []
        sched._add_request_to_queue = lambda req, is_retracted=False: requeued.append(
            (req, is_retracted)
        )
        batch = SimpleNamespace(
            reqs=[victim], preempt_decode=Mock(side_effect=lambda idx: [victim])
        )
        with patch("time.perf_counter", return_value=100.0):
            sched._preempt_for_starving_waiters(batch)
        batch.preempt_decode.assert_called_once_with((0,))
        self.assertEqual(requeued, [(victim, True)])
        self.assertEqual(victim.preempt_stall_start_s, 100.0)


if __name__ == "__main__":
    unittest.main()
