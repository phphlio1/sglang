"""Rollback of a prefill admission whose prepare_for_extend hit KV OOM.

Regression guard for the H200 scheduler deaths (ada j50, seed 77188): the
rollback must release exactly the locks this admission pass took, keep a
chunk continuation tracked by self.chunked_req alone (requeueing it too
admitted it twice in one batch and corrupted the allocator), undo the
inflight_middle_chunks increment so the req can still be released, and
restore extend_range so the next pass's stash gate does not checkpoint
a chunk that was never dispatched.
"""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import Req  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.mem_cache.base_prefix_cache import DecLockRefParams  # noqa: E402
from sglang.srt.sampling.sampling_params import SamplingParams  # noqa: E402
from sglang.srt.utils.common import Range  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_req(rid: str) -> Req:
    return Req(rid, "", array("q", [1, 2, 3]), SamplingParams())


def _make_scheduler(*, waiting_queue, chunked_req) -> Scheduler:
    sched = Scheduler.__new__(Scheduler)
    sched.waiting_queue = list(waiting_queue)
    sched.chunked_req = chunked_req
    sched.tree_cache = SimpleNamespace(dec_lock_ref=Mock())

    def _free_row(req):
        # Mirrors ReqToTokenPool.free's visible effect on the req.
        req.kv.req_pool_idx = None

    sched.req_to_token_pool = SimpleNamespace(free=Mock(side_effect=_free_row))
    return sched


def _admit_fresh(sched: Scheduler, req: Req) -> DecLockRefParams:
    """Mirror add_one_req's side effects up to just before prepare: the
    adder matched a node and _req_inc_lock_ref rebound the receipt."""
    req.last_node = object()
    admission_receipt = DecLockRefParams()
    req.lock_receipt = admission_receipt
    req.kv.req_pool_idx = 7  # alloc_req_slots registered a row
    sched.waiting_queue = [r for r in sched.waiting_queue if r is not req]
    return admission_receipt


def _rollback(
    sched: Scheduler, *, new_chunked_req, can_run_list, receipts, ranges, fresh_rows
):
    sched._rollback_failed_prefill_admission(
        adder=SimpleNamespace(new_chunked_req=new_chunked_req),
        can_run_list=can_run_list,
        # The scheduler captures this list before prepare_for_extend, i.e.
        # before alloc_req_slots registers the reqs' rows.
        fresh_row_reqs=fresh_rows,
        receipts_before=receipts,
        extend_ranges_before=ranges,
    )


class TestPrefillOomRollback(CustomTestCase):
    def test_fresh_admission_lock_released_and_receipt_restored(self):
        req = _make_req("fresh")
        sched = _make_scheduler(waiting_queue=[req], chunked_req=None)
        receipts = sched._snapshot_admission_lock_receipts()
        ranges = sched._snapshot_admission_extend_ranges()
        receipt_before = req.lock_receipt

        fresh_rows = [req]  # captured before prepare registered the row
        admission_receipt = _admit_fresh(sched, req)

        _rollback(
            sched,
            new_chunked_req=None,
            can_run_list=[req],
            receipts=receipts,
            ranges=ranges,
            fresh_rows=fresh_rows,
        )

        sched.tree_cache.dec_lock_ref.assert_called_once_with(
            req.last_node, admission_receipt
        )
        self.assertIs(req.lock_receipt, receipt_before)
        self.assertIsNone(req.extend_range)
        sched.req_to_token_pool.free.assert_called_once_with(req)
        self.assertIsNone(req.kv.req_pool_idx)
        self.assertEqual([r.rid for r in sched.waiting_queue], ["fresh"])

    def test_continuation_lock_survives_and_stays_pointer_tracked(self):
        cont = _make_req("cont")
        cont.kv.req_pool_idx = 5  # holds KV from its first chunk
        sched = _make_scheduler(waiting_queue=[], chunked_req=cont)
        receipts = sched._snapshot_admission_lock_receipts()
        ranges = sched._snapshot_admission_extend_ranges()
        receipt_before = cont.lock_receipt

        # add_chunked_req admits the continuation without taking a new lock;
        # the scheduler's pre-prepare bookkeeping still runs.
        cont.last_node = object()
        cont.inflight_middle_chunks += 1

        _rollback(
            sched,
            new_chunked_req=None,
            can_run_list=[cont],
            receipts=receipts,
            ranges=ranges,
            fresh_rows=[],
        )

        sched.tree_cache.dec_lock_ref.assert_not_called()
        self.assertIs(cont.lock_receipt, receipt_before)
        self.assertIs(sched.chunked_req, cont)
        self.assertNotIn(cont, sched.waiting_queue)
        self.assertEqual(cont.inflight_middle_chunks, 0)
        sched.req_to_token_pool.free.assert_not_called()
        self.assertIsNotNone(cont.kv.req_pool_idx)

    def test_new_chunked_req_requeued_with_bookkeeping_undone(self):
        req = _make_req("giant")
        sched = _make_scheduler(waiting_queue=[req], chunked_req=None)
        receipts = sched._snapshot_admission_lock_receipts()
        ranges = sched._snapshot_admission_extend_ranges()

        fresh_rows = [req]  # captured before prepare registered the row
        admission_receipt = _admit_fresh(sched, req)
        # _commit_prefill_admission marked the req as the new chunked req,
        # counted the in-flight middle chunk, and set its first extend_range.
        sched.chunked_req = req
        req.inflight_middle_chunks += 1
        req.set_extend_range(0, 4096)

        _rollback(
            sched,
            new_chunked_req=req,
            can_run_list=[req],
            receipts=receipts,
            ranges=ranges,
            fresh_rows=fresh_rows,
        )

        self.assertIsNone(sched.chunked_req)
        self.assertEqual([r.rid for r in sched.waiting_queue], ["giant"])
        self.assertEqual(req.inflight_middle_chunks, 0)
        self.assertIsNone(req.extend_range)
        sched.tree_cache.dec_lock_ref.assert_called_once_with(
            req.last_node, admission_receipt
        )

    def test_rollback_restores_extend_range_so_stash_gate_stays_closed(self):
        # add_chunked_req sets extend_range before prepare_for_extend runs;
        # an OOM rollback that leaves it makes the next pass's stash gate
        # (extend_range.end > len(prefix_indices)) checkpoint a chunk that
        # was never dispatched, binding unwritten req_to_token rows into
        # prefix_indices and the radix tree -- the H200 illegal-access crash.
        cont = _make_req("cont")
        cont.kv.req_pool_idx = 5
        cont.prefix_indices = torch.arange(8192, dtype=torch.int64)
        parked = Range(4096, 8192)  # end == len(prefix_indices): nothing new
        cont.extend_range = parked
        sched = _make_scheduler(waiting_queue=[], chunked_req=cont)
        receipts = sched._snapshot_admission_lock_receipts()
        ranges = sched._snapshot_admission_extend_ranges()

        # The failed pass: add_chunked_req rebased extend_range onto the
        # chunk it was about to dispatch when prepare hit the OOM.
        cont.set_extend_range(8192, 8192 + 4096)
        cont.inflight_middle_chunks += 1

        _rollback(
            sched,
            new_chunked_req=None,
            can_run_list=[cont],
            receipts=receipts,
            ranges=ranges,
            fresh_rows=[],
        )

        self.assertIs(cont.extend_range, parked)
        self.assertFalse(
            cont.extend_range.end > len(cont.prefix_indices),
            "stash gate must stay closed after a rolled-back admission",
        )

    def test_mixed_batch_rolls_back_only_fresh_admissions(self):
        cont = _make_req("cont")
        cont.kv.req_pool_idx = 5
        fresh_a = _make_req("a")
        fresh_b = _make_req("b")
        sched = _make_scheduler(waiting_queue=[fresh_a, fresh_b], chunked_req=cont)
        receipts = sched._snapshot_admission_lock_receipts()
        ranges = sched._snapshot_admission_extend_ranges()

        fresh_rows = [fresh_a, fresh_b]  # captured before rows were registered
        receipt_a = _admit_fresh(sched, fresh_a)
        receipt_b = _admit_fresh(sched, fresh_b)
        cont.last_node = object()
        cont.inflight_middle_chunks += 1

        _rollback(
            sched,
            new_chunked_req=None,
            can_run_list=[cont, fresh_a, fresh_b],
            receipts=receipts,
            ranges=ranges,
            fresh_rows=fresh_rows,
        )

        self.assertEqual(
            sched.tree_cache.dec_lock_ref.call_count,
            2,
            "continuation lock must not be released",
        )
        sched.req_to_token_pool.free.assert_any_call(fresh_a)
        sched.req_to_token_pool.free.assert_any_call(fresh_b)
        self.assertEqual(sched.req_to_token_pool.free.call_count, 2)
        # Fresh reqs requeue ahead of untouched waiters; the continuation
        # stays tracked by the pointer alone.
        self.assertEqual([r.rid for r in sched.waiting_queue], ["a", "b"])
        self.assertIs(sched.chunked_req, cont)
        self.assertEqual(cont.inflight_middle_chunks, 0)
        self.assertIs(fresh_a.lock_receipt, receipts["a"])
        self.assertIs(fresh_b.lock_receipt, receipts["b"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
