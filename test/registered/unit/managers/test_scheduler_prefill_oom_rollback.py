"""Rollback of a prefill admission whose prepare_for_extend hit KV OOM.

Regression guard for the H200 scheduler deaths (ada j50, seed 77188): the
rollback must release exactly the locks this admission pass took, keep a
chunk continuation tracked by self.chunked_req alone (requeueing it too
admitted it twice in one batch and corrupted the allocator), undo the
inflight_middle_chunks increment so the req can still be released, and
restore extend_end so the next pass's stash gate does not checkpoint
a chunk that was never dispatched.
"""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import Req  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.mem_cache.base_prefix_cache import (  # noqa: E402
    DecLockRefParams,
    TreeLock,
)
from sglang.srt.sampling.sampling_params import SamplingParams  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_req(rid: str) -> Req:
    return Req(rid, "", array("q", [1, 2, 3]), SamplingParams())


def _make_scheduler(*, waiting_queue, chunked_req) -> Scheduler:
    sched = Scheduler.__new__(Scheduler)
    sched.waiting_queue = list(waiting_queue)
    sched.chunked_req = chunked_req
    sched.tree_cache = SimpleNamespace(unlock=Mock())

    def _free_row(req):
        # Mirrors ReqToTokenPool.free's visible effect on the req.
        req.kv.req_pool_idx = None

    sched.req_to_token_pool = SimpleNamespace(free=Mock(side_effect=_free_row))
    return sched


def _admit_fresh(sched: Scheduler, req: Req) -> TreeLock:
    """Mirror _commit_prefill_admission's side effects up to just before
    prepare: the adder matched a node and took a fresh tree lock."""
    req.last_node = object()
    admission_receipt = TreeLock(req.last_node, DecLockRefParams())
    req.lock = admission_receipt
    req.kv.req_pool_idx = 7  # alloc_req_slots registered a row
    sched.waiting_queue = [r for r in sched.waiting_queue if r is not req]
    return admission_receipt


def _rollback(
    sched: Scheduler, *, prev_chunked_req, can_run_list, receipts, ranges, fresh_rows
):
    sched._rollback_failed_prefill_admission(
        prev_chunked_req=prev_chunked_req,
        can_run_list=can_run_list,
        # The scheduler captures this list before prepare_for_extend, i.e.
        # before alloc_req_slots registers the reqs' rows.
        fresh_row_reqs=fresh_rows,
        locks_before=receipts,
        extend_ends_before=ranges,
    )


class TestPrefillOomRollback(CustomTestCase):
    def test_fresh_admission_lock_released_and_receipt_restored(self):
        req = _make_req("fresh")
        sched = _make_scheduler(waiting_queue=[req], chunked_req=None)
        receipts = sched._snapshot_admission_locks()
        ranges = sched._snapshot_admission_extend_ends()
        receipt_before = req.lock

        fresh_rows = [req]  # captured before prepare registered the row
        admission_receipt = _admit_fresh(sched, req)

        _rollback(
            sched,
            prev_chunked_req=None,
            can_run_list=[req],
            receipts=receipts,
            ranges=ranges,
            fresh_rows=fresh_rows,
        )

        sched.tree_cache.unlock.assert_called_once_with(admission_receipt)
        self.assertIs(req.lock, receipt_before)
        self.assertIsNone(req.extend_end)
        sched.req_to_token_pool.free.assert_called_once_with(req)
        self.assertIsNone(req.kv.req_pool_idx)
        self.assertEqual([r.rid for r in sched.waiting_queue], ["fresh"])

    def test_continuation_lock_survives_and_stays_pointer_tracked(self):
        cont = _make_req("cont")
        cont.kv.req_pool_idx = 5  # holds KV from its first chunk
        sched = _make_scheduler(waiting_queue=[], chunked_req=cont)
        receipts = sched._snapshot_admission_locks()
        ranges = sched._snapshot_admission_extend_ends()
        receipt_before = cont.lock

        # add_chunked_req admits the continuation without taking a new lock;
        # the scheduler's pre-prepare bookkeeping still runs.
        cont.last_node = object()
        cont.inflight_middle_chunks += 1

        _rollback(
            sched,
            prev_chunked_req=cont,
            can_run_list=[cont],
            receipts=receipts,
            ranges=ranges,
            fresh_rows=[],
        )

        sched.tree_cache.unlock.assert_not_called()
        self.assertIs(cont.lock, receipt_before)
        self.assertIs(sched.chunked_req, cont)
        self.assertNotIn(cont, sched.waiting_queue)
        self.assertEqual(cont.inflight_middle_chunks, 0)
        sched.req_to_token_pool.free.assert_not_called()
        self.assertIsNotNone(cont.kv.req_pool_idx)

    def test_new_chunked_req_requeued_with_bookkeeping_undone(self):
        req = _make_req("giant")
        sched = _make_scheduler(waiting_queue=[req], chunked_req=None)
        receipts = sched._snapshot_admission_locks()
        ranges = sched._snapshot_admission_extend_ends()

        fresh_rows = [req]  # captured before prepare registered the row
        admission_receipt = _admit_fresh(sched, req)
        # _commit_prefill_admission marked the req as the new chunked req,
        # counted the in-flight middle chunk, and set its first extend_end.
        sched.chunked_req = req
        req.inflight_middle_chunks += 1
        req.extend_end = 4096

        _rollback(
            sched,
            prev_chunked_req=None,
            can_run_list=[req],
            receipts=receipts,
            ranges=ranges,
            fresh_rows=fresh_rows,
        )

        self.assertIsNone(sched.chunked_req)
        self.assertEqual([r.rid for r in sched.waiting_queue], ["giant"])
        self.assertEqual(req.inflight_middle_chunks, 0)
        self.assertIsNone(req.extend_end)
        sched.tree_cache.unlock.assert_called_once_with(admission_receipt)

    def test_rollback_restores_extend_end_so_stash_gate_stays_closed(self):
        # add_chunked_req sets extend_end before prepare_for_extend runs; an
        # OOM rollback that leaves it makes the next pass's stash gate
        # (extend_end > prefix_len) checkpoint a chunk that was never
        # dispatched, binding unwritten req_to_token rows into the radix tree.
        cont = _make_req("cont")
        cont.kv.req_pool_idx = 5
        cont.prefix_len = 8192
        parked = 8192  # extend_end == prefix_len: nothing new
        cont.extend_end = parked
        sched = _make_scheduler(waiting_queue=[], chunked_req=cont)
        receipts = sched._snapshot_admission_locks()
        ranges = sched._snapshot_admission_extend_ends()

        # The failed pass: add_chunked_req moved extend_end onto the chunk
        # it was about to dispatch when prepare hit the OOM.
        cont.extend_end = 8192 + 4096
        cont.inflight_middle_chunks += 1

        _rollback(
            sched,
            prev_chunked_req=cont,
            can_run_list=[cont],
            receipts=receipts,
            ranges=ranges,
            fresh_rows=[],
        )

        self.assertEqual(cont.extend_end, parked)
        self.assertFalse(
            cont.extend_end > cont.prefix_len,
            "stash gate must stay closed after a rolled-back admission",
        )

    def test_mixed_batch_rolls_back_only_fresh_admissions(self):
        cont = _make_req("cont")
        cont.kv.req_pool_idx = 5
        fresh_a = _make_req("a")
        fresh_b = _make_req("b")
        sched = _make_scheduler(waiting_queue=[fresh_a, fresh_b], chunked_req=cont)
        receipts = sched._snapshot_admission_locks()
        ranges = sched._snapshot_admission_extend_ends()

        fresh_rows = [fresh_a, fresh_b]  # captured before rows were registered
        receipt_a = _admit_fresh(sched, fresh_a)
        receipt_b = _admit_fresh(sched, fresh_b)
        cont.last_node = object()
        cont.inflight_middle_chunks += 1

        _rollback(
            sched,
            prev_chunked_req=cont,
            can_run_list=[cont, fresh_a, fresh_b],
            receipts=receipts,
            ranges=ranges,
            fresh_rows=fresh_rows,
        )

        self.assertEqual(
            sched.tree_cache.unlock.call_count,
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
        self.assertIs(fresh_a.lock, receipts["a"])
        self.assertIs(fresh_b.lock, receipts["b"])

    def test_final_chunk_continuation_stays_tracked_not_requeued(self):
        # add_chunked_req returns None for a final chunk, so the pass clears
        # self.chunked_req before prepare. Requeued, the continuation kept its row
        # and lock; its next admission overwrote req.lock and the old lock pinned
        # its prefix forever (protected KV left at idle on the H200 replay).
        cont = _make_req("cont")
        cont.kv.req_pool_idx = 5
        sched = _make_scheduler(waiting_queue=[], chunked_req=cont)
        receipts = sched._snapshot_admission_locks()
        ranges = sched._snapshot_admission_extend_ends()
        receipt_before = cont.lock
        cont.extend_end = 4096
        sched.chunked_req = None  # final chunk: add_chunked_req returned None

        _rollback(
            sched,
            prev_chunked_req=cont,
            can_run_list=[cont],
            receipts=receipts,
            ranges=ranges,
            fresh_rows=[],
        )

        self.assertIs(sched.chunked_req, cont)
        self.assertNotIn(cont, sched.waiting_queue)
        self.assertIs(cont.lock, receipt_before)
        self.assertEqual(cont.inflight_middle_chunks, 0)
        sched.tree_cache.unlock.assert_not_called()
        sched.req_to_token_pool.free.assert_not_called()

    def test_final_chunk_with_new_chunked_req_restores_the_continuation(self):
        cont = _make_req("cont")
        cont.kv.req_pool_idx = 5
        giant = _make_req("giant")
        sched = _make_scheduler(waiting_queue=[giant], chunked_req=cont)
        receipts = sched._snapshot_admission_locks()
        ranges = sched._snapshot_admission_extend_ends()
        # The final chunk cleared self.chunked_req, so a waiting giant became
        # the new chunked req in the same pass.
        sched.chunked_req = None
        admission_receipt = _admit_fresh(sched, giant)
        sched.chunked_req = giant
        giant.inflight_middle_chunks += 1

        _rollback(
            sched,
            prev_chunked_req=cont,
            can_run_list=[cont, giant],
            receipts=receipts,
            ranges=ranges,
            fresh_rows=[giant],
        )

        self.assertIs(sched.chunked_req, cont)
        self.assertEqual([r.rid for r in sched.waiting_queue], ["giant"])
        self.assertEqual(giant.inflight_middle_chunks, 0)
        sched.tree_cache.unlock.assert_called_once_with(admission_receipt)
        sched.req_to_token_pool.free.assert_called_once_with(giant)


class TestPrefillOomHold(CustomTestCase):
    """The sticky prefill-OOM hold: after a prefill-step OOM the scheduler
    must stop re-attempting the admission every pass (the H200 ada-j50
    hammer ran 5 failed 4096-token extends in 0.4 s right before the CUDA
    illegal access) and resume only once the pool can fit the alloc that
    failed."""

    OOM_MSG = (
        "Prefill out of memory. Try to lower your batch size.\n"
        "Try to allocate 4096 tokens.\n"
        "Available full tokens: 256 (full_available_size=256 "
        "+ full_evictable_size_=0)\n"
    )

    @staticmethod
    def _make_hold_scheduler(pool_tokens: int) -> Scheduler:
        sched = Scheduler.__new__(Scheduler)
        sched.page_size = 64
        sched._prefill_oom_hold_tokens = None
        sched._prefill_oom_hold_passes = 0
        sched.tree_cache = SimpleNamespace(
            available_and_evictable_size=Mock(return_value=pool_tokens)
        )
        return sched

    def test_arm_holds_requested_tokens_plus_a_page(self):
        sched = self._make_hold_scheduler(pool_tokens=256)
        sched._arm_prefill_oom_hold(RuntimeError(self.OOM_MSG))
        self.assertEqual(sched._prefill_oom_hold_tokens, 4096 + 64)
        self.assertEqual(sched._prefill_oom_hold_passes, 200)

    def test_arm_ignores_message_without_token_count(self):
        sched = self._make_hold_scheduler(pool_tokens=256)
        sched._arm_prefill_oom_hold(RuntimeError("out of memory: no count here"))
        self.assertIsNone(sched._prefill_oom_hold_tokens)

    def test_hold_blocks_until_pool_fits_then_clears(self):
        sched = self._make_hold_scheduler(pool_tokens=256)
        sched._arm_prefill_oom_hold(RuntimeError(self.OOM_MSG))
        # Pool still short (256 < 4096+64): blocked, hold stays armed.
        self.assertTrue(sched._prefill_oom_hold_blocks())
        self.assertEqual(sched._prefill_oom_hold_tokens, 4096 + 64)
        # Decode freed memory: the hold clears and prefill resumes.
        sched.tree_cache.available_and_evictable_size.return_value = 64000
        self.assertFalse(sched._prefill_oom_hold_blocks())
        self.assertIsNone(sched._prefill_oom_hold_tokens)
        self.assertFalse(sched._prefill_oom_hold_blocks())

    def test_hold_escapes_after_pass_limit(self):
        sched = self._make_hold_scheduler(pool_tokens=256)
        sched._arm_prefill_oom_hold(RuntimeError(self.OOM_MSG))
        sched._prefill_oom_hold_passes = 3
        self.assertTrue(sched._prefill_oom_hold_blocks())
        self.assertTrue(sched._prefill_oom_hold_blocks())
        # Pass budget exhausted: fall back to the un-held retry path rather
        # than wedging prefill on a pool that can never satisfy the hold.
        self.assertFalse(sched._prefill_oom_hold_blocks())
        self.assertIsNone(sched._prefill_oom_hold_tokens)


if __name__ == "__main__":
    unittest.main(verbosity=2)
