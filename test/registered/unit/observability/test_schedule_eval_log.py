"""Unit tests for the per-request schedule-eval log (--log-schedule-eval-path)."""

import json
import os
import tempfile
import unittest
from array import array
from pathlib import Path

from sglang.srt.managers.schedule_batch import FINISH_LENGTH, Req
from sglang.srt.observability.schedule_eval_log import (
    ScheduleEvalLogger,
    stamp_admission_prefix_matches,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_req(rid):
    return Req(rid, "", array("q", list(range(100))), SamplingParams())


def _finish(req, *, retracted=False):
    """Drive a req through the stamp lifecycle the scheduler produces."""
    req.time_stats.set_scheduler_recv_time(10.0)
    req.time_stats.set_wait_queue_entry_time(11.0)
    if req.first_queue_entry_time is None:
        req.first_queue_entry_time = req.time_stats.wait_queue_entry_time
    req.num_matched_prefix_tokens = 64
    stamp_admission_prefix_matches([req])
    req.time_stats.set_forward_entry_time(15.0)
    if retracted:
        req.reset_for_retract()
        req.time_stats.set_wait_queue_entry_time(20.0)
        req.num_matched_prefix_tokens = 0
        stamp_admission_prefix_matches([req])
        req.time_stats.set_forward_entry_time(25.0)
    req.time_stats.set_prefill_finished_time(30.0)
    req.time_stats.set_completion_time(40.0)
    req.finished_reason = FINISH_LENGTH(length=100)
    return req


class TestStampAdmissionPrefixMatches(CustomTestCase):
    def test_first_admission_is_stamp_once_last_is_refreshed(self):
        req = _make_req("r")
        req.num_matched_prefix_tokens = 64
        stamp_admission_prefix_matches([req])
        req.num_matched_prefix_tokens = 0  # retracted: nothing cached anymore
        stamp_admission_prefix_matches([req])
        self.assertEqual(req.matched_prefix_tokens_first_admission, 64)
        self.assertEqual(req.matched_prefix_tokens_last_admission, 0)

    def test_retraction_reset_keeps_the_bookkeeping(self):
        # reset_for_retract clears the cache-match state but must keep the
        # first-admission stamp (it is the first-attempt hit rate) and the
        # first queue entry (queue_wait spans the retraction cycle).
        req = _make_req("r")
        req.first_queue_entry_time = 11.0
        req.num_matched_prefix_tokens = 64
        stamp_admission_prefix_matches([req])
        req.reset_for_retract()
        self.assertEqual(req.matched_prefix_tokens_first_admission, 64)
        self.assertEqual(req.first_queue_entry_time, 11.0)
        self.assertEqual(req.retraction_count, 1)


class TestScheduleEvalLogger(CustomTestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="schedule_eval_log_")

        def _cleanup():
            for p in Path(self.dir).glob("*"):
                p.unlink(missing_ok=True)
            os.rmdir(self.dir)

        self.addCleanup(_cleanup)
        self.path = os.path.join(self.dir, "eval.jsonl")

    def read_records(self):
        with open(self.path) as handle:
            return [json.loads(line) for line in handle]

    def test_one_line_per_finished_req_with_expected_fields(self):
        logger = ScheduleEvalLogger(
            path=self.path, policy="boost", boost_gamma=1e-4, boost_bin_size=256
        )
        finished = _finish(_make_req("done"))
        pending = _make_req("pending")  # still running: must not be logged
        logger.log_finished_batch([pending, finished])
        logger.close()

        records = self.read_records()
        self.assertEqual([r["rid"] for r in records], ["done"])
        rec = records[0]
        self.assertEqual(
            {
                "rid",
                "policy",
                "gamma",
                "bin_size",
                "arrival_wall_s",
                "queue_wait_s",
                "ttft_s",
                "e2e_s",
                "input_len",
                "output_len",
                "matched_prefix_tokens_first_admission",
                "matched_prefix_tokens_last_admission",
                "num_retractions",
                "boost_arrival_tokens",
            },
            set(rec.keys()),
        )
        self.assertEqual(rec["policy"], "boost")
        self.assertEqual(rec["gamma"], 1e-4)
        self.assertEqual(rec["bin_size"], 256)
        self.assertEqual(rec["input_len"], 100)
        self.assertEqual(rec["matched_prefix_tokens_first_admission"], 64)
        self.assertEqual(rec["num_retractions"], 0)
        # First admission (15.0) minus first entry (11.0), on the scheduler clock.
        self.assertAlmostEqual(rec["queue_wait_s"], 4.0)
        self.assertAlmostEqual(rec["ttft_s"], 20.0)
        self.assertAlmostEqual(rec["e2e_s"], 30.0)
        self.assertIsNotNone(rec["arrival_wall_s"])

    def test_queue_wait_spans_retraction_from_the_first_entry(self):
        # The stamp-once contract (spec §4): FIRST admission minus FIRST entry.
        # wait_queue_entry_time re-stamps on re-entry and forward_entry_time
        # stays at the first admission, so a retracted req reports the same
        # first-attempt wait it would have without the retraction; the
        # retraction itself is visible via num_retractions and the
        # last-admission prefix match.
        logger = ScheduleEvalLogger(
            path=self.path, policy="boost", boost_gamma=1e-4, boost_bin_size=256
        )
        logger.log_finished_batch([_finish(_make_req("retracted"), retracted=True)])
        logger.close()

        rec = self.read_records()[0]
        self.assertEqual(rec["num_retractions"], 1)
        self.assertEqual(rec["matched_prefix_tokens_first_admission"], 64)
        self.assertEqual(rec["matched_prefix_tokens_last_admission"], 0)
        self.assertAlmostEqual(rec["queue_wait_s"], 15.0 - 11.0)

    def test_finished_req_logged_once_across_batches(self):
        # Overlap mode can surface a finished req in two result batches; the
        # rid set keeps the file one-line-per-request.
        logger = ScheduleEvalLogger(
            path=self.path, policy="fcfs", boost_gamma=1e-4, boost_bin_size=256
        )
        req = _finish(_make_req("done"))
        logger.log_finished_batch([req])
        logger.log_finished_batch([req])
        logger.close()
        self.assertEqual(len(self.read_records()), 1)

    def test_unwritable_path_warns_and_disables_without_raising(self):
        with self.assertLogs(
            "sglang.srt.observability.schedule_eval_log", level="WARNING"
        ):
            logger = ScheduleEvalLogger(
                path=os.path.join(self.dir, "missing_dir", "eval.jsonl"),
                policy="boost",
                boost_gamma=1e-4,
                boost_bin_size=256,
            )
        # Disabled, not crashed: logging a finished batch is a no-op.
        logger.log_finished_batch([_finish(_make_req("done"))])
        logger.close()
        self.assertFalse(os.path.exists(os.path.join(self.dir, "missing_dir")))


if __name__ == "__main__":
    unittest.main()
