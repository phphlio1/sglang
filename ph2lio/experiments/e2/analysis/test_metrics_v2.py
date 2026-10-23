"""Unit tests for analysis/metrics_v2.py -- every fixture hand-computed.

Run: python3 analysis/test_metrics_v2.py
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics_v2 import (
    STRETCH_UNCACHED_FLOOR_TOKENS,
    _ms_to_s,
    cache_hit_rate,
    compute_metrics_v2,
    itl_chunk_gaps,
    jain_queue_wait,
    percentile,
    stats,
    stretch_values,
    tok_per_s,
    tpot_values_ms,
    worst5_wait_share_pct,
    write_metrics_v2,
)


class TestPercentile(unittest.TestCase):
    def test_empty_and_single(self):
        self.assertIsNone(percentile([], 95))
        self.assertEqual(percentile([5.0], 95), 5.0)
        self.assertEqual(percentile([5.0], 50), 5.0)

    def test_hand_linear_interpolation(self):
        # [1,2,3,4]: k = 3*p/100; p50 k=1.5 -> 2.5; p95 k=2.85 -> 3.85; p99 k=2.97 -> 3.97
        vals = [4, 2, 1, 3]
        self.assertEqual(percentile(vals, 50), 2.5)
        self.assertEqual(percentile(vals, 95), 3.85)
        self.assertEqual(percentile(vals, 99), 3.97)

    def test_stats_shape(self):
        s = stats([1.0, 2.0, 3.0], (50, 95))
        self.assertEqual(s["n"], 3)
        self.assertEqual(s["mean"], 2.0)
        self.assertEqual(s["p50"], 2.0)
        self.assertEqual(sorted(s), ["mean", "n", "p50", "p95"])


class TestUnitConversion(unittest.TestCase):
    def test_ms_to_s(self):
        self.assertAlmostEqual(_ms_to_s({"value": 178.828142, "unit": "ms"}), 0.178828142, places=12)
        self.assertEqual(_ms_to_s({"value": 2.5, "unit": "s"}), 2.5)
        # Unrecognized unit drops the record visibly, never mis-scales it.
        self.assertIsNone(_ms_to_s({"value": 1.0, "unit": "banana"}))
        self.assertIsNone(_ms_to_s({"unit": "ms"}))


class TestStretch(unittest.TestCase):
    def test_hand_basic(self):
        # wait 10 over own work (15-10)=5 -> 2.0; uncached 300 >= floor
        values, n_den, n_ttft = stretch_values([10.0], [15.0], [300])
        self.assertEqual(values, [2.0])
        self.assertEqual((n_den, n_ttft), (0, 0))

    def test_floor_excludes_small_uncached(self):
        # uncached 255 < 256 floor -> excluded even with a huge wait
        values, n_den, n_ttft = stretch_values([10.0, 10.0], [15.0, 15.0], [256, 255])
        self.assertEqual(values, [2.0])
        self.assertEqual((n_den, n_ttft), (0, 0))
        self.assertEqual(STRETCH_UNCACHED_FLOOR_TOKENS, 256)

    def test_zero_and_negative_denominator_dropped_and_counted(self):
        # ttft == wait (den 0) and ttft < wait (den < 0): dropped, counted
        values, n_den, n_ttft = stretch_values([10.0, 10.0, 10.0], [10.0, 5.0, 15.0], [300, 300, 300])
        self.assertEqual(values, [2.0])
        self.assertEqual((n_den, n_ttft), (2, 0))

    def test_missing_ttft_dropped_and_counted_separately(self):
        # queued-but-never-prefilled (window closed while in queue):
        # no ttft_s -> dropped into its own counter, never a bogus value
        values, n_den, n_ttft = stretch_values([10.0, 150.0], [15.0, None], [300, 300])
        self.assertEqual(values, [2.0])
        self.assertEqual((n_den, n_ttft), (0, 1))

    def test_all_warm_empty_population(self):
        # matched == input everywhere -> uncached 0 -> empty, nothing dropped
        values, n_den, n_ttft = stretch_values([1.0, 2.0], [3.0, 4.0], [0, 0])
        self.assertEqual(values, [])
        self.assertEqual((n_den, n_ttft), (0, 0))

    def test_hand_mixed(self):
        # (wait, ttft, uncached): (10,15,300)->2.0; (20,20,300)->dropped; (3,4,100)->floored
        values, n_den, n_ttft = stretch_values([10.0, 20.0, 3.0], [15.0, 20.0, 4.0], [300, 300, 100])
        self.assertEqual(values, [2.0])
        self.assertEqual((n_den, n_ttft), (1, 0))


class TestWorst5Share(unittest.TestCase):
    def test_hand_n20(self):
        # waits 1..20: ceil(0.05*20)=1 top wait 20, total 210 -> 20/210
        waits = [float(i) for i in range(1, 21)]
        self.assertAlmostEqual(worst5_wait_share_pct(waits), 100 * 20 / 210)

    def test_small_n_ceil(self):
        # n=3: ceil(0.15)=1 -> 3/6 = 50%
        self.assertAlmostEqual(worst5_wait_share_pct([1.0, 2.0, 3.0]), 50.0)

    def test_n1(self):
        self.assertEqual(worst5_wait_share_pct([7.5]), 100.0)

    def test_even_spread_reads_five(self):
        # n=100 equal waits: top 5 sum 25, total 500 -> exactly 5%
        self.assertAlmostEqual(worst5_wait_share_pct([5.0] * 100), 5.0)

    def test_empty_and_all_zero(self):
        self.assertIsNone(worst5_wait_share_pct([]))
        self.assertIsNone(worst5_wait_share_pct([0.0, 0.0]))


class TestJainOnWait(unittest.TestCase):
    def test_hand(self):
        self.assertEqual(jain_queue_wait([1.0, 1.0, 1.0, 1.0]), 1.0)
        # [0,0,0,10]: 10^2 / (4*100) = 0.25
        self.assertEqual(jain_queue_wait([0.0, 0.0, 0.0, 10.0]), 0.25)
        # [2,8]: 10^2 / (2*68) = 0.735294...
        self.assertAlmostEqual(jain_queue_wait([2.0, 8.0]), 100 / 136)

    def test_all_zero_waits_fair(self):
        self.assertEqual(jain_queue_wait([0.0, 0.0]), 1.0)

    def test_empty(self):
        self.assertIsNone(jain_queue_wait([]))


class TestITL(unittest.TestCase):
    def test_per_chunk_pooling(self):
        rows = [
            {"chunk_gaps_ms": [10.0, 20.0], "ttft_s": 0.5, "latency_s": 2.5, "osl": 11},
            {"chunk_gaps_ms": [30.0], "ttft_s": 0.2, "latency_s": 0.4, "osl": 1},
        ]
        gaps = itl_chunk_gaps(rows)
        self.assertEqual(gaps, [10.0, 20.0, 30.0])
        # sorted [10,20,30]: p50 k=1 -> 20; p95 k=1.9 -> 29; p99 k=1.98 -> 29.8
        self.assertEqual(percentile(gaps, 50), 20.0)
        self.assertEqual(percentile(gaps, 95), 29.0)
        self.assertEqual(percentile(gaps, 99), 29.8)

    def test_fallback_when_no_arrays(self):
        rows = [{"chunk_gaps_ms": [], "ttft_s": 0.5, "latency_s": 2.5, "osl": 11}]
        self.assertIsNone(itl_chunk_gaps(rows))
        # TPOT fallback: (2.5-0.5)*1000/(11-1) = 200 ms; osl=1 excluded
        self.assertEqual(tpot_values_ms(rows + [{"chunk_gaps_ms": [], "ttft_s": 1.0, "latency_s": 2.0, "osl": 1}]), [200.0])


class TestCacheHit(unittest.TestCase):
    def test_hand(self):
        rows = [
            {"input_len": 1000, "matched_first": 500},
            {"input_len": 1000, "matched_first": 0},
        ]
        self.assertEqual(cache_hit_rate(rows), 0.25)
        self.assertIsNone(cache_hit_rate([]))
        self.assertIsNone(cache_hit_rate([{"input_len": 0, "matched_first": 0}]))


class TestTokPerS(unittest.TestCase):
    def test_hand_window(self):
        # window = max(end)-min(start) = 12-10 = 2 s; tokens 11+1 = 12 -> 6.0
        rows = [
            {"start_ns": 10_000_000_000, "end_ns": 12_000_000_000, "osl": 11},
            {"start_ns": 11_000_000_000, "end_ns": 11_500_000_000, "osl": 1},
        ]
        tps, window = tok_per_s(rows)
        self.assertEqual(tps, 6.0)
        self.assertEqual(window["window_s"], 2.0)
        self.assertEqual(window["output_tokens"], 12)

    def test_empty(self):
        self.assertIsNone(tok_per_s([])[0])


def _profile_record(phase, ttft_ms, lat_ms, isl, osl, gaps=None, error=False, start_ns=None, end_ns=None):
    metrics = {
        "time_to_first_token": {"value": ttft_ms, "unit": "ms"},
        "request_latency": {"value": lat_ms, "unit": "ms"},
        "input_sequence_length": {"value": isl, "unit": "tokens"},
        "output_sequence_length": {"value": osl, "unit": "tokens"},
    }
    if gaps is not None:
        metrics["inter_chunk_latency"] = {"value": gaps, "unit": "ms"}
    return {
        "metadata": {
            "benchmark_phase": phase,
            "request_start_ns": start_ns,
            "request_end_ns": end_ns,
        },
        "metrics": metrics,
        **({"error": "boom"} if error else {}),
    }


def _eval_record(rid, arrival, wait=None, ttft=None, input_len=0, matched=0):
    record = {
        "rid": rid,
        "arrival_wall_s": arrival,
        "input_len": input_len,
        "matched_prefix_tokens_first_admission": matched,
    }
    if wait is not None:
        record["queue_wait_s"] = wait
    if ttft is not None:
        record["ttft_s"] = ttft
    return record


def _build_run_dir(root: Path, with_server_export: bool) -> Path:
    run_dir = root / ("cell_export" if with_server_export else "cell_noexport")
    run_dir.mkdir()
    (run_dir / "profile_export.jsonl").write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                _profile_record("warmup", 100, 200, 10, 5, start_ns=1_000_000_000, end_ns=2_000_000_000),
                _profile_record("profiling", 0, 0, 0, 0, error=True),
                _profile_record("profiling", 500, 2500, 1000, 11, gaps=[100, 300], start_ns=10_000_000_000, end_ns=12_000_000_000),
                _profile_record("profiling", 200, 400, 10, 1, gaps=[], start_ns=11_000_000_000, end_ns=11_500_000_000),
            ]
        )
        + "\n"
    )
    (run_dir / "schedule_eval.jsonl").write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                _eval_record("preoffset", 1.0, wait=99.0, ttft=99.0, input_len=99, matched=0),
                _eval_record("HEALTH_CHECK_1", 2.0, wait=0.0),
                _eval_record("warmup_arrival", 5.0, wait=16.0, ttft=49.0, input_len=1000, matched=0),
                _eval_record("cold_kept", 11.0, wait=2.0, ttft=3.0, input_len=1000, matched=0),
                _eval_record("warm_kept", 12.0, wait=1.0, ttft=2.0, input_len=1000, matched=900),
                {"rid": "no_wait_field", "arrival_wall_s": 13.0},
            ]
        )
        + "\n"
    )
    (run_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "policy": "lpm",
                "concurrency": 4,
                "seed": 42,
                "sglang_commit": "f" * 40,
                "timestamp": "2026-10-08T00:00:00",
                "schedule_eval": {"offset_lines": 1},
            }
        )
    )
    if with_server_export:
        (run_dir / "server_metrics_export.jsonl").write_text(
            json.dumps({"benchmark_phase": "profiling", "timestamp_ns": 10_500_000_000}) + "\n"
        )
    # Old v1 metrics, read-only: the cross-check block must quote it verbatim.
    (run_dir / "e2_metrics.json").write_text(
        json.dumps(
            {
                "queue": {
                    "queue_time_p50_s": 0.11,
                    "queue_time_p95_s": 0.55,
                    "queue_time_p99_s": 1.99,
                    "queue_time_n": 2,
                    "starvation_threshold_s": 0.2,
                    "starvation_rate": 0.5,
                }
            }
        )
    )
    return run_dir


class TestComputeMetricsV2(unittest.TestCase):
    """End-to-end on a hand-built run dir; every expected value below is
    computed by hand in the assertions' comments."""

    def _compute(self, with_server_export):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _build_run_dir(Path(tmp), with_server_export)
            frozen_before = (run_dir / "e2_metrics.json").read_bytes()
            metrics = compute_metrics_v2(run_dir)
            self.assertEqual((run_dir / "e2_metrics.json").read_bytes(), frozen_before)
            return metrics

    def test_populations(self):
        for with_export in (False, True):
            metrics = self._compute(with_export)
            # offset 1 sliced, 1 health check, 1 warmup arrival, 1 no-wait
            counts = metrics["populations"]["p_wait"]
            self.assertEqual(metrics["n"]["p_wait"], 2, with_export)
            self.assertEqual(counts["n_lines"], 6)
            self.assertEqual(counts["n_offset_sliced"], 1)
            self.assertEqual(counts["n_health_check"], 1)
            self.assertEqual(counts["n_warmup_cut"], 1)
            self.assertEqual(counts["n_no_queue_wait"], 1)
            # 4 records: 1 warmup phase, 1 error, 2 kept
            self.assertEqual(metrics["n"]["p_done"], 2)
            self.assertEqual(metrics["populations"]["p_done"]["n_phase_excluded"], 1)
            self.assertEqual(metrics["populations"]["p_done"]["n_error"], 1)
            self.assertEqual(metrics["populations"]["join"], "not_joinable")

    def test_warmup_cut_source(self):
        self.assertEqual(
            self._compute(False)["populations"]["p_wait"]["warmup_cut_source"],
            "profile_first_profiling_request_start",
        )
        self.assertEqual(
            self._compute(True)["populations"]["p_wait"]["warmup_cut_source"],
            "server_first_profiling_scrape",
        )

    def test_metric_values(self):
        metrics = self._compute(False)
        # waits [2, 1]: mean 1.5; sorted [1,2]: p50 1.5, p90 1.9, p95 1.95, p99 1.99
        q = metrics["queue_wait_s"]
        self.assertEqual(q["n"], 2)
        self.assertEqual(q["mean"], 1.5)
        self.assertEqual(q["p50"], 1.5)
        self.assertEqual(q["p90"], 1.9)
        self.assertEqual(q["p95"], 1.95)
        self.assertEqual(q["p99"], 1.99)
        # ttft_ms [500, 200]: mean 350; p50 350; p95 200+0.95*300=485; p99 497
        self.assertEqual(metrics["ttft_ms"]["mean"], 350.0)
        self.assertEqual(metrics["ttft_ms"]["p50"], 350.0)
        self.assertEqual(metrics["ttft_ms"]["p95"], 485.0)
        self.assertEqual(metrics["ttft_ms"]["p99"], 497.0)
        # ITL per-chunk pooled [100, 300]: mean 200, p50 200, p90 280, p95 290, p99 298
        self.assertEqual(metrics["itl_ms"]["source"], "per_chunk")
        self.assertEqual(metrics["itl_ms"]["n_chunks"], 2)
        self.assertEqual(metrics["itl_ms"]["mean"], 200.0)
        self.assertEqual(metrics["itl_ms"]["p50"], 200.0)
        self.assertEqual(metrics["itl_ms"]["p90"], 280.0)
        self.assertEqual(metrics["itl_ms"]["p95"], 290.0)
        self.assertEqual(metrics["itl_ms"]["p99"], 298.0)
        # tokens 12 over window (12-10)=2 s -> 6.0
        self.assertEqual(metrics["tok_per_s"]["value"], 6.0)
        self.assertEqual(metrics["tok_per_s"]["window"]["window_s"], 2.0)
        # cache hit (0+900)/2000 = 0.45
        self.assertEqual(metrics["cache_hit"]["rate"], 0.45)
        # Jain [2,1]: 9/10
        self.assertEqual(metrics["jain_queue_wait"]["value"], 0.9)
        # stretch: cold uncached 1000 >= 256, den 3-2=1 -> 2.0; warm floored
        self.assertEqual(metrics["stretch"]["n"], 1)
        self.assertEqual(metrics["stretch"]["p95"], 2.0)
        self.assertEqual(metrics["stretch"]["n_dropped_denominator"], 0)
        self.assertEqual(metrics["stretch"]["n_dropped_no_ttft"], 0)
        # worst-5%: ceil(0.1)=1 top wait 2 of total 3 -> 66.67%
        self.assertAlmostEqual(metrics["worst5_wait_share_pct"]["value"], 200 / 3)

    def test_cross_check_block(self):
        metrics = self._compute(False)
        frozen = metrics["cross_check"]["queue_histogram_frozen_v1"]
        self.assertEqual(frozen["p95_s"], 0.55)
        self.assertEqual(frozen["p99_s"], 1.99)
        starv = metrics["cross_check"]["starvation_rate_at"]
        # waits [2,1]: both > 0.2 -> 1.0; none > 10 -> 0.0
        self.assertEqual(starv["0.2_s"], 1.0)
        self.assertEqual(starv["10_s"], 0.0)
        self.assertEqual(starv["recorded_threshold_s"], 0.2)
        # cold [2], warm [1]: gap p99 = 1.0
        cw = metrics["cross_check"]["cold_warm_queue_wait"]
        self.assertEqual(cw["cold"]["n"], 1)
        self.assertEqual(cw["warm"]["n"], 1)
        self.assertEqual(cw["gap_p99_s"], 1.0)

    def test_write_does_not_touch_v1(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _build_run_dir(Path(tmp), False)
            frozen_before = (run_dir / "e2_metrics.json").read_bytes()
            write_metrics_v2(run_dir)
            self.assertEqual((run_dir / "e2_metrics.json").read_bytes(), frozen_before)
            written = json.loads((run_dir / "e2_metrics_v2.json").read_text())
            self.assertEqual(written["schema"], "e2_metrics_v2/1")
            self.assertEqual(written["n"]["p_wait"], 2)


if __name__ == "__main__":
    unittest.main()
