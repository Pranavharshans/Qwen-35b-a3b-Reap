"""Regression tests for the direction-aware VRAM guard (torch-free)."""

from __future__ import annotations

from reverse_reap.bench_guard import blocking_oom, predict_batch_safe

#: Realistic Phase-1 regime: device-level static ~22 GiB (weights + retained
#: allocator cache) on a 24124 MiB GPU; B8 measured peak 22438 MiB.
STATIC = {0: 22000, 1: 22010, 2: 21995, 3: 22005}
TOTAL = {0: 24124, 1: 24124, 2: 24124, 3: 24124}
B8_PEAKS = {0: 22438, 1: 22410, 2: 22420, 3: 22400}


def _completed(*entries):
    return list(entries)


def test_first_batch_approved_with_probe_reason():
    ok, reason = predict_batch_safe(
        8, completed=[], static_per_gpu=STATIC, total_per_gpu=TOTAL,
        remaining_seconds=7000.0, need_remaining_seconds=420.0,
    )
    assert ok
    assert "first batch" in reason


def test_wall_clock_floor_refuses():
    ok, reason = predict_batch_safe(
        7, completed=_completed((8, "ok", B8_PEAKS)),
        static_per_gpu=STATIC, total_per_gpu=TOTAL,
        remaining_seconds=100.0, need_remaining_seconds=420.0,
    )
    assert not ok
    assert "remaining" in reason


def test_descending_ladder_scales_down_from_larger_prior():
    """The Phase-1 bug: B4/B1 were refused because only larger priors existed."""
    ok, reason = predict_batch_safe(
        4, completed=_completed((8, "ok", B8_PEAKS)),
        static_per_gpu=STATIC, total_per_gpu=TOTAL,
        remaining_seconds=7000.0, need_remaining_seconds=420.0,
    )
    assert ok
    assert "scale-down" in reason


def test_ascending_ladder_scales_up_from_smaller_prior():
    b4_peaks = {g: STATIC[g] + 200 for g in STATIC}
    ok, reason = predict_batch_safe(
        8, completed=_completed((4, "ok", b4_peaks)),
        static_per_gpu=STATIC, total_per_gpu=TOTAL,
        remaining_seconds=7000.0, need_remaining_seconds=420.0,
    )
    assert ok
    assert "scale-up" in reason


def test_nearest_prior_wins_and_ties_resolve_larger():
    b8 = _completed((8, "ok", B8_PEAKS))
    b4_peaks = {g: STATIC[g] + 200 for g in STATIC}
    ok, reason = predict_batch_safe(
        6, completed=b8 + [(4, "ok", b4_peaks)],
        static_per_gpu=STATIC, total_per_gpu=TOTAL,
        remaining_seconds=7000.0, need_remaining_seconds=420.0,
    )
    assert ok
    # |6-8| == |6-4|: tie resolves to n=8 (scale-down, conservative side).
    assert "from n=8" in reason
    assert "scale-down" in reason


def test_high_static_regime_still_approves_small_dynamic():
    """The degenerate-formula regression: static ~91% of physical must not
    veto batches whose measured dynamic component is small."""
    for n_next in (7, 6, 4, 1):
        ok, _reason = predict_batch_safe(
            n_next, completed=_completed((8, "ok", B8_PEAKS)),
            static_per_gpu=STATIC, total_per_gpu=TOTAL,
            remaining_seconds=7000.0, need_remaining_seconds=1500.0,
        )
        assert ok, f"n={n_next} refused despite small dynamic need"


def test_margin_blocks_prediction_exceeding_free_memory():
    tight_static = {g: 23800 for g in TOTAL}  # only 324 MiB free < 1024 margin
    ok, reason = predict_batch_safe(
        8, completed=_completed((4, "ok", {g: 24000 for g in TOTAL})),
        static_per_gpu=tight_static, total_per_gpu=TOTAL,
        remaining_seconds=7000.0, need_remaining_seconds=420.0,
    )
    assert not ok
    assert "margin" in reason


def test_failed_and_oob_statuses_are_not_priors():
    ok, reason = predict_batch_safe(
        6, completed=_completed((8, "error", B8_PEAKS), (4, "skipped", {})),
        static_per_gpu=STATIC, total_per_gpu=TOTAL,
        remaining_seconds=7000.0, need_remaining_seconds=420.0,
    )
    assert ok
    assert "first batch" in reason


def test_smaller_oom_blocks_larger_but_not_smaller():
    completed = _completed((4, "oom", {}), (8, "ok", B8_PEAKS))
    assert blocking_oom(6, completed=completed) == "OOM already occurred at n=4"
    assert blocking_oom(8, completed=completed) == "OOM already occurred at n=4"
    assert blocking_oom(4, completed=completed) is None
    assert blocking_oom(2, completed=completed) is None


def test_no_oom_means_no_block():
    assert blocking_oom(8, completed=_completed((8, "ok", B8_PEAKS))) is None
    assert blocking_oom(8, completed=[]) is None
