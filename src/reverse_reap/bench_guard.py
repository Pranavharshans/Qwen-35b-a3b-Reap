"""Pre-attempt VRAM safety predictions for the benchmark batch ladder.

Pure logic (no torch, no GPU): given the device-level memory baseline captured
after model load, the measured peaks of already-completed batch stages, and the
remaining wall-clock budget, decide whether attempting the next batch size is
predicted safe. The benchmark script calls these helpers; this module exists so
the prediction rule is unit-testable without a GPU.

Fixed 2026-09-07 — two defects in the original inline guard
(``scripts/benchmark_intervention_paths.py`` v0, Phase-1 run):

1. Direction blindness. The prior lookup only considered completed stages with
   a SMALLER batch size (``n < n_next``). The ladder runs largest-first
   (B8, B7, B6, ...), so at guard time for B4/B1 the only completed stages
   were LARGER — the guard found "no completed smaller-batch stage" and
   refused, even though scaling a measured larger-batch peak DOWN is the
   safest possible extrapolation. Fix: nearest completed batch in EITHER
   direction wins; ties resolve to the larger ``n`` (scaling down
   overpredicts, which is the conservative side).
2. Degenerate formula in high-static regimes. The prediction was
   ``static + dynamic * ratio + 2048`` compared against the 92% ceiling.
   Device-level readings include allocator-retained cache, so measured
   ``static`` already sat at ~91% of physical; adding the 8% ceiling margin
   AND a flat 2 GiB on top meant the guard could never approve anything —
   not even the batch it had just measured. It never engaged (the first
   batch bypassed it), so nothing that ran was affected. Fix: predict only
   the DYNAMIC component (``dynamic_prev * n_next / n_prev``) and require it
   to fit in currently-free memory minus ONE explicit margin
   (:data:`GUARD_FREE_MARGIN_MIB`). OOM risk comes from active allocation
   attempts against free memory, and retained cache is reusable by the
   allocator, so the free-based comparison is the sound quantity — with a
   single hard margin instead of two stacked ones.

What is NOT changed: the 92% VRAM safety ceiling itself. It still governs
the batch-selection verdict on MEASURED peaks (``peak <= 0.92 * physical``),
unchanged and unweakened. This module only repairs the pre-attempt heuristic
that decides whether a batch is worth trying at all.
"""

from __future__ import annotations

#: Free-memory margin (MiB) that must remain beyond the predicted dynamic need
#: on the tightest GPU before a batch size is attempted.
GUARD_FREE_MARGIN_MIB = 1024


def predict_batch_safe(
    n_next: int,
    *,
    completed: list[tuple[int, str, dict[int, int]]],
    static_per_gpu: dict[int, int],
    total_per_gpu: dict[int, int],
    remaining_seconds: float,
    need_remaining_seconds: float,
) -> tuple[bool, str]:
    """Predict whether attempting batch size ``n_next`` is safe.

    ``completed`` is ``(batch_size, status, peak_vram_mib_per_gpu)`` per
    finished stage. ``static_per_gpu`` is the device-level baseline captured
    after model load; ``total_per_gpu`` is physical memory per GPU.
    """
    if remaining_seconds < need_remaining_seconds:
        return False, (
            f"remaining {remaining_seconds:.0f}s < {need_remaining_seconds:.0f}s needed"
        )
    priors = [(n, peaks) for (n, status, peaks) in completed if status == "ok" and peaks]
    if not priors:
        # First batch of the ladder: there is nothing to extrapolate from, so
        # the prefill-probe OOM catch plus the per-chunk OOM catch in the
        # stage runner substitute for the extrapolating guard.
        return True, (
            "first batch: no completed stage to extrapolate from; "
            "prefill-probe OOM catch + per-chunk OOM catch apply"
        )
    # Nearest completed batch in either direction; ties resolve to the larger
    # n so the scale factor is <= 1 (scaling down overpredicts: conservative).
    n_prev, prev_peaks = min(priors, key=lambda item: (abs(item[0] - n_next), -item[0]))
    ratio = n_next / n_prev
    direction = "same-size" if ratio == 1 else ("scale-down" if ratio < 1 else "scale-up")
    needs: dict[int, float] = {}
    for gpu in total_per_gpu:
        static = static_per_gpu.get(gpu, 0)
        prev = prev_peaks.get(gpu, static)
        dynamic = max(prev - static, 0)
        needs[gpu] = dynamic * ratio
    worst_gpu = max(needs, key=lambda gpu: needs[gpu])
    free = min(
        total - static_per_gpu.get(gpu, 0) for gpu, total in total_per_gpu.items()
    )
    ok = needs[worst_gpu] <= free - GUARD_FREE_MARGIN_MIB
    return ok, (
        f"predicted dynamic {needs[worst_gpu]:.0f} MiB on GPU{worst_gpu} vs "
        f"{free:.0f} MiB free ({direction} x{ratio:.2f} from n={n_prev}, "
        f"static {static_per_gpu.get(worst_gpu)}, prev peak "
        f"{prev_peaks.get(worst_gpu)}, margin {GUARD_FREE_MARGIN_MIB} MiB)"
    )


def blocking_oom(
    n_next: int,
    *,
    completed: list[tuple[int, str, dict[int, int]]],
) -> str | None:
    """Return a refusal reason if a SMALLER batch already OOMed, else None.

    An OOM at batch ``n`` only blocks larger batches: attempting a smaller
    batch afterwards is still legitimate (it may fit), and same-size retries
    are governed by the stage runner, not the guard.
    """
    smaller_ooms = sorted(n for (n, status, _peaks) in completed if status == "oom" and n < n_next)
    if smaller_ooms:
        return f"OOM already occurred at n={smaller_ooms[0]}"
    return None
