"""Same-kernel intervention equivalence diagnostic (standalone, GPU).

Directive 2026-09-07: the layer-local diagnostic
(scripts/validate_layer_local_deltas.py) FAILED without revising its
tolerance — its cross-kernel comparisons (fused grouped_mm vs eager
per-expert replay) sit on a measured full_agreement noise floor of
rel-L2 ~= 4.1e-3, ~4000x above the 1e-6 assumed in its pre-run
justification. That file and its 5e-3 threshold are PRESERVED UNCHANGED
as the failed record; this script is a NEW pre-commit that never invokes
the eager replay, so the cross-kernel floor cannot contaminate it.

Per target layer L with target experts E_L (from the frozen top-four),
using ONLY the native fused experts.forward (identical kernel both
sides) on fixed (hidden, idx, W) captured from the exact pinned
checkpoint:

  full          = orig(h, idx, W)
  masked_manual = orig(h, idx, W_zeroed)   # targeted routes zeroed by hand
  only_manual   = orig(h, idx, W_only)     # only targeted routes kept
  intervened    = orig(h, idx, W) under intervene_qwen35(masked=...)
  empty_out     = orig(h, idx, W) under intervene_qwen35(masked=frozenset())

  1. GATE (bitwise): intervened == masked_manual. Same kernel, same
     effective inputs — must be bit-identical. Catches any wrapper
     behaviour beyond clone+zero (renormalization, rerouting, dtype
     changes, extra passes).
  2. GATE (bitwise): empty_out == full. Empty mask must be a no-op.
  3. GATE (bitwise, on the REAL tensor): the router-weight tensor the
     patched forward actually passes inward must equal W_zeroed, with
     targeted routes exactly 0.0.
  4. GATE (tolerance): full == masked_manual + only_manual (weight-space
     additivity through the same kernel — zeroing is subtraction, no
     renormalization, no rerouting). The sum is formed in fp32 so BF16
     summation rounding is not attributed to the kernel.
  5. GATE (bitwise): rows of (full - masked_manual) whose token routed
     to no target expert must be bitwise 0.0 (same kernel, same
     contributing inputs on both sides for those rows).
  6. FAIL-CLOSED coverage: a target with zero routed tokens across all
     calibration captures makes that layer INCONCLUSIVE, because gates
     1/4/5 would be vacuous for it.

TOLERANCE JUSTIFICATION (committed BEFORE the first GPU run of this
script, 2026-09-07 — not fitted to results):

  Both sides of comparison 4 run the IDENTICAL fused grouped_mm kernel
  on bit-identical BF16 inputs except for the weight pattern (full vs
  split-then-added). Kernel blocking, tiling, and internal precision are
  the same on both sides, so unlike the cross-kernel diagnostic there is
  no method difference to compound — the only divergence is fp32
  accumulation association from the changed zero pattern plus one final
  BF16 rounding, expected ~1e-6 relative (precedent: the fp32 spy unit
  tests agree BITWISE across summation orders). The worst differing term
  is the final BF16 rounding of two near-boundary values: <= ~5 ulps,
  i.e. max-abs <= 0.02 x peak reference magnitude (5 x 2^-8 ~= 0.0195).
  SAMEKERNEL_REL_L2_TOL = 5e-3 is ~1000x above expected same-kernel
  rounding noise, while the smallest systematic semantic bug still moves
  the aggregate >= 10x above it whenever >= 1 routed token exists: one
  missed zeroed route at typical router weight 0.05-0.4 changes that
  token's residual by >= 5%; renormalization by 1/(1-w) shifts EVERY
  surviving token of affected rows by >= 5%.
  SAMEKERNEL_COS_MIN = 0.9999 follows: 5e-3 of orthogonal noise gives
  cos ~= 1 - (5e-3)^2/2 = 0.9999875, so 0.9999 catches any sign flip,
  substitution, or dropped-token error that preserves norms.
  Route-hit counts are recorded per target expert (see gate 6).

Report shape mirrors the layer-local diagnostic (``status`` pass/fail/
error plus ``passed`` bool) so scripts/benchmark_intervention_paths.py
--layer-diagnostic consumes it unchanged; the old full-model oracle
result and its 0.0625 threshold stay untouched, and this report is
separately named (stage-same-kernel / same-kernel-probe.json).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reverse_reap.causal import load_expert_set  # noqa: E402
from reverse_reap.config import load_config  # noqa: E402
from reverse_reap.datasets import load_manifest  # noqa: E402
from reverse_reap.instrumentation import FROZEN_SELECTED_TOP4, intervene_qwen35  # noqa: E402
from reverse_reap.qwen35 import inspect_qwen35_moe  # noqa: E402
from reverse_reap.runtime import load_donor, validate_donor_contract  # noqa: E402

PINNED_REVISION = "59d61f3ce65a6d9863b86d2e96597125219dc754"

#: A priori same-kernel tolerances — see TOLERANCE JUSTIFICATION above.
SAMEKERNEL_REL_L2_TOL = 5e-3
SAMEKERNEL_COS_MIN = 0.9999
SAMEKERNEL_MAXABS_FRAC = 0.02  # x peak abs magnitude of the reference tensor

TOLERANCE_JUSTIFICATION = (
    "Identical fused grouped_mm kernel both sides on bit-identical BF16 "
    "inputs except the weight pattern (full vs split-then-added); no method "
    "difference, so only fp32 association from the changed zero pattern "
    "plus one final BF16 rounding differs (expected ~1e-6 relative; fp32 "
    "spy unit tests agree bitwise across summation orders). Worst-case "
    "final BF16 rounding ~5 ulps => max-abs <= 0.02x peak (5 x 2^-8 "
    "~= 0.0195). 5e-3 rel-L2 is ~1000x above expected same-kernel noise "
    "and >= 10x below the smallest systematic semantic bug (one missed "
    "zeroed route at weight 0.05-0.4, or 1/(1-w) renormalization, shifts "
    "affected residuals by >= 5%). Cosine 0.9999 follows from 5e-3 "
    "orthogonal noise (cos ~= 0.9999875) and catches sign/substitution "
    "errors."
)


def _metrics(a: Any, b: Any) -> dict[str, float]:
    """max-abs, relative-L2 and cosine between two same-shape tensors."""
    af, bf = a.float(), b.float()
    diff = af - bf
    max_abs = float(diff.abs().amax().item())
    norm_a = float(af.norm().item())
    norm_b = float(bf.norm().item())
    denom = max(norm_a, norm_b)
    if denom > 0:
        rel_l2 = float(diff.norm().item() / denom)
    else:
        rel_l2 = 0.0 if max_abs == 0.0 else float("inf")
    if norm_a > 0 and norm_b > 0:
        cosine = float(((af.flatten() @ bf.flatten()) / (norm_a * norm_b)).item())
    else:
        cosine = 1.0 if max_abs == 0.0 else 0.0
    return {
        "max_abs": max_abs,
        "rel_l2": rel_l2,
        "cosine": cosine,
        "peak_ref": float(bf.abs().amax().item()),
    }


def _zero_targeted_weights(weights: Any, idx: Any, target_set: set[int]) -> Any:
    """Clone of ``weights`` with every route to a targeted expert zeroed."""
    modified = weights.clone()
    for expert in target_set:
        modified[idx == expert] = 0
    return modified


def _only_targeted_weights(weights: Any, idx: Any, target_set: set[int]) -> Any:
    """Zeros shaped like ``weights`` keeping only targeted routes."""
    import torch

    only = torch.zeros_like(weights)
    for expert in target_set:
        only[idx == expert] = weights[idx == expert]
    return only


def _within_tolerance(m: dict[str, float]) -> bool:
    """Gate-4 predicate: additivity residual within the pre-committed bound."""
    return (
        m["rel_l2"] <= SAMEKERNEL_REL_L2_TOL
        and m["cosine"] >= SAMEKERNEL_COS_MIN
        and m["max_abs"] <= SAMEKERNEL_MAXABS_FRAC * max(m["peak_ref"], 1e-12)
    )


def assess_capture(
    full: Any,
    masked_manual: Any,
    only_manual: Any,
    intervened: Any,
    empty_out: Any,
    real_modified: Any,
    weights: Any,
    idx: Any,
    target_set: set[int],
) -> dict[str, Any]:
    """Evaluate all same-kernel gates for one captured (h, idx, W) triple.

    Pure tensor logic shared by the GPU run and the synthetic tests: no
    model, no kernel choice, no I/O. ``real_modified`` is the exact weight
    tensor the patched forward passed inward. Returns per-gate booleans,
    the additivity metrics, and route-hit counts.
    """
    import torch

    expected_zeroed = _zero_targeted_weights(weights, idx, target_set)
    wrapper_fidelity = bool(torch.equal(intervened, masked_manual))
    empty_is_noop = bool(torch.equal(empty_out, full))
    if target_set:
        weights_exact = bool(
            torch.equal(real_modified, expected_zeroed)
            and all(
                bool(((real_modified[idx == expert]) == 0).all())
                for expert in target_set
            )
        )
    else:
        weights_exact = bool(torch.equal(real_modified, weights))
    reconstructed = masked_manual.float() + only_manual.float()
    m_additivity = _metrics(reconstructed, full.float())
    additivity_ok = _within_tolerance(m_additivity)
    routed_target = torch.zeros(idx.shape[0], dtype=torch.bool, device=idx.device)
    for expert in target_set:
        routed_target |= (idx == expert).any(dim=1)
    residual = full - masked_manual
    nontargeted_max = (
        float(residual[~routed_target].abs().amax().item())
        if (~routed_target).any()
        else 0.0
    )
    nontargeted_zero = nontargeted_max == 0.0
    route_hits = {expert: int((idx == expert).sum().item()) for expert in target_set}
    return {
        "wrapper_fidelity_bitwise": wrapper_fidelity,
        "empty_mask_is_noop_bitwise": empty_is_noop,
        "surviving_weights_bitwise_exact": weights_exact,
        "additivity_within_tolerance": additivity_ok,
        "additivity_metrics": m_additivity,
        "nontargeted_delta_bitwise_zero": nontargeted_zero,
        "nontargeted_delta_max_abs": nontargeted_max,
        "route_hits": route_hits,
    }


def _encode_one(tokenizer: Any, sample: Any, config: Any, model: Any) -> dict:
    """Single-sample prefill encoding (same template path as the benchmark).

    Mirrors scripts/benchmark_intervention_paths.py::_encode_batch for one
    sample (no padding needed for a single row).
    """
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": sample.prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=config.runtime.enable_thinking,
    )
    encoded = tokenizer(text, return_tensors="pt")
    return {k: v.to(model.get_input_embeddings().weight.device) for k, v in encoded.items()}


def _capture_real_modified(
    experts_mod: Any, hidden: Any, idx: Any, weights: Any
) -> Any:
    """Return the EXACT weight tensor the patched forward passes inward.

    Under ``intervene_qwen35`` the patched experts.forward closes over the
    original as ``_original``; temporarily swap that closure slot for a
    recorder so the clone the intervention built is captured, not recomputed.
    """
    seen: dict[str, Any] = {}

    patched = experts_mod.forward
    kw = dict(patched.__func__.__kwdefaults__ or {})
    inner = kw["_original"]

    def recorder(hs: Any, ti: Any, tw: Any) -> Any:
        seen["weights"] = tw.detach().clone()
        return inner(hs, ti, tw)

    patched.__func__.__kwdefaults__ = {**kw, "_original": recorder}
    try:
        import torch

        with torch.inference_mode():
            patched(hidden, idx, weights)
    finally:
        patched.__func__.__kwdefaults__ = kw
    return seen["weights"]


def main() -> int:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selected-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="JSON report path (e.g. <out>/same-kernel-probe.json)")
    parser.add_argument("--calibration-sample-ids", nargs="+", default=[
        "4371146b066c3f9643baafd4",
        "7db4d0460f2cb926e2cbf1a4",
    ])
    parser.add_argument("--expect-gpu-count", type=int, default=4)
    parser.add_argument("--expect-gpu-name", type=str, default="3090")
    args = parser.parse_args()

    report: dict[str, Any] = {
        "diagnostic": "same-kernel-intervention-equivalence",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pinned_revision": PINNED_REVISION,
        "calibration_sample_ids": list(args.calibration_sample_ids),
        "tolerances": {
            "rel_l2": SAMEKERNEL_REL_L2_TOL,
            "cosine_min": SAMEKERNEL_COS_MIN,
            "max_abs_frac_of_peak": SAMEKERNEL_MAXABS_FRAC,
            "justification": TOLERANCE_JUSTIFICATION,
            "committed": "2026-09-07, before the first GPU run (not fitted)",
        },
        "status": "error",
    }

    def write(status: str, passed: bool | None = None) -> int:
        report["status"] = status
        if passed is not None:
            report["passed"] = passed
        report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": status, "passed": report.get("passed"),
                          "layers": report.get("layers", {})}, indent=1, default=str),
              flush=True)
        return {"pass": 0, "fail": 1, "error": 2}[status]

    try:
        # ---------- host preflight (fail closed) ----------
        n_gpus = torch.cuda.device_count()
        names = [torch.cuda.get_device_name(i) for i in range(n_gpus)]
        if n_gpus != args.expect_gpu_count or not all(
            args.expect_gpu_name in name for name in names
        ):
            report["error"] = f"host is not as expected: count={n_gpus} names={names}"
            return write("error")

        config = load_config(args.config)
        model, tokenizer = load_donor(args.model_path, config)
        architecture = inspect_qwen35_moe(model)
        validate_donor_contract(model, architecture)

        # ---------- manifest is authority ----------
        masked_set = load_expert_set(args.selected_manifest)
        if masked_set != FROZEN_SELECTED_TOP4:
            report["error"] = f"selected manifest != frozen top-4: {sorted(masked_set)}"
            return write("error")
        targets = sorted(masked_set)  # [(layer, expert)] ascending by layer

        by_id = {s.sample_id: s for s in load_manifest(args.manifest)
                 if s.split == "validation"}
        missing = [i for i in args.calibration_sample_ids if i not in by_id]
        if missing:
            report["error"] = f"calibration samples missing from manifest: {missing}"
            return write("error")
        cal_samples = [by_id[i] for i in args.calibration_sample_ids]

        # ---------- capture fixed (hidden, idx, W) per target layer ----------
        target_layers = sorted({layer for layer, _ in targets})
        captures: dict[int, list[tuple[Any, Any, Any]]] = {layer: [] for layer in target_layers}
        handles = []
        for layer in target_layers:
            experts_mod = architecture.layers[layer].mlp.experts

            def hook(_mod: Any, hook_args: Any, _out: Any, _layer: int = layer) -> None:
                h, idx, w = hook_args[0], hook_args[1], hook_args[2]
                captures[_layer].append(
                    (h.detach().clone(), idx.detach().clone(), w.detach().clone()))

            handles.append(experts_mod.register_forward_hook(hook))
        try:
            with torch.inference_mode():
                for sample in cal_samples:
                    encoded = _encode_one(tokenizer, sample, config, model)
                    model(**encoded, use_cache=False)
                    del encoded
        finally:
            for handle in handles:
                handle.remove()
        torch.cuda.empty_cache()

        report["captures_per_layer"] = {
            str(layer): len(captures[layer]) for layer in target_layers
        }
        if any(len(captures[layer]) != len(cal_samples) for layer in target_layers):
            report["error"] = "hook captured an unexpected call count"
            return write("error")

        # ---------- per-layer same-kernel comparisons ----------
        layers_report: dict[str, Any] = {}
        all_pass = True
        for layer, expert in targets:
            experts_mod = architecture.layers[layer].mlp.experts
            original = experts_mod.forward
            target_set = {expert}
            layer_caps = captures[layer]
            total_hits = sum(int((idx == expert).sum().item()) for _, idx, _ in layer_caps)

            per_capture = []
            layer_ok = True
            worst_add: dict[str, float] | None = None
            for ci, (h, idx, w) in enumerate(layer_caps):
                w_zeroed = _zero_targeted_weights(w, idx, target_set)
                w_only = _only_targeted_weights(w, idx, target_set)
                with torch.inference_mode():
                    full = original(h, idx, w)
                    masked_manual = original(h, idx, w_zeroed)
                    only_manual = original(h, idx, w_only)
                    with intervene_qwen35(architecture, masked=masked_set):
                        intervened = experts_mod.forward(h, idx, w)
                        real_modified = _capture_real_modified(experts_mod, h, idx, w)
                    with intervene_qwen35(architecture, masked=frozenset()):
                        empty_out = experts_mod.forward(h, idx, w)
                verdict = assess_capture(
                    full, masked_manual, only_manual, intervened,
                    empty_out, real_modified, w, idx, target_set)
                per_capture.append({
                    "capture": ci,
                    "sample_id": args.calibration_sample_ids[ci],
                    "route_hits": verdict["route_hits"].get(expert, 0),
                    "additivity": verdict["additivity_metrics"],
                    "gates": {
                        "wrapper_fidelity_bitwise":
                            verdict["wrapper_fidelity_bitwise"],
                        "empty_mask_is_noop_bitwise":
                            verdict["empty_mask_is_noop_bitwise"],
                        "surviving_weights_bitwise_exact":
                            verdict["surviving_weights_bitwise_exact"],
                        "additivity_within_tolerance":
                            verdict["additivity_within_tolerance"],
                        "nontargeted_delta_bitwise_zero":
                            verdict["nontargeted_delta_bitwise_zero"],
                    },
                    "nontargeted_delta_max_abs": verdict["nontargeted_delta_max_abs"],
                    "surviving_weights_exact":
                        verdict["surviving_weights_bitwise_exact"],
                })
                m = verdict["additivity_metrics"]
                if worst_add is None or (m["rel_l2"], -m["cosine"], m["max_abs"]) > (
                        worst_add["rel_l2"], -worst_add["cosine"], worst_add["max_abs"]):
                    worst_add = m
                layer_ok = layer_ok and (
                    verdict["wrapper_fidelity_bitwise"]
                    and verdict["empty_mask_is_noop_bitwise"]
                    and verdict["surviving_weights_bitwise_exact"]
                    and verdict["additivity_within_tolerance"]
                    and verdict["nontargeted_delta_bitwise_zero"]
                )
                for t in (full, masked_manual, only_manual, intervened,
                          empty_out, w_zeroed, w_only):
                    del t
            torch.cuda.empty_cache()

            coverage_ok = total_hits >= 1
            passed = bool(layer_ok and coverage_ok)
            all_pass = all_pass and passed
            layers_report[f"({layer},{expert})"] = {
                "layer": layer,
                "expert": expert,
                "route_hits_total": total_hits,
                "coverage": "ok" if coverage_ok else "INCONCLUSIVE (no routed token)",
                "worst_case_additivity": worst_add,
                "passed": passed,
                "per_capture": per_capture,
            }

        report["layers"] = layers_report
        return write("pass" if all_pass else "fail", all_pass)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        report["error"] = "CUDA OOM — cleared safely"
        return write("error")
    except Exception as exc:  # fail closed, report survives
        report["error"] = f"{type(exc).__name__}: {exc}"[:300]
        return write("error")


if __name__ == "__main__":
    sys.exit(main())
