"""Layer-local intervention equivalence diagnostic (standalone, GPU).

Directive 2026-09-07: the full-model oracle probe (prefill logits after 40
MoE layers) FAILs at max_abs 11.45 vs the 0.0625 spy-calibrated tolerance.
That tolerance was calibrated on same-code-both-sides comparisons; across the
fused grouped_mm kernel vs the eager per-expert replay, BF16 evaluation-order
noise compounds through 40 layers of nonlinearity, so the full-model number
cannot distinguish "semantics broken" from "summation order differs". This
diagnostic tests the SAME semantic claim one layer at a time, where the
cross-kernel difference is bounded a priori (see TOLERANCE JUSTIFICATION
below), using fixed hidden states / indices / weights captured from the exact
pinned checkpoint.

Per target layer L with target experts E_L (from the frozen top-four):

  1. delta_fast   = native_out(h,idx,W) - fast_masked_out(h,idx,W)
     delta_slow   = eager_full(h,idx,W) - eager_masked(h,idx,W)
     -> report max-abs / relative-L2 / cosine of (delta_fast, delta_slow).
  2. native_out vs eager_full   -> same three metrics (cross-kernel, full).
  3. fast_masked_out vs eager_masked -> same three metrics (cross-kernel, masked).
  4. GATE: explicit_targeted = eager replay restricted to E_L routes only
     must equal delta_fast within tolerance (proves the fused kernel removed
     EXACTLY the targeted contributions — no renormalization, no reroute,
     no collateral zeroing).
  5. GATE (exact): every non-targeted element of delta_fast must be bitwise
     0.0 (same shapes, same code path on both sides of the subtraction).
  6. GATE (exact, on the REAL tensor): the router-weight tensor the patched
     forward actually passes to the fused kernel must equal W everywhere
     except the E_L routes, which must be exactly 0.0.

TOLERANCE JUSTIFICATION (committed BEFORE the first GPU run of this script,
2026-09-07 — not fitted to results):

  Both sides of comparison 4 compute the identical weighted sum over the
  identical routed tokens with identical BF16 inputs; only the summation
  ASSOCIATION differs (fused grouped_mm tiles vs sequential per-expert
  index_add_), and both accumulate in fp32 with a single final BF16 rounding.
  Reordering error for fp32 accumulation over K~8 experts x seq terms is
  ~1e-6 relative (precedent: the fp32 spy unit tests agree BITWISE across the
  same two summation orders). The worst differing term is the final BF16
  rounding of two near-boundary values: <= ~5 ulps, i.e. max-abs <= 0.02 x
  peak reference magnitude (5 x 2^-8 ~= 0.0195).
  LAYER_REL_L2_TOL = 5e-3 is ~1000x above expected association noise, while
  the smallest systematic semantic bug still moves the aggregate >= 10x above
  it whenever >= 1 routed token exists: one missed zeroed route at typical
  router weight 0.05-0.4 changes that token's delta by >= 5%; renormalization
  by 1/(1-w) shifts EVERY surviving token of affected rows by >= 5%.
  LAYER_COS_MIN = 0.9999 follows: 5e-3 of orthogonal noise gives
  cos ~= 1 - (5e-3)^2/2 = 0.9999875, so 0.9999 catches any sign flip,
  substitution, or dropped-token error that preserves norms.
  Route-hit counts are recorded per target expert; a target with zero routed
  tokens across all calibration captures makes that layer INCONCLUSIVE
  (fail-closed), because comparison 4 would be vacuous for it.

The old full-model result and its 0.0625 threshold are PRESERVED untouched in
scripts/benchmark_intervention_paths.py::ORACLE_BF16_TOLERANCE. This
diagnostic is separately named (stage-layer-delta / layer-delta-probe.json)
and never overwrites oracle-probe records.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
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

#: A priori per-layer tolerances — see TOLERANCE JUSTIFICATION above.
LAYER_REL_L2_TOL = 5e-3
LAYER_COS_MIN = 0.9999
LAYER_MAXABS_FRAC = 0.02  # x peak abs magnitude of the reference tensor

TOLERANCE_JUSTIFICATION = (
    "Same math, same bit-identical BF16 inputs on both sides; only fp32 "
    "summation association differs (fused grouped_mm tiles vs sequential "
    "per-expert index_add_). Expected association noise ~1e-6 relative "
    "(fp32 spy unit tests agree bitwise across the same two orders); "
    "worst-case final BF16 rounding ~5 ulps => max-abs <= 0.02x peak "
    "(5 x 2^-8 ~= 0.0195). 5e-3 rel-L2 is ~1000x above expected noise and "
    ">= 10x below the smallest systematic semantic bug (one missed zeroed "
    "route at weight 0.05-0.4, or 1/(1-w) renormalization, shifts affected "
    "deltas by >= 5%). Cosine 0.9999 follows from 5e-3 orthogonal noise "
    "(cos ~= 0.9999875) and catches sign/substitution errors."
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


def _eager_replay(
    experts_mod: Any,
    hidden: Any,
    idx: Any,
    weights: Any,
    num_experts: int,
    *,
    skip: set[int] | None = None,
    only: set[int] | None = None,
) -> Any:
    """Eager per-expert replay mirroring instrument_qwen35's side path.

    Bit-for-bit the same op sequence as the masked_total accumulation in
    src/reverse_reap/instrumentation.py (one_hot mask, per-hit loop,
    gate_up linear, silu gate, down linear, weight, index_add_) — with
    ``skip`` zeroing experts (slow_masked) or ``only`` keeping experts
    (explicit targeted contribution). ``skip=None, only=None`` is the full
    unmasked replay (slow_unmasked).
    """
    import torch
    import torch.nn.functional as F

    total = torch.zeros_like(hidden)
    mask = F.one_hot(idx, num_classes=num_experts).permute(2, 1, 0)
    for expert_tensor in torch.nonzero(mask.sum(dim=(1, 2))).flatten():
        expert = int(expert_tensor.item())
        if skip is not None and expert in skip:
            continue
        if only is not None and expert not in only:
            continue
        ranks, tokens = torch.where(mask[expert_tensor])
        current = hidden[tokens]
        gate_up = F.linear(current, experts_mod.gate_up_proj[expert])
        gate, up = gate_up.chunk(2, dim=-1)
        current = experts_mod.act_fn(gate) * up
        current = F.linear(current, experts_mod.down_proj[expert])
        weighted = current * weights[tokens, ranks, None]
        total.index_add_(0, tokens, weighted.to(total.dtype))
    return total


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
    import torch

    seen: dict[str, Any] = {}
    patched = experts_mod.forward
    kw = dict(patched.__func__.__kwdefaults__ or {})
    inner = kw["_original"]

    def recorder(this: Any, hs: Any, ti: Any, tw: Any) -> Any:
        seen["weights"] = tw.detach().clone()
        return inner(hs, ti, tw)

    patched.__func__.__kwdefaults__ = {**kw, "_original": recorder}
    try:
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
                        help="JSON report path (e.g. <out>/layer-delta-probe.json)")
    parser.add_argument("--calibration-sample-ids", nargs="+", default=[
        "4371146b066c3f9643baafd4",
        "84ee9615bd1e09a6f3b83770",
        "7db4d0460f2cb926e2cbf1a4",
        "2542b6cd29768cd223836536",
    ])
    parser.add_argument("--expect-gpu-count", type=int, default=4)
    parser.add_argument("--expect-gpu-name", type=str, default="3090")
    args = parser.parse_args()

    report: dict[str, Any] = {
        "diagnostic": "layer-local-intervention-equivalence",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pinned_revision": PINNED_REVISION,
        "calibration_sample_ids": list(args.calibration_sample_ids),
        "tolerances": {
            "rel_l2": LAYER_REL_L2_TOL,
            "cosine_min": LAYER_COS_MIN,
            "max_abs_frac_of_peak": LAYER_MAXABS_FRAC,
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

        # ---------- per-layer comparisons ----------
        layers_report: dict[str, Any] = {}
        all_pass = True
        for layer, expert in targets:
            experts_mod = architecture.layers[layer].mlp.experts
            original = experts_mod.forward
            target_set = {expert}
            layer_caps = captures[layer]
            total_hits = sum(int((idx == expert).sum().item()) for _, idx, _ in layer_caps)

            per_capture = []
            gate_explicit = True
            gate_nontargeted = True
            gate_weights = True
            worst = {
                "delta_agreement": None,
                "full_agreement": None,
                "masked_agreement": None,
                "explicit_vs_delta": None,
            }
            for ci, (h, idx, w) in enumerate(layer_caps):
                with torch.inference_mode():
                    native_out = original(h, idx, w)
                    with intervene_qwen35(architecture, masked=masked_set):
                        fast_out = experts_mod.forward(h, idx, w)
                        real_modified = _capture_real_modified(experts_mod, h, idx, w)
                    slow_full = _eager_replay(
                        experts_mod, h, idx, w, architecture.num_experts)
                    slow_masked = _eager_replay(
                        experts_mod, h, idx, w, architecture.num_experts,
                        skip=target_set)
                    explicit = _eager_replay(
                        experts_mod, h, idx, w, architecture.num_experts,
                        only=target_set)
                delta_fast = native_out - fast_out
                delta_slow = slow_full - slow_masked
                m_delta = _metrics(delta_fast, delta_slow)
                m_full = _metrics(native_out, slow_full)
                m_masked = _metrics(fast_out, slow_masked)
                m_explicit = _metrics(explicit, delta_fast)
                # Non-targeted rows of delta_fast must be bitwise zero: rows
                # where the token routed to no target expert.
                routed_target = (idx == expert).any(dim=1)
                nontargeted_max = (
                    float(delta_fast[~routed_target].abs().amax().item())
                    if (~routed_target).any() else 0.0
                )
                # Surviving weights on the REAL tensor passed to the kernel.
                expected_mod = w.clone()
                expected_mod[idx == expert] = 0
                weights_exact = bool(
                    torch.equal(real_modified, expected_mod)
                    and bool(((real_modified[idx == expert]) == 0).all()))
                per_capture.append({
                    "capture": ci,
                    "sample_id": args.calibration_sample_ids[ci],
                    "route_hits": int((idx == expert).sum().item()),
                    "delta_agreement": m_delta,
                    "full_agreement": m_full,
                    "masked_agreement": m_masked,
                    "explicit_vs_delta": m_explicit,
                    "nontargeted_delta_max_abs": nontargeted_max,
                    "surviving_weights_exact": weights_exact,
                })
                for key, m in (("delta_agreement", m_delta), ("full_agreement", m_full),
                               ("masked_agreement", m_masked),
                               ("explicit_vs_delta", m_explicit)):
                    prev = worst[key]
                    if prev is None or (m["rel_l2"], -m["cosine"], m["max_abs"]) > (
                            prev["rel_l2"], -prev["cosine"], prev["max_abs"]):
                        worst[key] = m
                gate_explicit = gate_explicit and (
                    m_explicit["rel_l2"] <= LAYER_REL_L2_TOL
                    and m_explicit["cosine"] >= LAYER_COS_MIN
                    and m_explicit["max_abs"]
                    <= LAYER_MAXABS_FRAC * max(m_explicit["peak_ref"], 1e-12)
                )
                gate_nontargeted = gate_nontargeted and nontargeted_max == 0.0
                gate_weights = gate_weights and weights_exact
                for t in (native_out, fast_out, slow_full, slow_masked, explicit,
                          delta_fast, delta_slow):
                    del t
            torch.cuda.empty_cache()

            coverage_ok = total_hits >= 1
            layer_pass = bool(
                gate_explicit and gate_nontargeted and gate_weights and coverage_ok)
            all_pass = all_pass and layer_pass
            layers_report[f"({layer},{expert})"] = {
                "layer": layer,
                "expert": expert,
                "route_hits_total": total_hits,
                "coverage": "ok" if coverage_ok else "INCONCLUSIVE (no routed token)",
                "worst_case": worst,
                "gates": {
                    "explicit_targeted_within_tolerance": gate_explicit,
                    "nontargeted_delta_bitwise_zero": gate_nontargeted,
                    "surviving_weights_bitwise_exact": gate_weights,
                },
                "passed": layer_pass,
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
