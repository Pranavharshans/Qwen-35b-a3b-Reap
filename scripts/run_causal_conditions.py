"""GPU batch runner: one model load, many causal conditions, resumable.

Generates responses for every condition in the given phase of the causal-pilot
conditions spec. Scoring is intentionally NOT performed here so a GPU host
never needs Docker; see scripts/run_causal_scoring.py.

Resumability: a condition whose destination file already exists is skipped
(fail-closed capture semantics — destinations are never overwritten), so a
retry after a timeout or crash re-enters cheaply without redoing finished
conditions.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from reverse_reap.causal import CausalError, generate_condition
from reverse_reap.config import load_config
from reverse_reap.qwen35 import inspect_qwen35_moe
from reverse_reap.runtime import load_donor, validate_donor_contract


def load_conditions(spec_path: Path, phase: str) -> list[dict]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("schema_version") != 1:
        raise SystemExit(f"unsupported conditions spec schema: {spec.get('schema_version')}")
    conditions = [c for c in spec["conditions"] if c["phase"] == phase]
    if not conditions:
        raise SystemExit(f"no conditions found for phase {phase!r}")
    return conditions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument(
        "--phase",
        required=True,
        choices=["validation-baselines", "validation-interventions", "random"],
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    conditions = load_conditions(args.conditions, args.phase)
    expected_experts = {
        c["condition_id"]: (None if c["expert_manifest"] is None else Path(c["expert_manifest"]))
        for c in conditions
    }
    for condition_id, manifest in expected_experts.items():
        if manifest is not None and not manifest.exists():
            raise SystemExit(f"frozen expert manifest missing for {condition_id}: {manifest}")

    print(f"loading donor once for {len(conditions)} conditions...", flush=True)
    started = time.monotonic()
    model, tokenizer = load_donor(args.model_path, config)
    architecture = inspect_qwen35_moe(model)
    validate_donor_contract(model, architecture)
    print(f"donor loaded in {time.monotonic() - started:.1f}s", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    completed, skipped = [], []
    for condition in conditions:
        condition_id = condition["condition_id"]
        destination = args.output_dir / f"{condition_id}.jsonl"
        summary_path = args.summary_dir / f"{condition_id}.summary.json"
        if destination.exists():
            skipped.append(condition_id)
            print(f"[skip] {condition_id}: {destination} already complete", flush=True)
            summary = {
                "condition_id": condition_id,
                "samples": sum(
                    1 for line in destination.read_text(encoding="utf-8").splitlines() if line
                ),
                "resumed": True,
            }
        else:
            summary = generate_condition(
                model,
                tokenizer,
                architecture,
                args.dataset_manifest,
                destination,
                config,
                split=condition["split"],
                condition_id=condition_id,
                expert_manifest=expected_experts[condition_id],
                limit=args.limit,
                instrument_noop=bool(condition.get("instrument_noop")),
            )
            summary["resumed"] = False
            completed.append(condition_id)
            print(
                f"[done] {condition_id}: samples={summary['samples']} "
                f"truncation={summary['truncation_rate']:.3f} "
                f"mean_latency={summary['mean_latency_seconds']:.1f}s",
                flush=True,
            )
        summary["expert_manifest"] = condition["expert_manifest"]
        summary["split"] = condition["split"]
        destination_tmp = summary_path.with_name(f".{summary_path.name}.")
        destination_tmp.write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")
        destination_tmp.replace(summary_path)

    print(
        f"phase {args.phase} complete: {len(completed)} generated, {len(skipped)} resumed",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CausalError as error:
        print(f"CausalError: {error}", file=sys.stderr)
        sys.exit(2)
