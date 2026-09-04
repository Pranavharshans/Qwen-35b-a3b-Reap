"""Score generation-only causal condition records on the CPU host with Docker.

Applies reverse_reap.causal.score_response per record (unit_tests run inside
the digest-pinned evaluator container; exact_match/multiple_choice locally)
and writes ``{condition_id}.preswebench.jsonl``. SWE-bench rows stay
scoreable=False here — they are resolved ONLY by the pinned official harness
in scripts/run_swebench_harness.py, which merges completed/resolved verdicts
into the final ``{condition_id}.jsonl``. They are never silently excluded.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from reverse_reap.causal import CausalError, score_condition


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--generations-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluator-image-file", type=Path, required=True)
    args = parser.parse_args()

    spec = json.loads(args.conditions.read_text(encoding="utf-8"))
    if spec.get("schema_version") != 1:
        raise SystemExit(f"unsupported conditions spec schema: {spec.get('schema_version')}")
    image = args.evaluator_image_file.read_text(encoding="utf-8").strip()
    if "@sha256:" not in image:
        raise SystemExit(f"evaluator image pin is not digest-pinned: {image!r}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for condition in spec["conditions"]:
        condition_id = condition["condition_id"]
        generated = args.generations_dir / f"{condition_id}.jsonl"
        destination = args.output_dir / f"{condition_id}.preswebench.jsonl"
        if destination.exists():
            print(f"[skip] {condition_id}: already pre-scored", flush=True)
            continue
        if not generated.exists():
            raise SystemExit(f"generation file missing: {generated}")
        started = time.monotonic()
        summary = score_condition(
            generated, args.dataset_manifest, destination, evaluator_image=image
        )
        summary_path = args.output_dir / f"{condition_id}.summary.json"
        summary_path.write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")
        print(
            f"[scored] {condition_id}: samples={summary['samples']} "
            f"scoreable={summary['scoreable_fraction']:.3f} "
            f"pass_rate={summary['pass_rate']:.3f} "
            f"({time.monotonic() - started:.1f}s, swebench pending harness)",
            flush=True,
        )
    print("pre-scoring complete", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CausalError as error:
        print(f"CausalError: {error}", file=sys.stderr)
        sys.exit(2)
