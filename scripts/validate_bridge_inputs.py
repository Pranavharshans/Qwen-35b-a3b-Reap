#!/usr/bin/env python3
"""Validate immutable inputs before a targeted donor capture is launched."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from reverse_reap.bridge_capture import BridgeCaptureError, _candidate_manifest_metadata
from reverse_reap.config import load_config
from reverse_reap.datasets import load_manifest
from reverse_reap.donors import donor_contract


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("candidate_manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--require-full", action="store_true")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if not args.source_manifest.is_file():
            raise BridgeCaptureError(f"source manifest is unavailable: {args.source_manifest}")
        samples = load_manifest(args.source_manifest)
        candidate_hash, experts = _candidate_manifest_metadata(args.candidate_manifest)
        contract = donor_contract(config.model.id)
        invalid = [
            (layer, expert)
            for layer, expert in experts
            if not (0 <= layer < contract.num_hidden_layers and 0 <= expert < contract.num_experts)
        ]
        if invalid:
            raise BridgeCaptureError(
                f"candidate manifest contains experts outside {config.model.id}: {invalid}"
            )
        if args.require_full and config.datasets.split != "full":
            raise BridgeCaptureError("bridge capture requires a config with datasets.split=full")
        if Path(config.datasets.manifest) != args.source_manifest:
            raise BridgeCaptureError("config dataset manifest differs from the supplied source")
        source_hash = hashlib.sha256(args.source_manifest.read_bytes()).hexdigest()
        result = {
            "passed": True,
            "config_sha256": config.fingerprint(),
            "source_manifest_sha256": source_hash,
            "candidate_manifest_sha256": candidate_hash,
            "model_id": config.model.id,
            "model_revision": config.model.revision,
            "source_manifest": str(args.source_manifest),
            "candidate_manifest": str(args.candidate_manifest),
            "sample_count": len(samples),
            "target_expert_count": len(experts),
            "target_experts": [
                {"layer": layer, "expert": expert} for layer, expert in experts
            ],
        }
    except (BridgeCaptureError, OSError, ValueError, KeyError) as error:
        result = {"passed": False, "error": str(error)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
