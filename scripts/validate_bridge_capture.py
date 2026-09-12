#!/usr/bin/env python3
"""Validate all completed targeted-capture shards and coverage state."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from reverse_reap.bridge_capture import (
    BridgeCaptureError,
    load_bridge_manifest,
    load_target_capture_state,
    validate_target_shard,
)
from reverse_reap.donors import donor_contract


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_root", type=Path)
    parser.add_argument("capture_manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--hidden-size", type=int)
    args = parser.parse_args()
    try:
        manifest = load_bridge_manifest(args.capture_manifest)
        hidden_size = args.hidden_size or donor_contract(manifest.model_id).hidden_size
        state_path = args.capture_root / "capture-state.json"
        if not state_path.is_file():
            raise BridgeCaptureError("capture-state.json is missing")
        state = load_target_capture_state(
            state_path,
            run_id=manifest.run_id,
            capture_manifest_sha256=manifest.manifest_sha256,
        )
        if state is None:
            raise BridgeCaptureError("capture state is missing")
        if not state.complete or not state.coverage.get("capture_success"):
            raise BridgeCaptureError("capture state does not prove successful coverage")
        shard_paths = sorted(
            path for path in args.capture_root.glob("shard-*") if path.is_dir()
        )
        if {path.name for path in shard_paths} != set(state.completed_shards):
            raise BridgeCaptureError("state and shard directory sets differ")
        reports = [
            validate_target_shard(path, hidden_size=hidden_size) for path in shard_paths
        ]
        target_experts = sorted((item.layer, item.expert) for item in manifest.experts)
        for path in shard_paths:
            payload = json.loads((path / "shard.json").read_text(encoding="utf-8"))
            shard_experts = sorted(
                (item["layer"], item["expert"]) for item in payload["target_experts"]
            )
            if shard_experts != target_experts:
                raise BridgeCaptureError("shard candidate set differs from manifest")
            if (
                payload.get("run_id") != manifest.run_id
                or payload.get("source_manifest_hash") != manifest.source_manifest_sha256
                or payload.get("config_sha256") != manifest.config_sha256
                or payload.get("candidate_manifest_sha256")
                != manifest.candidate_manifest_sha256
            ):
                raise BridgeCaptureError("shard provenance hashes differ from manifest")
        report = {
            "passed": True,
            "run_id": manifest.run_id,
            "manifest_sha256": manifest.manifest_sha256,
            "shard_count": len(reports),
            "record_count": sum(item["record_count"] for item in reports),
            "analyzed_tokens": int(state.coverage["analyzed_tokens"]),
            "routed_target_tokens": sum(item["analyzed_tokens"] for item in reports),
            "coverage": state.coverage,
        }
    except (BridgeCaptureError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        report = {"passed": False, "error": str(error)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
