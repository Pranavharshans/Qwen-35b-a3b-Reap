"""Streaming validation for the token-level routing telemetry contract."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


class TelemetryError(ValueError):
    """Raised when routing evidence violates an identity or numeric invariant."""


def merge_telemetry(inputs: list[Path], destination: Path) -> dict[str, Any]:
    if len(inputs) < 2:
        raise TelemetryError("at least two telemetry inputs are required")
    if destination.exists():
        raise TelemetryError(f"refusing to overwrite merged telemetry: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    seen: set[tuple[str, int, int, int]] = set()
    splits: set[str] = set()
    count = 0
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            for path in inputs:
                with path.open(encoding="utf-8") as source:
                    for line in source:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        identity = (
                            str(row["sample_id"]),
                            int(row["token_index"]),
                            int(row["layer_index"]),
                            int(row["route_rank"]),
                        )
                        if identity in seen:
                            raise TelemetryError(
                                f"duplicate routing identity while merging: {identity}"
                            )
                        seen.add(identity)
                        splits.add(str(row["split"]))
                        output.write(json.dumps(row, sort_keys=True) + "\n")
                        count += 1
            output.flush()
            os.fsync(output.fileno())
        if splits - {"calibration", "selection"}:
            raise TelemetryError(f"merged candidate telemetry leaks held-out splits: {splits}")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {
        "routing_rows": count,
        "splits": sorted(splits),
        "inputs": [str(path) for path in inputs],
    }


def validate_telemetry(
    path: Path, *, num_layers: int = 40, num_experts: int = 256, top_k: int = 8
) -> dict[str, Any]:
    groups: dict[tuple[str, int, int], dict[str, set[int] | int]] = defaultdict(
        lambda: {"ranks": set(), "experts": set(), "rows": 0}
    )
    tokens: set[tuple[str, int]] = set()
    chunks: set[str] = set()
    row_count = 0
    required = {
        "schema_version",
        "run_id",
        "sample_id",
        "condition_id",
        "segment",
        "token_index",
        "token_id",
        "layer_index",
        "expert_index",
        "route_rank",
        "router_weight",
        "expert_output_l2",
        "chunk_id",
    }
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = required - row.keys()
            if missing:
                raise TelemetryError(f"line {line_number} missing fields: {sorted(missing)}")
            layer, expert, rank = (
                int(row["layer_index"]),
                int(row["expert_index"]),
                int(row["route_rank"]),
            )
            if not 0 <= layer < num_layers:
                raise TelemetryError(f"line {line_number} layer out of range")
            if not 0 <= expert < num_experts:
                raise TelemetryError(f"line {line_number} expert out of range")
            if not 0 <= rank < top_k:
                raise TelemetryError(f"line {line_number} rank out of range")
            weight, norm = float(row["router_weight"]), float(row["expert_output_l2"])
            if not math.isfinite(weight) or weight < 0 or not math.isfinite(norm) or norm < 0:
                raise TelemetryError(f"line {line_number} has invalid numeric telemetry")
            token = (str(row["sample_id"]), int(row["token_index"]))
            key = (*token, layer)
            group = groups[key]
            group["rows"] = int(group["rows"]) + 1
            ranks = group["ranks"]
            experts = group["experts"]
            assert isinstance(ranks, set) and isinstance(experts, set)
            ranks.add(rank)
            experts.add(expert)
            tokens.add(token)
            chunks.add(str(row["chunk_id"]))
            row_count += 1
    if not row_count:
        raise TelemetryError("telemetry is empty")
    expected_groups = len(tokens) * num_layers
    if len(groups) != expected_groups:
        raise TelemetryError(f"expected {expected_groups} token-layer groups, got {len(groups)}")
    for key, group in groups.items():
        if group["rows"] != top_k or len(group["ranks"]) != top_k or len(group["experts"]) != top_k:
            raise TelemetryError(f"token-layer group {key} does not have top-k unique routes")
    expected_rows = len(tokens) * num_layers * top_k
    if row_count != expected_rows:
        raise TelemetryError(f"expected {expected_rows} rows, got {row_count}")
    return {
        "valid": True,
        "routing_rows": row_count,
        "analysed_tokens": len(tokens),
        "token_layer_groups": len(groups),
        "chunks": len(chunks),
        "num_layers": num_layers,
        "num_experts": num_experts,
        "top_k": top_k,
    }
