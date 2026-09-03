#!/usr/bin/env python3
"""Fail-closed hardware/runtime check for the approved 4x3090 environment."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def collect() -> dict:
    import torch

    gpus = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        gpus.append(
            {
                "index": index,
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "capability": list(torch.cuda.get_device_capability(index)),
            }
        )
    report = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "gpus": gpus,
        "disk_free_bytes": shutil.disk_usage(Path.cwd()).free,
    }
    if shutil.which("nvidia-smi"):
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        )
        report["nvidia_driver"] = sorted(set(result.stdout.split()))
    return report


def validate(report: dict) -> list[str]:
    errors = []
    if not report["cuda_available"]:
        errors.append("CUDA is unavailable")
    if report["gpu_count"] != 4:
        errors.append(f"expected exactly 4 GPUs, found {report['gpu_count']}")
    for gpu in report["gpus"]:
        if "3090" not in gpu["name"]:
            errors.append(f"GPU {gpu['index']} is not an RTX 3090: {gpu['name']}")
        if gpu["total_memory_bytes"] < 23 * 1024**3:
            errors.append(f"GPU {gpu['index']} has less than 23 GiB VRAM")
        if gpu["capability"] < [8, 6]:
            errors.append(f"GPU {gpu['index']} compute capability is below 8.6")
    if report["disk_free_bytes"] < 220 * 1024**3:
        errors.append("less than 220 GiB disk is free")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = collect()
    errors = validate(report)
    report["passed"] = not errors
    report["errors"] = errors
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if not errors else 2


if __name__ == "__main__":
    sys.exit(main())
