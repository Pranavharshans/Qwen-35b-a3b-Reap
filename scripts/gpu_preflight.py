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


def validate(report: dict, profile: str = "4x3090") -> list[str]:
    errors = []
    if not report["cuda_available"]:
        errors.append("CUDA is unavailable")
    if profile == "pro6000":
        # Single RTX PRO 6000 Blackwell 96 GB (sm_120) hosting the B8-qualified
        # causal-generation run. Torch <2.11 (cu126, sm<=90) cannot execute a
        # single CUDA op on sm_120, so the stack version is gated here, not
        # merely recorded; the empirical gates (diagnostic, noop-equiv)
        # re-validate the stack on the host before any intervention
        # generation.
        if report["gpu_count"] != 1:
            errors.append(f"expected exactly 1 GPU, found {report['gpu_count']}")
        for gpu in report["gpus"]:
            if "PRO 6000" not in gpu["name"]:
                errors.append(f"GPU {gpu['index']} is not an RTX PRO 6000: {gpu['name']}")
            if gpu["total_memory_bytes"] < 90 * 1024**3:
                errors.append(f"GPU {gpu['index']} has less than 90 GiB VRAM")
            if gpu["capability"] < [12, 0]:
                errors.append(f"GPU {gpu['index']} compute capability is below 12.0")
        torch_version = str(report.get("torch", ""))
        if not torch_version.startswith("2.11."):
            errors.append(f"PRO 6000 sm_120 requires torch 2.11.x+cu128, found {torch_version}")
        cuda_runtime = str(report.get("cuda_runtime", ""))
        if "12.8" not in cuda_runtime:
            errors.append(f"torch 2.11+cu128 requires CUDA runtime 12.8, found {cuda_runtime}")
        if report["disk_free_bytes"] < 120 * 1024**3:
            errors.append("less than 120 GiB disk is free")
    elif profile == "4x3090":
        if report["gpu_count"] != 4:
            errors.append(f"expected exactly 4 GPUs, found {report['gpu_count']}")
        for gpu in report["gpus"]:
            if "3090" not in gpu["name"]:
                errors.append(f"GPU {gpu['index']} is not an RTX 3090: {gpu['name']}")
            if gpu["total_memory_bytes"] < 23 * 1024**3:
                errors.append(f"GPU {gpu['index']} has less than 23 GiB VRAM")
            if gpu["capability"] < [8, 6]:
                errors.append(f"GPU {gpu['index']} compute capability is below 8.6")
        if report["disk_free_bytes"] < 100 * 1024**3:
            errors.append("less than 100 GiB disk is free")
    else:
        errors.append(f"unknown preflight profile: {profile}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile", choices=["4x3090", "pro6000"], default="4x3090")
    args = parser.parse_args()
    report = collect()
    errors = validate(report, profile=args.profile)
    report["profile"] = args.profile
    report["passed"] = not errors
    report["errors"] = errors
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if not errors else 2


if __name__ == "__main__":
    sys.exit(main())
