#!/usr/bin/env python3
"""Build and verify a digest-pinned official EvalPlus scoring image."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

EVALPLUS_REVISION = "e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2"
HUMANEVAL_PLUS_VERSION = "v0.1.10"
HUMANEVAL_PLUS_SHA256 = "e62f4130146963d969da64553f407a66e52d095adbfed4ee6733b4d59e14a3ed"
HUMANEVAL_PLUS_PATH = "/opt/evalplus-data/HumanEvalPlus.jsonl.gz"
BASE_TAG = "python:3.12-slim"
IMAGE_REPO = "localhost:5000/reverse-reap-evalplus"


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print(f"$ {' '.join(command)}", flush=True)
    return subprocess.run(command, capture_output=True, text=True, check=check)


def digest(reference: str) -> str:
    result = run(
        ["docker", "image", "inspect", "--format", "{{index .RepoDigests 0}}", reference]
    )
    value = result.stdout.strip()
    if "@sha256:" not in value:
        raise SystemExit(f"image lacks repository digest: {reference}")
    return value


def verify_image_metadata(reference: str) -> dict[str, str]:
    labels_result = run(
        ["docker", "image", "inspect", "--format", "{{json .Config.Labels}}", reference]
    )
    try:
        labels = json.loads(labels_result.stdout)
    except json.JSONDecodeError as error:
        raise SystemExit("built image has invalid OCI labels") from error
    if not isinstance(labels, dict):
        raise SystemExit("built image is missing OCI labels")
    expected = {
        "org.opencontainers.image.revision": EVALPLUS_REVISION,
        "org.opencontainers.image.humanevalplus.version": HUMANEVAL_PLUS_VERSION,
        "org.opencontainers.image.humanevalplus.path": HUMANEVAL_PLUS_PATH,
        "org.opencontainers.image.humanevalplus.sha256": HUMANEVAL_PLUS_SHA256,
    }
    for key, value in expected.items():
        if labels.get(key) != value:
            raise SystemExit(f"built image label {key!r} is not pinned to {value!r}")

    content = run(
        [
            "docker",
            "run",
            "--rm",
            "--network=none",
            "--read-only",
            reference,
            "sha256sum",
            HUMANEVAL_PLUS_PATH,
        ],
        check=False,
    )
    actual = content.stdout.strip().split(maxsplit=1)[0] if content.stdout.strip() else ""
    if content.returncode != 0 or actual != HUMANEVAL_PLUS_SHA256:
        raise SystemExit("built image HumanEval+ artifact hash does not match its pin")
    return {
        "humanevalplus_version": HUMANEVAL_PLUS_VERSION,
        "humanevalplus_path": HUMANEVAL_PLUS_PATH,
        "humanevalplus_sha256": HUMANEVAL_PLUS_SHA256,
    }


def ensure_registry() -> None:
    state = run(
        ["docker", "inspect", "-f", "{{.State.Running}}", "reap-registry"], check=False
    )
    if state.returncode == 0 and state.stdout.strip() == "true":
        return
    if state.returncode == 0:
        run(["docker", "rm", "-f", "reap-registry"])
    run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            "reap-registry",
            "-p",
            "127.0.0.1:5000:5000",
            "registry:2",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if run(["docker", "version"], check=False).returncode != 0:
        raise SystemExit("Docker with a running daemon is required")
    ensure_registry()
    run(["docker", "pull", BASE_TAG])
    base = digest(BASE_TAG)
    tag = f"{IMAGE_REPO}:{EVALPLUS_REVISION[:8]}"
    run(
        [
            "docker",
            "build",
            "-f",
            "deploy/evalplus.Dockerfile",
            "--build-arg",
            f"BASE_IMAGE={base}",
            "--build-arg",
            f"EVALPLUS_REVISION={EVALPLUS_REVISION}",
            "--build-arg",
            f"HUMANEVAL_PLUS_VERSION={HUMANEVAL_PLUS_VERSION}",
            "--build-arg",
            f"HUMANEVAL_PLUS_SHA256={HUMANEVAL_PLUS_SHA256}",
            "-t",
            tag,
            ".",
        ]
    )
    run(["docker", "push", tag])
    pinned = digest(tag)
    revision = run(
        [
            "docker",
            "image",
            "inspect",
            "--format",
            '{{index .Config.Labels "org.opencontainers.image.revision"}}',
            pinned,
        ]
    ).stdout.strip()
    dataset = verify_image_metadata(pinned)
    # The pinned EvalPlus revision exposes its CLI through Google Fire, and
    # ``--help`` exits with status 2. Probe an offline import of the exact
    # evaluate entrypoint instead of relying on a help exit code.
    probe_command = (
        "from evalplus.evaluate import evaluate; "
        "from evalplus.sanitize import sanitize; "
        "print('evalplus-import-ok')"
    )
    probe = run(
        [
            "docker",
            "run",
            "--rm",
            "--network=none",
            pinned,
            "python",
            "-c",
            probe_command,
        ],
        check=False,
    )
    if (
        revision != EVALPLUS_REVISION
        or probe.returncode != 0
        or "evalplus-import-ok" not in probe.stdout
    ):
        raise SystemExit("built image failed EvalPlus revision/CLI verification")
    (args.output_dir / "evalplus-image.txt").write_text(pinned + "\n", encoding="utf-8")
    report = {
        "base_image": base,
        "evalplus_revision": EVALPLUS_REVISION,
        "evalplus_image": pinned,
        **dataset,
        "cli_probe": "PASS (offline import of evaluate and sanitize)",
    }
    (args.output_dir / "evalplus-image-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(pinned)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
