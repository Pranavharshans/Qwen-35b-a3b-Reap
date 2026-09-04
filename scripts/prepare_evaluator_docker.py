"""Prepare Docker and the digest-pinned evaluator image on the GPU host.

Idempotent and fail-closed. Performs, in order:

1. Verify (or install) Docker and a running daemon.
2. Resolve a digest pin for ubuntu:24.04 and build the evaluator base image
   (python3 + JDK 21) from deploy/evaluator-base.Dockerfile.
3. Start a localhost-only registry, push the base, rebuild the frozen
   evaluator/Dockerfile against the base digest, and push the evaluator.
4. Sanity-run the final digest under the exact lockdown flags that
   reverse_reap.evaluator uses, for both python3 and javac.
5. Write evaluator-image.txt (the ``name@sha256:...`` pin consumed by the
   scoring task) and docker-report.json (every resolved digest + versions).

Any failure exits nonzero; nothing is left half-pinned.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

UBUNTU_TAG = "ubuntu:24.04"
REGISTRY = "localhost:5000"
BASE_REPO = f"{REGISTRY}/reverse-reap-evaluator-base"
EVAL_REPO = f"{REGISTRY}/reverse-reap-evaluator"


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(command)}", flush=True)
    return subprocess.run(command, capture_output=True, text=True, check=check)


def docker_available() -> bool:
    try:
        result = run(["docker", "version", "--format", "{{.Server.Version}}"], check=False)
    except FileNotFoundError:
        return False
    return result.returncode == 0


def ensure_docker(report: dict) -> None:
    if docker_available():
        report["docker_installed_now"] = False
        return
    print("docker missing; installing docker.io via apt...", flush=True)
    run(["apt-get", "update"])
    run(["apt-get", "install", "-y", "--no-install-recommends", "docker.io"])
    if not docker_available():
        run(["service", "docker", "start"], check=False)
        for _ in range(30):
            if docker_available():
                break
            time.sleep(2)
    if not docker_available():
        raise SystemExit("docker daemon did not become ready")
    report["docker_installed_now"] = True
    report["docker_version"] = run(
        ["docker", "version", "--format", "{{.Server.Version}}"]
    ).stdout.strip()


def repo_digest(reference: str) -> str:
    result = run(["docker", "image", "inspect", "--format", "{{index .RepoDigests 0}}", reference])
    digest = result.stdout.strip()
    if "@sha256:" not in digest:
        raise SystemExit(f"no sha256 repo digest for {reference}: {digest!r}")
    return digest


def ensure_registry() -> None:
    state = run(
        ["docker", "inspect", "-f", "{{.State.Running}}", "reap-registry"], check=False
    )
    if state.returncode == 0 and state.stdout.strip() == "true":
        return
    if state.returncode == 0:
        run(["docker", "rm", "-f", "reap-registry"])
    run(["docker", "run", "-d", "--name", "reap-registry", "-p", "127.0.0.1:5000:5000",
         "registry:2"])
    for _ in range(30):
        check = run(["docker", "inspect", "-f", "{{.State.Running}}", "reap-registry"],
                    check=False)
        if check.returncode == 0 and check.stdout.strip() == "true":
            return
        time.sleep(2)
    raise SystemExit("local registry did not become ready")


def sanity_run(image: str, output_dir: Path) -> None:
    program = "assert 1 + 1 == 2\nprint('sanity-ok')\n"
    with tempfile.TemporaryDirectory(prefix="reap-eval-sanity-") as directory:
        source = Path(directory) / "submission.py"
        source.write_text(program, encoding="utf-8")
        result = run(
            [
                "docker", "run", "--rm", "--network=none", "--read-only", "--cap-drop=ALL",
                "--security-opt=no-new-privileges", "--pids-limit=64", "--memory=512m",
                "--cpus=1", "--tmpfs=/tmp:rw,noexec,nosuid,size=64m", "--user=65534:65534",
                "--mount", f"type=bind,src={source},dst=/submission.py,readonly",
                image, "python3", "-I", "-B", "/submission.py",
            ],
            check=False,
        )
    if result.returncode != 0 or "sanity-ok" not in result.stdout:
        raise SystemExit(f"python sanity run failed: rc={result.returncode} {result.stderr[-500:]}")
    javac = run(["docker", "run", "--rm", "--network=none", image, "javac", "-version"],
                check=False)
    if javac.returncode != 0:
        raise SystemExit(f"javac sanity run failed: {javac.stderr[-500:]}")
    (output_dir / "sanity-python.txt").write_text(result.stdout, encoding="utf-8")
    (output_dir / "sanity-javac.txt").write_text(javac.stdout + javac.stderr, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pin_path = args.output_dir / "evaluator-image.txt"
    report_path = args.output_dir / "docker-report.json"

    report: dict = {}
    ensure_docker(report)

    if pin_path.exists():
        pinned = pin_path.read_text(encoding="utf-8").strip()
        sanity_run(pinned, args.output_dir)
        report["evaluator_digest"] = pinned
        report["reused_existing_pin"] = True
        report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n",
                               encoding="utf-8")
        print(f"evaluator image already pinned and sane: {pinned}", flush=True)
        return 0

    ensure_registry()
    run(["docker", "pull", UBUNTU_TAG])
    ubuntu_digest = repo_digest(UBUNTU_TAG)
    report["ubuntu_digest"] = ubuntu_digest

    run(["docker", "build", "-f", "deploy/evaluator-base.Dockerfile",
         "--build-arg", f"BASE_IMAGE={ubuntu_digest}", "-t", f"{BASE_REPO}:pinned", "deploy"])
    run(["docker", "push", f"{BASE_REPO}:pinned"])
    base_digest = repo_digest(f"{BASE_REPO}:pinned")
    report["base_digest"] = base_digest

    run(["docker", "build", "-f", "evaluator/Dockerfile",
         "--build-arg", f"BASE_IMAGE={base_digest}", "-t", f"{EVAL_REPO}:pinned", "."])
    run(["docker", "push", f"{EVAL_REPO}:pinned"])
    evaluator_digest = repo_digest(f"{EVAL_REPO}:pinned")
    report["evaluator_digest"] = evaluator_digest

    sanity_run(evaluator_digest, args.output_dir)
    pin_path.write_text(evaluator_digest + "\n", encoding="utf-8")
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8")
    print(f"evaluator image pinned: {evaluator_digest}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
