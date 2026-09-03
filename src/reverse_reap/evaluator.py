"""Generated-code evaluation in a locked-down, disposable Docker container."""

from __future__ import annotations

import hashlib
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


class EvaluationError(RuntimeError):
    """Raised when the evaluator itself is unavailable or misconfigured."""


@dataclass(frozen=True)
class EvaluationResult:
    passed: bool
    return_code: int | None
    timed_out: bool
    stdout: str
    stderr: str
    program_sha256: str


def evaluate_python(
    generated_code: str,
    tests: str,
    *,
    image: str,
    timeout_seconds: int = 10,
    memory_mb: int = 512,
) -> EvaluationResult:
    """Execute untrusted Python with no network, capabilities, writable root, or host access."""
    if "@sha256:" not in image:
        raise EvaluationError("evaluator image must be pinned by sha256 digest")
    if not 1 <= timeout_seconds <= 120:
        raise EvaluationError("timeout_seconds must be within [1, 120]")
    if not 64 <= memory_mb <= 4096:
        raise EvaluationError("memory_mb must be within [64, 4096]")
    program = generated_code.rstrip() + "\n\n" + tests.lstrip()
    digest = hashlib.sha256(program.encode()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="reverse-reap-eval-") as directory:
        root = Path(directory)
        source = root / "submission.py"
        source.write_text(program, encoding="utf-8")
        command = [
            "docker",
            "run",
            "--rm",
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--pids-limit=64",
            f"--memory={memory_mb}m",
            "--cpus=1",
            "--tmpfs=/tmp:rw,noexec,nosuid,size=64m",
            "--user=65534:65534",
            "--mount",
            f"type=bind,src={source},dst=/submission.py,readonly",
            image,
            "python3",
            "-I",
            "-B",
            "/submission.py",
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except FileNotFoundError as error:
            raise EvaluationError("Docker is required for generated-code evaluation") from error
        except subprocess.TimeoutExpired as error:
            return EvaluationResult(
                passed=False,
                return_code=None,
                timed_out=True,
                stdout=(error.stdout or "")[-8192:],
                stderr=(error.stderr or "")[-8192:],
                program_sha256=digest,
            )
    return EvaluationResult(
        passed=completed.returncode == 0,
        return_code=completed.returncode,
        timed_out=False,
        stdout=completed.stdout[-8192:],
        stderr=completed.stderr[-8192:],
        program_sha256=digest,
    )


def evaluate_java(
    generated_code: str,
    tests: str,
    *,
    image: str,
    timeout_seconds: int = 20,
    memory_mb: int = 768,
) -> EvaluationResult:
    """Compile and execute untrusted Java inside the same locked-down boundary."""
    if "@sha256:" not in image:
        raise EvaluationError("evaluator image must be pinned by sha256 digest")
    program = generated_code.rstrip() + "\n\n" + tests.lstrip()
    digest = hashlib.sha256(program.encode()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="reverse-reap-java-eval-") as directory:
        source = Path(directory) / "Main.java"
        source.write_text(program, encoding="utf-8")
        command = [
            "docker",
            "run",
            "--rm",
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--pids-limit=64",
            f"--memory={memory_mb}m",
            "--cpus=1",
            "--tmpfs=/tmp:rw,exec,nosuid,size=128m",
            "--user=65534:65534",
            "--mount",
            f"type=bind,src={source},dst=/submission/Main.java,readonly",
            image,
            "sh",
            "-c",
            "javac -d /tmp/classes /submission/Main.java && java -cp /tmp/classes Main",
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except FileNotFoundError as error:
            raise EvaluationError("Docker is required for generated-code evaluation") from error
        except subprocess.TimeoutExpired as error:
            return EvaluationResult(
                False,
                None,
                True,
                (error.stdout or "")[-8192:],
                (error.stderr or "")[-8192:],
                digest,
            )
    return EvaluationResult(
        completed.returncode == 0,
        completed.returncode,
        False,
        completed.stdout[-8192:],
        completed.stderr[-8192:],
        digest,
    )
