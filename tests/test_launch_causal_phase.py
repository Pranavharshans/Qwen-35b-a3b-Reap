"""Tests for scripts/launch_causal_phase.py (single-resolution run-ID launch)."""

import json
import re
import subprocess
import sys
from pathlib import Path

from reverse_reap.config import load_config

SCRIPT = Path(__file__).parents[1] / "scripts" / "launch_causal_phase.py"


def _run(args, cwd):
    return subprocess.run(
        [sys.executable, str(SCRIPT)] + args, capture_output=True, text=True, cwd=cwd, timeout=120
    )


def _config(root):
    return load_config(root / "configs" / "pinned-3090-bf16-gen.yaml")


def test_dry_run_resolves_run_id_once_and_scopes_state_dir(tmp_path):
    root = Path(__file__).parents[1]
    result = _run(
        [
            "--config",
            "configs/pinned-3090-bf16-gen.yaml",
            "--plan",
            "configs/execution-plan-causal-gen.yaml",
            "--state-root",
            str(tmp_path / "state"),
            "--dry-run",
        ],
        cwd=root,
    )
    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout)
    run_id = record["run_id"]
    # <UTC stamp>-qwen35a3b-direct-<sha8>-<fp8>
    parts = run_id.split("-")
    assert len(parts) == 5 and parts[1] == "qwen35a3b" and parts[2] == "direct"
    assert re.fullmatch(r"\d{8}T\d{6}Z", parts[0]), parts[0]
    assert record["state_dir"] == str(tmp_path / "state" / run_id)
    # A dry run must not create anything.
    assert not (tmp_path / "state").exists()


def test_fresh_launch_refuses_existing_state_dir(tmp_path):
    root = Path(__file__).parents[1]
    existing = tmp_path / "state" / "20260904T000000Z-qwen35a3b-direct-2a7239d5-97b91839"
    existing.mkdir(parents=True)
    result = _run(
        [
            "--config",
            "configs/pinned-3090-bf16-gen.yaml",
            "--plan",
            "configs/execution-plan-causal-gen.yaml",
            "--state-root",
            str(tmp_path / "state"),
        ],
        cwd=root,
    )
    # Refusal can come from --resume validation (exit message) or the
    # state-dir guard; either way it must not launch and must name the dir.
    assert result.returncode != 0
    assert "20260904T000000Z" not in result.stdout


def test_resume_derives_run_id_from_state_dir_name(tmp_path):
    root = Path(__file__).parents[1]
    state = tmp_path / "state" / "20260904T120000Z-qwen35a3b-direct-2a7239d5-97b91839"
    state.mkdir(parents=True)
    result = _run(
        [
            "--config",
            "configs/pinned-3090-bf16-gen.yaml",
            "--plan",
            "configs/execution-plan-causal-gen.yaml",
            "--resume",
            str(state),
            "--dry-run",
        ],
        cwd=root,
    )
    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout)
    assert record["run_id"] == state.name
    assert record["state_dir"] == str(state)


def test_resume_rejects_non_scoped_and_missing_dirs(tmp_path):
    root = Path(__file__).parents[1]
    config_arg = ["--config", "configs/pinned-3090-bf16-gen.yaml"]
    plan_arg = ["--plan", "configs/execution-plan-causal-gen.yaml"]

    bad_name = tmp_path / "state" / "not-a-run-id"
    bad_name.mkdir(parents=True)
    result = _run([*config_arg, *plan_arg, "--resume", str(bad_name), "--dry-run"], cwd=root)
    assert result.returncode != 0 and "run-ID-scoped" in result.stderr

    missing = tmp_path / "state" / "20260904T120000Z-qwen35a3b-direct-2a7239d5-97b91839"
    result = _run([*config_arg, *plan_arg, "--resume", str(missing), "--dry-run"], cwd=root)
    assert result.returncode != 0 and "does not exist" in result.stderr

    both = _run([*config_arg, *plan_arg, "--resume", str(bad_name), "--state-root", "x"], cwd=root)
    assert both.returncode != 0  # argparse error: exactly one of the two

    config = _config(root)
    assert config.run_id is None  # gen config must keep run_id open for derivation


def _import_script():
    import importlib.util

    spec = importlib.util.spec_from_file_location("launch_causal_phase", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launch_record_never_lands_in_state_dir(tmp_path, monkeypatch):
    """recover_stale_tasks globs state_dir/*.json as task state — the record
    must go to the run dir instead, or the next launch fails closed."""

    module = _import_script()
    captured = {}

    def fake_run_all(plan, config, state_dir, *, run_id):
        captured["state_dir"] = state_dir
        captured["run_id"] = run_id
        return {"status": "COMPLETE"}

    monkeypatch.setattr(module, "run_all", fake_run_all)
    monkeypatch.setattr(module, "git_sha", lambda: "f1c4e5e2" * 5)
    monkeypatch.chdir(tmp_path)
    root = Path(__file__).parents[1]
    argv = [
        "launch_causal_phase.py",
        "--config",
        str(root / "configs" / "pinned-3090-bf16-gen.yaml"),
        "--plan",
        str(root / "configs" / "execution-plan-causal-gen.yaml"),
        "--state-root",
        str(tmp_path / "state"),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert module.main() == 0
    state_dir = captured["state_dir"]
    assert not list(Path(state_dir).glob("launch-record.json"))
    record = json.loads(
        (tmp_path / "runs/causal-pilot" / captured["run_id"] / "launch-record.json").read_text()
    )
    assert record["run_id"] == captured["run_id"]
