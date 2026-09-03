from datetime import UTC, datetime
from pathlib import Path

from reverse_reap.config import load_config


def test_3090_smoke_disables_thinking() -> None:
    config = load_config(Path("configs/smoke-3090-bf16.yaml"))
    assert config.model.id == "Qwen/Qwen3.5-35B-A3B"
    assert config.model.execution_precision == "bf16"
    assert config.runtime.enable_thinking is False
    assert config.runtime.use_cache is False


def test_run_id_separates_thinking_condition() -> None:
    direct = load_config(Path("configs/smoke-3090-bf16.yaml"))
    thinking = load_config(Path("configs/thinking-pilot-3090-bf16.yaml"))
    now = datetime(2026, 9, 3, tzinfo=UTC)
    assert "-direct-" in direct.resolve_run_id("a" * 40, now)
    assert "-think-" in thinking.resolve_run_id("a" * 40, now)
    assert direct.fingerprint() != thinking.fingerprint()
