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


def test_batch_size_allows_only_qualified_sizes() -> None:
    from pydantic import ValidationError

    from reverse_reap.config import RuntimeConfig

    base = {
        "seed": 20260903,
        "deterministic": True,
        "max_input_tokens": 1024,
        "max_new_tokens": 1024,
        "enable_thinking": False,
        "use_cache": True,
        "speculative_decoding": False,
    }
    assert RuntimeConfig(batch_size=1, **base).batch_size == 1
    # B8 admitted only via the PRO 6000 benchmark qualification.
    assert RuntimeConfig(batch_size=8, **base).batch_size == 8
    # B6/B4 admitted only for the human-authorized source-v4 3072-token probe
    # after B8 OOMed on the longest prompt.
    assert RuntimeConfig(batch_size=6, **base).batch_size == 6
    assert RuntimeConfig(batch_size=4, **base).batch_size == 4
    for bad in (0, 2, 5, 16):
        try:
            RuntimeConfig(batch_size=bad, **base)
        except ValidationError:
            pass
        else:
            raise AssertionError(f"batch_size={bad} must be rejected")
