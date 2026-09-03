from pathlib import Path

import pytest

from reverse_reap.state import RunState, Status, atomic_write_state, load_state


def make_state() -> RunState:
    return RunState(run_id="run", task_id="M1", config_sha256="a" * 64)


def test_valid_state_path_is_resumable(tmp_path: Path) -> None:
    state = make_state()
    state.transition(Status.PREFLIGHTED)
    state.transition(Status.RUNNING)
    state.transition(Status.VALIDATING)
    state.transition(Status.COMPLETE)
    path = tmp_path / "state.json"
    atomic_write_state(path, state)
    assert load_state(path) == state


def test_invalid_transition_fails_closed() -> None:
    state = make_state()
    with pytest.raises(ValueError, match="invalid state transition"):
        state.transition(Status.COMPLETE)


def test_third_identical_failure_is_rejected() -> None:
    state = make_state()
    state.register_retry("oom")
    state.register_retry("oom")
    with pytest.raises(ValueError, match="exceeded two retries"):
        state.register_retry("oom")

