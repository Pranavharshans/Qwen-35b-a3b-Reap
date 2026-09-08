"""Multi-turn Qwen driver over the validated swe-edit-v1 scaffold.

The scaffold (``swe_edit``) owns every repository side effect: sessions are created
from the six-field task projection, blobs are materialized from the exact frozen
base commit, and each model turn executes at most one tool action with
deterministic, recorded results. This module owns the conversation loop only:

- The donor model is loaded once by the caller and reused across sessions.
- Every turn sends the frozen issue plus the deterministic tool history.
- Exactly one raw model response is accepted per turn and parsed strictly by
  the scaffold: no Markdown extraction, JSON repair, or guessing.
- Every prompt and response is counted with the exact donor tokenizer.
- Production generation uses greedy decoding with thinking disabled, wrapped
  per forward in the existing selected-expert intervention
  (``intervene_qwen35``), identically for baseline, repeated-baseline, no-op,
  and selected conditions.
- The final patch is generated mechanically by the scaffold from accepted
  exact-match edits. Model-selected code is never rewritten or repaired.

The driver is model-agnostic: production passes a Qwen ``generate_fn`` while
tests pass a scripted fake with the same ``messages -> raw response`` contract.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reverse_reap.swe_edit import (
    EditPolicy,
    EditSession,
    EditSessionError,
    canonical_json,
    protocol_prompt,
    write_source_binding,
)

#: Consecutive identical tool errors before the session stops fail-closed.
#: Two is conservative: a deterministic model that repeats the exact failing
#: action has no new information to gain from a third attempt.
MAX_CONSECUTIVE_IDENTICAL_ERRORS = 2

#: Raw model responses become ``generate_fn`` input as chat messages. The tool
#: result is appended as a user message so the conversation stays within the
#: user/assistant roles supported by the donor chat template.
TOOL_ROLE = "user"


class AgenticError(RuntimeError):
    """The agentic loop cannot continue without weakening an invariant."""


#: ``generate_fn`` renders one assistant turn from the full conversation.
#: Production wraps donor generation (plus per-condition intervention);
#: tests use a scripted fake. It must be deterministic for fixed inputs.
GenerateFn = Callable[[list[dict[str, str]]], str]


def build_initial_messages(task: dict[str, Any], policy: EditPolicy) -> list[dict[str, str]]:
    """Frozen issue plus tool protocol as the first user turn."""
    return [{"role": "user", "content": protocol_prompt(task, policy)}]


def render_tool_result(result: dict[str, Any]) -> str:
    """Deterministic tool-result rendering shared by every condition."""
    return canonical_json(result).decode()


def count_input_tokens(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    """Exact donor-tokenizer prompt count with thinking disabled.

    Rendered first and encoded with the tokenizer ``__call__`` path, matching
    the production left-padded batch encoding exactly. (``apply_chat_template``
    with ``tokenize=True`` is not used: several transformers 5.x releases
    return per-turn ``Encoding`` objects instead of token ids there.)
    """
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    return len(tokenizer(text)["input_ids"])


def count_output_tokens(tokenizer: Any, raw_response: str) -> int:
    """Exact donor-tokenizer response count."""
    return len(tokenizer.encode(raw_response))


def _error_signature(result: dict[str, Any]) -> tuple[str, str, str] | None:
    if "error" not in result:
        return None
    return (str(result.get("action")), str(result.get("error")), str(result.get("message")))


@dataclass
class SessionSummary:
    """Hash-bound outcome of one driven sample-condition session."""

    sample_id: str
    condition_id: str
    run_id: str
    status: str
    turns: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    stop_reason: str
    patch_sha256: str | None = None
    transcript_sha256: str | None = None
    error: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)


class SessionDriver:
    """Drive one scaffold session turn by turn until finish or fail-closed stop."""

    def __init__(
        self,
        session: EditSession,
        policy: EditPolicy,
        tokenizer: Any,
        *,
        condition_id: str,
        run_id: str,
        model_max_input_tokens: int,
    ) -> None:
        self.session = session
        self.policy = policy
        self.tokenizer = tokenizer
        self.condition_id = condition_id
        self.run_id = run_id
        self.model_max_input_tokens = model_max_input_tokens
        # Rebuild the exact initial user turn from the bound task record. The
        # scaffold stores the task without the local-only repo_dir, so the
        # source directory is recovered from the session sidecar.
        self._messages = build_initial_messages(
            {**session.state["task"], "repo_dir": self._source_dir(session)}, policy,
        )
        # Replay completed turns so a resumed session continues with the exact
        # conversation (and token counts) it would have had uninterrupted.
        # Rendering reuses the stored tool results verbatim.
        for record in session.state.get("transcript", []):
            result = record.get("tool_result", {})
            if result.get("action") == "finish" and result.get("status") == "COMPLETE":
                break
            self._messages.append({"role": "assistant",
                                   "content": record.get("model_response", "")})
            self._messages.append({"role": TOOL_ROLE,
                                   "content": render_tool_result(result)})
        self._consecutive_identical_errors = 0
        self._last_error_signature: tuple[str, str, str] | None = None
        # Continue an error streak that was already running when the session
        # was interrupted, so a crash cannot reset the fail-closed counter.
        for record in reversed(session.state.get("transcript", [])):
            signature = _error_signature(record.get("tool_result", {}))
            if signature is None or (self._last_error_signature is not None
                                     and signature != self._last_error_signature):
                break
            self._last_error_signature = signature
            self._consecutive_identical_errors += 1

    @staticmethod
    def _source_dir(session: EditSession) -> str:
        marker = session.root / "source-path.txt"
        if marker.exists():
            return marker.read_text(encoding="utf-8").strip()
        return ""

    @property
    def messages(self) -> list[dict[str, str]]:
        return [dict(message) for message in self._messages]

    def _check_model_context(self, input_tokens: int) -> None:
        if input_tokens > self.model_max_input_tokens:
            raise AgenticError(
                f"turn prompt {input_tokens} exceeds model input limit "
                f"{self.model_max_input_tokens}"
            )

    def _note_tool_result(self, result: dict[str, Any]) -> None:
        signature = _error_signature(result)
        if signature is None:
            self._consecutive_identical_errors = 0
            self._last_error_signature = None
            return
        if signature == self._last_error_signature:
            self._consecutive_identical_errors += 1
        else:
            self._consecutive_identical_errors = 1
            self._last_error_signature = signature
        if self._consecutive_identical_errors >= MAX_CONSECUTIVE_IDENTICAL_ERRORS:
            raise AgenticError(f"repeated identical tool error: {signature[0]} {signature[1]}")

    def run(self, generate_fn: GenerateFn) -> SessionSummary:
        """Run turns until scaffold COMPLETE or any fail-closed stop."""
        sample_id = str(self.session.state["task"].get("sample_id", ""))
        history: list[dict[str, Any]] = []
        if self.session.state.get("status") != "RUNNING":
            return self._summary(
                sample_id, "ERROR", "already-terminal", history,
                error="session is not running",
            )
        try:
            while True:
                input_tokens = count_input_tokens(self.tokenizer, self._messages)
                self._check_model_context(input_tokens)
                raw_response = generate_fn(self.messages)
                if not isinstance(raw_response, str) or not raw_response:
                    raise AgenticError("model turn must be a non-empty string")
                output_tokens = count_output_tokens(self.tokenizer, raw_response)
                try:
                    result = self.session.process_turn(
                        raw_response,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )
                except EditSessionError as exc:
                    return self._summary(
                        sample_id, "ERROR", str(exc), history,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                history.append({"input_tokens": input_tokens,
                                "output_tokens": output_tokens,
                                "result": result})
                if result.get("status") == "COMPLETE":
                    artifacts = result.get("artifacts", {})
                    return self._summary(
                        sample_id, "COMPLETE", "finish", history,
                        patch_sha256=artifacts.get("patch", {}).get("sha256"),
                        transcript_sha256=artifacts.get("transcript", {}).get("sha256"),
                    )
                if "error" in result:
                    try:
                        self._note_tool_result(result)
                    except AgenticError as exc:
                        return self._summary(
                            sample_id, "ERROR", "repeated-tool-error", history,
                            error=str(exc),
                        )
                self._messages.append({"role": "assistant", "content": raw_response})
                self._messages.append({ "role": TOOL_ROLE,
                                        "content": render_tool_result(result)})
        except AgenticError as exc:
            return self._summary(
                sample_id, "ERROR", "driver-stop", history, error=str(exc),
            )

    def _summary(
        self,
        sample_id: str,
        status: str,
        stop_reason: str,
        history: list[dict[str, Any]],
        *,
        patch_sha256: str | None = None,
        transcript_sha256: str | None = None,
        error: str | None = None,
    ) -> SessionSummary:
        state = self.session.state
        return SessionSummary(
            sample_id=sample_id,
            condition_id=self.condition_id,
            run_id=self.run_id,
            status=status,
            turns=int(state.get("turns", 0)),
            tool_calls=int(state.get("tool_calls", 0)),
            input_tokens=int(state.get("input_tokens", 0)),
            output_tokens=int(state.get("output_tokens", 0)),
            stop_reason=stop_reason,
            patch_sha256=patch_sha256,
            transcript_sha256=transcript_sha256,
            error=error,
            history=history,
        )


def run_session(
    *,
    session_dir: Path,
    task: dict[str, Any],
    policy: EditPolicy,
    tokenizer: Any,
    generate_fn: GenerateFn,
    run_id: str,
    condition_id: str,
    config_sha256: str,
    model_revision: str,
    tokenizer_sha256: str,
    model_max_input_tokens: int,
    expected_binding: dict[str, str] | None = None,
) -> SessionSummary:
    """Create (or resume) one session and drive it to a terminal summary.

    Pass ``expected_binding`` to resume an interrupted session instead of
    creating it; binding, policy, or workspace drift fails closed inside the
    scaffold. The caller owns ``generate_fn`` (donor or scripted fake).
    """
    session_dir = Path(session_dir)
    if expected_binding is None:
        write_source_binding(session_dir, Path(task["repo_dir"]))
        source_binding = (session_dir / "source-path.txt").read_bytes()
        (session_dir / "source-path.txt").unlink()
        session = EditSession.create(
            session_dir, task, policy, run_id=run_id,
            config_sha256=config_sha256, model_revision=model_revision,
            tokenizer_sha256=tokenizer_sha256,
        )
        (session_dir / "source-path.txt").write_bytes(source_binding)
    else:
        session = EditSession.open(session_dir, policy, expected_binding)
    driver = SessionDriver(
        session, policy, tokenizer, condition_id=condition_id, run_id=run_id,
        model_max_input_tokens=model_max_input_tokens,
    )
    return driver.run(generate_fn)


def summaries_equal(first: SessionSummary, second: SessionSummary) -> bool:
    """Exact transcript/patch equivalence for baseline-repeat and no-op gates."""
    return (
        first.status == second.status == "COMPLETE"
        and first.patch_sha256 == second.patch_sha256
        and first.transcript_sha256 == second.transcript_sha256
    )


def plan_sessions(
    tasks: list[dict[str, Any]], condition_ids: list[str],
) -> list[tuple[str, dict[str, Any]]]:
    """Frozen B1 execution order: conditions outer, task-file order inner.

    Identical for every run of the same inputs; every condition sees every
    task exactly once, so batching and ordering cannot leak across arms.
    """
    if not tasks or not condition_ids:
        raise AgenticError("probe needs a non-empty task list and condition list")
    if len({task["sample_id"] for task in tasks}) != len(tasks):
        raise AgenticError("duplicate task identities")
    if len(set(condition_ids)) != len(condition_ids):
        raise AgenticError("duplicate condition identities")
    return [(condition_id, task) for condition_id in condition_ids for task in tasks]
