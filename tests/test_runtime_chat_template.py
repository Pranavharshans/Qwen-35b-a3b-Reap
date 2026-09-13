import pytest

torch = pytest.importorskip("torch")

from reverse_reap.datasets import normalize_sample  # noqa: E402
from reverse_reap.runtime import RuntimeCompatibilityError, _render_ids  # noqa: E402


def _sample():
    return normalize_sample(
        {
            "source": "fixture",
            "source_revision": "abc",
            "source_id": "chat-template",
            "domain": "coding",
            "stratum": "synthesis",
            "language": "python",
            "prompt": "write code",
            "reference": "pass",
            "scorer": "exact_match",
        },
        seed=1,
    )


class ThinkingTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        if len(messages) == 2:
            return {"input_ids": torch.tensor([[1, 2, 3, 8, 4, 5]])}
        if kwargs["add_generation_prompt"]:
            return {"input_ids": torch.tensor([[1, 2, 3, 9]])}
        return {"input_ids": torch.tensor([[1, 2, 3]])}


def test_teacher_forcing_uses_maximal_stable_chat_template_prefix():
    prompt, full = _render_ids(ThinkingTokenizer(), _sample(), True)
    assert prompt.tolist() == [[1, 2, 3]]
    assert full.tolist() == [[1, 2, 3, 8, 4, 5]]


class BrokenUserPrefixTokenizer(ThinkingTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        if len(messages) == 2:
            return {"input_ids": torch.tensor([[1, 7, 3, 8, 4]])}
        return super().apply_chat_template(messages, **kwargs)


def test_teacher_forcing_still_fails_closed_on_user_prefix_drift():
    with pytest.raises(RuntimeCompatibilityError, match="user-message prefix"):
        _render_ids(BrokenUserPrefixTokenizer(), _sample(), True)
