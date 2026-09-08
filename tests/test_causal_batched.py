"""Hardware-free tests for the B8-qualified batched production generation path."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from reverse_reap.causal import CausalError, generate_condition_batched
from reverse_reap.config import load_config


def _manifest_row(index, split="validation"):
    return {
        "schema_version": 1,
        "sample_id": f"s{index:03d}",
        "source": "fixture",
        "source_revision": "abc",
        "source_id": f"id-{index}",
        "domain": "coding",
        "stratum": "synthesis",
        "language": "python",
        "prompt": f"prompt {index}",
        "reference": "#### 42",
        "scorer": "exact_match",
        "split": split,
        "prompt_template_version": "v1",
        "content_sha256": "a" * 64,
    }


def _write_manifest(path: Path, count: int) -> Path:
    path.write_text("".join(json.dumps(_manifest_row(i)) + "\n" for i in range(count)))
    return path


class _FakeEmbeddings:
    weight = SimpleNamespace(device=torch.device("cpu"))


class _FakeModel:
    """Deterministic stand-in: appends fixed new ids to the (padded) input."""

    def __init__(self, new_ids=(7, 8, 9)):
        self.new_ids = list(new_ids)
        self.calls = 0

    def get_input_embeddings(self):
        return _FakeEmbeddings()

    def generate(self, **kwargs):
        self.calls += 1
        input_ids = kwargs["input_ids"]
        batch = input_ids.shape[0]
        new = torch.tensor([self.new_ids] * batch, dtype=torch.long)
        return torch.cat([input_ids, new], dim=1)


class _FakeTokenizer:
    eos_token_id = 0
    eos_token = "<eos>"
    pad_token_id = None
    padding_side = "right"

    def apply_chat_template(self, messages, **kwargs):
        if kwargs.get("tokenize", False):
            # Sequential _generate path: single-row tensor.
            text = messages[0]["content"]
            return torch.tensor(
                [[100 + (len(text) % 7)] * (4 + len(text) % 3)], dtype=torch.long
            )
        return "USER: " + messages[0]["content"]

    def __call__(self, texts, return_tensors=None, padding=False):
        # Variable prompt widths with left padding, mirroring the real call.
        rows = []
        for position, text in enumerate(texts):
            length = 4 + (position % 3)
            rows.append([100 + position] * length)
        width = max(len(r) for r in rows)
        padded = [[self.eos_token_id] * (width - len(r)) + r for r in rows]
        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(
                [[0] * (width - len(r)) + [1] * len(r) for r in rows],
                dtype=torch.long,
            ),
        }

    def decode(self, ids, skip_special_tokens=True):
        kept = [int(i) for i in ids.tolist() if int(i) != self.eos_token_id]
        if not kept:
            return ""  # mirrors real behavior on zero-length generation
        return "resp-" + "-".join(map(str, kept))

    def __len__(self):
        return 1000


def _architecture(num_layers=0):
    layers = tuple(
        SimpleNamespace(mlp=SimpleNamespace(experts=SimpleNamespace(forward=None)))
        for _ in range(num_layers)
    )
    return SimpleNamespace(num_layers=num_layers, layers=layers)


def _config(tmp_path: Path, batch_size: int):
    root = Path(__file__).parents[1]
    config = load_config(root / "configs" / "pinned-3090-bf16-gen.yaml")
    return config.model_copy(
        update={"runtime": config.runtime.model_copy(update={"batch_size": batch_size})}
    )


def _run(tmp_path, count=10, batch_size=8, **kwargs):
    manifest = _write_manifest(tmp_path / "manifest.jsonl", count)
    out = tmp_path / "generations" / "cX.jsonl"
    params = {
        "model": _FakeModel(),
        "tokenizer": _FakeTokenizer(),
        "architecture": _architecture(),
        "dataset_manifest": manifest,
        "destination": out,
        "config": _config(tmp_path, batch_size),
        "split": "validation",
        "condition_id": "cX",
        "checkpoint_dir": tmp_path / "checkpoints",
        "heartbeat_path": tmp_path / "heartbeat.json",
        "run_id": "test-run",
    }
    params.update(kwargs)
    summary = generate_condition_batched(**params)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    return summary, rows


def test_batched_generation_preserves_manifest_order_and_schema(tmp_path):
    summary, rows = _run(tmp_path, count=10, batch_size=8)
    assert summary["samples"] == 10
    assert summary["batch_size"] == 8
    assert summary["chunks"] == 2  # 8 + 2 partial
    assert [row["sample_id"] for row in rows] == [f"s{i:03d}" for i in range(10)]
    for row in rows:
        assert row["condition_id"] == "cX"
        assert row["split"] == "validation"
        assert row["masked_experts"] == 0
        assert row["response"].startswith("resp-")
        assert row["generated_tokens"] == 3
        assert row["truncated"] is False
    # Batch-size-8 chunking issues exactly ceil(10/8) forwards on one model.
    assert summary["chunks"] == 2


def test_batched_record_matches_sequential_schema_keys(tmp_path):
    from reverse_reap.causal import generate_condition

    manifest = _write_manifest(tmp_path / "manifest.jsonl", 3)
    config = _config(tmp_path, 1)
    seq_out = tmp_path / "seq.jsonl"
    generate_condition(
        _FakeModel(),
        _FakeTokenizer(),
        _architecture(),
        manifest,
        seq_out,
        config,
        split="validation",
        condition_id="cX",
    )
    seq_keys = set(json.loads(seq_out.read_text().splitlines()[0]))
    _, batched_rows = _run(tmp_path, count=3, batch_size=8)
    assert seq_keys <= set(batched_rows[0])


def test_batched_resume_reuses_validated_chunks(tmp_path):
    manifest = _write_manifest(tmp_path / "manifest.jsonl", 10)
    out = tmp_path / "generations" / "cX.jsonl"
    checkpoint_dir = tmp_path / "checkpoints"
    model = _FakeModel()
    kwargs = {
        "model": model,
        "tokenizer": _FakeTokenizer(),
        "architecture": _architecture(),
        "dataset_manifest": manifest,
        "destination": out,
        "config": _config(tmp_path, 8),
        "split": "validation",
        "condition_id": "cX",
        "checkpoint_dir": checkpoint_dir,
        "heartbeat_path": tmp_path / "heartbeat.json",
        "run_id": "test-run",
    }
    generate_condition_batched(**kwargs)
    first_calls = model.calls
    out.unlink()  # final missing, chunks cached -> regenerate without forwards
    model.calls = 0
    summary = generate_condition_batched(**kwargs)
    assert summary["samples"] == 10
    assert model.calls == 0


def test_batched_stale_checkpoint_is_terminal(tmp_path):
    manifest = _write_manifest(tmp_path / "manifest.jsonl", 4)
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "cX.chunk-0000.json").write_text(
        json.dumps({"sample_id": "WRONG"}) + "\n"
    )
    with pytest.raises(CausalError, match="disagree with the manifest slice"):
        _run(
            tmp_path,
            count=4,
            batch_size=8,
            dataset_manifest=manifest,
            checkpoint_dir=checkpoint_dir,
        )


def test_batched_refuses_to_overwrite_destination(tmp_path):
    manifest = _write_manifest(tmp_path / "manifest.jsonl", 2)
    out = tmp_path / "generations" / "cX.jsonl"
    out.parent.mkdir(parents=True)
    out.write_text("{}\n")
    with pytest.raises(CausalError, match="refusing to overwrite"):
        generate_condition_batched(
            _FakeModel(),
            _FakeTokenizer(),
            _architecture(),
            manifest,
            out,
            _config(tmp_path, 8),
            split="validation",
            condition_id="cX",
        )


def test_batched_empty_response_is_fail_closed(tmp_path):
    class _Empty(_FakeModel):
        def generate(self, **kwargs):
            self.calls += 1
            input_ids = kwargs["input_ids"]
            return input_ids  # zero new tokens -> empty response

    manifest = _write_manifest(tmp_path / "manifest.jsonl", 2)
    with pytest.raises(CausalError, match="empty response"):
        generate_condition_batched(
            _Empty(),
            _FakeTokenizer(),
            _architecture(),
            manifest,
            tmp_path / "out.jsonl",
            _config(tmp_path, 8),
            split="validation",
            condition_id="cX",
            checkpoint_dir=tmp_path / "checkpoints",
        )


def test_batched_heartbeat_tracks_progress(tmp_path):
    heartbeat = tmp_path / "heartbeat.json"
    _run(tmp_path, count=10, batch_size=8, heartbeat_path=heartbeat)
    payload = json.loads(heartbeat.read_text())
    assert payload["run_id"] == "test-run"
    assert payload["condition_id"] == "cX"
    assert payload["completed_items"] == 10
    assert payload["total_items"] == 10
    assert payload["completed_chunks"] == 2
    assert payload["total_chunks"] == 2
    assert payload["status"] == "complete"
    assert payload["batch_size"] == 8


def test_batched_masked_arm_records_mask_size(tmp_path):
    manifest = _write_manifest(tmp_path / "manifest.jsonl", 3)
    expert_path = tmp_path / "experts.json"
    expert_path.write_text(
        json.dumps({"experts": [{"layer": 0, "expert": 1}, {"layer": 0, "expert": 2}]})
    )
    _, rows = _run(
        tmp_path,
        count=3,
        batch_size=8,
        dataset_manifest=manifest,
        architecture=_architecture(num_layers=1),
        expert_manifest=expert_path,
    )
    assert {row["masked_experts"] for row in rows} == {2}
