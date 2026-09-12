"""Approved donor contracts shared by configuration, preflight, and runtime."""

from __future__ import annotations

from dataclasses import dataclass

QWEN35_MODEL_ID = "Qwen/Qwen3.5-35B-A3B"
QWEN38_MODEL_ID = "Qwen/Qwen3.8-Flash-Next"


@dataclass(frozen=True)
class DonorContract:
    model_id: str
    slug: str
    root_model_type: str
    architecture: str
    text_model_type: str
    num_hidden_layers: int
    hidden_size: int
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    dtype: str = "bfloat16"
    expected_revision: str | None = None

    def expected_text_config(self) -> dict[str, object]:
        return {
            "model_type": self.text_model_type,
            "num_hidden_layers": self.num_hidden_layers,
            "hidden_size": self.hidden_size,
            "num_experts": self.num_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "moe_intermediate_size": self.moe_intermediate_size,
            "shared_expert_intermediate_size": self.shared_expert_intermediate_size,
            "dtype": self.dtype,
        }


DONOR_CONTRACTS = {
    QWEN35_MODEL_ID: DonorContract(
        model_id=QWEN35_MODEL_ID,
        slug="qwen35a3b",
        root_model_type="qwen3_5_moe",
        architecture="Qwen3_5MoeForConditionalGeneration",
        text_model_type="qwen3_5_moe_text",
        num_hidden_layers=40,
        hidden_size=2048,
        num_experts=256,
        num_experts_per_tok=8,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        expected_revision="59d61f3ce65a6d9863b86d2e96597125219dc754",
    ),
    QWEN38_MODEL_ID: DonorContract(
        model_id=QWEN38_MODEL_ID,
        slug="qwen38flashnext",
        root_model_type="qwen4_exp",
        architecture="Qwen4ExpForConditionalGeneration",
        text_model_type="qwen4_exp_text",
        num_hidden_layers=48,
        hidden_size=2560,
        num_experts=512,
        num_experts_per_tok=10,
        moe_intermediate_size=640,
        shared_expert_intermediate_size=640,
        expected_revision="de4b8e4d43b917e7706784d8bb445c9af86a3540",
    ),
}


def donor_contract(model_id: str) -> DonorContract:
    try:
        return DONOR_CONTRACTS[model_id]
    except KeyError as error:
        raise ValueError(f"unsupported donor model: {model_id}") from error
