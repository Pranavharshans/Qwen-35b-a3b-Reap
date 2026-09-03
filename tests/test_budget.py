from datetime import UTC, datetime, timedelta

from reverse_reap.budget import evaluate_budget
from reverse_reap.config import BudgetConfig


def budget() -> BudgetConfig:
    return BudgetConfig(
        max_gpu_hours=10,
        max_cost_usd=10,
        provider_rate_usd_per_hour=1,
        storage_limit_gb=250,
        deadline_utc=datetime.now(UTC) + timedelta(days=1),
    )


def test_budget_preserves_twenty_percent_reserve() -> None:
    assert evaluate_budget(
        budget(), consumed_gpu_hours=4, projected_stage_gpu_hours=4
    ).allowed
    assert not evaluate_budget(
        budget(), consumed_gpu_hours=4, projected_stage_gpu_hours=4.1
    ).allowed


def test_final_stage_may_use_reserve() -> None:
    assert evaluate_budget(
        budget(), consumed_gpu_hours=4, projected_stage_gpu_hours=6, use_reserve=True
    ).allowed
