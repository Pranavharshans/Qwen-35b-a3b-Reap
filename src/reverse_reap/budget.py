"""Fail-closed GPU-hour and monetary budget checks."""

from dataclasses import dataclass
from datetime import UTC, datetime

from reverse_reap.config import BudgetConfig


@dataclass(frozen=True)
class BudgetDecision:
    allowed: bool
    projected_total_cost_usd: float
    projected_total_gpu_hours: float
    reason: str


def evaluate_budget(
    budget: BudgetConfig,
    *,
    consumed_gpu_hours: float,
    projected_stage_gpu_hours: float,
    use_reserve: bool = False,
    now: datetime | None = None,
) -> BudgetDecision:
    current = now or datetime.now(UTC)
    total_hours = consumed_gpu_hours + projected_stage_gpu_hours
    total_cost = total_hours * budget.provider_rate_usd_per_hour
    cost_limit = budget.max_cost_usd if use_reserve else budget.max_cost_usd * 0.8
    hour_limit = budget.max_gpu_hours if use_reserve else budget.max_gpu_hours * 0.8
    if current >= budget.deadline_utc:
        return BudgetDecision(False, total_cost, total_hours, "deadline reached")
    if total_hours > hour_limit:
        return BudgetDecision(False, total_cost, total_hours, "GPU-hour reserve would be exceeded")
    if total_cost > cost_limit:
        return BudgetDecision(False, total_cost, total_hours, "cost reserve would be exceeded")
    return BudgetDecision(True, total_cost, total_hours, "within configured budget")
