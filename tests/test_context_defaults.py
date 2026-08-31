from forge.agent import ContextBudget


def test_default_context_budget_preserves_more_recent_working_history() -> None:
    budget = ContextBudget()

    assert budget.max_context_characters == 80_000
    assert budget.recent_observation_count == 8
    assert budget.max_single_observation_characters == 6_000
