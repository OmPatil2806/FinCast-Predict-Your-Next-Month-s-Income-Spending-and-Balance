"""The training pipeline runs the model steps in the right order (step 7.11)."""

from pipelines.run_training import model_steps


def labels(**flags):
    return [label.strip() for label, _, _ in model_steps(**flags)]


def test_default_order():
    assert labels() == ["Baselines", "Champion rule", "Export + registry", "Short histories", "Drift reference",
                        "SHAP explanations", "Anomaly model"]


def test_tuning_runs_before_the_champion_is_chosen():
    order = labels(tune=True, experiments=True)
    assert order.index("Tuning (Optuna)") < order.index("Champion rule") < order.index("Export + registry")
    assert {"LightGBM median", "LightGBM ranges", "ETS / SARIMAX"} <= set(order)


def test_every_step_is_callable():
    assert all(callable(run) for _, _, run in model_steps(tune=True, experiments=True))
