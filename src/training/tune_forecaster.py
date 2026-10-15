"""Step 7.6: tune the LightGBM settings per family with Optuna.

Honest split of the rolling backtest (test months, oldest -> newest):

    ... | tune folds (6) | hold-out folds (last 6)

Every trial trains on the past and is scored on the tune folds only (mean MAE of the median model).
The last 6 test months are never seen during the search; on them the default and the tuned settings
are compared once, so the before/after number is not flattered by the search itself. The search uses
a seeded TPE sampler and prunes trials whose first folds are already worse than the median trial.

Each trial is a nested MLflow run under one parent run per family. The best settings are written to
models/forecast_params.json, which steps 7.8 / 7.9 use, only when they beat the defaults on the
hold-out months; otherwise the defaults are kept and that is recorded.

Run:  python -m src.training.tune_forecaster [--trials 30]
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import optuna
import pandas as pd

from src.evaluation.backtest import backtest_folds
from src.evaluation.metrics import mae
from src.features.build_features import build_features
from src.training.train_forecaster import DEFAULT_PARAMS, FAMILIES, FAMILY_MODE, ForecastModel, family_rows
from src.utils.helpers import load_config, resolve_path
from src.utils.mlflow_utils import setup_experiment


def search_space(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 128, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def split_folds(rows: pd.DataFrame, tune_folds: int, holdout_folds: int) -> tuple[list, list]:
    """(tune folds, hold-out folds): the hold-out folds are the newest test months."""
    folds = backtest_folds(rows)
    if len(folds) < tune_folds + holdout_folds:
        raise ValueError(f"need {tune_folds + holdout_folds} test months, have {len(folds)}")
    return folds[-(tune_folds + holdout_folds):-holdout_folds], folds[-holdout_folds:]


def fold_maes(rows: pd.DataFrame, folds, family: str, feature_columns: list[str], params: dict,
              on_fold=None) -> list[float]:
    """MAE of the median model on each fold (train on the past, predict the fold's month)."""
    mode = FAMILY_MODE[family]
    levels = sorted(rows["series"].unique())
    scores = []
    for number, fold in enumerate(folds):
        train, test = rows.iloc[fold.train_idx], rows.iloc[fold.test_idx]
        model = ForecastModel(family, mode, 0.5, feature_columns, levels, params).fit(train)
        scores.append(mae(test["target"], model.predict(test)))
        if on_fold:
            on_fold(number, float(np.mean(scores)))           # raises TrialPruned when hopeless
    return scores


def tune_family(table: pd.DataFrame, family: str, feature_columns: list[str], trials: int = 30,
                tune_folds: int = 6, holdout_folds: int = 6, seed: int = 42,
                log: bool = True, base_params: dict | None = None) -> dict:
    """Optuna search on the tune folds, then default vs tuned on the untouched hold-out folds."""
    rows = family_rows(table, family)
    tune, holdout = split_folds(rows, tune_folds, holdout_folds)
    base = {**DEFAULT_PARAMS, **(base_params or {})}

    def objective(trial: optuna.Trial) -> float:
        params = {**base, **search_space(trial)}

        def on_fold(number, running_mae):
            trial.report(running_mae, number)
            if trial.should_prune():
                raise optuna.TrialPruned()

        try:
            score = float(np.mean(fold_maes(rows, tune, family, feature_columns, params, on_fold)))
        except optuna.TrialPruned:
            if log:
                with mlflow.start_run(run_name=f"trial_{trial.number:02d}_pruned", nested=True):
                    mlflow.log_params(trial.params)
                    mlflow.set_tag("state", "pruned")
            raise
        if log:
            with mlflow.start_run(run_name=f"trial_{trial.number:02d}", nested=True):
                mlflow.log_params(trial.params)
                mlflow.log_metric("tune_mae", score)
        return score

    if log:
        setup_experiment(FAMILIES[family])
        mlflow.start_run(run_name=f"optuna_{FAMILY_MODE[family]}")
        mlflow.log_params({"family": family, "mode": FAMILY_MODE[family], "trials": trials,
                           "tune_months": f"{tune[0].test_month} to {tune[-1].test_month}",
                           "holdout_months": f"{holdout[0].test_month} to {holdout[-1].test_month}"})
    try:
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed),
                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))
        study.enqueue_trial({k: DEFAULT_PARAMS[k] for k in search_space_keys()})   # trial 0 = the defaults
        study.optimize(objective, n_trials=trials)

        tuned = {**base, **study.best_params}
        default_tune = study.trials[0].value
        default_holdout = float(np.mean(fold_maes(rows, holdout, family, feature_columns, base)))
        tuned_holdout = float(np.mean(fold_maes(rows, holdout, family, feature_columns, tuned)))
        result = {
            "family": family, "mode": FAMILY_MODE[family],
            "best_params": study.best_params,
            "tune_months": f"{tune[0].test_month} to {tune[-1].test_month}",
            "holdout_months": f"{holdout[0].test_month} to {holdout[-1].test_month}",
            "trials": len(study.trials),
            "pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
            "default_tune_mae": default_tune, "best_tune_mae": study.best_value,
            "default_holdout_mae": default_holdout, "tuned_holdout_mae": tuned_holdout,
            "holdout_improvement": 1 - tuned_holdout / default_holdout if default_holdout else 0.0,
        }
        result["use_tuned"] = result["holdout_improvement"] > 0
        if log:
            mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
            mlflow.log_metrics({k: result[k] for k in ("default_tune_mae", "best_tune_mae", "default_holdout_mae",
                                                       "tuned_holdout_mae", "holdout_improvement", "pruned")})
            mlflow.set_tag("use_tuned", str(result["use_tuned"]))
    finally:
        if log:
            mlflow.end_run()
    return result


def search_space_keys() -> list[str]:
    return ["n_estimators", "learning_rate", "num_leaves", "min_child_samples", "subsample",
            "colsample_bytree", "reg_lambda"]


def save_params(results: dict, path: Path | None = None) -> Path:
    """Settings per family for the final models: tuned if they won on the hold-out months, else defaults."""
    config = load_config()
    path = path or resolve_path(config["paths"]["models"]) / config["forecast"]["params_file"]
    payload = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "families": {}}
    for family, r in results.items():
        chosen = r["best_params"] if r["use_tuned"] else {k: DEFAULT_PARAMS[k] for k in search_space_keys()}
        payload["families"][family] = {
            "mode": r["mode"], "source": "tuned" if r["use_tuned"] else "default", "params": chosen,
            "holdout_months": r["holdout_months"],
            "holdout_mae": {"default": round(r["default_holdout_mae"], 2), "tuned": round(r["tuned_holdout_mae"], 2)},
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


def _print(results: dict) -> None:
    for family, r in results.items():
        print(f"\n=== {family} ({r['mode']}): {r['trials']} trials ({r['pruned']} pruned), "
              f"tuned on {r['tune_months']}, checked on {r['holdout_months']}")
        print(f"    tune months   MAE default €{r['default_tune_mae']:,.2f} -> best €{r['best_tune_mae']:,.2f}")
        print(f"    hold-out      MAE default €{r['default_holdout_mae']:,.2f} -> tuned €{r['tuned_holdout_mae']:,.2f} "
              f"({r['holdout_improvement']:+.1%}) -> using {'tuned' if r['use_tuned'] else 'default'} settings")
        print("    best params:", {k: round(v, 4) if isinstance(v, float) else v for k, v in r["best_params"].items()})


def main(argv: list[str] | None = None) -> None:
    config = load_config()
    tuning = config["forecast"]["tuning"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=tuning["trials"])
    args = parser.parse_args(argv)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logging.getLogger("mlflow").setLevel(logging.WARNING)

    monthly = pd.read_parquet(resolve_path(config["paths"]["data_processed"]) / config["processed"]["monthly_file"])
    table, info = build_features(monthly)
    results = {family: tune_family(table, family, info["feature_columns"], args.trials, tuning["tune_folds"],
                                   tuning["holdout_folds"], config["project"]["seed"])
               for family in FAMILIES}
    _print(results)
    path = save_params(results)
    print(f"\nSaved {path.relative_to(resolve_path('.'))}")


if __name__ == "__main__":
    main()
