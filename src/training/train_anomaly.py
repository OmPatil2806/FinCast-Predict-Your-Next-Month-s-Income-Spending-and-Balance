"""Step 7.10: train, evaluate and export the anomaly model (Isolation Forest).

The synthetic data has no "anomaly" column, so three kinds of known unusual expenses are scored:
    shock         the generator's rare big costs (SHOCKS: car repair, dentist, laptop, ...), found by
                  their description
    duplicate     planted for evaluation only: a real expense charged again 0-2 days later
    price_rise    planted for evaluation only: one subscription charge 15-60 % dearer than usual
Planting happens in a copy of the validation and test months; the exported model is trained on the
unchanged data.

Time split (no shuffling): months 1-18 train the forest; months 19-24 choose the flag rate
(contamination) by F1; months 25-36 are the test. A simple rule (any z-score >= 3 or the same amount
again within 3 days) is scored next to it as the bar to beat. As with the forecaster's champion rule
(7.8), whichever has the better F1 on the validation months decides the flags of the exported
detector; the forest's score is exported either way to rank how unusual an expense is. Results go to the MLflow experiment
"anomaly"; the final detector is saved to models/anomaly.joblib with models/anomaly_meta.json.

Run:  python -m src.training.train_anomaly
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score

from src.features.anomaly_features import MODEL_FEATURES as FEATURES
from src.features.anomaly_features import AnomalyDetector, build_anomaly_features, rule_flags
from src.ingestion.synthetic_generator import SHOCKS
from src.utils.helpers import display_path, load_config, resolve_path
from src.utils.mlflow_utils import setup_experiment
from src.utils.logger import get_logger

log = get_logger(__name__)

KINDS = ("shock", "duplicate", "price_rise")
FOREST = {"n_estimators": 200, "max_samples": 1024, "random_state": 42, "n_jobs": -1}


def plant_anomalies(transactions: pd.DataFrame, start, end, n_duplicates: int, n_price_rises: int,
                    seed: int = 42) -> pd.DataFrame:
    """Copy of the data with duplicates and subscription price rises planted between start and end.

    Adds a column `planted` ("", "duplicate" or "price_rise").
    """
    rng = np.random.default_rng(seed)
    df = transactions.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["planted"] = ""
    window = df["date"].between(pd.Timestamp(start), pd.Timestamp(end)) & (df["amount"] < 0)

    subscriptions = df.index[window & (df["category"] == "Subscriptions")]
    rises = rng.choice(subscriptions, size=min(n_price_rises, len(subscriptions)), replace=False)
    df.loc[rises, "amount"] = (df.loc[rises, "amount"] * rng.uniform(1.15, 1.6, size=len(rises))).round(2)
    df.loc[rises, "planted"] = "price_rise"

    candidates = df.index[window & ~df["category"].isin(["Rent", "Subscriptions"]) & (df["planted"] == "")]
    originals = rng.choice(candidates, size=min(n_duplicates, len(candidates)), replace=False)
    copies = df.loc[originals].copy()
    copies["date"] = copies["date"] + pd.to_timedelta(rng.integers(0, 3, size=len(copies)), unit="D")
    copies["planted"] = "duplicate"
    return pd.concat([df, copies], ignore_index=True)        # copies come after their originals


def label(features: pd.DataFrame) -> pd.Series:
    """Known kind of each expense: shock / duplicate / price_rise, or "" for normal."""
    kind = features.get("planted", pd.Series("", index=features.index)).fillna("")
    shocks = features["description"].isin([s[0] for s in SHOCKS])
    return kind.where(kind != "", np.where(shocks, "shock", ""))


def scores_report(kinds: pd.Series, flags: np.ndarray, scores: np.ndarray | None = None) -> dict:
    positive = (kinds != "").to_numpy()
    tp = int((flags & positive).sum())
    report = {
        "flag_rate": float(flags.mean()),
        "precision": tp / flags.sum() if flags.sum() else 0.0,
        "recall": tp / positive.sum() if positive.sum() else 0.0,
        "false_alarm_rate": float(flags[~positive].mean()),
        "flags": int(flags.sum()), "known_anomalies": int(positive.sum()),
    }
    report["f1"] = (2 * report["precision"] * report["recall"] / (report["precision"] + report["recall"])
                    if report["precision"] + report["recall"] else 0.0)
    for kind in KINDS:
        mask = (kinds == kind).to_numpy()
        report[f"recall_{kind}"] = float(flags[mask].mean()) if mask.any() else float("nan")
        report[f"n_{kind}"] = int(mask.sum())
    if scores is not None:
        report["average_precision"] = float(average_precision_score(positive, scores))
    return report


def fit_forest(features: pd.DataFrame) -> IsolationForest:
    return IsolationForest(**FOREST).fit(features[FEATURES].to_numpy(dtype=float))


def evaluate(transactions: pd.DataFrame, config: dict) -> dict:
    a = config["anomaly"]
    dates = pd.to_datetime(transactions["date"])
    first = dates.min().to_period("M")
    val_start = (first + a["train_months"]).to_timestamp()
    test_start = (first + a["train_months"] + a["validation_months"]).to_timestamp()
    end = dates.max()
    seed = config["project"]["seed"]

    planted = plant_anomalies(transactions, val_start, val_start + pd.offsets.MonthEnd(a["validation_months"]),
                              a["planted_duplicates"], a["planted_price_rises"], seed)
    planted = plant_anomalies(planted.drop(columns="planted").assign(_p=planted["planted"]), test_start, end,
                              a["planted_duplicates"], a["planted_price_rises"], seed + 1)
    planted["planted"] = planted["planted"].where(planted["planted"] != "", planted.pop("_p").fillna(""))
    features = build_anomaly_features(planted)
    features["kind"] = label(features)

    train = features[features["date"] < val_start]
    val = features[(features["date"] >= val_start) & (features["date"] < test_start)]
    test = features[features["date"] >= test_start]
    forest = fit_forest(train[train["kind"].isin(["", "shock"])])         # nothing planted in training
    train_scores = -forest.score_samples(train[FEATURES].to_numpy(dtype=float))

    def flags_for(part, contamination):
        threshold = float(np.quantile(train_scores, 1 - contamination))
        part_scores = -forest.score_samples(part[FEATURES].to_numpy(dtype=float))
        return part_scores >= threshold, part_scores

    grid = []
    for c in a["contamination_grid"]:
        flags, s = flags_for(val, c)
        grid.append({"contamination": c, **scores_report(val["kind"], flags, s)})
    grid = pd.DataFrame(grid)
    best = float(grid.loc[grid["f1"].idxmax(), "contamination"])
    rule_validation = scores_report(val["kind"], rule_flags(val))
    method = "rule" if rule_validation["f1"] > grid["f1"].max() else "forest"

    flags, s = flags_for(test, best)
    test_report = scores_report(test["kind"], flags, s)
    rule_report = scores_report(test["kind"], rule_flags(test))
    test = test.assign(flagged=flags, score=s)
    return {"contamination": best, "grid": grid, "test": test_report, "rule": rule_report, "method": method,
            "validation_f1": {"forest": float(grid["f1"].max()), "rule": rule_validation["f1"]},
            "periods": {"train": f"{first} to {(val_start - pd.Timedelta(days=1)):%Y-%m}",
                        "validation": f"{val_start:%Y-%m} to {(test_start - pd.Timedelta(days=1)):%Y-%m}",
                        "test": f"{test_start:%Y-%m} to {end:%Y-%m}"},
            "test_rows": test}


def train_final(transactions: pd.DataFrame, contamination: float, method: str = "forest",
                version: int | None = None) -> AnomalyDetector:
    """The exported detector: fitted on all (unplanted) expenses, threshold at the chosen flag rate."""
    features = build_anomaly_features(transactions)
    forest = fit_forest(features)
    scores = -forest.score_samples(features[FEATURES].to_numpy(dtype=float))
    return AnomalyDetector(model=forest, threshold=float(np.quantile(scores, 1 - contamination)),
                           features=list(FEATURES), contamination=contamination, method=method, version=version)


def export(detector: AnomalyDetector, results: dict, out_dir: Path | None = None) -> tuple[Path, Path]:
    config = load_config()
    out_dir = out_dir or resolve_path(config["paths"]["models"])
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / config["anomaly"]["meta_file"]
    version = json.loads(meta_path.read_text(encoding="utf-8")).get("version", 0) + 1 if meta_path.exists() else 1
    detector.version = version
    model_path = out_dir / config["anomaly"]["model_file"]
    joblib.dump(detector, model_path, compress=3)
    meta = {
        "version": version, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "IsolationForest", "params": FOREST, "features": detector.features,
        "contamination": detector.contamination, "threshold": round(detector.threshold, 6),
        "flag_method": detector.method,
        "flag_method_reason": (f"validation F1: rule {results['validation_f1']['rule']:.3f} vs forest "
                               f"{results['validation_f1']['forest']:.3f}"),
        "periods": results["periods"],
        "test": {k: round(v, 4) if isinstance(v, float) else v for k, v in results["test"].items()},
        "rule_baseline": {k: round(v, 4) if isinstance(v, float) else v for k, v in results["rule"].items()},
        "libraries": {"scikit-learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        "file": model_path.name, "size_kb": round(model_path.stat().st_size / 1024, 1),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return model_path, meta_path


def log_to_mlflow(results: dict, model_path: Path, meta_path: Path) -> None:
    setup_experiment("anomaly")
    with mlflow.start_run(run_name="isolation_forest"):
        mlflow.log_params({**FOREST, "features": len(FEATURES), "contamination": results["contamination"],
                           "flag_method": results["method"],
                           **{f"period_{k}": v for k, v in results["periods"].items()}})
        mlflow.log_metrics({f"test_{k}": v for k, v in results["test"].items() if np.isfinite(v)})
        mlflow.log_metrics({f"rule_{k}": v for k, v in results["rule"].items() if np.isfinite(v)})
        for row in results["grid"].itertuples():
            mlflow.log_metric("validation_f1", row.f1, step=int(round(row.contamination * 1000)))
        mlflow.log_artifact(str(meta_path))
        mlflow.log_artifact(str(model_path))


def _print(results: dict) -> None:
    p = results["periods"]
    log.info(f"train {p['train']} · choose flag rate {p['validation']} · test {p['test']}\n")
    log.info("Validation (choosing the flag rate):")
    log.info(results["grid"][["contamination", "flag_rate", "precision", "recall", "f1"]].to_string(
           index=False, formatters={c: "{:.1%}".format for c in ("contamination", "flag_rate", "precision", "recall", "f1")}))
    rows = []
    for name, r in (("isolation forest", results["test"]), ("simple rule", results["rule"])):
        rows.append({"model": name, "flagged": f"{r['flag_rate']:.2%}", "precision": f"{r['precision']:.1%}",
                     "recall": f"{r['recall']:.1%}", **{k: f"{r[f'recall_{k}']:.1%}" for k in KINDS},
                     "false alarms": f"{r['false_alarm_rate']:.2%}"})
    t = results["test"]
    log.info(f"\nTest (flag rate {results['contamination']:.1%}; known anomalies: {t['n_shock']} shocks, "
             f"{t['n_duplicate']} duplicates, {t['n_price_rise']} price rises; average precision {t['average_precision']:.3f}):")
    log.info(pd.DataFrame(rows).to_string(index=False))
    v = results["validation_f1"]
    log.info(f"\nFlags in the exported detector: {results['method']} (validation F1 rule {v['rule']:.1%} "
             f"vs forest {v['forest']:.1%})")


def main() -> None:
    config = load_config()
    transactions = pd.read_csv(resolve_path(config["paths"]["data_raw"]) / config["synthetic"]["transactions_file"])
    results = evaluate(transactions, config)
    _print(results)
    detector = train_final(transactions, results["contamination"], results["method"])
    model_path, meta_path = export(detector, results)
    log_to_mlflow(results, model_path, meta_path)
    log.info(f"\nSaved {display_path(model_path)} (version {detector.version}) and {display_path(meta_path)}")


if __name__ == "__main__":
    main()
