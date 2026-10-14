"""Train the text classifier that categorises transactions the keyword rules miss.

Model: TF-IDF on character 2-5-grams of the normalised description + the money direction
(in/out), fed to a class-balanced logistic regression.

The synthetic data repeats the same descriptions many times, so the model is trained on
DISTINCT (description, direction, category) rows, and evaluated three ways:
  A. new spellings of known merchants   (random split of distinct descriptions)
  B. completely unseen merchants        (split by merchant, the hard case)
  C. rules + model on held-out users    (row level, the number the blueprint targets: >= 90%)
plus the hand-labelled sample bank export as an independent check.

Run:  python -m src.training.train_categorizer
Saves models/categorizer.joblib (+ categorizer_meta.json) and logs the run to the MLflow
experiment "categorizer" (view with: mlflow ui --backend-store-uri sqlite:///mlflow.db).
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
import mlflow
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.ingestion.bank_csv_parser import parse_bank_csv
from src.preprocessing.categorizer import categorize, categorizer_path, merchant_key, model_inputs
from src.utils.helpers import load_config, mlflow_tracking_uri, resolve_path

matplotlib.use("Agg")   # draw figures without a screen

FEATURES = ["text", "direction"]
MODEL_PARAMS = {"analyzer": "char_wb", "ngram_range": "2-5", "C": 10.0, "class_weight": "balanced",
                "classifier": "LogisticRegression"}
SAMPLE_LABELS = "sample_labels.csv"


def build_model(seed: int = 42) -> Pipeline:
    features = ColumnTransformer([
        ("text", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True), "text"),
        ("direction", OneHotEncoder(handle_unknown="ignore"), ["direction"]),
    ])
    classifier = LogisticRegression(max_iter=3000, C=MODEL_PARAMS["C"], class_weight="balanced", random_state=seed)
    return Pipeline([("features", features), ("classifier", classifier)])


def to_features(df: pd.DataFrame) -> pd.DataFrame:
    """description + amount -> the two model inputs."""
    return model_inputs(df)


def distinct_examples(transactions: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct (text, direction, category), with how often it occurs."""
    examples = to_features(transactions).assign(category=transactions["category"].values)
    return (examples.groupby(FEATURES + ["category"], as_index=False).size()
            .rename(columns={"size": "count"}))


def _scores(truth: pd.Series, predicted) -> dict:
    return {"accuracy": round(float(accuracy_score(truth, predicted)), 4),
            "macro_f1": round(float(f1_score(truth, predicted, average="macro", zero_division=0)), 4),
            "n": int(len(truth))}


def evaluate_new_spellings(examples: pd.DataFrame, seed: int) -> dict:
    """A: random split of distinct descriptions (merchants are mostly known, spellings are new)."""
    train, test = train_test_split(examples, test_size=0.25, random_state=seed, stratify=examples["category"])
    model = build_model(seed).fit(train[FEATURES], train["category"])
    return _scores(test["category"], model.predict(test[FEATURES]))


def evaluate_unseen_merchants(examples: pd.DataFrame, seed: int, repeats: int = 20,
                              thresholds=(0.3, 0.4, 0.5, 0.6, 0.7)) -> dict:
    """B: whole merchants held out, repeated over several random splits (the result varies a lot).

    Also shows, per confidence threshold, how many guesses the model would keep and how
    accurate those kept guesses are - this is how the categorizer threshold is chosen.
    """
    groups = merchant_key(examples["text"])
    splitter = GroupShuffleSplit(n_splits=repeats, test_size=0.2, random_state=seed)
    runs, confidence, correct = [], [], []
    for train_idx, test_idx in splitter.split(examples, groups=groups):
        train, test = examples.iloc[train_idx], examples.iloc[test_idx]
        model = build_model(seed).fit(train[FEATURES], train["category"])
        probabilities = model.predict_proba(test[FEATURES])
        predicted = model.classes_[probabilities.argmax(axis=1)]
        runs.append(_scores(test["category"], predicted))
        confidence += list(probabilities.max(axis=1))
        correct += list(predicted == test["category"].values)
    confidence, correct = np.array(confidence), np.array(correct)
    by_threshold = {
        t: {"kept": round(float((confidence >= t).mean()), 3),
            "accuracy_of_kept": round(float(correct[confidence >= t].mean()), 3) if (confidence >= t).any() else None}
        for t in thresholds
    }
    return {"accuracy": round(float(np.mean([r["accuracy"] for r in runs])), 4),
            "accuracy_min": min(r["accuracy"] for r in runs),
            "accuracy_max": max(r["accuracy"] for r in runs),
            "macro_f1": round(float(np.mean([r["macro_f1"] for r in runs])), 4),
            "n": int(np.mean([r["n"] for r in runs])), "splits": repeats,
            "by_threshold": by_threshold}


def predict_with_rules(model: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """Rules + model + confidence threshold: exactly what the app uses (categorize)."""
    return categorize(df, model)


def evaluate_pipeline_on_users(transactions: pd.DataFrame, seed: int) -> tuple[dict, dict, pd.DataFrame]:
    """C: train on 80% of users, score rules + model on every transaction of the other 20%."""
    users = transactions["user_id"].unique()
    rng = np.random.default_rng(seed)
    test_users = set(rng.choice(users, size=len(users) // 5, replace=False))
    is_test = transactions["user_id"].isin(test_users)
    model = build_model(seed).fit(*_xy(distinct_examples(transactions[~is_test])))
    test = transactions[is_test]
    predicted = predict_with_rules(model, test)
    by_source = {source: _scores(test["category"][part.index], part["category"])
                 for source, part in predicted.groupby("category_source")}
    by_source["sent_to_review"] = int(predicted["needs_review"].sum())
    per_category = f1_score(test["category"], predicted["category"], average=None,
                            labels=sorted(test["category"].unique()), zero_division=0)
    labels = sorted(test["category"].unique())
    return ({**_scores(test["category"], predicted["category"]), "by_source": by_source},
            {label: float(v) for label, v in zip(labels, np.round(per_category, 3))},
            pd.DataFrame({"truth": test["category"].values, "predicted": predicted["category"].values}))


def evaluate_sample_file(model: Pipeline) -> dict | None:
    """Rules + model on the 60-row sample bank export, labelled by hand."""
    sample_dir = resolve_path(load_config()["paths"]["data_sample"])
    labels_path = sample_dir / SAMPLE_LABELS
    if not labels_path.exists():
        return None
    sample = parse_bank_csv(sample_dir / "sample_bank_export.csv")
    labels = pd.read_csv(labels_path)
    sample = sample.merge(labels, on="description", how="left")
    predicted = predict_with_rules(model, sample)
    wrong = sample.assign(predicted=predicted["category"])
    wrong = wrong[wrong["predicted"] != wrong["category"]][["description", "category", "predicted"]]
    return {**_scores(sample["category"], predicted["category"]), "mistakes": wrong.to_dict("records")}


def _xy(examples: pd.DataFrame):
    return examples[FEATURES], examples["category"]


def train_categorizer(transactions: pd.DataFrame, seed: int = 42) -> tuple[Pipeline, dict]:
    """Evaluate (A, B, C, sample) and fit the final model on all distinct examples."""
    examples = distinct_examples(transactions)
    pipeline_scores, f1_per_category, predictions = evaluate_pipeline_on_users(transactions, seed)
    final_model = build_model(seed).fit(*_xy(examples))
    metrics = {
        "distinct_examples": len(examples),
        "A_new_spellings": evaluate_new_spellings(examples, seed),
        "B_unseen_merchants": evaluate_unseen_merchants(examples, seed),
        "C_rules_plus_model_heldout_users": pipeline_scores,
        "C_f1_per_category": f1_per_category,
        "sample_file": evaluate_sample_file(final_model),
        "C_predictions": predictions,
    }
    return final_model, metrics


def save_model(model: Pipeline, metrics: dict, threshold: float, path: Path | None = None) -> tuple[Path, Path]:
    """Write the model (joblib) and a small JSON with what it is and how good it was."""
    path = Path(path) if path else categorizer_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)
    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,     # the app must load it with the same version
        "params": MODEL_PARAMS,
        "confidence_threshold": threshold,
        "classes": [str(c) for c in model.classes_],
        "distinct_examples": metrics["distinct_examples"],
        "metrics": {
            "rules_plus_model_heldout_users_accuracy": metrics["C_rules_plus_model_heldout_users"]["accuracy"],
            "unseen_merchants_accuracy": metrics["B_unseen_merchants"]["accuracy"],
            "unseen_merchants_kept_at_threshold": metrics["B_unseen_merchants"]["by_threshold"].get(threshold),
            "sample_file_accuracy": (metrics["sample_file"] or {}).get("accuracy"),
        },
    }
    meta_path = path.with_name(load_config()["categorizer"]["meta_file"])
    meta_path.write_text(json.dumps(meta, indent=2))
    return path, meta_path


def confusion_figure(predictions: pd.DataFrame):
    """Confusion matrix of rules + model on held-out users (row-normalised, one hue)."""
    import matplotlib.pyplot as plt

    labels = sorted(set(predictions["truth"]) | set(predictions["predicted"]))
    matrix = confusion_matrix(predictions["truth"], predictions["predicted"], labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted"); ax.set_ylabel("true category")
    for (i, j), v in np.ndenumerate(matrix):
        if v >= 0.005:
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7, color="white" if v > 0.5 else "#0b0b0b")
    ax.set_title("Categoriser on held-out users (rules + model)", loc="left")
    fig.tight_layout()
    return fig


def log_to_mlflow(metrics: dict, threshold: float, model_path: Path, meta_path: Path) -> str:
    """One run in the 'categorizer' experiment: parameters, metrics, plots and the model files."""
    config = load_config()
    mlflow.set_tracking_uri(mlflow_tracking_uri())
    name = config["mlflow"]["experiments"]["categorizer"]
    if mlflow.get_experiment_by_name(name) is None:
        artifacts = resolve_path("mlruns") / name
        mlflow.create_experiment(name, artifact_location=artifacts.as_uri())
    mlflow.set_experiment(name)

    flat = {
        "rules_plus_model_accuracy": metrics["C_rules_plus_model_heldout_users"]["accuracy"],
        "rules_plus_model_macro_f1": metrics["C_rules_plus_model_heldout_users"]["macro_f1"],
        "new_spellings_accuracy": metrics["A_new_spellings"]["accuracy"],
        "unseen_merchants_accuracy": metrics["B_unseen_merchants"]["accuracy"],
        "unseen_merchants_macro_f1": metrics["B_unseen_merchants"]["macro_f1"],
        "sent_to_review_heldout_users": metrics["C_rules_plus_model_heldout_users"]["by_source"]["sent_to_review"],
    }
    if metrics["sample_file"]:
        flat["sample_file_accuracy"] = metrics["sample_file"]["accuracy"]
    for t, row in metrics["B_unseen_merchants"]["by_threshold"].items():
        flat[f"unseen_kept_at_{t}"] = row["kept"]
        if row["accuracy_of_kept"] is not None:
            flat[f"unseen_accuracy_of_kept_at_{t}"] = row["accuracy_of_kept"]
    for category, f1 in metrics["C_f1_per_category"].items():
        flat[f"f1_{category}"] = f1

    with mlflow.start_run(run_name="tfidf_char_logreg") as run:
        mlflow.log_params({**MODEL_PARAMS, "confidence_threshold": threshold,
                           "distinct_examples": metrics["distinct_examples"]})
        mlflow.log_metrics(flat)
        mlflow.log_figure(confusion_figure(metrics["C_predictions"]), "confusion_matrix.png")
        mlflow.log_dict({k: v for k, v in metrics.items() if k != "C_predictions"}, "metrics.json")
        mlflow.log_artifact(str(model_path), "model")
        mlflow.log_artifact(str(meta_path), "model")
        return run.info.run_id


def main() -> None:
    config = load_config()
    threshold = config["categorizer"]["confidence_threshold"]
    path = resolve_path(config["paths"]["data_raw"]) / config["synthetic"]["transactions_file"]
    transactions = pd.read_csv(path)
    model, metrics = train_categorizer(transactions, seed=config["project"]["seed"])

    print(f"Distinct training examples: {metrics['distinct_examples']}")
    for key in ("A_new_spellings", "B_unseen_merchants", "C_rules_plus_model_heldout_users", "sample_file"):
        print(f"{key}: {metrics[key]}")
    print("F1 per category (C):", metrics["C_f1_per_category"])

    model_path, meta_path = save_model(model, metrics, threshold)
    print(f"Saved {model_path.relative_to(resolve_path('.'))} ({model_path.stat().st_size / 1024:.0f} KB) "
          f"and {meta_path.name}")
    run_id = log_to_mlflow(metrics, threshold, model_path, meta_path)
    print(f"MLflow run {run_id} in experiment '{config['mlflow']['experiments']['categorizer']}'")


if __name__ == "__main__":
    main()
