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
(Step 5.5 adds MLflow logging and saves the model to models/categorizer.joblib.)
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.ingestion.bank_csv_parser import parse_bank_csv
from src.preprocessing.categorizer import categorize, merchant_key, model_inputs
from src.utils.helpers import load_config, resolve_path

FEATURES = ["text", "direction"]
SAMPLE_LABELS = "sample_labels.csv"


def build_model(seed: int = 42) -> Pipeline:
    features = ColumnTransformer([
        ("text", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True), "text"),
        ("direction", OneHotEncoder(handle_unknown="ignore"), ["direction"]),
    ])
    classifier = LogisticRegression(max_iter=3000, C=10.0, class_weight="balanced", random_state=seed)
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


def evaluate_pipeline_on_users(transactions: pd.DataFrame, seed: int) -> tuple[dict, dict]:
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
    return ({**_scores(test["category"], predicted["category"]), "by_source": by_source},
            dict(zip(sorted(test["category"].unique()), np.round(per_category, 3))))


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
    pipeline_scores, f1_per_category = evaluate_pipeline_on_users(transactions, seed)
    final_model = build_model(seed).fit(*_xy(examples))
    metrics = {
        "distinct_examples": len(examples),
        "A_new_spellings": evaluate_new_spellings(examples, seed),
        "B_unseen_merchants": evaluate_unseen_merchants(examples, seed),
        "C_rules_plus_model_heldout_users": pipeline_scores,
        "C_f1_per_category": f1_per_category,
        "sample_file": evaluate_sample_file(final_model),
    }
    return final_model, metrics


def main() -> None:
    config = load_config()
    path = resolve_path(config["paths"]["data_raw"]) / config["synthetic"]["transactions_file"]
    transactions = pd.read_csv(path)
    model, metrics = train_categorizer(transactions, seed=config["project"]["seed"])
    print(f"Distinct training examples: {metrics['distinct_examples']}")
    for key in ("A_new_spellings", "B_unseen_merchants", "C_rules_plus_model_heldout_users", "sample_file"):
        print(f"{key}: {metrics[key]}")
    print("F1 per category (C):", metrics["C_f1_per_category"])


if __name__ == "__main__":
    main()
