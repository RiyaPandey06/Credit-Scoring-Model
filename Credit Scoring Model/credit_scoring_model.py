"""Train and evaluate a transparent credit-risk classification pipeline.

The included data generator is intentionally for demonstration only.  Replace it
with a consented, representative, historically labelled production dataset before
using this project to make any real lending decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42
TARGET = "defaulted"
NUMERIC_FEATURES = [
    "annual_income",
    "total_debt",
    "payment_history_score",
    "late_payments_12m",
    "credit_utilization",
    "credit_history_years",
    "open_credit_accounts",
    "employment_years",
    "loan_amount",
    "debt_to_income",
    "loan_to_income",
    "income_per_account",
]
CATEGORICAL_FEATURES = ["employment_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def make_demo_data(rows: int = 3000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Create a reproducible synthetic, financially plausible training set."""
    rng = np.random.default_rng(seed)
    income = np.clip(rng.lognormal(np.log(62_000), 0.55, rows), 15_000, 300_000)
    accounts = rng.integers(1, 13, rows)
    utilization = np.clip(rng.beta(2.1, 2.8, rows), 0.01, 0.99)
    debt = np.clip(income * rng.uniform(0.05, 0.85, rows), 500, 250_000)
    loan = np.clip(income * rng.uniform(0.03, 0.65, rows), 1_000, 150_000)
    history = np.clip(rng.gamma(3, 2.8, rows), 0.2, 35)
    late = rng.poisson(1.2, rows)
    payment = np.clip(98 - late * 8 - utilization * 18 + rng.normal(0, 5, rows), 20, 100)
    employment = rng.choice(
        ["salaried", "self_employed", "contract", "unemployed"],
        p=[0.56, 0.20, 0.17, 0.07], size=rows,
    )
    employment_years = np.clip(history * rng.uniform(0.25, 0.9, rows), 0, 25)
    debt_to_income = debt / income
    loan_to_income = loan / income
    # Latent default probability: a target label based on payment behaviour and affordability.
    log_odds = (
        -2.1 + 3.0 * utilization + 2.0 * debt_to_income + 1.4 * loan_to_income
        + 0.22 * late - 0.055 * (payment - 75) - 0.045 * history
        + 0.35 * (employment == "contract") + 1.0 * (employment == "unemployed")
        + rng.normal(0, 0.45, rows)
    )
    probability = 1 / (1 + np.exp(-log_odds))
    defaulted = rng.binomial(1, probability)
    return pd.DataFrame({
        "annual_income": income.round(2), "total_debt": debt.round(2),
        "payment_history_score": payment.round(2), "late_payments_12m": late,
        "credit_utilization": utilization.round(4), "credit_history_years": history.round(2),
        "open_credit_accounts": accounts, "employment_years": employment_years.round(2),
        "loan_amount": loan.round(2), "employment_type": employment,
        TARGET: defaulted,
    })


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build affordability features while avoiding divide-by-zero values."""
    data = frame.copy()
    income = data["annual_income"].clip(lower=1)
    data["debt_to_income"] = data["total_debt"] / income
    data["loan_to_income"] = data["loan_amount"] / income
    data["income_per_account"] = income / data["open_credit_accounts"].clip(lower=1)
    return data


def build_pipeline(model_name: str) -> Pipeline:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])
    if model_name == "logistic":
        estimator = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
    elif model_name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=350, min_samples_leaf=4, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        )
    else:
        raise ValueError("model_name must be 'logistic' or 'random_forest'")
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def score_model(name: str, pipeline: Pipeline, x_train, x_test, y_train, y_test) -> tuple[Pipeline, dict]:
    pipeline.fit(x_train, y_train)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average="binary", zero_division=0)
    return pipeline, {
        "model": name,
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(y_test, predictions, digits=4, zero_division=0),
    }


def train(data_path: str | None, artifacts_dir: Path) -> dict:
    raw = pd.read_csv(data_path) if data_path else make_demo_data()
    if TARGET not in raw.columns:
        raise ValueError(f"Input CSV must include a '{TARGET}' column (1 = default, 0 = non-default).")
    data = add_engineered_features(raw)
    x_train, x_test, y_train, y_test = train_test_split(
        data[FEATURES], data[TARGET], test_size=0.25, stratify=data[TARGET], random_state=RANDOM_STATE
    )
    results = []
    trained = {}
    for name in ("logistic", "random_forest"):
        trained[name], result = score_model(name, build_pipeline(name), x_train, x_test, y_train, y_test)
        results.append(result)
    best = max(results, key=lambda item: item["roc_auc"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(trained[best["model"]], artifacts_dir / "credit_risk_model.joblib")
    with (artifacts_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump({"best_model": best["model"], "test_rows": len(x_test), "results": results}, file, indent=2)
    if data_path is None:
        Path("data").mkdir(exist_ok=True)
        raw.to_csv("data/credit_data.csv", index=False)
    return {"best_model": best["model"], "results": results, "artifact": str(artifacts_dir / "credit_risk_model.joblib")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a credit-risk classifier.")
    parser.add_argument("--data", help="Optional labelled CSV. Required target column: defaulted.")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Directory for saved model and metrics.")
    args = parser.parse_args()
    outcome = train(args.data, Path(args.artifacts_dir))
    print(f"Best model: {outcome['best_model']}")
    for result in outcome["results"]:
        print("{model}: precision={precision}, recall={recall}, f1={f1_score}, roc_auc={roc_auc}".format(**result))
    print(f"Saved model: {outcome['artifact']}")


if __name__ == "__main__":
    main()
