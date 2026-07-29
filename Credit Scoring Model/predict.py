"""Score one applicant with the saved demonstration model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from credit_scoring_model import add_engineered_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict credit default risk for one applicant JSON record.")
    parser.add_argument("--input", required=True, help="Path to a JSON object containing applicant fields.")
    parser.add_argument("--model", default="artifacts/credit_risk_model.joblib")
    parser.add_argument("--threshold", type=float, default=0.50, help="Default-risk alert threshold (0-1).")
    args = parser.parse_args()
    applicant = json.loads(Path(args.input).read_text(encoding="utf-8"))
    model = joblib.load(args.model)
    risk = float(model.predict_proba(add_engineered_features(pd.DataFrame([applicant])))[:, 1][0])
    print(json.dumps({
        "default_risk_probability": round(risk, 4),
        "risk_band": "higher_risk" if risk >= args.threshold else "lower_risk",
        "threshold": args.threshold,
        "notice": "Demonstration output only; do not use as the sole basis for a lending decision.",
    }, indent=2))


if __name__ == "__main__":
    main()
