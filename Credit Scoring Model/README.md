# Credit Scoring Model

A complete solution for CodeAlpha Machine Learning Task 1. It predicts the probability of loan default using financial-history data and reports precision, recall, F1-score, and ROC-AUC.

## Included

- Financial feature engineering: debt-to-income, loan-to-income, and income per account.
- Missing-value handling, numeric scaling, and categorical encoding.
- Logistic Regression and Random Forest comparison; the best ROC-AUC model is saved.
- Saved pipeline and metrics JSON, plus a command-line prediction script.

## Run

```powershell
python -m pip install -r requirements.txt
python credit_scoring_model.py
python predict.py --input sample_applicant.json
```

The default training run uses a deterministic synthetic demonstration dataset, then creates `data/credit_data.csv`, `artifacts/credit_risk_model.joblib`, and `artifacts/metrics.json`.

To train on a real, permissioned dataset:

```powershell
python credit_scoring_model.py --data path\to\labelled_credit_data.csv
```

The CSV must contain `annual_income`, `total_debt`, `payment_history_score`, `late_payments_12m`, `credit_utilization`, `credit_history_years`, `open_credit_accounts`, `employment_years`, `loan_amount`, `employment_type`, and `defaulted` (0/1).

## Responsible use

The bundled data is synthetic and educational only. Production credit models require consented, representative historical data, compliance and fairness review, calibration and drift testing, human oversight, and appropriate adverse-action procedures. They must never be the sole basis of a lending decision.
