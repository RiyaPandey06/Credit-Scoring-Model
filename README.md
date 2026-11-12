# Credit Scoring Model 

A Machine Learning-based Credit Scoring Model. The project predicts the probability of loan default based on applicants' financial history and credit-related information.

## Project Overview

The model analyzes financial information to estimate the likelihood of a loan applicant defaulting on a loan. It uses feature engineering, data preprocessing, and classification algorithms to generate predictions and evaluate model performance.

The project compares Logistic Regression and Random Forest and selects the best-performing model based on ROC-AUC.

## Features

- Financial feature engineering:
  - Debt-to-Income Ratio
  - Loan-to-Income Ratio
  - Income per Account
- Missing-value imputation
- Numerical feature scaling
- Categorical feature encoding
- Logistic Regression and Random Forest comparison
- Model evaluation using Accuracy, Precision, Recall, F1-score, and ROC-AUC
- Model serialization using Joblib
- Command-line prediction using JSON input
- Support for custom labelled credit datasets

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib

## Project Structure

```text
Credit-Scoring-Model/
│
├── credit_scoring_model.py
├── predict.py
├── sample_applicant.json
├── requirements.txt
├── README.md
│
├── data/
│   └── credit_data.csv
│
└── artifacts/
    ├── credit_risk_model.joblib
    └── metrics.json
```

## Installation and Setup

**1. Clone the repository**

```bash
git clone https://github.com/your-username/Credit-Scoring-Model.git
cd Credit-Scoring-Model
```

**2. Create a virtual environment**

```bash
python -m venv .venv
```

**3. Activate the environment**

Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

**4. Install dependencies**

```bash
python -m pip install -r requirements.txt
```

## Run the Project

**Train the model:**

```bash
python credit_scoring_model.py
```

The training process generates:

- `data/credit_data.csv`
- `artifacts/credit_risk_model.joblib`
- `artifacts/metrics.json`

**Generate a prediction:**

```bash
python predict.py --input sample_applicant.json
```

Example output:

```json
{
  "default_risk_probability": 0.1429,
  "risk_band": "lower_risk",
  "threshold": 0.5,
  "notice": "Demonstration output only; do not use as the sole basis for a lending decision."
}
```

## Model Performance

Both models were evaluated on a test dataset containing **750 records**.

### Logistic Regression

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 67.73% |
| Precision | 70.44% |
| Recall | 70.10% |
| F1-Score | 70.27% |
| ROC-AUC | **75.33%** |

Confusion Matrix:

```text
[[222, 120],
 [122, 286]]
```

### Random Forest

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 66.40% |
| Precision | 69.12% |
| Recall | 69.12% |
| F1-Score | 69.12% |
| ROC-AUC | 73.11% |

Confusion Matrix:

```text
[[216, 126],
 [126, 282]]
```

### Model Comparison

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 67.73% | 66.40% |
| Precision | 70.44% | 69.12% |
| Recall | 70.10% | 69.12% |
| F1-Score | 70.27% | 69.12% |
| ROC-AUC | 75.33% | 73.11% |

**Best Model: Logistic Regression**

Logistic Regression achieved the highest ROC-AUC score of **0.7533**, outperforming Random Forest (0.7311). Therefore, Logistic Regression was selected and saved as the final model pipeline.

## Train on a Custom Dataset

The project also supports training on a real, permissioned labelled dataset.

```bash
python credit_scoring_model.py --data path/to/labelled_credit_data.csv
```

The CSV must contain the following columns:

- `annual_income`
- `total_debt`
- `payment_history_score`
- `late_payments_12m`
- `credit_utilization`
- `credit_history_years`
- `open_credit_accounts`
- `employment_years`
- `loan_amount`
- `employment_type`
- `defaulted`

The target column `defaulted` must contain binary values (0 or 1).

## Responsible Use

The default dataset is synthetic and intended only for educational and demonstration purposes. Model results should not be interpreted as evidence of real-world lending performance.

Production credit-scoring systems require consented and representative historical data, fairness and compliance assessments, probability calibration, drift monitoring, human oversight, and appropriate adverse-action procedures.

This model must never be used as the sole basis for lending decisions.

## Author

**Riya Pandey**

Machine Learning | Python | Data Science

**Project:** CodeAlpha Machine Learning Internship – Task 1

---

*Built for learning and experimentation with machine learning-based credit risk prediction.*
