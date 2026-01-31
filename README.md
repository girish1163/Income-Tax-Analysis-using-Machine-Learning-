# Suspicious Tax Return Classification

Build a binary classifier to detect fraudulent tax returns using engineered signals and multiple models (Logistic Regression, Random Forest, XGBoost, LightGBM).

## Features engineered
- Income vs Tax Paid ratio
- Sudden spikes in declared income (YoY change and Z-score)
- Unusually high deductions (relative to income and peer cohort)
- Mismatched salary vs employer reports
- Filing delays and amendments

## Project layout
- `requirements.txt` – dependencies
- `.env.example` – environment variables template
- `src/`
  - `data_api.py` – fetch from Income Tax API or CSV (temporary)
  - `features.py` – feature engineering
  - `models.py` – model definitions and training utilities
  - `evaluate.py` – metrics and evaluation
  - `plots.py` – required graphical representations
  - `train.py` – CLI: load → features → train → evaluate → plots
  - `predict.py` – CLI: load model → infer on new data
- `data/` – local cache and input CSVs (you provide)
- `models/` – saved trained models
- `reports/` – generated plots

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment
Copy and edit .env.example as .env:
```
INCOME_TAX_API_KEY=your_key_here
API_BASE_URL=https://api.example.gov
```

## Quick start (CSV-based until API wired)
Prepare a CSV with at least these columns:
- `taxpayer_id`
- `period_end_date` (YYYY-MM-DD)
- `income`
- `tax_paid`
- `deductions`
- `salary_reported`
- `salary_employer`
- `filed_date` (YYYY-MM-DD)
- `amendment_count` (integer)
- `is_fraud` (0/1)

Run training (all models):
```bash
python -m src.train --input data/returns.csv --models all
```
Outputs:
- Saves models under `models/`
- Writes metrics to console
- Plots to `reports/`

Predict on new data:
```bash
python -m src.predict --model-path models/logreg.joblib --input data/new_returns.csv --output predictions.csv
```

## Notes
- Class imbalance is common; Logistic Regression uses class_weight='balanced'.
- For API mode, set env vars and we will wire `data_api.py` to fetch from your service securely.
