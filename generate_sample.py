#!/usr/bin/env python3
import os
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

N_TAXPAYERS = 200
YEARS = [2022, 2023, 2024]
RECORDS_PER_TAXPAYER = 3  # one per year


def make_title(idx: int) -> str:
    return f"Sample Film {idx:04d}"


def main(out_path: str = "data/returns_sample.csv"):
    rows = []
    taxpayer_ids = [f"TP{100000+i}" for i in range(N_TAXPAYERS)]

    idx = 0
    for tp in taxpayer_ids:
        base_income = np.random.lognormal(mean=11.0, sigma=0.5)  # around ~60k-200k
        base_deductions_rate = np.clip(np.random.normal(0.1, 0.05), 0, 0.5)
        employer_salary_factor = np.clip(np.random.normal(1.0, 0.05), 0.8, 1.2)
        for y in YEARS:
            idx += 1
            # income with occasional spikes
            spike_flag = (np.random.rand() < 0.12)
            income = base_income * np.random.uniform(0.85, 1.15)
            if spike_flag:
                income *= np.random.uniform(1.6, 2.5)
            # tax paid roughly progressive
            tax_rate = np.clip(np.random.normal(0.18, 0.03), 0.05, 0.4)
            tax_paid = income * tax_rate
            # deductions
            ded_rate = np.clip(np.random.normal(base_deductions_rate, 0.05), 0.0, 0.8)
            deductions = income * ded_rate
            # salaries
            salary_employer = income * employer_salary_factor * 0.7
            mismatch_flag = (np.random.rand() < 0.1)
            salary_reported = salary_employer * (np.random.uniform(0.7, 1.3) if mismatch_flag else np.random.uniform(0.95, 1.05))
            # filing
            period_end_date = datetime(y, 3, 31)
            delay_days = int(np.random.normal(45, 30))
            if np.random.rand() < 0.1:
                delay_days += int(np.random.uniform(120, 300))
            filed_date = period_end_date + timedelta(days=max(delay_days, -100))
            amendment_count = int(np.random.rand() < 0.08)
            # tickets_sold/title to enable trending plot (proxy from income)
            tickets_sold = int(max(0, np.random.normal(income / 1000.0, 500)))
            title = make_title(idx)

            # construct red-flag features for label
            tax_to_income_ratio = tax_paid / max(income, 1e-6)
            deduct_ratio = deductions / max(income, 1e-6)
            salary_rel_diff = abs(salary_reported - salary_employer) / max(abs(salary_employer), 1e-6)
            is_fraud = 0
            risk_score = 0
            risk_score += (tax_to_income_ratio < 0.08) * 1
            risk_score += (deduct_ratio > 0.35) * 1
            risk_score += spike_flag * 1
            risk_score += (salary_rel_diff > 0.2) * 1
            risk_score += (delay_days > 120) * 1
            risk_score += (amendment_count > 0) * 1
            # convert to probability and sample
            p = min(0.05 + 0.15 * risk_score, 0.95)
            is_fraud = int(np.random.rand() < p)

            rows.append({
                'taxpayer_id': tp,
                'period_end_date': period_end_date.strftime('%Y-%m-%d'),
                'income': round(income, 2),
                'tax_paid': round(tax_paid, 2),
                'deductions': round(deductions, 2),
                'salary_reported': round(salary_reported, 2),
                'salary_employer': round(salary_employer, 2),
                'filed_date': filed_date.strftime('%Y-%m-%d'),
                'amendment_count': amendment_count,
                'is_fraud': is_fraud,
                'title': title,
                'tickets_sold': tickets_sold,
            })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == '__main__':
    main()
