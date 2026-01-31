from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple, List

REQUIRED_COLUMNS = [
    'taxpayer_id','period_end_date','income','tax_paid','deductions',
    'salary_reported','salary_employer','filed_date','amendment_count'
]


def ensure_columns(df: pd.DataFrame, require_label: bool = True) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if require_label and 'is_fraud' not in df.columns:
        missing.append('is_fraud')
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['period_end_date'] = pd.to_datetime(df['period_end_date'])
    df['filed_date'] = pd.to_datetime(df['filed_date'])
    df['period_year'] = df['period_end_date'].dt.year
    return df


def engineer_features(df: pd.DataFrame, require_label: bool = True) -> Tuple[pd.DataFrame, pd.Series | None, List[str]]:
    """Create features specified in the project description.
    Returns (X, y, feature_names)
    """
    ensure_columns(df, require_label=require_label)
    df = parse_dates(df)

    X = pd.DataFrame(index=df.index)

    # 1) Income vs tax paid ratio
    X['tax_to_income_ratio'] = (df['tax_paid'] / (df['income'].replace(0, np.nan))).fillna(0.0).clip(0, 10)

    # 2) Sudden spikes in declared income: YoY change and z-score within taxpayer
    df_sorted = df.sort_values(['taxpayer_id', 'period_end_date'])
    income_prev = df_sorted.groupby('taxpayer_id')['income'].shift(1)
    yoy_change = (df_sorted['income'] - income_prev) / income_prev.replace(0, np.nan)
    df_sorted['income_yoy_change'] = yoy_change.fillna(0.0).clip(-10, 10)
    # Z-score within taxpayer
    def zscore(s: pd.Series) -> pd.Series:
        m = s.mean()
        sd = s.std(ddof=0)
        if sd == 0 or np.isnan(sd):
            return pd.Series(0.0, index=s.index)
        return (s - m) / sd
    df_sorted['income_z'] = df_sorted.groupby('taxpayer_id')['income'].transform(zscore).fillna(0.0).clip(-10, 10)
    # restore original order
    X['income_yoy_change'] = df_sorted['income_yoy_change'].reindex(df.index).fillna(0.0)
    X['income_z'] = df_sorted['income_z'].reindex(df.index).fillna(0.0)

    # 3) Unusually high deductions: ratio and peer percentile (peer by period_year)
    X['deduct_to_income_ratio'] = (df['deductions'] / (df['income'].replace(0, np.nan))).fillna(0.0).clip(0, 10)
    # peer percentile by year
    def pct_rank(g):
        return g.rank(pct=True)
    peer_rank = df.groupby('period_year')['deductions'].transform(pct_rank).fillna(0.0)
    X['deduct_peer_percentile'] = peer_rank

    # 4) Mismatched salary vs employer reports: absolute and relative diff
    diff = (df['salary_reported'] - df['salary_employer']).abs()
    X['salary_abs_diff'] = diff
    X['salary_rel_diff'] = (diff / (df['salary_employer'].abs() + 1e-6)).clip(0, 10)

    # 5) Filing delays and amendments
    delays = (df['filed_date'] - df['period_end_date']).dt.days
    X['filing_delay_days'] = delays.clip(lower=-3650, upper=3650).fillna(0)
    X['has_amendment'] = (df['amendment_count'] > 0).astype(int)
    X['amendment_count'] = df['amendment_count'].fillna(0).clip(0, 100)

    # Basic stabilizers
    X = X.replace([np.inf, -np.inf], 0).fillna(0)

    y = None
    if require_label:
        y = df['is_fraud'].astype(int)

    return X, y, list(X.columns)
