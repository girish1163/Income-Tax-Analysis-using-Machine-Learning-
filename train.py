import argparse
import os
from typing import List

import numpy as np
import pandas as pd

from .features import engineer_features
from .models import get_models, TrainConfig, split_data, train_and_save
from .evaluate import classification_metrics
from .plots import (
    plot_income_tax_ratio,
    plot_income_spikes,
    plot_deductions,
    plot_salary_mismatch,
    plot_filing_delays,
    plot_trending_year,
    plot_roc_pr_curves,
    plot_confusion,
    plot_precision_recall_vs_threshold,
    plot_feature_importance,
)
from .data_api import fetch_full_url, normalize_records


def ensure_dirs(base: str):
    os.makedirs(os.path.join(base, 'models'), exist_ok=True)
    os.makedirs(os.path.join(base, 'reports'), exist_ok=True)


def run_training(input_path: str | None,
                 api_url: str | None,
                 models_to_run: List[str] | None,
                 out_base: str,
                 trending_title_col: str | None,
                 trending_tickets_col: str | None,
                 trending_year: int | None,
                 show: bool = False,
                 no_trending: bool = False):
    ensure_dirs(out_base)

    if (input_path is None) == (api_url is None):
        raise ValueError("Provide exactly one of --input (CSV) or --api-url (full request URL)")

    if input_path is not None:
        df = pd.read_csv(input_path)
    else:
        payload = fetch_full_url(api_url)
        # Expect a list of records or an object with a key holding records
        if isinstance(payload, dict):
            # try common keys
            for key in ('data', 'results', 'items', 'records'):
                if key in payload and isinstance(payload[key], list):
                    payload = payload[key]
                    break
        if not isinstance(payload, list):
            raise RuntimeError("API response format unsupported. Expected a list of records or an object with a list under 'data'/'results'/'items'/'records'.")
        df = normalize_records(payload)
    X_df, y, feat_names = engineer_features(df, require_label=True)
    X = X_df.values.astype(np.float32)
    y = y.values.astype(np.int32)

    cfg = TrainConfig()
    X_tr, X_val, y_tr, y_val = split_data(X, y, cfg)

    models = get_models()
    if models_to_run and 'all' not in models_to_run:
        models = {k: v for k, v in models.items() if k in models_to_run}
        if not models:
            raise ValueError("No valid models selected. Options: logreg, rf, xgb, lgbm, or 'all'")

    results = train_and_save(models, X_tr, y_tr, X_val, y_val, os.path.join(out_base, 'models'))

    # Reports directory
    reports_dir = os.path.join(out_base, 'reports')
    print("\nGenerating plots in reports/ ...")
    # Evaluate each model on validation set and generate eval plots
    print("\nValidation metrics:")
    for name, _, model_path in results:
        from joblib import load
        pipe = load(model_path)
        proba = pipe.predict_proba(X_val)[:, 1]
        metrics = classification_metrics(y_val, proba)
        print(f"- {name}: AUC={metrics['roc_auc']:.3f}, PR-AUC={metrics['pr_auc']:.3f}, F1={metrics['f1']:.3f}, P={metrics['precision']:.3f}, R={metrics['recall']:.3f}")
        # Evaluation plots per model
        y_pred = (proba >= 0.5).astype(int)
        plot_roc_pr_curves(y_val, proba, reports_dir, prefix=name, show=show)
        plot_confusion(y_val, y_pred, reports_dir, prefix=name, show=show)
        plot_precision_recall_vs_threshold(y_val, proba, reports_dir, prefix=name, show=show)
        plot_feature_importance(pipe, feat_names, reports_dir, prefix=name, show=show)
    # Plots of engineered signals
    print("\nGenerating plots in reports/ ...")
    plot_income_tax_ratio(X_df, reports_dir, show=show)
    plot_income_spikes(X_df, reports_dir, show=show)
    plot_deductions(X_df, reports_dir, show=show)
    plot_salary_mismatch(X_df, reports_dir, show=show)
    plot_filing_delays(X_df, reports_dir, show=show)

    # Trending plot (optional) — disabled if no_trending is True
    if not no_trending:
        title_col = trending_title_col or ('title' if 'title' in df.columns else None)
        tickets_col = trending_tickets_col or ('tickets_sold' if 'tickets_sold' in df.columns else None)
        if title_col and tickets_col and title_col in df.columns and tickets_col in df.columns:
            plot_trending_year(df, 'period_year', title_col, tickets_col, trending_year, reports_dir, show=show)

    print("Done. Models in models/, plots in reports/.")


def parse_args():
    p = argparse.ArgumentParser(description="Suspicious Tax Return Classification - Training")
    p.add_argument('--input', help='Input CSV path with labeled data')
    p.add_argument('--api-url', help='Full API request URL returning JSON array or object with data/results/items/records')
    p.add_argument('--models', nargs='+', default=['all'], help="Models to run: logreg rf xgb lgbm or 'all'")
    p.add_argument('--out', default='.', help='Project base folder (default .)')
    p.add_argument('--title-col', default=None, help='Title column for trending plot (optional)')
    p.add_argument('--tickets-col', default=None, help='Tickets/units column for trending plot (optional)')
    p.add_argument('--year', type=int, default=None, help='Year for trending plot (optional)')
    p.add_argument('--show', action='store_true', help='Pop up plot windows during training')
    p.add_argument('--no-trending', action='store_true', help='Do not generate the trending-by-year plot')
    return p.parse_args()


def main():
    args = parse_args()
    run_training(args.input, args.api_url, args.models, args.out, args.title_col, args.tickets_col, args.year, show=args.show, no_trending=args.no_trending)


if __name__ == '__main__':
    main()
