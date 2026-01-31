#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
import joblib

from .features import engineer_features


def run_predict(model_path: str, input_path: str, output_path: str):
    pipe = joblib.load(model_path)
    df = pd.read_csv(input_path)
    X_df, _, _ = engineer_features(df, require_label=False)
    proba = pipe.predict_proba(X_df.values.astype(np.float32))[:, 1]
    out = df.copy()
    out['fraud_proba'] = proba
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Wrote predictions to {output_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Predict fraud probability for tax returns")
    p.add_argument('--model-path', required=True, help='Path to a saved model .joblib')
    p.add_argument('--input', required=True, help='Input CSV path (unlabeled)')
    p.add_argument('--output', required=True, help='Output CSV path')
    return p.parse_args()


def main():
    args = parse_args()
    run_predict(args.model_path, args.input, args.output)


if __name__ == '__main__':
    main()
