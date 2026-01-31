from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import joblib

# Optional imports for gradient boosting libraries
try:
    from xgboost import XGBClassifier  # type: ignore
except Exception:  # pragma: no cover
    XGBClassifier = None  # type: ignore

try:
    from lightgbm import LGBMClassifier  # type: ignore
except Exception:  # pragma: no cover
    LGBMClassifier = None  # type: ignore


@dataclass
class TrainConfig:
    test_size: float = 0.2
    random_state: int = 42


def split_data(X: np.ndarray, y: np.ndarray, cfg: TrainConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y)


def get_models() -> Dict[str, Pipeline]:
    models: Dict[str, Pipeline] = {}

    models['logreg'] = Pipeline([
        ('scaler', StandardScaler(with_mean=False)),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', n_jobs=None, solver='lbfgs')),
    ])

    models['rf'] = Pipeline([
        ('clf', RandomForestClassifier(n_estimators=300, max_depth=None, class_weight='balanced_subsample', random_state=42, n_jobs=-1))
    ])

    if XGBClassifier is not None:
        models['xgb'] = Pipeline([
            ('clf', XGBClassifier(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                objective='binary:logistic',
                eval_metric='auc',
                n_jobs=-1,
                tree_method='hist',
                random_state=42,
            ))
        ])

    if LGBMClassifier is not None:
        models['lgbm'] = Pipeline([
            ('clf', LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                objective='binary',
                random_state=42,
                n_jobs=-1,
            ))
        ])

    return models


def train_and_save(models: Dict[str, Pipeline], X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, out_dir: str) -> List[Tuple[str, float, str]]:
    os.makedirs(out_dir, exist_ok=True)
    results: List[Tuple[str, float, str]] = []
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, proba)
        model_path = os.path.join(out_dir, f"{name}.joblib")
        joblib.dump(pipe, model_path)
        results.append((name, float(auc), model_path))
    return results
