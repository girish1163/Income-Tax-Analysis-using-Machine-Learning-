from __future__ import annotations
import os
from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_income_tax_ratio(X: pd.DataFrame, out_dir: str, fname: str = "ratio_distribution.png", show: bool = False) -> str:
    _ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(X['tax_to_income_ratio'].clip(0, 5), ax=ax, fill=True, color="#4e79a7")
    ax.set_title("Income vs Tax Paid Ratio Distribution")
    ax.set_xlabel("tax_paid / income")
    path = os.path.join(out_dir, fname)
    fig.tight_layout()
    fig.savefig(path)
    if show:
        plt.show()
    plt.close(fig)
    return path


# --- Evaluation plots ---
def plot_roc_pr_curves(y_true: np.ndarray, y_proba: np.ndarray, out_dir: str, prefix: str, show: bool = False) -> tuple[str, str]:
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    _ensure_dir(out_dir)
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    ax1.plot(fpr, tpr, label=f"ROC AUC={roc_auc:.3f}", color="#4e79a7")
    ax1.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax1.set_title("ROC Curve")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend(loc='lower right')
    roc_path = os.path.join(out_dir, f"{prefix}_roc.png")
    fig1.tight_layout(); fig1.savefig(roc_path)
    if show: plt.show()
    plt.close(fig1)

    # Precision-Recall
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.plot(recall, precision, label=f"AP={ap:.3f}", color="#e15759")
    ax2.set_title("Precision-Recall Curve")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend(loc='lower left')
    pr_path = os.path.join(out_dir, f"{prefix}_pr.png")
    fig2.tight_layout(); fig2.savefig(pr_path)
    if show: plt.show()
    plt.close(fig2)

    return roc_path, pr_path


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, out_dir: str, prefix: str, show: bool = False) -> str:
    from sklearn.metrics import confusion_matrix
    _ensure_dir(out_dir)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    path = os.path.join(out_dir, f"{prefix}_confusion.png")
    fig.tight_layout(); fig.savefig(path)
    if show: plt.show()
    plt.close(fig)
    return path


def plot_precision_recall_vs_threshold(y_true: np.ndarray, y_proba: np.ndarray, out_dir: str, prefix: str, show: bool = False) -> str:
    from sklearn.metrics import precision_recall_curve
    _ensure_dir(out_dir)
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    thresholds = np.append(thresholds, 1.0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, precision, label='Precision', color="#4e79a7")
    ax.plot(thresholds, recall, label='Recall', color="#e15759")
    ax.set_title('Precision/Recall vs Threshold')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.legend()
    path = os.path.join(out_dir, f"{prefix}_pr_threshold.png")
    fig.tight_layout(); fig.savefig(path)
    if show: plt.show()
    plt.close(fig)
    return path


def plot_feature_importance(model, feature_names: list[str], out_dir: str, prefix: str, top_k: int = 20, show: bool = False) -> str | None:
    _ensure_dir(out_dir)
    clf = getattr(model, 'named_steps', {}).get('clf', model)
    importances = None
    if hasattr(clf, 'feature_importances_'):
        importances = np.array(clf.feature_importances_, dtype=float)
    elif hasattr(clf, 'coef_'):
        coef = getattr(clf, 'coef_', None)
        if coef is not None:
            importances = np.abs(np.ravel(coef))
    if importances is None:
        return None
    idx = np.argsort(importances)[::-1][:top_k]
    fig, ax = plt.subplots(figsize=(8, max(4, top_k * 0.3)))
    ax.barh(np.array(feature_names)[idx][::-1], importances[idx][::-1], color="#59a14f")
    ax.set_title('Top Feature Importance')
    ax.set_xlabel('Importance')
    path = os.path.join(out_dir, f"{prefix}_feature_importance.png")
    fig.tight_layout(); fig.savefig(path)
    if show: plt.show()
    plt.close(fig)
    return path


def plot_income_spikes(X: pd.DataFrame, out_dir: str, fname: str = "income_spikes.png", show: bool = False) -> str:
    _ensure_dir(out_dir)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(X['income_yoy_change'].clip(-2, 2), bins=40, ax=axes[0], color="#e15759")
    axes[0].set_title("YoY Income Change (clipped)")
    axes[0].set_xlabel("YoY change")
    sns.histplot(X['income_z'].clip(-5, 5), bins=40, ax=axes[1], color="#76b7b2")
    axes[1].set_title("Income Z-score by Taxpayer (clipped)")
    axes[1].set_xlabel("Z-score")
    path = os.path.join(out_dir, fname)
    fig.tight_layout()
    fig.savefig(path)
    if show:
        plt.show()
    plt.close(fig)
    return path


def plot_deductions(X: pd.DataFrame, out_dir: str, fname: str = "deductions.png", show: bool = False) -> str:
    _ensure_dir(out_dir)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.kdeplot(X['deduct_to_income_ratio'].clip(0, 2), ax=axes[0], fill=True, color="#59a14f")
    axes[0].set_title("Deductions to Income Ratio")
    axes[0].set_xlabel("deductions / income")
    sns.histplot(X['deduct_peer_percentile'], bins=40, ax=axes[1], color="#edc948")
    axes[1].set_title("Deductions Percentile (peer by year)")
    axes[1].set_xlabel("percentile")
    path = os.path.join(out_dir, fname)
    fig.tight_layout()
    fig.savefig(path)
    if show:
        plt.show()
    plt.close(fig)
    return path


def plot_salary_mismatch(X: pd.DataFrame, out_dir: str, fname: str = "salary_mismatch.png", show: bool = False) -> str:
    _ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(x=X['salary_abs_diff'], y=X['salary_rel_diff'].clip(0, 5), ax=ax, s=12, alpha=0.5)
    ax.set_title("Salary Mismatch: Absolute vs Relative")
    ax.set_xlabel("abs diff")
    ax.set_ylabel("relative diff")
    path = os.path.join(out_dir, fname)
    fig.tight_layout()
    fig.savefig(path)
    if show:
        plt.show()
    plt.close(fig)
    return path


def plot_filing_delays(X: pd.DataFrame, out_dir: str, fname: str = "filing_delays.png", show: bool = False) -> str:
    _ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(x=X['filing_delay_days'].clip(-365, 365), ax=ax, color="#bab0ab")
    ax.set_title("Filing Delay (days)")
    ax.set_xlabel("days (clipped -365..365)")
    path = os.path.join(out_dir, fname)
    fig.tight_layout()
    fig.savefig(path)
    if show:
        plt.show()
    plt.close(fig)
    return path


def plot_trending_year(df_raw: pd.DataFrame, year_col: str, title_col: str, tickets_col: str, year: Optional[int], out_dir: str, fname: str = "trending_year.png", show: bool = False) -> str:
    _ensure_dir(out_dir)
    if year is None:
        year = int(pd.to_datetime(df_raw['period_end_date']).dt.year.max())
    view = df_raw.copy()
    if year_col not in view.columns:
        view[year_col] = pd.to_datetime(view['period_end_date']).dt.year
    view = view[view[year_col] == year]
    view = view.sort_values(tickets_col, ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(y=view[title_col], x=view[tickets_col], ax=ax, color="#4e79a7")
    ax.set_title(f"Trending Films {year} by {tickets_col}")
    ax.set_xlabel(tickets_col)
    ax.set_ylabel("")
    path = os.path.join(out_dir, fname)
    fig.tight_layout()
    fig.savefig(path)
    if show:
        plt.show()
    plt.close(fig)
    return path
