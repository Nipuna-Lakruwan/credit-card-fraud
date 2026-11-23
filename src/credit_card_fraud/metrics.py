"""Metrics for rare-event detection: PR-AUC with bootstrap CIs, operating points and a cost model."""
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

REVIEW_COST = 5.0  # euros an analyst spends checking one flagged transaction (assumption, see README)


def recall_at_precision(y, s, min_precision: float) -> float:
    p, r, _ = precision_recall_curve(y, s)
    ok = p >= min_precision
    return float(r[ok].max()) if ok.any() else 0.0


def precision_at_recall(y, s, min_recall: float) -> float:
    p, r, _ = precision_recall_curve(y, s)
    ok = r >= min_recall
    return float(p[ok].max()) if ok.any() else 0.0


def summary(y, s) -> dict:
    return {
        "pr_auc": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)),
        "recall_at_p90": recall_at_precision(y, s, 0.90),
        "precision_at_r80": precision_at_recall(y, s, 0.80),
    }


def bootstrap(y, s, n: int = 2000, seed: int = 0) -> dict:
    """95% percentile intervals, resampling transactions with replacement."""
    y, s = np.asarray(y), np.asarray(s)
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in summary(y, s)}
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0:
            continue
        for k, v in summary(y[idx], s[idx]).items():
            draws[k].append(v)
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in draws.items()}


def cost(y, flagged, amount, review_cost: float = REVIEW_COST) -> dict:
    """Missed fraud costs its amount; every flag costs one review. Caught fraud is assumed recovered."""
    y, flagged, amount = map(np.asarray, (y, flagged, amount))
    missed = float(amount[(y == 1) & ~flagged].sum())
    reviews = float(flagged.sum() * review_cost)
    return {"missed_fraud_eur": missed, "review_eur": reviews, "total_eur": missed + reviews,
            "flags": int(flagged.sum()), "caught": int((flagged & (y == 1)).sum()), "frauds": int(y.sum())}


def best_threshold(y, s, amount, review_cost: float = REVIEW_COST) -> float:
    """Threshold on the score that minimises total cost (choose it on validation, never on test)."""
    s = np.asarray(s)
    cands = np.unique(np.quantile(s, np.linspace(0.9, 1.0, 2001)))
    totals = [cost(y, s >= t, amount, review_cost)["total_eur"] for t in cands]
    return float(cands[int(np.argmin(totals))])
