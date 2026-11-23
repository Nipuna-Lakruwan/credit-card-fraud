import numpy as np
import pandas as pd

from credit_card_fraud.data import features, time_split
from credit_card_fraud.metrics import best_threshold, cost, precision_at_recall, recall_at_precision, summary


def toy(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f"V{i}": rng.normal(size=n) for i in range(1, 29)})
    df["Time"] = np.sort(rng.uniform(0, 172800, n))
    df["Amount"] = rng.exponential(50, n)
    df["Class"] = (rng.uniform(size=n) < 0.05).astype(int)
    return df


def test_time_split_is_ordered_and_disjoint():
    df = toy()
    tr, va, te = time_split(df)
    assert len(tr) + len(va) + len(te) == len(df)
    assert tr.Time.max() <= va.Time.min() and va.Time.max() <= te.Time.min()


def test_features_shape_and_hour_circle():
    f = features(toy())
    assert f.shape[1] == 31
    assert np.allclose(f.hour_sin**2 + f.hour_cos**2, 1)


def test_perfect_scores():
    y = np.array([0, 0, 1, 1]); s = np.array([0.1, 0.2, 0.8, 0.9])
    m = summary(y, s)
    assert m["pr_auc"] == 1 and m["roc_auc"] == 1
    assert recall_at_precision(y, s, 0.9) == 1 and precision_at_recall(y, s, 0.8) == 1


def test_cost_and_threshold():
    y = np.array([0, 0, 0, 1]); amount = np.array([10, 10, 10, 500.0]); s = np.array([0.1, 0.2, 0.3, 0.9])
    c = cost(y, s >= 0.5, amount, review_cost=5)
    assert c == {"missed_fraud_eur": 0.0, "review_eur": 5.0, "total_eur": 5.0, "flags": 1, "caught": 1, "frauds": 1}
    assert cost(y, s >= 2, amount)["missed_fraud_eur"] == 500
    t = best_threshold(y, s, amount, review_cost=5)
    assert 0.3 < t <= 0.9
