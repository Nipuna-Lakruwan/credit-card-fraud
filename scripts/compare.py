"""Paired bootstrap of the test PR-AUC gap between LightGBM and logistic regression."""
import json

import numpy as np
from sklearn.metrics import average_precision_score

from credit_card_fraud.data import ROOT


def main() -> None:
    z = np.load(ROOT / "results" / "test_scores.npz")
    y, a, b = z["y"], z["lightgbm"], z["logreg"]
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum():
            diffs.append(average_precision_score(y[i], a[i]) - average_precision_score(y[i], b[i]))
    diffs = np.array(diffs)
    out = {"lightgbm_minus_logreg_pr_auc": float(average_precision_score(y, a) - average_precision_score(y, b)),
           "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
           "share_of_resamples_lightgbm_better": float((diffs > 0).mean())}
    (ROOT / "results" / "paired_comparison.json").write_text(json.dumps(out, indent=2))
    print(out)


if __name__ == "__main__":
    main()
