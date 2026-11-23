"""Run every experiment and write results/metrics.json.

1. Validation grid: {logreg, lightgbm} x {none, class_weight, smote, undersample} x 5 seeds, plus an
   isolation forest, trained on the first 60% of time and scored on the next 20%.
2. Final test: the best imbalance setting per family (by mean validation PR-AUC), a 5-seed average,
   scored once on the last 20%. The alert threshold is chosen on validation by the cost model.
3. Random-split check: the same LightGBM on a shuffled stratified 80/20 split, to show how much a
   random split flatters the numbers.
"""
import json
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from credit_card_fraud.data import ROOT, features, load, time_split
from credit_card_fraud.metrics import REVIEW_COST, best_threshold, bootstrap, cost, summary
from credit_card_fraud.models import IMBALANCE, fit_score

SEEDS = [0, 1, 2, 3, 4]
OUT = ROOT / "results"


def scores(family, how, X_tr, y_tr, X_ev, seeds=SEEDS):
    seeds = [0] if (family == "logreg" and how in ("none", "class_weight")) else seeds  # deterministic
    return [fit_score(family, how, s, X_tr, y_tr, X_ev) for s in seeds]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    df = load()
    tr, va, te = time_split(df)
    X = {k: features(d).to_numpy() for k, d in (("tr", tr), ("va", va), ("te", te))}
    y = {"tr": tr.Class.to_numpy(), "va": va.Class.to_numpy(), "te": te.Class.to_numpy()}
    res = {"split": {k: {"rows": int(len(v)), "fraud": int(v.sum())} for k, v in y.items()}, "validation": {}}

    # 1. validation grid
    for family in ["logreg", "lightgbm"]:
        for how in IMBALANCE:
            t0 = time.time()
            aps = [summary(y["va"], s)["pr_auc"] for s in scores(family, how, X["tr"], y["tr"], X["va"])]
            res["validation"][f"{family}/{how}"] = {"pr_auc_mean": float(np.mean(aps)), "pr_auc_sd": float(np.std(aps)),
                                                    "seeds": len(aps), "seconds": round(time.time() - t0, 1)}
            print(f"{family:9s} {how:12s} val PR-AUC {np.mean(aps):.3f} ± {np.std(aps):.3f}  ({time.time()-t0:.0f}s)", flush=True)
    s_if = scores("isolation_forest", "none", X["tr"], y["tr"], X["va"])
    res["validation"]["isolation_forest/none"] = {"pr_auc_mean": float(np.mean([summary(y["va"], s)["pr_auc"] for s in s_if]))}
    print("isolation forest val PR-AUC", res["validation"]["isolation_forest/none"]["pr_auc_mean"])

    # 2. final test, one shot
    res["test"] = {}
    curves = {}
    for family in ["logreg", "lightgbm", "isolation_forest"]:
        hows = IMBALANCE if family != "isolation_forest" else ["none"]
        how = max(hows, key=lambda h: res["validation"][f"{family}/{h}"]["pr_auc_mean"])
        s_va = np.mean(scores(family, how, X["tr"], y["tr"], X["va"]), axis=0)
        s_te = np.mean(scores(family, how, X["tr"], y["tr"], X["te"]), axis=0)
        thr = best_threshold(y["va"], s_va, va.Amount.to_numpy())
        res["test"][family] = {
            "imbalance": how, **summary(y["te"], s_te), "ci95": bootstrap(y["te"], s_te),
            "threshold_from_val": thr,
            "cost_test": cost(y["te"], s_te >= thr, te.Amount.to_numpy()),
        }
        curves[family] = s_te
        r = res["test"][family]
        print(f"TEST {family:16s} ({how}) PR-AUC {r['pr_auc']:.3f} CI {r['ci95']['pr_auc'][0]:.3f}-{r['ci95']['pr_auc'][1]:.3f} "
              f"ROC {r['roc_auc']:.3f} R@P90 {r['recall_at_p90']:.2f} cost {r['cost_test']}", flush=True)
    np.savez(OUT / "test_scores.npz", y=y["te"], amount=te.Amount.to_numpy(), **curves)

    # baselines for the cost model: flag nothing, flag everything
    amt = te.Amount.to_numpy()
    res["cost_baselines_test"] = {"flag_nothing": cost(y["te"], np.zeros_like(y["te"], bool), amt),
                                  "flag_everything": cost(y["te"], np.ones_like(y["te"], bool), amt),
                                  "review_cost_eur": REVIEW_COST}

    # 3. random split comparison (what most notebooks report)
    best_lgb = res["test"]["lightgbm"]["imbalance"]
    Xall, yall = features(df).to_numpy(), df.Class.to_numpy()
    rand = []
    for rep, (i, j) in enumerate(StratifiedShuffleSplit(5, test_size=0.2, random_state=0).split(Xall, yall)):
        rand.append(summary(yall[j], fit_score("lightgbm", best_lgb, rep, Xall[i], yall[i], Xall[j]))["pr_auc"])
    res["random_split_lightgbm"] = {"pr_auc_mean": float(np.mean(rand)), "pr_auc_sd": float(np.std(rand)), "repeats": 5}
    print("random split LightGBM PR-AUC", np.mean(rand), np.std(rand))

    (OUT / "metrics.json").write_text(json.dumps(res, indent=2))
    print("wrote", OUT / "metrics.json")


if __name__ == "__main__":
    main()
