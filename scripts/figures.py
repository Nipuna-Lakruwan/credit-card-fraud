"""Draw the figures from results/metrics.json and results/test_scores.npz."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

from credit_card_fraud.data import ROOT, features, load, time_split
from credit_card_fraud.metrics import cost
from credit_card_fraud.models import make

FIG = ROOT / "figures"
NAMES = {"logreg": "Logistic regression", "lightgbm": "LightGBM", "isolation_forest": "Isolation forest (no labels)"}
COLORS = {"logreg": "#4C72B0", "lightgbm": "#DD8452", "isolation_forest": "#8C8C8C"}


def main() -> None:
    FIG.mkdir(exist_ok=True)
    m = json.loads((ROOT / "results" / "metrics.json").read_text())
    z = np.load(ROOT / "results" / "test_scores.npz")
    y, amount = z["y"], z["amount"]

    # 1. precision-recall curves on the test period
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for fam in ["lightgbm", "logreg", "isolation_forest"]:
        p, r, _ = precision_recall_curve(y, z[fam])
        t = m["test"][fam]
        ax.plot(r, p, color=COLORS[fam], lw=2,
                label=f"{NAMES[fam]}: PR-AUC {t['pr_auc']:.2f} [{t['ci95']['pr_auc'][0]:.2f}, {t['ci95']['pr_auc'][1]:.2f}]")
    ax.axhline(y.mean(), color="k", ls=":", lw=1, label=f"Random guessing ({y.mean()*100:.2f}% fraud)")
    ax.set(xlabel="Recall (share of fraud caught)", ylabel="Precision (share of flags that are fraud)",
           title=f"Test period (last 7.7 h, {int(y.sum())} frauds)", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(FIG / "pr_curves_test.png", dpi=150); plt.close(fig)

    # 2. validation grid: how much does each imbalance trick help?
    hows = ["none", "class_weight", "smote", "undersample"]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    w = 0.38
    for k, fam in enumerate(["logreg", "lightgbm"]):
        v = [m["validation"][f"{fam}/{h}"] for h in hows]
        ax.bar(np.arange(4) + (k - 0.5) * w, [d["pr_auc_mean"] for d in v], w, yerr=[d["pr_auc_sd"] for d in v],
               color=COLORS[fam], label=NAMES[fam], capsize=3)
    ax.axhline(m["validation"]["isolation_forest/none"]["pr_auc_mean"], color=COLORS["isolation_forest"], ls="--",
               label=NAMES["isolation_forest"])
    ax.set_xticks(range(4), ["nothing", "class weights", "SMOTE", "undersample 1:10"])
    ax.set(ylabel="Validation PR-AUC", title="Imbalance handling (mean ± sd over 5 seeds)", ylim=(0, 1))
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "imbalance_validation.png", dpi=150); plt.close(fig)

    # 3. cost against threshold on the test period, with the threshold chosen on validation
    s = z["lightgbm"]
    thr = np.unique(np.quantile(s, np.linspace(0.95, 1, 800)))
    totals = np.array([cost(y, s >= t, amount)["total_eur"] for t in thr])
    chosen = m["test"]["lightgbm"]["threshold_from_val"]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(thr, totals, color=COLORS["lightgbm"])
    ax.axvline(chosen, color="k", ls="--", lw=1, label=f"threshold picked on validation (€{m['test']['lightgbm']['cost_test']['total_eur']:,.0f})")
    ax.axhline(m["cost_baselines_test"]["flag_nothing"]["total_eur"], color="r", ls=":", label=f"flag nothing (€{m['cost_baselines_test']['flag_nothing']['total_eur']:,.0f})")
    ax.set(xscale="log", xlabel="LightGBM score threshold (log scale)", ylabel="Test-period cost (€)",
           title=f"Missed fraud + €{m['cost_baselines_test']['review_cost_eur']:.0f} per review")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "cost_vs_threshold.png", dpi=150); plt.close(fig)

    # 4. feature importance (gain) of one LightGBM fitted on the training period
    df = load(); tr, _, _ = time_split(df)
    Xtr = features(tr)
    model = make("lightgbm", m["test"]["lightgbm"]["imbalance"], 0).fit(Xtr.to_numpy(), tr.Class.to_numpy())
    gain = model.booster_.feature_importance("gain"); order = np.argsort(gain)[-12:]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.barh(np.array(Xtr.columns)[order], gain[order] / gain.sum(), color=COLORS["lightgbm"])
    ax.set(xlabel="Share of total split gain", title="LightGBM: top 12 features")
    fig.tight_layout(); fig.savefig(FIG / "feature_importance.png", dpi=150); plt.close(fig)
    print("figures in", FIG)


if __name__ == "__main__":
    main()
