"""Save the 5-seed LightGBM (class weights, trained on the training period) for the Hugging Face model repo."""
import json

from credit_card_fraud.data import ROOT, features, load, time_split
from credit_card_fraud.models import make


def main() -> None:
    out = ROOT / "models"; out.mkdir(exist_ok=True)
    m = json.loads((ROOT / "results" / "metrics.json").read_text())
    tr, _, _ = time_split(load())
    X, y = features(tr), tr.Class.to_numpy()
    for seed in range(5):
        make("lightgbm", "class_weight", seed).fit(X.to_numpy(), y).booster_.save_model(str(out / f"lightgbm_seed{seed}.txt"))
    (out / "config.json").write_text(json.dumps({
        "features": list(X.columns), "score": "mean of the 5 boosters' predicted probability",
        "alert_threshold": m["test"]["lightgbm"]["threshold_from_val"],
        "test_pr_auc": m["test"]["lightgbm"]["pr_auc"], "test_pr_auc_ci95": m["test"]["lightgbm"]["ci95"]["pr_auc"]}, indent=2))
    print("saved", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
