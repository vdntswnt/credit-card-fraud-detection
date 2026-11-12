"""Train, compare and evaluate credit card fraud models.

Usage:  python src/train.py

Pipeline
--------
1. Load data, drop exact duplicate rows, sort by time.
2. Chronological 60/20/20 train/validation/test split (no leakage from the future).
3. Compare: Logistic Regression, Random Forest, XGBoost, and XGBoost + SMOTE.
4. Pick the best model by validation PR-AUC.
5. Choose the decision threshold on VALIDATION data by minimising business cost.
6. Report final metrics on the untouched TEST set; save model, metrics and figures.
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FEATURES, FIG_DIR, MODEL_DIR, ROOT, add_features, load_data, time_split  # noqa: E402

SEED = 42
REVIEW_COST = 5.0  # assumed cost (in currency units) of a human reviewing one flagged transaction


def business_cost(y_true, y_prob, amounts, threshold):
    """Cost = value of missed frauds (full amount lost) + review cost of every flagged transaction."""
    flagged = y_prob >= threshold
    missed = (~flagged) & (y_true == 1)
    return float(amounts[missed].sum() + REVIEW_COST * flagged.sum())


def best_threshold(y_true, y_prob, amounts):
    grid = np.unique(np.concatenate([np.linspace(0.01, 0.99, 99), np.quantile(y_prob, np.linspace(0.9, 0.9999, 60))]))
    costs = [business_cost(y_true, y_prob, amounts, t) for t in grid]
    return float(grid[int(np.argmin(costs))])


def metrics_at(y_true, y_prob, threshold):
    pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred)), 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_data()
    df = add_features(raw)
    train, val, test = time_split(df)
    Xtr, ytr = train[FEATURES], train["Class"]
    Xva, yva = val[FEATURES], val["Class"]
    Xte, yte = test[FEATURES], test["Class"]
    spw = float((ytr == 0).sum() / (ytr == 1).sum())

    print(f"Rows after de-duplication: {len(df):,} | frauds: {int(df.Class.sum())} ({df.Class.mean():.3%})")
    for name, d in [("train", train), ("val", val), ("test", test)]:
        print(f"  {name:5s}: {len(d):>7,} rows, {int(d.Class.sum()):>3} frauds")

    def make_xgb():
        return XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            eval_metric="aucpr", n_jobs=-1, random_state=SEED,
        )

    candidates = {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED
        ),
        "XGBoost (class weights)": make_xgb().set_params(scale_pos_weight=spw),
    }

    fitted, results = {}, {}
    for name, model in candidates.items():
        model.fit(Xtr, ytr)
        fitted[name] = model

    # XGBoost + SMOTE: oversample the minority class in the training set only.
    smote = SMOTE(sampling_strategy=0.1, random_state=SEED)
    Xs, ys = smote.fit_resample(Xtr, ytr)
    xgb_smote = make_xgb()
    xgb_smote.fit(Xs, ys)
    fitted["XGBoost + SMOTE"] = xgb_smote

    for name, model in fitted.items():
        pv = model.predict_proba(Xva)[:, 1]
        pt = model.predict_proba(Xte)[:, 1]
        results[name] = {
            "val_pr_auc": round(float(average_precision_score(yva, pv)), 4),
            "val_roc_auc": round(float(roc_auc_score(yva, pv)), 4),
            "test_pr_auc": round(float(average_precision_score(yte, pt)), 4),
            "test_roc_auc": round(float(roc_auc_score(yte, pt)), 4),
        }
        print(f"{name:26s} val PR-AUC {results[name]['val_pr_auc']:.4f} | test PR-AUC {results[name]['test_pr_auc']:.4f}")

    # --- model selection on VALIDATION only
    best_name = max(results, key=lambda k: results[k]["val_pr_auc"])
    best = fitted[best_name]
    print(f"\nSelected model (best validation PR-AUC): {best_name}")

    pva = best.predict_proba(Xva)[:, 1]
    pte = best.predict_proba(Xte)[:, 1]
    thr = best_threshold(yva.values, pva, val["Amount"].values)

    default_test = metrics_at(yte, pte, 0.5)
    tuned_test = metrics_at(yte, pte, thr)
    cost_none = float(test.loc[yte == 1, "Amount"].sum())  # no model: every fraud is lost
    cost_default = business_cost(yte.values, pte, test["Amount"].values, 0.5)
    cost_tuned = business_cost(yte.values, pte, test["Amount"].values, thr)

    final = {
        "selected_model": best_name,
        "data": {
            "rows_after_dedup": int(len(df)),
            "fraud_rate_pct": round(float(df.Class.mean() * 100), 3),
            "train_rows": int(len(train)), "val_rows": int(len(val)), "test_rows": int(len(test)),
            "test_frauds": int(yte.sum()),
        },
        "model_comparison": results,
        "threshold_validation_tuned": round(thr, 4),
        "test_at_default_0.5": default_test,
        "test_at_tuned_threshold": tuned_test,
        "test_cost_assumptions": {"review_cost_per_flag": REVIEW_COST},
        "test_cost": {
            "no_model_loss": round(cost_none, 2),
            "default_threshold": round(cost_default, 2),
            "tuned_threshold": round(cost_tuned, 2),
            "savings_vs_no_model_pct": round(100 * (1 - cost_tuned / cost_none), 1),
        },
    }
    print(json.dumps(final, indent=2))

    # --- save artefacts
    joblib.dump({"model": best, "features": FEATURES, "threshold": thr, "model_name": best_name},
                MODEL_DIR / "fraud_model.joblib")
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(final, indent=2))

    # sample transactions for the Streamlit demo (real rows from the held-out test set)
    demo = test.copy()
    demo_frauds = demo[demo.Class == 1].sample(min(25, int(yte.sum())), random_state=SEED)
    demo_legit = demo[demo.Class == 0].sample(75, random_state=SEED)
    pd.concat([demo_frauds, demo_legit]).sample(frac=1, random_state=SEED)[
        ["Time", *[f"V{i}" for i in range(1, 29)], "Amount", "Class"]
    ].to_csv(ROOT / "data" / "sample_transactions.csv", index=False)

    # --- figures
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, model in fitted.items():
        p = model.predict_proba(Xte)[:, 1]
        pr, rc, _ = precision_recall_curve(yte, p)
        ax.plot(rc, pr, label=f"{name} (AP={results[name]['test_pr_auc']:.3f})", lw=2)
    ax.axhline(yte.mean(), color="grey", ls="--", label=f"Random guess ({yte.mean():.4f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-Recall curves (held-out test set)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "pr_curves.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, (title, thr_v) in zip(axes, [("Default threshold 0.5", 0.5), (f"Cost-tuned threshold {thr:.3f}", thr)]):
        cm = confusion_matrix(yte, (pte >= thr_v).astype(int))
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black", fontsize=12)
        ax.set_xticks([0, 1], ["Legit", "Fraud"]); ax.set_yticks([0, 1], ["Legit", "Fraud"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title, fontsize=10); ax.grid(False)
    fig.suptitle(f"{best_name} - confusion matrices on test set", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG_DIR / "confusion_matrices.png", dpi=150); plt.close(fig)

    # SHAP only works directly on tree models
    tree_model = best if hasattr(best, "get_booster") or hasattr(best, "estimators_") else None
    if tree_model is not None:
        sample = Xte.sample(3000, random_state=SEED)
        sample = sample.copy()
        explainer = shap.TreeExplainer(tree_model)
        sv = explainer.shap_values(sample)
        if isinstance(sv, list):
            sv = sv[1]
        elif getattr(sv, "ndim", 2) == 3:
            sv = sv[:, :, 1]
        plt.figure()
        shap.summary_plot(sv, sample, show=False, max_display=15)
        plt.title("Top features driving fraud predictions (SHAP)")
        plt.tight_layout(); plt.savefig(FIG_DIR / "shap_summary.png", dpi=150, bbox_inches="tight"); plt.close()

    print("\nSaved model, metrics.json, sample_transactions.csv and figures.")


if __name__ == "__main__":
    main()
