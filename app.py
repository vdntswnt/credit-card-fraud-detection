"""Streamlit demo: score credit card transactions for fraud.

Run:  streamlit run app.py
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from common import FEATURES, PCA_COLS, add_features  # noqa: E402

st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="🛡️", layout="wide")


@st.cache_resource
def load_artifacts():
    bundle = joblib.load(ROOT / "models" / "fraud_model.joblib")
    metrics = json.loads((ROOT / "reports" / "metrics.json").read_text())
    samples = pd.read_csv(ROOT / "data" / "sample_transactions.csv")
    return bundle, metrics, samples


def score(df: pd.DataFrame, bundle) -> pd.Series:
    X = add_features(df)[FEATURES]
    return pd.Series(bundle["model"].predict_proba(X)[:, 1], index=df.index, name="fraud_probability")


bundle, metrics, samples = load_artifacts()
tuned_thr = float(bundle["threshold"])

st.title("🛡️ Credit Card Fraud Detector")
st.caption(
    f"Model: **{bundle['model_name']}** · trained on a chronological 60% of the ULB dataset · "
    f"decision threshold tuned on validation data to minimise business cost"
)

tab_single, tab_batch, tab_perf = st.tabs(["Score a transaction", "Batch scoring", "Model performance"])

# ------------------------------------------------------------------ single transaction
with tab_single:
    st.write(
        "Pick one of 100 real transactions from the **held-out test period** (25 frauds, 75 legitimate) "
        "and see what the model says. The true label is hidden until you reveal it."
    )
    threshold = st.slider(
        "Alert threshold (higher = fewer false alarms, more missed fraud)",
        0.01, 0.99, min(max(tuned_thr, 0.01), 0.99), 0.01,
        help=f"The cost-optimal value found on validation data was {tuned_thr:.2f}.",
    )
    idx = st.selectbox(
        "Transaction",
        samples.index,
        format_func=lambda i: f"#{i}  |  amount {samples.loc[i, 'Amount']:,.2f}  |  t = {int(samples.loc[i, 'Time']):,}s",
    )
    row = samples.loc[[idx]]
    prob = float(score(row, bundle).iloc[0])
    flagged = prob >= threshold

    c1, c2, c3 = st.columns(3)
    c1.metric("Fraud probability", f"{prob:.1%}")
    c2.metric("Decision", "🚨 FLAG FOR REVIEW" if flagged else "✅ Approve")
    c3.metric("Amount", f"{row['Amount'].iloc[0]:,.2f}")
    st.progress(min(prob, 1.0))

    if st.checkbox("Reveal the true label"):
        truth = int(row["Class"].iloc[0])
        correct = truth == int(flagged)
        st.write(f"True label: **{'Fraud' if truth else 'Legitimate'}** — model was {'✅ correct' if correct else '❌ wrong'}.")

    with st.expander("Raw features"):
        st.dataframe(row.drop(columns=["Class"]).T.rename(columns={idx: "value"}))

# ------------------------------------------------------------------ batch scoring
with tab_batch:
    st.write(
        "Upload a CSV with columns `Time`, `V1`…`V28`, `Amount` (same format as the Kaggle file). "
        "A `Class` column is optional."
    )
    up = st.file_uploader("CSV file", type="csv")
    use_demo = st.checkbox("Use the 100 demo transactions instead", value=up is None)
    data = samples.copy() if (use_demo and up is None) else (pd.read_csv(up) if up else None)

    if data is not None:
        needed = ["Time", *PCA_COLS, "Amount"]
        missing = [c for c in needed if c not in data.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            data = data.copy()
            data["fraud_probability"] = score(data, bundle)
            data["flagged"] = data["fraud_probability"] >= tuned_thr
            n_flag = int(data["flagged"].sum())
            st.success(f"Scored {len(data):,} transactions — {n_flag:,} flagged at threshold {tuned_thr:.2f}.")
            if "Class" in data.columns:
                tp = int(((data.flagged) & (data.Class == 1)).sum())
                st.write(f"Of {int(data.Class.sum())} true frauds, **{tp}** were caught; **{n_flag - tp}** false alarms.")
            show = data.sort_values("fraud_probability", ascending=False)
            st.dataframe(show[["Time", "Amount", "fraud_probability", "flagged"]
                              + (["Class"] if "Class" in data.columns else [])].head(200), width="stretch")
            st.download_button("Download scored CSV", show.to_csv(index=False).encode(), "scored_transactions.csv", "text/csv")

# ------------------------------------------------------------------ performance
with tab_perf:
    d = metrics["data"]
    st.subheader("Held-out test set results")
    st.write(
        f"{d['test_rows']:,} transactions, {d['test_frauds']} frauds ({d['fraud_rate_pct']}% base rate). "
        "Models are evaluated with **PR-AUC** because accuracy is meaningless at this imbalance."
    )
    comp = pd.DataFrame(metrics["model_comparison"]).T.rename(columns={
        "val_pr_auc": "Val PR-AUC", "val_roc_auc": "Val ROC-AUC", "test_pr_auc": "Test PR-AUC", "test_roc_auc": "Test ROC-AUC"})
    st.dataframe(comp)

    a, b = metrics["test_at_default_0.5"], metrics["test_at_tuned_threshold"]
    cc1, cc2 = st.columns(2)
    cc1.metric("Default threshold 0.5", f"P {a['precision']:.0%} / R {a['recall']:.0%}", f"{a['fp']} false alarms, {a['fn']} missed")
    cc2.metric(f"Cost-tuned threshold {b['threshold']}", f"P {b['precision']:.0%} / R {b['recall']:.0%}",
               f"{b['fp']} false alarms, {b['fn']} missed")
    st.caption(
        f"Assumed cost: a missed fraud loses its full amount; each flagged transaction costs {metrics['test_cost_assumptions']['review_cost_per_flag']:.0f} to review. "
        f"Tuned model cuts fraud-related cost by **{metrics['test_cost']['savings_vs_no_model_pct']}%** vs. having no model."
    )
    for fig in ["pr_curves.png", "confusion_matrices.png", "shap_summary.png"]:
        path = ROOT / "reports" / "figures" / fig
        if path.exists():
            st.image(str(path), width="stretch")
