"""
app.py
------
Streamlit dashboard for loan officers.

Run:   streamlit run app.py
Needs: `python train.py` to have been run first (creates models/credit_model.joblib and reports/).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from data_pipeline import CATEGORY_LEVELS, FEATURES, prepare_input
from train import (
    REPORT_DIR, build_lime, grouped_contributions, lime_explain, load_bundle,
    make_shap_explainer, predict_risk, risk_category,
)

st.set_page_config(page_title="Credit Risk Assessment", page_icon="💳", layout="wide")

BADGE = {"Low": "#2e9d4f", "Medium": "#e0a100", "High": "#d9363e"}


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model...")
def get_bundle() -> dict:
    return load_bundle()


@st.cache_resource(show_spinner=False)
def get_explainer():
    b = get_bundle()
    return make_shap_explainer(b["pipeline"], b["background"])


@st.cache_resource(show_spinner=False)
def get_lime():
    b = get_bundle()
    return build_lime(b["pipeline"], b["calibrator"], b["lime_ref"])


def show_image(name: str, caption: str) -> None:
    """Display a saved report image, or a hint if it has not been generated."""
    p = Path(REPORT_DIR) / name
    if p.exists():
        st.image(str(p), caption=caption)
    else:
        st.info(f"`{name}` not found - re-run `python train.py`.")


def show_table(name: str, title: str) -> None:
    p = Path(REPORT_DIR) / name
    if p.exists():
        st.markdown(f"**{title}**")
        st.dataframe(pd.read_csv(p), hide_index=True)


def badge(category: str) -> str:
    return (f"<span style='background:{BADGE[category]};color:white;padding:6px 16px;"
            f"border-radius:16px;font-weight:600;font-size:1.05rem'>{category} risk</span>")


def fmt_value(v) -> str:
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if isinstance(v, (float, np.floating)):
        return "n/a" if np.isnan(v) else (f"{v:,.2f}".rstrip("0").rstrip("."))
    return str(v)


def bar_chart(labels, values, title, xlabel):
    """Horizontal diverging bar chart (red = raises risk, blue = lowers it)."""
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(labels) + 1.2))
    colors = ["#d9363e" if v > 0 else "#2f6fb3" for v in values]
    ax.barh(range(len(labels)), values, color=colors)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Load model
# --------------------------------------------------------------------------- #
try:
    bundle = get_bundle()
except FileNotFoundError as e:
    st.error(f"{e}")
    st.stop()
except Exception as e:  # e.g. library version mismatch when unpickling
    st.error(f"Could not load the model: {e}. Re-run `python train.py` in this environment.")
    st.stop()

thr = float(bundle["threshold"])
unit = "probability" if bundle["model_name"] == "random_forest" else "log-odds"

st.title("💳 Loan Default Risk Assessment")
st.caption(
    f"Model: **{bundle['model_name']}** · imbalance strategy: **{bundle['strategy']}** · "
    f"decline threshold: **{thr:.0%}** (cost of missed default : wrong rejection = "
    f"{bundle['cost_fn']:g} : {bundle['cost_fp']:g}) · risk bands: Low < {0.5 * thr:.0%}, "
    f"Medium < {2 * thr:.0%}, High above"
)

# --------------------------------------------------------------------------- #
# Sidebar: single-application inputs
# --------------------------------------------------------------------------- #
sb = st.sidebar
sb.header("Applicant & loan details")
person_age = sb.number_input("Person Age", 18, 100, 26)
person_income = sb.number_input("Person Income ($/yr)", 1_000, 10_000_000, 55_000, step=1_000)
person_home_ownership = sb.selectbox("Person Home Ownership", CATEGORY_LEVELS["person_home_ownership"])
emp_unknown = sb.checkbox("Employment length unknown")
person_emp_length = np.nan if emp_unknown else sb.number_input("Person Employment Length (years)", 0.0, 60.0, 4.0, step=0.5)
loan_intent = sb.selectbox("Loan Intent", CATEGORY_LEVELS["loan_intent"])
loan_grade = sb.selectbox("Loan Grade", CATEGORY_LEVELS["loan_grade"], index=1)
loan_amnt = sb.number_input("Loan Amount ($)", 500, 1_000_000, 8_000, step=500)
rate_unknown = sb.checkbox("Interest rate unknown")
loan_int_rate = np.nan if rate_unknown else sb.number_input("Loan Interest Rate (%)", 1.0, 40.0, 11.0, step=0.1)
loan_percent_income = round(loan_amnt / person_income, 2)
sb.caption(f"Loan Percent Income (auto) = **{loan_percent_income:.2f}**")
cb_person_default_on_file = sb.selectbox("Credit History - Default on File", CATEGORY_LEVELS["cb_person_default_on_file"])
cb_person_cred_hist_length = sb.number_input("Credit History Length (years)", 0, 60, 4)

tab_single, tab_batch, tab_insights = st.tabs(
    ["Single application assessment", "Batch loan risk screening", "Model & dataset insights"]
)

# --------------------------------------------------------------------------- #
# Tab 1: single assessment
# --------------------------------------------------------------------------- #
with tab_single:
    row = pd.DataFrame([{
        "person_age": person_age, "person_income": person_income, "person_home_ownership": person_home_ownership,
        "person_emp_length": person_emp_length, "loan_intent": loan_intent, "loan_grade": loan_grade,
        "loan_amnt": loan_amnt, "loan_int_rate": loan_int_rate, "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": cb_person_default_on_file,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
    }])
    try:
        row, _ = prepare_input(row)
        risk = float(predict_risk(bundle, row)[0])
        approve = risk < thr
        category = risk_category(risk, thr)

        c1, c2, c3 = st.columns(3)
        c1.metric("Default risk score", f"{risk:.1%}")
        with c2:
            st.markdown("**Decision**")
            (st.success if approve else st.error)("APPROVE" if approve else "DECLINE / MANUAL REVIEW")
        with c3:
            st.markdown("**Risk category**")
            st.markdown(badge(category), unsafe_allow_html=True)

        st.subheader("Why this score? (SHAP)")
        contrib = grouped_contributions(get_explainer(), bundle["pipeline"], row).head(11)
        labels = [f"{r.feature} = {fmt_value(r.value)}" for r in contrib.itertuples()]
        fig = bar_chart(labels, contrib["shap"].tolist(), "Feature impact on this applicant's default risk",
                        f"SHAP value ({unit}); red raises risk, blue lowers it")
        st.pyplot(fig)
        plt.close(fig)

        if st.checkbox("Also show a LIME explanation (local surrogate model, a few seconds)"):
            with st.spinner("Running LIME..."):
                exp = lime_explain(get_lime(), row)
                items = exp.as_list()
            fig = bar_chart([k for k, _ in items], [v for _, v in items],
                            "LIME: local feature weights for P(default)", "weight; red raises risk, blue lowers it")
            st.pyplot(fig)
            plt.close(fig)
    except Exception as e:  # surface input/model problems instead of crashing the app
        st.error(f"Could not score this application: {e}")

# --------------------------------------------------------------------------- #
# Tab 2: batch screening
# --------------------------------------------------------------------------- #
with tab_batch:
    st.write("Upload a CSV with these columns (`loan_percent_income` is derived automatically if absent):")
    st.code(", ".join(FEATURES))
    sample = bundle["lime_ref"].head(5)
    st.download_button("Download a sample file", sample.to_csv(index=False), "sample_applications.csv", "text/csv")
    up = st.file_uploader("Applications CSV", type="csv")
    if up is not None:
        try:
            raw = pd.read_csv(up)
            if raw.empty:
                raise ValueError("The file has no rows.")
            feats, notes = prepare_input(raw)
            for n in notes:
                st.warning(n)
            p = predict_risk(bundle, feats)

            t = st.slider("Decline threshold (default = cost-optimal)", 0.01, 0.99, float(round(thr, 2)), 0.01)
            out = raw.copy()
            out["default_risk"] = np.round(p, 4)
            out["risk_category"] = [risk_category(v, t) for v in p]
            out["decision"] = np.where(p >= t, "DECLINE / REVIEW", "APPROVE")

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Applications", f"{len(out):,}")
            k2.metric("Flagged (decline/review)", f"{int((p >= t).sum()):,}")
            k3.metric("Flag rate", f"{(p >= t).mean():.1%}")
            k4.metric("Mean risk", f"{p.mean():.1%}")

            fig, ax = plt.subplots(figsize=(7, 2.6))
            ax.hist(p, bins=40, color="#4C78A8")
            ax.axvline(t, color="r", ls="--", label=f"threshold {t:.2f}")
            ax.set(xlabel="Predicted default risk", ylabel="Applications")
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            view = st.radio("Show", ["All", "Flagged only", "Approved only"], horizontal=True)
            shown = {"All": out, "Flagged only": out[p >= t], "Approved only": out[p < t]}[view]
            st.dataframe(shown)
            st.download_button("Download scored CSV (all rows)", out.to_csv(index=False),
                               "scored_applications.csv", "text/csv")
        except Exception as e:
            st.error(f"Could not process the file: {e}")

# --------------------------------------------------------------------------- #
# Tab 3: insights
# --------------------------------------------------------------------------- #
with tab_insights:
    mp = Path(REPORT_DIR) / "metrics.json"
    if mp.exists():
        m = json.loads(mp.read_text())["test_at_chosen_threshold"]
        st.subheader("Held-out test performance")
        cols = st.columns(6)
        for col, (k, lab) in zip(cols, [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("precision", "Precision"),
                                        ("recall", "Recall"), ("f1", "F1"), ("fn", "Missed defaults (FN)")]):
            col.metric(lab, f"{m[k]:,.3f}" if isinstance(m[k], float) else f"{m[k]:,}")
        st.caption(f"Precision / recall / F1 at the cost-optimal threshold {thr:.0%}.")

    cr = bundle.get("clean_report")
    if cr:
        st.subheader("Dataset")
        st.write(
            f"{cr['rows_raw']:,} raw rows → **{cr['rows_clean']:,}** after removing {cr['dropped_outliers']} impossible "
            f"age/employment outliers and {cr['dropped_duplicates']} duplicates. Default rate **{cr['default_rate']:.1%}**. "
            f"Missing values (imputed): {cr['missing']}."
        )
    show_image("eda_target.png", "Class balance and default rate by segment")
    show_image("eda_distributions.png", "Numeric feature distributions by outcome")
    show_image("eda_correlation.png", "Correlation matrix")

    st.subheader("Model evaluation")
    a, b = st.columns(2)
    with a:
        show_image("roc_pr_curves.png", "ROC and precision-recall curves (test set)")
    with b:
        show_image("confusion_matrix.png", "Confusion matrix at the chosen threshold")
    show_image("threshold_analysis.png", "Threshold analysis (out-of-fold)")
    show_table("threshold_analysis.csv", "Operating points for the final model (test set)")
    show_table("model_comparison_test.csv", "Model comparison at each model's cost-optimal threshold (test set)")
    show_table("strategy_cv_comparison.csv", "Imbalance strategy comparison (5-fold CV on training split)")

    st.subheader("Explainability")
    a, b = st.columns(2)
    with a:
        show_image("shap_summary.png", "Global SHAP summary (beeswarm)")
        show_image("shap_waterfall_high_risk.png", "SHAP waterfall: highest-risk test applicant")
    with b:
        show_image("shap_importance_bar.png", "Mean |SHAP| importance")
        show_image("shap_waterfall_low_risk.png", "SHAP waterfall: lowest-risk test applicant")
    a, b = st.columns(2)
    with a:
        show_image("lime_approved.png", "LIME: approved applicant")
    with b:
        show_image("lime_rejected.png", "LIME: rejected applicant")
