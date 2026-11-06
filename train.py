"""
train.py
--------
Model training, imbalance management, calibration, cost-sensitive threshold tuning,
SHAP / LIME explainability, and artifact export for `credit_risk_dataset.csv`.

Usage:
    python train.py                         # full run (several minutes)
    python train.py --data path/to.csv --cost-fn 5 --cost-fp 1 --cv 5

Outputs:
    models/credit_model.joblib   pipeline + calibrator + threshold + metadata (used by app.py)
    reports/                     plots (png), tables (csv) and metrics.json

Design notes
------------
* Imbalance: for each model we compare  none | class_weight (scale_pos_weight for XGBoost) |
  SMOTE | ADASYN  with stratified CV on the TRAIN split only, then pick the best by PR-AUC.
* Class weighting / resampling distort predicted probabilities, so each model's scores are
  Platt-calibrated on out-of-fold predictions. The "risk %" shown in the app is therefore a
  real probability, and the threshold is comparable across models.
* The decision threshold minimises  cost_fn*FN + cost_fp*FP  on out-of-fold predictions
  (no test leakage); the untouched test split is used only for reporting.
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, auc, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from xgboost import XGBClassifier

from data_pipeline import (
    CATEGORICAL_COLS, DATA_PATH, FEATURES, NUMERIC_COLS, build_preprocessor, load_clean, save_eda_plots,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

SEED = 42
MODEL_PATH = Path("models/credit_model.joblib")
REPORT_DIR = Path("reports")
MODELS = ["logreg", "random_forest", "xgboost"]
MODEL_LABELS = {"logreg": "Logistic Regression", "random_forest": "Random Forest", "xgboost": "XGBoost"}
STRATEGIES = ["none", "class_weight", "smote", "adasyn"]
# A missed default (FN) costs 5x a wrongly rejected good loan (FP). Override with --cost-fn/--cost-fp.
DEFAULT_COST_FN, DEFAULT_COST_FP = 5.0, 1.0
TARGET_RECALL = 0.90  # operating point "catch 90% of defaults"


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def build_model(name: str, strategy: str, y: pd.Series) -> Pipeline:
    """preprocessing -> optional resampler -> classifier.

    Resamplers act only during fit; predict/predict_proba on new data is unaffected.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}")
    weighted = strategy == "class_weight"
    cw = "balanced" if weighted else None
    neg_pos = float((y == 0).sum()) / max(float((y == 1).sum()), 1.0)

    if name == "logreg":
        clf = LogisticRegression(max_iter=3000, class_weight=cw, random_state=SEED)
    elif name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=250, min_samples_leaf=3, class_weight=cw, n_jobs=-1, random_state=SEED
        )
    elif name == "xgboost":
        clf = XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            min_child_weight=2, scale_pos_weight=neg_pos if weighted else 1.0,
            eval_metric="logloss", n_jobs=-1, random_state=SEED,
        )
    else:
        raise ValueError(f"Unknown model {name!r}")

    steps = [("prep", build_preprocessor())]
    # ponytail: SMOTE/ADASYN interpolate the one-hot columns too. SMOTENC is the stricter choice.
    if strategy == "smote":
        steps.append(("sampler", SMOTE(random_state=SEED)))
    elif strategy == "adasyn":
        steps.append(("sampler", ADASYN(random_state=SEED)))
    steps.append(("clf", clf))
    return Pipeline(steps)


def compare_strategies(X: pd.DataFrame, y: pd.Series, cv_splits: int) -> pd.DataFrame:
    """Stratified-CV comparison of every model x imbalance strategy (train split only)."""
    cv = StratifiedKFold(cv_splits, shuffle=True, random_state=SEED)
    rows = []
    for m in MODELS:
        for s in STRATEGIES:
            try:
                res = cross_validate(
                    build_model(m, s, y), X, y, cv=cv, n_jobs=1,
                    scoring={"pr_auc": "average_precision", "roc_auc": "roc_auc", "recall": "recall", "f1": "f1"},
                )
                row = {"model": m, "strategy": s, "pr_auc": res["test_pr_auc"].mean(),
                       "roc_auc": res["test_roc_auc"].mean(), "recall@0.5": res["test_recall"].mean(),
                       "f1@0.5": res["test_f1"].mean()}
            except Exception as e:  # e.g. ADASYN can fail to generate samples; don't kill the whole run
                log.warning("CV failed for %s + %s: %s", m, s, e)
                row = {"model": m, "strategy": s, "pr_auc": np.nan, "roc_auc": np.nan,
                       "recall@0.5": np.nan, "f1@0.5": np.nan}
            rows.append(row)
            log.info("CV %-13s %-12s PR-AUC=%.4f ROC-AUC=%.4f", m, s, row["pr_auc"], row["roc_auc"])
    return pd.DataFrame(rows).round(4)


# --------------------------------------------------------------------------- #
# Calibration, thresholds, metrics
# --------------------------------------------------------------------------- #
def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_calibrator(p_oof: np.ndarray, y: pd.Series) -> LogisticRegression:
    """Platt scaling (monotone, so ranking metrics such as AUC are unchanged)."""
    return LogisticRegression(C=1e4, max_iter=1000).fit(_logit(p_oof), y)


def calibrate(cal: LogisticRegression, p_raw: np.ndarray) -> np.ndarray:
    return cal.predict_proba(_logit(p_raw))[:, 1]


def predict_risk(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    """Calibrated default probability for rows already in FEATURES format."""
    return calibrate(bundle["calibrator"], bundle["pipeline"].predict_proba(X[FEATURES])[:, 1])


def risk_category(p: float, thr: float) -> str:
    """Low < thr/2 <= Medium < 2*thr <= High  (bands scale with the cost-optimal threshold)."""
    return "Low" if p < 0.5 * thr else ("Medium" if p < 2.0 * thr else "High")


def threshold_sweep(y_true, p, grid: np.ndarray | None = None) -> pd.DataFrame:
    """Confusion counts + precision/recall/F1 for every threshold on the grid (vectorised)."""
    y = np.asarray(y_true).astype(bool)
    p = np.asarray(p)
    grid = np.linspace(0.01, 0.99, 99) if grid is None else np.asarray(grid)
    pred = p[None, :] >= grid[:, None]
    tp = (pred & y).sum(1)
    fp = (pred & ~y).sum(1)
    fn = (~pred & y).sum(1)
    tn = (~pred & ~y).sum(1)
    prec = np.divide(tp, tp + fp, out=np.zeros(len(grid)), where=(tp + fp) > 0)
    rec = tp / max(y.sum(), 1)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(len(grid)), where=(prec + rec) > 0)
    return pd.DataFrame({"threshold": grid, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                         "precision": prec, "recall": rec, "f1": f1})


def tune_threshold(y_true, p, cost_fn: float, cost_fp: float) -> tuple[float, pd.DataFrame]:
    """Threshold minimising cost_fn*FN + cost_fp*FP. Returns (threshold, sweep with cost column)."""
    sweep = threshold_sweep(y_true, p)
    sweep["cost"] = cost_fn * sweep["fn"] + cost_fp * sweep["fp"]
    return float(sweep.loc[sweep["cost"].idxmin(), "threshold"]), sweep


def operating_points(sweep: pd.DataFrame) -> dict[str, float]:
    """Named thresholds chosen from an (out-of-fold) sweep that already has a `cost` column."""
    ok = sweep[sweep["recall"] >= TARGET_RECALL]
    return {
        "Default (0.50)": 0.5,
        "Max F1": float(sweep.loc[sweep["f1"].idxmax(), "threshold"]),
        f"Recall >= {TARGET_RECALL:.0%}": float(ok["threshold"].max()) if len(ok) else float(sweep["threshold"].min()),
        "Min cost (chosen)": float(sweep.loc[sweep["cost"].idxmin(), "threshold"]),
    }


def metrics_at(y_true, p, thr: float, cost_fn: float = DEFAULT_COST_FN, cost_fp: float = DEFAULT_COST_FP) -> dict:
    """AUCs plus thresholded precision / recall / F1, confusion counts and total cost."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(p) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(thr), 3),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, p),
        "roc_auc": roc_auc_score(y_true, p),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "cost": float(cost_fn * fn + cost_fp * fp),
    }


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def save_eval_plots(y_test, probs: dict[str, np.ndarray], final: str, thr: float, out: Path) -> None:
    """ROC + PR curves for all models, and the final model's confusion matrix at the tuned threshold."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        lw = 2.5 if name == final else 1.2
        ax[0].plot(fpr, tpr, lw=lw, label=f"{MODEL_LABELS[name]} (AUC={auc(fpr, tpr):.3f})")
        pr, rc, _ = precision_recall_curve(y_test, p)
        ax[1].plot(rc, pr, lw=lw, label=f"{MODEL_LABELS[name]} (AP={average_precision_score(y_test, p):.3f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (test set)")
    ax[1].axhline(np.mean(y_test), color="k", ls="--", lw=0.8)
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve (test set)")
    for a in ax:
        a.legend(loc="best", fontsize=8)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "roc_pr_curves.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, (probs[final] >= thr).astype(int), display_labels=["Non-default", "Default"],
        cmap="Blues", ax=ax, colorbar=False,
    )
    ax.set_title(f"{MODEL_LABELS[final]} - confusion matrix\n(test set, threshold {thr:.2f})")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=130)
    plt.close(fig)


def save_threshold_plot(sweep: pd.DataFrame, points: dict[str, float], out: Path) -> None:
    """Precision/recall/F1 and total cost vs. threshold, with the named operating points marked."""
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for col in ("precision", "recall", "f1"):
        ax[0].plot(sweep["threshold"], sweep[col], label=col)
    ax[1].plot(sweep["threshold"], sweep["cost"], color="#E45756")
    colors = ["gray", "#59A14F", "#F28E2B", "#B07AA1"]
    for (label, t), c in zip(points.items(), colors):
        for a in ax:
            a.axvline(t, ls="--", color=c, lw=1.2, label=f"{label}: {t:.2f}" if a is ax[0] else None)
    ax[0].set(xlabel="Decision threshold", ylabel="Score", title="Precision / recall / F1 vs. threshold (out-of-fold)")
    ax[1].set(xlabel="Decision threshold", ylabel="Total cost (FN & FP weighted)", title="Cost vs. threshold")
    ax[0].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "threshold_analysis.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Explainability: SHAP
# --------------------------------------------------------------------------- #
def feature_names_of(pipe: Pipeline) -> list[str]:
    """Readable transformed-feature names (strip the 'num__' / 'cat__' prefixes)."""
    return [n.split("__", 1)[-1] for n in pipe.named_steps["prep"].get_feature_names_out()]


def make_shap_explainer(pipe: Pipeline, background: np.ndarray):
    """TreeExplainer for tree models, LinearExplainer for logistic regression."""
    import shap

    clf = pipe.named_steps["clf"]
    if isinstance(clf, LogisticRegression):
        return shap.LinearExplainer(clf, background)
    return shap.TreeExplainer(clf)


def shap_explain(explainer, pipe: Pipeline, X_raw: pd.DataFrame):
    """shap.Explanation (default class) for raw rows. `data` is shown on the ORIGINAL scale
    for numeric features (readable in plots); `values` are in the model's margin units
    (log-odds for XGBoost / logistic regression, probability for random forest)."""
    import shap

    prep = pipe.named_steps["prep"]
    Xt = np.asarray(prep.transform(X_raw[FEATURES]), dtype=float)
    ex = explainer(Xt, check_additivity=False) if explainer.__class__.__name__ == "TreeExplainer" else explainer(Xt)
    vals, base = np.asarray(ex.values), np.asarray(ex.base_values)
    if vals.ndim == 3:  # random forest -> (n, features, classes); keep the default class
        vals, base = vals[:, :, 1], base[:, 1]
    data = Xt.copy()
    n_num = len(NUMERIC_COLS)
    data[:, :n_num] = prep.named_transformers_["num"].named_steps["scale"].inverse_transform(Xt[:, :n_num])
    return shap.Explanation(values=vals, base_values=base, data=data, feature_names=feature_names_of(pipe))


def grouped_contributions(explainer, pipe: Pipeline, row: pd.DataFrame) -> pd.DataFrame:
    """One row of SHAP values summed back to the 11 original features (one-hot columns grouped).
    Returns columns: feature, value, shap  sorted by |shap| descending."""
    ex = shap_explain(explainer, pipe, row)
    names, vals = list(ex.feature_names), ex.values[0]
    out = []
    for c in FEATURES:
        idx = [i for i, n in enumerate(names) if n == c or n.startswith(c + "_")]
        out.append({"feature": c, "value": row.iloc[0][c], "shap": float(vals[idx].sum())})
    return pd.DataFrame(out).sort_values("shap", key=np.abs, ascending=False).reset_index(drop=True)


def save_shap_plots(explainer, bundle: dict, X_test: pd.DataFrame, out: Path, n_summary: int = 1000) -> None:
    """Global beeswarm + bar importance, and waterfall plots for the highest- and lowest-risk test applicants."""
    import shap

    pipe = bundle["pipeline"]
    sample = X_test.sample(min(n_summary, len(X_test)), random_state=SEED)
    ex = shap_explain(explainer, pipe, sample)

    plt.figure()
    shap.summary_plot(ex.values, ex.data, feature_names=ex.feature_names, show=False, max_display=15)
    plt.title("Global SHAP summary (beeswarm)")
    plt.savefig(out / "shap_summary.png", dpi=130, bbox_inches="tight")
    plt.close("all")

    plt.figure()
    shap.summary_plot(ex.values, ex.data, feature_names=ex.feature_names, plot_type="bar", show=False, max_display=15)
    plt.title("Mean |SHAP| feature importance")
    plt.savefig(out / "shap_importance_bar.png", dpi=130, bbox_inches="tight")
    plt.close("all")

    p = predict_risk(bundle, X_test)
    for label, idx in (("high_risk", int(np.argmax(p))), ("low_risk", int(np.argmin(p)))):
        one = shap_explain(explainer, pipe, X_test.iloc[[idx]])
        plt.figure()
        shap.plots.waterfall(one[0], max_display=12, show=False)
        plt.title(f"SHAP waterfall - {label.replace('_', ' ')} test applicant (risk {p[idx]:.1%})", fontsize=10)
        plt.savefig(out / f"shap_waterfall_{label}.png", dpi=130, bbox_inches="tight")
        plt.close("all")


# --------------------------------------------------------------------------- #
# Explainability: LIME (on RAW features so explanations read naturally)
# --------------------------------------------------------------------------- #
def build_lime(pipe: Pipeline, calibrator: LogisticRegression, X_ref: pd.DataFrame):
    """Return (explainer, encode_fn, predict_fn) for LIME; probabilities are the calibrated ones."""
    from lime.lime_tabular import LimeTabularExplainer

    cols = list(X_ref.columns)
    fill = {c: X_ref[c].median() for c in NUMERIC_COLS}
    fill.update({c: X_ref[c].mode()[0] for c in CATEGORICAL_COLS})
    ref = X_ref.fillna(fill)
    cats = {c: sorted(ref[c].unique()) for c in CATEGORICAL_COLS}

    def encode(df: pd.DataFrame) -> np.ndarray:
        out = df[cols].fillna(fill).copy()
        for c in CATEGORICAL_COLS:
            out[c] = out[c].map({v: i for i, v in enumerate(cats[c])})
        return out.to_numpy(dtype=float)

    def predict_fn(arr: np.ndarray) -> np.ndarray:
        df = pd.DataFrame(arr, columns=cols)
        for c in CATEGORICAL_COLS:
            df[c] = df[c].round().astype(int).map(dict(enumerate(cats[c])))
        p = calibrate(calibrator, pipe.predict_proba(df)[:, 1])
        return np.column_stack([1 - p, p])

    explainer = LimeTabularExplainer(
        encode(ref), feature_names=cols, class_names=["Non-default", "Default"],
        categorical_features=[cols.index(c) for c in CATEGORICAL_COLS],
        categorical_names={cols.index(c): cats[c] for c in CATEGORICAL_COLS},
        mode="classification", random_state=SEED,
    )
    return explainer, encode, predict_fn


def lime_explain(lime_tools, row: pd.DataFrame, num_features: int = 10, num_samples: int = 3000):
    """LIME explanation object for the first row of `row` (default class)."""
    explainer, encode, predict_fn = lime_tools
    return explainer.explain_instance(encode(row)[0], predict_fn, num_features=num_features, num_samples=num_samples)


def save_lime_plots(bundle: dict, X_test: pd.DataFrame, out: Path) -> None:
    """LIME plots for one clearly-approved and one clearly-rejected test applicant."""
    tools = build_lime(bundle["pipeline"], bundle["calibrator"], bundle["lime_ref"])
    thr = bundle["threshold"]
    p = predict_risk(bundle, X_test)
    groups = {"approved": p < thr, "rejected": p >= thr}
    for label, mask in groups.items():
        if not mask.any():
            log.warning("No %s applicants in the test set; skipping LIME plot", label)
            continue
        q = np.quantile(p[mask], 0.10 if label == "approved" else 0.90)
        i = int(np.argmin(np.where(mask, np.abs(p - q), np.inf)))
        exp = lime_explain(tools, X_test.iloc[[i]])
        fig = exp.as_pyplot_figure()
        fig.set_size_inches(9, 5)
        fig.suptitle(f"LIME - {label.upper()} applicant (risk {p[i]:.1%}, threshold {thr:.0%})", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"lime_{label}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        log.info("LIME %s: test row %d, risk=%.3f", label, i, p[i])


# --------------------------------------------------------------------------- #
# Artifact loading (used by app.py)
# --------------------------------------------------------------------------- #
def load_bundle(path: Path | str = MODEL_PATH) -> dict:
    """Load the saved model bundle; raises a helpful error if training has not been run."""
    if not Path(path).exists():
        raise FileNotFoundError(f"'{path}' not found - run `python train.py` first.")
    return joblib.load(path)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _fmt(d: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train & evaluate credit-risk models")
    ap.add_argument("--data", type=Path, default=DATA_PATH, help="path to credit_risk_dataset.csv")
    ap.add_argument("--cost-fn", type=float, default=DEFAULT_COST_FN, help="cost of a missed default (FN)")
    ap.add_argument("--cost-fp", type=float, default=DEFAULT_COST_FP, help="cost of a wrongly rejected good loan (FP)")
    ap.add_argument("--cv", type=int, default=5, help="CV folds")
    ap.add_argument("--skip-explain", action="store_true", help="skip SHAP/LIME plots")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)

    REPORT_DIR.mkdir(exist_ok=True)
    MODEL_PATH.parent.mkdir(exist_ok=True)

    # ---- data -------------------------------------------------------------
    X, y, df_clean, clean_report = load_clean(args.data)
    save_eda_plots(df_clean, REPORT_DIR)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    log.info("Train=%d Test=%d | default rate train=%.3f test=%.3f", len(X_tr), len(X_te), y_tr.mean(), y_te.mean())

    # ---- Step 2: imbalance strategies x models (CV, train split only) -------
    cv_table = compare_strategies(X_tr, y_tr, args.cv)
    cv_table.to_csv(REPORT_DIR / "strategy_cv_comparison.csv", index=False)
    print("\n=== CV comparison: model x imbalance strategy (train split) ===")
    print(cv_table.to_string(index=False))

    # ---- Step 3: per model -> best strategy, OOF calibration, threshold, test metrics
    cv = StratifiedKFold(args.cv, shuffle=True, random_state=SEED)
    valid = cv_table.dropna(subset=["pr_auc"])
    best = valid.sort_values("pr_auc", ascending=False).groupby("model").head(1).set_index("model")
    res: dict[str, dict] = {}
    for m in MODELS:
        strat = best.loc[m, "strategy"]
        log.info("Fitting %s + %s ...", m, strat)
        oof_raw = cross_val_predict(build_model(m, strat, y_tr), X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        cal = fit_calibrator(oof_raw, y_tr)
        oof = calibrate(cal, oof_raw)
        thr, sweep = tune_threshold(y_tr, oof, args.cost_fn, args.cost_fp)
        pipe = build_model(m, strat, y_tr).fit(X_tr, y_tr)
        p_te = calibrate(cal, pipe.predict_proba(X_te)[:, 1])
        res[m] = {"strategy": strat, "pipe": pipe, "cal": cal, "thr": thr, "sweep": sweep, "p_te": p_te,
                  "cv_pr_auc": float(best.loc[m, "pr_auc"])}

    comp = pd.DataFrame([
        {"model": MODEL_LABELS[m], "strategy": r["strategy"], **metrics_at(y_te, r["p_te"], r["thr"], args.cost_fn, args.cost_fp)}
        for m, r in res.items()
    ]).round(4)
    comp.to_csv(REPORT_DIR / "model_comparison_test.csv", index=False)
    print("\n=== Test-set comparison at each model's cost-optimal threshold ===")
    print(comp[["model", "strategy", "threshold", "precision", "recall", "f1", "pr_auc", "roc_auc", "fn", "fp", "cost"]]
          .to_string(index=False))

    # final model: best CV PR-AUC (selected without looking at the test set)
    final = max(res, key=lambda m: res[m]["cv_pr_auc"])
    R = res[final]
    log.info("Final model: %s + %s (threshold %.2f)", final, R["strategy"], R["thr"])

    # ---- threshold analysis for the final model (chosen on OOF, reported on test)
    points = operating_points(R["sweep"])
    thr_table = pd.DataFrame([
        {"operating_point": name, **metrics_at(y_te, R["p_te"], t, args.cost_fn, args.cost_fp)}
        for name, t in points.items()
    ]).round(4)
    thr_table.to_csv(REPORT_DIR / "threshold_analysis.csv", index=False)
    print(f"\n=== Threshold analysis: {MODEL_LABELS[final]} (cost FN={args.cost_fn:g}, FP={args.cost_fp:g}) ===")
    print(thr_table[["operating_point", "threshold", "precision", "recall", "f1", "fn", "fp", "cost"]].to_string(index=False))

    save_eval_plots(y_te, {m: r["p_te"] for m, r in res.items()}, final, R["thr"], REPORT_DIR)
    save_threshold_plot(R["sweep"], points, REPORT_DIR)

    # ---- bundle ------------------------------------------------------------
    pipe = R["pipe"]
    background = np.asarray(pipe.named_steps["prep"].transform(X_tr.sample(200, random_state=SEED)), dtype=float)
    bundle = {
        "pipeline": pipe, "calibrator": R["cal"], "threshold": R["thr"], "model_name": final,
        "strategy": R["strategy"], "cost_fn": args.cost_fn, "cost_fp": args.cost_fp,
        "background": background, "lime_ref": X_tr.sample(min(3000, len(X_tr)), random_state=SEED).reset_index(drop=True),
        "operating_points": points, "clean_report": clean_report,
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__},
    }
    joblib.dump(bundle, MODEL_PATH)
    (REPORT_DIR / "metrics.json").write_text(json.dumps({
        "final_model": final, "strategy": R["strategy"], "threshold": R["thr"],
        "cost_fn": args.cost_fn, "cost_fp": args.cost_fp,
        "test_at_chosen_threshold": metrics_at(y_te, R["p_te"], R["thr"], args.cost_fn, args.cost_fp),
        "clean_report": clean_report,
    }, indent=2, default=float))
    log.info("Saved model -> %s", MODEL_PATH)

    # ---- Step 4: explainability -------------------------------------------
    if not args.skip_explain:
        explainer = make_shap_explainer(pipe, background)
        save_shap_plots(explainer, bundle, X_te, REPORT_DIR)
        save_lime_plots(bundle, X_te, REPORT_DIR)
    log.info("Done. Reports in %s/. Launch the app with: streamlit run app.py", REPORT_DIR)


if __name__ == "__main__":
    main()
