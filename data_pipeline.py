"""
data_pipeline.py
----------------
Data loading, cleaning, outlier handling, preprocessing pipeline and EDA plots
for `credit_risk_dataset.csv`.

Everything schema-related lives here so train.py and app.py share one source of truth.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

log = logging.getLogger(__name__)

DATA_PATH = Path("data/credit_risk_dataset.csv")
TARGET = "loan_status"
NUMERIC_COLS = [
    "person_age", "person_income", "person_emp_length", "loan_amnt",
    "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
]
CATEGORICAL_COLS = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]
FEATURES = NUMERIC_COLS + CATEGORICAL_COLS

# Allowed values (used for the app's dropdowns)
CATEGORY_LEVELS = {
    "person_home_ownership": ["RENT", "OWN", "MORTGAGE", "OTHER"],
    "loan_intent": ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
    "loan_grade": list("ABCDEFG"),
    "cb_person_default_on_file": ["N", "Y"],
}
# Physically impossible values (per the dataset spec)
MAX_AGE, MAX_EMP_LENGTH = 100, 100


# --------------------------------------------------------------------------- #
# Loading & cleaning
# --------------------------------------------------------------------------- #
def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce numerics (bad values -> NaN) and trim/upper-case categoricals (plain object dtype)."""
    df = df.copy()
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in CATEGORICAL_COLS:
        s = df[c].astype("object").where(df[c].notna(), None)
        df[c] = s.map(lambda v: str(v).strip().upper() if v is not None else np.nan).astype("object")
    return df


def load_raw(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Read the CSV and verify that every expected column exists."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at '{path}'. Put credit_risk_dataset.csv in data/ or pass --data.")
    df = pd.read_csv(path)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove impossible outliers, exact duplicates and rows without a valid target.

    Returns (clean_df, report). Missing values in features are NOT dropped - the
    preprocessing pipeline imputes them (so the same logic applies at inference).
    """
    report = {"rows_raw": int(len(df))}
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET].isin([0, 1])]
    report["dropped_bad_target"] = report["rows_raw"] - len(df)

    df = _normalise_columns(df)
    n = len(df)
    # NaN comparisons are False, so missing age/emp_length rows are kept (and imputed later)
    df = df[~(df["person_age"] > MAX_AGE) & ~(df["person_emp_length"] > MAX_EMP_LENGTH)]
    report["dropped_outliers"] = n - len(df)

    n = len(df)
    df = df.drop_duplicates()
    report["dropped_duplicates"] = n - len(df)

    df = df.reset_index(drop=True)
    df[TARGET] = df[TARGET].astype(int)
    report["rows_clean"] = int(len(df))
    report["default_rate"] = round(float(df[TARGET].mean()), 4)
    report["missing"] = {k: int(v) for k, v in df[FEATURES].isna().sum().items() if v > 0}
    return df, report


def load_clean(path: Path | str = DATA_PATH) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, dict]:
    """Load + clean. Returns (X, y, clean_df, report)."""
    df, report = clean(load_raw(path))
    log.info("Cleaning report: %s", report)
    return df[FEATURES], df[TARGET], df, report


# --------------------------------------------------------------------------- #
# Inference-time input handling (single form + batch upload)
# --------------------------------------------------------------------------- #
def prepare_input(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Validate/clean user-supplied rows for scoring. Returns (frame[FEATURES], warnings).

    - derives loan_percent_income from loan_amnt / person_income if absent
    - coerces numerics, normalises categorical text
    - impossible ages / employment lengths become NaN (imputed) instead of crashing
    Raises ValueError if required columns are missing.
    """
    df = df.copy()
    notes: list[str] = []
    if "loan_percent_income" not in df.columns and {"loan_amnt", "person_income"} <= set(df.columns):
        inc = pd.to_numeric(df["person_income"], errors="coerce").replace(0, np.nan)
        df["loan_percent_income"] = (pd.to_numeric(df["loan_amnt"], errors="coerce") / inc).round(2)
        notes.append("loan_percent_income was missing and has been derived as loan_amnt / person_income.")
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    out = _normalise_columns(df[FEATURES])
    bad = (out["person_age"] > MAX_AGE) | (out["person_emp_length"] > MAX_EMP_LENGTH)
    if bad.any():
        out.loc[out["person_age"] > MAX_AGE, "person_age"] = np.nan
        out.loc[out["person_emp_length"] > MAX_EMP_LENGTH, "person_emp_length"] = np.nan
        notes.append(f"{int(bad.sum())} row(s) had impossible age/employment values; they were treated as missing.")
    for c, levels in CATEGORY_LEVELS.items():
        unseen = sorted(set(out[c].dropna()) - set(levels))
        if unseen:
            notes.append(f"Unrecognised {c} values {unseen} are scored as 'unknown'.")
    return out, notes


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def build_preprocessor() -> ColumnTransformer:
    """Median imputation + RobustScaler (income/age have extreme values) for numerics;
    most-frequent imputation + one-hot (unknown-safe) for categoricals."""
    num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", RobustScaler())])
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("num", num, NUMERIC_COLS), ("cat", cat, CATEGORICAL_COLS)], sparse_threshold=0.0)


# --------------------------------------------------------------------------- #
# EDA
# --------------------------------------------------------------------------- #
def save_eda_plots(df: pd.DataFrame, out_dir: Path) -> None:
    """Save target/segment default rates, numeric distributions by class, and a correlation heatmap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    palette = {0: "#4C78A8", 1: "#E45756"}

    # 1) class balance + default rate by segment
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    ax = axes[0, 0]
    counts = df[TARGET].value_counts().sort_index()
    ax.bar(["Non-default", "Default"], counts.values, color=[palette[0], palette[1]])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}\n({v / len(df):.1%})", ha="center", va="bottom", fontsize=9)
    ax.set_title("Class balance")
    ax.set_ylim(0, counts.max() * 1.18)
    for ax, col in zip(axes.flat[1:5], CATEGORICAL_COLS):
        (df.groupby(col)[TARGET].mean().sort_values(ascending=False)).plot.bar(ax=ax, color="#F58518")
        ax.axhline(df[TARGET].mean(), color="k", ls="--", lw=0.8)
        ax.set_title(f"Default rate by {col}")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
    ax = axes[1, 2]
    ax.axis("off")
    ax.text(0, 0.9, f"Rows: {len(df):,}\nDefault rate: {df[TARGET].mean():.1%}\n"
                    f"Missing values:\n{df[FEATURES].isna().sum()[lambda s: s > 0].to_string()}",
            va="top", fontsize=11, family="monospace")
    fig.tight_layout()
    fig.savefig(out_dir / "eda_target.png", dpi=120)
    plt.close(fig)

    # 2) numeric distributions by class (clip display at 99th pct so outliers don't flatten plots)
    fig, axes = plt.subplots(2, 4, figsize=(17, 7))
    for ax, col in zip(axes.flat, NUMERIC_COLS):
        s = df[[col, TARGET]].dropna()
        s = s[s[col] <= s[col].quantile(0.99)]
        sns.histplot(data=s, x=col, hue=TARGET, palette=palette, stat="density", common_norm=False,
                     element="step", bins=40, ax=ax)
        ax.set_title(col)
        if ax.get_legend():
            ax.get_legend().set_title("default")
    axes.flat[-1].axis("off")
    fig.suptitle("Numeric feature distributions by outcome (clipped at 99th percentile)")
    fig.tight_layout()
    fig.savefig(out_dir / "eda_distributions.png", dpi=120)
    plt.close(fig)

    # 3) correlation heatmap
    corr = df[NUMERIC_COLS + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation matrix (numeric features + target)")
    fig.tight_layout()
    fig.savefig(out_dir / "eda_correlation.png", dpi=120)
    plt.close(fig)
