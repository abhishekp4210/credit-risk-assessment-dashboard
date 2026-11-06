"""Minimal self-check:  python test_basic.py   (needs the project's dependencies installed)."""
import numpy as np

from data_pipeline import clean, load_raw
from train import threshold_sweep, tune_threshold


def main() -> None:
    df, rep = clean(load_raw())
    assert df["person_age"].max() <= 100 and df["person_emp_length"].max() <= 100, rep
    assert rep["rows_clean"] == len(df) and 0.2 < rep["default_rate"] < 0.25, rep

    # toy check: perfect scores -> a mid threshold gives zero errors
    y = np.array([0, 0, 0, 1, 1]); p = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    s = threshold_sweep(y, p)
    assert s.loc[s.threshold.sub(0.5).abs().idxmin(), ["fp", "fn"]].sum() == 0
    thr, sweep = tune_threshold(y, p, cost_fn=5, cost_fp=1)
    assert 0.3 < thr <= 0.8 and sweep["cost"].min() == 0
    print("OK", rep)


if __name__ == "__main__":
    main()
