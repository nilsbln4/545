"""Problem 1 -- Predict step: how much data sits behind each correlation.

Builds the table of jointly observed days for every pair of series, and counts
the days on which all five trade. Nothing is fitted here; this is the
descriptive input the Predict questions (a)-(c) ask you to reason from.
"""

import pandas as pd

from common import DATA_DIR, section


def main() -> None:
    df = pd.read_csv(DATA_DIR / "problem1.csv").drop(columns="Day")
    observed = df.notna().astype(int)

    # Entry (i, j) = number of days both series i and j are observed.
    # The diagonal is the number of days each series is observed on its own.
    joint_days = observed.T @ observed

    section("Days each pair of series is jointly observed (diagonal = days observed)")
    print(joint_days.to_string())

    n_all = int(observed.all(axis=1).sum())
    section("Days on which all five series trade (complete rows)")
    print(f"{n_all} of {len(df)} days")

    section("Missing days per series")
    print((len(df) - observed.sum()).to_string())


if __name__ == "__main__":
    main()
