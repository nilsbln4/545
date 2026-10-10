"""Shared helpers for the Assignment 2 scripts."""

import sys
from pathlib import Path

# Lets the scripts import the course package without `pip install -e .`.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = DATA_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def section(title: str) -> None:
    bar = "=" * len(title)
    print(f"\n{title}\n{bar}")


class Report:
    """Prints lines and keeps them, so a script's output can be saved to
    output/<name>.txt for pasting into the write-up."""

    def __init__(self) -> None:
        self.lines: list = []

    def say(self, text: str = "") -> None:
        print(text)
        self.lines.append(text)

    def section(self, title: str) -> None:
        self.say(f"\n{title}\n{'=' * len(title)}")

    def save(self, name: str) -> None:
        path = DATA_DIR / "output" / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("\n".join(self.lines), encoding="utf-8")
        print(f"\nSaved to {path.relative_to(DATA_DIR)}")


def problem2_returns():
    """Problem 2's arithmetic returns with the sample mean removed (as the
    problem asks). Returns (returns, removed_mean)."""
    import pandas as pd
    import risk545 as rl

    prices = pd.read_csv(DATA_DIR / "problem2.csv")
    ret = rl.return_calculate(prices[["Day", "Price"]], method="DISCRETE",
                              date_column="Day")["Price"].to_numpy()
    return ret - ret.mean(), ret.mean()


def ew_weights(lam: float, n_obs: int):
    """Week 3's normalized exponential weights, newest observation heaviest."""
    import numpy as np

    w = (1 - lam) * lam ** np.arange(n_obs)[::-1]
    return w / w.sum()
