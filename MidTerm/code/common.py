"""Shared helpers for the midterm scripts."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Lets the scripts import the course package without `pip install -e .`.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import risk545 as rl  # noqa: E402

MIDTERM_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = MIDTERM_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


class Report:
    """Prints lines and keeps them, so the output can be saved to output/<name>."""

    def __init__(self) -> None:
        self.lines: list = []

    def say(self, text: str = "") -> None:
        print(text)
        self.lines.append(text)

    def section(self, title: str) -> None:
        self.say(f"\n{title}\n{'=' * len(title)}")

    def save(self, name: str) -> None:
        path = OUTPUT_DIR / name
        path.write_text("\n".join(self.lines), encoding="utf-8")
        print(f"\nSaved to {path.relative_to(MIDTERM_DIR)}")


# ----------------------------------------------------------------------------
# Problem 3 helpers (used by problem3a/3b/3c)
# ----------------------------------------------------------------------------

def load_q3_returns() -> pd.DataFrame:
    """Arithmetic returns of S1..S10 (optionally de-meaned, see settings.py)."""
    import settings

    prices = pd.read_csv(MIDTERM_DIR / "problem3.csv")
    ret = rl.return_calculate(prices, method="DISCRETE", date_column="Date")
    stocks = [c for c in ret.columns if c != "Date"]
    out = ret[stocks].copy()
    if settings.DEMEAN_Q3:
        out = out - out.mean()
    return out


def load_q3_positions() -> pd.DataFrame:
    """Holdings valued at the last price in the file: Stock, Book, Holding, Price, Value."""
    prices = pd.read_csv(MIDTERM_DIR / "problem3.csv")
    holdings = pd.read_csv(MIDTERM_DIR / "problem3_holdings.csv")
    last = prices.iloc[-1]
    holdings["Price"] = holdings["Stock"].map(lambda s: float(last[s]))
    holdings["Value"] = holdings["Holding"] * holdings["Price"]
    return holdings


def fit_margin(x: np.ndarray, kind: str):
    """Fit one margin: kind is "normal", "t" or "nig"."""
    if kind == "normal":
        return rl.fit_normal(x)
    if kind == "t":
        return rl.fit_general_t(x)
    if kind == "nig":
        return rl.fit_nig_mle(x)
    raise ValueError(f"unknown distribution {kind!r}: use 'normal', 't' or 'nig'")


def require_choices(need_copula: bool = False) -> dict:
    """Return the per-stock distribution choices from settings.py, or stop if
    they have not been filled in."""
    import settings

    missing = [s for s, k in settings.DISTRIBUTION_CHOICE.items() if k is None]
    if missing:
        raise SystemExit(f"Fill in DISTRIBUTION_CHOICE for {', '.join(missing)} in settings.py first.")
    if need_copula and settings.COPULA_CHOICE not in ("gaussian", "t"):
        raise SystemExit("Set COPULA_CHOICE to 'gaussian' or 't' in settings.py first.")
    return settings.DISTRIBUTION_CHOICE


def margin_quantile(fit, kind: str, u: np.ndarray) -> np.ndarray:
    """Map uniforms to returns through a fitted margin's inverse CDF."""
    if kind != "nig":
        return fit.eval(u)
    # The NIG quantile has no closed form and scipy inverts it numerically one
    # draw at a time, which is slow for tens of thousands of draws. Evaluate it
    # once on a grid (dense in both tails) and interpolate.
    grid = np.concatenate([np.linspace(1e-6, 1e-3, 100),
                           np.linspace(1e-3, 1 - 1e-3, 1000)[1:],
                           np.linspace(1 - 1e-3, 1 - 1e-6, 100)[1:]])
    return np.interp(u, grid, fit.error_model.ppf(grid))
