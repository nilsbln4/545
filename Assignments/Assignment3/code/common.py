"""Shared helpers for the Assignment 3 scripts."""

import sys
from datetime import date
from pathlib import Path

import pandas as pd

# Lets the scripts import the course package without `pip install -e .`.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = DATA_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


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


# ----------------------------------------------------------------------------
# Problem 1 inputs (from the assignment text)
# ----------------------------------------------------------------------------
P1_SPOT = 182.40
P1_RATE = 0.0425           # risk free rate, continuously compounded
P1_DIV_YIELD = 0.0060      # continuously compounded dividend yield
P1_TODAY = date(2026, 11, 13)
P1_EXPIRY = date(2026, 12, 18)
# Calendar days over 365, as the problem specifies.
P1_T = (P1_EXPIRY - P1_TODAY).days / 365


def p1_chain() -> pd.DataFrame:
    """problem1.csv reshaped to one row per strike, with mids and spreads for
    the call and the put side by side."""
    q = pd.read_csv(DATA_DIR / "problem1.csv")
    q["Mid"] = (q["Bid"] + q["Ask"]) / 2
    q["Spread"] = q["Ask"] - q["Bid"]
    calls = q[q["Type"] == "Call"].set_index("Strike").drop(columns="Type").add_prefix("C_")
    puts = q[q["Type"] == "Put"].set_index("Strike").drop(columns="Type").add_prefix("P_")
    return calls.join(puts, how="outer").sort_index()


# ----------------------------------------------------------------------------
# Problem 2 inputs (from the assignment text)
# ----------------------------------------------------------------------------
P2_SPOT = 48.50
P2_STRIKE = 50.0
P2_T = 0.5
P2_RATE = 0.045
P2_VOL = 0.30
P2_DIV_AMOUNTS = [0.25, 1.50]
P2_DIV_TIMES = [0.10, 0.40]     # years; tree steps must fall on these, so N is a multiple of 5


# ----------------------------------------------------------------------------
# Problem 3 inputs (from the assignment text)
# ----------------------------------------------------------------------------
P3_RATE = 0.0425           # ABC pays no dividends, so b = r
P3_DAYS_PER_YEAR = 255     # trading days; DaysToExpiry is in trading days
P3_HORIZON = 10            # trading days


def p3_spot() -> float:
    """Last price of ABC in problem3.csv; every position is valued here."""
    return float(pd.read_csv(DATA_DIR / "problem3.csv")["ABC"].iloc[-1])


def p3_positions() -> pd.DataFrame:
    """problem3_portfolios.csv with DaysToExpiry / ImpliedVol / Strike as floats (NaN for stock)."""
    return pd.read_csv(DATA_DIR / "problem3_portfolios.csv")


# ----------------------------------------------------------------------------
# Problem 4 inputs (from the assignment text)
# ----------------------------------------------------------------------------
P4_RF_ANNUAL = 0.03
P4_RF_MONTHLY = P4_RF_ANNUAL / 12     # simple monthly rate; (1.03)^(1/12) - 1 = 0.002466 differs by 0.003%


def p4_returns() -> pd.DataFrame:
    """Monthly total returns of EQ, BOND, STK, PUTW, indexed by month end."""
    df = pd.read_csv(DATA_DIR / "problem4.csv", parse_dates=["Month"])
    return df.set_index("Month")


# ----------------------------------------------------------------------------
# Problem 5 inputs
# ----------------------------------------------------------------------------
P5_DAYS_PER_YEAR = 255     # the instructor annualizes daily data with 255 (Week09/week09.jl)
P5_FACTORS = ["Mkt-RF", "SMB", "HML", "Mom"]


def p5_factors() -> pd.DataFrame:
    """Fama-French 3 factors, RF and momentum, daily, as decimals (the files store
    percentages), indexed by date. Inner join of the two files."""
    ff = pd.read_csv(DATA_DIR / "F-F_Research_Data_Factors_daily.CSV")
    mom = pd.read_csv(DATA_DIR / "F-F_Momentum_Factor_daily.CSV")
    mom.columns = [c.strip() for c in mom.columns]          # header is "Mom   "
    df = ff.merge(mom, on="Date", how="inner")
    df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m%d")
    df = df.set_index("Date")
    return df[P5_FACTORS + ["RF"]] / 100


def p5_data() -> pd.DataFrame:
    """Fund returns joined to the factors on date; adds FUND1_ex and FUND2_ex (excess of RF)."""
    funds = pd.read_csv(DATA_DIR / "problem5.csv", parse_dates=["Date"]).set_index("Date")
    df = funds.join(p5_factors(), how="inner")
    for f in ("FUND1", "FUND2"):
        df[f"{f}_ex"] = df[f] - df["RF"]
    return df
