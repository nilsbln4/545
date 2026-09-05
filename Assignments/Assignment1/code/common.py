"""Shared helpers used by more than one problem script."""

from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = DATA_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def aicc(loglik: float, k: int, n: int) -> float:
    """Corrected AIC. k = number of estimated parameters, n = sample size.

    Undefined once n <= k + 1; that only happens with far more parameters
    than data, which none of these fits approach.
    """
    aic = 2 * k - 2 * loglik
    return aic + (2 * k * (k + 1)) / (n - k - 1)


def section(title: str) -> None:
    bar = "=" * len(title)
    print(f"\n{title}\n{bar}")
