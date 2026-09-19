"""Shared helpers for the per-test scripts in this folder.

Each test_N_M.py computes one functional test's result with the risk545
package, writes it to outputs/ under the reference file's name, and compares
it against the reference in testfiles/data/. Importing this module also puts
the repo root on sys.path, so the scripts run without `pip install -e .`.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA = REPO_ROOT / "testfiles" / "data"
OUTPUT = Path(__file__).resolve().parent / "outputs"

# Deterministic linear algebra: should match the reference near machine precision.
TIGHT = dict(rtol=1e-6, atol=1e-8)
# Numerical optimizers (scipy here, Ipopt in the Julia reference): agree to 6+ figures.
MLE = dict(rtol=1e-4, atol=1e-6)
# Monte Carlo with 100,000 draws: sampling noise, so a looser bar.
MONTE_CARLO = dict(rtol=0.08, atol=0.0015)


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name)


def matrix(name: str) -> np.ndarray:
    return load(name).to_numpy(dtype=float)


def matrix_df(m: np.ndarray) -> pd.DataFrame:
    """Wrap a matrix with the x1..xn column names the reference files use."""
    return pd.DataFrame(m, columns=[f"x{i + 1}" for i in range(m.shape[1])])


def run(test_id: str, description: str, output: pd.DataFrame, out_name: str,
        tol: dict, reference=None) -> bool:
    """Write `output` to outputs/<out_name>, compare it to the reference, and
    print one PASS/FAIL line.

    `reference` defaults to the file of the same name in testfiles/data/. Pass
    an array instead when the right target is not that file (test 5 checks the
    simulated covariance against the matrix it was drawn from).
    """
    OUTPUT.mkdir(exist_ok=True)
    output.to_csv(OUTPUT / out_name, index=False)

    if reference is None:
        reference = load(out_name)
    if isinstance(reference, pd.DataFrame):
        reference = reference.select_dtypes("number").to_numpy(dtype=float)
    actual = output.select_dtypes("number").to_numpy(dtype=float)

    try:
        np.testing.assert_allclose(actual, reference, **tol)
        status, detail = "PASS", ""
    except AssertionError as e:
        status = "FAIL"
        detail = "\n      " + "\n      ".join(str(e).strip().splitlines()[:6])

    print(f"{status}  {test_id:<4} {description}  ->  outputs/{out_name}{detail}")
    return status == "PASS"
