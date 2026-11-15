"""Runs every Assignment 3 script in order and reproduces every number and figure
in the write-up.

    python run_all.py

Takes about 5 minutes; nearly all of it is problem2_fit.py, whose non-recombining
dividend tree is large at the biggest N in the convergence study.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

SCRIPTS = [
    "problem1_predict.py", "problem1_fit.py", "problem1_reconcile.py",
    "problem2_predict.py", "problem2_fit.py", "problem2_reconcile.py",
    "problem3_predict.py", "problem3_fit.py", "problem3_reconcile.py",
    "problem4_predict.py", "problem4_fit.py", "problem4_reconcile.py",
    "problem5_predict.py", "problem5_fit.py", "problem5_reconcile.py",
]


def main() -> int:
    for script in SCRIPTS:
        print(f"\n{'#' * 12} {script} {'#' * 12}", flush=True)
        result = subprocess.run([sys.executable, str(HERE / script)], cwd=HERE)
        if result.returncode != 0:
            print(f"\n{script} failed with exit code {result.returncode}")
            return result.returncode
    print("\nAll scripts finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
