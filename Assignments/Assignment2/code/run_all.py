"""Runs every Assignment 2 script in order, with the settings used in the write-up.

    python run_all.py

The scripts that take --recent-days use 30: the last 30 returns of Problem 2
look different from the rest (part 2b), and that split drives parts 2b and 2f.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

RUNS = [
    ("problem1_counts.py", []),
    ("problem1_fit.py", []),
    ("problem2_predict.py", ["--recent-days", "30"]),
    ("problem2_fit.py", ["--recent-days", "30"]),
    ("problem3_predict.py", []),
    ("problem3_fit.py", []),
    ("problem4_predict.py", []),
    ("problem4_fit.py", []),
    ("problem5_fit.py", []),
]


def main() -> int:
    for script, args in RUNS:
        print(f"\n{'#' * 12} {script} {' '.join(args)} {'#' * 12}")
        result = subprocess.run([sys.executable, str(HERE / script), *args], cwd=HERE)
        if result.returncode != 0:
            print(f"\n{script} failed with exit code {result.returncode}")
            return result.returncode
    print("\nAll scripts finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
