"""Runs the midterm scripts in order.

    python run_all.py

problem3b.py and problem3c.py stop with a message until the choices in
settings.py are filled in.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = ["problem1.py", "problem2.py", "problem3a.py", "problem3b.py", "problem3c.py"]

for script in SCRIPTS:
    print(f"\n{'#' * 12} {script} {'#' * 12}")
    result = subprocess.run([sys.executable, str(HERE / script)], cwd=HERE)
    if result.returncode != 0:
        print(f"\n{script} stopped with exit code {result.returncode}")
        break
