"""Run every functional test (1.1 through 7.6) and print one PASS/FAIL line each.

    python functional_tests/run_all.py

Each test's computed output is written to functional_tests/outputs/ under the
reference file's name, so it can be diffed against testfiles/data/. Exit code
is 0 only if every test passes.
"""

import importlib
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _order(path: Path):
    return tuple(int(n) for n in re.findall(r"\d+", path.stem))


def main() -> int:
    scripts = sorted(HERE.glob("test_*.py"), key=_order)
    results = [importlib.import_module(s.stem).main() for s in scripts]

    passed = sum(results)
    print(f"\n{passed}/{len(results)} functional tests passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
