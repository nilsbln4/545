"""Runs all five problems in order and writes their combined console output
to ../output/results.txt, in addition to printing it."""

import contextlib
import io
from pathlib import Path

import problem1, problem2, problem3, problem4, problem5

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    buf = io.StringIO()
    tee = _Tee(buf)
    with contextlib.redirect_stdout(tee):
        for name, mod in [("Problem 1", problem1), ("Problem 2", problem2),
                           ("Problem 3", problem3), ("Problem 4", problem4),
                           ("Problem 5", problem5)]:
            print(f"\n\n########## {name} ##########")
            mod.main()

    out_path = OUTPUT_DIR / "results.txt"
    out_path.write_text(buf.getvalue())
    print(f"\nAll output also written to {out_path.relative_to(OUTPUT_DIR.parent)}")


class _Tee(io.TextIOBase):
    """Writes to both the buffer and the real stdout."""

    def __init__(self, buf: io.StringIO):
        self.buf = buf
        self.stdout = __import__("sys").stdout

    def write(self, s: str) -> int:
        self.buf.write(s)
        self.stdout.write(s)
        return len(s)


if __name__ == "__main__":
    main()
