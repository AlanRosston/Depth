#!/usr/bin/env python3
"""Command line for multi-segment tasks: check / free / ladder / report (see depth/harness/run.py)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depth.harness import run as R  # noqa: E402


def gate_args(p):
    p.add_argument("--reserve", type=float, default=None,
                   help="stop starting new runs once the OpenRouter balance is below this many dollars")
    p.add_argument("--max-cost", type=float, default=None,
                   help="stop starting new runs once the cost recorded in this invocation reaches this many dollars")
    p.add_argument("--stop-file", default=None, help="stop starting new runs once this file exists")


def main():
    ap = argparse.ArgumentParser(description=R.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="oracle passes every segment, nop fails")
    c.add_argument("tasks", nargs="+")
    f = sub.add_parser("free", help=f"{R.K} runs without a cap")
    f.add_argument("tasks", nargs="+")
    f.add_argument("--model", default=None, help=f"default: {R.MODEL}")
    gate_args(f)
    lad = sub.add_parser("ladder", help="measure D with the model set " + ", ".join(R.MODELS))
    lad.add_argument("tasks", nargs="+")
    gate_args(lad)
    rp = sub.add_parser("report", help="D of measured tasks, or with --model the pass rate of that model")
    rp.add_argument("--model", default=None)
    a = ap.parse_args()
    {"check": R.cmd_check, "free": R.cmd_free, "ladder": R.cmd_ladder, "report": R.cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
