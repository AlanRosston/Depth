#!/usr/bin/env python3
"""Command line for measuring D on existing tasks: free / ladder / report (see depth/audit/run.py)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depth.audit import run as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=R.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, what in (("free", f"{R.K} runs without a cap"), ("ladder", "measure D with the model set " + ", ".join(R.MODELS))):
        p = sub.add_parser(name, help=what)
        p.add_argument("tasks", nargs="+", help="Harbor task directories (grouped by their parent directory)")
        if name == "free":
            p.add_argument("--model", default=None, help=f"default: {R.MODEL}")
        p.add_argument("--reserve", type=float, default=None,
                       help="stop starting new runs once the OpenRouter balance is below this many dollars")
        p.add_argument("--max-cost", type=float, default=None,
                       help="stop starting new runs once the cost recorded in this invocation reaches this many dollars")
        p.add_argument("--stop-file", default=None, help="stop starting new runs once this file exists")
    rp = sub.add_parser("report", help="N and D per task")
    rp.add_argument("tasks", nargs="*", help="restrict to these task directories")
    a = ap.parse_args()
    {"free": R.cmd_free, "ladder": R.cmd_ladder, "report": R.cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
