"""Where results are written. Everything goes under OUT (environment variable DEPTH_OUT, default <repo>/out)."""
import os
from pathlib import Path

from depth import REPO

OUT = Path(os.environ.get("DEPTH_OUT", REPO / "out"))
HARNESS_OUT = OUT / "harness"               # multi-segment tasks: runs/<model>/results.jsonl, check/, trials
AUDIT_OUT = OUT / "audit"                   # existing tasks: runs/<model>/results.jsonl, task copies, trials
