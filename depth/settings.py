"""Runtime settings, overridable by environment variables only.

Models are called through OpenRouter; tasks run in local Docker through Harbor.
"""
import os
from pathlib import Path

# ---- model API ---------------------------------------------------------------
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
BASE_URL = os.environ.get("DEPTH_BASE_URL", "https://openrouter.ai/api/v1")


def agent_env():
    """Environment handed to Harbor: the agent calls the model through litellm, which reads OPENROUTER_API_KEY."""
    return {"OPENROUTER_API_KEY": API_KEY}


# ---- Harbor ------------------------------------------------------------------
HARBOR = os.environ.get("DEPTH_HARBOR", str(Path.home() / ".local" / "bin" / "harbor"))

# ---- sandbox -----------------------------------------------------------------
ORACLE_TIMEOUT = int(os.environ.get("DEPTH_ORACLE_TIMEOUT", "2400"))  # wall-clock limit (s) of one Harbor trial when the caller gives none
WORKERS = int(os.environ.get("DEPTH_WORKERS", "3"))                   # Harbor trials running at once
MIN_FREE_GB = 12                # do not start a new trial while free disk space is below this
MIN_MEM_MB = int(os.environ.get("DEPTH_MIN_MEM_MB", "3000"))          # ... or while available memory is below this
START_GAP = int(os.environ.get("DEPTH_START_GAP", "15"))              # seconds between starting two Harbor processes
MAX_LOAD5 = float(os.environ.get("DEPTH_MAX_LOAD5", "0"))             # if > 0, wait until the 5-minute load average is below it
START_LOCK = Path(os.environ.get("DEPTH_START_LOCK", "/tmp/depth_harbor_start.lock"))   # serializes starts across processes
