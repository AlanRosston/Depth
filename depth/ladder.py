"""Result files, the next cap to run, and the depth of a task under the model set.

Rows (one line of results.jsonl per run) have at least `tier` (the cap; 0 = no cap), `status` and `n_episodes`.
The functions take rows already filtered to measured runs.

Protocol (constants in depth.protocol):
  1. Runs without a cap (`free`) are separate from measuring D (`ladder`). The first model of the set runs every task
     K times without a cap. If no run passes, the task is not solved and gives no estimate. Otherwise N is the fewest
     turns among its passing runs, and caps at or above N count as passed without being run.
  2. Model 0 tries the caps below N in increasing order, up to K runs each, the second run skipped once the first passes.
     Its estimate D_0 is the first cap with a passing run; if none passes, the smallest cap at or above N, or ABOVE
     if N exceeds the largest cap.
  3. Each later model i runs no task without a cap. It tries only the caps below the smallest estimate of models
     0..i-1, in the same order with the same number of runs, and its estimate is the first cap it passes.
  4. D = the smallest estimate over the models that give one (set_depth).
"""
import collections
import fcntl
import json
import threading
import time
import urllib.request

ABOVE = 99      # D of a task passed only without a cap: above the largest cap


# ------------------------------------------------------------------ result files --
def load(out):
    p = out / "results.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def append_row(path, r):
    """Append one row to results.jsonl; a file lock keeps rows from several processes from interleaving."""
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(r) + "\n")
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)


# ------------------------------------------------------------------ caps --
def _by_tier(rs):
    runs = collections.defaultdict(list)
    for r in rs:
        runs[r["tier"]].append(r)
    return runs


def _passed(runs, t):
    return any(r["status"] == "passed" for r in runs[t])


def free_n(rs, unlimited):
    """N: the fewest turns among the passing runs without a cap (None if none passes)."""
    ns = [r["n_episodes"] or unlimited for r in rs if not r["tier"] and r["status"] == "passed"]
    return min(ns) if ns else None


def next_tier(rs, tiers, k, unlimited):
    """Model 0. Measured rows of one task -> the next cap to run, or None when done.
    Needs k runs without a cap first; returns 0 (no cap) while fewer than k exist, so callers that only measure D
    must check for them before (they are produced by `free`)."""
    runs = _by_tier(rs)
    if len(runs[0]) < k:
        return 0
    n = free_n(rs, unlimited)
    if n is None:
        return None                         # not solved: no estimate
    for t in tiers:
        if t >= n or _passed(runs, t):
            return None
        if len(runs[t]) < k:
            return t
    return None


def next_below(rs, tiers, k, cap):
    """Later models. Rows of one task -> the next cap below `cap` to run, or None when done. No runs without a cap."""
    runs = _by_tier(rs)
    for t in tiers:
        if t >= cap or _passed(runs, t):
            return None
        if len(runs[t]) < k:
            return t
    return None


def depth_tier(rs, tiers, unlimited=None):
    """Estimate of model 0: the smallest passing cap; caps not run count as passed from the smallest one at or above N;
    ABOVE if the task passes only without a cap; None if it never passes."""
    ok = [r["tier"] for r in rs if r["tier"] and r["status"] == "passed"]
    n = free_n(rs, unlimited or float("inf"))
    if n is not None:
        ok += [t for t in tiers if t >= n][:1]
    return min(ok) if ok else (ABOVE if n is not None else None)


def below_depth(rs, cap):
    """Estimate of a later model: the smallest cap below `cap` that it passes, or None."""
    return min((r["tier"] for r in rs if r["tier"] and r["tier"] < cap and r["status"] == "passed"), default=None)


def set_depth(rows, tiers, unlimited=None):
    """rows[i] = measured rows of model i on one task (models in the order of protocol.MODELS).
    -> (D, [estimate of each model]). D is None if model 0 does not solve the task; a later model that does not lower
    the estimate so far, or is not run, gets None."""
    if not rows:
        return None, []
    d = depth_tier(rows[0], tiers, unlimited)
    est = [d]
    for rs in rows[1:]:
        e = below_depth(rs, d) if d is not None else None
        est.append(e)
        if e is not None:
            d = e
    return d, est


def cap_for(rows, tiers, unlimited=None):
    """The cap a later model runs under: the smallest estimate of the models whose rows are given (None = do not run)."""
    return set_depth(rows, tiers, unlimited)[0]


# ------------------------------------------------------------------ spending --
def openrouter_left(key):
    """Dollars left on OpenRouter: the smaller of the key's remaining limit and the account credits; None if unknown."""
    def get(path):
        req = urllib.request.Request("https://openrouter.ai/api/v1/" + path, headers={"Authorization": "Bearer " + key})
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.load(r)["data"]
    try:
        k, c = get("key"), get("credits")
        left = [c["total_credits"] - c["total_usage"]]
        if k.get("limit") is not None:
            left.append(k["limit_remaining"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return min(left)


class Budget:
    """Gate on starting new runs: closes when the cost recorded in this process reaches max_cost, when the OpenRouter
    balance (checked at most every 2 minutes) is below reserve, or when stop_file exists. None disables a condition."""

    def __init__(self, key, max_cost=None, reserve=None, stop_file=None):
        self.key, self.max_cost, self.reserve, self.stop_file = key, max_cost, reserve, stop_file
        self.spent, self.checked, self.stopped = 0.0, 0.0, None
        self.lock = threading.Lock()

    def add(self, cost):
        with self.lock:
            self.spent += cost or 0

    def ok(self):
        with self.lock:
            if self.stopped:
                return False
            if self.stop_file is not None and self.stop_file.exists():
                self.stopped = f"found {self.stop_file}"
            elif self.max_cost is not None and self.spent >= self.max_cost:
                self.stopped = "recorded cost $%.2f reached --max-cost %.2f" % (self.spent, self.max_cost)
            elif self.reserve is not None and time.time() - self.checked > 120:
                self.checked = time.time()
                left = openrouter_left(self.key)
                if left is not None and left < self.reserve:
                    self.stopped = "OpenRouter balance $%.2f is below --reserve %.2f" % (left, self.reserve)
            if self.stopped:
                print(f"budget: {self.stopped}; no new runs are started", flush=True)
            return not self.stopped
