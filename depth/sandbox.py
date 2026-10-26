"""Harbor trials in local Docker (`harbor trial start -e docker --delete`).
trial() runs a single-instruction task; run_harbor() starts and supervises one harbor process."""
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

from depth import settings as C, solvers

SEM = threading.BoundedSemaphore(C.WORKERS)


def free_gb():
    return shutil.disk_usage("/").free / 2**30


def _tail(p, n):
    try:
        s = Path(p).read_text(errors="replace")
    except OSError:
        return ""
    return s[-n:]


def _has_test_results(tdir, stdout):
    """True if the verifier left ctrf.json or a pytest summary line."""
    return (Path(tdir) / "verifier" / "ctrf.json").exists() or bool(re.search(r"\d+ (passed|failed|errors?)", stdout))


def classify(res, timed_out):
    """Status of a trial: passed, zero_reward, timeout, build_failed, runtime_error or unknown."""
    if timed_out:
        return "timeout"
    if res is None:
        return "unknown"
    exc = res.get("exception_info")
    if exc:
        blob = json.dumps(exc).lower()
        if "timeout" in blob:
            return "timeout"
        if "build" in blob or "dockerfile" in blob:
            return "build_failed"
        return "runtime_error"
    rw = ((res.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    if rw is None:
        return "unknown"
    return "passed" if float(rw) >= 1.0 else "zero_reward"


# verifier output that marks a network failure of the verifier itself
INFRA = re.compile(r"uvx?: command not found|curl: \(\d+\)|Could not resolve host|Temporary failure in name resolution|"
                   r"Failed to fetch|Network is unreachable|Failed to download|error sending request|"
                   r"Connection (failed|reset|refused)|/\.local/bin/env: No such file")


def trial(task_dir, trials_dir, name, agent="oracle", model=None, agent_kwargs=(), timeout=None, extra_env=None,
          infra_retries=2, agent_timeout=None):
    """Run one trial; rerun it up to infra_retries times while its status is infra_error (the verifier produced no
    test results and its output matches INFRA)."""
    for n in range(infra_retries + 1):
        r = _trial(task_dir, trials_dir, name, agent, model, agent_kwargs, timeout, extra_env, agent_timeout)
        if r["status"] != "infra_error":
            return r
        time.sleep(30 * (n + 1))
    return r


def harbor_env(task_dir, trials_dir, name, limit_secs):
    """Task directory and environment arguments handed to Harbor."""
    return Path(task_dir).resolve(), ["-e", "docker"]


def _trial(task_dir, trials_dir, name, agent, model, agent_kwargs, timeout, extra_env, agent_timeout):
    trials_dir = Path(trials_dir)
    trials_dir.mkdir(parents=True, exist_ok=True)
    tdir = trials_dir / name
    if tdir.exists():
        shutil.rmtree(tdir)
    task_path, env_args = harbor_env(task_dir, trials_dir, name, timeout or C.ORACLE_TIMEOUT)
    cmd = [C.HARBOR, "trial", "start", "-p", str(task_path), "-a", agent, *env_args,
           "--trials-dir", str(trials_dir.resolve()), "--trial-name", name, "--delete"]
    if model:
        cmd += ["-m", solvers.harbor_model(model)]
    for kv in agent_kwargs:
        cmd += ["--agent-kwarg", kv]
    if agent_timeout:
        cmd += ["--agent-timeout", str(agent_timeout)]
    env = dict(os.environ, COLUMNS="200", NO_COLOR="1")
    env.update(extra_env or {})
    _cleanup(name)
    with SEM:
        timed_out, secs = run_harbor(cmd, env, trials_dir / f"{name}.harbor.log", timeout or C.ORACLE_TIMEOUT, name)
    res = None
    try:
        res = json.loads((tdir / "result.json").read_text())
    except (OSError, ValueError):
        pass
    status = classify(res, timed_out)
    reward = ((res or {}).get("verifier_result") or {}).get("rewards", {}).get("reward") if res else None
    vstdout = _tail(tdir / "verifier" / "test-stdout.txt", 200000)
    if status == "zero_reward" and not _has_test_results(tdir, vstdout) and INFRA.search(vstdout):
        status = "infra_error"
    ar = (res or {}).get("agent_result") or {}
    return {"status": status, "reward": reward, "secs": secs, "trial_dir": str(tdir),
            "exception": (res or {}).get("exception_info"), "agent_cost": ar.get("cost_usd"),
            "n_episodes": (ar.get("metadata") or {}).get("n_episodes")}


def mem_avail_mb():
    """MemAvailable from /proc/meminfo in MB (a large number if unreadable)."""
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 1 << 30


def _start(cmd, log, env):
    """Start a harbor process: wait while the 5-minute load is at least MAX_LOAD5 (if set); then, holding START_LOCK,
    wait for MIN_MEM_MB of available memory, start the process and keep the lock for START_GAP seconds."""
    import fcntl
    while C.MAX_LOAD5 and os.getloadavg()[1] >= C.MAX_LOAD5:
        time.sleep(30)
    with open(C.START_LOCK, "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            while mem_avail_mb() < C.MIN_MEM_MB:
                time.sleep(30)
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            time.sleep(C.START_GAP)
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)
    return p


def run_harbor(cmd, env, log_path, timeout, name):
    """Run one harbor process (the caller holds a SEM slot): wait while free disk is below MIN_FREE_GB, write output to
    log_path, and after `timeout` seconds send SIGTERM, then SIGKILL after 60 s, and clean up.
    -> (timed out, seconds since start)."""
    while free_gb() < C.MIN_FREE_GB:
        time.sleep(30)
    timed_out = False
    with open(log_path, "w") as log:
        p = _start(cmd, log, env)
        t0 = time.time() - C.START_GAP
        try:
            p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
            _cleanup(name)
    return timed_out, round(time.time() - t0, 1)


def _cleanup(name):
    """Bring down the compose project `<trial name, lower case>__env` of a trial, if it exists."""
    proj = f"{name.lower()}__env"
    rc = subprocess.run(["docker", "compose", "ls", "-a", "-q", "--filter", f"name={proj}"],
                        capture_output=True, text=True)
    if proj in rc.stdout.split():
        subprocess.run(["docker", "compose", "-p", proj, "down", "--volumes", "--remove-orphans"],
                       capture_output=True, timeout=300)
