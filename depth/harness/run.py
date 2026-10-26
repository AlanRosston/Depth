"""Multi-segment tasks: runs without a cap, measuring D under the model set, and pass rates.

A task is a Harbor multi-step task (the [[steps]] of task.toml); the agent is staged_agent.StagedTerminus.
Commands (scripts/harness.py):

  check  <task>...             oracle passes every segment and nop fails the task
  free   <task>... [--model M] K runs without a cap (default model: protocol.MODEL)
  ladder <task>...             measure D with the model set protocol.MODELS; needs the runs of `free`
  report [--model M]           N and D per task; with --model, the pass rate of M's runs without a cap

Every run stops at the first segment that fails. Without a cap the K runs are made at once; under a cap the
second run is made only if the first fails. Model 0 writes runs/<model>/results.jsonl, each later model
runs/<model>__below/results.jsonl, one row per run. Measured rows are not rerun.
"""
import collections
import hashlib
import json
import os
import re
import shutil
import statistics as st
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from depth import REPO, ladder as L, paths, protocol, sandbox, settings, solvers
from depth.ladder import Budget, append_row, load  # noqa: F401

OUT = paths.HARNESS_OUT
CHECK = OUT / "check"

AGENT = "depth.harness.staged_agent:StagedTerminus"
MODEL = protocol.MODEL
MODELS = protocol.MODELS
LADDER = protocol.LADDER
K = protocol.K
UNLIMITED = protocol.UNLIMITED
STAGE_TIMEOUT = protocol.AGENT_TIMEOUT      # agent time limit (s) of each segment
AGENT_BROKE_TERMINAL = ("error connecting to /tmp/tmux", "no server running on")   # counted as a failed run

GATE = None                                 # Budget set by free / ladder


def runs_dir(model, below=False):
    return OUT / "runs" / (solvers.tag(model) + ("__below" if below else ""))


# ------------------------------------------------------------------ tasks --
def stages(task):
    """Names of the [[steps]] of task.toml, in order."""
    names, in_step = [], False
    for ln in (Path(task) / "task.toml").read_text().splitlines():
        s = ln.strip()
        if s.startswith("["):
            in_step = s == "[[steps]]"
        elif in_step and (m := re.match(r'name\s*=\s*"([^"]+)"', s)):
            names.append(m.group(1))
            in_step = False
    return names


def w_lines(task):
    """W: non-blank, non-comment lines of the solve.sh of all segments."""
    n = 0
    for p in sorted(Path(task).glob("steps/*/solution/solve.sh")):
        n += sum(1 for ln in p.read_text(errors="replace").splitlines()
                 if ln.strip() and not ln.strip().startswith("#"))
    return n


# ------------------------------------------------------------------ one trial --
def staged_kwargs(model, k, T):
    kw = solvers.solver_kwargs(model, T or UNLIMITED)
    kw.append(f"n_stages={k}")
    if T:
        kw.append("announce_turn_limit=true")
    return kw


def multi_trial(task, trials_dir, name, agent, model=None, agent_kwargs=(), extra_env=None):
    """Run one Harbor multi-step trial; returns one result row for the task and its segments."""
    task, trials_dir = Path(task).resolve(), Path(trials_dir).resolve()
    k = len(stages(task))
    trials_dir.mkdir(parents=True, exist_ok=True)
    tdir = trials_dir / name
    if tdir.exists():
        subprocess.run(["rm", "-rf", str(tdir)])
    task_path, env_args = sandbox.harbor_env(task, trials_dir, name, k * STAGE_TIMEOUT + 1800)
    cmd = [settings.HARBOR, "trial", "start", "-p", str(task_path), "-a", agent, *env_args,
           "--trials-dir", str(trials_dir), "--trial-name", name, "--delete",
           "--agent-timeout", str(STAGE_TIMEOUT)]
    if ":" in agent:                    # module:class agents resume the conversation; oracle and nop do not
        cmd.append("--resume-trajectory")
    if model:
        cmd += ["-m", solvers.harbor_model(model)]
    for kv in agent_kwargs:
        cmd += ["--agent-kwarg", kv]
    env = dict(os.environ, COLUMNS="200", NO_COLOR="1")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO), env.get("PYTHONPATH")]))
    env.update(extra_env or {})
    sandbox._cleanup(name)
    with sandbox.SEM:
        if GATE is not None and not GATE.ok():
            r = summarize(None, tdir, k, 0.0, False)
            r.update(status="gated", exception=GATE.stopped)
            return r
        timed_out, secs = sandbox.run_harbor(cmd, env, trials_dir / f"{name}.harbor.log", k * STAGE_TIMEOUT + 1800, name)
    try:
        res = json.loads((tdir / "result.json").read_text())
    except (OSError, ValueError):
        res = None
    return summarize(res, tdir, k, secs, timed_out)


def summarize(res, tdir, k, secs, timed_out):
    parts = []
    for s in (res or {}).get("step_results") or []:
        ar = s.get("agent_result") or {}
        md = ar.get("metadata") or {}
        exc = s.get("exception_info") or {}
        parts.append({
            "name": s["step_name"],
            "reward": ((s.get("verifier_result") or {}).get("rewards") or {}).get("reward"),
            "turns": md.get("n_episodes"),
            "skipped": md.get("skipped_no_turns_left"),
            "cost": ar.get("cost_usd"),
            "exception": exc.get("exception_type"),
            "broke_terminal": any(x in (exc.get("exception_message") or "") for x in AGENT_BROKE_TERMINAL),
        })
    reward = (((res or {}).get("verifier_result") or {}).get("rewards") or {}).get("reward")
    part_exc = {p["exception"] for p in parts if p["exception"] and not p["broke_terminal"]}
    if timed_out:
        status = "timeout"
    elif res is None or (res.get("exception_info") and not parts):
        status = sandbox.classify(res, False)
    elif part_exc - {"AgentTimeoutError"}:
        status = "runtime_error"
    else:
        status = "passed" if reward is not None and float(reward) >= 1.0 else "zero_reward"
    turns = [p["turns"] for p in parts if p["turns"] is not None]
    return {"status": status, "reward": reward, "n_stages": k,
            "stages_passed": sum(p["reward"] is not None and float(p["reward"]) >= 1.0 for p in parts),
            "n_episodes": sum(turns) if turns else None,
            "agent_cost": round(sum(p["cost"] or 0 for p in parts), 5),
            "secs": secs, "trial_dir": str(tdir), "parts": parts}


def rollout(task, out, model, T, attempt):
    """One run of `model` on `task` under cap T (0 = no cap)."""
    task = Path(task)
    k = len(stages(task))
    name = f"{task.name}__{solvers.tag(model)}__T{T}__a{attempt}"
    r = multi_trial(early_stop_task(task), out / "trials", name, AGENT, model, staged_kwargs(model, k, T),
                    extra_env=settings.agent_env())
    return {"task_id": task.name, "tier": T, "attempt": attempt, "model": model, "W": w_lines(task), **r}


_EARLY_LOCK = threading.Lock()


def _tree_sig(d):
    h = hashlib.sha1()
    for p in sorted(Path(d).rglob("*")):
        if p.is_file():
            s = p.stat()
            h.update(f"{p.relative_to(d)}\0{s.st_size}\0{s.st_mtime_ns}\n".encode())
    return h.hexdigest()


def with_min_reward(toml, names):
    """Add `min_reward = 1.0` to the [[steps]] in `names` that have no min_reward."""
    lines, blocks, cur = toml.splitlines(), [], None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("["):
            cur = {"name": None, "at": None, "has": False} if s == "[[steps]]" else None
            if cur is not None:
                blocks.append(cur)
        elif cur is not None:
            m = re.match(r'name\s*=\s*"([^"]+)"', s)
            if m and cur["name"] is None:
                cur["name"], cur["at"] = m.group(1), i
            elif s.startswith("min_reward"):
                cur["has"] = True
    add = {b["at"] for b in blocks if b["name"] in names and not b["has"]}
    return "\n".join(ln + ("\nmin_reward = 1.0" if i in add else "") for i, ln in enumerate(lines)) + "\n"


def early_stop_task(task):
    """Copy of the task in out/harness/tasks_early_stop/<name> with min_reward = 1.0 on every segment but the last.
    Rebuilt when the task changes."""
    task = Path(task).resolve()
    dst = OUT / "tasks_early_stop" / task.name
    sig_file = dst.with_name(task.name + ".sig")
    with _EARLY_LOCK:
        sig = _tree_sig(task)
        if dst.exists() and sig_file.exists() and sig_file.read_text() == sig:
            return dst
        tmp = dst.with_name(task.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(task, tmp, symlinks=True)
        names = stages(task)
        (tmp / "task.toml").write_text(with_min_reward((task / "task.toml").read_text(), set(names[:-1])))
        shutil.rmtree(dst, ignore_errors=True)
        tmp.rename(dst)
        sig_file.write_text(sig)
    return dst


# ------------------------------------------------------------------ measuring D --
def measured(r):
    return r["status"] not in ("runtime_error", "build_failed", "unknown", "timeout", "gated")


def task_rows(task_id, model, below=False):
    return [r for r in load(runs_dir(model, below)) if measured(r) and r["task_id"] == task_id]


def model_rows(task_id):
    """Measured rows of each model of the set on one task, in the order of MODELS."""
    return [task_rows(task_id, m, i > 0) for i, m in enumerate(MODELS)]


def task_depth(task_id):
    """(D, estimate of each model) of one task."""
    return L.set_depth(model_rows(task_id), LADDER, UNLIMITED)


def task_done(rows):
    """True if no model of the set has a cap left to run on the task."""
    if L.next_tier(rows[0], LADDER, K, UNLIMITED) is not None:
        return False
    return all(L.next_below(rs, LADDER, K, L.cap_for(rows[:i], LADDER, UNLIMITED)) is None
               for i, rs in enumerate(rows) if i and L.cap_for(rows[:i], LADDER, UNLIMITED) is not None)


class Runner:
    """Runs one model on tasks and records rows."""

    def __init__(self, model, below=False):
        self.model, self.out = model, runs_dir(model, below)
        self.out.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def one(self, task, k, t, i, extra=None):
        r = rollout(task, self.out, self.model, t, i)
        if r["status"] == "gated":
            return r
        r.update(extra or {})
        with self.lock:
            append_row(self.out / "results.jsonl", r)
        if GATE is not None:
            GATE.add(r["agent_cost"])
        print(f"{task.name} {solvers.tag(self.model)} T={t} a{i} {r['status']} segments {r['stages_passed']}/{k} "
              f"turns={r['n_episodes']} {r['secs']:.0f}s", flush=True)
        return r

    def tier(self, task, k, t, rs, extra=None):
        """Run cap t (0 = no cap) up to K times, skipping measured attempts; append measured rows to rs.
        Returns False if a run was gated or not measured."""
        have = {r["attempt"] for r in rs if r["tier"] == t}
        todo = [i for i in range(K) if i not in have]
        if t == 0:
            with ThreadPoolExecutor(max(1, len(todo))) as ex:
                got = list(ex.map(lambda i: self.one(task, k, t, i, extra), todo))
        else:
            got = []
            for i in todo:
                if any(r["tier"] == t and r["status"] == "passed" for r in rs + got):
                    break
                got.append(self.one(task, k, t, i, extra))
                if not measured(got[-1]):
                    break
        rs += [r for r in got if measured(r)]
        if any(r["status"] == "gated" for r in got):
            return False
        if not all(map(measured, got)):
            print(f"{task.name}: run not measured ({[r['status'] for r in got]})", flush=True)
            return False
        return True


def set_gate(a):
    global GATE
    stop = Path(a.stop_file) if getattr(a, "stop_file", None) else None
    if getattr(a, "reserve", None) is not None or getattr(a, "max_cost", None) is not None or stop is not None:
        GATE = Budget(settings.API_KEY, getattr(a, "max_cost", None), getattr(a, "reserve", None), stop)


def _run_all(pipeline, tasks):
    with ThreadPoolExecutor(len(tasks)) as ex:
        finished = list(ex.map(pipeline, tasks))
    if not all(finished):
        raise SystemExit(1)


def cmd_free(a):
    set_gate(a)
    run = Runner(a.model or MODEL)

    def pipeline(task):
        task = Path(task)
        k = len(stages(task))
        with run.lock:
            rs = [r for r in load(run.out) if measured(r) and r["task_id"] == task.name]
        if not run.tier(task, k, 0, rs):
            return False
        ok = sorted(r["n_episodes"] or UNLIMITED for r in rs if r["tier"] == 0 and r["status"] == "passed")
        print(f"-- {task.name}: {len(ok)}/{sum(r['tier'] == 0 for r in rs)} passed without a cap, turns {ok}", flush=True)
        return True

    _run_all(pipeline, a.tasks)


def cmd_ladder(a):
    set_gate(a)
    runners = [Runner(m, i > 0) for i, m in enumerate(MODELS)]

    def pipeline(task):
        task = Path(task)
        k = len(stages(task))
        rows = []
        for i, run in enumerate(runners):
            with run.lock:
                rs = [r for r in load(run.out) if measured(r) and r["task_id"] == task.name]
            if i == 0:
                n0 = sum(r["tier"] == 0 for r in rs)
                if n0 < K:
                    print(f"-- {task.name}: {n0}/{K} runs without a cap; run `free` first", flush=True)
                    return False
                while (t := L.next_tier(rs, LADDER, K, UNLIMITED)) is not None:
                    if not run.tier(task, k, t, rs):
                        return False
            else:
                cap = L.cap_for(rows, LADDER, UNLIMITED)
                if cap is None:
                    break
                while (t := L.next_below(rs, LADDER, K, cap)) is not None:
                    if not run.tier(task, k, t, rs, extra={"cap": cap}):
                        return False
            rows.append(rs)
        d, est = L.set_depth(rows, LADDER, UNLIMITED)
        print(f"-- {task.name}: D={d} (estimates {dict(zip(map(solvers.tag, MODELS), est))})", flush=True)
        return True

    _run_all(pipeline, a.tasks)


def cmd_check(a):
    tasks = list(map(Path, a.tasks))
    with ThreadPoolExecutor(2 * len(tasks)) as ex:
        futs = {(t.name, ag): ex.submit(multi_trial, t, CHECK / "trials", f"{t.name}__{ag}", ag)
                for t in tasks for ag in ("oracle", "nop")}
    bad = 0
    for task in tasks:
        o, n = futs[task.name, "oracle"].result(), futs[task.name, "nop"].result()
        ok = o["status"] == "passed" and all(p["reward"] == 1 for p in o["parts"]) and n["status"] != "passed"
        bad += not ok
        (CHECK / f"{task.name}.json").write_text(json.dumps({"oracle": o, "nop": n}, indent=1))
        print(f"{'OK ' if ok else 'BAD'} {task.name}: oracle {o['status']} {[p['reward'] for p in o['parts']]}; "
              f"nop {n['status']} {[p['reward'] for p in n['parts']]}", flush=True)
    sys.exit(1 if bad else 0)


def cmd_report(a):
    if a.model:
        return report_model(a.model)
    tids = sorted({r["task_id"] for r in load(runs_dir(MODEL)) if measured(r)})
    summary = {}
    for tid in tids:
        rows = model_rows(tid)
        d, est = L.set_depth(rows, LADDER, UNLIMITED)
        s = {"k": rows[0][0]["n_stages"], "W": rows[0][0]["W"], "N": L.free_n(rows[0], UNLIMITED), "D": d,
             "estimates": dict(zip(MODELS, est)), "finished": task_done(rows)}
        summary[tid] = s
        print(f"{tid}: k={s['k']} W={s['W']} N={s['N']} D={d}{'' if s['finished'] else ' (not finished)'}")
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    (OUT / "runs" / "summary.json").write_text(json.dumps(summary, indent=1))


def report_model(model):
    """Pass rate and fraction of segments passed of a model's runs without a cap, per task and averaged."""
    by = collections.defaultdict(list)
    for r in load(runs_dir(model)):
        if measured(r) and r["tier"] == 0:
            by[r["task_id"]].append(r)
    if not by:
        print(f"no runs without a cap for {model}")
        return
    rates, segs = {}, {}
    for tid, rs in sorted(by.items()):
        rates[tid] = st.mean(r["status"] == "passed" for r in rs)
        segs[tid] = st.mean(r["stages_passed"] / r["n_stages"] for r in rs)
        print(f"{tid}: pass rate {rates[tid]:.2f}, segments {segs[tid]:.2f}")
    print(f"{model}: pass@1 {st.mean(rates.values()):.3f}, segments {st.mean(segs.values()):.3f}, {len(rates)} tasks")
