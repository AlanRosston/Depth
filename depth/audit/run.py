"""Measuring D on existing tasks: Harbor tasks with a single instruction, run in terminus-2.

Commands (scripts/audit.py); TASK is a Harbor task directory, grouped by the name of its parent directory:
  free   TASK... [--model M]   K runs without a cap (default model: protocol.MODEL)
  ladder TASK...               measure D with the model set protocol.MODELS; needs the runs of `free`
  report [TASK...]             N and D per task, distribution of D per dataset

Under cap T, HINT is appended to the instruction. The agent time limit is protocol.AGENT_TIMEOUT. Runs of one task
are sequential; under a cap the second run is made only if the first fails. Model 0 writes
out/audit/runs/<model>/results.jsonl, each later model out/audit/runs/<model>__below/results.jsonl, one row per run.
Measured rows are not rerun.
"""
import collections
import json
import shutil
import statistics as st
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from depth import ladder as L, paths, protocol, sandbox, settings, solvers
from depth.ladder import Budget, load
from depth.taskenv import bake_tmux, prebuilt_image, tmux_image

OUT = paths.AUDIT_OUT
MODEL, MODELS = protocol.MODEL, protocol.MODELS
LADDER, K, UNLIMITED = protocol.LADDER, protocol.K, protocol.UNLIMITED
TRIAL_TIMEOUT = 3600                # wall-clock limit (s) of one Harbor trial, including the image build

HINT = """

---
Turn limit: you have at most {T} turns for this task. One turn is one of your responses; a response may contain
several commands. The session is cut off after turn {T} and whatever state the environment is in will be graded.
Plan for this limit: batch commands where you can, and do not spend turns on steps that are not needed.
"""


def runs_dir(model, below=False):
    return OUT / "runs" / (solvers.tag(model) + ("__below" if below else ""))


def key(p):
    """(dataset, task id) of a task directory."""
    p = Path(p)
    return p.parent.name, p.name


def measured(r):
    if r["status"] in ("infra_error", "unknown", "runtime_error", "build_failed"):
        return False
    return not (r["status"] == "timeout" and r["reward"] is None and not r["n_episodes"])


def by_task(out):
    by = collections.defaultdict(list)
    for r in load(out):
        if measured(r):
            by[(r["dataset"], r["task_id"])].append(r)
    return by


# ------------------------------------------------------------------ task copies --
BAKE_MARK = ".tmux_baked"


def prebake(tasks):
    """Build the tmux layer of every prebuilt image among `tasks`."""
    bases = sorted({prebuilt_image(p) for p in tasks if prebuilt_image(p)})
    if bases:
        with ThreadPoolExecutor(settings.WORKERS) as ex:
            list(ex.map(tmux_image, bases))


def task_copy(src, dst, tier):
    """Copy of the task for cap `tier`, with tmux installed at build time and HINT appended to the instruction."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=True)
    bake_tmux(dst)
    (dst / BAKE_MARK).write_text("")
    if tier:
        p = dst / "instruction.md"
        p.write_text(p.read_text().rstrip("\n") + HINT.format(T=tier))


# ------------------------------------------------------------------ running --
class Runner:
    """Runs one model on existing tasks and records rows."""

    def __init__(self, model, below, budget):
        self.model, self.out, self.budget = model, runs_dir(model, below), budget
        self.out.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def rows(self, p):
        ds, tid = key(p)
        return [r for r in load(self.out) if measured(r) and r["dataset"] == ds and r["task_id"] == tid]

    def rollout(self, p, t, i, extra=None):
        ds, tid = key(p)
        name = f"{tid}__T{t}"
        copy = OUT / "tasks" / solvers.tag(self.model) / ds / name
        with self.lock:
            if not (copy / BAKE_MARK).exists():
                task_copy(Path(p), copy, t)
        r = sandbox.trial(copy, OUT / "trials" / ds / solvers.tag(self.model), f"{name}__a{i}", agent="terminus-2",
                          model=self.model, agent_kwargs=solvers.solver_kwargs(self.model, t or UNLIMITED),
                          timeout=TRIAL_TIMEOUT, extra_env=settings.agent_env(), agent_timeout=protocol.AGENT_TIMEOUT)
        self.budget.add(r["agent_cost"])
        row = {"dataset": ds, "task_id": tid, "tier": t, "attempt": i, "model": self.model,
               **{x: r[x] for x in ("status", "reward", "secs", "agent_cost", "n_episodes")}, **(extra or {})}
        with self.lock:
            L.append_row(self.out / "results.jsonl", row)
        print(f"{ds} {tid} {solvers.tag(self.model)} T={t} a{i} {r['status']} turns={r['n_episodes']} "
              f"{r['secs']:.0f}s", flush=True)
        return row

    def tier(self, p, t, rs, extra=None):
        """Run cap t (0 = no cap) up to K times, skipping measured attempts; under a cap, stop after the first pass.
        Returns False if the budget closed or a run was not measured."""
        have = {r["attempt"] for r in rs if r["tier"] == t}
        for i in range(K):
            if i in have:
                continue
            if t and any(r["tier"] == t and r["status"] == "passed" for r in rs):
                break
            if not self.budget.ok():
                return False
            r = self.rollout(p, t, i, extra)
            if not measured(r):
                print(f"{key(p)[1]}: run not measured ({r['status']})", flush=True)
                return False
            rs.append(r)
        return True


def _budget(a):
    return Budget(settings.API_KEY, a.max_cost, a.reserve, Path(a.stop_file) if a.stop_file else None)


def cmd_free(a):
    budget = _budget(a)
    run = Runner(a.model or MODEL, False, budget)
    tasks = [Path(p) for p in a.tasks]
    prebake(tasks)

    def pipeline(p):
        rs = run.rows(p)
        if run.tier(p, 0, rs):
            ok = sorted(r["n_episodes"] for r in rs if r["tier"] == 0 and r["status"] == "passed" and r["n_episodes"])
            print(f"-- {key(p)[1]}: {len(ok)}/{sum(r['tier'] == 0 for r in rs)} passed without a cap, turns {ok}",
                  flush=True)

    with ThreadPoolExecutor(settings.WORKERS) as ex:
        list(ex.map(pipeline, tasks))


def cmd_ladder(a):
    budget = _budget(a)
    runners = [Runner(m, i > 0, budget) for i, m in enumerate(MODELS)]
    tasks = [Path(p) for p in a.tasks]
    prebake(tasks)

    def pipeline(p):
        rows = []
        for i, run in enumerate(runners):
            rs = run.rows(p)
            if i == 0:
                n0 = sum(r["tier"] == 0 for r in rs)
                if n0 < K:
                    print(f"-- {key(p)[1]}: {n0}/{K} runs without a cap; run `free` first", flush=True)
                    return
                while (t := L.next_tier(rs, LADDER, K, UNLIMITED)) is not None:
                    if not run.tier(p, t, rs):
                        return
            else:
                cap = L.cap_for(rows, LADDER, UNLIMITED)
                if cap is None:
                    break
                while (t := L.next_below(rs, LADDER, K, cap)) is not None:
                    if not run.tier(p, t, rs, extra={"cap": cap}):
                        return
            rows.append(rs)
        d, est = L.set_depth(rows, LADDER, UNLIMITED)
        print(f"-- {key(p)[1]}: D={d} (estimates {dict(zip(map(solvers.tag, MODELS), est))})", flush=True)

    with ThreadPoolExecutor(settings.WORKERS) as ex:
        list(ex.map(pipeline, tasks))


# ------------------------------------------------------------------ report --
def task_depths(only=None):
    """{(dataset, task id): (N, D, estimates, finished)} over the tasks with rows of model 0."""
    by = [by_task(runs_dir(m, i > 0)) for i, m in enumerate(MODELS)]
    res = {}
    for k0, rs0 in by[0].items():
        if only is not None and k0 not in only:
            continue
        rows = [rs0] + [b.get(k0, []) for b in by[1:]]
        d, est = L.set_depth(rows, LADDER, UNLIMITED)
        done = L.next_tier(rs0, LADDER, K, UNLIMITED) is None and all(
            L.next_below(rows[i], LADDER, K, c) is None
            for i in range(1, len(rows)) if (c := L.cap_for(rows[:i], LADDER, UNLIMITED)) is not None)
        res[k0] = (L.free_n(rs0, UNLIMITED), d, est, done)
    return res


def cmd_report(a):
    only = {key(p) for p in a.tasks} if a.tasks else None
    summary = collections.defaultdict(dict)
    for (ds, tid), (n, d, est, done) in sorted(task_depths(only).items()):
        summary[ds][tid] = {"N": n, "D": d, "estimates": dict(zip(MODELS, est)), "finished": done}
    for ds, tasks in summary.items():
        solved = [s["D"] for s in tasks.values() if s["finished"] and s["D"] is not None]
        print(f"== {ds}: {len(tasks)} tasks, {sum(s['finished'] for s in tasks.values())} finished, "
              f"{len(solved)} solved; D {dict(sorted(collections.Counter(solved).items()))} "
              f"({L.ABOVE} = passed only without a cap); median D {st.median(solved) if solved else None}")
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    (OUT / "runs" / "summary.json").write_text(json.dumps(summary, indent=1))
