#!/usr/bin/env python3
"""Self-test of measuring D on existing tasks, without docker or model API calls: sandbox.trial is replaced by a
fake that passes by model and cap. Checks the order of runs under the model set and the turn limit in the instruction.

    python3 tests/audit_selftest.py
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depth.audit import run as A  # noqa: E402

M0, M1 = "selftest/a0", "selftest/a1"
TASK = A.OUT / "selftest" / "toyset" / "t1"


def make_task():
    shutil.rmtree(TASK.parent, ignore_errors=True)
    (TASK / "environment").mkdir(parents=True)
    (TASK / "environment" / "Dockerfile").write_text("FROM ubuntu:22.04\n")
    (TASK / "instruction.md").write_text("Do the thing.\n")
    (TASK / "task.toml").write_text('version = "1.0"\n\n[agent]\ntimeout_sec = 120.0\n')


def main():
    fails, calls, hints = [], [], {}

    def expect(ok, what):
        print(("  ok   " if ok else "  FAIL ") + what, flush=True)
        if not ok:
            fails.append(what)

    def fake_trial(task_dir, trials_dir, name, agent=None, model=None, agent_kwargs=(), timeout=None, extra_env=None,
                   agent_timeout=None):
        t = int(name.split("__T")[1].split("__")[0])
        calls.append((model, t))
        hints[t] = (Path(task_dir) / "instruction.md").read_text()
        ok = (t == 0 or t >= 5) if model == M0 else t >= 2
        return {"status": "passed" if ok else "zero_reward", "reward": 1.0 if ok else 0.0, "secs": 0.0,
                "agent_cost": 0.0, "n_episodes": 8 if ok and not t else t}

    saved = A.sandbox.trial, A.MODEL, A.MODELS
    for m in (M0, M1):
        A.solvers.SOLVERS[m] = A.solvers.SOLVERS[A.MODEL]
    A.sandbox.trial, A.MODEL, A.MODELS = fake_trial, M0, [M0, M1]
    for i, m in enumerate((M0, M1)):
        shutil.rmtree(A.runs_dir(m, i > 0), ignore_errors=True)
    make_task()
    ns = lambda **kw: argparse.Namespace(tasks=[str(TASK)], model=None, reserve=None, max_cost=None, stop_file=None, **kw)
    try:
        A.cmd_ladder(ns())
        expect(calls == [], f"ladder does not run before the runs without a cap exist: {calls}")
        A.cmd_free(ns())
        expect(calls == [(M0, 0)] * A.K, f"free: K runs of model 0 without a cap: {calls}")
        calls.clear()
        A.cmd_ladder(ns())
        want = [(M0, 1), (M0, 1), (M0, 2), (M0, 2), (M0, 3), (M0, 3), (M0, 5), (M1, 1), (M1, 1), (M1, 2)]
        res = A.task_depths()[("toyset", "t1")]
        expect(calls == want and res[0] == 8 and res[1] == 2 and res[2] == [5, 2] and res[3],
               f"ladder: model 0 from cap 1 to its first pass below N=8, then model 1 below 5; D=2: {calls}, {res}")
        expect("at most 3 turns for this task" in hints[3] and "Turn limit" not in hints[0],
               "the turn limit is appended to the instruction under a cap only")
        calls.clear()
        A.cmd_ladder(ns())
        expect(calls == [], f"ladder again: nothing rerun: {calls}")
        A.cmd_report(argparse.Namespace(tasks=[]))
    finally:
        A.sandbox.trial, A.MODEL, A.MODELS = saved
        for m in (M0, M1):
            A.solvers.SOLVERS.pop(m, None)
        for i, m in enumerate((M0, M1)):
            shutil.rmtree(A.runs_dir(m, i > 0), ignore_errors=True)
        shutil.rmtree(TASK.parent, ignore_errors=True)
        shutil.rmtree(A.OUT / "tasks" / "selftest_a0", ignore_errors=True)
        shutil.rmtree(A.OUT / "tasks" / "selftest_a1", ignore_errors=True)
    print("self-test: %s" % ("all passed" if not fails else "%d failed" % len(fails)))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
