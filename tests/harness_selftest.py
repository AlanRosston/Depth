#!/usr/bin/env python3
"""Self-test of the multi-segment harness, without model API calls: a local fake model (OpenAI-compatible endpoint)
answers from a script while Harbor runs a three-segment toy task.

    SELFTEST_TOY_IMAGE=<image with bash, python3, tmux and /app> python3 tests/harness_selftest.py    # needs docker, ~3 minutes

Toy task: p1 writes /app/a.txt = 1; p2 runs /app/work/tool.sh (delivered with this segment, prints a+1) into
/app/b.txt; p3 writes c = 2b. Fake model: p1 one turn (command + complete); p2 two turns (command, then complete);
p3 one turn plus the confirmation. The command of p2 also runs `ls /tests`, and its check requires that the tests of
the previous segment are gone.

Checks:
  0 (no docker): the early-stop copy of a task; the model set (free and ladder are separate, model 0 tries caps from 1
    upward, a later model only caps below the estimate so far, under a cap the second run is skipped once the first
    passes); the budget gate
  1. oracle passes every segment, nop fails the task
  2. resume: the first request of segment 2 contains the whole conversation of segment 1
  3. the confirmation is asked once, in the last segment
  4. turns per segment 1 / 2 / 2, 5 in total; the tests of a segment are removed before the next one
  5. cap T=3 stated: the first message states 3 turns, segment 2 reports 1 turn used, segment 3 sends no request
  6. a run stops at the first segment that fails (fake model does nothing in p2): p3 sends no request
  7. EMPTY_LIMIT empty replies in a row end the trial as runtime_error (not measured); p3 sends no request
"""
import argparse
import json
import os
import re
import shutil
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depth.harness import run as R  # noqa: E402

TOY = R.OUT / "selftest" / "toy3"
LOG = R.OUT / "selftest" / "fake_llm.jsonl"
IMAGE = os.environ.get("SELFTEST_TOY_IMAGE")                  # needs bash, python3 and tmux
CONFIRM = "Are you sure you want to mark the task as complete"

PARTS = {
    "p1": ("Write the number 1 to /app/a.txt.", 'test "$(cat /app/a.txt 2>/dev/null)" = 1',
           "echo 1 > /app/a.txt", {}),
    "p2": ("Run /app/work/tool.sh and save what it prints to /app/b.txt.",
           'test "$(cat /app/b.txt 2>/dev/null)" = 2 && '
           '{ [ ! -e /app/tests_seen.txt ] || grep -q "No such file" /app/tests_seen.txt; }',
           "/app/work/tool.sh > /app/b.txt",
           {"work/tool.sh": '#!/bin/sh\necho $(( $(cat /app/a.txt) + 1 ))\n'}),
    "p3": ("Write twice the number in /app/b.txt to /app/c.txt.", 'test "$(cat /app/c.txt 2>/dev/null)" = 4',
           "echo $(( $(cat /app/b.txt) * 2 )) > /app/c.txt", {}),
}

# Script: (segment, turn within the segment) -> (commands, task_complete). Segments in SKIP_PARTS do nothing and
# complete at once; segments in EMPTY_PARTS get empty replies.
SKIP_PARTS = set()
EMPTY_PARTS = set()
SCRIPT = {
    (1, 0): (["echo 1 > /app/a.txt\n"], True),
    (2, 0): (["ls /tests > /app/tests_seen.txt 2>&1; /app/work/tool.sh > /app/b.txt\n"], False),
    (2, 1): ([], True),
    (3, 0): (["echo $(( $(cat /app/b.txt) * 2 )) > /app/c.txt\n"], True),
}


def make_toy():
    if TOY.exists():
        shutil.rmtree(TOY)
    (TOY / "environment").mkdir(parents=True)
    (TOY / "environment" / "Dockerfile").write_text("FROM %s\n" % IMAGE)
    (TOY / "task.toml").write_text(
        'version = "1.0"\nmulti_step_reward_strategy = "mean"\n\n[verifier]\ntimeout_sec = 120.0\n\n'
        '[agent]\ntimeout_sec = 300.0\n\n[environment]\ndocker_image = "%s"\nbuild_timeout_sec = 600.0\n'
        'cpus = 1\nmemory = "2G"\nmemory_mb = 2048\n' % IMAGE
        + "".join('\n[[steps]]\nname = "%s"\n' % n for n in PARTS))
    for name, (text, check, solve, files) in PARTS.items():
        d = TOY / "steps" / name
        for sub in ("tests", "solution"):
            (d / sub).mkdir(parents=True)
        (d / "instruction.md").write_text(text + "\n")
        (d / "tests" / "test.sh").write_text(
            "#!/bin/bash\nmkdir -p /logs/verifier\nif %s; then echo 1; else echo 0; fi > /logs/verifier/reward.txt\n"
            % check)
        (d / "solution" / "solve.sh").write_text("#!/bin/bash\nset -e\n%s\n" % solve)
        for rel, body in files.items():
            p = d / "workdir" / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(body)
            p.chmod(0o755)
        for p in (d / "tests" / "test.sh", d / "solution" / "solve.sh"):
            p.chmod(0o755)


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        msgs = body["messages"]
        text = lambda m: m["content"] if isinstance(m["content"], str) else json.dumps(m["content"])
        users = [i for i, m in enumerate(msgs) if m["role"] == "user"]
        last = text(msgs[users[-1]])
        part, since = None, None
        for i in reversed(users):
            hits = re.findall(r"[Pp]art (\d+) of \d+", text(msgs[i]))
            if hits:
                part, since = int(hits[-1]), i
                break
        turn = sum(m["role"] == "assistant" for m in msgs[since:]) if since is not None else None
        if CONFIRM in last or part in SKIP_PARTS:
            cmds, done = [], True
        else:
            cmds, done = SCRIPT.get((part, turn), ([], True))
        reply = json.dumps({"analysis": "scripted", "plan": "scripted",
                            "commands": [{"keystrokes": c, "duration": 0.5} for c in cmds], "task_complete": done})
        if part in EMPTY_PARTS:
            reply = ""
        with open(LOG, "a") as f:
            f.write(json.dumps({"n_messages": len(msgs), "part": part, "turn": turn, "first": text(msgs[0]),
                                "last_user": last, "reply": reply}) + "\n")
        resp = {"id": "fake", "object": "chat.completion", "created": int(time.time()), "model": "fake-staged",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": reply}}]}
        if reply:
            resp["usage"] = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
        out = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


def fake_run(port, cap, announce, tag, task=TOY):
    LOG.unlink(missing_ok=True)
    info = {"max_input_tokens": 200000, "max_output_tokens": 8000, "input_cost_per_token": 0,
            "output_cost_per_token": 0, "litellm_provider": "openai", "mode": "chat"}
    kw = ["api_base=http://127.0.0.1:%d/v1" % port, "model_info=" + json.dumps(info), "n_stages=3",
          "max_turns=%d" % cap] + (["announce_turn_limit=true"] if announce else [])
    r = R.multi_trial(task, R.OUT / "selftest" / "trials", "toy3__fake_" + tag, R.AGENT, "openai/fake-staged", kw,
                      extra_env={"OPENAI_API_KEY": "fake"})
    reqs = [json.loads(x) for x in LOG.read_text().splitlines()] if LOG.exists() else []
    return r, reqs


def unit(expect):
    """0. Without docker: the early-stop copy of a task."""
    print("0. early-stop copy (no docker)", flush=True)
    toml = ('version = "1.0"\n\n[environment]\ncpus = 1\n\n[[steps]]\nname = "s01"\n\n[[steps]]\nname = "s02"\n'
            'min_reward = 0.5\n\n[[steps]]\nname = "s03"\n')
    got = R.with_min_reward(toml, {"s01", "s02"})
    expect(got.count("min_reward") == 2 and 'name = "s01"\nmin_reward = 1.0' in got and "min_reward = 0.5" in got
           and 'name = "s03"\n' in got and not got.rstrip().endswith("1.0"),
           "with_min_reward: added to s01, s02 keeps its own, none for the last segment")
    shutil.rmtree(R.OUT / "tasks_early_stop" / TOY.name, ignore_errors=True)
    ft = R.early_stop_task(TOY)
    ft_toml = (ft / "task.toml").read_text()
    expect(ft.name == TOY.name and ft_toml.count("min_reward = 1.0") == 2 and 'name = "p3"\nmin_reward' not in ft_toml
           and "min_reward" not in (TOY / "task.toml").read_text() and (ft / "steps" / "p2" / "workdir").is_dir(),
           "early_stop_task: same name, min_reward on p1 and p2 only, original untouched, files complete")
    mt = (ft / "task.toml").stat().st_mtime_ns
    R.early_stop_task(TOY)
    expect((ft / "task.toml").stat().st_mtime_ns == mt, "the copy is not rebuilt while the task is unchanged")
    model_set(expect)
    budget(expect)


def model_set(expect):
    """0 (cont.). The model set with fake runs: model 0 passes without a cap in 30 turns and under caps >= 20; model 1
    passes under caps >= need. Results go to directories of fake model names and are removed afterwards."""
    m0, m1 = "selftest/m0", "selftest/m1"
    saved = R.MODEL, R.MODELS, R.rollout
    calls = []

    def fake(need1):
        def run(task, out, model, T, attempt):
            calls.append((model, T))
            ok = (T == 0 or T >= 20) if model == m0 else T >= need1
            return {"task_id": Path(task).name, "tier": T, "attempt": attempt, "model": model, "W": 3, "n_stages": 3,
                    "status": "passed" if ok else "zero_reward", "stages_passed": 3 if ok else 1,
                    "n_episodes": 30 if ok and not T else T, "agent_cost": 0.0, "secs": 0.0, "trial_dir": "x",
                    "parts": []}
        return run

    def go(cmd, **kw):
        ns = argparse.Namespace(tasks=[str(TOY)], model=None, attempts=None, reserve=None, max_cost=None,
                                stop_file=None, **kw)
        try:
            (R.cmd_free if cmd == "free" else R.cmd_ladder)(ns)
            return True
        except SystemExit:
            return False

    def clean():
        for i, m in enumerate((m0, m1)):
            shutil.rmtree(R.runs_dir(m, i > 0), ignore_errors=True)

    R.MODEL, R.MODELS = m0, [m0, m1]
    clean()
    try:
        R.rollout = fake(10)
        ok = go("ladder")
        expect(not ok and calls == [], f"ladder does not run before the runs without a cap exist: {calls}")
        ok = go("free")
        expect(ok and calls == [(m0, 0)] * R.K, f"free: K runs of model 0 without a cap: {calls}")
        calls.clear()
        ok = go("free")
        expect(ok and calls == [], f"free again: nothing rerun: {calls}")
        ok = go("ladder")
        below = [t for t in R.LADDER if t < 20]
        want0 = [(m0, t) for t in below for _ in range(R.K)] + [(m0, 20)]
        want1 = [(m1, t) for t in below if t < 10 for _ in range(R.K)] + [(m1, 10)]
        d, est = R.task_depth(TOY.name)
        rows1 = R.task_rows(TOY.name, m1, True)
        expect(ok and calls == want0 + want1 and d == 10 and est == [20, 10] and all(r.get("cap") == 20 for r in rows1),
               f"ladder: model 0 from cap 1 up to its first pass (second run skipped), then model 1 below 20; D=10: "
               f"{calls}, D={d}, estimates={est}")
        expect(R.task_done(R.model_rows(TOY.name)), "the task is finished for every model")
        calls.clear()
        ok = go("ladder")
        expect(ok and calls == [], f"ladder again: nothing rerun: {calls}")
        shutil.rmtree(R.runs_dir(m1, True), ignore_errors=True)
        calls.clear()
        R.rollout = fake(99)
        ok = go("ladder")
        d, est = R.task_depth(TOY.name)
        expect(ok and calls == [(m1, t) for t in below for _ in range(R.K)] and d == 20 and est == [20, None],
               f"a later model that passes no cap below the estimate leaves D unchanged: {calls}, D={d}, {est}")
    finally:
        R.MODEL, R.MODELS, R.rollout = saved
        clean()


def budget(expect):
    """0 (cont.). The budget gate: an unknown balance lets runs start, a balance below reserve closes it for good;
    with the gate closed, free starts no Harbor process and writes no row."""
    left = R.L.openrouter_left
    m3 = "selftest/gate"
    try:
        seq = iter([None, 9.0, 3.0, 100.0])
        R.L.openrouter_left = lambda key: next(seq)
        g = R.Budget("x", None, 5.0)
        got = []
        for _ in range(4):
            g.checked = 0.0
            got.append(g.ok())
        expect(got == [True, True, False, False], f"budget: unknown -> open, 9 > 5 open, 3 < 5 closed, stays closed: {got}")
        R.L.openrouter_left = lambda key: 0.0
        R.solvers.SOLVERS[m3] = R.solvers.SOLVERS[R.MODEL]
        shutil.rmtree(R.runs_dir(m3), ignore_errors=True)
        try:
            R.cmd_free(argparse.Namespace(tasks=[str(TOY)], model=m3, attempts=None, reserve=5.0, max_cost=None,
                                          stop_file=None))
            ok = True
        except SystemExit:
            ok = False
        logs = list((R.runs_dir(m3) / "trials").glob("*.harbor.log"))
        expect(not ok and not R.load(R.runs_dir(m3)) and not logs,
               f"gate closed: free starts no Harbor and writes no row: ok={ok} logs={logs}")
    finally:
        R.L.openrouter_left, R.GATE = left, None
        R.solvers.SOLVERS.pop(m3, None)
        shutil.rmtree(R.runs_dir(m3), ignore_errors=True)


def main():
    if not IMAGE:
        sys.exit("set SELFTEST_TOY_IMAGE: an image with bash, python3 and tmux")
    make_toy()
    fails = []

    def expect(ok, what):
        print(("  ok   " if ok else "  FAIL ") + what, flush=True)
        if not ok:
            fails.append(what)

    unit(expect)

    print("1. oracle / nop", flush=True)
    o = R.multi_trial(TOY, R.OUT / "selftest" / "trials", "toy3__oracle", "oracle")
    n = R.multi_trial(TOY, R.OUT / "selftest" / "trials", "toy3__nop", "nop")
    expect([p["reward"] for p in o["parts"]] == [1, 1, 1], "oracle passes every segment: %s" % [p["reward"] for p in o["parts"]])
    expect(n["status"] != "passed", "nop fails: %s %s" % (n["status"], [p["reward"] for p in n["parts"]]))

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]

    print("2-4. fake model, no cap", flush=True)
    r, reqs = fake_run(port, 150, False, "free")
    expect(r["status"] == "passed", "task passed: %s (%s)" % (r["status"], r["trial_dir"]))
    expect([p["turns"] for p in r["parts"]] == [1, 2, 2] and r["n_episodes"] == 5,
           "turns per segment 1/2/2, 5 in total: %s" % [p["turns"] for p in r["parts"]])
    p2 = [q for q in reqs if q["part"] == 2]
    expect(bool(p2) and p2[0]["n_messages"] >= 3 and "Write the number 1" in p2[0]["first"],
           "segment 2 continues the conversation (first request has %s messages, including segment 1)" % (p2[0]["n_messages"] if p2 else None))
    conf = [q["part"] for q in reqs if CONFIRM in q["last_user"]]
    expect(conf == [3], "confirmation asked once, in the last segment: %s" % conf)
    expect(r["parts"][1]["reward"] == 1, "segment 2 cannot see the tests of segment 1 (checked by p2)")
    steps = []
    for name in PARTS:
        f = Path(r["trial_dir"]) / "steps" / name / "agent" / "trajectory.json"
        steps.append(len(json.loads(f.read_text()).get("steps", [])) if f.exists() else None)
    expect(None not in steps and steps == sorted(set(steps)),
           "the saved trajectory grows segment by segment: %s" % steps)

    print("5. fake model, cap T=3 stated", flush=True)
    r, reqs = fake_run(port, 3, True, "T3")
    expect([p["turns"] for p in r["parts"]] == [1, 2, 0] and r["parts"][2]["skipped"],
           "turns 1/2/0, segment 3 skipped: %s" % [(p["turns"], p["skipped"]) for p in r["parts"]])
    expect([p["reward"] for p in r["parts"]] == [1, 1, 0] and r["status"] == "zero_reward",
           "rewards 1/1/0, task fails: %s %s" % ([p["reward"] for p in r["parts"]], r["status"]))
    expect(bool(reqs) and "at most 3 turns" in reqs[0]["first"], "the first message states the cap of 3 turns")
    p2 = [q for q in reqs if q["part"] == 2]
    expect(bool(p2) and "Turns used so far: 1 of 3." in p2[0]["last_user"], "segment 2 reports 1 turn used")
    expect(not [q for q in reqs if q["part"] == 3], "segment 3 sends no request")

    print("6. a run stops at the first failing segment; the fake model does nothing in p2", flush=True)
    SKIP_PARTS.add(2)
    r, reqs = fake_run(port, 150, False, "early", task=R.early_stop_task(TOY))
    SKIP_PARTS.clear()
    expect([(p["name"], p["reward"]) for p in r["parts"]] == [("p1", 1), ("p2", 0)] and r["status"] == "zero_reward",
           "p1 passes, p2 fails and the run stops, p3 not run: %s %s" % ([(p["name"], p["reward"]) for p in r["parts"]], r["status"]))
    expect(not [q for q in reqs if q["part"] == 3], "segment 3 sends no request")

    print("7. empty replies in p2", flush=True)
    limit = int(re.search(r"^EMPTY_LIMIT = (\d+)", (Path(R.__file__).parent / "staged_agent.py").read_text(), re.M).group(1))
    EMPTY_PARTS.add(2)
    r, reqs = fake_run(port, 150, False, "empty")
    EMPTY_PARTS.clear()
    p2 = [q for q in reqs if q["part"] == 2]
    expect(len(p2) == limit and not [q for q in reqs if q["part"] == 3],
           "p2 stops after %d requests, p3 sends none: p2 sent %d" % (limit, len(p2)))
    parts = [(p["name"], p["reward"], p["exception"]) for p in r["parts"]]
    expect(r["status"] == "runtime_error" and not R.measured(r) and parts[0][:2] == ("p1", 1)
           and len(parts) > 1 and parts[1][0] == "p2" and parts[1][2] == "EmptyResponsesError",
           "runtime_error, not measured, p1 passed, p2 EmptyResponsesError: %s %s" % (r["status"], parts))
    srv.shutdown()

    print("self-test: %s" % ("all passed" if not fails else "%d failed" % len(fails)))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
