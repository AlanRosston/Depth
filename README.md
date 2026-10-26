# DEPTH

Code for *Is Your Long-Horizon Benchmark Really Deep? Measuring and Synthesizing Depth in Terminal Tasks*.

The **depth** $D$ of a terminal task is the smallest cap on the number of turns under which a model still solves it,
where one turn is one reply of the model, the commands in it, and the output it sees. This repository contains the
code for **measuring $D$** on existing tasks (Harbor tasks with one instruction) and on multi-segment tasks, under an
ordered model set. The pipeline that generates multi-segment tasks is not included in this release.

## Requirements

- Linux with Docker (BuildKit) and Python ≥ 3.10 with `requests` and `PyYAML`.
- [Harbor](https://github.com/laude-institute/harbor) 0.23.0 with the tmux patch in `patches/` (it only changes the
  failure path of sending long commands to tmux):
  ```bash
  uv tool install harbor==0.23.0
  SITE=$(~/.local/share/uv/tools/harbor/bin/python -c "import harbor, os; print(os.path.dirname(os.path.dirname(harbor.__file__)))")
  patch -p1 -d "$SITE" < patches/harbor-0.23.0-tmux_session.patch
  ```
  Set `DEPTH_HARBOR` if the `harbor` executable is not at `~/.local/bin/harbor`.
- A static tmux image, used for images whose package sources are offline:
  `docker build -t hd/tmux-static -f depth/tools/tmux-static.Dockerfile depth/tools`
- An OpenRouter key: `export OPENROUTER_API_KEY=...`. Every model is pinned to the providers listed in
  `depth/solvers.py`, with fallbacks disabled.

Results are written to `out/` (or `$DEPTH_OUT`). Commands can be rerun at any time: runs already recorded are not
repeated.

## Measurement protocol

The protocol is in `depth/protocol.py` and the rules in `depth/ladder.py`:

- Caps $\mathcal{T}=\{1,2,3,5,10,20,40,80\}$ for every task; the cap is stated to the agent and the run is stopped
  after it.
- `MODELS` is the ordered model set; `MODELS[0]` must be the model that runs without a cap (`MODEL`).
- **Runs without a cap** (`free`) come first and are separate from measuring $D$: `MODELS[0]` runs each task twice.
  If neither run passes, the task is not solved. Otherwise $N$ is the fewest turns among its passing runs.
- **Measuring $D$** (`ladder`): `MODELS[0]` tries the caps below $N$ from the smallest, up to two runs each (the second
  only if the first fails), and stops at the first cap passed; caps at or above $N$ count as passed. Each later model
  tries only the caps below the smallest estimate so far, in the same way, without runs without a cap.
  $D$ is the smallest estimate.

## Measuring $D$ on existing tasks

A task is a Harbor task directory with a single instruction; tasks are grouped by their parent directory.

```bash
python3 scripts/audit.py free   tasks/tb2/*      # two runs without a cap per task
python3 scripts/audit.py ladder tasks/tb2/*      # D under the model set
python3 scripts/audit.py report                  # N and D per task, distribution of D per dataset
```

Under cap $T$ the turn limit is appended to the instruction, and the agent time limit is 30 minutes.

## Multi-segment tasks

A multi-segment task is a Harbor multi-step task (`[[steps]]` in `task.toml`). Its segments are delivered one at a
time in one conversation and one container (`depth/harness/staged_agent.py`, an extension of terminus-2); the cap covers
all segments together, and each segment has 30 minutes.

```bash
python3 scripts/harness.py check  TASK...              # oracle passes every segment, doing nothing fails
python3 scripts/harness.py free   TASK...              # two runs without a cap (MODELS[0])
python3 scripts/harness.py ladder TASK...              # D under the model set
python3 scripts/harness.py report                      # N and D per task
python3 scripts/harness.py free TASK... --model M      # evaluate another registered model
python3 scripts/harness.py report --model M            # its pass rate
```

## Layout

| Path | Content |
| --- | --- |
| `depth/protocol.py`, `depth/ladder.py` | measurement protocol, order of caps, $D$ under the model set |
| `depth/solvers.py`, `depth/settings.py` | registered models, settings |
| `depth/sandbox.py`, `depth/taskenv.py` | Harbor trials in Docker, task copies and images |
| `depth/audit/` | measuring $D$ on existing tasks |
| `depth/harness/` | the multi-segment agent, trials and measuring $D$ on multi-segment tasks |
| `scripts/` | command-line entry points |
| `tests/` | self-tests (no model API calls) |
| `patches/` | the Harbor patch |

## Tests

```bash
python3 tests/audit_selftest.py                                          # seconds, no docker
SELFTEST_TOY_IMAGE=<image with bash, python3, tmux and /app> python3 tests/harness_selftest.py
```

## License

Apache License 2.0, see [LICENSE](LICENSE). `patches/harbor-0.23.0-tmux_session.patch` modifies Harbor 0.23.0, which is also distributed under the Apache License 2.0.
