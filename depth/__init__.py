"""DEPTH: measure the depth of terminal tasks.

Library modules (command-line entry points are in scripts/):
  protocol    the measurement protocol: model set, caps, runs per cap, turn and time limits
  ladder      result files, the order in which caps are tried, and the depth D of a task under the model set
  solvers     registered models and the providers they are pinned to on OpenRouter
  settings    environment-variable settings (API key, Harbor path, concurrency, resource guards)
  sandbox     running one Harbor trial in local Docker, with concurrency, disk and memory guards
  taskenv     task directories and images: pre-installing tmux, normalizing task.toml
  paths       where results are written (DEPTH_OUT, default ./out)
Subpackages:
  harness     multi-segment tasks: the agent that delivers segments one at a time, trials, measuring D
  audit       measuring D on existing single-instruction tasks
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
