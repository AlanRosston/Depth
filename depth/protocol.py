"""The measurement protocol, shared by existing tasks (depth.audit) and multi-segment tasks (depth.harness).
How caps are tried and how D is taken is in depth.ladder."""

# The model that runs every task without a cap: its passing runs give N and decide which tasks are solved.
MODEL = "openrouter/deepseek/deepseek-v4.1-flash"

# The model set, measured in this order. Each later model tries only the caps below the smallest estimate of the
# models before it; D is the smallest estimate. Every model must be registered in depth.solvers.
MODELS = [
    "openrouter/deepseek/deepseek-v4.1-flash",
    "openrouter/z-ai/glm-5.3-flash",
]
assert MODELS[0] == MODEL, "MODELS[0] must be the model that runs without a cap (MODEL)"

LADDER = [1, 2, 3, 5, 10, 20, 40, 80]   # caps on the number of turns, tried from the smallest
K = 2                                   # runs without a cap, and at most this many runs per cap
UNLIMITED = 150                         # the turn limit of a run "without a cap"
AGENT_TIMEOUT = 1800                    # seconds: agent time limit of an existing task (replacing the dataset's),
                                        # and of each segment of a multi-segment task
