"""Agent for multi-segment tasks: terminus-2 that delivers the segments of a Harbor multi-step task one at a time
in one conversation and one container.

Differences from terminus-2:
1. resume(): from segment 2 on, the instruction of the segment and the current terminal screen are appended to the
   conversation as a new user message.
2. The "Are you sure?" confirmation is asked only in the last segment.
3. max_turns covers all segments together; segments reached with no turns left are graded without a turn.
   With announce_turn_limit=true the cap is stated in the first message and each segment reports the turns used.
4. The tests of a segment are removed from the container before the next segment starts.
5. HTTP 429 is retried after the waits in RL_WAITS; EMPTY_LIMIT empty replies in a row raise EmptyResponsesError.

Usage:
    harbor trial start -p <task> -a depth.harness.staged_agent:StagedTerminus --resume-trajectory \
        --agent-kwarg n_stages=6 --agent-kwarg max_turns=20 --agent-kwarg announce_turn_limit=true ...
With n_stages unset or 1 the agent is terminus-2.
"""
import asyncio
from datetime import datetime, timezone
from typing import override

from litellm.exceptions import RateLimitError
from pydantic import Field

from harbor.agents.capabilities import AgentCapabilities
from harbor.agents.terminus_2.terminus_2 import Terminus2, Terminus2Options
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trajectories import Step
from harbor.models.trial.paths import EnvironmentPaths

PREAMBLE = """This task is handed to you in {k} parts, one at a time, in this same terminal session. Below is part 1.
When a part is done, set "task_complete": true; the next part is then sent to you. Later parts build on what
earlier parts left in the environment. Each part is graded when you mark it complete, and you are not told the
grade; the task passes only if every part does.{limit}

Part 1 of {k}:

{instruction}"""

LIMIT = """

Turn limit: you have at most {T} turns for the whole task, all {k} parts together. One turn is one of your
responses; a response may contain several commands. The session is cut off after turn {T}; parts you have not
finished by then fail. Plan for this limit: batch commands where you can, and do not spend turns on steps that
are not needed."""

NEXT_PART = """Part {prev} is marked complete. Here is part {i} of {k}{last}:

{instruction}
{used}
Current terminal state:
{terminal_state}"""

USED = "\nTurns used so far: {u} of {T}.\n"

RL_WAITS = (20, 40, 60, 60, 60)     # seconds before each retry after HTTP 429
EMPTY_LIMIT = 5                     # empty replies in a row that end the trial


class EmptyResponsesError(RuntimeError):
    """Raised after EMPTY_LIMIT empty replies in a row."""


class StagedOptions(Terminus2Options):
    n_stages: int | None = Field(default=None, description="Number of segments ([[steps]] in task.toml).")
    announce_turn_limit: bool = Field(default=False, description="State max_turns and report the turns used.")


class StagedTerminus(Terminus2):
    capabilities = AgentCapabilities(atif=True, resume=True)
    options_model = StagedOptions
    options: StagedOptions

    def __init__(self, *args, n_stages: int | None = None, announce_turn_limit: bool = False, **kwargs):
        super().__init__(*args, n_stages=n_stages, announce_turn_limit=announce_turn_limit, **kwargs)
        self._cap = self.options.max_turns          # None: no cap
        self._empty_streak = 0
        self._stage = 0
        self._turns_used = 0
        self._instructions: list[str] = []

    @staticmethod
    @override
    def name() -> str:
        return "staged-terminus-2"

    def _k(self) -> int:
        return self.options.n_stages or 1

    def _final(self) -> bool:
        return self._stage >= self._k()

    def _remaining(self) -> int:
        return 1000000 if self._cap is None else self._cap - self._turns_used

    def _announce(self) -> bool:
        return self.options.announce_turn_limit and self._cap is not None

    def _totals(self) -> tuple[int, int, int, float]:
        c, s = self._chat, self._subagent_metrics
        return (c.total_input_tokens + s.total_prompt_tokens, c.total_output_tokens + s.total_completion_tokens,
                c.total_cache_tokens + s.total_cached_tokens, c.total_cost + s.total_cost_usd)

    def _stage_meta(self, context: AgentContext, skipped: bool = False) -> None:
        meta = dict(context.metadata or {})
        if "api_request_times_msec" in meta:
            meta["api_request_times_msec"] = list(meta["api_request_times_msec"])
        meta.update({"n_episodes": 0 if skipped else self._n_episodes, "part": self._stage, "n_parts": self._k(),
                     "turns_total": self._turns_used, "turn_cap": self._cap, "skipped_no_turns_left": skipped})
        context.metadata = meta

    @override
    async def _query_llm(self, chat, prompt, original_instruction="", session=None):
        """terminus-2's _query_llm, retried after HTTP 429 with the waits in RL_WAITS."""
        for wait in RL_WAITS:
            try:
                return await super()._query_llm(chat, prompt, original_instruction, session)
            except Exception as e:                       # noqa: BLE001
                if not (isinstance(e, RateLimitError) or getattr(e, "status_code", None) == 429):
                    raise
                self.logger.warning(f"rate limited (429), retrying in {wait} s: {str(e)[:200]}")
                await asyncio.sleep(wait)
        return await super()._query_llm(chat, prompt, original_instruction, session)

    @override
    async def _handle_llm_interaction(self, chat, prompt, original_instruction="", session=None):
        out = await super()._handle_llm_interaction(chat, prompt, original_instruction, session)
        self._empty_streak = 0 if (out[-1].content or "").strip() else self._empty_streak + 1
        if self._empty_streak >= EMPTY_LIMIT:
            raise EmptyResponsesError(f"{EMPTY_LIMIT} empty replies in a row; ending the trial")
        if not self._final():
            self._pending_completion = True
        return out

    @override
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        self._stage, self._turns_used, self._instructions = 1, 0, [instruction]
        if self._cap is not None:
            self._max_episodes = self._cap
        text = instruction
        if self._k() > 1:
            limit = LIMIT.format(T=self._cap, k=self._k()) if self._announce() else ""
            text = PREAMBLE.format(k=self._k(), limit=limit, instruction=instruction)
        try:
            await super().run(text, environment, context)
        finally:
            self._turns_used = self._n_episodes
            self._stage_meta(context)

    @override
    async def resume(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        if self._chat is None or self._session is None:
            raise RuntimeError("resume() called before run(): segment 1 of a multi-segment task must use run()")
        if self._empty_streak >= EMPTY_LIMIT:
            raise EmptyResponsesError(f"an earlier segment ended after {EMPTY_LIMIT} empty replies")
        self._stage += 1
        self._instructions.append(instruction)
        self._context = context
        await environment.exec(f"rm -rf {EnvironmentPaths.tests_dir}", user="root")

        left = self._remaining()
        if left <= 0:
            self._n_episodes = 0
            self._stage_meta(context, skipped=True)
            return

        self._max_episodes = left
        self._pending_completion = False
        self._n_episodes = 0
        terminal_state = self._limit_output_length(await self._session.get_incremental_output())
        prompt = NEXT_PART.format(
            prev=self._stage - 1, i=self._stage, k=self._k(), last=" (the last part)" if self._final() else "",
            instruction=instruction, terminal_state=terminal_state,
            used=USED.format(u=self._turns_used, T=self._cap) if self._announce() else "")
        self._trajectory_steps.append(Step(step_id=len(self._trajectory_steps) + 1,
                                           timestamp=datetime.now(timezone.utc).isoformat(),
                                           source="user", message=prompt))
        before, n_req = self._totals(), len(self._api_request_times)
        try:
            await self._run_agent_loop(initial_prompt=prompt, chat=self._chat,
                                       original_instruction="\n\n".join(self._instructions))
        finally:
            after = self._totals()
            context.n_input_tokens, context.n_output_tokens, context.n_cache_tokens = (
                after[0] - before[0], after[1] - before[1], after[2] - before[2])
            cost = after[3] - before[3]
            context.cost_usd = cost if cost > 0 else None
            context.metadata = {"api_request_times_msec": self._api_request_times[n_req:],
                                "summarization_count": self._summarization_count}
            self._turns_used += self._n_episodes
            self._stage_meta(context)
            self._dump_trajectory()
