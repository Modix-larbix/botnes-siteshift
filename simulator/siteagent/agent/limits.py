"""Limits of the agent: program resources, physical episode, agent time and optional caps.

The active profile is agent_limits_s5_v0_8.json (loaded by load_limits). Every limit is checked before the step it
bounds (instruction, skill call, observation, model call, correction) and nothing resets a counter to get around it.
In this profile null means no numeric cap on model calls, corrections, technical retries, episode cost, help reviews
and refused not_found claims; wall time, interpreter resources and physical budgets stay finite.

AgentLimitsV08 adds the per-episode limits of the non-final reprises (max_help_reviews, max_not_found_refusals) and
reprise_needs_new_observation: a reprise after the first one of the episode needs a new robot-side observation, so a
program that repeats a help or a not_found without looking cannot run until the wall time. They live in a subclass
because the declared identity of a profile hashes asdict(limits): new fields on AgentLimits would change the hash of
every older profile. Older profiles load as AgentLimits, and reprise_limit() gives them their original counts.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, fields
from decimal import Decimal, InvalidOperation
from pathlib import Path

LIMITS_FILE = Path(__file__).with_name("agent_limits_s5_v0_8.json")
LIMITS_FILE_V0_7 = Path(__file__).with_name("agent_limits_s5_v0_7.json")  # earlier profiles: not in this package
LIMITS_FILE_V0_6 = Path(__file__).with_name("agent_limits_s5_v0_6.json")
LIMITS_FILE_V0_5 = Path(__file__).with_name("agent_limits_s5_v0_5.json")
LIMITS_FILE_V0_4 = Path(__file__).with_name("agent_limits_s5_v0_4.json")
LIMITS_FILE_V0_3 = Path(__file__).with_name("agent_limits_s5_v0_3.json")
LIMITS_FILE_V0_2 = Path(__file__).with_name("agent_limits_s5_v0_2.json")
LIMITS_FILE_V0_1 = Path(__file__).with_name("agent_limits_s5_v0.json")


@dataclass(frozen=True)
class ProgramLimits:
    max_source_chars: int = 12000
    max_ast_nodes: int = 3000
    max_nesting_depth: int = 24
    max_steps: int = 20000
    max_loop_iterations: int = 200
    max_skill_calls: int = 60
    max_queries: int = 400
    max_interpreter_cpu_s: float = 5.0
    max_wall_s: float = 1800.0
    max_list_len: int = 500
    max_str_len: int = 4000
    max_value_items: int = 5000
    max_value_depth: int = 8
    max_int_abs: int = 10 ** 15
    max_log_lines: int = 100
    # botscript-0.3: top-level def functions. Not in the limits JSON files yet (defaults apply).
    max_functions: int = 20  # checked before running (too_many_functions)
    max_call_depth: int = 16  # nested calls at run time (limit max_call_depth); recursion is refused before running


@dataclass(frozen=True)
class EpisodeLimits:
    max_sim_s: float = 600.0
    max_skill_calls: int = 200
    max_observations: int = 40
    observation_sim_s_nominal: float = 1.0
    max_queries: int = 2000  # 0.2: robot-side readings per episode (0.1 had no cap)


@dataclass(frozen=True)
class AgentLimits:
    max_model_calls: int | None = 4
    max_corrections: int | None = 3
    max_technical_retries_per_run: int | None = 1
    max_completion_tokens: int = 8000  # 0.4 (0.3: 4000)
    max_prompt_chars: int = 56000  # 0.4 (0.3: 60000)
    temperature: float = 0.0
    max_episode_wall_s: float = 3600.0  # 0.3: checked before each model call or program run
    max_episode_cost_usd: str | None = "0.35"  # None disables only this monetary cap, never accounting
    correct_on: tuple = ("parse_error", "refused", "runtime_error", "limit", "give_up")  # 0.5 (0.4: + done_refused)
    reasoning_effort: str | None = None  # 0.6: Token Factory reasoning_effort sent with each request (None: not sent)

    def __post_init__(self):
        for name in ("max_model_calls", "max_corrections", "max_technical_retries_per_run"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a non-negative integer or None")
        if self.max_episode_cost_usd is not None:
            try:
                cost = Decimal(str(self.max_episode_cost_usd))
            except InvalidOperation:
                raise ValueError("max_episode_cost_usd must be finite and non-negative, or None") from None
            if not cost.is_finite() or cost < 0:
                raise ValueError("max_episode_cost_usd must be finite and non-negative, or None")
        if (type(self.max_episode_wall_s) not in (int, float)
                or not math.isfinite(self.max_episode_wall_s) or self.max_episode_wall_s < 0):
            raise ValueError("max_episode_wall_s must remain finite and non-negative")


# 0.8: the per-episode limits of the non-final reprises (agent.loop) and their earlier values
REPRISE_DEFAULTS = {"max_help_reviews": 1, "max_not_found_refusals": 2}
REPRISE_PROGRESS = "reprise_needs_new_observation"  # False for a profile without it


@dataclass(frozen=True)
class AgentLimitsV08(AgentLimits):
    """0.8: AgentLimits plus max_help_reviews (reviews of help(kind="search_incomplete") per episode)
    and max_not_found_refusals (refused not_found claims followed by a correction per episode). None: no numeric cap
    (agent_limits_s5_v0_8.json). The defaults are the earlier values (REPRISE_DEFAULTS). Plus
    reprise_needs_new_observation (True in 0.8; default False, the earlier behaviour): a reprise after the first of
    the episode needs a new robot-side observation since the previous one (agent.loop.reprise_has_progress)."""
    max_help_reviews: int | None = 1
    max_not_found_refusals: int | None = 2
    reprise_needs_new_observation: bool = False

    def __post_init__(self):
        super().__post_init__()
        for name in REPRISE_DEFAULTS:
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a non-negative integer or None")
        if type(self.reprise_needs_new_observation) is not bool:
            raise ValueError("reprise_needs_new_observation must be true or false")


def reprise_limit(agent, name: str) -> int | None:
    """0.8: the profile's max_help_reviews or max_not_found_refusals; the earlier value for a profile without it."""
    return getattr(agent, name, REPRISE_DEFAULTS[name])


def reprise_needs_new_observation(agent) -> bool:
    """The profile's reprise_needs_new_observation; False for a profile without it (0.7 and
    before: their counts bound the reprises, as before)."""
    return getattr(agent, REPRISE_PROGRESS, False) is True


@dataclass(frozen=True)
class Limits:
    version: str
    program: ProgramLimits = field(default_factory=ProgramLimits)
    episode: EpisodeLimits = field(default_factory=EpisodeLimits)
    agent: AgentLimits = field(default_factory=AgentLimits)


def _take(cls, d: dict):
    names = {f.name for f in fields(cls)}
    kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in names}
    return cls(**kw)


def load_limits(path: str | Path | None = None) -> Limits:
    d = json.loads(Path(path or LIMITS_FILE).read_text())
    agent_cls = AgentLimitsV08 if any(k in d["agent"] for k in (*REPRISE_DEFAULTS, REPRISE_PROGRESS)) \
        else AgentLimits  # 0.8
    return Limits(version=d["version"], program=_take(ProgramLimits, d["program"]),
                  episode=_take(EpisodeLimits, d["episode"]), agent=_take(agent_cls, d["agent"]))
