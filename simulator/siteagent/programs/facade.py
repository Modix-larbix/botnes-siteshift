"""Skill facade for task programs: episode budget and robot-side call log.

A task program receives this facade, not the Robot. It exposes the skills of robot.skills and a few robot-side
readings (belief, worktop height, joint command, believed tool pose, holding()). It has no simulator attribute, but
this is not isolation: the readings return the robot's own objects and Python introspection reaches private fields.
The agent's programs are isolated by the sandbox instead.

Budget (criteria config, "budget"): checked before each skill call; over the simulated time or the number of calls,
the call is refused with reason "budget_exceeded" and the arm does not move. A skill already running finishes. The
counters are read-only; the evaluator counts calls and time itself through the bench-held hook.

Call log: one record per skill call (name, rounded arguments, ok, reason, whitelisted robot-side data, robot clock).
facade.log returns a copy; the bench receives its own deep copy of every record.
"""
from __future__ import annotations

import copy
import math

from ..robot.skills import SkillResult

SKILLS = ("calibrate", "go_home", "look_pose", "move_to", "open_gripper", "close_gripper", "pick", "place",
          "turn_over", "open_drawer", "close_drawer", "press", "spray", "scrub")
READINGS = ("belief", "worktop_z", "q_cmd", "tcp_pose", "holding")
# Robot-side result fields kept in the log (scalars only).
DATA_KEYS = ("moved", "asked", "strokes", "width", "grasped", "arm_moved", "error", "where", "force_n", "stage",
             "end_stop", "expected", "felt_fraction", "force_mean", "t_start", "t_end", "stop_dz", "lift",
             "force_peak", "felt_s", "gesture")


def _clean(x, depth=0):
    """Rounded, JSON-safe copy of a call argument (numbers to 0.1 mm / 1e-4)."""
    if depth > 3:
        return str(type(x).__name__)
    if isinstance(x, bool) or x is None or isinstance(x, str):
        return x
    if isinstance(x, (int, float)):
        return round(float(x), 4) if math.isfinite(float(x)) else None
    if hasattr(x, "tolist"):
        return _clean(x.tolist(), depth + 1)
    if isinstance(x, (list, tuple)):
        return [_clean(v, depth + 1) for v in x]
    if isinstance(x, dict):
        return {str(k): _clean(v, depth + 1) for k, v in x.items()}
    return str(type(x).__name__)


class SkillFacade:
    def __init__(self, robot, budget: dict, bench=None):
        """bench: bench-side hook (taskcheck.BenchEvidence or None) with start(name) and end(record);
        it gets a deep copy of every record, refused calls included."""
        st = {"max_sim_s": float(budget["max_sim_s"]), "max_calls": int(budget["max_skill_calls"]),
              "n_calls": 0, "exceeded": None, "log": []}
        object.__setattr__(self, "_SkillFacade__robot", robot)
        object.__setattr__(self, "_SkillFacade__bench", bench)
        object.__setattr__(self, "_SkillFacade__st", st)
        st["t0"] = self._clock()

    def __setattr__(self, name, value):
        raise AttributeError(f"the skill facade is read-only (cannot set {name!r})")

    def __delattr__(self, name):
        raise AttributeError(f"the skill facade is read-only (cannot delete {name!r})")

    # Read-only views of the robot-side counters (the verifier does not use them).
    max_sim_s = property(lambda self: self.__st["max_sim_s"])
    max_calls = property(lambda self: self.__st["max_calls"])
    n_calls = property(lambda self: self.__st["n_calls"])
    exceeded = property(lambda self: self.__st["exceeded"])
    t0 = property(lambda self: self.__st["t0"])
    log = property(lambda self: copy.deepcopy(self.__st["log"]))

    def _clock(self) -> float:
        return float(self.__robot.sim.t)  # the robot's own clock (simulated time)

    def budget_state(self) -> dict:
        st = self.__st
        used = self._clock() - st["t0"]
        exceeded = st["exceeded"]
        if exceeded is None and used > st["max_sim_s"]:
            exceeded = "max_sim_s"
        return {"sim_s_used": round(used, 1), "max_sim_s": st["max_sim_s"], "skill_calls": st["n_calls"],
                "max_skill_calls": st["max_calls"], "exceeded": exceeded}

    def __getattr__(self, name):
        if name in READINGS:
            return getattr(self.__robot, name)
        if name in SKILLS:
            fn = getattr(self.__robot, name)

            def call(*args, **kwargs):
                return self._call(name, fn, args, kwargs)
            return call
        raise AttributeError(f"the skill facade has no {name!r}")

    def _call(self, name, fn, args, kwargs) -> SkillResult:
        st, bench = self.__st, self.__bench
        st["n_calls"] += 1
        if bench is not None:
            bench.start(name)
        t = self._clock()
        rec = {"i": st["n_calls"], "t_s": round(t, 2), "skill": name, "args": _clean(list(args)),
               "kwargs": _clean(kwargs)}
        if st["n_calls"] > st["max_calls"]:
            st["exceeded"] = st["exceeded"] or "max_skill_calls"
        if t - st["t0"] > st["max_sim_s"]:
            st["exceeded"] = st["exceeded"] or "max_sim_s"
        r = None
        try:
            if st["exceeded"]:
                r = SkillResult(False, "budget_exceeded", {"arm_moved": False})
            else:
                r = fn(*args, **kwargs)
        finally:
            if r is None:  # the skill raised: the bench still closes its record
                rec.update({"ok": False, "reason": "internal_error", "data": {"raised": True},
                            "t_end_s": round(self._clock(), 2)})
                if bench is not None:
                    bench.end(copy.deepcopy(rec))
        rec.update({"ok": bool(r.ok), "reason": r.reason,
                    "data": {k: _clean(r.data[k]) for k in DATA_KEYS
                             if k in r.data and isinstance(r.data[k], (bool, int, float, str))},
                    "t_end_s": round(self._clock(), 2)})
        st["log"].append(rec)
        if bench is not None:
            bench.end(copy.deepcopy(rec))
        return r
