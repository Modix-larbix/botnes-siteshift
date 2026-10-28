"""Observable feedback record of a task episode: what the robot side may give a model.

Built only from robot-side records: the public task statement, the program's own claim, the skill facade's call log
(what the program sent, results, reasons, the robot's own force and gripper readings, the robot clock), the
perception trace (anonymous track ids, categories, poses in the robot's believed frame, regions, coverage) and the
robot's site model (difference report and search scope), plus holding() at the end. It never receives the simulator,
the truth spec, the evaluator or its verdict.

contains_oracle_information is false only when no field, and no robot-side computation behind a field, uses an oracle
quantity; it is computed from perception.estimate.ORACLE_INPUTS, which must be empty. Declared and not an oracle
input: the simulated sensor decides which segments it reports with a truth-side visibility render, as part of the
simulated perception.
"""
from __future__ import annotations

from .perception.estimate import ORACLE_INPUTS

SCHEMA = "observable-feedback-1.2"
# visible_fraction is an ORACLE quantity of the simulated sensor (truth-side render). It is not in the feedback (schema
# 1.0 carried it) and is not used robot side at all (the robot-side visibility_estimate replaces
# it). n_pixels is sent.
DECLARED_ORACLE_FIELDS = ("visible_fraction",)
ORACLE_SCOPE = ("fields and the robot-side computations behind them, direct or indirect; the simulated sensor's "
                "own report threshold (truth-side visibility render) is part of the declared simulated perception")
DET_KEYS = ("track_id", "category", "region", "pose_source", "pose_step", "n_pixels")
ACTION_ARG_KEYS = ("region", "camera", "why", "attempt", "track_id", "pos", "yaw", "xy", "height_m", "expected",
                   "seconds", "force_n", "handle_pos", "pull_dir", "pose_source", "pose_step")


def public_task(crit: dict) -> dict:
    """The task statement the program is given (public part of the criteria config)."""
    return {"task_id": crit["task_id"], "text": crit["task_text"], "target_description": crit["target"],
            "scrub_request": crit["scrub_request"], "budget": crit["budget"], "criteria_version": crit["version"]}


def _obs(o: dict) -> dict:
    return {"obs_id": o["obs_id"], "camera": o["provenance"]["sensor"], "step": o["provenance"]["step"],
            "t_s": o["provenance"]["t_s"],
            "detections": [{**{k: d[k] for k in DET_KEYS if k in d}, "position_m": d["pose"]["position_m"],
                            "sigma_m": d["pose"]["sigma_m"], "yaw_rad": d["pose"].get("yaw_rad")}
                           for d in o["detections"]],
            "handles": [{"track_id": h["track_id"], "category": h["category"], "position_m": h["pose"]["position_m"],
                         "pull_dir": h["pull_dir"]} for h in o["handles"]],
            "regions": [{"region": r["region"], "coverage": r["coverage"], "seen_categories": r["seen_categories"]}
                        for r in o["regions"]]}


def _trace(trace: list[dict]) -> list[dict]:
    out = []
    for r in trace:
        e = {"step": r["step"], "action": r["action"],
             "args": {k: v for k, v in r["args"].items() if k in ACTION_ARG_KEYS},
             "result": {k: v for k, v in r["result"].items() if isinstance(v, (bool, int, float, str)) or v is None}}
        if r.get("note"):
            e["note"] = r["note"]
        if r.get("observation"):
            e["observation"] = _obs(r["observation"])
        out.append(e)
    return out


def observable_feedback(task: dict, program: str, claim: dict, skill_log: list[dict], budget_state: dict,
                        holding_at_end: bool, search_trace: list[dict] | None = None,
                        site_model=None) -> dict:
    """task: public_task(crit). skill_log: SkillFacade.log. budget_state: SkillFacade.budget_state()
    (robot clock and call counts). site_model: the robot's sitemodel.SiteModel, if the program has one."""
    calls = [{k: e[k] for k in ("i", "t_s", "t_end_s", "skill", "args", "kwargs", "ok", "reason", "data") if k in e}
             for e in skill_log]
    forces = [{"i": e["i"], "skill": e["skill"], **{k: e["data"][k] for k in ("force_mean", "felt_fraction", "t_start",
                                                                              "t_end") if k in e["data"]}}
              for e in skill_log if e["skill"] == "scrub"]
    fb = {"schema": SCHEMA, "label": "simulation, perception simulee",
          "contains_oracle_information": bool(ORACLE_INPUTS), "oracle_information_scope": ORACLE_SCOPE,
          "robot_side_oracle_inputs": list(ORACLE_INPUTS),
          "task": task, "program": program,
          "program_claim": {k: v for k, v in claim.items() if isinstance(v, (str, int, float, bool)) or v is None},
          "skill_calls": calls,
          "failed_calls": [{"i": c["i"], "skill": c["skill"], "reason": c["reason"]} for c in calls if not c["ok"]],
          "own_force_readings": forces,
          "robot_end_state": {"holding_something": bool(holding_at_end)},
          "budget_used": {k: budget_state[k] for k in ("sim_s_used", "max_sim_s", "skill_calls", "max_skill_calls",
                                                       "exceeded")}}
    if search_trace is not None:
        fb["perception_trace"] = _trace(search_trace)
    if site_model is not None:
        fb["site_knowledge"] = {"search_scope": site_model.search_scope("sponge"),
                                "difference_report": site_model.diff_report()}
    return fb
