"""Scripted adaptive chain: search and pick the sponge, then scrub the perceived bowl. No language model.

A programmed reference with the same skills, calibration and budget as the other programs.
1. siteagent.search.SiteSearch.run(): calibrate, look, read the dossier, search the candidate regions in a bounded
   order, pick the sponge from its perceived pose.
2. If the sponge is held: take the bowl track on the landing tray (one wrist look down at the tray if none was seen)
   and scrub its top surface for 3 s at 4 N.
Claims: "task_done", "not_found_in_inspected_scope", "not_found_scope_incomplete", "help_requested:not_reachable",
or "failed:<stage>:<reason>". The evaluator decides.
"""
from __future__ import annotations

import numpy as np

from ..perception.catalog import MODULE_SURFACES, OBJECT_SHAPES
from ..perception.contract import OBSERVED, STALE
from ..search import SiteSearch, _skill_rec

NAME = "scripted_adaptive_chain"
SCRUB_S, SCRUB_N = 3.0, 4.0


def _bowl_track(per, robot):
    ts = [t for t in per.tracks_of("bowl", (OBSERVED, STALE)) if t.region == "landing_tray"]
    if not ts:
        return None
    b = robot.belief.t[:2]
    return min(ts, key=lambda t: float(np.linalg.norm(t.pos[:2] - b)))


def run(robot, perception, dossier, hook=None) -> tuple[dict, SiteSearch]:
    """robot: the skill facade; perception: perception.Perception built on the same facade."""
    s = SiteSearch(robot, perception, dossier, target="sponge", hook=hook)
    res = s.run()
    out = res["outcome"]
    claim = {"search_outcome": out, "search_steps": res["n_steps"]}
    if out in ("not_found_in_inspected_scope", "not_found_scope_incomplete"):
        return dict(claim, claim=out), s
    if out.startswith("found_not_reachable"):
        return dict(claim, claim="help_requested:not_reachable"), s
    if out.startswith("found_pick_failed:"):
        return dict(claim, claim="failed:pick:" + out.split(":", 1)[1]), s
    if out != "found_and_picked":
        return dict(claim, claim=out if out.startswith("failed:") else f"failed:search:{out}"), s
    bowl = _bowl_track(perception, robot)
    if bowl is None:
        tray = next((t for t in perception.tracks.values() if t.category == "landing_tray"), None)
        if tray is not None:
            z = robot.worktop_z + MODULE_SURFACES["landing_tray"]["z"]
            rr = s.look_down_at(tray.pos[:2], z, height=0.55)
            if rr is not None and rr.ok:
                s._observe("wrist_cam", "find the bowl on the landing tray")
        bowl = _bowl_track(perception, robot)
    if bowl is None:
        s._record("find_target", {}, {"ok": False, "reason": "bowl_not_seen"})
        return dict(claim, claim="failed:find_bowl:not_seen"), s
    top = bowl.pos + np.array([0.0, 0.0, OBJECT_SHAPES["bowl"]["top_above_centre"]])
    r = robot.scrub(top, SCRUB_S, SCRUB_N)
    perception.note_action("scrub", {"ok": bool(r.ok)})
    s._record("scrub", {"track_id": bowl.track_id, "pos": [float(x) for x in top], "seconds": SCRUB_S,
                        "force_n": SCRUB_N, "pose_source": bowl.pose_source, "pose_step": bowl.pose_step},
              _skill_rec(r))
    if not r.ok:
        return dict(claim, claim=f"failed:scrub:{r.reason}"), s
    return dict(claim, claim="task_done"), s
