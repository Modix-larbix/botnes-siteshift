"""Scripted search for a tool in the kitchen. No language model.

The policy uses only the robot's skills, its perception (perception.estimate) and the site dossier; target poses come
from perception tracks, never from the bench. The bench may pass a hook called after every step to log hidden truth
on its side, apart from this trace.

Policy for one category (default "sponge"):
1. calibrate, look pose, observe with robot_eye;
2. if fewer drawer handles are perceived than the dossier declares, look down at the drawer unit with the wrist camera;
3. if the tool is seen, pick it; otherwise visit the candidate regions in a bounded order (the declared region first,
   then the other drawers left to right, then the open surfaces not yet covered). A drawer is opened by its perceived
   handle, observed with the wrist camera and closed again when the tool is not there. After a failed pull the robot
   looks for the handle before trusting any opening, and pulls again at most once. When the unseen part of a region
   could still hold the tool (sitemodel.could_hide), the robot looks once more over the unseen part;
4. stop when the tool is seen, the bounded scope is exhausted, or the budget is reached;
5. pick from the perceived pose (top grasp across the perceived long axis). If the tool lies too close to the unit
   top (less than MIN_TOP_CLEARANCE), pull the drawer to its full travel and look again; still too close: give up with
   "found_not_reachable:under_unit_top". After a failed pick, retry only on a fresh sighting.

Budget: max_regions is checked before each region; max_steps is a strict cap on observations and actions, enforced by
one guard before every operation (an operation starts only if it ends within the cap; a look reserves its move and
its observation together). Outcomes: found, not_found_in_inspected_scope (only when sitemodel.search_scope says so),
not_found_scope_incomplete, or a budget stop.
"""
from __future__ import annotations

import inspect
import math
from dataclasses import dataclass

import numpy as np

from .perception.catalog import (DRAWER_FLOOR_BELOW_HANDLE, DRAWER_MAX_OPENING, DRAWER_TOP_EDGE_BEHIND_CLOSED_HANDLE,
                                 MODULE_SURFACES, SINK_FAUCET_TOP, SINK_HAND_CLEARANCE, SINK_LOOK_FORWARD,
                                 SURFACE_REGION)
from .perception.contract import HELD, OBSERVED, STALE, to_dict
from .robot.design import wrist_cam_in_hand
from .robot.kinematics import TCP_OFFSET, hand_down
from .sitemodel import could_hide, frame_accepted, is_contact_reason, open_placements

KEEP_KEYS = ("moved", "asked", "strokes", "width", "arm_moved", "error", "where", "force_n", "stage",
             "end_stop", "expected", "felt_fraction")
# Versions (recorded by the harness and in the look records). Corrections of : look-unseen-2 -> look-unseen-3 (the
# tilted look arms the retreat out of its pose, robot.skills tilt-retreat-1)
LOOK_UNSEEN_VERSION = "look-unseen-3"
UNSEEN_AIM_VERSION = "unseen-placement-1"
OVERHEAD_REFUSED_OUTCOME = "found_pick_failed:overhead_obstruction"


def _plain(v):
    """A plain scalar (numpy scalars cast), a list of them, or None for anything else."""
    if v is None or isinstance(v, (bool, str)):
        return v
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if math.isfinite(float(v)) else None
    if isinstance(v, (list, tuple, np.ndarray)) and len(v) <= 8:
        return [_plain(x) for x in list(v)]
    return None


def plain_record(d: dict) -> dict:
    """A record of plain values (search records, host look records): non-plain entries are dropped."""
    out = {}
    for k, v in (d or {}).items():
        if k == "result" or not isinstance(k, str):
            continue
        p = _plain(v)
        if p is not None or v is None:
            out[k] = p
    return out


def reason_detail(r) -> dict:
    """contact-invalidates-1: the detail a perception note carries for a skill result: its reason (and data.lift for a
    lift_blocked) when it reports a contact (sitemodel.is_contact_reason), else {} (a success, or a failure without a
    contact: the note stays as before)."""
    if r is None or getattr(r, "ok", False):
        return {}
    data = getattr(r, "data", None)
    lift = data.get("lift") if isinstance(data, dict) else None
    reason = getattr(r, "reason", None)
    if not is_contact_reason(reason, lift):
        return {}
    return {"reason": str(reason), **({"lift": lift} if isinstance(lift, str) else {})}


def note_skill_reason(per, r):
    """contact-invalidates-1: attach a failed skill's reason to the action perception just noted for it (drawer and pick
    notes carry no reason in their signature; SiteModel.note_reason). A model without note_reason (test doubles) or a
    successful result or a failure without a contact: nothing."""
    d = reason_detail(r)
    model = getattr(per, "model", None)
    fn = getattr(model, "note_reason", None)
    if d and callable(fn):
        fn(int(per.step), d["reason"], d.get("lift"))


def top_clearance(per, t):
    """(handle track, clearance in m of the target centre in front of the unit top's edge) for a target in a drawer,
    else (None, None). Robot side: the handle's first (closed) view and the drawer unit catalog. (SiteSearch._pick's
    reachability rule; a module function, so the host reads the same rule, agent.host.FacadeBackend.)"""
    if not isinstance(getattr(t, "region", None), str) or not t.region.startswith("drawer_unit:"):
        return None, None
    h = per.handle_for_region(t.region)
    if h is None or h.closed_pos is None:
        return None, None
    pull = np.asarray(h.pull[:2]) / max(np.linalg.norm(h.pull[:2]), 1e-9)
    edge = h.closed_pos[:2] - DRAWER_TOP_EDGE_BEHIND_CLOSED_HANDLE * pull
    return h, float((t.pos[:2] - edge) @ pull)


def overhead_reading(per, t, pick_pos, kind: str | None = None, yaw: float | None = None):
    """pick-pose-3: Perception.overhead_check(track, pick_pos), the one robot-side check of a
    structure over the pick target (rule B on the view that set the target's pose; rule F as
    perception.overhead.RULE_F_ACTIVE says), for the scripted chain (SiteSearch.overhead_check) and the host
    (agent.host.FacadeBackend.overhead_check). kind and yaw (the grasp's finger yaw) are passed when the check takes
    them (rule F reads the finger column). Returns (refuse, plain record or None). Fail-closed: a check that raises
    or answers without a boolean "refuse" refuses (code "overhead_unreadable"). A perception without the check (test
    doubles): (False, None), unchecked. Because rule F may be active, plain_record drops the check's nested
    "rules" block, so rule F's count is kept as rule_f_points (an int, or None when rule F was not evaluated);
    n_points stays rule B's count. The host reports rule F's count when rule F alone refused
    (agent.host.GuardedHost.pick_precondition)."""
    fn = getattr(per, "overhead_check", None)
    if not callable(fn):
        return False, None
    try:
        params = inspect.signature(fn).parameters
        kw = {k: v for k, v in (("kind", kind), ("yaw", yaw)) if k in params and v is not None}
    except (TypeError, ValueError):
        kw = {}
    try:
        res = fn(t, np.asarray(pick_pos, float), **kw)
    except Exception as e:  # noqa: BLE001 (never a blind pick: an unreadable check refuses)
        return True, {"refuse": True, "code": "overhead_unreadable", "error_type": type(e).__name__[:60]}
    if not isinstance(res, dict) or not isinstance(res.get("refuse"), (bool, np.bool_)):
        return True, {"refuse": True, "code": "overhead_unreadable"}
    rec = plain_record(res)
    rec["refuse"] = bool(res["refuse"])
    rf = res.get("rules")
    rf = rf.get("F") if isinstance(rf, dict) else None
    if isinstance(rf, dict) and "rule_f_points" not in rec:  # rule F's count, flattened (records only)
        n = rf.get("n")
        ok = isinstance(n, (int, np.integer)) and not isinstance(n, (bool, np.bool_))
        rec["rule_f_points"] = int(n) if ok else None
    return rec["refuse"], rec


RETREAT_OUTCOMES = ("ok", "planning_failed", "failed", "aborted", "dropped")


def _takes_keyword(fn, name: str) -> bool:
    """Does callable fn accept keyword `name` (by name or through **kwargs)? False when its signature is unreadable."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind == p.VAR_KEYWORD for p in params.values())


def retreat_records(r) -> list[dict]:
    """tilt-retreat-1: the retreat events a robot result reports (data["retreat"], robot.skills), as plain records
    with "skill_contact": whether the skill that carried the retreat ended on a contact (sitemodel.is_contact_reason of
    its result). [] for a result without any."""
    data = getattr(r, "data", None)
    evs = data.get("retreat") if isinstance(data, dict) else None
    if not isinstance(evs, list):
        return []
    out = []
    for ev in evs:
        if not isinstance(ev, dict):
            continue
        rec = plain_record(ev)
        rec["skill_contact"] = bool(is_contact_reason(getattr(r, "reason", None), data.get("lift")))
        out.append(rec)
    return out


def retreat_summary(events: list[dict]) -> dict:
    """tilt-retreat-1 (records only): counts of retreat records by outcome, the planning failures and how many of
    them fell back to a skill that ended on a contact (the gate reading)."""
    out = {o: 0 for o in RETREAT_OUTCOMES}
    out["n"] = 0
    out["planning_failed_then_contact"] = 0
    for e in events or []:
        if not isinstance(e, dict):
            continue
        out["n"] += 1
        o = e.get("outcome")
        if o in out:
            out[o] += 1
        if o == "planning_failed" and e.get("skill_contact") is True:
            out["planning_failed_then_contact"] += 1
    return out


def _frame_centre(fr):
    """The world xy centre of a fitted module frame (Perception.module_frame): a mapping's "centre" (or "center"), the
    first item of a (centre, x axis, ...) sequence, or a 2- or 3-vector; None when unreadable."""
    c = None
    if isinstance(fr, dict):
        c = fr.get("centre", fr.get("center"))
    elif isinstance(fr, (list, tuple)) and fr and isinstance(fr[0], (list, tuple, np.ndarray)):
        c = fr[0]
    elif isinstance(fr, (list, tuple, np.ndarray)) and len(fr) in (2, 3):
        c = fr
    try:
        c = np.asarray(c, float).ravel()
    except (TypeError, ValueError):
        return None
    if c.size < 2 or not np.all(np.isfinite(c[:2])):
        return None
    return c[:2]


@dataclass
class SearchBudget:
    max_regions: int = 6  # regions opened or looked at after the first scan
    max_steps: int = 80  # perception step counter (observations + actions)
    pick_attempts: int = 2


def _skill_rec(r) -> dict:
    d = {k: r.data[k] for k in KEEP_KEYS if k in r.data and isinstance(r.data[k], (int, float, str, bool))}
    return {"ok": bool(r.ok), "reason": r.reason, **d}


class SiteSearch:
    def __init__(self, robot, perception, dossier, target: str = "sponge", budget: SearchBudget | None = None,
                 hook=None):
        self.robot = robot
        self.per = perception
        self.model = perception.model
        self.dossier = dossier
        self.target = target
        self.budget = budget or SearchBudget()
        self.hook = hook
        self.trace: list[dict] = []
        self.outcome = None
        self.action_starts: list[tuple[str, int]] = []  # (task action, perception step at its start)
        self.observation_starts: list[tuple[str, int]] = []  # (camera, perception step at its start)
        self.clearance_starts: list[tuple[str, int]] = []  # (clearance gesture, step at its start), unguarded
        self.clearance_steps = 0  # perception steps used by the clearance gestures
        self.refused: list[dict] = []  # operations the guard refused (budget)
        self.opened: dict[str, object] = {}  # region -> handle track of the drawers the robot pulled
        self.drawers_state_unknown: list[str] = []  # opening unknown and no look allowed to measure it
        self._running = False  # the guard of _observe applies while run() is searching
        self.retreats: list[dict] = []  # tilt-retreat-1: the retreat records of the results received (records)

    def _note_retreat(self, r):
        """tilt-retreat-1: keep the retreat events a robot result reports (retreat_records) in self.retreats and
        the trace (action "retreat"). No perception step, no motion."""
        for rec in retreat_records(r):
            self.retreats.append(rec)
            self._record("retreat", {k: rec.get(k) for k in ("version", "skill", "aim", "to") if k in rec},
                         {k: rec.get(k) for k in ("ok", "reason", "outcome", "fallback", "skill_ok", "skill_reason",
                                                  "skill_contact") if k in rec})

    # ------------------------------------------------------------ trace
    def _knowledge(self) -> dict:
        sc = self.model.search_scope(self.target)
        regions = {r: {"status": g["status"], "coverage": g["last"]["coverage"] if g["last"] else 0.0,
                       "seen": g["last"]["seen_categories"] if g["last"] else []}
                   for r, g in self.model.regions.items()}
        return {"target_outcome_so_far": sc["outcome"], "target_tracks": sc["found_tracks"], "regions": regions}

    def _record(self, action: str, args: dict | None = None, result: dict | None = None, obs=None,
                note: str = ""):
        rec = {"step": self.per.step, "action": action, "args": args or {}, "result": result or {},
               "observation": None if obs is None else to_dict(obs), "knowledge": self._knowledge()}
        if note:
            rec["note"] = note
        self.trace.append(rec)
        if self.hook is not None:
            self.hook(rec)
        return rec

    def _observe(self, camera: str, why: str):
        """One observation. While the search runs, the common guard is checked first (backstop: the
        look that precedes an observation already reserved its step); returns None if refused."""
        if self._running and not self._guard("observe", 1, camera=camera, why=why):
            return None
        self.observation_starts.append((camera, self.per.step))
        obs = self.per.observe(camera)
        self._record("observe", {"camera": camera, "why": why}, {"ok": True, "n_detections": len(obs.detections),
                                                                  "n_handles": len(obs.handles)}, obs)
        return obs

    # ------------------------------------------------------------ viewpoints
    def look_down_at(self, xy, z_floor, height: float = 0.42, min_tcp_z: float | None = None):
        """Move the hand so the wrist camera is `height` above (xy, z_floor), looking down.
        min_tcp_z : the TCP stays at or above this height (the camera is raised when needed), e.g. above a
        sink's faucet. A move stopped by the collision reflex ends the look (Fix: other angles were tried into the
        same obstacle). contact-invalidates-1: the note carries the contact reason of any attempt (a blocked pose
        before a pose that succeeded included); a look without contact is noted as before."""
        p_h, _ = wrist_cam_in_hand()
        off = p_h - np.array([0.0, 0.0, TCP_OFFSET])
        bt = self.robot.belief.t
        radial = math.atan2(xy[1] - bt[1], xy[0] - bt[0])
        last, contact = None, None
        for h in (height, height - 0.06, height + 0.06):
            for dyaw in (0.0, 0.5, -0.5, 1.0, -1.0):
                yaw = radial + math.pi / 2 + dyaw
                R = hand_down(yaw)
                cam = np.array([xy[0], xy[1], z_floor + h])
                tcp = cam - R @ off
                if min_tcp_z is not None and tcp[2] < min_tcp_z:
                    h = float(h + (min_tcp_z - tcp[2]))
                    cam[2] += min_tcp_z - tcp[2]
                    tcp[2] = min_tcp_z
                r = self.robot.move_to(tcp, orient=yaw, transit=True)
                self._note_retreat(r)  # tilt-retreat-1 (records)
                last = r
                if contact is None and is_contact_reason(r.reason, (r.data or {}).get("lift")):
                    contact = r.reason
                if r.ok:
                    self.per.note_action("look_down_at", {"height_m": h, **({"reason": contact} if contact else {})})
                    self._record("look_down_at", {"xy": [float(xy[0]), float(xy[1])], "height_m": h,
                                                  "yaw": yaw}, _skill_rec(r))
                    return r
                if r.reason == "collision":
                    self.per.note_action("look_down_at", {"ok": False, "reason": contact or r.reason})  # a contact
                    self._record("look_down_at", {"xy": [float(xy[0]), float(xy[1])], "height_m": h},
                                 _skill_rec(r))
                    return r
        self.per.note_action("look_down_at", {"ok": False, **({"reason": contact} if contact else {})})
        self._record("look_down_at", {"xy": [float(xy[0]), float(xy[1])], "height_m": height}, _skill_rec(last))
        return last

    # ------------------------------------------------------------ the unseen look T50 (look-unseen-2).
    # UNDER_TOP_TILT_DEG is a parameter: 60 (same aim) is the declared alternative.
    UNDER_TOP_TILT_DEG = 50.0
    UNDER_TOP_RADII_M = (0.40, 0.34, 0.46)  # camera distance from the aim, in the order tried
    UNDER_TOP_ROLLS = (1.0, -1.0)  # both rolls about the view axis
    UNDER_TOP_BEHIND_EDGE_M = 0.015  # aim behind the unit top's edge
    UNDER_TOP_ABOVE_FLOOR_M = 0.03  # aim above the believed drawer floor
    UNDER_TOP_MAX_ATTEMPTS = 6
    UNSEEN_DOWN_HEIGHT_M = 0.38  # straight-down unseen look (live height)
    MODULE_LOOK_HEIGHT_M = 0.55  # whole-module look (the surface centre look's height)

    def _region_record(self, rid: str):
        """The record the unseen aim reads: the region's merged record, else its latest inspection with a floor grid (a
        view that covered less than MIN_COVER), else None (no view of it yet)."""
        regions = getattr(self.model, "regions", None)
        region = regions.get(rid) if isinstance(regions, dict) else None
        if not region:
            return None
        return region.get("last") or next((r for r in reversed(region.get("inspections", [])) if r.get("grid_seen")),
                                          None)

    def _grid_unseen_points(self, rid: str, rec):
        """The region's floor points that no layer of rec's grid saw (grid "0", and "0" in the top layer when there is
        one): the live unseen set. None when rec has no grid or the region's geometry does not match it."""
        if not rec or not rec.get("grid_seen"):
            return None
        geoms = self.per.region_geometries()
        if rid not in geoms or len(geoms[rid][0]) != len(rec["grid_seen"]):
            return None
        pts = geoms[rid][0]
        top = rec.get("grid_seen_top") or ""
        unseen = np.array([c == "0" and (not top or top[i] == "0") for i, c in enumerate(rec["grid_seen"])])
        return pts[unseen]

    def surface_centre(self, cat: str, mod=None):
        """surface-frame-1: the world xy a surface look aims at: the centre of the module's accepted fitted
        outline (Perception.module_frame(category)), else the module track's position; None when neither is known."""
        if mod is None:
            mod = next((t for t in self.per.tracks.values() if t.category == cat), None)
        fn = getattr(self.per, "module_frame", None)
        fr = None
        if callable(fn):
            try:
                fr = fn(cat)
            except Exception:  # noqa: BLE001 (no fitted outline readable: the track, as before)
                fr = None
        c = _frame_centre(fr)
        if c is not None:
            return c
        return None if mod is None else np.asarray(mod.pos[:2], float)

    def under_top_aim(self, rid: str, rec=None):
        """Aim of the tilted look into a drawer (pure, robot side): {"target": aim (world xyz), "pull", "lat",
        "basis": "placements" | "grid" | "handle", "n_open", "lateral_m"}, or a failure reason
        (handle_not_perceived)."""
        h = self.per.handle_for_region(rid)
        if h is None or getattr(h, "closed_pos", None) is None:
            return "handle_not_perceived"
        pull = np.asarray(h.pull[:2], float)
        pull = pull / max(float(np.linalg.norm(pull)), 1e-9)
        lat = np.array([-pull[1], pull[0]])
        edge = np.asarray(h.closed_pos[:2], float) - DRAWER_TOP_EDGE_BEHIND_CLOSED_HANDLE * pull
        zf = float(h.pos[2]) - DRAWER_FLOOR_BELOW_HANDLE
        xy_open = open_placements(rec) if rec is not None else np.zeros((0, 2))
        grid = None if len(xy_open) else self._grid_unseen_points(rid, rec)
        if len(xy_open):
            olat, basis = float(((xy_open - edge) @ lat).mean()), "placements"
        elif grid is not None and len(grid):
            olat, basis = float(((np.asarray(grid)[:, :2] - edge) @ lat).mean()), "grid"
        else:
            olat, basis = float((np.asarray(h.pos[:2], float) - edge) @ lat), "handle"
        target = np.array([*(edge - self.UNDER_TOP_BEHIND_EDGE_M * pull + olat * lat), zf + self.UNDER_TOP_ABOVE_FLOOR_M])
        return {"target": target, "pull": pull, "lat": lat, "basis": basis, "n_open": int(len(xy_open)),
                "lateral_m": olat}

    @staticmethod
    def under_top_poses(target, pull, tilt_deg: float, radii=UNDER_TOP_RADII_M, rolls=UNDER_TOP_ROLLS):
        """The camera poses of the tilted look, in the order tried: [(radius, roll, camera position (3,), R (3, 3))]. The
        camera sits `radius` from the aim, pitched tilt_deg above the drawer axis (toward the robot along the pull); R
        is the hand frame whose third axis (the camera axis) points at the aim, rolled both ways about it."""
        T = np.asarray(target, float)
        p3 = np.array([float(pull[0]), float(pull[1]), 0.0])
        up = np.array([0.0, 0.0, 1.0])
        e = math.radians(float(tilt_deg))
        out = []
        for rr in radii:
            cam = T + rr * (math.cos(e) * p3 + math.sin(e) * up)
            zc = (T - cam) / np.linalg.norm(T - cam)
            hy0 = np.cross(zc, up)
            hy0 /= np.linalg.norm(hy0)
            for sgn in rolls:
                hy = sgn * hy0
                out.append((float(rr), float(sgn), cam, np.column_stack([np.cross(hy, zc), hy, zc])))
        return out

    def look_under_top(self, rid: str, tilt_deg: float | None = None, rec=None) -> dict:
        """look-unseen-3 (each pose move carries the aim, so the
        robot's next arm motion after a successful look first retreats out of the tilted pose, robot.skills
        tilt-retreat-1): the tilted look under the unit top of drawer rid (or T60 if tilt_deg is 60). Moves the hand
        only (no observation: the caller observes). Stops on a collision; at most UNDER_TOP_MAX_ATTEMPTS poses. Notes
        one perception action "look_down_at" (contact reason of any attempt). Returns the pose record: {"aim",
        "version", "region", "pose": "T50", "tilt_deg", "ok", "reason", "n_try", "radius_m", "roll", "basis",
        "n_open", "lateral_m", "target_xyz", "result"}; "result" is the last move's SkillResult (None when no move
        was tried), every other value is plain."""
        tilt = float(self.UNDER_TOP_TILT_DEG if tilt_deg is None else tilt_deg)
        out = {"aim": UNSEEN_AIM_VERSION, "version": LOOK_UNSEEN_VERSION, "region": rid,
               "pose": f"T{int(round(tilt))}", "tilt_deg": tilt}
        aim = self.under_top_aim(rid, self._region_record(rid) if rec is None else rec)
        if isinstance(aim, str):
            return {**out, "ok": False, "reason": aim, "n_try": 0, "basis": None, "n_open": 0, "result": None}
        p_h, _ = wrist_cam_in_hand()
        off = np.asarray(p_h, float) - np.array([0.0, 0.0, TCP_OFFSET])
        last, contact, used, n = None, None, None, 0
        # look-unseen-3 (tilt-retreat-1): each pose move carries the look's aim (rounded as target_xyz below), so a move
        # that ends ok makes the robot's next arm motion retreat first; a robot whose move_to does not take it (test
        # doubles) is moved as before
        kw = ({"retreat_aim": [round(float(x), 5) for x in aim["target"]]}
              if _takes_keyword(self.robot.move_to, "retreat_aim") else {})
        for rr, sgn, cam, Rm in self.under_top_poses(aim["target"], aim["pull"], tilt, radii=self.UNDER_TOP_RADII_M,
                                                     rolls=self.UNDER_TOP_ROLLS):
            if n >= self.UNDER_TOP_MAX_ATTEMPTS:
                break
            n += 1
            r = self.robot.move_to(cam - Rm @ off, orient=Rm, transit=True, **kw)
            self._note_retreat(r)  # tilt-retreat-1 (records)
            last = r
            if contact is None and is_contact_reason(r.reason, (r.data or {}).get("lift")):
                contact = r.reason
            if r.ok or r.reason == "collision":
                used = (rr, sgn)
                break
        ok = bool(last is not None and last.ok)
        detail = {"tilted_deg": tilt} if ok else {"ok": False}
        if contact:  # a contact met by any attempt (a collision stops the look)
            detail["reason"] = contact
        self.per.note_action("look_down_at", detail)
        return {**out, "ok": ok, "reason": None if last is None else str(last.reason), "n_try": n,
                "radius_m": None if used is None else used[0], "roll": None if used is None else used[1],
                "basis": aim["basis"], "n_open": aim["n_open"], "lateral_m": round(aim["lateral_m"], 5),
                "target_xyz": [round(float(x), 5) for x in aim["target"]], "result": last}

    def unseen_aim(self, rid: str, z_floor: float | None = None, without_record: bool = False) -> dict:
        """Aim "unseen-placement-1" (pure, no motion): what a look at the unseen part of region rid does. One rule for
        the host's look_at_region(region, "unseen") and this chain's re-look. Plain record {"aim", "region", "pose",
        "basis", "n_open", "could_hide"} with:
          pose None and "reason": no_earlier_view_of_region (no view of the region yet; the chain passes
            without_record=True and then aims as below), nothing_unseen (could_hide False: grid and placements),
            handle_not_perceived, module_not_perceived;
          pose "T50" (drawer, look_under_top);
          pose "module" (a surface whose latest lattice rests on no accepted outline, or with nothing else to aim at:
            the whole-module look at its centre, MODULE_LOOK_HEIGHT_M) and pose "down" (straight down at the centroid
            of the open placements, else at the grid's unseen points, UNSEEN_DOWN_HEIGHT_M), each with "xy",
            "z_floor", "height_m", "min_tcp_z" (sink forward rule and faucet clearance kept).
        Invariant: pose None with reason nothing_unseen only when could_hide is False."""
        out = {"aim": UNSEEN_AIM_VERSION, "region": rid, "pose": None, "basis": None, "n_open": 0, "could_hide": None}
        drawer = rid.startswith("drawer_unit:")
        h = mod = cat = None
        if drawer:  # what the live look checked first: the region's handle or module
            h = self.per.handle_for_region(rid)
            if h is None:
                return {**out, "reason": "handle_not_perceived"}
        else:
            cat = next((c for c, r in SURFACE_REGION.items() if r == rid), None)
            mod = next((t for t in self.per.tracks.values() if t.category == cat), None) if cat is not None else None
            if mod is None:
                return {**out, "reason": "module_not_perceived"}
        rec = self._region_record(rid)
        if rec is None and not without_record:
            return {**out, "reason": "no_earlier_view_of_region"}
        hide = None if rec is None else could_hide(rec, self.target)
        out["could_hide"] = hide
        if hide is False:
            return {**out, "reason": "nothing_unseen"}
        xy_open = open_placements(rec) if rec is not None else np.zeros((0, 2))
        out["n_open"] = int(len(xy_open))
        if drawer:
            if getattr(h, "closed_pos", None) is not None:
                return {**out, "pose": f"T{int(round(float(self.UNDER_TOP_TILT_DEG)))}"}
            # no closed handle position (no unit top edge known): straight down, as the live re-look did then
            z = float(h.pos[2]) - DRAWER_FLOOR_BELOW_HANDLE if z_floor is None else float(z_floor)
            pull = np.asarray(h.pull[:2], float) / max(float(np.linalg.norm(h.pull[:2])), 1e-9)
            grid = None if len(xy_open) else self._grid_unseen_points(rid, rec)
            if len(xy_open):
                xy, basis = xy_open.mean(axis=0), "placements"
            elif grid is not None and len(grid):
                xy, basis = np.asarray(grid)[:, :2].mean(axis=0), "grid"
            else:
                xy, basis = np.asarray(h.pos[:2], float) - 0.16 * pull, "handle"
            return {**out, "pose": "down", "basis": basis, "xy": [float(xy[0]), float(xy[1])], "z_floor": z,
                    "height_m": self.UNSEEN_DOWN_HEIGHT_M, "min_tcp_z": None}
        z = float(self.robot.worktop_z) + MODULE_SURFACES[cat]["z"] if z_floor is None else float(z_floor)
        min_tcp = self._min_tcp_z(rid)
        fitted = frame_accepted(rec.get("placement") if isinstance(rec, dict) else None) is not False
        grid = self._grid_unseen_points(rid, rec) if fitted and not len(xy_open) else None
        if fitted and len(xy_open):
            c, pose, basis = xy_open.mean(axis=0), "down", "placements"
        elif grid is not None and len(grid):
            c, pose, basis = np.asarray(grid)[:, :2].mean(axis=0), "down", "grid"
        else:  # no accepted outline under the latest lattice, or nothing else to aim at: the whole module
            c, pose, basis = self.surface_centre(cat, mod), "module", "module"
        xy = self._look_xy(rid, c[:2])
        return {**out, "pose": pose, "basis": basis, "xy": [float(xy[0]), float(xy[1])], "z_floor": z,
                "height_m": self.UNSEEN_DOWN_HEIGHT_M if pose == "down" else self.MODULE_LOOK_HEIGHT_M,
                "min_tcp_z": None if min_tcp is None else float(min_tcp)}

    # ------------------------------------------------------------ helpers
    def _target_track(self):
        ts = [t for t in self.per.tracks_of(self.target, (OBSERVED, STALE))]
        if not ts:
            return None
        return max(ts, key=lambda t: (t.status == OBSERVED, t.last_step, t.vis_estimate))

    def _drawer_handles_known(self) -> int:
        return sum(1 for t in self.per.tracks.values() if t.category == "drawer_handle")

    def _over_budget(self) -> bool:
        return self.per.step >= self.budget.max_steps

    # Perception steps of each guarded operation (a look = its move + its observation).
    COST = {"look_down_at": 2, "look_pose": 2}

    def _guard(self, action: str, cost: int, **args) -> bool:
        """The common budget guard: an operation starts only if it ends within the
        cap, step + cost <= max_steps. Refusals are recorded (trace "stop" and self.refused)."""
        if self.per.step + cost > self.budget.max_steps:
            rec = {"before": action, "cost": cost, "step": self.per.step, **args}
            self.refused.append(rec)
            self._record("stop", rec, {"ok": False, "reason": "budget"})
            return False
        return True

    def _may_start(self, action: str, **args) -> bool:
        """Guard of a task action (see _guard; cost from COST, else 1). The clearance gestures
        do not call this: they call _clearance."""
        if not self._guard(action, self.COST.get(action, 1), **args):
            return False
        self.action_starts.append((action, self.per.step))
        return True

    def _clearance(self, action: str, fn):
        """A declared clearance gesture (closing a drawer the robot opened, letting go of the rail
        after a fault): allowed past the cap, logged apart with the steps it used."""
        s0 = self.per.step
        self.clearance_starts.append((action, s0))
        out = fn()
        self.clearance_steps += self.per.step - s0
        return out

    def _todo(self, tried) -> list[str]:
        """Candidate regions not tried yet and not cleared for the target (never inspected, or
        inspected, e.g. by the first scan, with an unseen part that could still hold it)."""
        cleared = {row["region"] for row in self.model.search_scope(self.target)["scope"] if row["cleared"]}
        return [rid for rid in self.dossier.candidate_regions(self.target) if rid not in tried and rid not in cleared]

    # ------------------------------------------------------------ policy
    def run(self) -> dict:
        self._running = True
        rob, per = self.robot, self.per
        if not self._may_start("calibrate"):
            return self._finish(self.model.search_scope(self.target)["outcome"], note="search budget: nothing started")
        r = rob.calibrate()
        self._note_retreat(r)  # tilt-retreat-1 (records)
        per.note_action("calibrate", {"ok": bool(r.ok), **reason_detail(r)})
        self._record("calibrate", {}, {**_skill_rec(r), "belief": rob.belief.to_dict() if r.ok else None})
        if not r.ok:
            return self._finish(f"failed:calibrate:{r.reason}")
        if not self._may_start("look_pose"):
            return self._finish(self.model.search_scope(self.target)["outcome"], note="search budget: no first scan")
        r = rob.look_pose()
        self._note_retreat(r)  # tilt-retreat-1 (records)
        if r.ok:
            per.note_action("look_pose")
        else:  # contact-invalidates-1: the failure's reason reaches the site model
            per.note_action("look_pose", reason_detail(r))
        self._record("look_pose", {}, _skill_rec(r))
        if not r.ok:
            return self._finish(f"failed:look_pose:{r.reason}")
        self._observe("robot_eye", "initial scan from the look pose")
        n = self.dossier.n_drawers() or 0
        if self._drawer_handles_known() < n:
            unit = next((t for t in per.tracks.values() if t.category == "drawer_unit"), None)
            if unit is not None and self._may_start("look_down_at", why="handles"):
                rr = self.look_down_at(unit.pos[:2], rob.worktop_z, height=0.50)
                if rr is not None and rr.ok:
                    self._observe("wrist_cam", "find the drawer handles the scan did not show")
        tried: list[str] = []
        n_regions = 0
        while self._target_track() is None:
            todo = self._todo(tried)
            if not todo:
                break
            if self._over_budget() or n_regions >= self.budget.max_regions:
                self._record("stop", {}, {"ok": False, "reason": "budget", "regions_left": todo})
                break
            rid = todo[0]
            tried.append(rid)
            n_regions += 1
            if rid.startswith("drawer_unit:"):
                self._inspect_drawer(rid)
            else:
                self._inspect_surface(rid)
        t = self._target_track()
        if t is None:
            return self._finish(self.model.search_scope(self.target)["outcome"])
        return self._pick(t)

    # Opening asked. At 0.22 m the 6 cm strip still under the unit top can hold a sponge, and a sponge at the back
    # is too close to the top for the hand. Asked to pull the full slide travel (catalog, 0.24 m), the skill stops at
    # the end stop (measured on 12 of 12 drawers in 5 kitchens: true opening 0.2396 to 0.2401 m, hand travel 0.2376 to
    # 0.2391 m). 4 cm of floor then stays under the top.
    OPEN_TARGET = DRAWER_MAX_OPENING
    OPEN_ENOUGH = 0.15  # below this, too little of the drawer floor is outside the unit: try again
    MIN_TOP_CLEARANCE = 0.03  # target centre in front of the unit top's edge, for a top grasp
    RELOOK_IN_FRONT_OF_TOP = 0.10  # second look into a drawer: camera this far in front of the top's edge

    def _inspect_drawer(self, rid: str):
        rob, per = self.robot, self.per
        h = per.handle_for_region(rid)
        if h is None:
            self._record("skip_region", {"region": rid}, {"ok": False, "reason": "handle_not_perceived"})
            return
        if (h.sigma[0] > 0.010 or h.truncated or h.vis_estimate < 0.6) and self._may_start("look_down_at", region=rid):
            # The handle was only partly seen (arm in the way, image edge): look at it from above
            # before grasping it (robot-side visibility estimate).
            rr = self.look_down_at(h.pos[:2], float(h.pos[2]), height=0.40)
            if rr is not None and rr.ok:
                self._observe("wrist_cam", f"refine the handle of {rid}")
        pull = np.array([h.pull[0], h.pull[1], 0.0])
        for attempt in range(2):
            if not self._may_start("open_drawer", region=rid):
                break
            if not self._pull(h, rid, attempt + 1):
                break
            z_floor = float(h.pos[2]) - DRAWER_FLOOR_BELOW_HANDLE
            centre = h.pos[:2] - 0.16 * pull[:2]
            if not self._may_start("look_down_at", region=rid):
                break
            rr = self.look_down_at(centre, z_floor)
            if rr is not None and rr.ok:
                self._observe("wrist_cam", f"look into {rid}")
            pull = np.array([h.pull[0], h.pull[1], 0.0])
            if per.opening(h.track_id) >= self.OPEN_ENOUGH or self._target_track() is not None:
                break
        if self._target_track() is None and not self._over_budget():
            self._relook_unseen(rid, float(h.pos[2]) - DRAWER_FLOOR_BELOW_HANDLE)
        t = self._target_track()
        if t is not None and t.region == rid:
            return  # keep it open for the pick
        self._close(h, rid)

    def _pull(self, h, rid: str, attempt: int, distance: float | None = None) -> bool:
        """Pull a drawer toward OPEN_TARGET (or by `distance`), update the belief, record the
        believed opening and its source. False if the opening is unknown afterwards."""
        rob, per = self.robot, self.per
        pull = np.array([h.pull[0], h.pull[1], 0.0])
        want = self.OPEN_TARGET - per.opening(h.track_id) if distance is None else distance
        if want < 0.01 or (distance is None and want < 0.03):
            return True
        handle = h.pos.copy()
        self.opened[rid] = h
        r = rob.open_drawer(handle, pull, distance=float(min(max(want, 0.05) if distance is None else want,
                                                                self.OPEN_TARGET)),
                            min_distance=None if distance is None else max(0.0, want - 0.02))
        self._note_retreat(r)  # tilt-retreat-1 (records)
        if r.ok:
            per.note_drawer_motion(h.track_id, float(r.data["moved"]), "open_drawer")
        else:
            # How far the drawer moved is unknown (e.g. the handle slipped out of the fingers):
            # assume nothing, mark what was inside as stale, and measure it by looking.
            per.note_drawer_motion(h.track_id, 0.0, "open_drawer_failed")
            note_skill_reason(per, r)  # contact-invalidates-1
            self._release()
        open_step = per.step
        self._record("open_drawer", {"region": rid, "handle_track": h.track_id, "attempt": attempt,
                                     "handle_pos": [float(x) for x in handle],
                                     "pull_dir": [float(x) for x in pull]}, _skill_rec(r))
        if h.opening_unknown:
            if not self._may_start("look_down_at", region=rid, why="find_handle"):
                # Internal review: before, this fell through to an ok:True record while the opening was unknown. The
                # drawer may be partly open; the handle is not where the robot believes it, so no closing gesture is
                # attempted (declared).
                self._record("drawer_opening", {"region": rid},
                             {"ok": False, "reason": "opening_unknown_look_refused_by_budget"})
                if rid not in self.drawers_state_unknown:
                    self.drawers_state_unknown.append(rid)
                return False
            # Find the handle first: look over its whole travel (closed to fully open).
            rr = self.look_down_at(h.closed_pos[:2] + 0.08 * pull[:2], float(h.closed_pos[2]), height=0.50)
            if rr is not None and rr.ok:
                self._observe("wrist_cam", f"find the handle of {rid} after a failed pull")
            if h.opening_unknown:
                self._record("drawer_opening", {"region": rid},
                             {"ok": False, "reason": "handle_not_seen_opening_unknown"})
                if rid not in self.drawers_state_unknown:
                    self.drawers_state_unknown.append(rid)
                return False
        src = "handle_view" if h.last_step > open_step and h.pose_source == "this_view" else "hand_travel"
        self._record("drawer_opening", {"region": rid},
                     {"ok": True, "opening_m": per.opening(h.track_id), "source": src,
                      "handle_seen_at_step": h.last_step, "pulled_at_step": open_step})
        return True

    def _relook_unseen(self, rid: str, z_floor: float):
        """The target was not seen in this region, but its unseen part could still hold it: look once more.
        The aim is unseen_aim (the host's rule; without a view of the region yet, the chain still aims, as before):
        no look when could_hide is False (nothing_unseen); a drawer gets the T50 tilted look under the unit top
        (look_under_top); a surface a straight-down look at its open placements (else its grid's unseen points) or
        the whole-module look. The re-look runs whenever could_hide is not False (before: only when the grid had an
        unseen point)."""
        region = self.model.regions.get(rid)
        if region is None:
            return
        aim = self.unseen_aim(rid, z_floor, without_record=True)
        if aim["pose"] is None or not self._may_start("look_down_at", region=rid, why="relook"):
            return
        if aim["pose"].startswith("T"):
            lk = self.look_under_top(rid)
            r = lk["result"]
            self._record("look_under_top", plain_record(lk), _skill_rec(r) if r is not None else
                         {"ok": False, "reason": lk["reason"]})
            ok = bool(r is not None and r.ok)
            look = plain_record({**aim, **{k: lk[k] for k in ("ok", "reason", "n_try", "basis", "n_open")}})
        else:
            r = self.look_down_at(aim["xy"], aim["z_floor"], height=aim["height_m"], min_tcp_z=aim["min_tcp_z"])
            ok = bool(r is not None and r.ok)
            look = plain_record(aim)
        if ok:
            what = "under the unit top of" if aim["pose"].startswith("T") else "again over the unseen part of"
            self._observe("wrist_cam", f"look {what} {rid}")
        self._record("relook", {"region": rid, "look": look},
                     {"ok": True, "could_still_hide": self.model.could_hide(rid, self.target)})

    def _close(self, h, rid: str):
        """Clearance gesture: allowed past the step cap, logged apart (_clearance)."""
        if self.per.opening(h.track_id) < 0.02:
            return
        return self._clearance("close_drawer", lambda: self._close_now(h, rid))

    def _close_now(self, h, rid: str):
        rob, per = self.robot, self.per
        pull = np.array([h.pull[0], h.pull[1], 0.0])
        opening = per.opening(h.track_id)
        rc = rob.close_drawer(h.pos.copy(), -pull, expected=opening)
        self._note_retreat(rc)  # tilt-retreat-1 (records)
        closed = float(rc.data.get("moved", 0.0) or 0.0)
        if rc.ok:
            # End stop felt: the drawer is back where it was first seen (the hand travels a little
            # more than the drawer, the rail sits loosely between the fingers).
            per.note_drawer_motion(h.track_id, -opening, "close_drawer")
        else:
            per.note_drawer_motion(h.track_id, -min(closed, opening) if closed > 0.005 else 0.0,
                                   "close_drawer_failed")
            note_skill_reason(per, rc)  # contact-invalidates-1
            self._release_now()
        self._record("close_drawer", {"region": rid, "handle_track": h.track_id, "expected": opening},
                     _skill_rec(rc))

    def _release(self):
        """After a skill stopped on a fault (e.g. the collision reflex while pulling), the fingers
        may still be closed on the rail: open them and back up before any other motion. Clearance
        gesture: allowed past the step cap, logged apart."""
        return self._clearance("release", self._release_now)

    def _release_now(self):
        rob = self.robot
        r1 = rob.open_gripper()
        p, _ = rob.tcp_pose(rob.q_cmd)
        r2 = rob.move_to([p[0], p[1], p[2] + 0.08], orient=rob.tcp_pose(rob.q_cmd)[1], speed=0.1)
        self._note_retreat(r1)  # tilt-retreat-1 (records)
        self._note_retreat(r2)
        self.per.note_action("release", {"ok": bool(r1.ok and r2.ok), **reason_detail(r2 if not r2.ok else r1)})
        self._record("release", {}, {"ok": bool(r1.ok and r2.ok), "reason": r2.reason})

    def _look_xy(self, rid: str, xy):
        """Look point for a region: for the sink, at least SINK_LOOK_FORWARD in front of the
        perceived basin centre, toward the robot (the faucet spout is over the back half)."""
        xy = np.asarray(xy, float)[:2]
        if rid != SURFACE_REGION["sink"]:
            return xy
        mod = next((t for t in self.per.tracks.values() if t.category == "sink"), None)
        if mod is None:
            return xy
        d = self.robot.belief.t[:2] - mod.pos[:2]
        d = d / max(float(np.linalg.norm(d)), 1e-9)
        along = float((xy - mod.pos[:2]) @ d)
        return xy + max(0.0, SINK_LOOK_FORWARD - along) * d

    def _min_tcp_z(self, rid: str) -> float | None:
        """Lowest TCP height for a look down at a region: above the faucet for the sink (catalog)."""
        if rid == SURFACE_REGION["sink"]:
            return self.robot.worktop_z + SINK_FAUCET_TOP + SINK_HAND_CLEARANCE
        return None

    def _inspect_surface(self, rid: str):
        cat = next((c for c, r in SURFACE_REGION.items() if r == rid), None)
        mod = next((t for t in self.per.tracks.values() if t.category == cat), None)
        if mod is None:
            self._record("skip_region", {"region": rid}, {"ok": False, "reason": "module_not_perceived"})
            return
        z = self.robot.worktop_z + MODULE_SURFACES[cat]["z"]
        if not self._may_start("look_down_at", region=rid):
            return
        # At the module's fitted centre when perception accepted its outline, else at its track
        rr = self.look_down_at(self._look_xy(rid, self.surface_centre(cat, mod)), z, height=0.55,
                               min_tcp_z=self._min_tcp_z(rid))
        if rr is not None and rr.ok:
            self._observe("wrist_cam", f"look at {rid}")
        if self._target_track() is None and not self._over_budget():
            self._relook_unseen(rid, z)

    def _top_clearance(self, t):
        """(handle track, clearance in m of the target centre in front of the unit top's edge)
        for a target in a drawer, else (None, None). Robot side: the handle's first (closed) view and the drawer unit
        catalog. (the module function top_clearance, unchanged rule.)"""
        return top_clearance(self.per, t)

    def overhead_check(self, t, pick_pos, kind: str | None = None, yaw: float | None = None):
        """pick-pose-3: overhead_reading(self.per, ...): (refuse, plain record or None)."""
        return overhead_reading(self.per, t, pick_pos, kind=kind, yaw=yaw)

    def _pick(self, t):
        rob, per = self.robot, self.per
        h, clear = self._top_clearance(t)
        if clear is not None and clear < self.MIN_TOP_CLEARANCE:
            self._record("reachability", {"track_id": t.track_id, "region": t.region},
                         {"ok": False, "reason": "close_to_unit_top", "clearance_m": clear})
            more = self.OPEN_TARGET - per.opening(h.track_id)
            if more > 0.02 and self._may_start("open_drawer", region=t.region, why="further"):
                self._pull(h, t.region, attempt=3, distance=more)
                if self._may_start("look_down_at", region=t.region, why="after_further"):
                    rr = self.look_down_at(t.pos[:2], float(t.pos[2]) - 0.02)
                    if rr is not None and rr.ok:
                        self._observe("wrist_cam", "look again after pulling the drawer further")
                t2 = self._target_track()
                if t2 is not None:
                    t = t2
                _h, clear = self._top_clearance(t)
            if clear is not None and clear < self.MIN_TOP_CLEARANCE:
                self._record("reachability", {"track_id": t.track_id, "region": t.region},
                             {"ok": False, "reason": "under_unit_top", "clearance_m": clear})
                return self._finish("found_not_reachable:under_unit_top")
        last = None
        for attempt in range(self.budget.pick_attempts):
            # pick-pose-3: the overhead check on the pose frame, right before the pick motion
            refuse, oc = self.overhead_check(t, t.pos.copy(), kind=self.target,
                                             yaw=None if t.yaw is None else float(t.yaw) + math.pi / 2)
            if refuse:
                self._record("overhead_check", {"track_id": t.track_id, "pos": [float(x) for x in t.pos],
                                                "attempt": attempt + 1, "pose_step": t.pose_step},
                             {"ok": False, "reason": "overhead_obstruction", "arm_moved": False, **(oc or {})})
                return self._finish(OVERHEAD_REFUSED_OUTCOME)
            if not self._may_start("pick", attempt=attempt + 1):
                self._close_opened()
                return self._finish("found_pick_failed:search_budget")
            yaw = None if t.yaw is None else float(t.yaw) + math.pi / 2  # fingers across the long axis
            pos = t.pos.copy()
            r = rob.pick(pos, self.target, grasp="top", yaw=yaw)
            self._note_retreat(r)  # tilt-retreat-1 (records)
            per.note_pick(t.track_id, bool(r.ok))
            note_skill_reason(per, r)  # contact-invalidates-1; a pick invalidates anyway
            self._record("pick", {"track_id": t.track_id, "pos": [float(x) for x in pos], "yaw": yaw,
                                  "attempt": attempt + 1, "pose_source": t.pose_source,
                                  "pose_step": t.pose_step, **({"overhead": oc} if oc is not None else {})},
                         _skill_rec(r))
            last = r
            if r.ok:
                return self._finish("found_and_picked", picked_track=t.track_id)
            if attempt + 1 >= self.budget.pick_attempts:
                break
            # Look again before a second try (the object may have been pushed). Retry only on a
            # fresh sighting: the target track must have been seen after the failed pick.
            pick_step = per.step
            if not self._may_start("look_down_at", why="after_failed_pick"):
                self._close_opened()
                return self._finish(f"found_pick_failed:{last.reason}", note="search budget reached after a failed pick")
            rr = self.look_down_at(t.pos[:2], float(t.pos[2]) - 0.02)
            if rr is not None and rr.ok:
                self._observe("wrist_cam", "look again after a failed pick")
            t2 = self._target_track()
            if t2 is None or t2.last_step <= pick_step:
                return self._finish(f"found_pick_failed:{last.reason}",
                                    note="not_reseen: the tool was not seen again after the failed pick")
            t = t2
        return self._finish(f"found_pick_failed:{last.reason if last is not None else 'none'}")

    def _close_opened(self):
        """Clearance before a search stopped by its budget with the hand empty: close every drawer the
        robot pulled and still believes open (>= 2 cm, known opening). Allowed past the cap, logged
        apart (_close / _clearance)."""
        if self.robot.holding():
            return
        for rid, h in list(self.opened.items()):
            if rid in self.drawers_state_unknown or getattr(h, "opening_unknown", False):
                continue
            self._close(h, rid)

    def _finish(self, outcome: str, picked_track: str | None = None, note: str = "") -> dict:
        self._running = False
        self.outcome = outcome
        holding = bool(self.robot.holding())
        res = {"outcome": outcome, "robot_says_holding": holding, "picked_track": picked_track,
               "search_scope": self.model.search_scope(self.target), "diff_report": self.model.diff_report(),
               "site_model": self.model.status_table(), "n_steps": self.per.step,
               "step_budget": {"max_steps": self.budget.max_steps, "clearance_steps": self.clearance_steps,
                               "clearance_starts": [list(x) for x in self.clearance_starts],
                               "refused": [dict(x) for x in self.refused[:10]]},
               "drawers_state_unknown": list(self.drawers_state_unknown)}
        if note:
            res["note"] = note
        self._record("finish", {}, {"ok": outcome == "found_and_picked", "outcome": outcome}, note=note)
        return res
