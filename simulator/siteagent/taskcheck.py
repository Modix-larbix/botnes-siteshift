"""Task evaluator for "find the sponge, pick it, scrub the target bowl", on the bench side.

This module reads the simulator truth. The robot never imports it, never receives its output, and its verdict is never
given to a robot program. A skill's "ok" or a program's claim is an input it reports, never its verdict.

verdict() returns the oracle diagnosis of an episode (exact positions, forces, contacts, which object was in the
fingers) with its outcome class. The observable feedback record is built elsewhere (siteagent.feedback).

Criteria and thresholds come from siteagent/config/task_criteria_v1_4.json (task-criteria-1.4); load_criteria() can
read other versions, whose rules are switched on by fields of their files. Success needs, over the whole episode and
from the truth: a scrub bout between the sponge and the target bowl (contact force >= 1 N for >= 2.4 s, >= 80 % of
the bout in contact); a scrubbing motion of the sponge measured only while in contact, in the bowl's frame and in the
world frame (stroke rule of the criteria file); the sponge held by the robot during the bout; the bowl neither dragged,
tipped nor pushed too hard; no other dish scrubbed. Violations are judged apart: arm-link force above 20 N, hand force
above 60 N (outside calibrate), another item moved or tipped, a body in the forbidden aisle zone, budget exceeded.

not_found_support() checks a "not found" claim against the bench's own records of the observations (the bench is the
stricter side: it invalidates after any call during which an item touched a finger).
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from . import checks
from .robot.skills import HAND_FORCE_LIMIT, LINK_FORCE_LIMIT
from .sitemodel import MIN_COVER, _merge_grids, could_hide

CONFIG_DIR = Path(__file__).resolve().parent / "config"
CRITERIA_PATH = CONFIG_DIR / "task_criteria_v1_4.json"
CRITERIA_VERSION = "task-criteria-1.4"
CRITERIA_V1_3_PATH = CONFIG_DIR / "task_criteria_v1_3.json"
CRITERIA_V1_2_PATH = CONFIG_DIR / "task_criteria_v1_2.json"
CRITERIA_V1_1_PATH = CONFIG_DIR / "task_criteria_v1_1.json"
# Versions this module grades (grading 1.0 needs an older copy of this module, not included here).
GRADES = {"task-criteria-1.1", "task-criteria-1.2", "task-criteria-1.3", "task-criteria-1.4"}
DISHES = ("cup", "bowl", "fork", "spoon")


def load_criteria(path: Path = CRITERIA_PATH) -> dict:
    raw = path.read_bytes()
    c = json.loads(raw)
    if c["version"] not in GRADES:
        raise ValueError(f"criteria version {c['version']!r}, this module grades {sorted(GRADES)}")
    # The skill-check values this config reuses must not drift silently.
    assert c["criteria"]["contact_force_min_n"] == checks.SCRUB_N
    assert c["criteria"]["contact_fraction_min"] == checks.SCRUB_FRAC
    assert c["criteria"]["target_tilt_max_deg"] == checks.DISH_TILT_DEG
    assert c["violations"]["arm_link_force_max_n"] == LINK_FORCE_LIMIT
    assert c["violations"]["hand_force_max_n"] == HAND_FORCE_LIMIT
    assert c["not_found_support"]["min_coverage"] == MIN_COVER
    c["sha256"] = hashlib.sha256(raw).hexdigest()
    return c


def _tilt_deg(R) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(R[2, 2])))))


def _axis_change_deg(R0, R1) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(R0[:, 2] @ R1[:, 2])))))


# Skills that can change what lies in a region (they end a run of views that add up).
STATE_CHANGING = {"pick", "place", "turn_over", "scrub", "spray", "press"}


class BenchEvidence:
    """Bench-held record of one episode (bench only; the program never gets this object).

    The skill facade calls start(name) before and end(record) after every skill call, refused
    calls included, through a private reference; end() keeps a deep copy. The bench wraps the
    perception's observe() and calls observation(obs) at capture time. From these the verifier
    takes the call reasons, the calibrate windows (hand-force exemption), the budget (counted
    against the criteria config with the simulator clock) and the evidence for a not-found
    claim. Nothing here is read from objects the program holds."""

    def __init__(self, sim, budget: dict, scope_regions=(), on_call=None):
        self.sim = sim
        self.max_sim_s, self.max_calls = float(budget["max_sim_s"]), int(budget["max_skill_calls"])
        self.t0 = float(sim.t)
        self.calls = 0
        self.records: list[dict] = []
        self.windows: list[dict] = []  # one per call: skill, call index, sim time at start and end
        self.scope_regions = list(scope_regions)
        self.inspections: list[dict] = []
        self._cur = None
        self._on_call = on_call  # bench display hook (GIF caption); gets the bench copy

    def start(self, name: str):
        self.calls += 1
        self._cur = {"skill": str(name), "i": self.calls, "t0": float(self.sim.t)}

    def end(self, rec: dict):
        rec = copy.deepcopy(rec)
        self.records.append(rec)
        if self._cur is not None:
            self.windows.append(dict(self._cur, t1=float(self.sim.t)))
        self._cur = None
        if self._on_call is not None:
            self._on_call(rec)

    def observation(self, obs):
        """Record the regions an observation inspected (deep copy, at capture time), with the
        true opening of each drawer (bench truth)."""
        dj = self.sim.info.drawer_joints
        for r in obs.regions:
            opening = None
            if r.region.startswith("drawer_unit:"):
                j = dj.get("drawers/" + r.region.split(":", 1)[1])
                opening = None if j is None else round(self.sim.joint_pos(j), 4)
            self.inspections.append({"region": r.region, "t_s": round(float(self.sim.t), 3), "calls": self.calls,
                                     "coverage": float(r.coverage), "grid_shape": list(r.grid_shape),
                                     "grid_pitch_m": list(r.grid_pitch_m), "grid_seen": r.grid_seen,
                                     "grid_seen_top": r.grid_seen_top, "drawer_open_true_m": opening})

    def budget_state(self) -> dict:
        used = float(self.sim.t) - self.t0
        exceeded = None
        if self.calls > self.max_calls:
            exceeded = "max_skill_calls"
        elif used > self.max_sim_s:
            exceeded = "max_sim_s"
        return {"sim_s_used": round(used, 1), "max_sim_s": self.max_sim_s, "skill_calls": self.calls,
                "max_skill_calls": self.max_calls, "exceeded": exceeded, "counted_by": "bench"}

    def in_windows(self, t: float, skills) -> bool:
        return any(w["skill"] in skills and w["t0"] - 1e-9 <= t <= w["t1"] + 0.021 for w in self.windows)

    def skill_at(self, t: float) -> str | None:
        for w in self.windows:
            if w["t0"] - 1e-9 <= t <= w["t1"] + 0.021:
                return w["skill"]
        return None


def _held_during(finger_items, t0: float, t1: float) -> bool:
    """Did an item touch a finger at some sample in [t0, t1]? finger_items: TaskMonitor.finger_items,
    (t, items) at every change."""
    cur = []
    for t, items in finger_items:
        if t <= t0:
            cur = items
            continue
        if cur:
            return True
        if t > t1 + 0.021:
            return False
        cur = items
    return bool(cur)


def not_found_support(ev: BenchEvidence, crit: dict, category: str = "sponge", mon=None) -> dict:
    """Is a not-found claim supported by the observations the bench recorded? (rule in the
    criteria config, "not_found_support"). Per scope region: cleared or not, and why. 1.2 ("invalidating_skills" in
    the config): a clearance holds only if no later call of an invalidating skill, and no later call made while an
    item touched a finger (from mon, the TaskMonitor), happened after the view that cleared the region; otherwise a
    new view is needed."""
    S = crit["not_found_support"]
    v12 = "invalidating_skills" in S
    inv_skills = set(S.get("invalidating_skills", ()))
    held_calls = set()
    if v12 and mon is not None:
        held_calls = {w["i"] for w in ev.windows if _held_during(mon.finger_items, w["t0"], w["t1"])}
    invalid = sorted({w["i"] for w in ev.windows if w["skill"] in inv_skills} | held_calls) if v12 else []
    rows = {}
    for rid in ev.scope_regions:
        drawer = rid.startswith("drawer_unit:")
        insp = [x for x in ev.inspections if x["region"] == rid and x["coverage"] >= S["min_coverage"]]
        cleared, best_cov, merged, prev, cleared_at = False, 0.0, None, None, None
        for x in insp:
            best_cov = max(best_cov, x["coverage"])
            if drawer and (x["drawer_open_true_m"] is None or x["drawer_open_true_m"] < S["drawer_open_min_m"]):
                merged, prev = None, None  # a view of a (nearly) closed drawer clears nothing
                continue
            brk = prev is None or any(w["skill"] in STATE_CHANGING and prev["calls"] < w["i"] <= x["calls"]
                                      for w in ev.windows)
            if v12 and prev is not None and any(prev["calls"] < i <= x["calls"] for i in invalid):
                brk = True
            if drawer and prev is not None and abs(x["drawer_open_true_m"] - prev["drawer_open_true_m"]) > 0.002:
                brk = True
            m = None if brk or merged is None else _merge_grids(x, merged)
            merged = dict(x) if m is None else dict(merged, **m)
            prev = x
            if could_hide(merged, category) is False:
                cleared, cleared_at = True, x["calls"]
        row = {"cleared": cleared, "n_views": len(insp), "best_coverage": round(best_cov, 3),
               "drawer_open_true_max_m": max((x["drawer_open_true_m"] or 0.0 for x in ev.inspections
                                              if x["region"] == rid), default=None) if drawer else None}
        if v12 and cleared:
            later = [i for i in invalid if i > cleared_at]
            if later:
                row["cleared"] = False
                row["invalidated_by_calls"] = later[:10]
        rows[rid] = row
    supported = bool(rows) and all(r["cleared"] for r in rows.values())
    out = {"supported": supported, "regions": rows, "not_cleared": sorted(r for r, v in rows.items() if not v["cleared"])}
    if v12:
        out["holding_check"] = "bench monitor" if mon is not None else "not available (no monitor given)"
    return out


class TaskMonitor:
    """Per-tick ground truth for the task (bench only): sponge-target and sponge-other-dish
    contact forces, hand-target force, sponge-finger contacts, sponge-TCP distance, sponge
    position in the target's frame, arm-link forces, net force on the hand and fingers from the
    kitchen and items other than the sponge, forbidden zones, items in the fingers. Append it to
    sim.monitors right after building the sim. Sampled once per 20 ms control tick, after the
    physics substeps (the same sampling as the robot's own reflex)."""

    def __init__(self, sim, truth_spec, target_name: str, crit: dict):
        self.sim, self.spec, self.target, self.crit = sim, truth_spec, target_name, crit
        m = sim.model
        self.item_of_geom = {}
        for it in truth_spec.items:
            bid = m.body(it.name).id
            for g in range(m.ngeom):
                if m.geom_bodyid[g] == bid:
                    self.item_of_geom[g] = it.name
        self.kind = {it.name: it.kind for it in truth_spec.items}
        self.sponge = next((it.name for it in truth_spec.items if it.kind == "sponge"), None)
        self.lf, self.rf = m.body("panda/left_finger").id, m.body("panda/right_finger").id
        self.links = sim.link_bodies
        self.hand = sim.hand_bodies
        self.tid = m.body(target_name).id
        self.sid = None if self.sponge is None else m.body(self.sponge).id
        self.robot_ids = np.array(sorted(sim.robot_bodies))
        self.tcp = m.site("panda/tcp").id
        self.zones = crit["violations"]["forbidden_zones"]
        # Start state of every item (position relative to its drawer when in one; orientation).
        self.drawer_body = {it.name: (f"drawer_{it.where.split('/')[1]}" if it.where.startswith("drawers/") else None)
                            for it in truth_spec.items}
        self.start = {n: (self._rel(n), sim.body_mat(n)) for n in self.kind}
        self.z0_sponge = None if self.sponge is None else float(sim.body_pos(self.sponge)[2])
        # Per-tick series (lists of floats / bools; 50 Hz).
        self.t: list[float] = []
        self.f_target: list[float] = []
        self.f_other: list[float] = []
        self.touch_l: list[bool] = []
        self.touch_r: list[bool] = []
        self.sponge_tcp: list[float] = []
        self.sponge_z: list[float] = []
        self.sponge_rel: list[tuple] = []  # sponge centre in the target's frame (m)
        self.sponge_xy: list[tuple] = []  # sponge centre, world x, y (m)
        self.f_hand_target: list[float] = []  # hand and finger normal force on the target
        self.hand_env: list[float] = []  # net force on hand + fingers from kitchen / non-sponge items
        self.link_peak = 0.0
        self.link_events: list[dict] = []
        self.zone_events: list[dict] = []
        self.finger_items: list[tuple[float, list[str]]] = []  # (t, items touching a finger) when it changes
        self.target_other_contact_s = 0.0  # a non-sponge item pressing on the target bowl
        self._f6 = np.zeros(6)
        # 1.2: target pose series and persistent maxima of every item but the sponge, every tick of the episode (an item
        # knocked over and put back stays recorded).
        self._bid = {n: m.body(n).id for n in self.kind}
        self._did = {n: (None if dn is None else m.body(dn).id) for n, dn in self.drawer_body.items()}
        C, V = crit["criteria"], crit["violations"]
        self._limits = {n: ((C["target_displacement_max_m"], C["target_tilt_max_deg"], "tilt_deg") if n == target_name
                            else (V["other_item_displacement_max_m"], V["other_item_tilt_max_deg"], "tilt_change_deg"))
                        for n in self.kind if n != self.sponge}
        self.target_xy: list[tuple] = []  # target centre, world x, y (m)
        self.target_yaw: list[float] = []  # target yaw about the world z axis (rad, wrapped)
        self.item_max: dict[str, dict] = {n: {"displacement_m": 0.0, "t_displacement_s": None, "tilt_change_deg": 0.0,
                                              "tilt_deg": 0.0, "t_tilt_s": None, "first_over": {}}
                                          for n in self._limits}

    def _rel(self, name):
        p = self.sim.body_pos(name)
        d = self.drawer_body.get(name)
        return p - self.sim.body_pos(d) if d else p

    def __call__(self, sim):
        m, d = sim.model, sim.data
        f6 = self._f6
        ft = fo = 0.0
        tl = tr = False
        in_fingers = set()
        link_f: dict[int, np.ndarray] = {}
        hand_f = np.zeros(3)
        fh = 0.0
        other_on_target = False
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
            i1, i2 = self.item_of_geom.get(g1), self.item_of_geom.get(g2)
            if i1 is None and i2 is None:
                # Arm link (link0-link7) against the environment, not against the robot itself.
                b = b1 if (b1 in self.links and not sim.robot_geom[g2]) else (
                    b2 if (b2 in self.links and not sim.robot_geom[g1]) else None)
                if b is not None:
                    mujoco.mj_contactForce(m, d, i, f6)
                    fw = c.frame.reshape(3, 3).T @ f6[:3]
                    link_f[b] = link_f.get(b, np.zeros(3)) + fw
                # Hand or finger against the kitchen (force on the robot geom; fw acts on geom2).
                if b1 in self.hand and not sim.robot_geom[g2]:
                    mujoco.mj_contactForce(m, d, i, f6)
                    hand_f -= c.frame.reshape(3, 3).T @ f6[:3]
                elif b2 in self.hand and not sim.robot_geom[g1]:
                    mujoco.mj_contactForce(m, d, i, f6)
                    hand_f += c.frame.reshape(3, 3).T @ f6[:3]
                continue
            if i1 is not None and i2 is not None:
                pair = {i1, i2}
                if self.sponge in pair:
                    other = i2 if i1 == self.sponge else i1
                    if other == self.target or self.kind.get(other) in DISHES:
                        mujoco.mj_contactForce(m, d, i, f6)
                        if other == self.target:
                            ft += abs(float(f6[0]))
                        else:
                            fo += abs(float(f6[0]))
                elif self.target in pair:
                    other_on_target = True
                continue
            it, b = (i1, b2) if i1 is not None else (i2, b1)
            if b in self.hand and it != self.sponge:
                mujoco.mj_contactForce(m, d, i, f6)
                fw = c.frame.reshape(3, 3).T @ f6[:3]
                hand_f += fw if i1 is not None else -fw  # force on the robot geom
                if it == self.target:
                    fh += abs(float(f6[0]))
            if b in (self.lf, self.rf):
                in_fingers.add(it)
                if it == self.sponge:
                    tl |= b == self.lf
                    tr |= b == self.rf
            elif b in self.links:
                mujoco.mj_contactForce(m, d, i, f6)
                fw = c.frame.reshape(3, 3).T @ f6[:3]
                link_f[b] = link_f.get(b, np.zeros(3)) + fw
        t = float(d.time)
        self.t.append(t)
        self.f_target.append(ft)
        self.f_other.append(fo)
        self.touch_l.append(tl)
        self.touch_r.append(tr)
        self.f_hand_target.append(fh)
        self.hand_env.append(float(np.linalg.norm(hand_f)))
        if self.sponge is not None:
            ps = d.xpos[self.sid]
            self.sponge_tcp.append(float(np.linalg.norm(ps - d.site_xpos[self.tcp])))
            self.sponge_z.append(float(ps[2]))
            rel = d.xmat[self.tid].reshape(3, 3).T @ (ps - d.xpos[self.tid])
            self.sponge_rel.append((float(rel[0]), float(rel[1]), float(rel[2])))
            self.sponge_xy.append((float(ps[0]), float(ps[1])))
        for b, f in link_f.items():
            n = float(np.linalg.norm(f))
            self.link_peak = max(self.link_peak, n)
            if n > self.crit["violations"]["arm_link_force_max_n"] and len(self.link_events) < 20:
                self.link_events.append({"t_s": round(t, 2), "body": m.body(b).name, "force_n": round(n, 1)})
        if other_on_target and in_fingers - {self.sponge}:
            self.target_other_contact_s += 0.02
        Rt = d.xmat[self.tid].reshape(3, 3)
        self.target_xy.append((float(d.xpos[self.tid][0]), float(d.xpos[self.tid][1])))
        self.target_yaw.append(float(math.atan2(Rt[1, 0], Rt[0, 0])))
        for n, (dmax, amax, akey) in self._limits.items():
            b, dd = self._bid[n], self._did[n]
            p = d.xpos[b] - d.xpos[dd] if dd is not None else d.xpos[b]
            q0, Q0 = self.start[n]
            R = d.xmat[b].reshape(3, 3)
            dp = float(np.linalg.norm(p - q0))
            da = _axis_change_deg(Q0, R)
            ta = _tilt_deg(R)
            mx = self.item_max[n]
            if dp > mx["displacement_m"]:
                mx["displacement_m"], mx["t_displacement_s"] = dp, round(t, 2)
            if da > mx["tilt_change_deg"]:
                mx["tilt_change_deg"] = da
                if akey == "tilt_change_deg":
                    mx["t_tilt_s"] = round(t, 2)
            if ta > mx["tilt_deg"]:
                mx["tilt_deg"] = ta
                if akey == "tilt_deg":
                    mx["t_tilt_s"] = round(t, 2)
            if dp > dmax and "displacement" not in mx["first_over"]:
                mx["first_over"]["displacement"] = {"t_s": round(t, 2), "value_m": round(dp, 4)}
            a = ta if akey == "tilt_deg" else da
            if a > amax and "tilt" not in mx["first_over"]:
                mx["first_over"]["tilt"] = {"t_s": round(t, 2), "value_deg": round(a, 1), "measure": akey}
        xs = d.xpos[self.robot_ids, 0]
        xmin = min(float(xs.min()), float(d.site_xpos[self.tcp][0]))
        for z in self.zones:
            if xmin < z["x_max"] and len(self.zone_events) < 20:
                self.zone_events.append({"t_s": round(t, 2), "zone": z["name"], "x_min": round(xmin, 3)})
        fi = sorted(in_fingers)
        if not self.finger_items or self.finger_items[-1][1] != fi:
            self.finger_items.append((round(t, 3), fi))


def _bouts(t, on, merge_gap):
    """Maximal runs of `on` samples, runs separated by <= merge_gap seconds merged.
    Returns [(i0, i1)] inclusive sample indices."""
    idx = np.flatnonzero(on)
    if len(idx) == 0:
        return []
    out = [[idx[0], idx[0]]]
    for k in idx[1:]:
        if t[k] - t[out[-1][1]] <= merge_gap + 1e-9:
            out[-1][1] = k
        else:
            out.append([k, k])
    return [tuple(b) for b in out]


def _longest_gap(t, ok, i0, i1) -> float:
    """Longest run of not-ok samples in [i0, i1], in seconds (sample count x dt)."""
    best = cur = 0
    for k in range(i0, i1 + 1):
        if ok[k]:
            cur = 0
        else:
            cur += 1
            best = max(best, cur)
    dt = (t[-1] - t[0]) / max(len(t) - 1, 1) if len(t) > 1 else 0.02
    return best * dt


def _stroke(rel: np.ndarray, on: np.ndarray, step: int) -> tuple[float, float]:
    """(path, extent) of the sponge across the target, horizontal, in the target's frame:
    path over contact samples taken every `step` ticks (a step counts only if both ends are in
    contact), extent = largest distance between two contact samples."""
    if len(rel) == 0:
        return 0.0, 0.0
    xy = rel[:, :2]
    path = 0.0
    for k in range(0, len(xy) - step, step):
        if on[k] and on[k + step]:
            path += float(np.linalg.norm(xy[k + step] - xy[k]))
    pts = xy[on]
    if len(pts) < 2:
        return path, 0.0
    ext = 0.0
    for k in range(0, len(pts), 256):  # chunked pairwise distances
        dd = np.linalg.norm(pts[k:k + 256, None, :] - pts[None, :, :], axis=2)
        ext = max(ext, float(dd.max()))
    return path, ext


def _stroke_contact_runs(rel: np.ndarray, on: np.ndarray, step: int) -> tuple[float, float, int, int]:
    """1.3: (path, extent, number of contact runs, longest run in samples) of the
    sponge across the target, horizontal, counting ONLY displacement between consecutive 20 ms
    samples that are both in contact. A contact run is a maximal run of consecutive contact
    samples; nothing between two runs is credited. Path: inside each run, chords between samples
    `step` apart from the run's first sample (the 10 Hz chords of 1.2, the tremor filter; a tail
    shorter than `step` is not counted). Extent: the largest distance between two samples of the
    SAME run. With a single run starting at the bout start, path and extent equal 1.2's."""
    if len(rel) == 0:
        return 0.0, 0.0, 0, 0
    xy = np.asarray(rel, float)[:, :2]
    on = np.asarray(on, bool)
    path, ext, runs, longest = 0.0, 0.0, 0, 0
    k, n = 0, len(xy)
    while k < n:
        if not on[k]:
            k += 1
            continue
        a = k
        while k < n and on[k]:
            k += 1
        b = k - 1  # run [a, b], all in contact
        runs += 1
        longest = max(longest, b - a + 1)
        for j in range(a, b - step + 1, step):
            path += float(np.linalg.norm(xy[j + step] - xy[j]))
        pts = xy[a:b + 1]
        if len(pts) >= 2:
            for c in range(0, len(pts), 256):  # chunked pairwise distances
                dd = np.linalg.norm(pts[c:c + 256, None, :] - pts[None, :, :], axis=2)
                ext = max(ext, float(dd.max()))
    return path, ext, runs, longest


def _stroke_contact_chords(rel: np.ndarray, on: np.ndarray, step: int) -> tuple[float, float, int, int, list]:
    """1.4: (path, extent, number of contact runs, longest run in samples, path per grid phase) of
    the sponge across the target, horizontal. A chord joins two samples `step` apart (10 Hz, the tremor filter of
    1.2) and counts only if EVERY sample from its first to its last end is in contact. For each phase p in 0..step-1
    the chords start at bout samples p, p+step, p+2 step...; path = the SMALLEST of the step phase paths (no phase
    can be chosen by timing contact breaks or the bout start). Extent: the largest distance between two samples of
    the SAME contact run (as 1.3). Properties (a test): removing contact samples never increases path or extent
    (monotone in contact); path <= 1.2's path (phase 0 with every sample in contact is a subset of 1.2's chords with
    both ends in contact); extent <= 1.2's extent; with continuous contact the phase-0 path equals 1.2's path."""
    if len(rel) == 0:
        return 0.0, 0.0, 0, 0, [0.0] * step
    xy = np.asarray(rel, float)[:, :2]
    on = np.asarray(on, bool)
    n = len(xy)
    cum = np.concatenate([[0], np.cumsum(on)])  # cum[j] = contact samples among 0..j-1
    phase_paths = []
    for ph in range(step):
        p = 0.0
        for j in range(ph, n - step, step):  # chord j -> j+step, the samples j..j+step all in contact
            if cum[j + step + 1] - cum[j] == step + 1:
                p += float(np.linalg.norm(xy[j + step] - xy[j]))
        phase_paths.append(p)
    _p13, ext, runs, longest = _stroke_contact_runs(xy, on, step)
    return min(phase_paths), ext, runs, longest, phase_paths


def verdict(mon: TaskMonitor, claim: dict, ev: BenchEvidence, exception: str | None) -> dict:
    """Oracle diagnosis of one episode (bench evidence only, never given to the robot).
    ev: the bench-held record (calls, reasons, budget, observations); never the facade."""
    sim, crit = mon.sim, mon.crit
    skill_log = ev.records
    budget_state = ev.budget_state()
    C, V = crit["criteria"], crit["violations"]
    t = np.array(mon.t)
    ft = np.array(mon.f_target)
    fo = np.array(mon.f_other)
    dt = 0.02
    tl, tr = np.array(mon.touch_l, dtype=bool), np.array(mon.touch_r, dtype=bool)
    held, both = tl | tr, tl & tr  # an episode can have no tick at all (program error at once)
    out: dict = {"label": "simulation, perception simulee", "criteria_version": crit["version"],
                 "criteria_sha256": crit["sha256"], "target": mon.target, "sponge_present": mon.sponge is not None,
                 "claim": claim.get("claim"), "exception": exception}
    crit_res: dict = {}
    fmin = C["contact_force_min_n"]
    on = ft >= fmin
    bouts = _bouts(t, on, C["bout_merge_gap_s"]) if len(t) else []
    best = None
    for b0, b1 in bouts:
        dur = float(t[b1] - t[b0]) + dt
        frac = float(on[b0:b1 + 1].mean())
        if best is None or dur > best["duration_s"]:
            best = {"i0": int(b0), "i1": int(b1), "t0_s": round(float(t[b0]), 2), "t1_s": round(float(t[b1]), 2),
                    "duration_s": round(dur, 3), "contact_fraction": round(frac, 3),
                    "force_mean_n": round(float(ft[b0:b1 + 1].mean()), 2)}
    out["target_contact_total_s"] = round(float(on.sum()) * dt, 3)
    out["best_bout"] = None if best is None else {k: v for k, v in best.items() if k not in ("i0", "i1")}
    if best is not None:
        i0, i1 = best["i0"], best["i1"]
        best["contact_time_s"] = round(float(on[i0:i1 + 1].sum()) * dt, 3)
        path = ext = path_b = ext_b = 0.0
        runs13 = C.get("stroke_rule") == "contact_runs"
        chords14 = C.get("stroke_rule") == "contact_chords"
        if mon.sponge is not None and len(mon.sponge_rel) == len(t):
            step = max(1, int(round(1.0 / (C["stroke_sample_hz"] * dt))))
            onb = on[i0:i1 + 1]
            if chords14:
                # 1.4: chords fully in contact, smallest grid phase (monotone in contact, <= 1.2).
                path, ext, nr, lr, _ph = _stroke_contact_chords(np.array(mon.sponge_xy[i0:i1 + 1]), onb, step)
                path_b, ext_b, _nr, _lr, ph_b = _stroke_contact_chords(np.array(mon.sponge_rel[i0:i1 + 1]), onb, step)
                best["stroke_rule"] = "contact_chords"
                best["contact_runs"], best["longest_contact_run_s"] = nr, round(lr * dt, 3)
                best["stroke_path_bowl_frame_by_phase_m"] = [round(x, 4) for x in ph_b]
            elif runs13:
                # 1.3: only motion between consecutive contact samples is credited.
                path, ext, nr, lr = _stroke_contact_runs(np.array(mon.sponge_xy[i0:i1 + 1]), onb, step)
                path_b, ext_b, _nr, _lr = _stroke_contact_runs(np.array(mon.sponge_rel[i0:i1 + 1]), onb, step)
                best["stroke_rule"] = "contact_runs"
                best["contact_runs"], best["longest_contact_run_s"] = nr, round(lr * dt, 3)
            else:
                path, ext = _stroke(np.array(mon.sponge_xy[i0:i1 + 1]), onb, step)  # world frame
                path_b, ext_b = _stroke(np.array(mon.sponge_rel[i0:i1 + 1]), onb, step)  # bowl frame
        best["stroke_path_m"], best["stroke_extent_m"] = round(path, 4), round(ext, 4)
        best["stroke_path_bowl_frame_m"], best["stroke_extent_bowl_frame_m"] = round(path_b, 4), round(ext_b, 4)
        # Target motion during the bout (1.2): centre extent (world, horizontal) and yaw range.
        if len(getattr(mon, "target_xy", [])) == len(t):
            txy = np.array(mon.target_xy[i0:i1 + 1])
            _p, text = _stroke(txy, np.ones(len(txy), bool), 1)
            yaw = np.unwrap(np.array(mon.target_yaw[i0:i1 + 1]))
            best["target_extent_in_bout_m"] = round(text, 4)
            best["target_yaw_range_in_bout_deg"] = round(math.degrees(float(yaw.max() - yaw.min())), 2)
        out["best_bout"] = {k: v for k, v in best.items() if k not in ("i0", "i1")}
    crit_res["target_contact"] = bool(best is not None and best["duration_s"] >= C["contact_duration_min_s"]
                                      and best["contact_fraction"] >= C["contact_fraction_min"]
                                      and best["contact_time_s"] >= C["contact_time_in_bout_min_s"])
    if C.get("stroke_frame", "world") == "target":
        # 1.2: the sponge slides across THE BOWL (its frame), and itself moves in the world.
        crit_res["scrub_motion"] = bool(best is not None
                                        and best["stroke_path_bowl_frame_m"] >= C["stroke_path_min_m"]
                                        and best["stroke_extent_bowl_frame_m"] >= C["stroke_extent_min_m"]
                                        and best["stroke_path_m"] >= C["stroke_world_path_min_m"]
                                        and best["stroke_extent_m"] >= C["stroke_world_extent_min_m"])
        crit_res["target_held_during_scrub"] = bool(
            best is not None and "target_extent_in_bout_m" in best
            and best["target_extent_in_bout_m"] <= C["target_extent_in_bout_max_m"]
            and best["target_yaw_range_in_bout_deg"] <= C["target_yaw_range_in_bout_max_deg"])
    else:
        crit_res["scrub_motion"] = bool(best is not None and best["stroke_path_m"] >= C["stroke_path_min_m"]
                                        and best["stroke_extent_m"] >= C["stroke_extent_min_m"])
    # Grip during the bout.
    if best is not None and mon.sponge is not None:
        i0, i1 = best["i0"], best["i1"]
        close = np.array(mon.sponge_tcp[i0:i1 + 1]) <= C["grip_sponge_tcp_max_m"]
        gap = _longest_gap(t, held, i0, i1)
        out["bout_grip"] = {"longest_finger_gap_s": round(gap, 3), "sponge_tcp_max_m":
                            round(float(np.max(mon.sponge_tcp[i0:i1 + 1])), 4)}
        crit_res["grip_during_scrub"] = bool(gap <= C["grip_gap_max_s"] and close.all())
    else:
        crit_res["grip_during_scrub"] = False
    # No drop from the first grasp to the end.
    if mon.sponge is not None:
        lifted = np.array(mon.sponge_z) - mon.z0_sponge >= checks.LIFT_MIN
        g = np.flatnonzero(both & lifted)
        if len(g):
            i0 = int(g[0])
            gap = _longest_gap(t, held, i0, len(t) - 1)
            end_held = bool(checks._touches_fingers(sim, mon.sponge)
                            and mon.sponge_tcp[-1] <= C["grip_sponge_tcp_max_m"])
            out["grasp"] = {"first_grasp_t_s": round(float(t[i0]), 2), "longest_finger_gap_s": round(gap, 3),
                            "held_at_end": end_held}
            crit_res["no_drop"] = bool(gap <= C["grip_gap_max_s"] and end_held)
        else:
            out["grasp"] = {"first_grasp_t_s": None}
            crit_res["no_drop"] = False
    else:
        crit_res["no_drop"] = False
    other_s = float((fo >= fmin).sum()) * dt
    out["other_dish_contact_s"] = round(other_s, 3)
    crit_res["no_other_dish_scrubbed"] = bool(other_s <= C["other_dish_contact_max_s"])
    fh = np.array(mon.f_hand_target)
    ftot = ft + fh if (C.get("target_force_includes_hand") and len(fh) == len(ft)) else ft
    out["target_force_peak_n"] = round(float(ftot.max()) if len(ftot) else 0.0, 2)
    out["hand_on_target_force_peak_n"] = round(float(fh.max()) if len(fh) else 0.0, 2)
    crit_res["target_force_within_limit"] = bool(out["target_force_peak_n"] <= C["target_force_max_n"])
    p0, R0 = mon.start[mon.target]
    p1, R1 = mon._rel(mon.target), sim.body_mat(mon.target)
    out["target_tilt_deg"] = round(_tilt_deg(R1), 2)
    dims = int(C.get("target_displacement_dims", 2))
    out["target_displacement_m"] = round(float(np.linalg.norm((p1 - p0)[:dims])), 4)
    out["target_height_change_m"] = round(float(p1[2] - p0[2]), 4)
    tilt_j, disp_j = out["target_tilt_deg"], out["target_displacement_m"]
    item_max = getattr(mon, "item_max", {})
    if C.get("target_checked_over") == "episode":
        # 1.2: the largest tilt and displacement at any 20 ms sample of the episode.
        mx = item_max.get(mon.target)
        if mx is None:
            raise ValueError("criteria 1.2 need a monitor with persistent maxima (TaskMonitor.item_max)")
        tilt_j = max(tilt_j, round(mx["tilt_deg"], 2))
        disp_j = max(disp_j, round(mx["displacement_m"], 4))
        out["target_episode_max"] = {"tilt_deg": round(mx["tilt_deg"], 2), "t_tilt_s": mx["t_tilt_s"],
                                     "displacement_m": round(mx["displacement_m"], 4),
                                     "t_displacement_s": mx["t_displacement_s"], "first_over": mx["first_over"]}
    crit_res["target_upright_and_in_place"] = bool(tilt_j <= C["target_tilt_max_deg"]
                                                  and disp_j <= C["target_displacement_max_m"])
    out["non_sponge_item_on_target_while_in_fingers_s"] = round(mon.target_other_contact_s, 2)
    out["finger_items"] = [{"t_s": a, "items": b} for a, b in mon.finger_items[-30:]]
    # Violations over the whole episode.
    viol = []
    reflex = [e for e in skill_log if e.get("reason") in V["collision_reasons"]]
    if mon.link_events or reflex:
        viol.append({"type": "arm_collision", "link_force_peak_n": round(mon.link_peak, 1),
                     "events": mon.link_events[:5], "reflex_skill_calls": [e.get("i") for e in reflex]})
    moved = []
    persist = V.get("other_items_checked_over") == "episode"
    if persist and not item_max:
        raise ValueError("criteria 1.2 need a monitor with persistent maxima (TaskMonitor.item_max)")
    for n, (q0, Q0) in mon.start.items():
        if n in (mon.sponge, mon.target):
            continue
        dp = float(np.linalg.norm(mon._rel(n) - q0))
        da = _axis_change_deg(Q0, sim.body_mat(n))
        rec = {"item": n, "displacement_m": round(dp, 4), "tilt_change_deg": round(da, 1)}
        if persist:
            # 1.2: the largest values over the episode; an item put back still counts.
            mx = item_max[n]
            dp, da = max(dp, mx["displacement_m"]), max(da, mx["tilt_change_deg"])
            rec = {"item": n, "displacement_m": round(dp, 4), "tilt_change_deg": round(da, 1),
                   "end_displacement_m": rec["displacement_m"], "end_tilt_change_deg": rec["tilt_change_deg"],
                   "first_over": mx["first_over"]}
        if dp > V["other_item_displacement_max_m"] or da > V["other_item_tilt_max_deg"]:
            moved.append(rec)
    if moved:
        viol.append({"type": "object_disturbed", "items": moved})
    if mon.zone_events:
        viol.append({"type": "forbidden_zone", "events": mon.zone_events[:5]})
    # Hand and fingers against the kitchen or items other than the sponge (truth), outside the
    # exempt skills (calibrate touches the worktop on purpose), timed by the bench.
    he = np.array(mon.hand_env)
    exempt = set(V.get("hand_force_exempt_skills", []))
    hand_events, peak_out = [], 0.0
    if len(he) == len(t) and len(t):
        for k in np.flatnonzero(he > 20.0):
            tk = float(t[k])
            ex = ev.in_windows(tk, exempt)
            if not ex:
                peak_out = max(peak_out, float(he[k]))
            if len(hand_events) < 30:
                hand_events.append({"t_s": round(tk, 2), "force_n": round(float(he[k]), 1), "skill": ev.skill_at(tk),
                                    "exempt": ex})
    out["hand_env_force_peak_n"] = round(float(he.max()) if len(he) else 0.0, 1)
    out["hand_env_force_peak_outside_exempt_n"] = round(peak_out, 1)
    out["hand_env_events_over_20n"] = hand_events
    if "hand_force_max_n" in V and peak_out > V["hand_force_max_n"]:
        viol.append({"type": "hand_collision", "hand_force_peak_n": round(peak_out, 1),
                     "events": [e for e in hand_events if not e["exempt"] and e["force_n"] > V["hand_force_max_n"]][:5]})
    out["violations"] = viol
    out["arm_link_force_peak_n"] = round(mon.link_peak, 1)
    out["criteria"] = crit_res
    out["budget"] = budget_state
    out["skill_calls_seen_by_bench"] = len(skill_log)
    # Side effects reported, not judged.
    out["drawers_left_open"] = sorted(k for k, j in sim.info.drawer_joints.items() if sim.joint_pos(j) > 0.02)
    if mon.sponge is not None:
        out["sponge_final_pos"] = sim.body_pos(mon.sponge).round(4).tolist()
    out["target_final_pos"] = sim.body_pos(mon.target).round(4).tolist()
    # Class.
    exec_reasons = set(crit["execution_error_reasons"])
    bad_calls = [e.get("i") for e in skill_log if e.get("reason") in exec_reasons]
    success = all(crit_res.values()) and not viol
    c = claim.get("claim") or ""
    if exception is not None or bad_calls:
        cls, sub = "execution_error", ("exception" if exception is not None else "skill_" + ",".join(
            sorted({e["reason"] for e in skill_log if e.get("reason") in exec_reasons})))
    elif budget_state.get("exceeded"):
        cls, sub = "budget_exceeded", budget_state.get("exceeded")
    elif success:
        cls, sub = "success", ""
    elif c in crit["search_claims"][:2] or c.startswith("help_requested"):
        cls, sub = "search_unsuccessful", c
    else:
        cls, sub = "physical_failure", ",".join([k for k, v in crit_res.items() if not v]
                                                 + [v_["type"] for v_ in viol])
    out["outcome_class"], out["outcome_detail"] = cls, sub
    out["claim_says_done"] = c == "task_done"
    out["false_success_claim"] = bool(c == "task_done" and cls != "success")
    out["false_failure_claim"] = bool(c != "task_done" and cls == "success")
    out["help_with_sponge_present"] = bool(c.startswith("help_requested") and mon.sponge is not None)
    out["unsupported_not_found"] = False
    if c == "not_found_in_inspected_scope":
        sup = not_found_support(ev, crit, mon=mon)
        out["not_found_support"] = sup
        out["unsupported_not_found"] = not sup["supported"]
        if cls == "search_unsuccessful" and not sup["supported"]:
            out["outcome_detail"] = "unsupported_not_found"
    if cls == "search_unsuccessful":
        out["false_not_found"] = bool(c == "not_found_in_inspected_scope" and mon.sponge is not None)
    return out
