"""Overhead pre-check of a top pick: does something lie over the target, so that the hand going down would hit it?

Rule B (active): the depth points of the view that set the target's pose, back-projected with that view's believed
camera pose, that lie inside the target's inscribed disk (catalog short side / 2 - OVERHEAD_DISK_SHRINK_M) and from the
perceived top + OVERHEAD_TOP_MARGIN_M up to the pick's approach height. OVERHEAD_N_MIN points or more refuse the pick
before any motion (OVERHEAD_REFUSAL_CODE, not a contact incident). Wrist pixels closer than OVERHEAD_SELF_DEPTH_M are
the robot's own hand and are ignored.
Rule F (RULE_F_ACTIVE; otherwise recorded only): the same test over the vertical sweep of the robot's own finger
collision boxes, from the grasp pose up to the approach height.

Why: an uncatalogued board that is part of a drawer, over a perceived sponge, is invisible to the dossier and to the
catalog geometry, and a straight top grasp hits it. The thresholds are declared calibration constants of the simulated
sensor. Declared limits: a structure over the finger column only is not refused unless rule F is active; only the pose
frame is read; a target taller than its catalog height reads as an obstruction (errs toward refusing).

overhead_check() is a pure function of a depth frame, a camera pose and the pick geometry: it reads and writes no state
and never reads the simulator. hand_boxes() derives the finger boxes from the robot's own arm model.
"""
from __future__ import annotations

import math

import numpy as np

OVERHEAD_VERSION = "overhead-check-1"  # rule B active, rule F as RULE_F_ACTIVE says (pick-pose-3)
OVERHEAD_N_MIN = 20  # points in the pose frame: 20 or more refuse (prototype N_MIN)
OVERHEAD_TOP_MARGIN_M = 0.006  # points count from the perceived top + 6 mm (prototype MZ_TOP)
OVERHEAD_DISK_SHRINK_M = 0.005  # inscribed disk radius = catalog short side / 2 - 5 mm (prototype FP_SHRINK)
OVERHEAD_SELF_DEPTH_M = 0.15  # wrist-camera pixels closer than this are the robot's own hand (prototype SELF_DEPTH_M)
OVERHEAD_FINGER_BOX_MARGIN_M = 0.003  # finger boxes grown by 3 mm across (prototype MXY)
RULE_F_ABOVE_TOP_M = 0.010  # rule F counts finger-column points 10 mm or more above the perceived top
APPROACH_ABOVE_WORKTOP_M = 0.40  # the pick skill's approach height: max(worktop + 0.40, grasp TCP + 0.12)
APPROACH_ABOVE_TCP_M = 0.12
OVERHEAD_REFUSAL_CODE = "pick_overhead_obstruction"
RULE_B_ACTIVE = True
# Rule F is active only if it refuses none of the 234 recorded successful picks and its worst normal pick reads at most
# 10 points. The recorded runs met that (0 / 234 refused, worst normal pick 0 points), the same reading that made the
# host unit-top clearance active (agent.host.PICK_UNIT_TOP_ACTIVE).
RULE_F_ACTIVE = True

_BOXES: dict = {}


def approach_height(worktop_z: float, tcp_z: float) -> float:
    """The pick skill's approach height for a top grasp whose TCP ends at tcp_z (robot.skills.Robot.pick: `above`)."""
    return max(float(worktop_z) + APPROACH_ABOVE_WORKTOP_M, float(tcp_z) + APPROACH_ABOVE_TCP_M)


def hand_boxes(open_w: float) -> list:
    """[(part, x_half, y_lo, y_hi, z_lo, z_hi)] in the hand frame (z toward the fingertips) of the collision geoms of the
    robot's own hand and fingers, fingers open at open_w (symmetrised in x, the fingers by |y|). Self knowledge: the arm
    model the robot plans with (robot.kinematics.ArmKinematics), computed once per opening."""
    key = round(float(open_w), 4)
    if key in _BOXES:
        return _BOXES[key]
    import mujoco  # noqa: PLC0415 (the robot's own arm model, robot.kinematics)
    from ..robot.kinematics import ArmKinematics  # noqa: PLC0415
    kin = ArmKinematics()
    m = kin.m
    d = mujoco.MjData(m)
    for fj in ("finger_joint1", "finger_joint2"):
        j = m.joint(fj)
        d.qpos[j.qposadr[0]] = float(np.clip(open_w / 2, m.jnt_range[j.id, 0], m.jnt_range[j.id, 1]))
    mujoco.mj_kinematics(m, d)
    hid = m.body("hand").id
    Hp, HR = d.xpos[hid].copy(), d.xmat[hid].reshape(3, 3).copy()
    part_of = {m.body("left_finger").id: "finger", m.body("right_finger").id: "finger", hid: "hand"}
    out = []
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        if b not in part_of or not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
        corners = np.array([c + np.array([i, j, k]) * h for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])
        w = corners @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        hl = (w - Hp) @ HR
        xh = float(np.abs(hl[:, 0]).max())
        if part_of[b] == "finger":
            ya = np.abs(hl[:, 1])
            out.append(("finger", xh, float(ya.min()), float(ya.max()), float(hl[:, 2].min()), float(hl[:, 2].max())))
        else:
            out.append(("hand", xh, 0.0, float(np.abs(hl[:, 1]).max()), float(hl[:, 2].min()), float(hl[:, 2].max())))
    _BOXES[key] = out
    return out


def overhead_check(depth, intrinsics, cam_p, cam_R, pick_pos, top_z: float, r_in: float, above_z: float, *,
                   self_depth_m: float | None = None, finger_boxes=None, hand_origin=None, hand_R=None,
                   sweep_m: float | None = None, tcp_z: float | None = None) -> dict:
    """Rule B (active) and rule F (recorded) on one depth frame (module docstring). Pure function.

    depth: (H, W) z-depth image (m, 0 = invalid); intrinsics: (fx, fy, cx, cy); cam_p, cam_R: the believed camera pose
    of that frame (camera looks along -z); pick_pos: the pick position (world, the target's perceived centre); top_z:
    the target's perceived top; r_in: the radius of the target's inscribed disk (shrunk); above_z: the approach height;
    self_depth_m: ignore pixels closer than this (wrist camera: the robot's own hand). Rule F needs finger_boxes
    (hand_boxes), hand_origin and hand_R (the hand frame at the grasp pose), sweep_m (the vertical travel from the
    grasp pose to the approach height) and tcp_z (the grasp TCP height); without them it is not evaluated (None).
    Returns {"version", "rule": "B", "refuse", "code", "n_points", "n_min", "rules": {"B": {...}, "F": {...}}}."""
    fx, fy, cx, cy = (float(x) for x in intrinsics[:4])
    pos = np.asarray(pick_pos, float).reshape(3)
    dep = np.asarray(depth)
    v, u = np.nonzero(dep > 0)
    dd = dep[v, u].astype(float)
    if self_depth_m is not None:
        keep = dd >= float(self_depth_m)
        v, u, dd = v[keep], u[keep], dd[keep]
    x = (u + 0.5 - cx) / fx * dd
    y = -(v + 0.5 - cy) / fy * dd
    P = np.column_stack([x, y, -dd]) @ np.asarray(cam_R, float).T + np.asarray(cam_p, float)
    hz = P[:, 2]
    dxy = P[:, :2] - pos[:2]
    over = hz >= float(top_z) + OVERHEAD_TOP_MARGIN_M
    in_disk = np.hypot(dxy[:, 0], dxy[:, 1]) <= float(r_in)
    n_b = int((in_disk & over & (hz <= float(above_z))).sum())
    fires_b = n_b >= OVERHEAD_N_MIN
    n_f = None
    if (finger_boxes is not None and hand_origin is not None and hand_R is not None and sweep_m is not None
            and tcp_z is not None):
        # the prototype's coarse window: within 0.15 m of the pick position, from the TCP - 3 cm to the approach + 5 cm
        near = (np.hypot(dxy[:, 0], dxy[:, 1]) <= 0.15) & (hz >= float(tcp_z) - 0.03) & (hz <= float(above_z) + 0.05)
        ho = np.asarray(hand_origin, float)
        ph = (P - ho) @ np.asarray(hand_R, float)
        hz0 = float(ho[2])
        fing = np.zeros(len(P), bool)
        for part, xh, ylo, yhi, zlo, zhi in finger_boxes:
            if part != "finger":
                continue
            mg = OVERHEAD_FINGER_BOX_MARGIN_M
            inxy = (np.abs(ph[:, 0]) <= xh + mg) & (np.abs(ph[:, 1]) >= ylo - mg) & (np.abs(ph[:, 1]) <= yhi + mg)
            inz = (hz >= hz0 - zhi) & (hz <= hz0 - zlo + float(sweep_m))
            fing |= inxy & inz
        n_f = int((near & fing & (hz >= float(top_z) + RULE_F_ABOVE_TOP_M)).sum())
    fires_f = None if n_f is None else n_f >= OVERHEAD_N_MIN
    refuse = bool((RULE_B_ACTIVE and fires_b) or (RULE_F_ACTIVE and bool(fires_f)))
    rule = "F" if (RULE_F_ACTIVE and bool(fires_f) and not (RULE_B_ACTIVE and fires_b)) else "B"
    return {"version": OVERHEAD_VERSION, "rule": rule, "refuse": refuse, "code": OVERHEAD_REFUSAL_CODE if refuse else None,
            "n_points": n_b, "n_min": OVERHEAD_N_MIN,
            "rules": {"B": {"active": RULE_B_ACTIVE, "n": n_b, "fires": bool(fires_b)},
                      "F": {"active": RULE_F_ACTIVE, "n": n_f, "fires": fires_f}},
            "top_z": round(float(top_z), 5), "above_z": round(float(above_z), 5), "r_in": round(float(r_in), 5)}


def not_evaluated(why: str) -> dict:
    """The result when the check cannot be evaluated (no pose frame kept for the target): no refusal, recorded."""
    return {"version": OVERHEAD_VERSION, "rule": "B", "refuse": False, "code": None, "n_points": None,
            "n_min": OVERHEAD_N_MIN, "rules": {"B": {"active": RULE_B_ACTIVE, "n": None, "fires": None},
                                               "F": {"active": RULE_F_ACTIVE, "n": None, "fires": None}},
            "not_evaluated": str(why)}


def skill_top_grasp(kind: str) -> tuple:
    """(top_dz, finger opening) of a top grasp of this kind: the robot's own skill catalog (robot.skills.CATALOG)."""
    from ..robot.skills import CATALOG  # noqa: PLC0415 (self knowledge: the skill catalog)
    c = CATALOG[kind]
    return float(c["top_dz"]), float(c["open"])


def grasp_frame(pick_pos, yaw: float, top_dz: float, tcp_offset: float, hand_down) -> tuple:
    """(tcp, hand origin, hand rotation) of a top grasp at pick_pos with the given yaw: the pick skill's own geometry
    (tcp = pos + top_dz; hand origin = tcp - R @ [0, 0, tcp_offset], R = hand_down(yaw))."""
    pos = np.asarray(pick_pos, float).reshape(3)
    R = hand_down(float(yaw))
    tcp = pos + np.array([0.0, 0.0, float(top_dz)])
    return tcp, tcp - R @ np.array([0.0, 0.0, float(tcp_offset)]), R


def default_yaw(pick_pos, base_xy) -> float:
    """The pick skill's default top-grasp yaw: the radial direction from the believed base, + 90 degrees."""
    v = np.asarray(pick_pos, float)[:2] - np.asarray(base_xy, float)[:2]
    n = float(np.linalg.norm(v))
    v = v / n if n > 1e-6 else np.array([1.0, 0.0])
    return math.atan2(float(v[1]), float(v[0])) + math.pi / 2
