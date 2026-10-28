"""Robot skills: the gestures robot programs call, with guarded motions and machine-readable outcomes.

Every skill returns SkillResult(ok, reason, data) and never raises: a bad argument returns "bad_arg", an unexpected
exception "internal_error"; data["arm_moved"] says whether the arm moved before the skill returned. Reason codes:
- planning: "unreachable" (no IK solution), "no_path" (no continuous, collision-free motion);
- runtime safety: "collision" (force on an arm link above 20 N, or on the hand above 60 N outside a guarded segment:
  the motion stops, like a collision reflex), "blocked" (the arm stopped tracking its command);
- grasping: "collision_on_approach", "no_grasp", "slipped", "lift_blocked", "dropped", "not_holding",
  "already_holding", "hand_busy" and the tool-specific holding reasons;
- scrub: "no_contact", "scrub_cut_short", "weak_contact" (under 80 % of the asked time with 1 N or more felt at the
  hand), "force_limit" (own force reading above 15 N);
- drawers: "no_grasp_handle", "stuck", "partial", "lost_handle", "not_closed", "no_end_stop", "pull_<reason>",
  "push_<reason>"; calibration and placing reasons as named in the code.

Positions are world-frame meters (z up). The robot turns them into its base frame with its BaseBelief
(robot.belief); a wrong belief makes the real hand miss. A skill may use only its arguments, the arm joint positions
and its own commands, the gripper width, the IMU and the contact forces on robot links in the robot's own base frame
(siteagent.sim); never the poses of simulator bodies.

Motion corrections, recorded with their versions in the result's records only: open-ladder-1 (a drawer pull whose
full line cannot be planned falls back to slightly shorter strokes), tilt-retreat-1 (after a tilted look, the next
arm motion first retreats in front of the look's aim), look-exit-1 (after look_pose, the next motion first reverses
the direct HOME -> LOOK segment) and pinch-rest-2 (a rail pinch whose fingertips rest on the rail instead of
straddling it is lifted, reopened and pinched again under the same guard). None of them changes a threshold.
"""
from __future__ import annotations

import functools
import math
import sys
from dataclasses import dataclass, field

import mujoco
import numpy as np

from ..sim import CTRL_DT, GRIP_CLOSED, GRIP_OPEN
from ..world.build import SPRAY_DIR_LOCAL, SPRAY_NOZZLE_LOCAL
from .belief import BaseBelief, tilt_from_accel, rpy_to_mat
from .kinematics import (HOME, REST, ArmKinematics, hand_down, hand_horizontal, mat_from_quat,
                         quat_from_mat, rot_about)

MAX_OPEN = 0.08
# grip-state contracts (robot.held), selectable like the scrub gestures (Robot(held_contract=...)). robot-held-1: the
# default of every condition (S, A and the agent), see Robot._reconcile_held; held-0: the earlier contract, kept to
# reproduce earlier runs.
ROBOT_HELD_CONTRACT = "robot-held-1"
LEGACY_HELD_CONTRACT = "held-0"
# open-ladder-1: the line lengths a drawer pull stroke falls back to when its whole line cannot be planned. Without
# it (0.215, 0.21, ...) a full pull (host open-full-1, about 0.24 m) or the scripted chain's 0.24 m fell to 0.215 m,
# below the 0.22 m the live command pulls from the same pinch. The rungs 0.235, 0.23, 0.225 and 0.22 are tried first;
# only requests over 0.224 m reach them (a rung is tried only below the request minus 4 mm), so a 0.22 m request tries
# exactly the same lengths as without the ladder.
OPEN_LADDER_VERSION = "open-ladder-1"
OPEN_STROKE_FALLBACKS_M = (0.235, 0.23, 0.225, 0.22, 0.215, 0.21, 0.205, 0.20, 0.18, 0.16, 0.14, 0.12, 0.10, 0.08,
                           0.06)
# tilt-retreat-1: after a tilted look (a move_to given retreat_aim that ended ok: search.SiteSearch.
# look_under_top, used by the host's look_at_region(drawer, "unseen") and the scripted chain's re-look), the next
# top-level skill that moves the arm first makes a planned hand-down transit (move_to(orient="down", transit=True)) to
# RETREAT_IN_FRONT_M in front of the look's aim (horizontally, toward the hand) at worktop + RETREAT_ABOVE_WORKTOP_M,
# where the live straight-down look left the arm (Robot._retreat_now). See Robot._retreat_now for the rest.
TILT_RETREAT_VERSION = "tilt-retreat-1"
RETREAT_IN_FRONT_M = 0.10
RETREAT_ABOVE_WORKTOP_M = 0.40
RETREAT_PLANNING_REASONS = ("no_path", "unreachable")
RETREAT_FALLBACK = "skill_from_current_pose"  # what follows a retreat that did not reach its waypoint without a contact
# look-exit-1: the first arm motion after look_pose leaves LOOK by the exact reverse of look_pose's
# own direct joint segment. LOOK_SEGMENT_SPEED / LOOK_SEGMENT_MIN_T are that segment's speed (the literal 1.2 of
# look_pose) and _joint_move_raw's default min_t, which it used: same values, named.
LOOK_EXIT_VERSION = "look-exit-1"
LOOK_SEGMENT_SPEED = 1.2
LOOK_SEGMENT_MIN_T = 0.4
# pinch-rest: PINCH_GUARD_N is the rail pinch approach's descent
# guard (the literal 8.0 of _pinch_rail; pick and press keep their own literals), reused as the rest check's level;
# RAIL_HALF_THICKNESS_M is the rail handle's half height (robot-side catalog, equal to
# perception.catalog.HANDLE_HALF_THICKNESS; the robot package does not import perception). pinch-rest-2:
# the same two constants make the trigger (_pinch_resting): up above PINCH_GUARD_N and the
# gripper width below 2 x RAIL_HALF_THICKNESS_M (the 1.6 cm rail, strict).
PINCH_REST_VERSION = "pinch-rest-2"
PINCH_GUARD_N = 8.0
RAIL_HALF_THICKNESS_M = 0.008


def _pinch_resting(up_n: float, width_m: float) -> bool:
    """pinch-rest-2's rest predicate, used for the trigger (up, w) and for the re-check's still_resting (up2, w2): the
    hand's untared upward reaction above PINCH_GUARD_N with the fingers closed narrower than the rail
    (2 x RAIL_HALF_THICKNESS_M; both comparisons strict). Fingertips resting on the rail's top face close past its
    faces (corner pinches read 13.3-14.8 mm, simulation), while a pinch with the rail between the fingers cannot be
    narrower than the rail and, below that width, cannot read more than the guard level (the pads' friction bound)."""
    return bool(up_n > PINCH_GUARD_N and width_m < 2 * RAIL_HALF_THICKNESS_M)
# Franka joint speed limits (rad/s, joints 1-4 and 5-7) with a 10 % margin; a held object
# moves at most 0.8 rad/s per joint (a held cup is flung by fast joint moves).
QD_LIMIT = 0.9 * np.array([2.175] * 4 + [2.61] * 3)
QD_HELD = 0.8
QD_GUARDED = 0.4  # joint speed limit while a force guard watches for an expected contact
# Collision reflex: net contact force on any arm link (link0-link7), and on the hand plus
# fingers outside a guarded segment (where the skill's own force guard watches).
# Null-space pull toward the rest posture while tracking a path step by step: weaker than the
# IK default, so two consecutive solutions stay continuous (the default 0.05 made the first step
# after an unusual posture jump 0.4 rad), but not zero (long paths drift into joint limits).
TRACK_POSTURE = 0.03
LINK_FORCE_LIMIT = 20.0
HAND_FORCE_LIMIT = 60.0
# Site plan: no module rises more than 0.18 m above the worktop (dishwasher with its button,
# sink walls, drawer rails; the faucet stands at the back of the sink, 0.83 m from the base).
# A held object is carried with its bottom at least 2 cm above that.
ENVELOPE_H = 0.18
SINK_RIM_H = 0.14  # site plan: sink walls above the worktop (the spray bottle stays above them)
# Site plan: modules stand on a ring around the robot, their fronts at least 0.46 m from the base.
# Away from the skill's own target zone, arm links over that ring stay above the tallest module.
MODULE_RING_R = 0.46
ZONE_R = 0.30
TIP_BELOW_TCP = 0.008  # closed fingertips stick out this far past the TCP along the hand axis
TIP_HALF_X = 0.0105  # closed fingertips half thickness along the hand x axis
# Effective probe offset of the side-edge touch, measured once on calibration sites 2000-2021 not bench sites: 1.5 mm
# less than the front touch (systematic, std 0.1 mm).
TIP_SIDE_EXTRA = 0.0015

# Look pose: arm folded low beside the base, toward the aisle. Chosen on held-out sites 2000-2032 (not bench sites): the
# arm hides at most 3.4 % of the modules' pixels in robot_eye. Joint 4 at -2.9 pressed the left finger into link 1 (self
# contact, the arm stopped at -2.80); -2.75 is free and gives the same view.
LOOK = np.array([-2.5, 0.3, 0.0, -2.75, 0.0, 1.42, 0.785])

# The robot's object catalog (self knowledge about objects, like a grasp affordance table).
# hh: half height when upright; top_dz: TCP above the centre for a top grasp;
# side_dz: TCP above the centre for a side grasp; open: finger opening before closing.
CATALOG = {
    "cup": dict(hh=0.045, top_dz=0.015, side_dz=0.006, open=0.074, side=True),
    "bowl": dict(hh=0.022, top_dz=0.010, side_dz=None, open=0.074, side=False),
    "fork": dict(hh=0.006, top_dz=0.0, side_dz=None, open=0.050, side=False, low=True),
    "spoon": dict(hh=0.006, top_dz=0.0, side_dz=None, open=0.050, side=False, low=True),
    # Sponge: held by its upper half so the fingertips stay 1.6 cm above the scrubbing face. grip: closed to 4 cm, not
    # fully (at full force the fingers crept through the soft sponge within about 10 s and it fell). Also:
    # carried at <= 0.4 rad/s per joint (qd_held). At the general 0.8 rad/s the soft sponge crept 1 to 1.5 cm toward the
    # fingertips during carries (arm motion, not static holds) and stayed on the dish when the hand lifted after
    # scrubbing (reference kitchen: program A 19/20, scripted chain 6/10, every failure "dropped").
    "sponge": dict(hh=0.014, top_dz=0.010, side_dz=None, open=0.074, side=False, grip=0.040, qd_held=0.4,
                   # v2.2: closed to 4 cm as before, then held at 4.8 cm, a lighter squeeze. Held at 4 cm the fingers
                   # kept closing into the soft sponge (width 5.6 -> 4.2 cm in a carry) while it slid toward the
                   # fingertips (median 2-7 mm over the hold, 23 to 36 mm by the loss), until it left the pads; worst
                   # when the finger axis is skewed from the sponge's narrow axis (pick default yaw, 11-12 deg in the
                   # reference drawer). Reference kitchen, pick default yaw: 5/20 drops at 4 cm, 0/20 at 4.8 cm. Known
                   # limit: the squeeze per finger is about 250 N/m x (width -
                   # command) (net force vectors, equal and opposite): 2.2 N after the grasp, 0.2-0.4 N after a 35 s
                   # carry as the width creeps toward 4.8 cm. With a skewed grasp the sponge then rolled about 6 deg in
                   # the hand and the scrub loaded the bowl (default yaw: 45.7 N, bowl moved 4.3 cm; 1 of 169
                   # tuning placements). Squeezing again for the scrub (to 4 cm, or to width - 8 mm >= 4.2 cm) and a
                   # constant-squeeze carry were measured and discarded: drops after the scrub, sponge pushed out of the
                   # fingers, carry drops. With the perceived yaw (as program S) the same episode passes (4.5 N, bowl in
                   # place).
                   grip_v22=0.048),
    # Side grasp 6.2 cm above the centre (5.5 cm left the palm 3 mm above a sink's
    # front wall); the pads (+-9 mm) stay on the body, below the trigger head.
    "spray_bottle": dict(hh=0.075, top_dz=0.0, side_dz=0.062, open=0.070, side=True,
                         lying_hh=0.030, com_local=(0.0, 0.0, 0.014),
                         # Corners of the body (6 x 4.4 x 15 cm) and of the trigger head.
                         corners_local=tuple((x, y, z) for x in (-0.03, 0.03) for y in (-0.022, 0.022)
                                             for z in (-0.075, 0.075))
                         + tuple((x, y, 0.115) for x in (-0.012, 0.012) for y in (-0.015, 0.035))),
    "towel": dict(hh=0.0125, top_dz=0.0, side_dz=None, open=0.074, side=False, low=True),
}


class _Abort(Exception):
    """Stops the current top-level skill (collision reflex, blocked arm, bad argument)."""

    def __init__(self, reason: str, data: dict | None = None):
        super().__init__(reason)
        self.reason = reason
        self.data = data or {}


def _skill(fn):
    """Public skill wrapper: never raises, reports whether the arm moved. Nested skill calls
    let an abort through to the outermost skill, which stops."""
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        top = self._depth == 0
        n0 = self.n_cmd
        if top:  # tilt-retreat-1: the retreat state when this skill starts, and its events (Robot._retreat_close)
            pending0 = getattr(self, "_retreat_pending", None)
            self._retreat_events = []
            self._retreat_skill = fn.__name__
            # look-exit-1 and pinch-rest (pinch-rest-2): the events of this top-level skill (Robot._v473c_close)
            self._look_exit_events = []
            self._pinch_rest_events = []
        self._depth += 1
        try:
            r = fn(self, *args, **kwargs)
        except _Abort as e:
            if not top:
                raise
            self._recover()
            r = SkillResult(False, e.reason, dict(e.data))
        except Exception as e:  # noqa: BLE001 (the S5 sandbox must never see an exception)
            if not top:
                raise
            self._recover()
            r = SkillResult(False, "internal_error", {"error": f"{type(e).__name__}: {e}"})
        finally:
            self._depth -= 1
            if top:
                self._zone = None
        if top:
            r.data["arm_moved"] = bool(self.n_cmd != n0)
            self._retreat_close(r, pending0, n0)
            self._v473c_close(r)
        return r
    return wrapper


def _vec(name, v, n=3) -> np.ndarray:
    try:
        a = np.asarray(v, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        raise _Abort("bad_arg", {"error": f"{name} must be {n} numbers"})
    if a.shape != (n,) or not np.all(np.isfinite(a)):
        raise _Abort("bad_arg", {"error": f"{name} must be {n} finite numbers, got {v!r}"})
    return a


def _hdir(name, v) -> np.ndarray:
    try:
        a = np.asarray(v, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        raise _Abort("bad_arg", {"error": f"{name} must be a direction"})
    if a.shape not in ((2,), (3,)) or not np.all(np.isfinite(a)) or np.hypot(a[0], a[1]) < 0.1:
        raise _Abort("bad_arg", {"error": f"{name} must be a horizontal direction (x, y) with norm > 0.1, got {v!r}"})
    h = np.array([a[0], a[1], 0.0])
    return h / np.linalg.norm(h)


def _num(name, v, lo, hi, allow_none=False):
    if v is None and allow_none:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise _Abort("bad_arg", {"error": f"{name} must be a number, got {v!r}"})
    if not (math.isfinite(x) and lo <= x <= hi):
        raise _Abort("bad_arg", {"error": f"{name} must be in [{lo}, {hi}], got {v!r}"})
    return x


def _quat(name, v):
    if v is None:
        return None
    q = _vec(name, v, 4)
    n = float(np.linalg.norm(q))
    if abs(n - 1.0) > 0.05:
        raise _Abort("bad_arg", {"error": f"{name} must be a unit quaternion (w, x, y, z), got norm {n:.3f}"})
    return q / n


@dataclass
class SkillResult:
    ok: bool
    reason: str = "ok"
    data: dict = field(default_factory=dict)

    def __bool__(self):
        return self.ok


def _smooth(s: float) -> float:
    return 3 * s * s - 2 * s ** 3


def _slerp(q0, q1, s):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    d = float(q0 @ q1)
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + s * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1 - s) * th) * q0 + math.sin(s * th) * q1) / math.sin(th)


def _rot_angle(R0, R1) -> float:
    c = (np.trace(R0.T @ R1) - 1) / 2
    return math.acos(max(-1.0, min(1.0, c)))


def _flip_z(R):
    """Same hand pose turned 180 deg about its own axis (parallel gripper symmetry)."""
    return R @ np.diag([-1.0, -1.0, 1.0])


class Robot:
    PLAN_EDGES = {"front_x": 0.29, "side_y": 0.29}  # site plan: worktop inner edges (U opening)

    SCRUB_GESTURES = ("v1", "v2", "v2.1", "v2.2")
    DEFAULT_SCRUB_GESTURE = "v2.2"  # run_task's --scrub-gesture default follows it (A/B/C references)
    HELD_CONTRACTS = (LEGACY_HELD_CONTRACT, ROBOT_HELD_CONTRACT)
    DEFAULT_HELD_CONTRACT = ROBOT_HELD_CONTRACT  # run_task's --held-contract default follows it

    def __init__(self, sim, belief: BaseBelief | None = None, worktop_z: float = 0.90, scrub_gesture: str = "v2.2",
                 *, held_contract: str = ROBOT_HELD_CONTRACT):
        """scrub_gesture: "v2.2" (default: v2.1 plus the lighter sponge grip, CATALOG sponge grip_v22, see pick),
        "v2.1" (v2 plus the grip checks below; the previous default, kept to reproduce earlier runs), "v2" (kept to
        reproduce earlier runs) or "v1" (the first gesture, kept for comparison runs). The sponge is only picked to
        scrub, so its grasp policy is versioned with the scrub gesture. held_contract (keyword only): "robot-held-1"
        (default, the same for every condition: S, A and the agent; robot.held ends when holding() reads False, see
        _reconcile_held) or "held-0" (robot.held kept after open_gripper and close_gripper). The bench records it
        next to scrub_gesture (bench_task)."""
        if scrub_gesture not in self.SCRUB_GESTURES:
            raise ValueError(f"scrub_gesture must be one of {self.SCRUB_GESTURES}")
        if held_contract not in self.HELD_CONTRACTS:
            raise ValueError(f"held_contract must be one of {self.HELD_CONTRACTS}")
        self.scrub_gesture = scrub_gesture
        self.held_contract = held_contract
        self.sim = sim
        self.kin = ArmKinematics()
        self.worktop_z = float(worktop_z)
        self.belief = belief if belief is not None else BaseBelief.nominal(self.worktop_z)
        self.q_cmd = sim.arm_q()
        self.grip_cmd = sim.gripper_cmd()
        self.held: dict | None = None
        self.trace: list[str] = []
        self.edges = dict(self.PLAN_EDGES)
        self.n_cmd = 0  # arm commands sent (tells whether a skill moved the arm)
        self._depth = 0
        self._guarded = False
        self._zone = None  # (x, y) of the current skill's target: modules near it may be reached into
        # tilt-retreat-1: the look aim of the last tilted look while its retreat is due ({"aim": [x, y, z]}), the
        # retreat events of the current top-level skill (copied into its result, data["retreat"]) and its name
        self._retreat_pending: dict | None = None
        self._retreat_events: list[dict] = []
        self._retreat_skill: str | None = None
        # look-exit-1: the exit due after look_pose ({"t_s", "direct", "q_dev_from_look"}), the last joint move's route
        # (records, read by look_pose) and the events of the current top-level skill (data["look_exit"]); pinch-rest
        # (pinch-rest-2): the rest checks of the current top-level skill (data["pinch_rest"])
        self._look_exit_pending: dict | None = None
        self._last_joint_route: dict | None = None
        self._look_exit_events: list[dict] = []
        self._pinch_rest_events: list[dict] = []

    # ================================================================ frames
    def _to_base(self, p_w, R_w=None):
        pb = self.belief.world_to_base(p_w)
        Rb = None if R_w is None else self.belief.rot_world_to_base(R_w)
        return pb, Rb

    def tcp_pose(self, q=None):
        """Believed TCP pose in the world (from joint positions and the belief)."""
        q = self.sim.arm_q() if q is None else q
        pb, Rb = self.kin.fk(q)
        return self.belief.base_to_world(pb), self.belief.rot_base_to_world(Rb)

    def _radial(self, p_w) -> np.ndarray:
        """Horizontal unit vector from the (believed) base toward p_w."""
        v = np.asarray(p_w, float)[:2] - self.belief.t[:2]
        n = np.linalg.norm(v)
        v = v / n if n > 1e-6 else np.array([1.0, 0.0])
        return np.array([v[0], v[1], 0.0])

    # ================================================================ IK helpers
    def _seeds(self, pb, q_hint=None):
        th = math.atan2(pb[1], pb[0])
        seeds = []
        if q_hint is not None:
            seeds.append(np.asarray(q_hint, float))
        seeds.append(self.q_cmd.copy())
        for base in (REST, HOME, np.array([0.0, 0.2, 0.0, -1.9, 0.0, 2.1, 0.785]),
                     np.array([0.0, -0.8, 0.0, -2.6, 0.0, 1.8, 0.785])):
            s = base.copy()
            s[0] = th
            seeds.append(s)
        for s7 in (-2.0, 2.0):
            s = REST.copy()
            s[0], s[6] = th, s7
            seeds.append(s)
        # Hand horizontal near the worktop: shoulder forward, wrist turned (q5 near +-2.4).
        for base in (np.array([1.2, 1.3, -0.7, -2.3, -2.45, 1.85, 1.57]),
                     np.array([-1.2, 1.25, 0.6, -2.5, 2.35, 1.65, 0.0]),
                     np.array([1.2, 1.3, -0.7, -2.3, -2.45, 1.85, -1.57]),
                     np.array([-1.2, 1.25, 0.6, -2.5, 2.35, 1.65, 2.6])):
            s = base.copy()
            s[0] = float(np.clip(th + base[0], -2.8, 2.8))
            seeds.append(s)
        return seeds

    def _solve(self, p_w, R_w, sym=True, q_hint=None, free_axis_w=None):
        """IK for a world pose under the current belief. Returns (q, ok, pos_err, R_used_world)."""
        cands = [R_w, _flip_z(R_w)] if sym else [R_w]
        best = None
        for Rw in cands:
            pb, Rb = self._to_base(p_w, Rw)
            fa = None if free_axis_w is None else self.belief.rot_world_to_base(np.eye(3)) @ free_axis_w
            q, ok, ep, er = self.kin.solve(pb, Rb, self._seeds(pb, q_hint), free_yaw_axis=fa,
                                           q_ref=self.q_cmd)
            cost = float(np.abs(q - self.q_cmd).sum()) + (0 if ok else 100.0 + 100 * ep)
            if best is None or cost < best[0]:
                best = (cost, q, ok, ep, Rw)
        _, q, ok, ep, Rw = best
        return q, ok, ep, Rw

    # ================================================================ execution and reflex
    def _cmd(self, q):
        """Send one 50 Hz arm command, advance one control period, run the collision reflex.
        look-exit-1 (records only): a command sent while the exit is still due is recorded as missed."""
        if getattr(self, "_look_exit_pending", None) is not None:
            self._look_exit_missed(sys._getframe(1).f_code.co_name)
        self.q_cmd = np.asarray(q, float)
        self.sim.set_arm(self.q_cmd)
        self.sim.tick()
        self.n_cmd += 1
        self._reflex()

    def _wait(self, n: int = 1):
        for _ in range(n):
            self.sim.tick()
            self._reflex()

    def _reflex(self):
        """Collision reflex on the robot's own force readings: any arm link pushed by more than
        LINK_FORCE_LIMIT, or the hand by more than HAND_FORCE_LIMIT outside a guarded segment,
        stops the skill."""
        F = self.sim.link_forces_base()
        if not F:
            return
        for b, f in F.items():
            if b in self.sim.link_bodies:
                n = float(np.linalg.norm(f))
                if n > LINK_FORCE_LIMIT:
                    raise _Abort("collision", {"where": "arm_link", "force_n": n})
        if not self._guarded:
            fh = np.zeros(3)
            for b, f in F.items():
                if b in self.sim.hand_bodies:
                    fh += f
            n = float(np.linalg.norm(fh))
            if n > HAND_FORCE_LIMIT:
                raise _Abort("collision", {"where": "hand", "force_n": n})

    def _hold(self):
        """Stop where the arm is (after a guarded stop on contact): command = measured joints,
        so the servo does not keep pushing toward a point beyond the contact."""
        q = self.sim.arm_q()
        self.q_cmd = q
        self.sim.set_arm(q)

    def _recover(self):
        """After an abort: trigger off, hold the arm where it is, update what is held."""
        self.sim.flags["spray_trigger"] = False
        self._guarded = False
        q = self.sim.arm_q()
        self.q_cmd = q
        self.sim.set_arm(q)
        self.sim.tick(10)
        if self.held is not None and not self.holding():
            self.held = None

    def _settle(self, max_s=0.6, tol=0.01):
        """Wait for the arm to reach its command. An arm still 0.02 rad away is blocked."""
        for _ in range(int(max_s / CTRL_DT)):
            if np.abs(self.sim.arm_q() - self.q_cmd).max() < tol:
                return True
            self._wait()
        err = float(np.abs(self.sim.arm_q() - self.q_cmd).max())
        if err > 0.02:
            raise _Abort("blocked", {"track_err_rad": err})
        return False

    # ================================================================ clearance (self knowledge + site plan)
    link_margin = 0.06

    def _clearance(self, q, margin=None) -> float:
        """Signed clearance (m) of a configuration; >= 0 is clear. Checks: self-collision; camera mast; own mount
        (no point, hand included, inside a cylinder of radius 0.22 m below 0.12 m, base frame);
        arm links (link2..link7, not the hand) at least `margin` above the worktop surface where
        there is worktop (outside the U opening of the site plan's edges)."""
        if not self.kin.clear_of_mast(q) or self.kin.self_collides(q):
            return -1.0
        margin = self.link_margin if margin is None else margin
        P = self.kin.link_points(q)
        A = P[3:14]
        mount = float(np.min(np.maximum(np.hypot(A[:, 0], A[:, 1]) - 0.22, A[:, 2] - 0.12)))
        # The mount plate itself (0.26 m square, its top at the base origin) against the links'
        # bounding boxes (conservative already: box corners lie outside the meshes), 1 cm margin.
        B = self.kin.link_box_points(q)
        mount = min(mount, float(np.min(np.maximum(np.maximum(np.abs(B[:, 0]), np.abs(B[:, 1])) - self.MOUNT_BOX_HALF,
                                                   B[:, 2] - 0.01))))
        Q = np.vstack([P[0:6], P[7:12], P[14:16]])  # link2..link7 and the segments between them
        W = Q @ self.belief.R.T + self.belief.t
        fx, sy = self.edges["front_x"] - 0.02, self.edges["side_y"] - 0.02
        over = (W[:, 0] > fx) | (np.abs(W[:, 1]) > sy)
        wt = float(np.min(W[over, 2] - (self.worktop_z + margin))) if np.any(over) else 1.0
        # Module ring, outside the target zone: links above the tallest module (+4 cm, link radius).
        r = np.hypot(W[:, 0] - self.belief.t[0], W[:, 1] - self.belief.t[1])
        ring = r > MODULE_RING_R
        if self._zone is not None:
            ring &= np.hypot(W[:, 0] - self._zone[0], W[:, 1] - self._zone[1]) > ZONE_R
        if np.any(ring) and margin >= 0:
            wt = min(wt, float(np.min(W[ring, 2] - (self.worktop_z + ENVELOPE_H + 0.04))))
        return min(mount, wt)

    def _links_clear(self, q, margin=None) -> bool:
        return self._clearance(q, margin) >= 0.0

    def _floor(self, q_start) -> float:
        """Clearance a path from q_start must keep: clear, or no worse than where it starts."""
        # 1 cm of tolerance: the worktop margin (6 cm) already includes the links' radius, and a
        # motion that starts right at the margin (a grasp near the tray) must be able to leave.
        return min(0.0, self._clearance(q_start)) - 0.01

    def _joint_path_ok(self, q0, q1, n=16) -> bool:
        q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
        fl = min(self._floor(q0), self._floor(q1))
        return all(self._clearance(q0 + s * (q1 - q0)) >= fl for s in np.linspace(0, 1, n)[1:-1])

    def _joint_route(self, q0, q1):
        """Joint waypoints from q0 to q1 whose straight joint interpolations stay clear: direct,
        through HOME, or folded first at the current azimuth. None if none is clear."""
        q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
        if self._joint_path_ok(q0, q1):
            return [q1]
        if (np.abs(q0 - HOME).max() > 1e-3 and np.abs(q1 - HOME).max() > 1e-3
                and self._joint_path_ok(q0, HOME) and self._joint_path_ok(HOME, q1)):
            return [HOME.copy(), q1]
        # Fold first at the current azimuth (joint 1 unchanged), then go.
        for base in (REST, HOME):
            w = base.copy()
            w[0] = q0[0]
            if (np.abs(q0 - w).max() > 1e-3 and not self.kin.self_collides(w)
                    and self._joint_path_ok(q0, w) and self._joint_path_ok(w, q1)):
                return [w, q1]
        return None

    # ================================================================ motion primitives
    def _carry_R(self, target):
        """Hand-down orientation for carrying the held object to `target`: the current gripper
        yaw, turned by the change of azimuth around the base (the wrist does not wind up)."""
        p_now, R_now = self.tcp_pose(self.q_cmd)
        f0, f1 = self.belief.world_to_base(p_now), self.belief.world_to_base(target)
        da = math.atan2(f1[1], f1[0]) - math.atan2(f0[1], f0[0])
        yaw = math.atan2(R_now[1, 1], R_now[0, 1]) + da
        return hand_down(yaw)

    # Turning circle around the base at transit height. The sink's faucet stands about 0.83 m
    # from the base and its walls rise 0.14 m above the worktop; transit height keeps the hand
    # above them, and _swing_route checks every leg against the plan.
    SWING_R = 0.55
    TURN_HIGH = True
    MOUNT_BOX_HALF = 0.14  # mount plate half width 0.13 m + 1 cm
    PRE_DEV = 0.015  # hand deviation allowed while re-reaching the start of a planned line

    def _swing_route(self, q_from, q_target):
        """Joint waypoints from q_from to q_target through via-points on the turning circle at
        transit height (fallback when no continuous Cartesian path exists). Via-points the arm
        cannot take are skipped, but every joint leg is checked; None if no clear route."""
        p0, R0 = self.tcp_pose(q_from)
        p1, _ = self.tcp_pose(q_target)
        vias, da = self._route(p0, p1)
        route, q_prev = [], np.asarray(q_from, float)
        for k, p in enumerate(vias[1:-1], start=1):
            Rk = rot_about([0, 0, 1], da * k / max(len(vias) - 1, 1)) @ R0
            pb, Rb = self._to_base(p, Rk)
            q, ep, er = self.kin.ik(pb, Rb, q_prev, iters=150)
            # A via-point the arm cannot take is skipped; every joint leg that is kept (including
            # the one that then spans the gap) is checked for clearance.
            if (ep < 3e-3 and er < 0.05 and np.abs(q - q_prev).max() < 1.0
                    and self._links_clear(q) and self._joint_path_ok(q_prev, q)):
                route.append(q)
                q_prev = q
        sub = self._joint_route(q_prev, q_target)
        return None if sub is None else route + sub

    def _joint_move(self, q_target, speed=1.0, min_t=0.4):
        """Smooth joint-space move from the current command to q_target, through HOME if the
        straight joint path is not clear, after a 15 cm lift if needed. Aborts with "no_path". look-exit-1: a due
        exit out of LOOK is made first (_look_exit_now); the route taken is kept (records, read by look_pose)."""
        if self._depth >= 1:
            self._look_exit_now("joint_move")
        q_target = np.asarray(q_target, float)
        route = self._joint_route(self.q_cmd, q_target)
        if route is None:
            p, _ = self.tcp_pose(self.q_cmd)
            if self._line(p + [0, 0, 0.15], None, speed=0.15).ok:
                route = self._joint_route(self.q_cmd, q_target)
        if route is None:
            raise _Abort("no_path", {"stage": "joint_move"})
        self._last_joint_route = {"q0": self.q_cmd.copy(), "route": [np.array(q, float) for q in route]}
        for q in route:
            self._joint_move_raw(q, speed, min_t)

    def _held_qd_limit(self) -> np.ndarray:
        """Joint speed limits while holding something: QD_HELD, or the object's own catalog limit
        when it is lower (earlier: the soft sponge)."""
        lim = np.minimum(QD_LIMIT, QD_HELD)
        q = CATALOG.get((self.held or {}).get("kind"), {}).get("qd_held")
        return lim if q is None else np.minimum(lim, q)

    def _joint_move_raw(self, q_target, speed=1.0, min_t=0.4):
        if self._depth >= 1:  # look-exit-1: a due exit out of LOOK first
            self._look_exit_now("joint_move_raw")
        lim = QD_LIMIT
        if self.held is not None:
            speed = min(speed, 0.5)  # a held cup is flung by fast joint moves
            lim = self._held_qd_limit()
        q0 = self.q_cmd.copy()
        dq = np.abs(np.asarray(q_target, float) - q0)
        # Smoothstep peak speed is 1.5 times the mean: keep it within each joint's limit.
        T = max(min_t, float(dq.max()) / speed, 1.5 * float((dq / lim).max()))
        n = max(int(math.ceil(T / CTRL_DT)), 1)
        for k in range(1, n + 1):
            self._cmd(q0 + _smooth(k / n) * (q_target - q0))
        self._settle()

    def _plan_line(self, p0, R0, p1, R1, n, linear, q_start):
        """Joint path (one entry per control tick) for a straight Cartesian line, or None. Every
        configuration keeps the clearance (no worse than at the start)."""
        fl = self._floor(q_start)
        qa0, qa1 = quat_from_mat(R0), quat_from_mat(R1)
        targets = []
        for k in range(1, n + 1):
            s = k / n if linear else _smooth(k / n)
            R = mat_from_quat(_slerp(qa0, qa1, s))
            targets.append(self._to_base(p0 + s * (p1 - p0), R))
        # Plain tracking first: each step's IK seeded from the previous one (same arm branch).
        path, q_prev, okp = [], q_start.copy(), True
        for pb, Rb in targets:
            q, ep, er = self.kin.ik(pb, Rb, q_prev, iters=60, posture=TRACK_POSTURE)
            if (ep > 2e-3 or er > 0.02 or np.abs(q - q_prev).max() > 0.12
                    or self._clearance(q) < fl):
                okp = False
                break
            path.append(q)
            q_prev = q
        if okp:
            return path
        ends = []
        pb1, Rb1 = targets[-1]
        # First the end configuration reached from the start with almost no posture pull (the
        # same arm branch; a strong pull fights joint limits near a twisted start), then seeds.
        for i, seed in enumerate([q_start, q_start] + self._seeds(pb1, q_start)):
            q, ep, er = self.kin.ik(pb1, Rb1, seed, iters=300 if i == 0 else 200,
                                    posture=TRACK_POSTURE if i == 0 else 0.05)
            if (ep < 2e-3 and er < 0.02 and self._clearance(q) >= fl
                    and not any(np.abs(q - e).max() < 0.05 for e in ends)):
                ends.append(q)
            if len(ends) >= 3:
                break
        ends.sort(key=lambda q: float(np.abs(q - q_start).sum()))
        for q_end in ends:
            path, q_prev, okp = [], q_start.copy(), True
            for k, (pb, Rb) in enumerate(targets, start=1):
                seed = q_prev + (q_end - q_prev) / (n - k + 1)
                q, ep, er = self.kin.ik(pb, Rb, seed, iters=60, posture=TRACK_POSTURE)
                if ep > 2e-3 or er > 0.02:
                    q, ep, er = self.kin.ik(pb, Rb, q_prev, iters=150, posture=TRACK_POSTURE)
                if (ep > 3e-3 or er > 0.03 or np.abs(q - q_prev).max() > 0.12
                        or self._clearance(q) < fl):
                    okp = False
                    break
                path.append(q)
                q_prev = q
            if okp:
                return path
        # Backward: track the line from a good end configuration back to the start pose, then
        # reach that start configuration by a short joint move (hand stays near the start).
        for q_end in ends:
            back, q_prev, okp = [q_end], q_end.copy(), True
            for pb, Rb in reversed(targets[:-1]):
                q, ep, er = self.kin.ik(pb, Rb, q_prev, iters=60, posture=TRACK_POSTURE)
                if (ep > 2e-3 or er > 0.02 or np.abs(q - q_prev).max() > 0.12
                        or self._clearance(q) < fl):
                    okp = False
                    break
                back.append(q)
                q_prev = q
            if not okp:
                continue
            pb0, Rb0 = self._to_base(p0, R0)
            q0b, ep, er = self.kin.ik(pb0, Rb0, q_prev, iters=60, posture=TRACK_POSTURE)
            if ep > 2e-3 or er > 0.02 or np.abs(q0b - q_prev).max() > 0.12:
                continue
            m = max(int(float(np.abs(q0b - q_start).max()) / 0.03), 1)
            pre, dev, okp = [], 0.0, True
            for j in range(1, m + 1):
                q = q_start + _smooth(j / m) * (q0b - q_start)
                if self._clearance(q) < fl:
                    okp = False
                    break
                pre.append(q)
                dev = max(dev, float(np.linalg.norm(self.kin.fk(q)[0] - pb0)))
            if not okp or dev > self.PRE_DEV:  # 5 cm let a back-off push into the edge it left
                continue
            return pre + [q0b] + list(reversed(back))
        # Fallback: joint interpolation, accepted if the hand stays within 1 cm of the line.
        for q_end in ends:
            path, okp = [], True
            for k, (pb, Rb) in enumerate(targets, start=1):
                s = k / n if linear else _smooth(k / n)
                q = q_start + s * (q_end - q_start)
                pf, Rf = self.kin.fk(q)
                if np.linalg.norm(pf - pb) > 0.01 or _rot_angle(Rf, Rb) > 0.1 or self._clearance(q) < fl:
                    okp = False
                    break
                path.append(q)
            if okp:
                return path
        return None

    def _line(self, p_w, R_w=None, speed=0.2, guard=None, linear=False, ang_speed=1.2,
              sym=False, min_t=0.15):
        """Straight Cartesian path (believed frame) to (p_w, R_w), planned then executed at 50 Hz
        within the joint speed limits.

        guard: callable returning True to stop (contact). Returns SkillResult with data {'stopped': bool, 'fraction':
        s}. The whole path is planned before moving, so a motion never starts toward an unreachable target. look-exit-1:
        a due exit out of LOOK is made first (_look_exit_now).
"""
        if self._depth >= 1 and getattr(self, "_look_exit_pending", None) is not None:
            self._look_exit_now("line:" + sys._getframe(1).f_code.co_name)
        p0, R0 = self.tcp_pose(self.q_cmd)
        R1 = R0 if R_w is None else np.asarray(R_w, float)
        if sym and R_w is not None and _rot_angle(R0, _flip_z(R1)) < _rot_angle(R0, R1):
            R1 = _flip_z(R1)
        p1 = np.asarray(p_w, float)
        dist = float(np.linalg.norm(p1 - p0))
        ang = _rot_angle(R0, R1)
        T = max(dist / speed, ang / ang_speed, min_t)
        n = max(int(T / CTRL_DT), 1)
        path = self._plan_line(p0, R0, p1, R1, n, linear, self.q_cmd.copy())
        if path is None:
            q_end, ok, ep, _ = self._solve(p1, R1, sym=False, q_hint=self.q_cmd)
            return SkillResult(False, "unreachable" if not ok else "no_path",
                               {"pos_err": ep, "target": p1.tolist()})
        path, _ = self._retime(self.q_cmd, path, [0] * len(path), guarded_seg=0 if guard is not None else None)
        self._guarded = guard is not None
        try:
            for k, q in enumerate(path, start=1):
                self._cmd(q)
                if guard is not None and guard():
                    self._hold()
                    return SkillResult(True, "ok", {"stopped": True, "fraction": k / len(path)})
        finally:
            self._guarded = False
        self._settle()
        return SkillResult(True, "ok", {"stopped": False, "fraction": 1.0})

    def _ik_candidates(self, p_w, R_w, sym=True, max_n=4):
        """Distinct IK solutions for a world pose (closest to the current command first)."""
        out = []
        for Rw in ([R_w, _flip_z(R_w)] if sym else [R_w]):
            pb, Rb = self._to_base(p_w, Rw)
            az = math.atan2(pb[1], pb[0])
            for seed in self._seeds(pb, self.q_cmd):
                q, ep, er = self.kin.ik(pb, Rb, seed, iters=200)
                if (ep < 2e-3 and er < 0.02 and self._links_clear(q)
                        and not any(np.abs(q - o[1]).max() < 0.05 for o in out)):
                    out.append((self.kin.posture_cost(q, az) + 0.05 * float(np.abs(q - self.q_cmd).sum()), q))
        out.sort(key=lambda cq: cq[0])
        return [q for _, q in out[:max_n]]

    # ================================================================ retreat out of a tilted look
    def _retreat_now(self):
        """tilt-retreat-1 ("retreat"): when a tilted look's retreat is due
        and a skill is running, plan and make the hand-down transit out of the tilted pose first. Called where a skill's
        first arm motion starts: _lift_to (every skill whose motion starts with _approach: move_to transit, the looks,
        pick, open_drawer, close_drawer, scrub, ...; the hook of the verified variant) and go_home (go_home, look_pose,
        calibrate, whose first motion is a straight lift or a joint move; not in the variant, see the module note). The
        waypoint: RETREAT_IN_FRONT_M from the look's aim horizontally toward the hand (the current TCP), at worktop +
        RETREAT_ABOVE_WORKTOP_M; move_to(waypoint, orient="down", transit=True), a planned transit, nested in the skill.
        As in the verified variant, that nested move_to leaves its waypoint as the target zone (_zone) of the rest of
        the skill (the module ring is then checked near the skill's own target: more conservative there; declared).
        Outcomes (one event each, data["retreat"] of the top-level result, _retreat_close):
          ok: the skill goes on from the waypoint; planning_failed (no_path / unreachable, returned or raised by a joint
          move): fallback RETREAT_FALLBACK, the skill
            goes on from where the arm is (today's behaviour);
          failed (any other non-contact result): the same fallback; aborted (a collision reflex, a blocked arm, any
          other stop raised inside it): the skill stops on it, its result is
            that abort (a contact is reported as the skill's own contact incident)."""
        pend = getattr(self, "_retreat_pending", None)
        if pend is None or self._depth < 1:
            return
        self._retreat_pending = None
        tgt = [float(x) for x in pend["aim"]]
        p0 = self.tcp_pose(self.q_cmd)[0]
        u = np.array([p0[0] - tgt[0], p0[1] - tgt[1]])
        u = u / max(float(np.linalg.norm(u)), 1e-9)
        way = np.array([tgt[0] + RETREAT_IN_FRONT_M * u[0], tgt[1] + RETREAT_IN_FRONT_M * u[1],
                        self.worktop_z + RETREAT_ABOVE_WORKTOP_M])
        ev = {"version": TILT_RETREAT_VERSION, "aim": [round(x, 5) for x in tgt],
              "from": [round(float(x), 4) for x in p0], "to": [round(float(x), 4) for x in way],
              "t_s": self._clock_s()}
        self._retreat_events.append(ev)
        try:
            rr = self.move_to(way, orient="down", transit=True)
        except _Abort as e:
            ev.update(ok=False, reason=str(e.reason))
            if e.reason in RETREAT_PLANNING_REASONS:
                ev.update(outcome="planning_failed", fallback=RETREAT_FALLBACK)
                return
            ev["outcome"] = "aborted"
            fn = e.data.get("force_n") if isinstance(e.data, dict) else None
            if isinstance(fn, (int, float)) and not isinstance(fn, bool):
                ev["force_n"] = round(float(fn), 1)
            raise
        ev.update(ok=bool(rr.ok), reason=str(rr.reason))
        if rr.ok:
            ev["outcome"] = "ok"
        else:
            ev.update(outcome="planning_failed" if rr.reason in RETREAT_PLANNING_REASONS else "failed",
                      fallback=RETREAT_FALLBACK)

    def _retreat_close(self, r, pending0, n0):
        """tilt-retreat-1, at the end of a top-level skill: a retreat that was due when the skill started and is still
        due although the skill moved the arm (a straight move_to; no retreat point reached) is dropped, so a later skill
        never retreats from a pose that is no longer the tilted one (event outcome "dropped"). The skill's retreat
        events, with its name and result, go into r.data["retreat"] (a list of plain dicts; neither the facade log nor
        the host's program result keeps it: records only)."""
        events = list(getattr(self, "_retreat_events", None) or [])
        if (pending0 is not None and getattr(self, "_retreat_pending", None) is pending0
                and self.n_cmd != n0):
            self._retreat_pending = None
            events.append({"version": TILT_RETREAT_VERSION, "aim": [round(float(x), 5) for x in pending0["aim"]],
                           "outcome": "dropped", "t_s": self._clock_s()})
        self._retreat_events = []
        if not events:
            return
        name = getattr(self, "_retreat_skill", None)
        for ev in events:
            ev.update(skill=None if name is None else str(name), skill_ok=bool(r.ok), skill_reason=str(r.reason))
        r.data["retreat"] = events

    def _clock_s(self):
        try:
            return round(float(self.sim.t), 2)
        except Exception:  # noqa: BLE001 (a sim double without a clock: records only)
            return None

    # ================================================================ look-exit-1 (records: data["look_exit"])
    def _look_exit_now(self, where):
        """look-exit-1 (lookexit_tool.py arm lookexit, _lx_now, ported): when the exit armed
        by look_pose is due and a skill is running, leave LOOK by the exact reverse of look_pose's direct joint segment,
        _joint_move_raw(HOME, LOOK_SEGMENT_SPEED, LOOK_SEGMENT_MIN_T) (the same function, speed and smoothstep), before
        the skill's first arm motion: no route search, no clearance decision. Called at _lift_to (every _approach),
        _line (straight moves), _joint_move, _joint_move_raw and at the start of go_home (look_pose, calibrate). The due
        state is cleared before the segment, so the segment's own primitives never recurse. Outcomes (one event each in
        the top-level result's data["look_exit"]):
          ok: the segment ran; the skill goes on from HOME;
          not_at_look / forward_not_direct (guards): the command is not LOOK, or look_pose's last move was not the
            direct segment HOME -> LOOK: no command is sent, the skill goes on as before;
          aborted: a collision reflex or a blocked arm in the segment: the skill stops, its result is that abort."""
        pend = getattr(self, "_look_exit_pending", None)
        if pend is None or self._depth < 1:
            return
        self._look_exit_pending = None
        q = np.asarray(self.q_cmd, float).copy()
        ev = {"version": LOOK_EXIT_VERSION, "where": str(where), "skill": getattr(self, "_retreat_skill", None),
              "armed_t_s": pend.get("t_s"), "t0_s": self._clock_s(), "q_dev_from_look": float(np.abs(q - LOOK).max()),
              "held": None if self.held is None else self.held.get("kind")}
        self._v473c_events("_look_exit_events").append(ev)
        if ev["q_dev_from_look"] > 1e-9:
            ev["outcome"] = "not_at_look"
            return
        if not pend.get("direct"):
            ev["outcome"] = "forward_not_direct"
            return
        n0 = self.n_cmd
        try:
            self._joint_move_raw(HOME.copy(), LOOK_SEGMENT_SPEED, LOOK_SEGMENT_MIN_T)  # exact reverse of look_pose's segment
        except _Abort as e:
            ev.update(outcome="aborted", reason=str(e.reason), t1_s=self._clock_s(), n_cmd=self.n_cmd - n0)
            data = e.data if isinstance(e.data, dict) else {}
            fn = data.get("force_n")
            if isinstance(fn, (int, float)) and not isinstance(fn, bool):
                ev["force_n"] = round(float(fn), 1)
            if data.get("where") is not None:
                ev["where_contact"] = str(data.get("where"))
            raise  # the contact is the skill's own result
        ev.update(outcome="ok", t1_s=self._clock_s(), n_cmd=self.n_cmd - n0,
                  q_end_dev_from_home=float(np.abs(np.asarray(self.q_cmd, float) - HOME).max()))

    def _look_exit_missed(self, caller):
        """look-exit-1 (records only): an arm command sent while the exit is still due (no hooked primitive was passed,
        none known); recorded as missed and cleared, the command goes on (the prototype's detector)."""
        self._look_exit_pending = None
        self._v473c_events("_look_exit_events").append(
            {"version": LOOK_EXIT_VERSION, "outcome": "missed", "t_s": self._clock_s(),
             "skill": getattr(self, "_retreat_skill", None), "caller": str(caller)})

    def _v473c_events(self, attr):
        """The current top-level skill's event list `attr` (_look_exit_events or _pinch_rest_events), created if absent
        (records only)."""
        events = getattr(self, attr, None)
        if not isinstance(events, list):
            events = []
            setattr(self, attr, events)
        return events

    def _v473c_close(self, r):
        """At the end of a top-level skill: its look-exit-1 and pinch-rest events, with its name and result, go into
        r.data["look_exit"] and r.data["pinch_rest"] when there are any (lists of plain dicts; neither the facade log
        nor the host's program result keeps them: records only)."""
        name = getattr(self, "_retreat_skill", None)
        for key, attr in (("look_exit", "_look_exit_events"), ("pinch_rest", "_pinch_rest_events")):
            events = list(getattr(self, attr, None) or [])
            setattr(self, attr, [])
            if not events:
                continue
            for ev in events:
                ev.update(skill=None if name is None else str(name), skill_ok=bool(r.ok), skill_reason=str(r.reason))
            r.data[key] = events

    def _lift_to(self, z):
        """Up from the current pose to height z: straight up, else up and toward the base (a
        hand reaching far out cannot always rise in place), else as high as reachable. tilt-retreat-1: a due retreat
        out of a tilted look runs first (_retreat_now). look-exit-1: before it, a due exit out of LOOK
        (_look_exit_now)."""
        if self._depth >= 1:
            self._look_exit_now("lift_to")
        if getattr(self, "_retreat_pending", None) is not None and self._depth >= 1:
            self._retreat_now()
        p_now, R_now = self.tcp_pose(self.q_cmd)
        if p_now[2] >= z - 0.01:
            return SkillResult(True, "ok", {})
        inward = -self._radial(p_now)
        # The hand may also turn about the vertical while it rises (a held object stays upright).
        for zz in (z, z - 0.06, z - 0.12):
            if zz <= p_now[2]:
                break
            for back in (0.0, 0.05, 0.10):
                for yaw in (0.0, 0.6, -0.6, 1.2, -1.2):
                    R = rot_about([0, 0, 1], yaw) @ R_now if yaw else R_now
                    r = self._line(np.array([p_now[0], p_now[1], zz]) + back * inward, R, speed=0.25)
                    if r.ok:
                        return r
        return SkillResult(False, "lift_failed", {})

    def _path_targets(self, pts, Rs, speeds, linear_last):
        """Per-tick (position, rotation) targets in the base frame along a polyline. Rs[i] is the
        orientation at pts[i]; orientation is interpolated along each segment."""
        out, seg_index = [], []
        for i in range(len(pts) - 1):
            p0, p1 = np.asarray(pts[i], float), np.asarray(pts[i + 1], float)
            ang = _rot_angle(Rs[i], Rs[i + 1])
            T = max(np.linalg.norm(p1 - p0) / speeds[i], ang / 1.2, 0.06)
            n = max(int(T / CTRL_DT), 1)
            lin = linear_last and i == len(pts) - 2
            qa0, qa1 = quat_from_mat(Rs[i]), quat_from_mat(Rs[i + 1])
            for k in range(1, n + 1):
                sv = k / n if lin else _smooth(k / n)
                Rk = mat_from_quat(_slerp(qa0, qa1, sv))
                out.append(self._to_base(p0 + sv * (p1 - p0), Rk))
                seg_index.append(i)
        return out, seg_index

    def _plan_backward(self, pts, Rs, speeds, linear_last, max_ends=4, exit_check=None):
        """Plan the whole polyline backward from a good final configuration. Returns
        (q_at_pts0, path, seg_index) or None."""
        targets, seg_index = self._path_targets(pts, Rs, speeds, linear_last)
        start = self._to_base(pts[0], Rs[0])
        for q_end in self._ik_candidates(pts[-1], Rs[-1], sym=False, max_n=max_ends):
            if exit_check is not None and not exit_check(q_end):
                continue
            q, ok, rev = q_end.copy(), True, [q_end.copy()]
            for pb, Rb in list(reversed(targets[:-1])) + [start]:
                qn, ep, er = self.kin.ik(pb, Rb, q, iters=60, posture=TRACK_POSTURE)
                if ep > 2e-3 or er > 0.02 or np.abs(qn - q).max() > 0.12 or not self._links_clear(qn):
                    ok = False
                    break
                rev.append(qn)
                q = qn
            if ok:
                path = list(reversed(rev))  # path[0] at pts[0], path[k] = targets[k-1]
                return path[0], path[1:], seg_index
        return None

    def _retime(self, q0, path, seg_index, guarded_seg=None):
        """Insert intermediate commands wherever a joint would move faster than its limit (near
        wrist singularities a small hand motion can need fast joints). In a guarded segment
        (contact expected) every joint stays under QD_GUARDED: the wrist servos lag about
        0.1 s x speed, and a lagging arm overshoots when the contact comes."""
        lim = (self._held_qd_limit() if self.held else QD_LIMIT) * CTRL_DT
        lim_g = np.minimum(lim, QD_GUARDED * CTRL_DT)
        out, idx, prev = [], [], np.asarray(q0, float)
        for q, si in zip(path, seg_index):
            lm = lim_g if (guarded_seg is not None and si == guarded_seg) else lim
            n = int(math.ceil(float((np.abs(q - prev) / lm).max())))
            for j in range(1, max(n, 1)):
                out.append(prev + (q - prev) * j / n)
                idx.append(si)
            out.append(q)
            idx.append(si)
            prev = q
        return out, idx

    def _plan_forward(self, pts, Rs, speeds, linear_last):
        """Track the polyline forward from the current joint configuration (no reconfiguration
        at the start). Returns (path, seg_index) or None."""
        targets, seg_index = self._path_targets(pts, Rs, speeds, linear_last)
        q, path = self.q_cmd.copy(), []
        for pb, Rb in targets:
            qn, ep, er = self.kin.ik(pb, Rb, q, iters=60, posture=TRACK_POSTURE)
            if ep < 2e-3 and er < 0.02:
                qu, epu, eru = self.kin.untwist(qn, pb, Rb)
                if epu < 2e-3 and eru < 0.02:
                    qn = qu
            if ep > 2e-3 or er > 0.02 or np.abs(qn - q).max() > 0.12 or not self._links_clear(qn):
                return None
            path.append(qn)
            q = qn
        return path, seg_index

    def _route(self, p_from, p_to):
        """Waypoints from p_from to p_to (both at transit height): a turn around the base of more
        than 20 deg goes along a circle of radius SWING_R, over free space."""
        b = self.belief.t
        # Azimuths in the base frame, not wrapped: the arm never turns through the back (joint 1
        # stops at +-166 deg and the camera mast stands there).
        f0, f1 = self.belief.world_to_base(p_from), self.belief.world_to_base(p_to)
        da = math.atan2(f1[1], f1[0]) - math.atan2(f0[1], f0[0])
        a0 = math.atan2(p_from[1] - b[1], p_from[0] - b[0])
        if abs(da) < math.radians(20):
            return [], 0.0
        z = max(p_from[2], p_to[2])
        r0 = float(np.hypot(p_from[0] - b[0], p_from[1] - b[1]))
        r1 = float(np.hypot(p_to[0] - b[0], p_to[1] - b[1]))
        rs = min(self.SWING_R, max(r0, r1))  # near-base moves (calibration touches) stay near the base
        n = max(2, int(abs(da) / math.radians(12)))
        vias = [np.array([b[0] + rs * math.cos(a0 + da * k / n), b[1] + rs * math.sin(a0 + da * k / n), z])
                for k in range(0, n + 1)]
        return vias, da

    def _approach(self, pts, R, speeds=None, guard=None, linear_last=True, sym=True, lift_z=None,
                  exit_check=None):
        """Lift, travel (around the base if needed) and follow the polyline pts at orientation R.
        The whole motion after the lift is planned backward from a good final configuration, so
        the last, delicate segment is always continuous. The guard (if any) watches the last
        segment. With sym, the gripper may be turned 180 deg about its axis. exit_check(q_end):
        a plan is kept only if the skill's next motion is possible from its final configuration
        (e.g. the lift after a grasp), so the skill fails before moving rather than after."""
        pts = [np.asarray(p, float) for p in pts]
        speeds = speeds or [0.25] * (len(pts) - 2) + [0.08]
        lift_z = pts[0][2] if lift_z is None else lift_z
        if self.held is not None and "tcp_above_bottom" in self.held:
            # Carry height: the held object's bottom above the tallest module of the site plan.
            lift_z = max(lift_z, self.worktop_z + ENVELOPE_H + 0.02 + self.held["tcp_above_bottom"])
        self._lift_to(lift_z)
        p_now, R_now = self.tcp_pose(self.q_cmd)
        best = None
        for drop in (0.0, 0.06, 0.12):
            pp = list(pts)
            if drop and len(pp) > 2:
                pp[0] = np.array([pp[0][0], pp[0][1], max(pp[1][2], pp[0][2] - drop)])
            vias, da = self._route(p_now, pp[0])
            # Turn the hand at transit height, above the target, then go down with the final
            # orientation (a hand still turning while it descends sweeps its 21 cm palm low).
            z_tr = max(p_now[2], pp[0][2])
            mid = np.array([pp[0][0], pp[0][1], z_tr])
            mids = [mid] if (self.TURN_HIGH and z_tr - pp[0][2] > 0.02 and np.linalg.norm(p_now - mid) > 0.005) else []
            for Rv in ([R, _flip_z(R)] if sym else [R]):
                travel = [p_now] + vias + mids + [pp[0]] if np.linalg.norm(p_now - pp[0]) > 0.005 else [pp[0]]
                full = travel + pp[1:]
                # Orientation along the travel: the current one turned with the base (around the
                # vertical), then Rv from pts[0] on.
                nt = len(travel)
                Rs = [R_now] if nt > 1 else []
                for k in range(len(vias)):
                    Rs.append(rot_about([0, 0, 1], da * k / max(len(vias) - 1, 1)) @ R_now)
                Rs += [Rv] * (len(full) - len(Rs))
                sp = [0.2 if self.held else 0.3] * (nt - 1) + speeds
                fwd = self._plan_forward(full, Rs, sp, linear_last) if nt > 1 else None
                if fwd is not None and (exit_check is None or exit_check(fwd[0][-1])):
                    best = (0.0, self.q_cmd.copy(), fwd[0], fwd[1], len(full) - 2, [], nt - 1)
                    break
                plan = self._plan_backward(full, Rs, sp, linear_last, exit_check=exit_check)
                if plan is None:
                    continue
                q0, path, seg_index = plan
                pre = [] if np.abs(q0 - self.q_cmd).max() <= 1e-3 else self._joint_route(self.q_cmd, q0)
                if pre is None:
                    continue
                reconf = float(np.abs(q0 - self.q_cmd).max())
                if best is None or reconf < best[0]:
                    best = (reconf, q0, path, seg_index, len(full) - 2, pre, nt - 1)
            if best is not None:
                break
        if best is None:
            # Fallback: plan only the approach (pts) backward, reach its first configuration by
            # joint moves through via-points on the turning circle. The planned part starts a
            # little above pts[0] when possible (12 or 6 cm), so the joint-space swing ends higher
            # and the hand goes down already turned.
            for drop in (0.0, 0.06, 0.12):
                base_pp = list(pts)
                if drop and len(base_pp) > 2:
                    base_pp[0] = np.array([base_pp[0][0], base_pp[0][1], max(base_pp[1][2], base_pp[0][2] - drop)])
                for up in (0.12, 0.06, 0.0):
                    if up:
                        pp = [base_pp[0] + [0, 0, up]] + base_pp
                        sp_f = [0.15] + list(speeds)
                    else:
                        pp, sp_f = base_pp, list(speeds)
                    for Rv in ([R, _flip_z(R)] if sym else [R]):
                        if len(pp) < 2:
                            break
                        plan = self._plan_backward(pp, [Rv] * len(pp), sp_f, linear_last, exit_check=exit_check)
                        if plan is None:
                            continue
                        q0, path, seg_index = plan
                        route = self._swing_route(self.q_cmd, q0)
                        if route is None:
                            continue
                        best = (0.0, q0, path, seg_index, len(pp) - 2, route, 0)
                        break
                    if best is not None:
                        break
                if best is not None:
                    break
        if best is None:
            q, ok, ep, _ = self._solve(pts[-1], R, sym=sym)
            return SkillResult(False, "unreachable" if not ok else "no_path",
                               {"pos_err": ep, "target": pts[-1].tolist()})
        _, q0, path, seg_index, last_seg, pre, first_seg = best
        for q in pre:
            # Same hand pose, other joint configuration (or via-points): slowly if something is held.
            self._joint_move_raw(q, speed=0.5 if self.held else 1.0, min_t=0.15)
        path, seg_index = self._retime(q0, path, seg_index, guarded_seg=last_seg if guard is not None else None)
        # The way in, known to be free for the hand: a grasp may leave by it (see _retrace).
        self._way_in = {"q0": q0.copy(), "path": path, "seg": seg_index, "first": first_seg}
        try:
            for k, q in enumerate(path):
                self._guarded = guard is not None and seg_index[k] == last_seg
                self._cmd(q)
                if self._guarded and guard():
                    self._hold()
                    return SkillResult(True, "ok", {"stopped": True})
        finally:
            self._guarded = False
        self._settle()
        return SkillResult(True, "ok", {"stopped": False})

    def _retrace(self, way, z_goal):
        """Leave by the way in, backward: back to the configuration at the start of the approach
        polyline (a few cm from the grasp), then back along the travel until the hand is at
        z_goal. Used when a hand reaching far out cannot rise in place with what it holds."""
        k = next((i for i, si in enumerate(way["seg"]) if si >= way["first"]), len(way["seg"]))
        q_entry = way["path"][k - 1] if k > 0 else way["q0"]
        self._joint_move(q_entry, speed=0.5, min_t=0.6)
        back = list(reversed(way["path"][:max(k - 1, 0)])) + [way["q0"]]
        back, _ = self._retime(self.q_cmd, back, [0] * len(back))
        for q in back:
            self._cmd(q)
            if self.tcp_pose(q)[0][2] >= z_goal:
                break
        self._settle()

    # ================================================================ sensing helpers
    def _hand_force(self) -> np.ndarray:
        """Contact force on hand and fingers from the environment, measured in the robot's own
        base frame and turned into believed world axes."""
        return self.belief.rot_base_to_world(self.sim.env_force_base(self.sim.hand_bodies))

    def _guard_dir(self, direction, thresh, tare=None, consecutive=2):
        """Stop when the reaction force against `direction` exceeds `thresh` newtons."""
        direction = np.asarray(direction, float) / np.linalg.norm(direction)
        tare = np.zeros(3) if tare is None else tare
        state = {"n": 0, "peak": 0.0}

        def g():
            f = self._hand_force() - tare
            push = float(-(f @ direction))
            state["peak"] = max(state["peak"], push)
            state["n"] = state["n"] + 1 if push > thresh else 0
            return state["n"] >= consecutive
        g.state = state
        return g

    def _tare(self, n=5):
        acc = np.zeros(3)
        for _ in range(n):
            self._wait()
            acc += self._hand_force()
        return acc / n

    # ================================================================ gripper
    def _grip(self, width_target, wait=0.6):
        u = float(np.clip(width_target / MAX_OPEN, 0, 1) * 255.0)
        self.grip_cmd = u
        self.sim.set_gripper(u)
        last = self.sim.gripper_width()
        for k in range(int(wait / CTRL_DT)):
            self.sim.tick()
            w = self.sim.gripper_width()
            if k > 8 and abs(w - last) < 2e-5:
                break
            last = w
        return self.sim.gripper_width()

    @_skill
    def open_gripper(self, width: float = MAX_OPEN) -> SkillResult:
        w = self._grip(_num("width", width, 0.0, MAX_OPEN))
        self._reconcile_held()  # robot-held-1: a full release ends robot.held
        return SkillResult(True, "ok", {"width": w})

    @_skill
    def close_gripper(self, width: float = 0.0) -> SkillResult:
        """Close (to `width`: 0 = full force; a soft object is gripped at a set width). It
        never sets robot.held (a hold by hand is the host's MANUAL_HOLD); under robot-held-1 a command that lets
        the held object go (a width above the grip) ends robot.held, like open_gripper."""
        w = self._grip(_num("width", width, 0.0, MAX_OPEN), wait=0.8)
        lt, rt = self.sim.finger_touch()
        grasped = bool(w > 0.004 and lt and rt)
        self._reconcile_held()
        return SkillResult(True, "ok", {"grasped": grasped, "width": w})

    def holding(self) -> bool:
        """Whether the gripper still holds something (finger width and finger contacts)."""
        if self.held is None:
            return False
        lt, rt = self.sim.finger_touch()
        # Both fingers on something (a soft sponge squeezed thin), or one finger on it with the
        # fingers apart (a rigid object pushed against one finger).
        return bool((lt and rt) or (self.sim.gripper_width() > 0.004 and (lt or rt)))

    def _reconcile_held(self) -> bool:
        """Contract robot-held-1: end robot.held when
        the robot's own grip reading holding() (finger switches and width) reads False. Before, only pick set it and
        only a skill that checks the grip cleared it, so after a full open_gripper the gates below still answered
        hand_busy / already_holding while holding() read False. Called (a) after open_gripper and close_gripper
        settle and (b) at the entry of the gestures gated on robot.held (pick: already_holding; open_drawer,
        close_drawer, press: hand_busy), after their argument checks: right after a 6.2 cm opening both switches can
        still touch and the sponge falls during the next motion, so the entry check catches a late fall. Never
        decided from the commanded width: a one-finger hang at 8 cm (recorded twice) still reads holding() True
        and keeps held. Not called by place, turn_over, spray and scrub, which keep their own check (a silent loss
        still answers "dropped", a grip-loss code of the host, unless a gated gesture's entry ended the hold first:
        then "not_holding*"). Reads only (finger switches, width): no tick, no RNG, so a chain without a stale hold
        is unchanged bit for bit. held-0: never clears. Returns whether held was cleared."""
        if self.held_contract != ROBOT_HELD_CONTRACT or self.held is None or self.holding():
            return False
        self.trace.append(f"held_cleared {self.held.get('kind')}: grip not felt")
        self.held = None
        return True

    # ================================================================ basic skills
    @_skill
    def go_home(self) -> SkillResult:
        self._look_exit_now("go_home")  # look-exit-1: a due exit out of LOOK first (also look_pose, calibrate)
        self._retreat_now()  # tilt-retreat-1: a due retreat out of a tilted look first (also look_pose, calibrate)
        p, R = self.tcp_pose(self.q_cmd)
        if p[2] < self.worktop_z + 0.30:
            r = self._line([p[0], p[1], self.worktop_z + 0.30], R, speed=0.25)
            if not r.ok:
                self.trace.append(f"go_home lift {r.reason}")
        self._joint_move(HOME, speed=1.2)  # aborts with "no_path" if no clear joint route
        return SkillResult(True, "ok", {})

    @_skill
    def look_pose(self) -> SkillResult:
        """look-exit-1: ended at LOOK, it arms the exit (the next arm motion leaves LOOK by the reverse of the
        joint segment it took, _look_exit_now), recording whether that segment was the direct one from HOME."""
        r = self.go_home()
        if not r.ok:
            return r
        self._joint_move(LOOK, speed=LOOK_SEGMENT_SPEED)
        lr = getattr(self, "_last_joint_route", None) or {}
        q0, route = lr.get("q0"), lr.get("route") or []
        direct = bool(q0 is not None and float(np.abs(np.asarray(q0, float) - HOME).max()) < 1e-4
                      and len(route) == 1 and float(np.abs(np.asarray(route[0], float) - LOOK).max()) < 1e-4)
        self._look_exit_pending = {"t_s": self._clock_s(), "direct": direct,
                                   "q_dev_from_look": float(np.abs(np.asarray(self.q_cmd, float) - LOOK).max())}
        return SkillResult(True, "ok", {})

    @_skill
    def move_to(self, pos, orient="down", speed: float = 0.25, transit: bool = False, retreat_aim=None) -> SkillResult:
        """retreat_aim (keyword, default None): the world aim of a tilted look made by this move;
        when the move ends ok, the next skill that moves the arm first retreats out of the tilted pose
        (_retreat_now)."""
        pos = _vec("pos", pos)
        speed = _num("speed", speed, 0.01, 0.5)
        aim = None if retreat_aim is None else _vec("retreat_aim", retreat_aim)
        self._zone = pos[:2].copy()
        if isinstance(orient, str):
            if orient != "down":
                raise _Abort("bad_arg", {"error": "orient must be 'down', a yaw angle (rad) or a 3x3 rotation matrix"})
            r = self._radial(pos)
            R = hand_down(math.atan2(r[1], r[0]) + math.pi / 2)
        elif np.ndim(orient) == 0:
            R = hand_down(_num("orient", orient, -2 * math.pi, 2 * math.pi))
        else:
            try:
                R = np.asarray(orient, dtype=float)
            except (TypeError, ValueError):
                R = np.zeros(0)
            if (R.shape != (3, 3) or not np.all(np.isfinite(R))
                    or np.abs(R.T @ R - np.eye(3)).max() > 1e-3 or np.linalg.det(R) < 0):
                raise _Abort("bad_arg", {"error": "orient must be 'down', a yaw angle (rad) or a 3x3 rotation matrix"})
        if transit:
            above = np.array([pos[0], pos[1], max(self.worktop_z + 0.40, pos[2])])
            res = self._approach([above, pos], R, [0.25], linear_last=False)
        else:
            res = self._line(pos, R, speed=speed, sym=True)
        p, _ = self.tcp_pose()
        res.data["believed_tcp"] = p.tolist()
        if aim is not None and res.ok:  # tilt-retreat-1: a tilted look ended here: its retreat is due
            self._retreat_pending = {"aim": [float(x) for x in aim]}
        return res

    # ================================================================ calibration
    @_skill
    def calibrate(self, edges: dict | None = None) -> SkillResult:
        """Measure the mount: tilt (IMU), height (touch down), yaw and x, y (edge touches).

        edges: site plan of the worktop inner edges, {"front_x": 0.29, "side_y": 0.29}
        (the U opening). Defaults are the plan values. They are kept for the clearance checks.
"""
        if edges is None:
            edges = dict(self.PLAN_EDGES)
        if not isinstance(edges, dict) or set(edges) != {"front_x", "side_y"}:
            raise _Abort("bad_arg", {"error": "edges must be {'front_x': m, 'side_y': m}"})
        edges = {k: _num(k, v, 0.15, 0.50) for k, v in edges.items()}
        self.edges = dict(edges)
        data = {}
        r = self.go_home()
        if not r.ok:
            return r
        self.open_gripper(0.0)
        # 1. Tilt from the accelerometer, 1 s average at 50 Hz while static.
        self._wait(10)
        acc = []
        for _ in range(50):
            self._wait()
            acc.append(self.sim.imu())
        a = np.mean(acc, axis=0)
        roll, pitch = tilt_from_accel(a)
        b = self.belief
        b.roll_deg, b.pitch_deg = roll, pitch
        data["imu_mean"] = a.tolist()
        R_rp = rpy_to_mat(roll, pitch, 0.0)

        # 2. Height: guarded touch of the worktop at a free spot in front of the base.
        down = hand_down(math.pi / 2)
        # Start 10 cm above the nominal worktop: the base may be 7 cm lower or 13 cm higher
        # than believed.
        spot = np.array([0.40, 0.0, self.worktop_z + 0.10])
        tare = self._tare()
        g = self._guard_dir([0, 0, -1], 4.0, tare, consecutive=1)  # hard surface: stop at once
        # The probe's end point lies below the believed worktop on purpose (the contact stops it
        # earlier), so the link-above-worktop rule is relaxed for this one motion.
        self.link_margin = -0.25
        try:
            r = self._approach([spot + [0, 0, 0.20], spot, spot - [0, 0, 0.26]], down, [0.25, 0.04], guard=g)
        finally:
            self.link_margin = 0.06
        if not r.ok or not r.data.get("stopped"):
            self.trace.append(f"height probe {r.reason} {r.data}")
            return SkillResult(False, "calib_height_no_contact", data)
        q_touch = self.sim.arm_q()
        pb, Rb = self.kin.fk(q_touch)
        tip_b = pb + Rb[:, 2] * TIP_BELOW_TCP
        h = -float((R_rp @ tip_b)[2])  # base origin above the touched surface, along gravity
        b.z = self.worktop_z + h
        data["height_above_worktop"] = h
        self._line(self.tcp_pose(self.q_cmd)[0] + [0, 0, 0.05], None, speed=0.1)

        # 3. Yaw and x, y: horizontal guarded touches of the worktop inner edges.
        # Front edge first (two touches, believed y = +-0.17 so that a 10 deg yaw error or a
        # 3 cm offset still lands on the front edge, away from the inner corners), then yaw and
        # x; then one touch of the side edge with the corrected belief, then y. Touch points stay
        # at least 0.27 m from the base axis (closer, a hand pointing down cannot reach down to
        # the worktop level).
        pts = []
        return self._calibrate_edges(edges, data, R_rp, pts)

    def _calibrate_edges(self, edges, data, R_rp, pts):
        b = self.belief
        for sign in (1.0, -1.0):
            c = None
            # Fallback spots and a shorter probe when the arm cannot reach the first one (a
            # very high or tilted base).
            for y_abs, x_start, reach in ((0.17, 0.07, 0.16), (0.14, 0.07, 0.16), (0.20, 0.06, 0.13)):
                c = self._edge_touch(np.array([edges["front_x"] - x_start, sign * y_abs]),
                                     np.array([1.0, 0.0]), reach=reach)
                if c is not None:
                    break
            if c is None:
                return SkillResult(False, "calib_edge_no_contact", data)
            pts.append(c)
        q1, q2 = (R_rp @ p for p in pts)  # gravity-aligned base frame, still rotated by yaw
        yaw = math.atan2(q1[0] - q2[0], q1[1] - q2[1])
        Rz = rpy_to_mat(0, 0, math.degrees(yaw))
        w1 = Rz @ q1
        # The fingertip's leading face touched the edge: add the tip half thickness.
        b.yaw_deg = math.degrees(yaw)
        b.x = edges["front_x"] - (w1[0] + TIP_HALF_X)
        # Yaw is known now: the side edge is within the x, y offset (+-3 cm) of the plan, so a
        # short probe will do; try a few spots along the edge if the arm cannot reach one.
        c = None
        for x0 in (0.20, 0.14, 0.25):
            c = self._edge_touch(np.array([x0, edges["side_y"] - 0.06]), np.array([0.0, 1.0]), reach=0.11)
            if c is not None:
                break
        if c is None:
            return SkillResult(False, "calib_edge_no_contact", data)
        pts.append(c)
        w3 = Rz @ (R_rp @ c)
        b.y = edges["side_y"] - (w3[1] + TIP_HALF_X - TIP_SIDE_EXTRA)
        b.calibrated = True
        data["touch_points_base"] = [p.tolist() for p in pts]
        r = self.go_home()
        if not r.ok:
            return SkillResult(False, r.reason, {**data, "stage": "go_home after calibration"})
        self.open_gripper()
        data["belief"] = b.to_dict()
        return SkillResult(True, "ok", data)

    def _edge_touch(self, start_xy, direction, reach=0.16):
        """Touch a vertical worktop edge: fingertip 1.2 cm below the surface, move along `direction`.

        Uses the current belief (yaw may still be nominal). Returns the TCP contact point in the
        base frame (position of the TCP axis at the fingertip height), or None.
"""
        z = self.worktop_z - 0.012 + TIP_BELOW_TCP
        d3 = np.array([direction[0], direction[1], 0.0])
        # Hand down, finger axis parallel to the edge: the fingertip's thin side leads.
        R = hand_down(math.atan2(direction[1], direction[0]) + math.pi / 2)
        low = np.array([start_xy[0], start_xy[1], z])
        tare = self._tare()
        g = self._guard_dir(d3, 3.0, tare, consecutive=1)  # hard edge: stop at once
        # One motion planned backward from the end of the probe: down in the U opening, then
        # slowly toward the edge until the fingertip feels it.
        r = self._approach([low + [0, 0, 0.15], low, low + reach * d3], R, [0.10, 0.02], guard=g,
                           linear_last=True, lift_z=self.worktop_z + 0.20)
        if not r.ok or not r.data.get("stopped"):
            self.trace.append(f"edge probe {r.reason} {r.data}")
            return None
        R = self.tcp_pose(self.q_cmd)[1]
        pb, Rb = self.kin.fk(self.sim.arm_q())
        tip_b = pb + Rb[:, 2] * TIP_BELOW_TCP
        # Back off and lift.
        self._line(self.tcp_pose(self.q_cmd)[0] - 0.03 * d3, R, speed=0.05)
        self._line(self.tcp_pose(self.q_cmd)[0] + [0, 0, 0.05], R, speed=0.1)
        return tip_b

    # ================================================================ pick and place
    def _lift_check(self, dzs, side_back=None):
        """exit_check for a grasp: the straight lift (and, for a side grasp, nothing else) must be
        plannable from the grasp configuration, so pick fails before moving if it is not."""
        def check(q_end):
            p_e, R_e = self.tcp_pose(q_end)
            for dz in dzs:
                n = max(int((dz / 0.12) / CTRL_DT), 1)
                if self._plan_line(p_e, R_e, p_e + [0, 0, dz], R_e, n, False, q_end) is not None:
                    return True
            return False
        return check

    @_skill
    def pick(self, pos, kind: str, grasp: str = "top", yaw: float | None = None,
             approach=None, obj_quat=None, upside_down: bool = False) -> SkillResult:
        if kind not in CATALOG:
            raise _Abort("bad_arg", {"error": f"kind must be one of {sorted(CATALOG)}, got {kind!r}"})
        if grasp not in ("top", "side"):
            raise _Abort("bad_arg", {"error": f"grasp must be 'top' or 'side', got {grasp!r}"})
        pos = _vec("pos", pos)
        yaw = _num("yaw", yaw, -2 * math.pi, 2 * math.pi, allow_none=True)
        approach = None if approach is None else _hdir("approach", approach)
        obj_quat = _quat("obj_quat", obj_quat)
        self._reconcile_held()  # robot-held-1: a hand that reads empty is not already holding
        if self.held is not None:
            return SkillResult(False, "already_holding", {"held": self.held["kind"]})
        self._zone = pos[:2].copy()
        c = CATALOG[kind]
        radial = self._radial(pos)
        grasp_pos = pos.copy()
        lift_dz = (0.12, 0.09, 0.06)
        if grasp == "top" and obj_quat is not None and "com_local" in c:
            # Take a long object at its centre of mass (perceived orientation + catalog).
            off = mat_from_quat(obj_quat) @ np.asarray(c["com_local"], float)
            grasp_pos[:2] += off[:2]
        a = None
        if grasp == "top":
            if yaw is None:
                yaw = math.atan2(radial[1], radial[0]) + math.pi / 2
            R = hand_down(yaw)
            tcp = grasp_pos + [0, 0, c["top_dz"]]
            pre = tcp + [0, 0, 0.12]
            above = np.array([tcp[0], tcp[1], max(self.worktop_z + 0.40, pre[2])])
            self.open_gripper(c["open"])
            tare = self._tare()
            if c.get("low"):
                # Guarded descent: flat objects lie on a floor; stop on contact, back off 3 mm.
                g = self._guard_dir([0, 0, -1], 3.0, tare)
                r = self._approach([above, pre, tcp - [0, 0, 0.02]], R, [0.25, 0.05], guard=g,
                                   exit_check=self._lift_check(lift_dz))
                if not r.ok:
                    return SkillResult(False, r.reason, r.data)
                if r.data.get("stopped"):
                    p_now = self.tcp_pose(self.q_cmd)[0]
                    self._line(p_now + [0, 0, 0.003], None, speed=0.02)
            else:
                g = self._guard_dir([0, 0, -1], 8.0, tare)
                r = self._approach([above, pre, tcp], R, [0.25, 0.08], guard=g, linear_last=False,
                                   exit_check=self._lift_check(lift_dz))
                if not r.ok:
                    return SkillResult(False, r.reason, r.data)
                if r.data.get("stopped"):
                    self._line(pre, None, speed=0.1)
                    return SkillResult(False, "collision_on_approach", {"force_peak": g.state["peak"]})
        else:
            if not c["side"]:
                raise _Abort("bad_arg", {"error": f"no side grasp for {kind}"})
            a = radial if approach is None else approach
            tcp = pos + [0, 0, c["side_dz"]]
            pre = tcp - 0.08 * a  # fingertips 5 cm clear of a 6 cm cup
            R = self._side_variant(tcp, hand_horizontal(a))
            if R is None:
                return SkillResult(False, "unreachable", {"target": tcp.tolist()})
            self.open_gripper(c["open"])
            tare = self._tare()
            g = self._guard_dir(a, 8.0, tare)
            # The last 10 cm down to `pre` slowly: the palm passes just above a tray rim or a sink
            # wall there, and a fast descent sags below the planned line.
            r = self._approach([pre + [0, 0, 0.10], pre, tcp], R, [0.08, 0.08], guard=g,
                               linear_last=False, sym=False, lift_z=self.worktop_z + 0.30,
                               exit_check=self._lift_check(lift_dz))
            if not r.ok:
                return SkillResult(False, r.reason, r.data)
            if r.data.get("stopped"):
                self._line(pre, None, speed=0.1)
                return SkillResult(False, "collision_on_approach", {"force_peak": g.state["peak"]})

        way_in = getattr(self, "_way_in", None)
        cg = self.close_gripper(c.get("grip", 0.0))
        if not cg.data["grasped"]:
            self.open_gripper(c["open"])
            self._line(self.tcp_pose(self.q_cmd)[0] + [0, 0, 0.08], None, speed=0.1)
            return SkillResult(False, "no_grasp", cg.data)
        grip_policy = grip_w = None
        if "grip_v22" in c and self.scrub_gesture == "v2.2":
            # v2.2: closed as v2.1 (same grasp and grasp check), then held with a lighter squeeze: the command is
            # relaxed to grip_v22, never wider than the measured width minus HOLD_MARGIN (a narrow chord keeps a
            # squeeze). Closing straight to 4.8 cm missed a sponge pushed on a surface (no_grasp).
            grip_policy = "v2.2"
            grip_w = float(min(c["grip_v22"], float(cg.data["width"]) - self.HOLD_MARGIN))
            self._grip(grip_w, wait=0.3)
        p_g, R_g = self.tcp_pose(self.q_cmd)
        q_grasp = self.q_cmd.copy()
        self.held = {"kind": kind, "grasp": grasp, "provisional": True}  # held-object speed limits
        r = None
        for dz in lift_dz:
            r = self._line(p_g + [0, 0, dz], R_g, speed=0.12)
            if r.ok:
                break
        if not r.ok:
            # Planned before the grasp, impossible now (the object moved the hand): let go and
            # back off rather than stay closed on it.
            self.held = None
            self.open_gripper(c["open"])
            p = self.tcp_pose(self.q_cmd)[0]
            if a is not None:
                self._line(p - 0.08 * a, R_g, speed=0.1)
            else:
                self._line(p + [0, 0, 0.05], R_g, speed=0.05)
            return SkillResult(False, "lift_blocked", {"lift": r.reason})
        q_lifted = self.q_cmd.copy()
        if grasp == "side":
            # Out of the basin or off the tray toward the robot, then up if possible.
            p1 = self.tcp_pose(self.q_cmd)[0]
            self._line(p1 - 0.08 * a, R_g, speed=0.1)
            carry = self.worktop_z + ENVELOPE_H + 0.02 + c["hh"] + c["side_dz"]
            if self.tcp_pose(self.q_cmd)[0][2] < carry - 0.01 and c["hh"] > 0.06:
                # A tall object (spray bottle) still low, e.g. in a sink: up to the carry height,
                # else out by the way the hand came in.
                self._lift_to(carry)
                if self.tcp_pose(self.q_cmd)[0][2] < carry - 0.01 and way_in is not None:
                    self._retrace(way_in, carry)
        self._wait(15)
        w = self.sim.gripper_width()
        lt, rt = self.sim.finger_touch()
        if w < 0.004 or not (lt or rt):
            self.held = None
            return SkillResult(False, "slipped", {"width": w})
        lying = kind == "spray_bottle" and grasp == "top"
        held = {"kind": kind, "grasp": grasp, "pick_pos": pos.tolist(), "approach": None,
                "width": w, "upside_down": bool(upside_down) and grasp == "top", "lying": lying}
        hh = c.get("lying_hh", c["hh"]) if lying else c["hh"]
        if grasp == "top":
            held["tcp_above_bottom"] = hh + c["top_dz"]
        else:
            held["tcp_above_bottom"] = hh + c["side_dz"]
            held["approach"] = a.tolist()
        held["surface_z"] = float(pos[2] - hh)
        held["pick_tcp_z"] = float(p_g[2])
        # Joint configurations of the way out (for turn_over: the way back is known to be free).
        held["way_out"] = [q_grasp.tolist(), q_lifted.tolist(), self.q_cmd.tolist()]
        if obj_quat is not None:
            # Object pose in the hand frame, from the perceived object orientation and the
            # believed hand pose at the moment of the grasp.
            R_obj = mat_from_quat(obj_quat)
            held["R_obj_in_hand"] = (R_g.T @ R_obj).tolist()
            held["p_obj_in_hand"] = (R_g.T @ (pos - p_g)).tolist()
        out = {"width": w, "held": held}
        if grip_policy:  # v2.2: the grip version and width command are recorded with the grasp
            held["grip_policy"] = out["grip_policy"] = grip_policy
            held["grip_cmd_w"] = out["grip_cmd_w"] = float(grip_w)
        self.held = held
        return SkillResult(True, "ok", out)

    HOLD_MARGIN = 0.003  # v2.2: the relaxed sponge command stays at least this far below the measured width

    @_skill
    def place(self, pos, orientation: str = "upright", yaw: float | None = None) -> SkillResult:
        if orientation not in ("upright", "upside_down"):
            raise _Abort("bad_arg", {"error": f"orientation must be 'upright' or 'upside_down', got {orientation!r}"})
        pos = _vec("pos", pos)
        yaw = _num("yaw", yaw, -2 * math.pi, 2 * math.pi, allow_none=True)
        if self.held is None:
            return SkillResult(False, "not_holding", {})
        if not self.holding():
            self.held = None
            return SkillResult(False, "dropped", {})
        h = self.held
        self._zone = pos[:2].copy()  # (turn_over, if needed, works at the pick spot: its zone)
        if orientation == "upside_down" and h["grasp"] != "side" and not h["upside_down"]:
            raise _Abort("bad_arg", {"error": "upside_down needs an object picked with grasp='side'"})
        if orientation == "upright" and h["upside_down"]:
            raise _Abort("bad_arg", {"error": "the object is held upside down"})
        if orientation == "upside_down" and h["grasp"] == "side":
            self._zone = np.array(h["pick_pos"][:2])
            r = self.turn_over()
            if not r.ok:
                return r
            h = self.held
            self._zone = pos[:2].copy()
        if h["grasp"] == "side":
            R = self.tcp_pose(self.q_cmd)[1]
            tcp = pos + [0, 0, h["tcp_above_bottom"]]
            return self._place_down(tcp, R, mode="side", clear_z=tcp[2] + 0.10)
        # Top grasp: hand down. Finger axis: `yaw` if given, else across the line from the base
        # (the palm, 21 cm long, then lies sideways and its thin side faces the robot).
        tcp = pos + [0, 0, h["tcp_above_bottom"]]
        if yaw is None:
            rad = self._radial(tcp)
            yaw = math.atan2(rad[1], rad[0]) + math.pi / 2
        return self._place_down(tcp, hand_down(yaw), mode="top")

    def _place_down(self, tcp, R, mode, clear_z=None) -> SkillResult:
        pre = tcp + [0, 0, 0.10]
        clear_z = self.worktop_z + 0.40 if clear_z is None else clear_z
        above = np.array([tcp[0], tcp[1], max(clear_z, pre[2])])
        tare = self._tare()
        g = self._guard_dir([0, 0, -1], 3.0, tare)
        r = self._approach([above, pre, tcp - [0, 0, 0.02]], R, [0.25, 0.05], guard=g, sym=(mode == "top"))
        if not r.ok:
            return SkillResult(False, r.reason, r.data)
        if not self.holding():
            self.held = None
            return SkillResult(False, "dropped", {"stage": "before release"})
        if not r.data.get("stopped"):
            # 2 cm below the expected support and nothing felt: do not let go in the air.
            self._line(self.tcp_pose(self.q_cmd)[0] + [0, 0, 0.06], None, speed=0.1)
            return SkillResult(False, "no_support_contact", {})
        self.open_gripper(min(MAX_OPEN, self.held["width"] + 0.025))
        p, Rn = self.tcp_pose(self.q_cmd)
        self.held = None
        retreat = True
        if mode == "side":
            back = -self._radial(p)
            retreat = self._line(p + 0.10 * back, Rn, speed=0.1).ok
        up = self._line(self.tcp_pose(self.q_cmd)[0] + [0, 0, 0.10], Rn, speed=0.1).ok
        if not (retreat or up):
            return SkillResult(False, "no_retreat", {"contact": True})
        return SkillResult(True, "ok", {"contact": True})

    def _side_variant(self, tcp, R):
        """Of the two equivalent horizontal hand poses (finger axis +-), the one whose wrist can
        still roll 180 deg within its joint range (needed to turn the object over)."""
        lo7, hi7 = self.kin.lo[6] + 0.05, self.kin.hi[6] - 0.05
        best = None
        for Rv in (R, _flip_z(R)):
            q, ok, ep, _ = self._solve(tcp, Rv, sym=False)
            if not ok:
                continue
            room = (q[6] - math.pi > lo7) or (q[6] + math.pi < hi7)
            key = (0 if room else 1, float(np.abs(q - self.q_cmd).sum()))
            if best is None or key < best[0]:
                best = (key, Rv)
        return None if best is None else best[1]

    def _flip_in_air(self) -> SkillResult:
        """Roll the wrist (joint 7) by 180 deg: the hand axis is horizontal and passes through
        the TCP, so the held object turns upside down in place."""
        lo7, hi7 = self.kin.lo[6] + 0.02, self.kin.hi[6] - 0.02
        q = self.q_cmd.copy()
        if q[6] - math.pi > lo7:
            q[6] -= math.pi
        elif q[6] + math.pi < hi7:
            q[6] += math.pi
        else:
            return SkillResult(False, "flip_joint_limit", {"q7": float(q[6])})
        delta = float(q[6] - self.q_cmd[6])
        self._joint_move(q, speed=1.5, min_t=1.5)
        h = self.held
        h["upside_down"] = True
        h["flip_delta"] = delta
        c = CATALOG[h["kind"]]
        h["tcp_above_bottom"] = c["hh"] - c["side_dz"]
        return SkillResult(True, "ok", {})

    @_skill
    def turn_over(self, regrasp: bool = True) -> SkillResult:
        """Turn the held object upside down and hold it again from the top.

        Needs an object picked with grasp="side". The wrist rolls 180 deg in the air, the object
        is put down where it was picked (that spot is known to be free), then taken again from
        the top: a hand pointing down reaches more places than a horizontal one (a dishwasher
        basket, for instance). place(..., "upside_down") calls this when needed; a program may
        call it earlier (before opening a dishwasher next to the tray: an open basket blocks the
        flat-hand moves). regrasp=False leaves it upside down where it was picked (hand empty):
        a program can then open the dishwasher and pick it with pick(..., upside_down=True)."""
        h = self.held
        if h is None:
            return SkillResult(False, "not_holding", {})
        if h["grasp"] != "side":
            if h["upside_down"]:
                return SkillResult(True, "ok", {"already": True})
            raise _Abort("bad_arg", {"error": "turn_over needs an object picked with grasp='side'"})
        if not self.holding():
            self.held = None
            return SkillResult(False, "dropped", {})
        self._zone = np.array(h["pick_pos"][:2])
        if not h["upside_down"]:
            r = self._flip_in_air()
            if not r.ok:
                return r
        kind = h["kind"]
        c = CATALOG[kind]
        spot = np.array(h["pick_pos"], float)
        surface = np.array([spot[0], spot[1], h["surface_z"]])
        a = np.array(h["approach"], float)
        # Put it down where and at the height it was picked from (the palm clears the tray lip
        # as it did at the pick; the rim drops about 1 cm). Way back: the joint configurations
        # of the way out, wrist turned by the flip (same arm, same free path).
        way = [np.array(q) for q in reversed(h.get("way_out", []))]
        exit_q = None
        if way and abs(float(np.abs(way[0][:6] - self.q_cmd[:6]).max())) < 0.05:
            for q in way[1:]:
                q = q.copy()
                q[6] += h.get("flip_delta", 0.0)
                self._joint_move(q, speed=0.4, min_t=0.6)
            if len(way) >= 3:
                exit_q = way[1].copy()  # the lifted configuration of the way out
                exit_q[6] += h.get("flip_delta", 0.0)
        else:
            R = self.tcp_pose(self.q_cmd)[1]
            tcp = np.array([spot[0], spot[1], h["pick_tcp_z"]])
            tare = self._tare()
            g = self._guard_dir([0, 0, -1], 3.0, tare)
            r = self._approach([tcp + [0, 0, 0.10], tcp], R, [0.04], guard=g, sym=False)
            if not r.ok:
                return SkillResult(False, "turn_over_" + r.reason, r.data)
        if not self.holding():
            self.held = None
            return SkillResult(False, "turn_over_dropped", {})
        self.open_gripper(c["open"])
        self.held = None
        p, Rn = self.tcp_pose(self.q_cmd)
        r = self._line(p - 0.12 * a, Rn, speed=0.1)
        if not r.ok:
            r = self._line(p + [0, 0, 0.10], Rn, speed=0.1)
        if not r.ok and exit_q is not None and self._joint_route(self.q_cmd, exit_q) is not None:
            # Up the way the cup came down (fingers open around it): known to be free.
            self._joint_move(exit_q, speed=0.4, min_t=0.6)
            r = SkillResult(True, "ok", {})
        if not r.ok:
            return SkillResult(False, "turn_over_no_retreat", {})
        if not regrasp:
            return SkillResult(True, "ok", {"left_at": (surface + [0, 0, c["hh"]]).tolist()})
        # Second grasp, from the top: the upside-down cup is the same cylinder.
        yaw = math.atan2(a[1], a[0]) + math.pi / 2
        r = self.pick(surface + [0, 0, c["hh"]], kind, grasp="top", yaw=yaw)
        if not r.ok:
            return SkillResult(False, "turn_over_pick_" + r.reason, r.data)
        self.held["upside_down"] = True
        return SkillResult(True, "ok", {})

    # ================================================================ drawers and buttons
    def _line_check(self, direction, dist, speed=0.06):
        """exit_check: a straight line of `dist` along `direction` (same hand orientation) can be
        planned from the final configuration of an approach (e.g. the pull after a pinch)."""
        d = np.array([direction[0], direction[1], 0.0])

        def check(q_end):
            p_e, R_e = self.tcp_pose(q_end)
            n = max(int(dist / speed / CTRL_DT), 1)
            return self._plan_line(p_e, R_e, p_e + dist * d, R_e, n, True, q_end) is not None
        return check

    def _pinch_rail(self, handle_pos, axis, exit_check=None):
        """Hand down, fingers on both sides of a rail handle along `axis` (horizontal).
        pinch-rest (pinch-rest-2): a pinch that reads ok is checked for fingertips resting on the rail (_pinch_rest)."""
        axis = np.array([axis[0], axis[1], 0.0])
        axis /= np.linalg.norm(axis)
        R = hand_down(math.atan2(axis[1], axis[0]))
        handle_pos = np.asarray(handle_pos, float)
        self.open_gripper(0.045)
        tare = self._tare()
        g = self._guard_dir([0, 0, -1], PINCH_GUARD_N, tare)
        above = np.array([handle_pos[0], handle_pos[1], max(self.worktop_z + 0.40, handle_pos[2] + 0.08)])
        r = self._approach([above, handle_pos + [0, 0, 0.08], handle_pos], R, [0.25, 0.06], guard=g,
                           linear_last=False, exit_check=exit_check)
        if not r.ok:
            return SkillResult(False, r.reason, r.data), None
        R = self.tcp_pose(self.q_cmd)[1]
        if r.data.get("stopped"):
            self._line(handle_pos + [0, 0, 0.08], R, speed=0.1)
            return SkillResult(False, "collision_on_approach", {"force_peak": g.state["peak"]}), None
        cg = self.close_gripper()
        # A pinched rail: the fingers stopped at the rail width (1.6 cm) and touch it. One finger
        # may stay a hair off (the fingers are coupled and the arm is stiff); pulling and pushing
        # do not need friction, the rail sits between the fingers.
        if not self._on_rail(pinch=True):
            self.open_gripper(0.045)
            self._line(handle_pos + [0, 0, 0.08], R, speed=0.1)
            return SkillResult(False, "no_grasp_handle", cg.data), None
        rr = self._pinch_rest(handle_pos, R)  # pinch-rest-2
        if rr is not None:
            return rr, None
        return SkillResult(True, "ok", {}), R

    def _pinch_rest(self, handle_pos, R):
        """pinch-rest-2 (g3rc/tools/instr.py pinch_rest_check, variant g1, ported as
        pinch-rest-1; its trigger narrowed by v473c_pinchfix_20261016). Right after a rail pinch that reads ok: the
        hand's upward reaction (_hand_force, read as the tested variant read it, untared) above PINCH_GUARD_N with
        the gripper width below the rail thickness 2 x RAIL_HALF_THICKNESS_M (_pinch_resting) means the fingertips
        rest on the rail instead of straddling it (a corner pinch: the pads' lower edge at or above the rail top, the
        fingers closed past the rail faces). Then: lift by RAIL_HALF_THICKNESS_M (relieves the press, else the
        fingers cannot open), open to 0.045 m, lower by 2 x RAIL_HALF_THICKNESS_M on a straight line guarded like the
        pinch approach's descent (PINCH_GUARD_N), close again and re-check. Returns None when the pinch stands (no
        trigger, or a re-pinch accepted), else the failure no_grasp_handle (the existing one: open, back up 8 cm
        above the handle). A re-pinch is accepted only if the lift line ran, the lowering ran its full length, not
        stopped by the guard (14.2), the rail is between the fingers and the re-check does not rest (_pinch_resting
        on up2, w2). Every number is the code's own: the guard level, the rail half height, the 0.06 m/s of the
        drawer lines, _pinch_rail's 0.045 m opening and 0.08 m back-up. Below the trigger nothing moves: _hand_force
        and the gripper width are reads (no tick, no random stream). Each check is one event in the top-level
        result's data["pinch_rest"]; f_n is the full untared hand force at the check (records only)."""
        f = self._hand_force()  # untared, as the tested variant read it
        up = float(f[2])
        w = float(self.sim.gripper_width())
        ev = {"version": PINCH_REST_VERSION, "t_s": self._clock_s(), "up_n": round(up, 2), "w": round(w, 5),
              "f_n": [round(float(x), 2) for x in f], "skill": getattr(self, "_retreat_skill", None)}
        self._v473c_events("_pinch_rest_events").append(ev)
        if not _pinch_resting(up, w):
            ev["action"] = "none"
            return None
        # relieve the press first (the arm holds the fingertips down on the rail; the gripper cannot open under it)
        p = self.tcp_pose(self.q_cmd)[0]
        ru = self._line(p + [0, 0, RAIL_HALF_THICKNESS_M], R, speed=0.06)
        self.open_gripper(0.045)
        ev["after_open_w"] = round(float(self.sim.gripper_width()), 5)
        ev["after_open_f"] = [round(float(x), 2) for x in self._hand_force()]
        p = self.tcp_pose(self.q_cmd)[0]
        tare = self._tare()
        g = self._guard_dir([0, 0, -1], PINCH_GUARD_N, tare)  # the pinch approach descent's own guard
        rl = self._line(p - [0, 0, 2 * RAIL_HALF_THICKNESS_M], R, speed=0.06, guard=g)
        stopped = bool(rl.data.get("stopped"))
        ev.update(up_ok=bool(ru.ok), up_reason=str(ru.reason), lower_ok=bool(rl.ok), lower_reason=str(rl.reason),
                  lower_stopped=stopped, lower_fraction=rl.data.get("fraction"),
                  lower_guard_peak=round(float(g.state["peak"]), 2))
        cg = self.close_gripper()
        up2 = float(self._hand_force()[2])
        on = bool(self._on_rail(pinch=True))
        w2 = float(self.sim.gripper_width())
        ev.update(action="repinch", up2_n=round(up2, 2), w2=round(w2, 5), on_rail=on)
        reject = [name for name, bad in (("lift_failed", not ru.ok), ("lower_failed", not rl.ok),
                                         ("lower_stopped", stopped), ("not_on_rail", not on),
                                         ("still_resting", _pinch_resting(up2, w2))) if bad]
        if not reject:
            ev.update(outcome="ok", t1_s=self._clock_s())
            return None
        ev.update(outcome="no_grasp_handle", reject=reject)
        self.open_gripper(0.045)
        self._line(np.asarray(handle_pos, float) + [0, 0, 0.08], R, speed=0.1)
        ev["t1_s"] = self._clock_s()
        return SkillResult(False, "no_grasp_handle", cg.data)

    def _on_rail(self, pinch=False) -> bool:
        """Rail between the fingers. Right after the pinch the fingers rest on the 1.6 cm rail;
        while pushing or pulling, the force-limited fingers may be spread by the rail, but if the
        rail had slipped out they would close on nothing (width near 0)."""
        w = self.sim.gripper_width()
        lt, rt = self.sim.finger_touch()
        if pinch:
            return bool(0.010 < w < 0.026 and (lt or rt))
        return bool(w > 0.010 and (lt or rt))

    def _home_soft(self) -> bool:
        """go_home as a retry step: a planning failure is not an abort of the calling skill."""
        try:
            return self.go_home().ok
        except _Abort as e:
            if e.reason in ("no_path", "unreachable"):
                self.trace.append(f"home_soft {e.reason}")
                return False
            raise

    def _release_rail(self):
        self.open_gripper(0.045)
        self._line(self.tcp_pose(self.q_cmd)[0] + [0, 0, 0.08], None, speed=0.1)

    @_skill
    def open_drawer(self, handle_pos, pull_dir, distance: float = 0.22, min_distance=None) -> SkillResult:
        handle_pos = _vec("handle_pos", handle_pos)
        pull = _hdir("pull_dir", pull_dir)
        distance = _num("distance", distance, 0.02, 0.30)
        min_distance = distance - 0.02 if min_distance is None else _num("min_distance", min_distance, 0.0, distance)
        self._reconcile_held()  # robot-held-1: refused only while the hand reads holding
        if self.held is not None:
            return SkillResult(False, "hand_busy", {})
        self._zone = handle_pos[:2] + 0.5 * distance * pull[:2]  # the handle's whole travel
        moved, h, strokes = 0.0, handle_pos.copy(), 0
        while strokes < 2 and moved < distance - 0.005:
            chk = self._line_check(pull, min(distance - moved, 0.10))
            r, R = self._pinch_rail(h, pull, exit_check=chk)
            if not r.ok and r.reason in ("unreachable", "no_path") and strokes == 0 and self._home_soft():
                r, R = self._pinch_rail(h, pull, exit_check=chk)  # again from home
            if not r.ok and r.reason in ("unreachable", "no_path"):
                r, R = self._pinch_rail(h, pull)  # a pinch from which a shorter pull is possible
            if not r.ok:
                if strokes == 0:
                    return r
                break
            strokes += 1
            start = self.tcp_pose(self.q_cmd)[0]
            tare = self._tare()
            g = self._guard_dir(pull, 60.0, tare, consecutive=5)
            # Pull what is left, or as far as the arm reaches in one continuous line; a second stroke re-grips the
            # handle where it is now (a fresh arm configuration). open-ladder-1: the fallback lengths
            # OPEN_STROKE_FALLBACKS_M (0.235-0.22 before 0.215)
            want = distance - moved
            rr = None
            for dist in [want] + [d for d in OPEN_STROKE_FALLBACKS_M if d < want - 0.004]:
                rr = self._line(start + dist * pull, R, speed=0.06, guard=g)
                if rr.ok or rr.reason not in ("unreachable", "no_path"):
                    break
            d = float((self.tcp_pose()[0] - start) @ pull)
            kept = self._on_rail()
            self._release_rail()
            if not rr.ok:
                if strokes == 1:
                    return SkillResult(False, "pull_" + rr.reason, {"moved": moved + max(d, 0.0)})
                break
            if not kept:
                return SkillResult(False, "lost_handle", {"moved_hand": moved + d})
            if rr.data.get("stopped"):
                return SkillResult(False, "stuck", {"moved": moved + d, "force_peak": g.state["peak"]})
            moved += d
            h = h + d * pull
        data = {"moved": moved, "asked": distance, "strokes": strokes}
        if moved < min_distance:
            return SkillResult(False, "partial", data)
        return SkillResult(True, "ok", data)

    @_skill
    def close_drawer(self, handle_pos, push_dir, expected=None, max_distance: float = 0.28) -> SkillResult:
        handle_pos = _vec("handle_pos", handle_pos)
        push = _hdir("push_dir", push_dir)
        expected = _num("expected", expected, 0.0, 0.30, allow_none=True)
        max_distance = _num("max_distance", max_distance, 0.02, 0.30)
        self._reconcile_held()  # robot-held-1: refused only while the hand reads holding
        if self.held is not None:
            return SkillResult(False, "hand_busy", {})
        self._zone = handle_pos[:2] + 0.5 * min(max_distance, 0.25) * push[:2]
        need = max((expected or 0.0) + 0.02, 0.05)
        r, R = self._pinch_rail(handle_pos, push, exit_check=self._line_check(push, need))
        if not r.ok and r.reason in ("unreachable", "no_path") and self._home_soft():
            # From where the arm was left no path was found: again from home.
            r, R = self._pinch_rail(handle_pos, push, exit_check=self._line_check(push, need))
        if not r.ok and r.reason in ("unreachable", "no_path"):
            r, R = self._pinch_rail(handle_pos, push)
        if not r.ok:
            return r
        start = self.tcp_pose(self.q_cmd)[0]
        tare = self._tare()
        g = self._guard_dir(push, 25.0, tare, consecutive=3)
        r = None
        for dist in [max_distance] + [d for d in (0.24, 0.21) if need <= d < max_distance]:
            r = self._line(start + dist * push, R, speed=0.06, guard=g, linear=True)
            if r.ok or r.reason not in ("unreachable", "no_path"):
                break
        moved = float((self.tcp_pose()[0] - start) @ push)
        kept = self._on_rail()
        self._release_rail()
        data = {"moved": moved, "end_stop": bool(r.data.get("stopped"))}
        if not r.ok:
            return SkillResult(False, "push_" + r.reason, data)
        if not kept:
            return SkillResult(False, "lost_handle", data)
        if not r.data.get("stopped"):
            # Pushed the whole way without feeling the end stop: the handle was not pushed home.
            return SkillResult(False, "no_end_stop", data)
        if expected is not None and moved < expected - 0.02:
            return SkillResult(False, "not_closed", {**data, "expected": expected})
        return SkillResult(True, "ok", data)

    PRESS_TOL = 0.006  # fingertip stop within 6 mm of the button top (panel top is 1 cm lower)

    @_skill
    def press(self, button_pos) -> SkillResult:
        button_pos = _vec("button_pos", button_pos)
        self._reconcile_held()  # robot-held-1: refused only while the hand reads holding
        if self.held is not None:
            return SkillResult(False, "hand_busy", {})
        self._zone = button_pos[:2].copy()
        self._grip(0.0, wait=0.4)
        R = hand_down(math.atan2(self._radial(button_pos)[1], self._radial(button_pos)[0]) + math.pi / 2)
        tcp = button_pos + [0, 0, TIP_BELOW_TCP]
        tare = self._tare()
        g = self._guard_dir([0, 0, -1], 8.0, tare)
        above = np.array([tcp[0], tcp[1], max(self.worktop_z + 0.40, tcp[2] + 0.06)])
        r = self._approach([above, tcp + [0, 0, 0.06], tcp - [0, 0, 0.03]], R, [0.25, 0.03], guard=g)
        if not r.ok:
            return SkillResult(False, r.reason, r.data)
        stop_dz = float(self.tcp_pose()[0][2] - tcp[2])  # where the fingertip stopped vs the button top
        R = self.tcp_pose(self.q_cmd)[1]
        self._wait(15)  # hold
        peak = g.state["peak"]
        self._line(self.tcp_pose(self.q_cmd)[0] + [0, 0, 0.06], R, speed=0.1)
        self.open_gripper()
        data = {"force_peak": peak, "stop_dz": stop_dz}
        if not r.data.get("stopped"):
            return SkillResult(False, "no_contact", data)
        if abs(stop_dz) > self.PRESS_TOL:
            return SkillResult(False, "contact_off_target", data)
        return SkillResult(True, "ok", data)

    # ================================================================ tools
    PIVOT_ARM_MAX = 0.008  # m: largest lever of the held bottle's weight about the finger axis

    def _pivot_arm(self, R_hand, com_hand) -> float:
        """Lever (m) of the held object's weight about the fingers' closing axis: a two-finger
        grasp turns freely about that axis, so the weight must hang (or stand) in line with it."""
        lever = R_hand @ com_hand
        return abs(float(np.cross(lever, [0.0, 0.0, -1.0]) @ R_hand[:, 1]))

    @_skill
    def spray(self, target_pos, seconds: float = 2.0, standoff: float = 0.30,
              tilt_deg: float = 50.0) -> SkillResult:
        target = _vec("target_pos", target_pos)
        seconds = _num("seconds", seconds, 0.1, 20.0)
        standoff = _num("standoff", standoff, 0.10, 0.60)
        tilt_deg = _num("tilt_deg", tilt_deg, 10.0, 80.0)
        h = self.held
        if h is None or h["kind"] != "spray_bottle":
            return SkillResult(False, "not_holding_spray", {})
        if "R_obj_in_hand" not in h:
            return SkillResult(False, "nozzle_unknown", {"reason": "pick() needs obj_quat"})
        if not self.holding():
            self.held = None
            return SkillResult(False, "dropped", {"stage": "before"})
        self._zone = target[:2].copy()
        R_oh = np.array(h["R_obj_in_hand"])
        p_oh = np.array(h["p_obj_in_hand"])
        n_hand = R_oh @ SPRAY_DIR_LOCAL  # spray direction in the hand frame
        tip_hand = p_oh + R_oh @ SPRAY_NOZZLE_LOCAL  # nozzle tip in the hand frame
        com_hand = p_oh + R_oh @ np.asarray(CATALOG["spray_bottle"]["com_local"], float)
        side = h["grasp"] == "side"
        _, R_now = self.tcp_pose(self.q_cmd)
        v = R_now @ n_hand
        az_now = math.atan2(v[1], v[0])
        rad = self._radial(target)
        t = math.radians(tilt_deg)
        # Candidate spray directions: tilted down by tilt_deg, azimuth near the nozzle's current
        # one (small hand rotation, no arm reconfiguration with the bottle in hand), pointing away
        # from the robot (the nozzle then sits on the robot side of the target, not by the faucet).
        cands = []
        for k in range(12):
            az = az_now + math.radians(30 * k)
            dh = np.array([math.cos(az), math.sin(az), 0.0])
            if dh @ rad < 0.3:
                continue  # nozzle on the robot's side of the target (the faucet stands behind it)
            d = dh * math.cos(t) + np.array([0, 0, -math.sin(t)])
            vv = R_now @ n_hand
            axis = np.cross(vv, d)
            sn = np.linalg.norm(axis)
            ang = math.atan2(sn, float(vv @ d))
            R = (rot_about(axis, ang) if sn > 1e-6 else np.eye(3)) @ R_now
            cands.append((ang, d, R))
        cands.sort(key=lambda c: c[0])
        chosen = None
        corners_hand = [p_oh + R_oh @ np.asarray(cn, float) for cn in CATALOG["spray_bottle"]["corners_local"]]
        floor_z = self.worktop_z + SINK_RIM_H + 0.015

        def low_point(tcp, R):
            return min(float((tcp + R @ cn)[2]) for cn in corners_hand)
        for ang, d, R0 in cands:
            # Nozzle `standoff` from the target along the spray direction, or farther if the
            # bottle would dip below the sink rim (site plan).
            st = standoff
            for st in [standoff] + [x for x in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60) if x > standoff]:
                if low_point(target - st * d - R0 @ tip_hand, R0) >= floor_z:
                    break
            nozzle_goal = target - st * d
            # First the hand turned by the smallest rotation (no twist about the spray axis),
            # then, if unreachable, with the twist about the spray axis left free.
            for free in (False, True):
                R = R0
                tcp = nozzle_goal - R @ tip_hand
                q, ok, _, _ = self._solve(tcp, R, sym=False, free_axis_w=d if free else None)
                for _ in range(3 if free else 0):
                    if not ok:
                        break
                    pb, Rb = self.kin.fk(q)
                    R = self.belief.rot_base_to_world(Rb)
                    tcp = nozzle_goal - R @ tip_hand
                    q, ok, _, _ = self._solve(tcp, R, sym=False, q_hint=q)
                if not ok:
                    continue
                if side and self._pivot_arm(R, com_hand) > self.PIVOT_ARM_MAX:
                    continue
                if low_point(tcp, R) < floor_z - 0.02:
                    continue
                r = self._approach([tcp + [0, 0, 0.08], tcp], R, [0.1], linear_last=False, sym=False,
                                   lift_z=self.worktop_z + 0.30)
                if r.ok:
                    chosen = (tcp, R)
                    break
                if r.reason not in ("no_path", "unreachable"):
                    return SkillResult(False, r.reason, r.data)
            if chosen is not None:
                break
        if chosen is None:
            return SkillResult(False, "unreachable", {"target": target.tolist()})
        if not self.holding():
            self.held = None
            return SkillResult(False, "dropped", {"stage": "approach"})
        self.sim.flags["spray_trigger"] = True
        t0 = self.sim.t
        try:
            for k in range(int(seconds / CTRL_DT)):
                self._wait()
                if k % 5 == 4 and not self.holding():
                    self.held = None
                    return SkillResult(False, "dropped", {"stage": "spraying", "t_start": t0, "t_drop": self.sim.t})
        finally:
            self.sim.flags["spray_trigger"] = False
        t1 = self.sim.t
        self._line(self.tcp_pose(self.q_cmd)[0] + [0, 0, 0.05], None, speed=0.1)
        if not self.holding():
            self.held = None
            return SkillResult(False, "dropped", {"stage": "after", "t_start": t0, "t_end": t1})
        return SkillResult(True, "ok", {"t_start": t0, "t_end": t1, "tilt_deg": tilt_deg, "standoff": st})

    SCRUB_SELF_FRAC = 0.6  # v1 only: share of the scrub with >= 1 N felt at the hand (own force reading)
    # v2, see scrub():
    SCRUB_STROKE_AMP = 0.01  # circle radius (m) along the sponge's long axis (hand x)
    SCRUB_STROKE_AMP_ACROSS = 0.01  # and across it (hand y, the fingers' closing axis)
    SCRUB_STROKE_HZ = 1.0  # turns per second
    SCRUB_RAMP_S = 0.25  # the stroke grows from 0 over this time
    SCRUB_FELT_N = 1.0  # own reading counted as contact (the evaluator's contact force)
    SCRUB_FELT_FRAC = 0.8  # share of the requested time with contact felt (the verifier's 80 %)
    SCRUB_MAX_DESCENT = 0.006  # hand never more than this below the first-contact height
    SCRUB_FORCE_STOP = 15.0  # own reading (N, norm): stop and lift ("force_limit")
    SCRUB_HOLD_MISSES = 2  # v2.1: consecutive ticks without the grip felt -> "dropped"

    @_skill
    def scrub(self, target_pos, seconds: float = 3.0, force: float = 4.0) -> SkillResult:
        """Holding the sponge: scrub the dish top surface at target_pos for `seconds` at about
        `force` newtons (own reading), then lift 6 cm.

        v2: 1 cm circles, 1 turn per second, CENTRED on target_pos and started from it with a 0.25 s ramp (v1: circles
        of the same size centred 1 cm beside the target, so the sponge and the fingers spent the scrub off-centre toward
        one side of the dish; straight back-and-forth strokes were tried in development and moved the bowl more); the
        hand never goes more than 6 mm below the height where it first
        felt the dish (fingertips 1.6 cm above the sponge's scrubbing face: they stay above the dish top); it stops and
        lifts if its own force reading exceeds 15 N ("force_limit": a finger or the hand pressing the dish), or as soon
        as the sponge is no longer held ("dropped"). ok only if the whole requested time was done, the sponge is still
        held, and contact (>= 1 N, own reading) was felt during >= 80 % of the requested time (else "weak_contact"). v1
        said ok from 60 %: it reported ok after 1.7 to 2.0 s of contact for a 3 s request, which the evaluator
        rejected. The evaluator, not this ok, decides the task (taskcheck).

        v2.1 (default): v2 checked the grip only when the skill started (before the transport) and every 10th tick of
        the scrub. In two bench episodes the sponge slipped
        out during the transport, the empty hand went down onto the bowl and the fingers pressed it (110.7 N
        and 32.2 N computed by the bench; the robot's net reading stayed under 15 N because the finger contacts were
        opposite). v2.1 also checks the grip on every tick of the guarded descent (the hand is still 5 cm or more
        above the dish when it starts), stops and lifts without touching the dish if the grip is gone ("dropped",
        stage "approach"), and checks the grip on every tick of the scrub (2 consecutive misses = "dropped"). Limit:
        the finger switches cannot tell a sponge from a dish: fingers closed on the bowl rim still read as holding."""
        target = _vec("target_pos", target_pos)
        seconds = _num("seconds", seconds, 0.2, 30.0)
        force = _num("force", force, 0.5, 15.0)
        h = self.held
        if h is None or h["kind"] != "sponge":
            return SkillResult(False, "not_holding_sponge", {})
        if not self.holding():
            self.held = None
            return SkillResult(False, "dropped", {})
        if self.scrub_gesture == "v1":
            return self._scrub_v1(target, seconds, force)
        strict = self.scrub_gesture in ("v2.1", "v2.2")  # v2.2: same scrub, lighter grip (pick)
        label = self.scrub_gesture
        self._zone = target[:2].copy()
        tcp0 = target + [0, 0, h["tcp_above_bottom"]]
        R = self._carry_R(tcp0)
        tare = self._tare()
        g = self._guard_dir([0, 0, -1], 0.6 * force, tare)
        lost = {"miss": 0, "gone": False}
        if strict:
            g_force = g

            def g():  # v2.1: the guarded descent also stops as soon as the grip is gone
                lost["miss"] = 0 if self.holding() else lost["miss"] + 1
                if lost["miss"] >= self.SCRUB_HOLD_MISSES:
                    lost["gone"] = True
                    return True
                return g_force()
        above = np.array([tcp0[0], tcp0[1], max(self.worktop_z + 0.40, tcp0[2] + 0.05)])
        r = self._approach([above, tcp0 + [0, 0, 0.05], tcp0 - [0, 0, 0.03]], R, [0.25, 0.01], guard=g,
                           sym=False)
        if not r.ok:
            return SkillResult(False, r.reason, r.data)
        if strict and (lost["gone"] or not self.holding()):
            self._line(self.tcp_pose(self.q_cmd)[0] + [0, 0, 0.06], R, speed=0.1)
            self.held = None
            return SkillResult(False, "dropped", {"stage": "approach", "gesture": label})
        if not r.data.get("stopped"):
            self._line(self.tcp_pose(self.q_cmd)[0] + [0, 0, 0.06], R, speed=0.1)
            return SkillResult(False, "no_contact", {})
        centre = self.tcp_pose(self.q_cmd)[0]
        ax = np.array([R[0, 0], R[1, 0], 0.0])  # hand x: the sponge's long axis (fingers close along hand y)
        ax /= max(float(np.linalg.norm(ax)), 1e-9)
        ay = np.array([-ax[1], ax[0], 0.0])
        z_off = 0.0
        t0 = self.sim.t
        q_prev = self.q_cmd.copy()
        forces, peak = [], 0.0
        n = int(round(seconds / CTRL_DT))
        done, reason = 0, None
        for k in range(n):
            f = self._hand_force() - tare
            f_up = float(f[2])
            forces.append(f_up)
            peak = max(peak, float(np.linalg.norm(f)))
            if peak > self.SCRUB_FORCE_STOP:
                reason = "force_limit"
                break
            if strict:
                lost["miss"] = 0 if self.holding() else lost["miss"] + 1
                if lost["miss"] >= self.SCRUB_HOLD_MISSES:
                    reason = "dropped"
                    break
            elif k % 10 == 9 and not self.holding():
                reason = "dropped"
                break
            z_off += float(np.clip(2e-5 * (force - f_up), -0.0006, 0.0006))
            z_off = float(np.clip(z_off, -self.SCRUB_MAX_DESCENT, 0.01))
            tk = k * CTRL_DT
            ramp = min(1.0, tk / self.SCRUB_RAMP_S)
            ph = 2 * math.pi * self.SCRUB_STROKE_HZ * tk
            p = (centre + ramp * self.SCRUB_STROKE_AMP * math.sin(ph) * ax
                 - ramp * self.SCRUB_STROKE_AMP_ACROSS * math.cos(ph) * ay + [0, 0, -z_off])
            pb, Rb = self._to_base(p, R)
            q, ep, er = self.kin.ik(pb, Rb, q_prev, iters=30, posture=TRACK_POSTURE)
            if ep > 3e-3:
                reason = "scrub_cut_short"
                break
            q_prev = q
            self._cmd(q)
            done += 1
        t1 = self.sim.t
        self._line(self.tcp_pose(self.q_cmd)[0] + [0, 0, 0.06], R, speed=0.1)
        fr = np.array(forces) if forces else np.zeros(0)
        felt = int((fr >= self.SCRUB_FELT_N).sum())
        data = {"t_start": t0, "t_end": t1, "force_mean": float(fr.mean()) if len(fr) else 0.0,
                "felt_fraction": felt / max(n, 1), "felt_s": felt * CTRL_DT, "force_peak": peak, "gesture": label}
        if reason is not None and reason != "dropped":
            return SkillResult(False, reason, data)
        if reason == "dropped" or not self.holding():
            self.held = None
            return SkillResult(False, "dropped", data)
        if felt < self.SCRUB_FELT_FRAC * n:
            return SkillResult(False, "weak_contact", data)
        return SkillResult(True, "ok", data)

    def _scrub_v1(self, target, seconds, force) -> SkillResult:
        """The first scrub gesture (v1) (1 cm circles beside target, 60 % felt), unchanged; comparison runs only."""
        h = self.held
        self._zone = target[:2].copy()
        tcp0 = target + [0, 0, h["tcp_above_bottom"]]
        R = self._carry_R(tcp0)
        tare = self._tare()
        g = self._guard_dir([0, 0, -1], 0.6 * force, tare)
        above = np.array([tcp0[0], tcp0[1], max(self.worktop_z + 0.40, tcp0[2] + 0.05)])
        r = self._approach([above, tcp0 + [0, 0, 0.05], tcp0 - [0, 0, 0.03]], R, [0.25, 0.01], guard=g,
                           sym=False)
        if not r.ok:
            return SkillResult(False, r.reason, r.data)
        if not r.data.get("stopped"):
            self._line(self.tcp_pose(self.q_cmd)[0] + [0, 0, 0.06], R, speed=0.1)
            return SkillResult(False, "no_contact", {})
        # Force-controlled circles: radius 1 cm, 1 turn per second, z adjusted from the
        # measured contact force (admittance on top of the joint position servo).
        centre = self.tcp_pose(self.q_cmd)[0]
        z_off = 0.0
        t0 = self.sim.t
        q_prev = self.q_cmd.copy()
        forces = []
        n = int(seconds / CTRL_DT)
        done = 0
        for k in range(n):
            f_up = float((self._hand_force() - tare)[2])
            forces.append(f_up)
            z_off += float(np.clip(2e-5 * (force - f_up), -0.0006, 0.0006))
            z_off = float(np.clip(z_off, -0.015, 0.01))
            ph = 2 * math.pi * k * CTRL_DT
            p = centre + [0.01 * math.cos(ph) - 0.01, 0.01 * math.sin(ph), -z_off]
            pb, Rb = self._to_base(p, R)
            q, ep, er = self.kin.ik(pb, Rb, q_prev, iters=30, posture=TRACK_POSTURE)
            if ep > 3e-3:
                break
            q_prev = q
            self._cmd(q)
            done += 1
        t1 = self.sim.t
        self._line(self.tcp_pose(self.q_cmd)[0] + [0, 0, 0.06], R, speed=0.1)
        frac = float(np.mean(np.array(forces) >= 1.0)) if forces else 0.0
        data = {"t_start": t0, "t_end": t1, "force_mean": float(np.mean(forces)) if forces else 0.0,
                "felt_fraction": frac}
        if done < n:
            return SkillResult(False, "scrub_cut_short", data)
        if not self.holding():
            self.held = None
            return SkillResult(False, "dropped", data)
        if frac < self.SCRUB_SELF_FRAC:
            return SkillResult(False, "weak_contact", data)
        return SkillResult(True, "ok", data)
