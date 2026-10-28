"""Ground-truth predicates for each skill. Bench only: the robot never calls this module.

Thresholds:
- calibration: roll and pitch error <= 0.5 deg; base height error <= 1 cm (yaw and x, y are reported too);
- drawer open: joint opening >= 0.12 m; closed: <= 0.01 m;
- pick: object lifted >= 5 cm and still held 0.3 s later (touching a finger);
- place: resting within 3 cm of the target, upright within 10 deg, on the surface, released, nearly still;
- press: contact force > 2 N on the start button; spray: the spray ray hits the sink floor most of the time;
- scrub: sponge-dish contact force >= 1 N during >= 80 % of the window, dish still upright (within 25 deg);
- move_to: true tool point within 1 cm of the target; look_pose: the arm hides <= 5 % of the modules' pixels.
wilson() gives the Wilson score interval used to report success rates.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from .robot.belief import belief_error
from .world.build import SPRAY_DIR_LOCAL, SPRAY_NOZZLE_LOCAL, item_half_height

CALIB_TILT_DEG, CALIB_POS_M = 0.5, 0.01
OPEN_MIN, CLOSED_MAX = 0.12, 0.01
LIFT_MIN = 0.05
PLACE_XY, PLACE_ANGLE_DEG, PLACE_Z = 0.03, 10.0, 0.01
DISH_TILT_DEG = 25.0
PRESS_N = 2.0
SPRAY_FRAC = 0.8
SCRUB_N, SCRUB_FRAC = 1.0, 0.8
REACH_M = 0.01
VIEW_OCCLUSION = 0.05


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def calibration(sim, belief) -> dict:
    bp, bR = sim.base_pose()
    e = belief_error(belief, bp, bR)
    e["pass"] = bool(e["roll_err_deg"] <= CALIB_TILT_DEG and e["pitch_err_deg"] <= CALIB_TILT_DEG
                     and e["z_err_m"] <= CALIB_POS_M)
    e["pass_full"] = bool(e["pass"] and e["yaw_err_deg"] <= CALIB_TILT_DEG and e["xy_err_m"] <= CALIB_POS_M)
    return e


def drawer_open(sim, joint: str) -> dict:
    v = sim.joint_pos(joint)
    return {"pass": bool(v >= OPEN_MIN), "opening_m": v}


def drawer_closed(sim, joint: str) -> dict:
    v = sim.joint_pos(joint)
    return {"pass": bool(v <= CLOSED_MAX), "opening_m": v}


def _touches_fingers(sim, body: str) -> bool:
    bid = sim.model.body(body).id
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        b1, b2 = sim.model.geom_bodyid[c.geom1], sim.model.geom_bodyid[c.geom2]
        if (b1 == bid and b2 in sim.finger_bodies) or (b2 == bid and b1 in sim.finger_bodies):
            return True
    return False


def lifted(sim, body: str, z0: float) -> dict:
    z1 = float(sim.body_pos(body)[2])
    sim.tick(15)
    z2 = float(sim.body_pos(body)[2])
    held = _touches_fingers(sim, body)
    return {"pass": bool(z1 - z0 >= LIFT_MIN and z2 - z0 >= LIFT_MIN and held),
            "lift_m": z2 - z0, "held": held}


def placed(sim, body: str, target, orientation: str) -> dict:
    sim.tick(25)  # let it settle
    p = sim.body_pos(body)
    R = sim.body_mat(body)
    axis = R[:, 2]
    up_angle = math.degrees(math.acos(max(-1.0, min(1.0, float(axis[2])))))
    angle = up_angle if orientation == "upright" else 180.0 - up_angle
    dxy = float(np.hypot(p[0] - target[0], p[1] - target[1]))
    kind = next(i.kind for i in sim.site.items if i.name == body)
    dz = float(p[2] - item_half_height(kind) - target[2])  # bottom (or rim) vs the support surface
    vel = float(np.linalg.norm(sim.data.body(body).cvel[3:]))
    released = not _touches_fingers(sim, body)
    ok = dxy <= PLACE_XY and angle <= PLACE_ANGLE_DEG and abs(dz) <= PLACE_Z and released and vel < 0.05
    return {"pass": bool(ok), "xy_err_m": dxy, "angle_deg": angle, "dz_m": dz, "released": released,
            "speed": vel}


def tcp_reach(sim, target) -> dict:
    p = sim.data.site("panda/tcp").xpos
    e = float(np.linalg.norm(p - np.asarray(target)))
    return {"pass": bool(e <= REACH_M), "err_m": e}


def view_occlusion(sim, camera: str = "robot_eye", size=(320, 240)) -> dict:
    """Share of the modules' and items' pixels that the arm hides in a camera image
    (segmentation render with and without the robot's geom groups 2 and 3)."""
    m, d = sim.model, sim.data
    r = mujoco.Renderer(m, size[1], size[0])
    r.enable_segmentation_rendering()
    opt = mujoco.MjvOption()
    r.update_scene(d, camera=camera, scene_option=opt)
    with_robot = r.render().copy()
    opt.geomgroup[2] = 0
    opt.geomgroup[3] = 0
    r.update_scene(d, camera=camera, scene_option=opt)
    without = r.render().copy()
    r.close()
    keep = np.array([not (m.body(m.geom_bodyid[g]).name in ("world", "pedestal", "mount_plate")
                          or m.body(m.geom_bodyid[g]).name.startswith("panda/")) for g in range(m.ngeom)])

    def scene_px(seg):
        ids = seg[:, :, 0]
        ok = (seg[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM) & (ids >= 0)
        out = np.zeros(ok.shape, bool)
        out[ok] = keep[ids[ok]]
        return out
    A, B = scene_px(with_robot), scene_px(without)
    occ = 1.0 - float((A & B).sum()) / max(int(B.sum()), 1)
    return {"pass": bool(occ <= VIEW_OCCLUSION), "occlusion": occ}


# ------------------------------------------------------------ monitors (per-tick ground truth)

class ButtonMonitor:
    def __init__(self, sim, geom: str = "dw_button"):
        self.gid = sim.model.geom(geom).id
        self.peak = 0.0

    def __call__(self, sim):
        f6 = np.zeros(6)
        for i in range(sim.data.ncon):
            c = sim.data.contact[i]
            if self.gid in (c.geom1, c.geom2):
                other = c.geom2 if c.geom1 == self.gid else c.geom1
                if sim.robot_geom[other]:
                    mujoco.mj_contactForce(sim.model, sim.data, i, f6)
                    self.peak = max(self.peak, abs(float(f6[0])))

    def result(self) -> dict:
        return {"pass": bool(self.peak > PRESS_N), "force_peak_n": self.peak}


class SprayMonitor:
    """Ray from the real nozzle along the real spray direction, against the basin floor."""

    def __init__(self, sim, bottle: str = "spray_bottle"):
        self.bottle = bottle
        info = sim.info
        f = info.module_frames["sink"]
        self.frame = f
        self.floor_z = f.z + 0.012
        self.half = (0.158 - 0.02, 0.188 - 0.02)
        self.on = 0
        self.hit = 0

    def __call__(self, sim):
        if not sim.flags.get("spray_trigger"):
            return
        self.on += 1
        p = sim.body_pos(self.bottle)
        R = sim.body_mat(self.bottle)
        o = p + R @ SPRAY_NOZZLE_LOCAL
        d = R @ SPRAY_DIR_LOCAL
        if d[2] >= -1e-3:
            return
        t = (self.floor_z - o[2]) / d[2]
        if t <= 0:
            return
        q = o + t * d
        f = self.frame
        c, s = math.cos(f.yaw), math.sin(f.yaw)
        lx = c * (q[0] - f.x) + s * (q[1] - f.y)
        ly = -s * (q[0] - f.x) + c * (q[1] - f.y)
        if abs(lx) <= self.half[0] and abs(ly) <= self.half[1]:
            self.hit += 1

    def result(self, seconds: float, ctrl_dt: float) -> dict:
        frac = self.hit / self.on if self.on else 0.0
        dur = self.on * ctrl_dt
        return {"pass": bool(frac >= SPRAY_FRAC and dur >= 0.9 * seconds), "aimed_fraction": frac,
                "duration_s": dur}


class ScrubMonitor:
    """Normal force between the sponge and one dish, sampled while the robot scrubs."""

    def __init__(self, sim, dish: str, sponge: str = "sponge"):
        m = sim.model
        self.sponge = {g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(sponge).id}
        self.dish = {g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(dish).id}
        self.dish_name = dish
        self.samples: list[tuple[float, float]] = []

    def __call__(self, sim):
        f6 = np.zeros(6)
        tot = 0.0
        for i in range(sim.data.ncon):
            c = sim.data.contact[i]
            if (c.geom1 in self.sponge and c.geom2 in self.dish) or (c.geom2 in self.sponge and c.geom1 in self.dish):
                mujoco.mj_contactForce(sim.model, sim.data, i, f6)
                tot += abs(float(f6[0]))
        self.samples.append((sim.t, tot))

    def result(self, sim, t0: float, t1: float, seconds: float) -> dict:
        win = [f for t, f in self.samples if t0 <= t <= t1]
        frac = (sum(f >= SCRUB_N for f in win) / len(win)) if win else 0.0
        R = sim.body_mat(self.dish_name)
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(R[2, 2])))))  # signed: upside down = 180
        ok = frac >= SCRUB_FRAC and (t1 - t0) >= 0.9 * seconds and tilt <= DISH_TILT_DEG
        return {"pass": bool(ok), "contact_fraction": frac, "window_s": t1 - t0,
                "force_mean_n": float(np.mean(win)) if win else 0.0, "dish_tilt_deg": tilt}
