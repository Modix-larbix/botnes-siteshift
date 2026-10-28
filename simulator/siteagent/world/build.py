"""Build a MuJoCo model of a back-kitchen workcell from a SiteSpec.

Layout: a U-shaped worktop around a robot on a pedestal. Modules (sink, dishwasher, drawer unit, landing tray, drying
mat, bins) stand in slots and face the robot. Each module has a local frame: origin at the worktop surface, +x
pointing to the robot. This is the simulator side: the robot never imports it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from .spec import WORKTOP_COLORS, Item, Module, SiteSpec, mount_quat, slot_xy

LAB = Path(__file__).resolve().parents[2]
PANDA_XML = LAB / "menagerie" / "franka_emika_panda" / "panda.xml"

BOX, CYL, PLANE = mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_PLANE
SLIDE = mujoco.mjtJoint.mjJNT_SLIDE

STEEL = [0.72, 0.74, 0.77, 1]
DARK = [0.18, 0.18, 0.20, 1]
WHITE = [0.93, 0.93, 0.95, 1]

ITEM_GEOMS = {
    # kind: list of (type, size, local pos, rgba, collides), mass
    # Sizes: cup radius 3.0 cm (earlier 3.4), ramekin 3.1 x 2.2 cm (earlier 3.8 x 2.5), folded towel
    # 6.4 cm wide (earlier 11 cm): the Franka hand opens 8 cm, it needs 1 cm of margin per side.
    "cup": ([(CYL, [0.030, 0.045], [0, 0, 0], WHITE, True),
             (CYL, [0.025, 0.001], [0, 0, 0.045], [0.25, 0.25, 0.28, 1], False)], 0.20),
    "bowl": ([(CYL, [0.031, 0.022], [0, 0, 0], [0.85, 0.80, 0.70, 1], True),
              (CYL, [0.026, 0.001], [0, 0, 0.022], [0.45, 0.40, 0.35, 1], False)], 0.15),
    "fork": ([(BOX, [0.075, 0.010, 0.006], [0, 0, 0], [0.80, 0.80, 0.82, 1], True)], 0.05),
    "spoon": ([(BOX, [0.070, 0.012, 0.006], [0, 0, 0], [0.75, 0.75, 0.70, 1], True)], 0.05),
    "sponge": ([(BOX, [0.045, 0.030, 0.014], [0, 0, 0], [0.98, 0.85, 0.20, 1], True),
                (BOX, [0.045, 0.030, 0.004], [0, 0, 0.018], [0.20, 0.55, 0.25, 1], True)], 0.03),
    # Trigger spray: flat body (6 x 4.4 cm, earlier a cylinder, it rolled in the drawer),
    # head on top; the nozzle points along the bottle's local +y.
    "spray_bottle": ([(BOX, [0.030, 0.022, 0.075], [0, 0, 0], [0.30, 0.55, 0.90, 1], True),
                      (BOX, [0.012, 0.025, 0.020], [0, 0.01, 0.095], WHITE, True),
                      (BOX, [0.004, 0.004, 0.004], [0, 0.037, 0.105], [0.9, 0.2, 0.2, 1], False)], 0.35),
    "towel": ([(BOX, [0.070, 0.032, 0.0125], [0, 0, 0], [0.80, 0.20, 0.20, 1], True)], 0.08),
}
# How each item's mass is shared between its colliding geoms (default: equally). The spray
# bottle's liquid is in the body, not in the trigger head (an earlier version split it 50/50).
ITEM_MASS_SPLIT = {"spray_bottle": [0.85, 0.15], "sponge": [0.8, 0.2]}
SPRAY_NOZZLE_LOCAL = np.array([0.0, 0.037, 0.105])  # nozzle tip, bottle frame
SPRAY_DIR_LOCAL = np.array([0.0, 1.0, 0.0])  # spray direction, bottle frame

# Robot cameras. They are part of the robot's own design: the constants and the extrinsics
# functions live in robot.design (robot-side code must not import the builder) and
# are imported here to place the cameras in the simulated robot. Values unchanged.
from ..robot.design import (EYE_FOVY, EYE_POS_PLATE, EYE_TARGET_PLATE, LINK0_ON_PLATE_Z,  # noqa: E402,F401
                            WRIST_CAM_FOVY, WRIST_CAM_POS_HAND, WRIST_CAM_QUAT_HAND,
                            robot_eye_in_base, wrist_cam_in_hand)
from ..robot.design import look_at_quat as _look_at_quat  # noqa: E402


def item_half_height(kind: str) -> float:
    geoms, _ = ITEM_GEOMS[kind]
    return max(p[2] + (s[1] if t == CYL else s[2]) for t, s, p, _, _ in geoms)


@dataclass
class Frame2D:
    """A module's pose on the worktop: position and yaw (module +x points to the robot)."""
    x: float
    y: float
    z: float
    yaw: float

    def to_world(self, local) -> np.ndarray:
        lx, ly, lz = local
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return np.array([self.x + c * lx - s * ly, self.y + s * lx + c * ly, self.z + lz])

    def quat(self) -> list[float]:
        return [math.cos(self.yaw / 2), 0, 0, math.sin(self.yaw / 2)]


@dataclass
class BuildInfo:
    """What the bench, the checker and the simulated sensors need to know about the scene."""
    spec: SiteSpec
    module_frames: dict[str, Frame2D] = field(default_factory=dict)
    # Semantic places: name -> (world xyz at build time, owning body name or None)
    places: dict[str, tuple[np.ndarray, str | None]] = field(default_factory=dict)
    drawer_joints: dict[str, str] = field(default_factory=dict)  # "drawers/0" -> joint name
    sponge_dish_friction: float | None = None  # world option, see SPONGE_DISH_FRICTION
    item_bodies: dict[str, str] = field(default_factory=dict)  # item name -> body name


def _box(body, name, half, pos, rgba, collide=True, **kw):
    g = body.add_geom(name=name, type=BOX, size=list(half), pos=list(pos), rgba=list(rgba), **kw)
    if not collide:
        g.contype = g.conaffinity = 0
    return g


def _open_box(body, prefix, half_xy, height, t, pos, rgba, front_height=None):
    """Floor plate + four walls. The +x wall can be lower (front_height)."""
    hx, hy = half_xy
    px, py, pz = pos
    _box(body, f"{prefix}_floor", (hx, hy, t / 2), (px, py, pz + t / 2), rgba)
    for name, half, off, h in (
        ("back", (t / 2, hy, height / 2), (-hx + t / 2, 0), height),
        ("front", (t / 2, hy, (front_height or height) / 2), (hx - t / 2, 0), front_height or height),
        ("left", (hx, t / 2, height / 2), (0, hy - t / 2), height),
        ("right", (hx, t / 2, height / 2), (0, -hy + t / 2), height),
    ):
        _box(body, f"{prefix}_{name}", (half[0], half[1], h / 2), (px + off[0], py + off[1], pz + h / 2), rgba)


def _add_place(info, name, frame: Frame2D, local, body=None):
    info.places[name] = (frame.to_world(local), body)


# ---------------------------------------------------------------- modules

def _sink(spec, info, f: Frame2D, m: Module):
    b = spec.worldbody.add_body(name="sink", pos=[f.x, f.y, f.z], quat=f.quat())
    _open_box(b, "sink", (0.17, 0.20), 0.14, 0.012, (0, 0, 0), STEEL, front_height=0.09)
    # Faucet at the back.
    _box(b, "faucet_post", (0.012, 0.012, 0.12), (-0.15, 0.0, 0.26), STEEL)
    _box(b, "faucet_spout", (0.07, 0.010, 0.010), (-0.09, 0.0, 0.37), STEEL)
    _add_place(info, "sink/basin", f, (0.0, 0.0, 0.02), "sink")
    _add_place(info, "sink/corner", f, (0.10, 0.14, 0.012), "sink")


DRAWER_D, DRAWER_WD, DRAWER_HU = 0.30, 0.17, 0.10  # unit depth, width per drawer, unit height
DRAWER_RANGE = 0.24


def _rail(body, prefix, x, z_top, half_len, rgba=DARK):
    """A rail handle standing 3 cm above the front edge: the hand pinches it from above."""
    _box(body, f"{prefix}_handle", (0.008, half_len, 0.008), (x, 0, z_top + 0.03), rgba)
    for side, y in (("l", half_len - 0.006), ("r", -half_len + 0.006)):
        _box(body, f"{prefix}_post_{side}", (0.006, 0.005, 0.012), (x, y, z_top + 0.012), rgba)


def _drawers(spec, info, f: Frame2D, m: Module):
    """Drawer unit, drawers side by side (earlier versions stacked them, and a hand reaching a lower
    handle hit the drawer above). Each drawer has a rail handle on its front top edge."""
    n = int(m.params.get("n", 2))
    D, Wd, Hu, t = DRAWER_D, DRAWER_WD, DRAWER_HU, 0.012
    W = n * Wd + 2 * t
    b = spec.worldbody.add_body(name="drawers", pos=[f.x, f.y, f.z], quat=f.quat())
    wood = [0.86, 0.84, 0.80, 1]
    _box(b, "drawers_back", (t / 2, W / 2, Hu / 2), (-D / 2 + t / 2, 0, Hu / 2), wood)
    _box(b, "drawers_left", (D / 2, t / 2, Hu / 2), (0, W / 2 - t / 2, Hu / 2), wood)
    _box(b, "drawers_right", (D / 2, t / 2, Hu / 2), (0, -W / 2 + t / 2, Hu / 2), wood)
    _box(b, "drawers_top", (D / 2, W / 2, t / 2), (0, 0, Hu - t / 2), wood)
    _box(b, "drawers_bottom", (D / 2, W / 2, t / 2), (0, 0, t / 2), wood)
    for k in range(1, n):
        _box(b, f"drawers_div_{k}", (D / 2 - 0.01, 0.003, Hu / 2 - t), (-0.01, -W / 2 + t + k * Wd, Hu / 2), wood)
    z0 = t + 0.002
    for k in range(n):
        yk = -W / 2 + t + (k + 0.5) * Wd
        d = b.add_body(name=f"drawer_{k}", pos=[0, yk, z0])
        d.add_joint(name=f"drawer_{k}_slide", type=SLIDE, axis=[1, 0, 0], range=[0, DRAWER_RANGE],
                    damping=8.0, frictionloss=1.5, armature=0.05)
        inner_hx, inner_hy = D / 2 - 0.02, Wd / 2 - 0.016
        _open_box(d, f"drawer_{k}", (inner_hx, inner_hy), 0.045, 0.008, (0, 0, 0), [0.80, 0.78, 0.74, 1])
        face_top = Hu - z0  # flush with the unit top
        _box(d, f"drawer_{k}_face", (0.008, Wd / 2 - 0.004, (face_top + 0.010) / 2),
             (D / 2 + 0.010, 0, (face_top - 0.010) / 2), [0.35, 0.50, 0.62, 1])
        _rail(d, f"drawer_{k}", D / 2 + 0.012, face_top, 0.05)
        d.add_site(name=f"drawer_{k}_handle_site", pos=[D / 2 + 0.012, 0, face_top + 0.03], size=[0.01, 0, 0])
        info.drawer_joints[f"drawers/{k}"] = f"drawer_{k}_slide"
        _add_place(info, f"drawers/{k}/handle", f, (D / 2 + 0.012, yk, Hu + 0.03), f"drawer_{k}")
        _add_place(info, f"drawers/{k}/inside", f, (0.0, yk, z0 + 0.008), f"drawer_{k}")
    info.places["drawers/height"] = (np.array([0, 0, Hu]), None)


# Cabinet 0.16 m tall (earlier: 0.22): with the basket out, the wrist of a hand placing a cup in
# the front row passes above the face and its rail.
DW_D, DW_W, DW_H = 0.34, 0.40, 0.16
DW_RANGE = 0.26
# Cup slots in the basket (basket frame): two rows near the front so they are clear of the
# cabinet top once the basket is out about 0.2 m (earlier at +-0.07 m, under the top),
# and 1.2 cm from the basket's front wall. Both rows are clear of the cabinet top for an opening
# >= 0.20 m (back row) or >= 0.11 m (front row).
DW_SLOTS = [(lx, ly) for lx in (0.005, 0.10) for ly in (-0.11, 0.0, 0.11)]


def _dishwasher(spec, info, f: Frame2D, m: Module):
    """Drawer-type dishwasher (like a DishDrawer): pull the basket out, load from above, push, press start."""
    D, W, H, t = DW_D, DW_W, DW_H, 0.012
    b = spec.worldbody.add_body(name="dishwasher", pos=[f.x, f.y, f.z], quat=f.quat())
    _box(b, "dw_back", (t / 2, W / 2, H / 2), (-D / 2 + t / 2, 0, H / 2), STEEL)
    _box(b, "dw_left", (D / 2, t / 2, H / 2), (0, W / 2 - t / 2, H / 2), STEEL)
    _box(b, "dw_right", (D / 2, t / 2, H / 2), (0, -W / 2 + t / 2, H / 2), STEEL)
    _box(b, "dw_top", (D / 2, W / 2, t / 2), (0, 0, H - t / 2), STEEL)
    _box(b, "dw_bottom", (D / 2, W / 2, t / 2), (0, 0, t / 2), STEEL)
    # Control strip with a start button on the top front edge, on the side (clear of the rail).
    _box(b, "dw_panel", (0.025, 0.06, 0.006), (D / 2 - 0.03, -0.12, H + 0.006), DARK)
    _box(b, "dw_led", (0.006, 0.02, 0.002), (D / 2 - 0.03, -0.075, H + 0.013), [0.1, 0.6, 1.0, 1], collide=False)
    b.add_geom(name="dw_button", type=CYL, size=[0.016, 0.006], pos=[D / 2 - 0.03, -0.13, H + 0.016], rgba=[0.15, 0.75, 0.25, 1])
    _add_place(info, "dishwasher/button", f, (D / 2 - 0.03, -0.13, H + 0.022), "dishwasher")
    z0 = t + 0.002
    basket = b.add_body(name="dw_basket", pos=[0, 0, z0])
    basket.add_joint(name="dw_basket_slide", type=SLIDE, axis=[1, 0, 0], range=[0, DW_RANGE],
                     damping=10.0, frictionloss=2.0, armature=0.05)
    bhx, bhy = D / 2 - 0.02, W / 2 - 0.025
    _open_box(basket, "dw_basket", (bhx, bhy), 0.07, 0.008, (0, 0, 0), [0.55, 0.57, 0.60, 1])
    # A peg behind each slot (decorative rack, clear of the fingers).
    for k, (lx, ly) in enumerate(DW_SLOTS):
        basket.add_geom(name=f"dw_peg_{k}", type=CYL, size=[0.004, 0.03],
                        pos=[lx - 0.055, ly, 0.035], rgba=[0.3, 0.3, 0.35, 1])
    face_top = H - z0
    _box(basket, "dw_face", (0.008, W / 2 - 0.006, (face_top + 0.010) / 2), (D / 2 + 0.010, 0, (face_top - 0.010) / 2), STEEL)
    _rail(basket, "dw", D / 2 + 0.012, face_top, 0.08)
    info.drawer_joints["dishwasher/basket"] = "dw_basket_slide"
    _add_place(info, "dishwasher/handle", f, (D / 2 + 0.012, 0, H + 0.03), "dw_basket")
    for k, (lx, ly) in enumerate(DW_SLOTS):
        _add_place(info, f"dishwasher/slot_{k}", f, (lx, ly, z0 + 0.008), "dw_basket")


def _landing(spec, info, f: Frame2D, m: Module):
    b = spec.worldbody.add_body(name="landing", pos=[f.x, f.y, f.z], quat=f.quat())
    # Tray 0.34 x 0.40 m (earlier: 0.30 x 0.36) for a wider dish grid (fingers between neighbours),
    # rim 1.2 cm (earlier: 2 cm) so a hand held flat can reach a cup over it.
    _open_box(b, "landing", (0.17, 0.20), 0.012, 0.01, (0, 0, 0), [0.28, 0.30, 0.33, 1])
    _add_place(info, "landing/center", f, (0, 0, 0.01), "landing")


def _drying_mat(spec, info, f: Frame2D, m: Module):
    b = spec.worldbody.add_body(name="drying_mat", pos=[f.x, f.y, f.z], quat=f.quat())
    _box(b, "mat", (0.15, 0.18, 0.004), (0, 0, 0.004), [0.25, 0.55, 0.80, 1])
    for k in range(5):
        _box(b, f"mat_ridge_{k}", (0.15, 0.004, 0.003), (0, -0.14 + 0.07 * k, 0.010), [0.20, 0.45, 0.70, 1], collide=False)
    _add_place(info, "drying_mat/center", f, (0, 0, 0.008), "drying_mat")


def _bins(spec, info, f: Frame2D, m: Module):
    b = spec.worldbody.add_body(name="bins", pos=[f.x, f.y, f.z], quat=f.quat())
    for k, (label, rgba) in enumerate((("trash", [0.25, 0.25, 0.25, 1]), ("recycling", [0.85, 0.75, 0.10, 1]))):
        ly = -0.09 + 0.18 * k
        _open_box(b, f"bin_{label}", (0.08, 0.08), 0.16, 0.008, (0, ly, 0), rgba)
        _add_place(info, f"bins/{label}", f, (0, ly, 0.01), "bins")


MODULE_BUILDERS = {"sink": _sink, "drawers": _drawers, "dishwasher": _dishwasher,
                   "landing": _landing, "drying_mat": _drying_mat, "bins": _bins}


# ---------------------------------------------------------------- items

def _container_frame(info: BuildInfo, where: str) -> tuple[Frame2D, tuple[float, float, float]]:
    """Frame and local base point where an item of this container rests."""
    if where.startswith("drawers/"):
        f = info.module_frames["drawers"]
        k = int(where.split("/")[1])
        _, yk, zk = _local_of(info, f"drawers/{k}/inside")
        return f, (0.0, yk, zk)
    if where == "landing":
        return info.module_frames["landing"], (0.0, 0.0, 0.01)
    if where == "sink_side":
        return info.module_frames["sink"], (0.0, 0.0, 0.012)
    raise ValueError(where)


def _local_of(info: BuildInfo, place: str) -> np.ndarray:
    """Module-frame coordinates of a place recorded at build time."""
    kind = place.split("/")[0]
    f = info.module_frames[kind]
    p = info.places[place][0]
    c, s_ = math.cos(f.yaw), math.sin(f.yaw)
    dx, dy = p[0] - f.x, p[1] - f.y
    return np.array([c * dx + s_ * dy, -s_ * dx + c * dy, p[2] - f.z])


def _add_item(spec, info, it: Item):
    f, base = _container_frame(info, it.where)
    geoms, mass = ITEM_GEOMS[it.kind]
    quat = np.array(f.quat())
    lying = it.kind == "spray_bottle" and it.where.startswith("drawers/")
    if lying:  # too tall for a drawer: it lies on its narrow side along the drawer, head to the back, nozzle sideways
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, quat, np.array([math.cos(-math.pi / 4), 0, math.sin(-math.pi / 4), 0]))
        quat = q
    hh = 0.030 if lying else item_half_height(it.kind)
    p = f.to_world((base[0] + it.pos[0], base[1] + it.pos[1], base[2] + hh + 0.002))
    b = spec.worldbody.add_body(name=it.name, pos=list(p), quat=list(quat))
    b.add_freejoint(name=f"{it.name}_free")
    collide_geoms = [g for g in geoms if g[4]]
    split = ITEM_MASS_SPLIT.get(it.kind, [1.0 / len(collide_geoms)] * len(collide_geoms))
    ci = 0
    for k, (t, size, lp, rgba, collides) in enumerate(geoms):
        g = b.add_geom(name=f"{it.name}_g{k}", type=t, size=list(size), pos=list(lp), rgba=list(rgba),
                       friction=[1.2, 0.01, 0.001], condim=4)
        if collides:
            g.mass = mass * split[ci]
            ci += 1
            if it.kind == "sponge":
                # A sponge is soft (about 1 cm of give under a few newtons), not rigid, and
                # rough (friction 2: a soft contact creeps out of the fingers otherwise).
                g.solref = [0.05, 1.0]
                g.solimp = [0.2, 0.9, 0.012, 0.5, 2.0]
                g.friction = [2.0, 0.02, 0.002]
        else:
            g.contype = g.conaffinity = 0
            g.mass = 0.0
    info.item_bodies[it.name] = it.name


# ---------------------------------------------------------------- robot and room

def _robot(spec, s: SiteSpec):
    m = s.mount
    top = s.worktop_height + m.height_offset
    ped = spec.worldbody.add_body(name="pedestal", pos=[m.x, m.y, 0])
    _box(ped, "pedestal_column", (0.11, 0.11, (top - 0.03) / 2), (0, 0, (top - 0.03) / 2), [0.30, 0.32, 0.36, 1])
    plate = ped.add_body(name="mount_plate", pos=[0, 0, top - 0.015], quat=list(mount_quat(m)))
    _box(plate, "mount_plate_geom", (0.13, 0.13, 0.015), (0, 0, 0), [0.45, 0.47, 0.50, 1])
    plate.add_site(name="imu", pos=[0, 0, 0.0], size=[0.01, 0, 0])
    # The robot's own camera, fixed to the mount: it tilts with the robot.
    # Mast moved from x = -0.20 to -0.30 on a bracket (the elbow hit it when the arm
    # reached a wing module).
    _box(plate, "eye_bracket", (0.085, 0.012, 0.01), (-0.215, 0.0, 0.0), [0.30, 0.32, 0.36, 1])
    _box(plate, "eye_mast", (0.012, 0.012, 0.52), (-0.30, 0.0, 0.53), [0.30, 0.32, 0.36, 1])
    plate.add_camera(name="robot_eye", pos=list(EYE_POS_PLATE),
                     quat=_look_at_quat(EYE_POS_PLATE, EYE_TARGET_PLATE), fovy=EYE_FOVY)

    panda = mujoco.MjSpec.from_file(str(PANDA_XML))
    for k in list(panda.keys):
        panda.delete(k)
    panda.body("hand").add_site(name="tcp", pos=[0, 0, 0.1034], size=[0.008, 0, 0])
    panda.body("hand").add_camera(name="wrist_cam", pos=list(WRIST_CAM_POS_HAND),
                                  quat=list(WRIST_CAM_QUAT_HAND), fovy=WRIST_CAM_FOVY)
    # Menagerie's gripper squeezes ~2.6 N; make it ~26 N (closer to a real Franka Hand).
    a = panda.actuator("actuator8")
    a.gainprm[0] = 0.1568627451
    a.biasprm[1], a.biasprm[2] = -1000.0, -100.0
    # The real Franka controller compensates the arm's own weight; do the same here
    # (a held object's weight is not compensated).
    for body in panda.bodies:
        if body.name != "world":
            body.gravcomp = 1.0
    frame = plate.add_frame(pos=[0, 0, LINK0_ON_PLATE_Z])
    frame.attach_body(panda.body("link0"), "panda/", "")
    spec.add_sensor(name="imu_acc", type=mujoco.mjtSensor.mjSENS_ACCELEROMETER,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")


def _room(spec, s: SiteSpec):
    H = s.worktop_height
    rgb = list(WORKTOP_COLORS[s.worktop_color]) + [1]
    cab = [0.92, 0.92, 0.90, 1]
    wb = spec.worldbody
    wb.add_geom(name="floor", type=PLANE, size=[3, 3, 0.05], rgba=[0.62, 0.60, 0.57, 1])
    # U-shaped worktop: front run and two wings, cabinets below. Inner edges at x = 0.29 and
    # y = +-0.29 (used by the calibration touches). Outer edges pushed out to carry the
    # 0.70 m slot ring (earlier: front run to x = 0.81, wings to |y| = 0.86).
    for name, c, half in (("front", (0.61, 0.0), (0.31, 0.98)),
                          ("wing_l", (-0.175, 0.64), (0.475, 0.34)),
                          ("wing_r", (-0.175, -0.64), (0.475, 0.34))):
        _box(wb, f"cabinet_{name}", (half[0], half[1], (H - 0.04) / 2), (c[0], c[1], (H - 0.04) / 2), cab)
        _box(wb, f"worktop_{name}", (half[0] + 0.01, half[1] + 0.01, 0.02), (c[0], c[1], H - 0.02), rgb)
    for name, pos, half in (("wall_front", (0.95, 0, 1.2), (0.02, 1.03, 1.2)),
                            ("wall_left", (0.125, 1.01, 1.2), (0.845, 0.02, 1.2)),
                            ("wall_right", (0.125, -1.01, 1.2), (0.845, 0.02, 1.2))):
        _box(wb, name, half, pos, [0.90, 0.88, 0.84, 1], collide=False)
    wb.add_light(name="sun", pos=[0.3, 0.2, 2.6], dir=[0, 0, -1], diffuse=[s.light] * 3, castshadow=True)
    wb.add_light(name="fill", pos=[-1.0, 0, 2.0], dir=[0.5, 0, -1], diffuse=[0.25, 0.25, 0.25], castshadow=False)
    for name, pos, target in (("overview", (-0.95, 0.0, 2.75), (0.30, 0.0, 0.85)),
                              ("corner", (-0.75, 0.95, 2.05), (0.40, -0.15, 0.90))):
        wb.add_camera(name=name, pos=list(pos), quat=_look_at_quat(pos, target), fovy=60)


# DECLARED CHANGE OF THE EXPERIMENTAL WORLD, applied to every condition programs A, S and the later B/C, every kitchen,
# every variant: the sliding friction between the sponge and the dishes (cup, bowl, fork, spoon) is SPONGE_DISH_FRICTION
# = 0.5, set by explicit contact pairs; every other contact is unchanged (sponge-finger stays 2.0, dish-tray 1.2). Why.
# The sponge's friction 2.0 was chosen for the GRIP (a soft contact crept out of the fingers), not as a value for
# a sponge on a dish; MuJoCo combines two geoms' frictions by their maximum, so 2.0 also applied to sponge-bowl
# contacts, above the bowl-tray 1.2: the sponge dragged the 150 g bowl instead of sliding on it. Measured in development
# staged and dev kitchens only: at 2.0 the sponge slides about 2 cm on the bowl per 3 s scrub whatever the force or
# stroke; at 1.0 it slides, but the lateral friction at the bowl's top (4.4 cm high, 3.1 cm base radius) tips the bowl
# (9 to 19 deg at 4 N, fingers then touch it) since mu h > r tips it above N = m g r / (mu h - r), about 3.5 N; at 0.5
# the bowl neither slides nor tips at any pressing force (0.5 x 4.4 < 3.1). Value: a wet, soapy sponge on a glazed dish,
# the setting of a real scrub, is expected well below the dry grip value; no measured value is available to us
# (knowledge base 06_Savoir_Physical_AI: "soapy films reduce friction by an unknown factor, no source found";
# Engineering ToolBox lists dry and lubricated coefficients, not re-checked in this session, no network). 0.5 is
# therefore a DECLARED MODELLING ASSUMPTION, not a measurement, chosen for a physical reason (no tipping, sliding before
# dragging) on development seeds only; runs at 1.0 and with the world before the change are kept as sensitivity rows.
# None restores the world before this change (comparison runs only). The pairs keep the contact parameters MuJoCo would
# mix for these geom pairs (solref, solimp averaged, condim and spin/roll friction maxima; a test compares with the
# dynamic contact).
SPONGE_DISH_FRICTION = 0.5
DISH_KINDS = ("cup", "bowl", "fork", "spoon")


def _sponge_dish_pairs(spec, s: SiteSpec, mu: float):
    def gp(name):
        g = spec.geom(name)
        return (np.array(g.friction, float), np.array(g.solref, float), np.array(g.solimp, float), int(g.condim),
                float(g.solmix))
    for sp in (it for it in s.items if it.kind == "sponge"):
        for dn in (it for it in s.items if it.kind in DISH_KINDS):
            for gs in (f"{sp.name}_g{k}" for k, x in enumerate(ITEM_GEOMS["sponge"][0]) if x[4]):
                for gd in (f"{dn.name}_g{k}" for k, x in enumerate(ITEM_GEOMS[dn.kind][0]) if x[4]):
                    f1, r1, i1, c1, m1 = gp(gs)
                    f2, r2, i2, c2, m2 = gp(gd)
                    w = m1 / (m1 + m2) if m1 + m2 > 0 else 0.5
                    p = spec.add_pair(geomname1=gs, geomname2=gd)
                    p.condim = max(c1, c2)
                    fr = np.maximum(f1, f2)
                    p.friction = [mu, mu, fr[1], fr[2], fr[2]]
                    p.solref = list(w * r1 + (1 - w) * r2)
                    p.solimp = list(w * i1 + (1 - w) * i2)


def build(s: SiteSpec, sponge_dish_friction: float | None = SPONGE_DISH_FRICTION) -> tuple[mujoco.MjModel, BuildInfo]:
    spec = mujoco.MjSpec()
    spec.modelname = f"botnes_site_{s.seed}"
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    # Friction settings recommended for grasping (elliptic cones, high impratio). With the
    # defaults, a held cup crept out of the fingers in about 2 s (soft-constraint slip).
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0
    # No-slip post-solver: removes the slow creep of soft friction (a held cup rotated a few
    # degrees per second in the fingers during long carries).
    spec.option.noslip_iterations = 3
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 960
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.35, 0.35, 0.35]
    info = BuildInfo(spec=s)
    _room(spec, s)
    _robot(spec, s)
    for m in s.modules:
        sx, sy = slot_xy(m.slot, m.kind)
        f = Frame2D(sx, sy, s.worktop_height, math.atan2(-sy, -sx))
        info.module_frames[m.kind] = f
        MODULE_BUILDERS[m.kind](spec, info, f, m)
    for it in s.items:
        _add_item(spec, info, it)
    if sponge_dish_friction is not None:
        _sponge_dish_pairs(spec, s, float(sponge_dish_friction))
    info.sponge_dish_friction = sponge_dish_friction
    model = spec.compile()
    return model, info
