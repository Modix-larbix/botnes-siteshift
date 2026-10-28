"""Development-only occlusion worlds, second set: version "occlusion-dev-2". Bench side, never a validation world.

It adds harder cases than occlusion-dev-1 (siteagent/world_dev/occlusion.py, unchanged): uncatalogued boards inside a
drawer, a sponge on its long edge, partial and surface covers. Its machinery (refusals, instance, applied) is
occlusion.py's.

Modes (the sponge's declared drawer is k):
- board_open (correct): an uncatalogued fixed board, part of drawer k's body, centered over the sponge, 4.0 cm above
  the drawer floor, with no front wall; board_open_48: the same board 4.8 cm above the floor; board_open_absent: the
  board with no sponge;
- edge_back (correct): no occluder; the sponge stands on its long edge against the back of drawer k, under the unit top;
- towel_offset (correct): towel_over's towel shifted across the drawer (a partial cover);
- towel_over_tray (correct): the sponge moved to a free cell of the landing tray with the towel on top of it.
Placement is done at build time through qpos0; post_settle_check measures the result bench side and raises
occlusion.WorldDevError when the world is not what the mode declares. A spec is "occlusion-dev-2:<mode>";
module_for(spec) gives the module of a spec of either version.
"""
from __future__ import annotations

import contextlib
import copy
import re

import numpy as np

from . import occlusion as O1
from .occlusion import (DEV_KITCHENS, DRAWER_FLOOR_TOP, OCCLUDER_NAME, SPAWN_CLEARANCE, SPONGE_REST_ABOVE_FLOOR,
                        TOWEL_ABOVE_SPONGE, WorldDevError)

OCCLUSION_DEV_VERSION = "occlusion-dev-2"
WORLD_DEV_SPEC = re.compile(r"^occlusion-dev-2:([a-z_0-9]+)$")
MODES = ("board_open", "board_open_48", "board_open_absent", "edge_back", "towel_offset", "towel_over_tray")
MODE_VARIANT = {"board_open": "correct", "board_open_48": "correct", "board_open_absent": "absent",
                "edge_back": "correct", "towel_offset": "correct", "towel_over_tray": "correct"}
BOARD_MODES = ("board_open", "board_open_48", "board_open_absent")
TOWEL_MODES = ("towel_offset", "towel_over_tray")
# Frozen geometry (drawer body frame as in occlusion.py: x along the drawer, + toward the robot).
BOARD_HALF = (0.048, 0.033, 0.002)
BOARD_BOTTOM_ABOVE_FLOOR = {"board_open": 0.040, "board_open_48": 0.048, "board_open_absent": 0.040}
BOARD_RGBA = (0.80, 0.78, 0.74, 1.0)  # the drawer's own colour
EDGE_BACK_POS = (-0.114, 0.0)  # the edge case
EDGE_SPONGE_HALF_UP = 0.030  # the sponge's half width: its centre's height above the floor when on its long edge
TOWEL_OFFSET_LAT_M = 0.009
TRAY_CELLS = ((-0.08, -0.14), (-0.08, 0.14), (0.08, 0.0), (-0.08, 0.0), (0.08, -0.14), (0.08, 0.14))  # landing frame
TRAY_FLOOR_ABOVE_BODY = 0.01  # the landing tray's resting surface above its body origin (world.build, "landing")
# post-settle tolerances (bench side)
TOL_REST_Z = 0.006
TOWEL_OVER_DZ = (0.028, 0.042)
TOWEL_OFFSET_DXY = (0.004, 0.02)
TOWEL_OVER_DXY = 0.02
EDGE_ALONG_MAX = -0.09
EDGE_UPRIGHT_MAX_TILT_DEG = 20.0  # angle between the sponge's local y (its 6 cm side) and the drawer's vertical
MAX_TILT_DEG = {"sponge": 20.0, OCCLUDER_NAME: 15.0}


def parse(spec: str) -> str:
    m = WORLD_DEV_SPEC.match(str(spec))
    if not m or m.group(1) not in MODES:
        raise WorldDevError(f"world_dev must be {OCCLUSION_DEV_VERSION}:<mode> with a mode among {MODES} "
                            f"(got {spec!r})")
    return m.group(1)


def spec_of(mode: str) -> str:
    if mode not in MODES:
        raise WorldDevError(f"unknown occlusion-dev-2 mode {mode!r}")
    return f"{OCCLUSION_DEV_VERSION}:{mode}"


def module_for(spec: str):
    """The module that builds a world_dev spec: occlusion (occlusion-dev-1) or this one (occlusion-dev-2)."""
    s = str(spec)
    if s.startswith(OCCLUSION_DEV_VERSION + ":"):
        import sys  # noqa: PLC0415
        return sys.modules[__name__]
    return O1


def refusal(kitchen: str, mode: str, noise_seeds=(), registry: dict | None = None, out_dir=None,
            variant: str | None = None, s5_root=None) -> str | None:
    """occlusion.refusal with this module's modes (the registry, noise-block and output-folder rules are occlusion.py's,
    applied through an occlusion-dev-1 mode of the same variant)."""
    if mode not in MODES:
        return f"refused: unknown occlusion mode {mode!r}"
    if variant is not None and variant != MODE_VARIANT[mode]:
        return f"refused: mode {mode} runs on variant {MODE_VARIANT[mode]}, not {variant!r}"
    proxy = "towel_over" if MODE_VARIANT[mode] == "correct" else "towel_over_absent"
    return O1.refusal(kitchen, proxy, noise_seeds, registry, out_dir, variant=MODE_VARIANT[mode], s5_root=s5_root)


def manifest_refusal(manifest: dict) -> str | None:
    """occlusion.manifest_refusal for manifests naming specs of either version."""
    jobs = [j for j in manifest.get("jobs", []) if isinstance(j, dict) and "world_dev" in j]
    named = bool(jobs) or "world_dev" in manifest
    if not named:
        return None
    if manifest.get("kind") == "validation":
        return "refused: a validation manifest never names a development world (world_dev)"
    if not (manifest.get("kind") == "batch" and manifest.get("development_tuning") is True):
        return "refused: world_dev jobs run only in a development batch (kind batch, development_tuning true)"
    for j in jobs:
        try:
            module_for(j["world_dev"]).parse(j["world_dev"])
        except WorldDevError as e:
            return f"refused: {e}"
        if str(j.get("kitchen")) not in DEV_KITCHENS:
            return f"refused: world_dev kitchen {j.get('kitchen')!r} is not a development drawer kitchen"
    return None


# ------------------------------------------------------------------------------------------------ the instance
def instance(kitchen: str, variant: str, mode: str, base_build_instance):
    """(truth, dossier spec, change log, setup log, target) with the mode's changes, and the sponge's declared drawer
    k. The dossier spec is never changed."""
    why = refusal(kitchen, mode, variant=variant)
    if why:
        raise WorldDevError(why)
    inst = base_build_instance(kitchen, variant)
    if inst is None:
        raise WorldDevError(f"kitchen {kitchen} variant {variant}: not applicable")
    truth, dspec, change_log, setup_log, target = inst
    truth = copy.deepcopy(truth)
    k = O1.declared_sponge_drawer(dspec)
    from ..world.spec import DRAWER_ITEM_POS, Item  # noqa: PLC0415
    sponge = next((i for i in truth.items if i.kind == "sponge"), None)
    if MODE_VARIANT[mode] == "correct":
        if sponge is None or sponge.where != f"drawers/{k}":
            raise WorldDevError(f"{mode}: the sponge is not in its declared drawer drawers/{k}")
    elif sponge is not None:
        raise WorldDevError(f"{mode}: the absent variant still has a sponge")
    others = [i for i in truth.items if i.where == f"drawers/{k}" and i.kind != "sponge"]
    if others:
        raise WorldDevError(f"{mode}: drawers/{k} holds another item ({others[0].kind}): one item per drawer")
    entry = {"change": "world_dev", "version": OCCLUSION_DEV_VERSION, "mode": mode, "drawer": f"drawers/{k}"}
    if mode in BOARD_MODES:
        centre = tuple(sponge.pos) if sponge is not None else tuple(DRAWER_ITEM_POS["sponge"])
        entry["board"] = {"half": list(BOARD_HALF), "bottom_above_floor_m": BOARD_BOTTOM_ABOVE_FLOOR[mode],
                          "centre_in_drawer": list(centre), "front_wall": False}
    if mode == "edge_back":
        sponge.pos = EDGE_BACK_POS
        entry["sponge"] = {"pose": "on its long edge, long axis across the drawer", "pos_in_drawer": list(EDGE_BACK_POS)}
    if mode == "towel_offset":
        truth.items.append(Item(OCCLUDER_NAME, "towel", sponge.where, tuple(sponge.pos)))
        entry["towel"] = {"name": OCCLUDER_NAME, "over": "sponge", "offset_across_m": TOWEL_OFFSET_LAT_M}
    if mode == "towel_over_tray":
        used = [tuple(i.pos) for i in truth.items if i.where == "landing"]
        cell = next((c for c in TRAY_CELLS if all(abs(c[0] - u[0]) > 0.03 or abs(c[1] - u[1]) > 0.03 for u in used)),
                    None)
        if cell is None:
            raise WorldDevError(f"{mode}: no free cell on the landing tray")
        sponge.where, sponge.pos = "landing", cell
        truth.items.append(Item(OCCLUDER_NAME, "towel", "landing", cell))
        entry["sponge"] = {"moved_to": "landing", "pos_in_tray": list(cell)}
        entry["towel"] = {"name": OCCLUDER_NAME, "over": "sponge", "where": "landing"}
    return (truth, dspec, list(change_log) + [entry], setup_log, target), k


# ------------------------------------------------------------------------------------------------ the build
def _board_centre(site, k: int):
    from ..world.spec import DRAWER_ITEM_POS  # noqa: PLC0415
    sp = next((i for i in site.items if i.kind == "sponge"), None)
    return tuple(sp.pos) if sp is not None and sp.where == f"drawers/{k}" else tuple(DRAWER_ITEM_POS["sponge"])


def _qpos(model, name):
    j = model.joint(f"{name}_free")
    return int(j.qposadr[0])


def _drawer_floor_and_rot(model, k: int):
    import mujoco  # noqa: PLC0415
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    b = model.body(f"drawer_{k}").id
    R = d.xmat[b].reshape(3, 3).copy()
    return d.xpos[b] + R @ np.array([0.0, 0.0, DRAWER_FLOOR_TOP]), R, d


def _quat_of(R) -> np.ndarray:
    import mujoco  # noqa: PLC0415
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel())
    return q


def build_world(site, mode: str | None = None, cover_drawer: int | None = None, **world_kw):
    """(model, info) of world.build.build(site, **world_kw) plus the mode's build-time changes. mode None: exactly
    world.build.build (as occlusion.build_world)."""
    B = O1._world_build()
    if mode is None:
        return B.build(site, **world_kw)
    if mode not in MODES:
        raise WorldDevError(f"unknown occlusion-dev-2 mode {mode!r}")
    if cover_drawer is None:
        raise WorldDevError(f"{mode}: the sponge's declared drawer is required")
    k = int(cover_drawer)
    if mode in BOARD_MODES:
        orig = B.MODULE_BUILDERS["drawers"]
        cx, cy = _board_centre(site, k)
        z = DRAWER_FLOOR_TOP + BOARD_BOTTOM_ABOVE_FLOOR[mode] + BOARD_HALF[2]

        def drawers_with_board(spec, info, f, m, _orig=orig):
            _orig(spec, info, f, m)
            B._box(spec.body(f"drawer_{k}"), f"drawer_{k}_board", BOARD_HALF, (cx, cy, z), list(BOARD_RGBA))
        B.MODULE_BUILDERS["drawers"] = drawers_with_board
        try:
            model, info = B.build(site, **world_kw)
        finally:
            B.MODULE_BUILDERS["drawers"] = orig
    else:
        model, info = B.build(site, **world_kw)
    sponge = next((i for i in site.items if i.kind == "sponge"), None)
    if sponge is None:
        return model, info
    sa = _qpos(model, sponge.name)
    if mode in ("board_open", "board_open_48", "towel_offset"):
        floor, R, _d = _drawer_floor_and_rot(model, k)
        model.qpos0[sa + 2] = floor[2] + SPONGE_REST_ABOVE_FLOOR + SPAWN_CLEARANCE
        if mode == "towel_offset":
            ta = _qpos(model, OCCLUDER_NAME)
            sp = model.qpos0[sa:sa + 3].copy()
            model.qpos0[ta:ta + 3] = sp + R @ np.array([0.0, TOWEL_OFFSET_LAT_M, 0.0]) + np.array(
                [0.0, 0.0, TOWEL_ABOVE_SPONGE])
            model.qpos0[ta + 3:ta + 7] = model.qpos0[sa + 3:sa + 7]
    elif mode == "edge_back":
        floor, R, _d = _drawer_floor_and_rot(model, k)
        # local x (9 cm) -> across the drawer, local y (6 cm) -> up, local z (3.6 cm) -> along
        RF = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        model.qpos0[sa + 2] = floor[2] + EDGE_SPONGE_HALF_UP + SPAWN_CLEARANCE
        model.qpos0[sa + 3:sa + 7] = _quat_of(R @ RF)
    elif mode == "towel_over_tray":
        import mujoco  # noqa: PLC0415
        d = mujoco.MjData(model)
        mujoco.mj_forward(model, d)
        lb = model.body("landing").id
        tray_floor_z = float((d.xpos[lb] + d.xmat[lb].reshape(3, 3) @ np.array([0.0, 0.0, TRAY_FLOOR_ABOVE_BODY]))[2])
        ta = _qpos(model, OCCLUDER_NAME)
        model.qpos0[sa + 2] = tray_floor_z + SPONGE_REST_ABOVE_FLOOR + SPAWN_CLEARANCE
        sp = model.qpos0[sa:sa + 3].copy()
        model.qpos0[ta:ta + 3] = sp + np.array([0.0, 0.0, TOWEL_ABOVE_SPONGE])
        model.qpos0[ta + 3:ta + 7] = model.qpos0[sa + 3:sa + 7]
    return model, info


# ------------------------------------------------------------------------------------------------ post-settle check
def _in_body(sim, body_frame: str, body: str) -> np.ndarray:
    m, d = sim.model, sim.data
    b = m.body(body_frame).id
    return d.xmat[b].reshape(3, 3).T @ (d.xpos[m.body(body).id] - d.xpos[b])


def _axis_tilt_deg(sim, frame_body: str, body: str, axis: int) -> float:
    m, d = sim.model, sim.data
    a = d.xmat[m.body(body).id].reshape(3, 3)[:, axis]
    z = d.xmat[m.body(frame_body).id].reshape(3, 3)[:, 2]
    return float(np.degrees(np.arccos(min(1.0, abs(float(a @ z))))))


def post_settle_check(sim, mode: str, k: int) -> dict:
    """Bench side, right after the Sim is built and settled: the sponge, the towel and the board where the mode
    declares them. Raises WorldDevError otherwise; returns the measured values."""
    B = O1._world_build()
    inner_hx, inner_hy = B.DRAWER_D / 2 - 0.02, B.DRAWER_WD / 2 - 0.016
    items = {i.name: i for i in sim.site.items}
    sponge = next((i for i in sim.site.items if i.kind == "sponge"), None)
    out = {"version": OCCLUSION_DEV_VERSION, "mode": mode, "drawer": f"drawers/{k}"}
    problems = []
    geoms = {sim.model.geom(g).name for g in range(sim.model.ngeom)}
    has_board = f"drawer_{k}_board" in geoms
    out["board_present"] = has_board
    if (mode in BOARD_MODES) != has_board:
        problems.append(f"board present {has_board} in mode {mode}")
    if any(g.endswith("_cover_plate") or g.endswith("_cover_front") for g in geoms):
        problems.append("an occlusion-dev-1 compartment is present")
    if (MODE_VARIANT[mode] == "correct") != (sponge is not None):
        problems.append("sponge presence does not match the mode")
    if (mode in TOWEL_MODES) != (OCCLUDER_NAME in items):
        problems.append("occluding towel presence does not match the mode")
    drawer_body = f"drawer_{k}"
    if sponge is not None and mode != "towel_over_tray":
        sp = _in_body(sim, drawer_body, sponge.name)
        out["sponge_in_drawer_m"] = [round(float(x), 4) for x in sp]
        rest = float(sp[2]) - DRAWER_FLOOR_TOP
        out["sponge_centre_above_floor_m"] = round(rest, 4)
        if not (abs(sp[0]) <= inner_hx and abs(sp[1]) <= inner_hy):
            problems.append(f"sponge outside drawer {k}'s floor ({sp.round(4).tolist()})")
        if mode == "edge_back":
            out["sponge_side_tilt_deg"] = round(_axis_tilt_deg(sim, drawer_body, sponge.name, 1), 2)
            if out["sponge_side_tilt_deg"] > EDGE_UPRIGHT_MAX_TILT_DEG:
                problems.append(f"sponge not on its long edge (local y {out['sponge_side_tilt_deg']} deg from vertical)")
            if abs(rest - EDGE_SPONGE_HALF_UP) > TOL_REST_Z:
                problems.append(f"sponge on its edge not resting (centre {rest:.4f} m above the floor)")
            if float(sp[0]) > EDGE_ALONG_MAX:
                problems.append(f"sponge not at the back (along {float(sp[0]):.4f})")
        else:
            out["sponge_tilt_deg"] = round(_axis_tilt_deg(sim, drawer_body, sponge.name, 2), 2)
            if out["sponge_tilt_deg"] > MAX_TILT_DEG["sponge"]:
                problems.append(f"sponge tilted {out['sponge_tilt_deg']} degrees")
            if abs(rest - SPONGE_REST_ABOVE_FLOOR) > TOL_REST_Z:
                problems.append(f"sponge not resting on the floor (centre {rest:.4f} m above it)")
        if has_board and mode != "board_open_absent":
            g = sim.model.geom(f"drawer_{k}_board").id
            m, d = sim.model, sim.data
            b = m.body(drawer_body).id
            gz = float((d.xmat[b].reshape(3, 3).T @ (d.geom_xpos[g] - d.xpos[b]))[2])
            board_bottom = gz - BOARD_HALF[2]
            top = float(sp[2]) + 0.022
            out["board_bottom_above_floor_m"] = round(board_bottom - DRAWER_FLOOR_TOP, 4)
            out["board_clearance_above_sponge_top_m"] = round(board_bottom - top, 4)
            if top >= board_bottom:
                problems.append("sponge top not under the board")
            gxy = (d.xmat[b].reshape(3, 3).T @ (d.geom_xpos[g] - d.xpos[b]))[:2]
            if float(np.hypot(*(gxy - sp[:2]))) > 0.01:
                problems.append("board not over the sponge")
    if sponge is not None and mode == "towel_over_tray":
        sp = _in_body(sim, "landing", sponge.name)
        out["sponge_in_tray_m"] = [round(float(x), 4) for x in sp]
        rest = float(sp[2]) - TRAY_FLOOR_ABOVE_BODY
        out["sponge_centre_above_tray_m"] = round(rest, 4)
        out["sponge_tilt_deg"] = round(_axis_tilt_deg(sim, "landing", sponge.name, 2), 2)
        if abs(rest - SPONGE_REST_ABOVE_FLOOR) > TOL_REST_Z:
            problems.append(f"sponge not resting on the tray (centre {rest:.4f} m above it)")
        if not (abs(sp[0]) <= 0.15 and abs(sp[1]) <= 0.18):
            problems.append(f"sponge outside the tray ({sp.round(4).tolist()})")
        if out["sponge_tilt_deg"] > MAX_TILT_DEG["sponge"]:
            problems.append(f"sponge tilted {out['sponge_tilt_deg']} degrees")
    if OCCLUDER_NAME in items and sponge is not None:
        frame = "landing" if mode == "towel_over_tray" else drawer_body
        tw, sp = _in_body(sim, frame, OCCLUDER_NAME), _in_body(sim, frame, sponge.name)
        out["towel_tilt_deg"] = round(_axis_tilt_deg(sim, frame, OCCLUDER_NAME, 2), 2)
        dz, dxy = float(tw[2] - sp[2]), float(np.hypot(*(tw[:2] - sp[:2])))
        out["towel_above_sponge_m"], out["towel_sponge_dxy_m"] = round(dz, 4), round(dxy, 4)
        if out["towel_tilt_deg"] > MAX_TILT_DEG[OCCLUDER_NAME]:
            problems.append(f"towel tilted {out['towel_tilt_deg']} degrees")
        lo, hi = TOWEL_OFFSET_DXY if mode == "towel_offset" else (0.0, TOWEL_OVER_DXY)
        if not TOWEL_OVER_DZ[0] <= dz <= TOWEL_OVER_DZ[1] or not lo <= dxy <= hi:
            problems.append(f"towel not on the sponge as declared (dz {dz:.4f}, dxy {dxy:.4f})")
    out["ok"] = not problems
    out["problems"] = problems
    if problems:
        raise WorldDevError(f"{OCCLUSION_DEV_VERSION}:{mode} post-settle check failed: {'; '.join(problems)}")
    return out


# ------------------------------------------------------------------------------------------------ applied()
@contextlib.contextmanager
def applied(world_dev: str, kitchen: str | None = None, registry: dict | None = None, noise_seeds=(), out_dir=None,
            s5_root=None):
    """As occlusion.applied, for occlusion-dev-2 specs (bench_task.build_instance and bench_task.Sim patched while the
    block runs, siteagent.sim.build during the Sim's construction only; the post-settle check after each Sim)."""
    mode = parse(world_dev)
    if kitchen is not None:
        why = refusal(kitchen, mode, noise_seeds, registry, out_dir, s5_root=s5_root)
        if why:
            raise WorldDevError(why)
    from .. import bench_task as BT  # noqa: PLC0415
    from .. import sim as SIM  # noqa: PLC0415
    orig_bi, orig_sim_cls, orig_build = BT.build_instance, BT.Sim, SIM.build
    state = {"version": OCCLUSION_DEV_VERSION, "mode": mode, "checks": [], "instances": [], "drawer": None}

    def build_instance(kitchen_, variant_):
        inst, k = instance(kitchen_, variant_, mode, orig_bi)
        state["drawer"] = k
        state["instances"].append({"kitchen": str(kitchen_), "variant": variant_, "drawer": f"drawers/{k}"})
        return inst

    class DevSim(orig_sim_cls):
        def __init__(self, site, *a, **kw):
            k = state["drawer"]
            if k is None:
                raise WorldDevError("development Sim built without its instance")

            def dev_build(s, **world_kw):
                return build_world(s, mode, cover_drawer=k, **world_kw)
            SIM.build = dev_build
            try:
                super().__init__(site, *a, **kw)
            finally:
                SIM.build = orig_build
            state["checks"].append(post_settle_check(self, mode, k))

    BT.build_instance, BT.Sim = build_instance, DevSim
    try:
        yield state
    finally:
        BT.build_instance, BT.Sim, SIM.build = orig_bi, orig_sim_cls, orig_build

