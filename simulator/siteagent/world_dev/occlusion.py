"""Development-only occlusion worlds, version "occlusion-dev-1". Bench side, never a validation world.

Modes, each built on the kitchen's own variant (the dossier is unchanged):
- covered_back (variant correct): the sponge's declared drawer gets a closed back compartment (a lid and a front wall,
  part of the drawer body), with the sponge under the lid: the region can never be cleared;
- covered_back_absent (variant absent): the same compartment, no sponge;
- towel_over (variant correct): a catalog towel lies on top of the sponge, centered on it, same yaw;
- towel_over_absent (variant absent): the same towel lies flat where the sponge would rest.
The towel is placed at build time (free joint qpos0) so a reset puts it back. After the Sim is built, a post-settle
check measures the sponge and the towel bench side and raises WorldDevError when the world is not what the mode
declares. The truth spec carries the towel; the change log gains a "world_dev" entry, recorded for the evaluator and
never given to the robot.

Without a mode the world is exactly world.build.build's. world/*.py is never edited: the compartment is added by
wrapping a builder while one build runs, and applied() patches bench_task while a development episode runs.

refusal() is checked before any episode: an unknown version or mode, a variant that is not the mode's, a kitchen outside
DEV_KITCHENS, a kitchen or noise seed whose seed registry entry is reserved, a noise seed of another kitchen's block, or
an output folder named real*, pilot*, val* or probe_format_*. manifest_refusal() refuses world_dev in any validation
manifest.
"""
from __future__ import annotations

import contextlib
import copy
import re
from pathlib import Path

import numpy as np

OCCLUSION_DEV_VERSION = "occlusion-dev-1"
WORLD_DEV_SPEC = re.compile(r"^occlusion-dev-1:([a-z_]+)$")
MODES = ("covered_back", "covered_back_absent", "towel_over", "towel_over_absent")
MODE_VARIANT = {"covered_back": "correct", "covered_back_absent": "absent", "towel_over": "correct",
                "towel_over_absent": "absent"}
MODE_TWIN = {"covered_back": "covered_back_absent", "covered_back_absent": "covered_back",
             "towel_over": "towel_over_absent", "towel_over_absent": "towel_over"}
COVERED_MODES = ("covered_back", "covered_back_absent")
TOWEL_MODES = ("towel_over", "towel_over_absent")
# the nine development kitchens whose dossier puts the sponge in a drawer
DEV_KITCHENS = ("reference", "31000", "41000", "41001", "41002", "41006", "41007", "41008", "41009")
OCCLUDER_NAME = "towel_occ"
RESERVED = "reserved_not_looked_at"
FORBIDDEN_OUT = re.compile(r"^(real|pilot|val)|^probe_format_")

# Frozen geometry (drawer body frame: x along the drawer, + toward the robot; z from the drawer body origin; the drawer
# floor plate's top is at DRAWER_FLOOR_TOP). Values measured on a development build: lid half sizes and centre,
# front wall half sizes and centre.
DRAWER_FLOOR_TOP = 0.008
COVER_LID_HALF, COVER_LID_POS = (0.055, 0.069, 0.002), (-0.075, 0.0, DRAWER_FLOOR_TOP + 0.050)
COVER_FRONT_HALF, COVER_FRONT_POS = (0.002, 0.069, 0.026), (-0.018, 0.0, DRAWER_FLOOR_TOP + 0.026)
COVER_RGBA = (0.80, 0.78, 0.74, 1.0)  # the drawer's own colour
SPONGE_ALONG_COVERED = -0.085  # sponge position along the drawer (the moved_jittered back bound), under the lid
# towel_over: build-time placement (qpos0). The sponge's body origin is the centre of its main box: it rests 0.014 m
# above the floor (geom half height) and its top is 0.022 m above its origin (world.build.item_half_height).
SPONGE_REST_ABOVE_FLOOR = 0.014
SPAWN_CLEARANCE = 0.0005  # the sponge spawns this far above its rest
TOWEL_ABOVE_SPONGE = 0.022 + 0.0125 + 0.001  # sponge top + towel half height + 1 mm (the placement offset)
# post-settle tolerances (bench side, drawer frame)
TOL_REST_Z = 0.006
TOWEL_OVER_DZ = (0.028, 0.042)  # towel centre above the sponge centre
TOWEL_OVER_DXY = 0.02
TOWEL_FLAT_ABOVE_FLOOR = (0.0125 - 0.004, 0.0125 + 0.004)
COVERED_ALONG = (-0.095, -0.065)
# Tilt bounds (degrees from the drawer's vertical). Measured on the nine kitchens: covered_back's sponge, spawned at
# along -0.085 against the back wall (the frozen geometry), settles tilted about 8.5 degrees; the towel of
# towel_over rests on the sponge and on the drawer's front wall top (about 5 degrees).
MAX_TILT_DEG = {"sponge": 20.0, OCCLUDER_NAME: 15.0}


class WorldDevError(RuntimeError):
    """A development world could not be built as declared (or was asked for where it is refused)."""


def parse(spec: str) -> str:
    """'occlusion-dev-1:<mode>' -> mode; anything else is refused."""
    m = WORLD_DEV_SPEC.match(str(spec))
    if not m or m.group(1) not in MODES:
        raise WorldDevError(f"world_dev must be {OCCLUSION_DEV_VERSION}:<mode> with a mode among {MODES} "
                            f"(got {spec!r})")
    return m.group(1)


def spec_of(mode: str) -> str:
    if mode not in MODES:
        raise WorldDevError(f"unknown occlusion mode {mode!r}")
    return f"{OCCLUSION_DEV_VERSION}:{mode}"


# ------------------------------------------------------------------------------------------------ refusals
def _numeric(kitchen):
    try:
        return int(kitchen)
    except (TypeError, ValueError):
        return None


def refusal(kitchen: str, mode: str, noise_seeds=(), registry: dict | None = None, out_dir=None,
            variant: str | None = None, s5_root=None) -> str | None:
    """None when this development world may be built; else why not (see the module docstring)."""
    if mode not in MODES:
        return f"refused: unknown occlusion mode {mode!r}"
    if variant is not None and variant != MODE_VARIANT[mode]:
        return f"refused: mode {mode} runs on variant {MODE_VARIANT[mode]}, not {variant!r}"
    if str(kitchen) not in DEV_KITCHENS:
        return f"refused: kitchen {kitchen!r} is not one of the development drawer kitchens {DEV_KITCHENS}"
    noise_seeds = [int(n) for n in noise_seeds]
    if registry is not None:
        k = _numeric(kitchen)
        listed = k is None
        own_block = None
        for e in registry.get("ranges", []):
            status = e.get("status")
            inside = k is not None and e.get("first") is not None and e["first"] <= k <= e["last"]
            if inside:
                listed = True
                if status == RESERVED:
                    return f"refused: kitchen {k} is in an entry still {RESERVED}"
                if status not in (None, "development"):
                    return f"refused: kitchen {k} has registry status {status!r} (development only)"
            ns = e.get("noise_seeds")
            if isinstance(ns, dict) and ns.get("first") is not None:
                lo, hi = int(ns["first"]), int(ns["last"])
                hit = [n for n in noise_seeds if lo <= n <= hi or lo <= n + 1 <= hi]
                if inside and isinstance(ns.get("multiplier"), int) and isinstance(ns.get("block"), int):
                    own_block = (ns["multiplier"] * k, ns["multiplier"] * k + ns["block"] - 1)
                if hit and (status == RESERVED or not inside):
                    return f"refused: noise seed {hit[0]} belongs to a registry block that is not this kitchen's"
        if not listed:
            return f"refused: kitchen {k} is not in the registry (never looked at)"
        if own_block is not None:
            out = [n for n in noise_seeds if not own_block[0] <= n <= n + 1 <= own_block[1]]
            if out:
                return f"refused: noise seed {out[0]} (camera {out[0] + 1}) outside kitchen {k}'s block {own_block}"
    if out_dir is not None:
        p = Path(out_dir).resolve()
        root = Path(s5_root).resolve() if s5_root is not None else None
        parts = p.relative_to(root).parts if root is not None and root in p.parents else p.parts
        if any(FORBIDDEN_OUT.match(x) for x in parts):
            return f"refused: a development world never writes into a real, pilot or validation folder ({p})"
    return None


def manifest_refusal(manifest: dict) -> str | None:
    """A manifest that names world_dev anywhere is refused unless it is a development batch (kind batch,
    development_tuning true); a validation manifest never carries it (the real-run launcher's rule)."""
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
            mode = parse(j["world_dev"])
        except WorldDevError as e:
            return f"refused: {e}"
        if str(j.get("kitchen")) not in DEV_KITCHENS:
            return f"refused: world_dev kitchen {j.get('kitchen')!r} is not a development drawer kitchen"
        del mode
    return None


# ------------------------------------------------------------------------------------------------ the instance
def declared_sponge_drawer(dspec) -> int:
    """The drawer the dossier declares for the sponge (robot-side knowledge, here only to place the occluder)."""
    sp = next((i for i in dspec.items if i.kind == "sponge"), None)
    if sp is None or not str(sp.where).startswith("drawers/"):
        raise WorldDevError("the dossier does not put the sponge in a drawer")
    return int(sp.where.split("/")[1])


def instance(kitchen: str, variant: str, mode: str, base_build_instance):
    """bench_task.build_instance(kitchen, variant) with the mode's changes: (truth, dossier spec, change log, setup log,
    target), and the drawer index of the occluder. The dossier spec is never changed."""
    why = refusal(kitchen, mode, variant=variant)
    if why:
        raise WorldDevError(why)
    inst = base_build_instance(kitchen, variant)
    if inst is None:
        raise WorldDevError(f"kitchen {kitchen} variant {variant}: not applicable")
    truth, dspec, change_log, setup_log, target = inst
    truth = copy.deepcopy(truth)
    k = declared_sponge_drawer(dspec)
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
    if mode == "covered_back":
        sponge.pos = (SPONGE_ALONG_COVERED, 0.0)
        entry["sponge_pos_in_drawer"] = list(sponge.pos)
    if mode in COVERED_MODES:
        entry["cover"] = {"lid_half": list(COVER_LID_HALF), "lid_pos": list(COVER_LID_POS),
                          "front_half": list(COVER_FRONT_HALF), "front_pos": list(COVER_FRONT_POS)}
    if mode == "towel_over":
        truth.items.append(Item(OCCLUDER_NAME, "towel", sponge.where, tuple(sponge.pos)))
        entry["towel"] = {"name": OCCLUDER_NAME, "over": "sponge", "pos_in_drawer": list(sponge.pos)}
    if mode == "towel_over_absent":
        pos = DRAWER_ITEM_POS["sponge"]
        truth.items.append(Item(OCCLUDER_NAME, "towel", f"drawers/{k}", tuple(pos)))
        entry["towel"] = {"name": OCCLUDER_NAME, "over": "floor", "pos_in_drawer": list(pos)}
    return (truth, dspec, list(change_log) + [entry], setup_log, target), k


# ------------------------------------------------------------------------------------------------ the build
def _world_build():
    """The module siteagent.world.build (siteagent.world exports its function build under the same name)."""
    import importlib  # noqa: PLC0415
    return importlib.import_module("siteagent.world.build")


def _add_cover(drawer_body, k: int) -> None:
    box = _world_build()._box
    box(drawer_body, f"drawer_{k}_cover_plate", COVER_LID_HALF, COVER_LID_POS, list(COVER_RGBA))
    box(drawer_body, f"drawer_{k}_cover_front", COVER_FRONT_HALF, COVER_FRONT_POS, list(COVER_RGBA))


def _place_towel_over(model, site) -> None:
    """towel_over: the sponge 0.5 mm above its rest and the towel on top of it, both through qpos0 (a reset keeps it)."""
    import mujoco  # noqa: PLC0415
    sponge = next(i for i in site.items if i.kind == "sponge")
    sj = model.joint(f"{sponge.name}_free")
    tj = model.joint(f"{OCCLUDER_NAME}_free")
    sa, ta = int(sj.qposadr[0]), int(tj.qposadr[0])
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)  # world pose of the drawer floor at qpos0
    k = int(sponge.where.split("/")[1])
    b = model.body(f"drawer_{k}").id
    floor = d.xpos[b] + d.xmat[b].reshape(3, 3) @ np.array([0.0, 0.0, DRAWER_FLOOR_TOP])
    sp = model.qpos0[sa:sa + 3].copy()
    sp[2] = floor[2] + SPONGE_REST_ABOVE_FLOOR + SPAWN_CLEARANCE
    model.qpos0[sa:sa + 3] = sp
    model.qpos0[ta:ta + 3] = sp + np.array([0.0, 0.0, TOWEL_ABOVE_SPONGE])
    model.qpos0[ta + 3:ta + 7] = model.qpos0[sa + 3:sa + 7]


def build_world(site, mode: str | None = None, cover_drawer: int | None = None, **world_kw):
    """(model, info) of world.build.build(site, **world_kw), plus the mode's build-time changes: the closed back
    compartment in drawer cover_drawer (covered modes) or the towel over the sponge (towel_over). mode None: exactly
    world.build.build."""
    B = _world_build()
    if mode is None:
        return B.build(site, **world_kw)
    if mode not in MODES:
        raise WorldDevError(f"unknown occlusion mode {mode!r}")
    if mode in COVERED_MODES:
        if cover_drawer is None:
            raise WorldDevError(f"{mode}: the drawer of the compartment is required")
        orig = B.MODULE_BUILDERS["drawers"]

        def drawers_with_cover(spec, info, f, m, _orig=orig):
            _orig(spec, info, f, m)
            _add_cover(spec.body(f"drawer_{int(cover_drawer)}"), int(cover_drawer))
        B.MODULE_BUILDERS["drawers"] = drawers_with_cover
        try:
            model, info = B.build(site, **world_kw)
        finally:
            B.MODULE_BUILDERS["drawers"] = orig
        return model, info
    model, info = B.build(site, **world_kw)
    if mode == "towel_over":
        _place_towel_over(model, site)
    return model, info


# ------------------------------------------------------------------------------------------------ post-settle check
def _in_drawer(sim, k: int, body: str) -> np.ndarray:
    """A body's position in drawer k's frame (bench truth)."""
    m, d = sim.model, sim.data
    b = m.body(f"drawer_{k}").id
    R = d.xmat[b].reshape(3, 3)
    return R.T @ (d.xpos[m.body(body).id] - d.xpos[b])


def _tilt_deg(sim, k: int, body: str) -> float:
    """Angle between a body's z axis and drawer k's z axis, degrees (bench truth)."""
    m, d = sim.model, sim.data
    zb = d.xmat[m.body(body).id].reshape(3, 3)[:, 2]
    zd = d.xmat[m.body(f"drawer_{k}").id].reshape(3, 3)[:, 2]
    return float(np.degrees(np.arccos(min(1.0, abs(float(zb @ zd))))))


def post_settle_check(sim, mode: str, k: int) -> dict:
    """Bench side, right after the Sim is built and settled: the sponge and the towel where the mode declares them, in
    drawer k's frame. Raises WorldDevError otherwise. Returns the measured values (recorded by the harness)."""
    B = _world_build()
    inner_hx, inner_hy = B.DRAWER_D / 2 - 0.02, B.DRAWER_WD / 2 - 0.016
    items = {i.name: i for i in sim.site.items}
    sponge = next((i for i in sim.site.items if i.kind == "sponge"), None)
    out = {"version": OCCLUSION_DEV_VERSION, "mode": mode, "drawer": f"drawers/{k}"}
    problems = []
    geoms = {sim.model.geom(g).name for g in range(sim.model.ngeom)}
    has_cover = f"drawer_{k}_cover_plate" in geoms and f"drawer_{k}_cover_front" in geoms
    out["cover_present"] = has_cover
    if (mode in COVERED_MODES) != has_cover:
        problems.append(f"cover present {has_cover} in mode {mode}")
    if (MODE_VARIANT[mode] == "correct") != (sponge is not None):
        problems.append("sponge presence does not match the mode")
    if (mode in TOWEL_MODES) != (OCCLUDER_NAME in items):
        problems.append("occluding towel presence does not match the mode")

    def inside(p):
        return abs(p[0]) <= inner_hx and abs(p[1]) <= inner_hy

    if sponge is not None:
        sp = _in_drawer(sim, k, sponge.name)
        out["sponge_in_drawer_m"] = [round(float(x), 4) for x in sp]
        rest = float(sp[2]) - DRAWER_FLOOR_TOP
        out["sponge_centre_above_floor_m"] = round(rest, 4)
        out["sponge_tilt_deg"] = round(_tilt_deg(sim, k, sponge.name), 2)
        if out["sponge_tilt_deg"] > MAX_TILT_DEG["sponge"]:
            problems.append(f"sponge tilted {out['sponge_tilt_deg']} degrees")
        if not inside(sp):
            problems.append(f"sponge outside drawer {k}'s floor ({sp.round(4).tolist()})")
        if abs(rest - SPONGE_REST_ABOVE_FLOOR) > TOL_REST_Z:
            problems.append(f"sponge not resting on the floor (centre {rest:.4f} m above it)")
        if mode == "covered_back":
            if not COVERED_ALONG[0] <= float(sp[0]) <= COVERED_ALONG[1]:
                problems.append(f"sponge along {float(sp[0]):.4f} outside {COVERED_ALONG} (not under the lid)")
            lid_bottom = COVER_LID_POS[2] - COVER_LID_HALF[2]
            out["lid_clearance_above_sponge_top_m"] = round(lid_bottom - (float(sp[2]) + 0.022), 4)
            if float(sp[2]) + 0.022 >= lid_bottom:
                problems.append("sponge top not under the lid")
    if OCCLUDER_NAME in items:
        tw = _in_drawer(sim, k, OCCLUDER_NAME)
        out["towel_in_drawer_m"] = [round(float(x), 4) for x in tw]
        out["towel_tilt_deg"] = round(_tilt_deg(sim, k, OCCLUDER_NAME), 2)
        if out["towel_tilt_deg"] > MAX_TILT_DEG[OCCLUDER_NAME]:
            problems.append(f"towel tilted {out['towel_tilt_deg']} degrees")
        if not inside(tw):
            problems.append(f"towel outside drawer {k}'s floor ({tw.round(4).tolist()})")
        if mode == "towel_over" and sponge is not None:
            sp = _in_drawer(sim, k, sponge.name)
            dz, dxy = float(tw[2] - sp[2]), float(np.hypot(*(tw[:2] - sp[:2])))
            out["towel_above_sponge_m"], out["towel_sponge_dxy_m"] = round(dz, 4), round(dxy, 4)
            if not TOWEL_OVER_DZ[0] <= dz <= TOWEL_OVER_DZ[1] or dxy > TOWEL_OVER_DXY:
                problems.append(f"towel not on the sponge (dz {dz:.4f}, dxy {dxy:.4f})")
        if mode == "towel_over_absent":
            h = float(tw[2]) - DRAWER_FLOOR_TOP
            out["towel_centre_above_floor_m"] = round(h, 4)
            if not TOWEL_FLAT_ABOVE_FLOOR[0] <= h <= TOWEL_FLAT_ABOVE_FLOOR[1]:
                problems.append(f"towel not flat on the floor (centre {h:.4f} m above it)")
    out["ok"] = not problems
    out["problems"] = problems
    if problems:
        raise WorldDevError(f"{OCCLUSION_DEV_VERSION}:{mode} post-settle check failed: {'; '.join(problems)}")
    return out


# ------------------------------------------------------------------------------------------------ applied()
@contextlib.contextmanager
def applied(world_dev: str, kitchen: str | None = None, registry: dict | None = None, noise_seeds=(), out_dir=None,
            s5_root=None):
    """While the block runs, bench_task.run_task_episode builds the development world: bench_task.build_instance gives
    the mode's instance and bench_task.Sim builds it (siteagent.sim.build is replaced during the Sim's construction
    only), then checks it after the settle. Yields a state dict: "mode", "checks" (one post_settle_check per Sim) and
    "instances" (kitchen, variant, drawer). Everything is restored on exit."""
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
