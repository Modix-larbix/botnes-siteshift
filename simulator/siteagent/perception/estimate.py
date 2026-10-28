"""Robot-side perception: from camera frames to tracks, region grids and placement evidence.

Uses only the CameraFrame of the simulated sensor (depth pixels and a category per visible segment), the robot's own
camera extrinsics (robot.design), its believed tool pose (joint encoders, kinematics, BaseBelief), its believed
worktop height, the public catalogs and the site plan. It never reads the simulator (a test scans this file) and
imports nothing from the kitchen generator or builder. ORACLE_INPUTS lists the oracle quantities used here: none.
Every number in this docstring is simulation, simulated perception, development kitchens only.

Pose estimates (heuristic, declared). Back-projected depth points of a segment, in the base frame, then in the world
through the BaseBelief. Objects: centre from the top-surface points minus the catalog height of that surface, yaw of
elongated objects from the principal axis (mod pi). Rail handles: top of the rail minus half its thickness, pull
direction perpendicular to the rail toward the robot. Modules: median of visible points (rough). Sigmas are a model,
not a calibration.

Visibility estimate. The segment's pixel count over the catalog silhouette projected with the believed camera pose at
the estimated pose. It feeds the sigma model, the handle-refine decision and the ranking of target tracks.

Region grids. Region floors are sampled on grids of about 1 cm (drawers) and 1.5 cm; each point is seen, covered by a
perceived object resting on it ("o"), or not seen. The site model uses the grid to say whether an object of a given
size could still be hidden (sitemodel.search_scope).

Closed reference of a drawer (DRAWER_REFERENCE_VERSION). A drawer's believed opening is its handle's offset along the
pull from the closed reference. A whole view that shows the handle more than CLOSED_REBASE_MAX_M outward never
replaces the reference (it is recorded as a measured opening), so a drawer left out does not read closed for good;
opening_unknown is cleared only by a view precise enough along the pull.

Floor under a perceived object (scope-occluder-1). A footprint codes "o" only when the object rests on that floor (its
room underneath is under OCCLUDER_MIN_GAP_M); points under a raised object stay "not seen", because a towel over a
sponge must not clear the sponge's place.

Placement evidence (SCOPE_COVER_VERSION). For the guarded target (the sponge), each region view also says, per
candidate placement of a catalog sponge box, whether the recognizer would have reported a sponge there: shown ('v')
only when, at every fitting yaw and both floor heights, the pixels whose measured depth reaches the top face reach
PLACEMENT_MIN_PX and PLACEMENT_MIN_FRAC of the box silhouette (the sensor's own thresholds). Resting objects exclude
the placements they occupy ('x'). The site model merges views per world position; the evidence only adds refusals.
Under configuration K1 the upright mode is evaluated by perception.upright_faces.

Surface outline (surface-frame-1). A surface region (landing tray, sink basin, drying mat) is placed on a fit of the
module's catalog outer footprint to the robot's own depth points over the episode's views, accepted only when it is
well pinned and supported; without an accepted outline the region never clears. Its outline is read on the first
track of the module's category.

Pose frame and overhead pre-check (pick-pose-3). The depth of the view that set a pickable track's pose is kept, and
overhead_check() runs perception.overhead on it before a top pick.

Skill reasons (contact-invalidates-1). note_action, note_drawer_motion and note_pick pass the skill's reason to the
site model, which decides which reasons invalidate clearances.

Configuration. One switch, SCOPE_CONFIGURATION ("K1" or "K2"), sets SCOPE_COVER_VERSION and PLACEMENT_MODES_ACTIVE;
configuration_check() refuses K1 without perception.upright_faces.

Wrist floor envelope (floor-env-1). A wrist view keeps its measured floor only when the measurement is valid, the band
rays lie far enough inside the region's floor rectangle, the band is flat enough and enough grid points project into
the image, and no drawer believed open overhangs a surface region; otherwise the view uses the believed floor. This
removes floor over-estimates from views of regions the look was not aimed at.

Drawer floor interior (drawer_floor_interior_1). The catalog drawer floor is the box's outer extent and includes the
8 mm walls; the floor band is measured on the interior rectangle, shrunk by DRAWER_WALL_THICKNESS_M on every side.

Drawer floor interior 2. Only the floor band uses the interior rectangle; the region grid (placements, coverage, grid
codes) is built on the catalog extent again, because the smaller grid caused false clearances.

Drawer deep end (DRAWER_DEEP_END_ACTIVE, off). When switched on, a drawer view's placement evidence also covers the
places up to DRAWER_DEEP_END_M beyond the believed deep end. Off in the recorded episodes.

Low level beyond (low_beyond_1). At the low test level a top-face sample counts as shown only when the measured depth
lies PLACEMENT_BEYOND_SIGMAS depth sigmas beyond it, so a thin cloth hugging a sponge cannot show the sponge's place.
A detection of the sponge is also read as standing on its long edge for the upright rest test.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field

import numpy as np

from ..robot.design import robot_eye_in_base, wrist_cam_in_hand
from ..robot.kinematics import TCP_OFFSET, hand_down
from ..site_plan import SLOT_ANGLES
from . import catalog as C
from . import overhead as OH
from .contract import (DECLARED, HELD, OBSERVED, STALE, Detection, HandleObservation, InspectedRegion,
                       ModuleObservation, Observation, Pose, Provenance)
from .sensor import depth_sigma

ORACLE_INPUTS: tuple = ()  # oracle quantities used robot side (none)
MODULE_SIGMA_XY = 0.09  # modules: no silhouette estimate (usually cut by the image edge or the arm)
ROUND_SIDES = 16
MIN_COVER_POINTS_IN_VIEW = 3
REACH_TOL = 0.03  # a line of sight that stops within 3 cm above a floor point counts as reaching it
GRID_INSET = 0.008  # floor grids stop 8 mm inside the floor edges
DRAWER_GRID_PITCH = (0.010, 0.015)  # along the drawer axis, across
# drawer_floor_interior_1 (module docstring, "Drawer floor interior"): the drawer's INTERIOR floor is the catalog
# floor (the box's outer extents, walls included) shrunk on every side by the wall thickness of the unit model
# (world.build's drawer box, checked by a test). Read only by region_geometries; every other use of the catalog drawer
# floor keeps catalog.DRAWER_FLOOR_*. The interior is computed as catalog value +/- the wall, so its floats are
# reproducible. drawer_floor_interior_2 (module docstring, "Drawer floor interior 2"): the interior rectangle drives
# only the floor band (and floor-env-1's edge condition, measured from it); the drawer region grid, the placement
# lattice and the candidate placements stay on the catalog extent (25 x 9). The deep-end extension stays off.
DRAWER_INTERIOR_VERSION = "drawer_floor_interior_2"
DRAWER_WALL_THICKNESS_M = 0.008
DRAWER_INTERIOR_BEHIND_HANDLE = (C.DRAWER_FLOOR_BEHIND_HANDLE[0] + DRAWER_WALL_THICKNESS_M,
                                 C.DRAWER_FLOOR_BEHIND_HANDLE[1] - DRAWER_WALL_THICKNESS_M)  # (0.040, 0.284)
DRAWER_INTERIOR_HALF_WIDTH = C.DRAWER_FLOOR_HALF_WIDTH - DRAWER_WALL_THICKNESS_M  # 0.061
# Deep-end extension (module docstring, "Drawer deep end"): written in advance, OFF. e_d = ceil(the largest in-view
# deep shortfall measured, 17.48 mm) + 2 mm.
DRAWER_DEEP_END_VERSION = "deep_end_1"
DRAWER_DEEP_END_ACTIVE = False
DRAWER_DEEP_END_M = 0.020
SURFACE_GRID_PITCH = 0.015
LYING_SPRAY_SIGMA_XY = 0.03
# Version of the drawer closed-reference rule (the earlier rule is implicitly
# drawer-reference-1). A whole view of a drawer believed closed may move its closed reference outward by at most
# CLOSED_REBASE_MAX_M in total (recorded views of resting drawers: at most 1.27 cm); a view further out is a measured
# opening. opening_unknown is cleared only by a whole view with a handle sigma along the pull of at most
# UNKNOWN_CLEAR_SIGMA_MAX_M (recorded whole views: wrist camera at most 9.9 mm, robot eye 1 % above 1.5 cm, those with
# the lowest visibility estimates; recorded clearing views at most 5.7 mm).
DRAWER_REFERENCE_VERSION = "drawer-reference-2"
CLOSED_REBASE_MAX_M = 0.02
UNKNOWN_CLEAR_SIGMA_MAX_M = 0.015
# scope-occluder-1 (module docstring): an unseen floor point under a perceived object codes "o" (never a hiding place)
# only when that object rests on the floor: its room underneath is under OCCLUDER_MIN_GAP_M, half the smallest catalog
# dimension of the guarded target. OCCLUDER_TARGET_CATEGORY is the agent's NOT_FOUND_GUARD_CATEGORY (agent.loop; this
# module imports nothing of the agent: a test checks they agree). The earlier rule (every footprint codes "o") is
# implicitly scope-occluder-0.
OCCLUDER_TARGET_CATEGORY = "sponge"
OCCLUDER_MIN_GAP_M = 0.5 * min(tuple(C.OBJECT_SHAPES[OCCLUDER_TARGET_CATEGORY]["footprint"])
                               + (C.OBJECT_SHAPES[OCCLUDER_TARGET_CATEGORY]["height"],))  # 0.018 m
# scope-placement-1 (module docstring, "Placement evidence"): the grid codes above stay (scope-occluder-1); the
# evidence that clears a region for the guarded target is a sensor-equivalent test per view and per candidate
# placement, merged per placement by world position. SCOPE_COVER_VERSION names the whole rule.
# "scope-placement-2-flat" = scope-placement-1 + the fitted surface outline (surface-frame-1) + the 3 cm surface margin,
# the sponge lying flat (configuration K2). Configuration K1 sets "scope-placement-2" together with
# PLACEMENT_MODES_ACTIVE = ("flat", "upright") and the face-aware upright test (perception.upright_faces); a test checks
# the two stay consistent. ONE switch names the configuration; SCOPE_COVER_VERSION and PLACEMENT_MODES_ACTIVE derive
# from it, and agent.host's full pull (open-full-1, drawer-booking-3) is active exactly when the upright mode is
# (host.open_full_active): the full pull ships only with the face-aware upright test. configuration_check() refuses K1
# without perception.upright_faces, at import. K1 was chosen on recorded development runs (E_K1 48 / 90 >= 0.8 x E_K2
# 43 / 90). K2 stays the fallback: set "K2" here and host.OPEN_FULL_ACTIVE = False.
SCOPE_CONFIGURATIONS = {"K1": ("scope-placement-2", ("flat", "upright")),
                        "K2": ("scope-placement-2-flat", ("flat",))}
SCOPE_CONFIGURATION = "K1"
SCOPE_COVER_VERSION = SCOPE_CONFIGURATIONS[SCOPE_CONFIGURATION][0]
PLACEMENT_MIN_PX = 40  # the recogniser's declared object thresholds (perception.sensor docstring: >= 40 visible
PLACEMENT_MIN_FRAC = 0.15  # pixels and a visible fraction >= 0.15); the robot side knows the spec, not the oracle value
PLACEMENT_MARGIN = 1.0  # factor on both thresholds (1: the recogniser's own; the numerator ignores the sides: errs to could-hide)
PLACEMENT_YAWS_DEG = (0.0, 30.0, 60.0, 90.0, 120.0, 150.0)  # yaws tested (relative to the grid's first axis, mod 180)
PLACEMENT_SETTLE_M = 0.004  # a resting target's top lies at most this far below its catalog height (declared)
# heights of the hypothetical top face tested: "low" = floor - floor error + height - settle (the strictest under a
# cover), "high" = floor + floor error + height (the strictest under an overhang)
PLACEMENT_TEST_LEVELS = ("low", "high")
PLACEMENT_DEPTH_TOL_M = 0.003  # + 2.5 depth sigma: a pixel shows the hypothetical top if its depth reaches it
# FIX_DRAFT_V2, thin covers; module docstring "Low level beyond": at the LOW test level a top-face sample is shown only
# when the line of sight passes the hypothetical top by at least PLACEMENT_BEYOND_SIGMAS depth sigmas ("beyond"),
# instead of reaching it within PLACEMENT_DEPTH_TOL_M + 2.5 sigma ("reach"). The low level is the lowest plausible top
# of a resting target (floor - floor error + height - settle): any surface at or above it, the target's own top or
# anything lying on it whatever its thickness, can no longer count as that top. The high level and the upright side
# faces keep "reach". Finding: thin_cover_20261019: a 1-4 mm cloth hugging an unrecognised sponge read as "the sponge's
# top" and cleared its place. The upright TOP faces use their own k,
# perception.upright_faces.UPRIGHT_LOW_BEYOND_SIGMAS (2.75); PLACEMENT_BEYOND_SIGMAS below is the flat test's.
PLACEMENT_LOW_RULE_VERSION = "low_beyond_1"
PLACEMENT_BEYOND_SIGMAS = 2.5
FLOOR_SEARCH_M = 0.04  # measured floor: grid rays whose measured surface lies within this of the believed floor
FLOOR_BAND_M = 0.012  # the floor is the median of the lowest band of that width (above the 5th percentile)
FLOOR_MIN_POINTS = 20  # fewer supporting rays: the floor is not measured in this view
FLOOR_MEASURED_ERR_M = 0.006  # declared error of a floor measured in the same view
FLOOR_BELIEF_ERR_M = 0.02  # declared error of the believed floor (handle track z - catalog; measured 1.64 cm)
# A camera listed here never measures its floor: its views always use the believed floor, whatever the envelope.
# wrist_cam is handled by the wrist floor envelope below instead, so the tuple is empty (the switch stays).
FLOOR_UNMEASURED_CAMERAS: tuple = ()
# Wrist floor envelope (module docstring): views of FLOOR_ENVELOPE_CAMERAS keep their measured floor only inside
# envelope floor-env-1, frozen on development data before validation: band edge median >= 12 mm, band IQR <= 4 mm,
# >= 20 % of the region grid in the image, plus the drawer-over-surface condition. robot_eye is not listed: unchanged.
# A frame without a camera name (test frames) is not listed either. The version id is floor-env-1 written with
# underscores: robot-side records never carry the word "floor" alone, a simulator geom name (no hidden names in
# robot-side JSON).
FLOOR_ENVELOPE_VERSION = "floor_env_1"
FLOOR_ENVELOPE_CAMERAS = ("wrist_cam",)
FLOOR_ENV_EDGE_MED_MIN_M = 0.012  # frozen "edge_med" [">=", 0.012]
FLOOR_ENV_BAND_IQR_MAX_M = 0.004  # frozen "z_iqr" ["<=", 0.004]
FLOOR_ENV_GRID_FRAC_MIN = 0.2  # frozen "frac_in" [">=", 0.2]
# The drawer-over-surface condition reads a drawer as open when its opening is unknown or its believed opening
# (Perception.opening) is more than this: the 5 mm bound under which _drawer_view treats a drawer as closed.
DRAWER_BELIEVED_CLOSED_M = 0.005
OBJECT_Z_ERR_M = (0.006, 0.016)  # declared error of a detection's top z: pose from this view / from an earlier one
# Hypotheses for the guarded target (catalog dimensions L >= W >= H): lying flat (footprint L x W, height H) and,
# with "upright", standing on its long edge (footprint L x H, height W). PLACEMENT_MODES_ACTIVE decides which ones a
# region must rule out; the others are computed for the records only.
PLACEMENT_MODES_ACTIVE = SCOPE_CONFIGURATIONS[SCOPE_CONFIGURATION][1]
# K1 only: an upright placement is 'v' only when, at every fitting yaw and both levels, its face-aware shown pixels (top
# face + camera-facing side faces, perception.upright_faces) reach PLACEMENT_MARGIN_UPRIGHT x both recogniser thresholds
# against the whole box silhouette. 1.25: the calibration check passed at 1.25 on recorded development views.
PLACEMENT_MARGIN_UPRIGHT = 1.25
# surface-frame-1 (module docstring, "Surface outline"). Declared calibration constants of the simulated sensor, frozen
# before validation (a 3 cm margin, U <= 0.08).
SURFACE_FRAME_VERSION = "surface-frame-1"
OUTLINE_TOL_M = 0.008  # module points inside the outer footprint grown by 8 mm, free worktop points outside it shrunk by 8 mm
OUTLINE_FREE_BELOW_M = 0.005  # a free worktop point lies at least 5 mm under the module's own measured top
OUTLINE_FREE_Z_WIN_M = 0.03  # ... and within 3 cm of the believed worktop
OUTLINE_NEAR_R_M = 0.45  # points within 45 cm of the prior centre
OUTLINE_STRIDE = 2  # every other pixel in both directions
OUTLINE_XY_WIN_M = 0.14  # search window around the prior: +-14 cm ...
OUTLINE_BIN_M = 0.002  # (summed-area bins of 2 mm; centre offsets every 2 bins: 4 mm steps)
OUTLINE_YAW_WIN_DEG = 10.0  # ... and +-10 degrees
OUTLINE_YAW_STEP_DEG = 0.5
OUTLINE_DELTA_MIN = 12  # pinning: the cost must rise by max(12, 0.4 % of the points)
OUTLINE_DELTA_FRAC = 0.004
OUTLINE_TIE_MIN = 2  # near-optimal ties: within max(2, 0.05 % of the points) of the best cost
OUTLINE_TIE_FRAC = 0.0005
OUTLINE_MIN_M_POINTS = 50  # fewer module points in the union: no fit
OUTLINE_EDGE_GUARD_M = 0.012  # an optimum within 12 mm (1 degree) of the window edge is searched again around itself
OUTLINE_EDGE_GUARD_DEG = 1.0
OUTLINE_MAX_M_PER_VIEW = 2500  # union buffers: points kept per view and in the union (even subsampling)
OUTLINE_MAX_F_PER_VIEW = 5000
OUTLINE_UNION_MAX_M = 8000
OUTLINE_UNION_MAX_F = 16000
OUTLINE_U_MAX_M = 0.08  # acceptance: pinning bound U <= 8 cm (pinned fits 0.037-0.059, unpinned 0.75 or infinite)
OUTLINE_SUPPORT_CELL_M = 0.02
OUTLINE_SUPPORT_MIN = 0.30  # acceptance: module points in >= 30 % of the outer footprint's 2 cm cells
OUTLINE_FREEZE_M = 0.01  # a later accepted fit within 1 cm and 1 degree keeps the frame
OUTLINE_FREEZE_DEG = 1.0
SURFACE_PLACEMENT_MARGIN_M = 0.03  # surface placement centres extend 3 cm beyond the believed centre range (4 cm)
SURFACE_EVIDENCE_EXTENSION_M = 0.04  # the surface evidence lattice is computed this far out (whole pitches), then masked
# pick-pose-3: categories whose pose frame's depth is kept for the overhead pre-check (every object category of the
# recogniser: the skill catalog can pick each one; a test checks the two catalogs agree).
PICKABLE_CATEGORIES = tuple(C.OBJECT_CATEGORIES)


def placement_modes(category: str = OCCLUDER_TARGET_CATEGORY) -> dict:
    """{mode: (footprint long, footprint short, height)} of the hypotheses tested for the guarded target (catalog)."""
    sh = C.OBJECT_SHAPES[category]
    L, W = max(sh["footprint"]), min(sh["footprint"])
    H = float(sh["height"])
    return {"flat": (float(L), float(W), H), "upright": (float(L), min(H, float(W)), max(H, float(W)))}


def _rect_taps(L: float, W: float, yaw: float, fa: float, fb: float):
    """Offsets (di, dj) of the fine lattice (steps fa, fb) inside an L x W rectangle centred on a lattice point,
    long side at yaw from the first axis."""
    R = 0.5 * math.hypot(L, W)
    ni, nj = int(math.ceil(R / fa)) + 1, int(math.ceil(R / fb)) + 1
    di, dj = np.meshgrid(np.arange(-ni, ni + 1), np.arange(-nj, nj + 1), indexing="ij")
    x, y = di.ravel() * fa, dj.ravel() * fb
    c, s = math.cos(yaw), math.sin(yaw)
    inside = (np.abs(x * c + y * s) <= 0.5 * L + 1e-9) & (np.abs(-x * s + y * c) <= 0.5 * W + 1e-9)
    return di.ravel()[inside], dj.ravel()[inside]


def _shoelace(u, v):
    """Areas of polygons given by vertex arrays (..., n) in order."""
    return 0.5 * np.abs(np.sum(u * np.roll(v, -1, axis=-1) - v * np.roll(u, -1, axis=-1), axis=-1))


def box_silhouette_px(fr, cam_p, cam_R, centres, ul, uw, L, W, z_top, H):
    """Projected area (pixels, not clipped to the image) of upright boxes: centres (N, 2) world xy, long-axis unit
    vectors ul (N, 2) and uw (N, 2), size L x W x H, top at z_top. Sum of the projected areas of the faces that face
    the camera (a convex box covers its silhouette once). inf when a corner lies behind the camera."""
    N = len(centres)
    corners = np.zeros((N, 2, 2, 2, 3))  # (N, sL, sW, top?, xyz)
    for a, sl in enumerate((-1.0, 1.0)):
        for b, sw in enumerate((-1.0, 1.0)):
            xy = centres + sl * 0.5 * L * ul + sw * 0.5 * W * uw
            corners[:, a, b, 0, :2] = xy
            corners[:, a, b, 1, :2] = xy
    corners[..., 0, 2] = z_top - H
    corners[..., 1, 2] = z_top
    pc = (corners - cam_p) @ cam_R
    Z = -pc[..., 2]
    bad = (Z <= 0.02).reshape(N, -1).any(axis=1)
    Zs = np.where(Z > 0.02, Z, 1.0)
    u = fr.cx + fr.fx * pc[..., 0] / Zs
    v = fr.cy - fr.fy * pc[..., 1] / Zs
    # Faces as corner index lists (sL, sW, top) in cyclic order, with outward normals.
    faces = [([(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)], None, 1.0),  # top
             ([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], None, -1.0),  # bottom
             ([(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)], ("l", 1.0), 0.0),
             ([(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)], ("l", -1.0), 0.0),
             ([(0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)], ("w", 1.0), 0.0),
             ([(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)], ("w", -1.0), 0.0)]
    total = np.zeros(N)
    for idx, side, nz in faces:
        fu = np.stack([u[:, a, b, t] for a, b, t in idx], axis=-1)
        fv = np.stack([v[:, a, b, t] for a, b, t in idx], axis=-1)
        fc = np.mean([corners[:, a, b, t] for a, b, t in idx], axis=0)
        to_cam = cam_p - fc
        if side is None:
            facing = nz * to_cam[:, 2] > 0
        else:
            n2 = (ul if side[0] == "l" else uw) * side[1]
            facing = (n2[:, 0] * to_cam[:, 0] + n2[:, 1] * to_cam[:, 1]) > 0
        total += np.where(facing, _shoelace(fu, fv), 0.0)
    return np.where(bad, np.inf, total)


def placement_excluded(xy, objects, mode: str, r_in: float) -> np.ndarray:
    """Per placement centre (world xy (N, 2)): does a resting object's surely-occupied footprint come within r_in (the
    radius of the disk the target contains in this mode, whatever its yaw)? Then the target cannot lie there."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    out = np.zeros(len(xy), bool)
    for ob in objects or ():
        if not ob.get("rests", {}).get(mode):
            continue
        for x, y, yaw, hl, hs in ob["fps"]:  # its footprint in each reading (a merged record keeps them all)
            dxy = xy - np.array([x, y])
            if yaw is None:
                dist = np.maximum(np.hypot(dxy[:, 0], dxy[:, 1]) - hl, 0.0)
            else:
                ax = np.array([math.cos(yaw), math.sin(yaw)])
                da = np.maximum(np.abs(dxy @ ax) - hl, 0.0)
                db = np.maximum(np.abs(dxy @ np.array([-ax[1], ax[0]])) - hs, 0.0)
                dist = np.hypot(da, db)
            out |= dist <= r_in
    return out


def _surface_beyond(ac, bc, code, W: float, grid, surface: dict):
    """surface-frame-1: a mode's centre range on the unextended grid, in the extended lattice's coordinates
    (the unextended grid starts surface["k"] pitches in); placements more than surface["margin"] beyond it are not
    candidates (' ' unless 'v'). ac, bc: placement centre offsets (N,) along the lattice axes; code: (N,) characters;
    W: the mode's footprint short side. Returns (code, number of placements masked)."""
    (na, nb), (pa, pb) = grid
    k = int(surface["k"])
    ext_a0, ext_b0 = (na - 2 * k - 1) * pa, (nb - 2 * k - 1) * pb
    lo_a, hi_a = k * pa - GRID_INSET + 0.5 * W, k * pa + ext_a0 + GRID_INSET - 0.5 * W
    lo_b, hi_b = k * pb - GRID_INSET + 0.5 * W, k * pb + ext_b0 + GRID_INSET - 0.5 * W
    off = np.maximum.reduce([lo_a - ac, ac - hi_a, lo_b - bc, bc - hi_b, np.zeros_like(ac)])
    beyond = (off > float(surface["margin"]) + 1e-9) & (code != " ") & (code != "v")
    return np.where(beyond, " ", code), int(beyond.sum())


def _surface_mask_record(rec: dict, W: float, grid, surface: dict) -> dict:
    """K1: _surface_beyond on a mode record built elsewhere (perception.upright_faces), from its own
    lattice description (k0, n, step: centres (k0 + i) x step along each axis, row-major as here)."""
    code = rec.get("code") or ""
    if not code:
        return rec
    (k0a, k0b), (n_a, n_b), (sa, sb) = rec["k0"], rec["n"], rec["step"]
    if len(code) != int(n_a) * int(n_b):
        raise ValueError("upright record: code length does not match its lattice")
    KA, KB = np.meshgrid(np.arange(int(n_a)) + int(k0a), np.arange(int(n_b)) + int(k0b), indexing="ij")
    masked, n = _surface_beyond(KA.ravel() * float(sa), KB.ravel() * float(sb), np.array(list(code)), W, grid, surface)
    return dict(rec, code="".join(masked), beyond_margin=n)


def _deep_end_beyond(ac, code, W: float, grid, deep: dict):
    """Deep-end extension (module docstring, "Drawer deep end"; used only when DRAWER_DEEP_END_ACTIVE): the lattice
    is the region grid extended by deep["k"] rows at the deep end only (the end of the first axis, away from the
    handle); placements whose centre lies more than deep["margin"] beyond the mode's deepest centre on the region grid
    are not candidates (' ' unless 'v'). ac: placement centre offsets (N,) along the first axis; code: (N,) characters;
    W: the mode's footprint short side. The front and the sides are not extended: nothing is masked there. Returns
    (code, number of placements masked)."""
    (na, _nb), (pa, _pb) = grid
    ext_a0 = (na - int(deep["k"]) - 1) * pa
    hi_a = ext_a0 + GRID_INSET - 0.5 * W
    beyond = (np.asarray(ac, float) - hi_a > float(deep["margin"]) + 1e-9) & (code != " ") & (code != "v")
    return np.where(beyond, " ", code), int(beyond.sum())


def _deep_end_mask_record(rec: dict, W: float, grid, deep: dict) -> dict:
    """_deep_end_beyond on a mode record built elsewhere (perception.upright_faces), from its own lattice description
    (k0, n, step), as _surface_mask_record."""
    code = rec.get("code") or ""
    if not code:
        return rec
    (k0a, _k0b), (n_a, n_b), (sa, _sb) = rec["k0"], rec["n"], rec["step"]
    if len(code) != int(n_a) * int(n_b):
        raise ValueError("upright record: code length does not match its lattice")
    KA = np.repeat(np.arange(int(n_a)) + int(k0a), int(n_b))
    masked, n = _deep_end_beyond(KA * float(sa), np.array(list(code)), W, grid, deep)
    return dict(rec, code="".join(masked), beyond_deep_end=n)


def _upright_faces():
    """K1 only: the face-aware upright test module (perception.upright_faces). Contract,
    fixed by its call in Perception._placement_evidence (the prototype's signature): upright_evidence(E, fr, cam_p,
    cam_R, pts, grid, f_used, f_err, objects, margin) -> the upright mode record {k0, n, step, r_in, code, share_pct,
    px, ...} on the lattice of this module (same placement centres, yaws, levels, codes 'v' 'x' '-' '.' ' '), E being
    this module. Resolved through the import system (sys.modules first), not the package
    attribute, so the module registered under siteagent.perception.upright_faces is the one used (a None entry raises
    ImportError)."""
    import importlib  # noqa: PLC0415 (K1 only)
    return importlib.import_module(f"{__package__}.upright_faces")


def configuration_check(configuration: str | None = None, modes=None) -> None:
    """Raises ValueError when the configuration is not one of SCOPE_CONFIGURATIONS, when the
    active modes do not pair with it, or when the upright mode is active without the face-aware test module. Run at
    import, so a K1 lab without perception.upright_faces fails loudly instead of clearing upright placements."""
    configuration = SCOPE_CONFIGURATION if configuration is None else configuration
    modes = tuple(PLACEMENT_MODES_ACTIVE if modes is None else modes)
    if configuration not in SCOPE_CONFIGURATIONS:
        raise ValueError(f"unknown configuration {configuration!r}")
    if modes != SCOPE_CONFIGURATIONS[configuration][1]:
        raise ValueError(f"configuration {configuration} pairs with modes {SCOPE_CONFIGURATIONS[configuration][1]}")
    if "upright" in modes:
        uf = _upright_faces()
        if not callable(getattr(uf, "upright_evidence", None)):
            raise ValueError("perception.upright_faces has no upright_evidence")


def occluder_top_and_height(category: str, z_centre: float, lying: bool) -> tuple[float, float] | None:
    """scope-occluder-1: the observed top of a detected object (its pose z plus the catalog height of its top
    above its centre) and its catalog height, lying values for a lying spray bottle. None for a category without a
    catalog shape. Robot side: the detection's pose and the public catalog only."""
    sh = C.OBJECT_SHAPES.get(category)
    if sh is None:
        return None
    if lying:
        return float(z_centre) + float(sh["lying_top_above_centre"]), float(sh["lying_height"])
    return float(z_centre) + float(sh["top_above_centre"]), float(sh["height"])


def occluder_gap(top_z: float, height: float, floor_z):
    """scope-occluder-1: room under an object above a floor of height floor_z (a float or an array):
    (observed top - floor z) - catalog height. The object rests on that floor when the gap is under
    OCCLUDER_MIN_GAP_M."""
    return (top_z - floor_z) - height


# ------------------------------------------------------------------ wrist floor envelope (floor-env-1)
def grid_rect(pts, inset: float = 0.0) -> tuple:
    """The plan rectangle of a row-major region grid (pts (N, 3) or (N, 2), at least 2 points along its second axis),
    grown by `inset` on every side: (corner xy (2,), unit axis a (2,), unit axis b (2,), length along a, length along b).
    Axis b runs along the grid's second index (pts[1] - pts[0]), axis a perpendicular to it. With inset GRID_INSET it is
    the region's believed floor rectangle (region_geometries places the grid GRID_INSET inside the floor's edges)."""
    xy = np.asarray(pts, float)[:, :2]
    eb = xy[1] - xy[0]
    eb = eb / max(float(np.linalg.norm(eb)), 1e-12)
    ea = np.array([eb[1], -eb[0]])
    A, B = (xy - xy[0]) @ ea, (xy - xy[0]) @ eb
    corner = xy[0] + (float(A.min()) - inset) * ea + (float(B.min()) - inset) * eb
    return (corner, ea, eb, float(A.max() - A.min()) + 2 * inset, float(B.max() - B.min()) + 2 * inset)


def rect_edge_distance(rect, xy) -> np.ndarray:
    """Distance (m) from each point xy (N, 2) to the nearest side of rect (grid_rect's tuple), positive inside."""
    corner, ea, eb, la, lb = rect
    d = np.asarray(xy, float).reshape(-1, 2) - corner
    a, b = d @ ea, d @ eb
    return np.minimum.reduce([a, la - a, b, lb - b])


def _rect_corners(rect) -> np.ndarray:
    corner, ea, eb, la, lb = rect
    return np.array([corner, corner + la * ea, corner + la * ea + lb * eb, corner + lb * eb])


def rects_overlap(r1, r2) -> bool:
    """Do two plan rectangles (grid_rect's tuple) overlap with a positive area (separating axis test; rectangles that
    only touch do not overlap)?"""
    c1, c2 = _rect_corners(r1), _rect_corners(r2)
    for ax in (r1[1], r1[2], r2[1], r2[2]):
        p1, p2 = c1 @ ax, c2 @ ax
        if min(p1.max(), p2.max()) - max(p1.min(), p2.min()) <= 1e-9:
            return False
    return True


def floor_band(qz, fz: float, edge=None, stats: dict | None = None):
    """The band rule of Perception._measured_floor on the measured heights qz (N,) of the valid rays of a view: the
    candidates lie within FLOOR_SEARCH_M of the believed floor fz; the band is the candidates within FLOOR_BAND_M above
    their 5th percentile; (median of the band, band size), or None with fewer than FLOOR_MIN_POINTS candidates or band
    rays. stats (records and the floor-env-1 envelope): gets "n_cand", "n_band", "band_iqr_m" (inter-quartile range of
    the band heights) and, with edge (N,) the distance of each ray's grid point to the believed floor boundary,
    "edge_med_m" (its median over the band rays). The two are rounded as the frozen envelope's analysis read them (the
    r6-hook-3 records: edge median to 0.1 mm, IQR to 0.01 mm), so the decision is the frozen candidate's on the same
    rays."""
    qz = np.asarray(qz, float)
    cm = np.abs(qz - fz) <= FLOOR_SEARCH_M
    c = qz[cm]
    if stats is not None:
        stats["n_cand"] = int(len(c))
    if len(c) < FLOOR_MIN_POINTS:
        return None
    lo = float(np.percentile(c, 5))
    bm = (c >= lo) & (c <= lo + FLOOR_BAND_M)
    band = c[bm]
    if stats is not None:
        stats["n_band"] = int(len(band))
    if len(band) < FLOOR_MIN_POINTS:
        return None
    if stats is not None:
        stats["band_iqr_m"] = round(float(np.percentile(band, 75) - np.percentile(band, 25)), 5)
        if edge is not None:
            stats["edge_med_m"] = round(float(np.median(np.asarray(edge, float)[cm][bm])), 4)
    return float(np.median(band)), int(len(band))


def floor_envelope_checks(valid: bool, edge_med_m, band_iqr_m, grid_frac, open_over=()) -> dict:
    """The floor-env-1 conditions (frozen envelope plus the drawer-over-surface condition) on a view's
    quantities: {condition: passed}. A missing quantity fails its condition. The view keeps its measured floor when
    every value is True."""
    return {"valid": bool(valid),
            "edge_med": edge_med_m is not None and bool(edge_med_m >= FLOOR_ENV_EDGE_MED_MIN_M),
            "band_iqr": band_iqr_m is not None and bool(band_iqr_m <= FLOOR_ENV_BAND_IQR_MAX_M),
            "grid_frac": grid_frac is not None and bool(grid_frac >= FLOOR_ENV_GRID_FRAC_MIN),
            "no_open_drawer_over": not list(open_over)}


# ------------------------------------------------------------------ surface outline fit (surface-frame-1)
def outline_half(category: str) -> tuple[float, float]:
    """Half sizes (x, y) of a surface module's catalog outer footprint (catalog.MODULE_OUTER_FOOTPRINTS)."""
    a, b = C.MODULE_OUTER_FOOTPRINTS[category]
    return 0.5 * float(a), 0.5 * float(b)


def _dilate1(mask: np.ndarray) -> np.ndarray:
    """Pixels within 1 pixel (8-neighbourhood) of a True pixel of a boolean image."""
    m = mask
    out = m.copy()
    out[1:, :] |= m[:-1, :]
    out[:-1, :] |= m[1:, :]
    out[:, 1:] |= m[:, :-1]
    out[:, :-1] |= m[:, 1:]
    out[1:, 1:] |= m[:-1, :-1]
    out[:-1, :-1] |= m[1:, 1:]
    out[1:, :-1] |= m[:-1, 1:]
    out[:-1, 1:] |= m[1:, :-1]
    return out


def segment_label_image(fr) -> np.ndarray:
    """(H, W) int16: k + 1 at the pixels of the frame's k-th segment (its reported pixel coordinates, floored), 0
    elsewhere; a later segment overwrites an earlier one."""
    lab = np.zeros((fr.height, fr.width), np.int16)
    for k, s in enumerate(fr.segments):
        if len(s.uv) == 0:
            continue
        u = np.clip(np.floor(s.uv[:, 0]).astype(int), 0, fr.width - 1)
        v = np.clip(np.floor(s.uv[:, 1]).astype(int), 0, fr.height - 1)
        lab[v, u] = k + 1
    return lab


def outline_points(depth, any_segment, module_mask, intr, cam_p, cam_R, prior_xy, worktop_z,
                   stride: int = OUTLINE_STRIDE):
    """(module points xy (N, 2), free worktop points xy (K, 2), module top z or None) of one view, robot side: the
    frame's depth back-projected with the believed camera pose (every stride-th pixel), within OUTLINE_NEAR_R_M of the
    prior centre. Module points: the module segment's pixels. Free points: pixels of no segment (not within 1 pixel of
    one) within OUTLINE_FREE_Z_WIN_M of the believed worktop and OUTLINE_FREE_BELOW_M under the module's own top (the
    median of its points near the worktop)."""
    fx, fy, cx, cy = intr[:4]
    H, W = depth.shape
    v, u = np.mgrid[0:H:stride, 0:W:stride]
    dep = depth[::stride, ::stride]
    near_seg = _dilate1(any_segment)[::stride, ::stride]
    mod = module_mask[::stride, ::stride]
    ok = dep > 0
    z = dep
    x = (u + 0.5 - cx) / fx * z
    y = -(v + 0.5 - cy) / fy * z
    pc = np.stack([x, y, -z], axis=-1)
    P = pc @ cam_R.T + cam_p
    near = ok & (np.hypot(P[..., 0] - prior_xy[0], P[..., 1] - prior_xy[1]) < OUTLINE_NEAR_R_M)
    Mp = P[near & mod]
    if len(Mp) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2)), None
    low = Mp[(Mp[:, 2] > worktop_z - 0.03) & (Mp[:, 2] < worktop_z + 0.05)]
    top_z = float(np.median(low[:, 2])) if len(low) >= 20 else float(worktop_z + 0.01)
    Fp = P[near & ~near_seg & (np.abs(P[..., 2] - worktop_z) < OUTLINE_FREE_Z_WIN_M)]
    Fp = Fp[Fp[:, 2] < top_z - OUTLINE_FREE_BELOW_M]
    return Mp[:, :2], Fp[:, :2], top_z


def _outline_sat(a, b, lo_a, lo_b, na, nb):
    ia = np.floor((a - lo_a) / OUTLINE_BIN_M).astype(int)
    ib = np.floor((b - lo_b) / OUTLINE_BIN_M).astype(int)
    ok = (ia >= 0) & (ia < na) & (ib >= 0) & (ib < nb)
    h = np.zeros((na + 1, nb + 1))
    np.add.at(h, (ia[ok] + 1, ib[ok] + 1), 1.0)
    return h.cumsum(0).cumsum(1)


def _outline_box(S, i0, i1, j0, j1):
    """Sums over bins [i0, i1) x [j0, j1) of a summed-area table (index arrays, clipped)."""
    n0, n1 = S.shape[0] - 1, S.shape[1] - 1
    i0, i1 = np.clip(i0, 0, n0), np.clip(i1, 0, n0)
    j0, j1 = np.clip(j0, 0, n1), np.clip(j1, 0, n1)
    return S[i1, j1] - S[i0, j1] - S[i1, j0] + S[i0, j0]


def _outline_cost_grid(M, F, yaw, c0, half):
    """Cost (#module points outside the footprint grown by OUTLINE_TOL_M + #free points inside it shrunk by
    OUTLINE_TOL_M) over the centre offsets (every 2 bins within +-OUTLINE_XY_WIN_M, in the yaw frame around c0)."""
    hx, hy = half
    ax = np.array([math.cos(yaw), math.sin(yaw)])
    ay = np.array([-ax[1], ax[0]])
    am, bm = (M - c0) @ ax, (M - c0) @ ay
    af, bf = (F - c0) @ ax, (F - c0) @ ay
    R = OUTLINE_XY_WIN_M + max(hx, hy) + 0.05
    n = int(round(2 * R / OUTLINE_BIN_M))
    Sm = _outline_sat(am, bm, -R, -R, n, n)
    Sf = _outline_sat(af, bf, -R, -R, n, n)
    offs = np.arange(-OUTLINE_XY_WIN_M, OUTLINE_XY_WIN_M + 1e-9, OUTLINE_BIN_M * 2)
    OA, OB = np.meshgrid(offs, offs, indexing="ij")

    def idx(x):
        return np.round((x + R) / OUTLINE_BIN_M).astype(int)
    tol = OUTLINE_TOL_M
    m_in = _outline_box(Sm, idx(OA - hx - tol), idx(OA + hx + tol), idx(OB - hy - tol), idx(OB + hy + tol))
    f_in = _outline_box(Sf, idx(OA - hx + tol), idx(OA + hx - tol), idx(OB - hy + tol), idx(OB + hy - tol))
    cost = (len(M) - m_in) + f_in
    return cost, offs, ax, ay


def outline_fit(M, F, category: str, prior_c, prior_yaw: float) -> dict | None:
    """Best pose (centre, yaw) of the module's outer footprint for module points M and free points F (world xy), searched
    around the prior, and its pinning bound U (module docstring). None with fewer than OUTLINE_MIN_M_POINTS points."""
    half = outline_half(category)
    if len(M) < OUTLINE_MIN_M_POINTS:
        return None
    prior_c = np.asarray(prior_c, float)
    best = None
    yaw_win, yaw_step = math.radians(OUTLINE_YAW_WIN_DEG), math.radians(OUTLINE_YAW_STEP_DEG)
    yaws = prior_yaw + np.arange(-yaw_win, yaw_win + 1e-9, yaw_step)
    costs_by_yaw = []
    for yaw in yaws:
        cost, offs, ax, ay = _outline_cost_grid(M, F, yaw, prior_c, half)
        k = np.unravel_index(int(np.argmin(cost)), cost.shape)
        costs_by_yaw.append(float(cost[k]))
        if best is None or cost[k] < best[0]:
            best = (float(cost[k]), float(yaw), prior_c + offs[k[0]] * ax + offs[k[1]] * ay)
    c_best, yaw_best, ctr = best
    # ties (a flat minimum): the middle of the near-optimal yaw interval, then the centroid of the near-optimal centres
    # at that yaw (an argmin would take the first tie, a bias toward negative offsets)
    tau = max(OUTLINE_TIE_MIN, OUTLINE_TIE_FRAC * (len(M) + len(F)))
    cyw = np.array(costs_by_yaw)
    near = np.flatnonzero(cyw <= c_best + tau)
    yaw_best = float(0.5 * (yaws[near.min()] + yaws[near.max()]))
    cost, offs, ax, ay = _outline_cost_grid(M, F, yaw_best, ctr, half)
    cmin = float(cost.min())
    ii, jj = np.nonzero(cost <= cmin + tau)
    ctr = ctr + float(np.mean(offs[ii])) * ax + float(np.mean(offs[jj])) * ay
    c_best = float(min(c_best, cmin))
    delta = max(OUTLINE_DELTA_MIN, OUTLINE_DELTA_FRAC * (len(M) + len(F)))
    # pinning along the best frame's axes (yaw fixed), then in yaw (best centre per yaw)
    cost, offs, ax, ay = _outline_cost_grid(M, F, yaw_best, ctr, half)
    i0 = int(np.argmin(np.abs(offs)))
    base = float(cost[i0, i0])
    prof_a, prof_b = cost[:, i0], cost[i0, :]

    def pin(prof, sign):
        rng = range(i0, len(offs)) if sign > 0 else range(i0, -1, -1)
        for i in rng:
            if prof[i] >= base + delta:
                return abs(float(offs[i]))
        return float("inf")
    pa = (pin(prof_a, -1), pin(prof_a, 1))
    pb = (pin(prof_b, -1), pin(prof_b, 1))
    iy = int(np.argmin(np.abs(yaws - yaw_best)))

    def pin_yaw(sign):
        rng = range(iy, len(yaws)) if sign > 0 else range(iy, -1, -1)
        for i in rng:
            if cyw[i] >= c_best + delta:
                return abs(float(yaws[i] - yaw_best))
        return float("inf")
    py = (pin_yaw(-1), pin_yaw(1))
    hx, hy = C.MODULE_SURFACES[category]["half"]
    r = math.hypot(hx, hy)
    sa, sb, sy = max(pa), max(pb), max(py)
    U = math.hypot(sa, sb) + r * (sy if math.isfinite(sy) else math.pi)
    return {"centre": [float(ctr[0]), float(ctr[1])], "yaw": yaw_best, "cost": c_best, "n_m": int(len(M)),
            "n_f": int(len(F)), "delta": delta, "pin_a": pa, "pin_b": pb, "pin_yaw_deg": [math.degrees(x) for x in py],
            "U": U}


def outline_subsample(P, n: int):
    """At most n rows of P, evenly spaced (deterministic)."""
    if len(P) <= n:
        return P
    idx = np.linspace(0, len(P) - 1, n).astype(int)
    return P[idx]


def outline_fit_robust(M, F, category: str, prior_c, prior_yaw: float, tries: int = 2) -> dict | None:
    """outline_fit with the window check: an optimum within OUTLINE_EDGE_GUARD_M (OUTLINE_EDGE_GUARD_DEG) of the search
    window's edge is searched again once around itself; still at the edge: "edge" True and U infinite (rejected)."""
    pc, py = np.asarray(prior_c, float), float(prior_yaw)
    r = None
    for _ in range(tries):
        r = outline_fit(M, F, category, pc, py)
        if r is None:
            return None
        dc = np.asarray(r["centre"]) - pc
        ax = np.array([math.cos(r["yaw"]), math.sin(r["yaw"])])
        ay = np.array([-ax[1], ax[0]])
        guard = OUTLINE_XY_WIN_M - OUTLINE_EDGE_GUARD_M
        at_edge = (abs(float(dc @ ax)) > guard or abs(float(dc @ ay)) > guard
                   or abs(r["yaw"] - py) > math.radians(OUTLINE_YAW_WIN_DEG - OUTLINE_EDGE_GUARD_DEG))
        if not at_edge:
            r["edge"] = False
            return r
        pc, py = np.asarray(r["centre"], float), float(r["yaw"])
    r["edge"] = True
    r["U"] = float("inf")
    return r


def outline_support(M, category: str, centre, yaw: float) -> float:
    """Share of the outer footprint's OUTLINE_SUPPORT_CELL_M cells that hold at least one module point."""
    hx, hy = outline_half(category)
    ax = np.array([math.cos(yaw), math.sin(yaw)])
    ay = np.array([-ax[1], ax[0]])
    a = (M - np.asarray(centre)) @ ax
    b = (M - np.asarray(centre)) @ ay
    na, nb = int(math.ceil(2 * hx / OUTLINE_SUPPORT_CELL_M)), int(math.ceil(2 * hy / OUTLINE_SUPPORT_CELL_M))
    ia = np.floor((a + hx) / OUTLINE_SUPPORT_CELL_M).astype(int)
    ib = np.floor((b + hy) / OUTLINE_SUPPORT_CELL_M).astype(int)
    ok = (ia >= 0) & (ia < na) & (ib >= 0) & (ib < nb)
    occ = np.zeros((na, nb), bool)
    occ[ia[ok], ib[ok]] = True
    return float(occ.mean())


def outline_accepted(fit: dict | None) -> bool:
    """Acceptance of a fit (module docstring): U <= OUTLINE_U_MAX_M, not at the window edge, support >=
    OUTLINE_SUPPORT_MIN."""
    return bool(fit is not None and math.isfinite(fit["U"]) and fit["U"] <= OUTLINE_U_MAX_M and not fit.get("edge")
                and fit.get("support", 0.0) >= OUTLINE_SUPPORT_MIN)


def outline_update(current: dict | None, fit: dict | None, step: int = 0) -> tuple[dict | None, str]:
    """(frame, decision) after a fit: "rejected" (not accepted: the frame is unchanged), "kept_frozen" (accepted within
    OUTLINE_FREEZE_M and OUTLINE_FREEZE_DEG of the current frame: unchanged) or "accepted" (the fit is the new frame)."""
    if not outline_accepted(fit):
        return current, "rejected"
    if (current is not None and math.hypot(fit["centre"][0] - current["centre"][0], fit["centre"][1] - current["centre"][1])
            <= OUTLINE_FREEZE_M and abs(fit["yaw"] - current["yaw"]) <= math.radians(OUTLINE_FREEZE_DEG)):
        return current, "kept_frozen"
    return {"centre": [float(fit["centre"][0]), float(fit["centre"][1])], "yaw": float(fit["yaw"]), "U": float(fit["U"]),
            "support": float(fit["support"]), "step": int(step)}, "accepted"


def outline_could_hide(ev: dict | None) -> bool:
    """surface-frame-1: True when ev is the placement evidence of a SURFACE region view whose
    module outline has no accepted fit ("module_frame" missing or not accepted): such a region can never be cleared.
    False for a drawer view and for a surface view on an accepted outline. Read by sitemodel.placement_could_hide."""
    if not ev:
        return False
    mf = ev.get("module_frame")
    if mf is None:
        return ev.get("region") in C.SURFACE_REGION.values()
    return not (isinstance(mf, dict) and mf.get("accepted") is True)


@dataclass
class Track:
    track_id: str
    category: str
    kind: str  # object | handle | module
    pos: np.ndarray
    sigma: np.ndarray
    yaw: float | None = None
    pull: np.ndarray | None = None
    unit: str | None = None
    first_step: int = 0
    last_step: int = 0
    status: str = OBSERVED
    region: str = "other"
    vis_estimate: float = 0.0  # robot-side visibility estimate of the last pose
    truncated: bool = False  # last pose came from a view cut by the image edge
    closed_pos: np.ndarray | None = None  # drawer handles: where it was first seen (assumed closed)
    moved_by_robot: float = 0.0  # drawer handles: opening the robot's own actions account for
    opening_unknown: bool = False  # drawer handles: an action moved it by an unknown amount
    closed_step: int = 0  # drawer handles: step of the view closed_pos comes from
    closed_inner_pos: np.ndarray | None = None  # drawer handles: innermost closed reference so far
    closed_inner_step: int = 0  # drawer handles: step of the view closed_inner_pos comes from
    pose_source: str = "this_view"  # where pos comes from (see perception.contract)
    pose_step: int = 0  # step of the view pos comes from
    history: list = field(default_factory=list)


def _reason_detail(reason) -> dict:
    """contact-invalidates-1: {"reason": reason} for a site-model action note, {} without a reason (the records
    of calls that pass none stay as before)."""
    return {} if reason is None else {"reason": str(reason)}


def _pca_axis(xy: np.ndarray):
    c = xy - xy.mean(axis=0)
    if len(c) < 3:
        return np.array([1.0, 0.0]), 1.0
    w, v = np.linalg.eigh(np.cov(c.T))
    ax = v[:, int(np.argmax(w))]
    ratio = float(w.min() / max(w.max(), 1e-12))
    return ax, ratio


def _yaw_mod_pi(ax) -> float:
    y = math.atan2(ax[1], ax[0])
    if y < -math.pi / 2:
        y += math.pi
    elif y >= math.pi / 2:
        y -= math.pi
    return y


def _quat_z(yaw: float) -> list[float]:
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


class Perception:
    """The robot's perception state: camera extrinsics, tracks, region geometry.

    observe(camera) captures a frame with the (simulated) sensor, estimates poses, associates
    them with tracks, computes region coverage, and returns an Observation (contract). If a
    SiteModel is attached, the observation is integrated into it.
"""

    def __init__(self, robot, sensor, declared_drawers: int | None = None, site_model=None):
        self.robot = robot
        self.sensor = sensor
        self.declared_drawers = declared_drawers
        self.model = site_model
        self.step = 0
        self._held_after_last = False  # grip reading after the previous noted action
        self.tracks: dict[str, Track] = {}
        self._n_tracks = 0
        self._n_obs = 0
        self.drawer_index: dict[str, int] | None = None  # handle track -> drawer index (from the left)
        self.frames: list = []  # (step, camera, CameraFrame) kept for the visual log (not serialised)
        self.keep_frames = False
        # The views where drawer-reference-2 decided differently from the earlier rule (records only, read by no
        # decision): {"step", "track_id", "rule": "outward_offset_is_opening" | "poor_view_keeps_unknown", ...}
        self.reference_events: list = []
        # scope-occluder-1: the region views where the rule coded "0" (not seen) floor points the earlier rule coded "o"
        # (records only, read by no decision): {"step", "obs_id", "region", "camera", "points", "objects": [[track_id,
        # category, gap_m], ...]} (the raised objects over those points)
        self.cover_events: list = []
        # scope-placement-1: per region view, the placement evidence for the guarded target (_placement_evidence), keyed
        # (obs_id, region). The site model takes it into its inspection records and merges it
        # (sitemodel.merge_placement); its search_scope and the guard read it (sitemodel.could_hide).
        self.placement_evidence: dict = {}
        # surface-frame-1: per surface module track, the union buffers of its views' module and free worktop points
        # ({track_id: [[M per view], [F per view]]}), its accepted outline frame (outline_update), and the per-view fit
        # records (records only: {"step", "obs_id", "track", "category", "fit", "decision", ...}).
        self._outline_acc: dict = {}
        self.outline_frames: dict = {}
        self.outline_events: list = []
        # pick-pose-3: {track_id: the depth frame (float32) and believed camera pose of the view that set the pose of
        # that pickable track}; overhead_events: the overhead checks run (records only).
        self.pose_frames: dict = {}
        self.overhead_events: list = []

    # ------------------------------------------------------------ frames
    def camera_pose(self, camera: str):
        """Believed camera pose in the world: (position, rotation; camera looks along -z)."""
        b = self.robot.belief
        if camera == "robot_eye":
            p_b, R_b = robot_eye_in_base()
            return b.base_to_world(p_b), b.R @ R_b
        if camera == "wrist_cam":
            # From the joint encoders (measured joints), not the last command (the
            # image is rendered where the arm really is, 0.7 to 3 mm from the commanded pose).
            p_t, R_t = self.robot.tcp_pose()
            p_h, R_h = wrist_cam_in_hand()
            off = p_h - np.array([0.0, 0.0, TCP_OFFSET])  # tcp frame = hand frame shifted along z
            return p_t + R_t @ off, R_t @ R_h
        raise ValueError(camera)

    @staticmethod
    def _backproject(fr, uv, d, cam_p, cam_R) -> np.ndarray:
        x = (uv[:, 0] - fr.cx) / fr.fx * d
        y = -(uv[:, 1] - fr.cy) / fr.fy * d
        pc = np.column_stack([x, y, -d])
        return pc @ cam_R.T + cam_p

    @staticmethod
    def project(fr, cam_p, cam_R, pts_w):
        """World points -> (u, v, expected depth, in_image mask)."""
        pc = (np.asarray(pts_w) - cam_p) @ cam_R
        z = -pc[:, 2]
        ok = z > 0.02
        zs = np.where(ok, z, 1.0)
        u = fr.cx + fr.fx * pc[:, 0] / zs
        v = fr.cy - fr.fy * pc[:, 1] / zs
        ok &= (u >= 1) & (u < fr.width - 1) & (v >= 1) & (v < fr.height - 1)
        return u, v, z, ok

    # ------------------------------------------------------------ estimation
    @staticmethod
    def _hull_area(u, v) -> float:
        """Area of the convex hull of image points (pixels^2), monotone chain + shoelace."""
        P = sorted(set(zip(np.round(u, 3).tolist(), np.round(v, 3).tolist())))
        if len(P) < 3:
            return 0.0

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        lo, hi = [], []
        for q in P:
            while len(lo) >= 2 and cross(lo[-2], lo[-1], q) <= 0:
                lo.pop()
            lo.append(q)
        for q in reversed(P):
            while len(hi) >= 2 and cross(hi[-2], hi[-1], q) <= 0:
                hi.pop()
            hi.append(q)
        H = lo[:-1] + hi[:-1]
        x = np.array([h[0] for h in H])
        y = np.array([h[1] for h in H])
        return float(0.5 * abs(x @ np.roll(y, -1) - y @ np.roll(x, -1)))

    def _silhouette_vis(self, corners, n_pixels, fr, cam) -> float:
        """Robot-side visibility estimate: n_pixels over the projected silhouette area of the
        catalog shape `corners` (world, believed frame), clipped to the image; in [0, 1]."""
        if cam is None:
            return 0.0
        cam_p, cam_R = cam
        pc = (np.asarray(corners) - cam_p) @ cam_R
        z = -pc[:, 2]
        if np.any(z <= 0.02):
            return 0.0  # part of the shape behind the camera: no estimate, the lowest value
        u = np.clip(fr.cx + fr.fx * pc[:, 0] / z, 0.0, fr.width)
        v = np.clip(fr.cy - fr.fy * pc[:, 1] / z, 0.0, fr.height)
        area = self._hull_area(u, v)
        if area <= 1.0:
            return 0.0
        return float(min(1.0, n_pixels / area))

    @staticmethod
    def _box_corners(c, ax, half_a, half_b, z0, z1):
        ax = np.array([ax[0], ax[1], 0.0])
        ax /= max(float(np.linalg.norm(ax)), 1e-9)
        by = np.array([-ax[1], ax[0], 0.0])
        out = []
        for sa in (-1, 1):
            for sb in (-1, 1):
                for z in (z0, z1):
                    q = np.array([c[0], c[1], 0.0]) + sa * half_a * ax + sb * half_b * by
                    q[2] = z
                    out.append(q)
        return np.array(out)

    def _object_silhouette(self, cat, cxy, cz, yaw, lying):
        sh = C.OBJECT_SHAPES[cat]
        if lying:
            fa, fb = sh["lying_footprint"]
            top, h = sh["lying_top_above_centre"], sh["lying_height"]
        else:
            fa, fb = sh["footprint"]
            top, h = sh["top_above_centre"], sh["height"]
        z1 = cz + top
        z0 = z1 - h
        if cat in C.ROUND_CATEGORIES and not lying:
            a = np.linspace(0, 2 * math.pi, ROUND_SIDES, endpoint=False)
            ring = np.column_stack([cxy[0] + 0.5 * fa * np.cos(a), cxy[1] + 0.5 * fa * np.sin(a)])
            return np.vstack([np.column_stack([ring, np.full(len(a), z0)]), np.column_stack([ring, np.full(len(a), z1)])])
        y = 0.0 if yaw is None else yaw
        return self._box_corners(cxy, (math.cos(y), math.sin(y)), 0.5 * fa, 0.5 * fb, z0, z1)

    def _object_pose(self, cat, pts, n_pixels, d_med, fr, cam=None):
        """Pose, note and robot-side visibility estimate of an object segment."""
        shape = C.OBJECT_SHAPES[cat]
        zt = float(np.percentile(pts[:, 2], 95))
        top = pts[pts[:, 2] >= zt - 0.006]
        note = ""
        top_h = shape["top_above_centre"]
        xy_src = top
        if cat == "spray_bottle":
            ax_all, _ = _pca_axis(pts[:, :2])
            proj = (pts[:, :2] - pts[:, :2].mean(0)) @ ax_all
            length = float(np.percentile(proj, 97) - np.percentile(proj, 3))
            # Lying if long seen from above, or if its top is too low for an upright bottle (an
            # upright one stands >= 13 cm above the worktop; one lying in a drawer about 8 cm).
            if length > shape["lying_min_length"] or zt - self.robot.worktop_z < 0.10:
                top_h = shape["lying_top_above_centre"]
                xy_src = pts
                note = "lying: centre biased toward the head, yaw only"
        cxy = xy_src[:, :2].mean(axis=0)
        cz = float(np.median(top[:, 2])) - top_h
        lying = bool(note)
        yaw = syaw = quat = None
        if shape["yaw"]:
            ax, ratio = _pca_axis(xy_src[:, :2])
            yaw = _yaw_mod_pi(ax)
            syaw = float(min(math.pi / 2, 0.05 + 1.5 * ratio))
            if not note:
                quat = _quat_z(yaw)
        else:
            note = "round: no yaw"
        vis = self._silhouette_vis(self._object_silhouette(cat, cxy, cz, yaw, lying), n_pixels, fr, cam)
        px = d_med / fr.fx
        sxy = 0.003 + px + 0.02 * (1 - vis)
        if lying:  # lying spray bottle: centre biased toward the head by up to 3 cm (measured)
            sxy = max(sxy, LYING_SPRAY_SIGMA_XY)
        sz = float(depth_sigma(d_med)) / math.sqrt(max(len(top), 1)) + 0.003 + (0.01 if lying else 0.0)
        pose = Pose([float(cxy[0]), float(cxy[1]), cz], [sxy, sxy, sz], yaw_rad=yaw, sigma_yaw_rad=syaw,
                    quat_wxyz=quat)
        return pose, note, vis

    def _handle_pose(self, cat, pts, n_pixels, d_med, fr, cam=None):
        """Pose, pull direction and robot-side visibility estimate of a rail handle segment."""
        zt = float(np.percentile(pts[:, 2], 95))
        top = pts[pts[:, 2] >= zt - 0.005]
        cxy = top[:, :2].mean(axis=0)
        cz = float(np.median(top[:, 2])) - C.HANDLE_HALF_THICKNESS
        ax, ratio = _pca_axis(top[:, :2])
        pull = np.array([-ax[1], ax[0]])
        to_base = self.robot.belief.t[:2] - cxy
        if pull @ to_base < 0:
            pull = -pull
        hb = C.HANDLE_HALF_THICKNESS
        corners = self._box_corners(cxy, ax, 0.5 * C.HANDLE_LENGTH.get(cat, 0.10), hb, cz - hb, cz + hb)
        vis = self._silhouette_vis(corners, n_pixels, fr, cam)
        px = d_med / fr.fx
        sxy = 0.003 + px + 0.02 * (1 - vis)
        sz = float(depth_sigma(d_med)) / math.sqrt(max(len(top), 1)) + 0.003
        yaw = _yaw_mod_pi(ax)
        pose = Pose([float(cxy[0]), float(cxy[1]), cz], [sxy, sxy, sz], yaw_rad=yaw,
                    sigma_yaw_rad=float(min(math.pi / 2, 0.03 + 1.5 * ratio)))
        return pose, np.array([pull[0], pull[1], 0.0]), vis

    def _module_pose(self, pts):
        wz = self.robot.worktop_z
        low = pts[(pts[:, 2] > wz - 0.02) & (pts[:, 2] < wz + 0.16)]
        src = low if len(low) >= 20 else pts
        cxy = np.median(src[:, :2], axis=0)
        bearing = math.degrees(math.atan2(cxy[1], cxy[0]))
        slot, best = None, 29.0
        for name, a in SLOT_ANGLES.items():
            dd = abs((bearing - a + 180) % 360 - 180)
            if dd < best:
                slot, best = name, dd
        s = MODULE_SIGMA_XY  # fixed: it does not depend on the oracle visible fraction
        return Pose([float(cxy[0]), float(cxy[1]), float(wz)], [s, s, 0.02]), bearing, slot

    # ------------------------------------------------------------ tracking
    def _new_track(self, cat, kind, pos, sigma, step) -> Track:
        self._n_tracks += 1
        t = Track(f"trk-{self._n_tracks:03d}", cat, kind, pos, sigma, first_step=step, last_step=step,
                  pose_step=step)
        if cat == "drawer_handle":
            t.closed_pos = np.array(pos, float)
            t.closed_step = step
            t.closed_inner_pos, t.closed_inner_step = np.array(pos, float), step
        self.tracks[t.track_id] = t
        return t

    def _associate(self, cat, kind, pos, sigma, used, step) -> Track:
        gate = {"object": 0.06, "handle": 0.05, "module": 0.25}[kind]
        best, bd = None, None
        for t in self.tracks.values():
            if t.category != cat or t.track_id in used or t.status == HELD:
                continue
            if cat == "drawer_handle" and t.pull is not None and t.closed_pos is not None:
                # A drawer handle only slides along its pull axis: match on the lateral and
                # vertical offsets from where it was first seen, whatever the opening.
                d = pos - t.closed_pos
                along = float(d[:2] @ t.pull[:2])
                lat = abs(float(d[0] * t.pull[1] - d[1] * t.pull[0]))
                if lat < 0.04 and abs(float(d[2])) < 0.03 and -0.04 < along < C.DRAWER_MAX_OPENING + 0.04:
                    if bd is None or lat < bd:
                        best, bd = t, lat
                    continue
            dxy = float(np.hypot(*(t.pos[:2] - pos[:2])))
            dz = abs(float(t.pos[2] - pos[2]))
            g = max(gate, 3 * float(max(sigma[0], t.sigma[0])))
            if dxy < g and dz < max(0.05, g) and (bd is None or dxy < bd):
                best, bd = t, dxy
        if best is None:
            best = self._new_track(cat, kind, pos, sigma, step)
        used.add(best.track_id)
        return best

    # ------------------------------------------------------------ regions (robot-side geometry)
    def _units(self):
        """Drawer handles grouped by drawer unit (nearest unit track), ordered left to right as
        seen from the robot. Returns {unit_track or None: [handle tracks]}."""
        units = [t for t in self.tracks.values() if t.category == "drawer_unit"]
        groups: dict = {}
        for h in self.tracks.values():
            if h.category != "drawer_handle":
                continue
            u = None
            if units:
                d = [float(np.hypot(*(h.pos[:2] - x.pos[:2]))) for x in units]
                if min(d) < 0.45:
                    u = units[int(np.argmin(d))].track_id
            h.unit = u
            groups.setdefault(u, []).append(h)
        for u, hs in groups.items():
            pull = np.mean([h.pull[:2] for h in hs], axis=0)
            facing = -pull / max(np.linalg.norm(pull), 1e-9)
            right = np.array([facing[1], -facing[0]])  # facing x z
            hs.sort(key=lambda h: float(h.pos[:2] @ right))
        return groups

    def _update_drawer_index(self):
        if self.drawer_index is not None:
            return
        groups = self._units()
        hs = [h for u, g in groups.items() for h in g]
        if len(groups) == 1 and self.declared_drawers and len(hs) == self.declared_drawers:
            self.drawer_index = {h.track_id: i for i, h in enumerate(next(iter(groups.values())))}

    def drawer_region_id(self, handle_track: str) -> str:
        if self.drawer_index and handle_track in self.drawer_index:
            return C.drawer_region(self.drawer_index[handle_track])
        return f"drawer_unit:?{handle_track}"

    def handle_for_region(self, region: str) -> Track | None:
        for h in self.tracks.values():
            if h.category == "drawer_handle" and self.drawer_region_id(h.track_id) == region:
                return h
        return None

    def region_geometries(self, with_grid: bool = False):
        """{region id: (grid points (N, 3) on the region floor, inside(p) test)} from the
        current handle and module tracks. Grid points are row-major (rows along the region's first axis: away from
        the handle for a drawer). with_grid adds (shape, pitch) as a third element. drawer_floor_interior_1, module
        docstring "Drawer floor interior": a drawer grid is built on the interior floor
        (DRAWER_INTERIOR_BEHIND_HANDLE, DRAWER_INTERIOR_HALF_WIDTH: the catalog floor without the walls); the inside
        test is unchanged. drawer_floor_interior_2 (module docstring "Drawer floor interior 2"): the returned drawer
        grid is the catalog's again (25 x 9); the interior grid (24 x 8) is kept in self.drawer_floor_pts[region id]
        for the floor band only."""
        out = {}
        floor_pts = {}  # per drawer region, the interior grid of the floor band
        for h in self.tracks.values():
            if h.category != "drawer_handle":
                continue
            pull = h.pull[:2] / np.linalg.norm(h.pull[:2])
            lat = np.array([-pull[1], pull[0]])
            fz = float(h.pos[2]) - C.DRAWER_FLOOR_BELOW_HANDLE
            built = {}
            # "grid" (placements, coverage, codes) on the catalog extent, "floor" (the band) on the interior
            for key, (a0, a1), hw in (("grid", C.DRAWER_FLOOR_BEHIND_HANDLE, C.DRAWER_FLOOR_HALF_WIDTH),
                                      ("floor", DRAWER_INTERIOR_BEHIND_HANDLE, DRAWER_INTERIOR_HALF_WIDTH)):
                na = int(round((a1 - a0 - 2 * GRID_INSET) / DRAWER_GRID_PITCH[0])) + 1
                nb = int(round((2 * hw - 2 * GRID_INSET) / DRAWER_GRID_PITCH[1])) + 1
                la = np.linspace(a0 + GRID_INSET, a1 - GRID_INSET, na)
                lb = np.linspace(-hw + GRID_INSET, hw - GRID_INSET, nb)
                A, B = np.meshgrid(la, lb, indexing="ij")
                xy = h.pos[:2] - np.outer(A.ravel(), pull) + np.outer(B.ravel(), lat)
                built[key] = (np.column_stack([xy, np.full(len(xy), fz)]),
                              ([na, nb], [float(la[1] - la[0]), float(lb[1] - lb[0])]))
            (pts, grid), floor_pts[self.drawer_region_id(h.track_id)] = built["grid"], built["floor"][0]

            def inside(p, h=h, pull=pull, lat=lat, fz=fz):
                d = h.pos[:2] - p[:2]
                a, b = float(d @ pull), float(abs(d @ lat))
                return (0.015 <= a <= 0.305 and b <= 0.085 and fz - 0.02 <= p[2] <= fz + 0.10)
            out[self.drawer_region_id(h.track_id)] = (pts, inside, grid) if with_grid else (pts, inside)
        self.drawer_floor_pts = floor_pts  # the floor band's points of each drawer region (interior)
        wz = self.robot.worktop_z
        # surface-frame-1: a surface region sits on its module's accepted outline frame (module_frame), else on the
        # module track as before (median, x axis toward the believed base); the last track of a category wins, as
        # before.
        for rid, sf in self._surface_frames().items():
            c, x_ax = sf["centre"], sf["x_ax"]
            y_ax = np.array([-x_ax[1], x_ax[0]])
            hx, hy = sf["half"]
            na = int(round((2 * hx - 2 * GRID_INSET) / SURFACE_GRID_PITCH)) + 1
            nb = int(round((2 * hy - 2 * GRID_INSET) / SURFACE_GRID_PITCH)) + 1
            la = np.linspace(-hx + GRID_INSET, hx - GRID_INSET, na)
            lb = np.linspace(-hy + GRID_INSET, hy - GRID_INSET, nb)
            A, B = np.meshgrid(la, lb, indexing="ij")
            xy = c + np.outer(A.ravel(), x_ax) + np.outer(B.ravel(), y_ax)
            grid = ([na, nb], [float(la[1] - la[0]), float(lb[1] - lb[0])])
            pts = np.column_stack([xy, np.full(len(xy), sf["z"])])

            def inside(p, c=c, x_ax=x_ax, y_ax=y_ax, hx=hx, hy=hy):
                d = p[:2] - c
                return (abs(float(d @ x_ax)) <= hx + 0.03 and abs(float(d @ y_ax)) <= hy + 0.03
                        and wz - 0.02 <= p[2] <= wz + 0.25)
            out[rid] = (pts, inside, grid) if with_grid else (pts, inside)
        return out

    # ------------------------------------------------------------ surface outline (surface-frame-1)
    def _first_track(self, category: str):
        """The first perceived track of a category (creation order), or None: the module track a surface region and its
        outline fit belong to (port fix, module docstring; p2_surf's next(t for t in tracks if t.category == cat))."""
        return next((t for t in self.tracks.values() if t.category == category), None)

    def _surface_frames(self) -> dict:
        """{surface region id: {"track", "category", "centre" (2,), "x_ax" (2,), "half", "z", "fit" (accepted outline
        frame or None)}}: the frame of each surface region, from the FIRST module track of its category (port fix of
        module docstring; until then the last track, as region_geometries did previously): its accepted outline frame
        (module docstring), else the track's position with the x axis toward the believed base."""
        out = {}
        wz = self.robot.worktop_z
        for m in self.tracks.values():
            if m.category not in C.MODULE_SURFACES:
                continue
            if C.SURFACE_REGION[m.category] in out:
                continue  # port fix: the FIRST track of the category, as the prototype's frame_of
            sd = C.MODULE_SURFACES[m.category]
            fit = self.outline_frames.get(m.track_id)
            if fit is not None:
                c = np.array(fit["centre"], float)
                x_ax = np.array([math.cos(fit["yaw"]), math.sin(fit["yaw"])])
            else:
                c = np.array(m.pos[:2], float)
                x_ax = self.robot.belief.t[:2] - m.pos[:2]
                x_ax = x_ax / max(np.linalg.norm(x_ax), 1e-9)
            out[C.SURFACE_REGION[m.category]] = {"track": m.track_id, "category": m.category, "centre": c, "x_ax": x_ax,
                                                 "half": sd["half"], "z": wz + sd["z"], "fit": fit}
        return out

    def module_frame(self, category: str) -> dict | None:
        """surface-frame-1: the accepted fitted outline frame of the surface module
        `category` (catalog module category, or its surface region id), or None. It is the frame region_geometries uses
        for that region (grid, region_of, coverage, placement evidence, exclusions); look aims read it (the module
        track's position stays the median, for association and slot naming)."""
        cat = category if category in C.SURFACE_REGION else next(
            (c for c, r in C.SURFACE_REGION.items() if r == category), None)
        sf = self._surface_frames().get(C.SURFACE_REGION[cat]) if cat is not None else None
        if sf is None or sf["fit"] is None:
            return None
        f = sf["fit"]
        return {"version": SURFACE_FRAME_VERSION, "accepted": True, "category": cat, "region": C.SURFACE_REGION[cat],
                "track_id": sf["track"], "centre": [float(f["centre"][0]), float(f["centre"][1])], "yaw": float(f["yaw"]),
                "x_axis": [float(sf["x_ax"][0]), float(sf["x_ax"][1])], "U": float(f["U"]),
                "support": float(f["support"]), "step": int(f["step"]), "half_m": [float(x) for x in sf["half"]],
                "z": float(sf["z"])}

    @staticmethod
    def _frame_block(sf: dict) -> dict:
        """The "module_frame" block of a surface view's placement evidence (records; outline_could_hide reads
        "accepted")."""
        f = sf["fit"]
        return {"version": SURFACE_FRAME_VERSION, "accepted": f is not None, "track": sf["track"],
                "category": sf["category"], "source": "outline_fit" if f is not None else "module_track",
                "centre": [round(float(sf["centre"][0]), 5), round(float(sf["centre"][1]), 5)],
                "yaw": round(float(math.atan2(sf["x_ax"][1], sf["x_ax"][0])), 6),
                "U": None if f is None else round(float(f["U"]), 5),
                "support": None if f is None else round(float(f["support"]), 4),
                "fit_step": None if f is None else int(f["step"]), "margin_m": SURFACE_PLACEMENT_MARGIN_M}

    @staticmethod
    def _drawer_deep_lattice(pts, grid, extend_m: float):
        """Deep-end extension (module docstring, "Drawer deep end"): (points (N, 3), grid, k) of a drawer region's
        grid (row-major, rows along the first axis: away from the handle) continued by k = ceil(extend_m / pitch) rows
        beyond its deepest row with the grid's own row step; the region grid's points are kept unchanged, first."""
        (na, nb), (pa, pb) = grid
        k = int(math.ceil(extend_m / pa - 1e-9))
        P = np.asarray(pts, float)
        last = P[(na - 1) * nb: na * nb]
        step = last - P[(na - 2) * nb: (na - 1) * nb]
        return (np.vstack([P] + [last + j * step for j in range(1, k + 1)]),
                ([int(na) + k, int(nb)], [float(pa), float(pb)]), k)

    @staticmethod
    def _surface_lattice(sf: dict, extend_m: float):
        """(points (N, 3), grid, k) of a surface region's grid extended by k = ceil(extend_m / SURFACE_GRID_PITCH)
        pitches on every side (the same lattice as region_geometries, continued outward)."""
        hx, hy = sf["half"]
        k = int(math.ceil(extend_m / SURFACE_GRID_PITCH - 1e-9))
        na = int(round((2 * hx - 2 * GRID_INSET) / SURFACE_GRID_PITCH)) + 1
        nb = int(round((2 * hy - 2 * GRID_INSET) / SURFACE_GRID_PITCH)) + 1
        pa = (2 * hx - 2 * GRID_INSET) / (na - 1)
        pb = (2 * hy - 2 * GRID_INSET) / (nb - 1)
        la = -hx + GRID_INSET + (np.arange(na + 2 * k) - k) * pa
        lb = -hy + GRID_INSET + (np.arange(nb + 2 * k) - k) * pb
        x_ax = np.asarray(sf["x_ax"], float)
        y_ax = np.array([-x_ax[1], x_ax[0]])
        A, B = np.meshgrid(la, lb, indexing="ij")
        xy = np.asarray(sf["centre"], float) + np.outer(A.ravel(), x_ax) + np.outer(B.ravel(), y_ax)
        pts = np.column_stack([xy, np.full(len(xy), float(sf["z"]))])
        return pts, ([na + 2 * k, nb + 2 * k], [float(pa), float(pb)]), k

    def _fit_outlines(self, fr, cam_p, cam_R, hits, step: int, obs_id: str):
        """Fit the outline of every surface module segment of this view (hits: [(segment index in fr.segments, segment,
        associated track or None)], every module segment of the frame in frame order) on the union of the views of
        the FIRST track of its category; update that track's accepted frame (outline_update). Port fix (module
        docstring): as the prototype's refine_view, the associated track does not choose the union (a partial view
        that started a second track of the category still feeds the first one); no track of the category yet: no fit.
        Robot side only: the frame's depth and segments, the believed camera pose and worktop, the catalog, the
        prior."""
        lab = None
        for k, s, seg_t in hits:
            if s.category not in C.MODULE_OUTER_FOOTPRINTS:
                continue
            t = self._first_track(s.category)
            if t is None:
                continue
            cur = self.outline_frames.get(t.track_id)
            if cur is not None:
                prior, prior_yaw = np.array(cur["centre"], float), float(cur["yaw"])
            else:  # the track's median, facing the site plan's centre (the origin of the slot bearings, _module_pose)
                prior = np.array(t.pos[:2], float)
                prior_yaw = math.atan2(-prior[1], -prior[0])
            rec = {"step": step, "obs_id": obs_id, "camera": fr.camera if hasattr(fr, "camera") else None,
                   "track": t.track_id, "category": t.category, "border": bool(s.at_image_border), "n_m_view": 0,
                   "fit": None, "decision": "rejected",
                   "segment_track": None if seg_t is None else seg_t.track_id}  # records: the associated track
            fit = None
            try:
                if lab is None:
                    lab = segment_label_image(fr)
                M, F, _top = outline_points(fr.depth, lab > 0, lab == k + 1, (fr.fx, fr.fy, fr.cx, fr.cy), cam_p, cam_R,
                                            prior, float(self.robot.worktop_z))
                acc = self._outline_acc.setdefault(t.track_id, [[], []])
                acc[0].append(outline_subsample(M, OUTLINE_MAX_M_PER_VIEW))
                acc[1].append(outline_subsample(F, OUTLINE_MAX_F_PER_VIEW))
                Mu = outline_subsample(np.vstack(acc[0]), OUTLINE_UNION_MAX_M)
                Fu = outline_subsample(np.vstack(acc[1]), OUTLINE_UNION_MAX_F)
                fit = outline_fit_robust(Mu, Fu, t.category, prior, prior_yaw)
                if fit is not None:
                    fit["support"] = outline_support(Mu, t.category, fit["centre"], fit["yaw"])
                rec["n_m_view"] = int(len(M))
            except Exception as e:  # noqa: BLE001 (never an accepted frame from an error: the region stays not clearable)
                rec["error"] = f"{type(e).__name__}: {e}"[:200]
                fit = None
            if fit is not None:
                rec["fit"] = {x: fit.get(x) for x in ("centre", "yaw", "U", "n_m", "n_f", "cost", "edge", "support")}
            new, decision = outline_update(cur, fit, step) if "error" not in rec else (cur, "rejected")
            rec["decision"] = decision
            if new is not None:
                self.outline_frames[t.track_id] = new
            self.outline_events.append(rec)

    def region_of(self, p, geoms=None) -> str:
        geoms = self.region_geometries() if geoms is None else geoms
        for rid, (_pts, inside, *_g) in geoms.items():
            if rid.startswith("drawer") and inside(p):
                return rid
        for rid, (_pts, inside, *_g) in geoms.items():
            if not rid.startswith("drawer") and inside(p):
                return rid
        return "other"

    def _coverage(self, fr, cam_p, cam_R, pts, objects=(), cover_record: list | None = None):
        """(coverage, n points, floor codes, top-layer codes). Floor codes per point: 1 seen, o
        under a perceived object resting on that floor, 0 not seen. Top-layer codes: 1 if the point
        VISIBILITY_LAYER_H above the floor point was in direct view. objects: (centre xy, yaw or None, half length,
        half width[, observed top z, catalog height[, track id, category]]) footprints of objects detected in this
        view. scope-occluder-1: an entry with its top and height codes "o" only where its room underneath
        (occluder_gap over the point's floor z) is under OCCLUDER_MIN_GAP_M; a 4-entry footprint (no height) always
        codes "o", as before. cover_record (records only): gets one dict when the rule coded "0" points that the rule
        coded "o": {"points": n, "objects": [[track id, category, gap], ...]}."""
        u, v, z, ok = self.project(fr, cam_p, cam_R, pts)
        n_in = int(ok.sum())
        if n_in < MIN_COVER_POINTS_IN_VIEW:
            return None
        meas = fr.depth[v[ok].astype(int), u[ok].astype(int)]
        ray = pts[ok] - cam_p
        rz = np.abs(ray[:, 2]) / np.linalg.norm(ray, axis=1)
        tol = np.minimum(REACH_TOL / np.maximum(rz, 0.2), 0.10)
        # The line of sight must stop near the floor: not more than 3 cm above it (something on
        # it, still counted as seen) and not more than 1.2 cm below it (then there is no floor
        # where the robot thinks the drawer is, e.g. the drawer did not open as far as believed).
        tol_down = np.minimum(0.012 / np.maximum(rz, 0.2), 0.04)
        reached = (meas > 0) & (meas >= z[ok] - tol) & (meas <= z[ok] + tol_down)
        seen = np.zeros(len(pts), bool)
        seen[np.flatnonzero(ok)[reached]] = True
        under = np.zeros(len(pts), bool)  # under an object resting on this floor: "o"
        raised = np.zeros(len(pts), bool)  # under a raised object (not an exclusion; records only)
        raised_by = []
        for ob in objects:
            c, yaw, ha, hb = ob[:4]
            d = pts[:, :2] - np.asarray(c)[:2]
            if yaw is None:
                foot = np.hypot(d[:, 0], d[:, 1]) <= ha
            else:
                ax = np.array([math.cos(yaw), math.sin(yaw)])
                foot = (np.abs(d @ ax) <= ha) & (np.abs(d @ np.array([-ax[1], ax[0]])) <= hb)
            if len(ob) >= 6 and ob[4] is not None and ob[5] is not None:
                gap = occluder_gap(float(ob[4]), float(ob[5]), pts[:, 2])  # scope-occluder-1
                rests = gap < OCCLUDER_MIN_GAP_M
                under |= foot & rests
                up = foot & ~rests
                if up.any():
                    raised |= up
                    raised_by.append((ob, up, float(np.min(gap[up]))))
            else:
                under |= foot
        code = "".join("1" if s_ else ("o" if o_ else "0") for s_, o_ in zip(seen, under))
        if cover_record is not None:
            rewritten = raised & ~under & ~seen  # "o" under the earlier rule, "0" now
            if rewritten.any():
                cover_record.append({"points": int(rewritten.sum()), "objects": [
                    [ob[6] if len(ob) > 6 else None, ob[7] if len(ob) > 7 else None, round(g, 4)]
                    for ob, up, g in raised_by if (up & rewritten).any()]})
        # Top layer: nothing in front of the point VISIBILITY_LAYER_H above the floor point.
        top = pts + np.array([0.0, 0.0, C.VISIBILITY_LAYER_H])
        u2, v2, z2, ok2 = self.project(fr, cam_p, cam_R, top)
        vis = np.zeros(len(pts), bool)
        if ok2.any():
            m2 = fr.depth[v2[ok2].astype(int), u2[ok2].astype(int)]
            ray2 = top[ok2] - cam_p
            rz2 = np.abs(ray2[:, 2]) / np.linalg.norm(ray2, axis=1)
            tol2 = np.minimum(0.01 / np.maximum(rz2, 0.2), 0.04)
            vis[np.flatnonzero(ok2)] = (m2 > 0) & (m2 >= z2[ok2] - tol2)
        code_top = "".join("1" if x else "0" for x in (vis | seen))
        return float(seen.sum()) / len(pts), len(pts), code, code_top

    # ------------------------------------------------------------ placement evidence (scope-placement-1)
    def _measured_floor(self, fr, cam_p, cam_R, pts, floor_rect=None, stats: dict | None = None):
        """(floor z, supporting rays) measured in this view, or None. The rays toward the region's grid points stop on
        the first surface; those that stop within FLOOR_SEARCH_M of the believed floor z are candidates, and the
        floor is the median of the lowest FLOOR_BAND_M band (above their 5th percentile): the lowest surface seen
        there (floor_band). floor-env-1 (records and the wrist envelope; the result is unchanged): stats,
        a dict, gets "n_pts" (grid points), "n_in" (those projecting into the image), "grid_frac" (n_in / n_pts) and
        floor_band's band quantities; "edge_med_m" is measured against floor_rect, the region's believed floor
        rectangle (grid_rect's tuple; grid_rect(pts, GRID_INSET) when not given)."""
        u, v, z, ok = self.project(fr, cam_p, cam_R, pts)
        idx = np.flatnonzero(ok)
        if stats is not None:
            stats.update(n_pts=int(len(pts)), n_in=int(len(idx)),
                         grid_frac=float(len(idx)) / len(pts) if len(pts) else None)
        if len(idx) < FLOOR_MIN_POINTS:
            return None
        meas = fr.depth[v[idx].astype(int), u[idx].astype(int)]
        good = meas > 0
        if good.sum() < FLOOR_MIN_POINTS:
            return None
        q = cam_p + (pts[idx[good]] - cam_p) * (meas[good] / z[idx[good]])[:, None]
        fz = float(pts[0, 2])
        edge = None
        if stats is not None:
            rect = grid_rect(pts, GRID_INSET) if floor_rect is None else floor_rect
            edge = rect_edge_distance(rect, np.asarray(pts, float)[idx[good], :2])
        return floor_band(q[:, 2], fz, edge, stats)

    def _open_drawers_over(self, pts) -> list:
        """floor-env-1 (drawer-over-surface condition): the drawer handle tracks believed open (opening
        unknown, or a believed opening, Perception.opening, of more than DRAWER_BELIEVED_CLOSED_M) whose floor footprint
        (catalog DRAWER_FLOOR_BEHIND_HANDLE x +-DRAWER_FLOOR_HALF_WIDTH behind the handle track's believed position,
        against its pull) overlaps the plan rectangle of the grid pts with a positive area. Robot side only: the handle
        tracks and the catalog."""
        if len(pts) < 2:
            return []
        g = grid_rect(pts, 0.0)
        a0, a1 = C.DRAWER_FLOOR_BEHIND_HANDLE
        hw = C.DRAWER_FLOOR_HALF_WIDTH
        out = []
        for h in self.tracks.values():
            if h.category != "drawer_handle" or h.pull is None:
                continue
            pull = np.asarray(h.pull[:2], float)
            n = float(np.linalg.norm(pull))
            if not n > 0.0:
                continue
            pull = pull / n
            lat = np.array([-pull[1], pull[0]])
            op = None if h.closed_pos is None else self.opening(h.track_id)
            if not (h.opening_unknown or (op is not None and op > DRAWER_BELIEVED_CLOSED_M)):
                continue
            foot = (np.asarray(h.pos[:2], float) - a1 * pull - hw * lat, pull, lat, float(a1 - a0), 2.0 * hw)
            if rects_overlap(foot, g):
                out.append(h.track_id)
        return out

    @staticmethod
    def _resting_objects(detections, rid, f_used, f_err, modes) -> list:
        """The objects of this view perceived in region rid, each with its surely-occupied footprint (the catalog
        footprint shrunk by the pose uncertainty, as for the grid codes) and, per mode, whether it RESTS on the
        floor: no room under it for the target in that mode, (observed top - floor) - catalog height under the mode's
        height minus PLACEMENT_SETTLE_M, the floor error and the pose's z error (OBJECT_Z_ERR_M). An object of
        another region (an open drawer over the tray, the tray under a drawer) never counts for this region. As in the
        module docstring, a detection of the guarded target (the sponge) is also read as standing on its long
        edge (catalog on-edge height, 0.060 m) for the upright rest test (it rests if either reading does), and its
        record carries gap_on_edge."""
        out = []
        for d in detections:
            sh = C.OBJECT_SHAPES.get(d.category)
            if sh is None or d.region != rid:
                continue
            lying = d.orientation_note.startswith("lying")
            top_z, height = occluder_top_and_height(d.category, float(d.pose.position_m[2]), lying)
            e_obj = OBJECT_Z_ERR_M[0] if d.pose_source == "this_view" else OBJECT_Z_ERR_M[1]
            gap = (top_z - f_used) - height
            fl, fs = sh["lying_footprint"] if lying else sh["footprint"]
            sg = float(d.pose.sigma_m[0])
            m_long, m_side = (LYING_SPRAY_SIGMA_XY, 0.01) if lying else (min(sg, 0.03), min(sg, 0.03))
            rnd = d.pose.yaw_rad is None or not sh["yaw"]
            fp = [round(float(d.pose.position_m[0]), 5), round(float(d.pose.position_m[1]), 5),
                  None if rnd else round(float(d.pose.yaw_rad), 5),
                  round(max(0.5 * min(fl, fs) - m_side, 0.005) if rnd else max(0.5 * fl - m_long, 0.005), 5),
                  round(0.0 if rnd else max(0.5 * fs - m_side, 0.005), 5)]
            rec = {"track": d.track_id, "category": d.category, "fps": [fp], "gap": round(float(gap), 5),
                   "rests": {m: bool(gap < H - PLACEMENT_SETTLE_M - f_err - e_obj) for m, (_L, _W, H) in modes.items()}}
            if d.category == OCCLUDER_TARGET_CATEGORY and "upright" in modes:
                # The robot does not estimate the target's orientation; a detected
                # target is also read as standing on its long edge (catalog on-edge height) for the upright rest test.
                # Records: gap_on_edge.
                gap_edge = (top_z - f_used) - max(min(sh["footprint"]), sh["height"])  # on-edge height 0.060 m
                rec["rests"]["upright"] = rec["rests"]["upright"] or bool(
                    gap_edge < modes["upright"][2] - PLACEMENT_SETTLE_M - f_err - e_obj)
                rec["gap_on_edge"] = round(float(gap_edge), 5)
            out.append(rec)
        return out

    def _placement_evidence(self, fr, cam_p, cam_R, pts, grid, rid, detections, floor_pts=None,
                            surface: dict | None = None, deep_end: dict | None = None) -> dict:
        """scope-placement-1, module docstring: what this view says about each candidate placement of
        the guarded target on the region floor. For each mode (placement_modes) and each placement centre of a
        half-pitch lattice (the grid's own axes), with every yaw of PLACEMENT_YAWS_DEG whose footprint fits on the
        floor: 'v' the view would have shown the target there (the visible pixels of its top face at two heights
        reach both recogniser thresholds raised by PLACEMENT_MARGIN, against the whole box silhouette), at every
        fitting yaw and both heights; 'x' a perceived object of this region RESTS there (no room under it for the
        target in that mode, with the floor and pose errors declared); '-' neither, the centre in the image; '.'
        neither, outside the image; ' ' no fitting yaw (not a placement). Robot side only: the frame's depth, the
        believed camera pose, the grid, the detections and the catalog. surface-frame-1: floor_pts (the region's own
        grid) measures the floor when pts is an extended lattice; surface = {"k": pitches the lattice was extended by
        on every side, "margin": m, "block": the module_frame block}: a placement more than m beyond its mode's
        centre range on the unextended grid is not a candidate (coded ' ' unless 'v', which later merges may use),
        and the block is recorded as "module_frame". Deep-end extension (module docstring, "Drawer deep end";
        only when DRAWER_DEEP_END_ACTIVE): deep_end = {"k": rows the drawer lattice was extended by at its deep end,
        "margin": m}: a placement more than m beyond its mode's deepest centre on the region grid is not a candidate
        (' ' unless 'v'); recorded as "deep_end". Module docstring, "Low level beyond": at the low test level a
        top-face sample is shown only when meas >= z + PLACEMENT_BEYOND_SIGMAS x depth_sigma(z), rule
        PLACEMENT_LOW_RULE_VERSION ("low_beyond_1"), recorded under the evidence's "low_rule" key; the high level
        keeps meas >= z - (PLACEMENT_DEPTH_TOL_M + 2.5 depth_sigma(z))."""
        target = OCCLUDER_TARGET_CATEGORY
        (na, nb), (pa, pb) = grid
        if na < 2 or nb < 2 or pa <= 0 or pb <= 0:
            return {"version": SCOPE_COVER_VERSION, "category": target, "error": "grid too small"}
        o = pts[0, :2].astype(float)
        ea = (pts[nb, :2] - pts[0, :2]) / pa
        eb = (pts[1, :2] - pts[0, :2]) / pb
        fz = float(pts[0, 2])
        # floor-env-1 (module docstring, "Wrist floor envelope"): the floor is measured on the region's own
        # grid, against its believed floor rectangle; a camera of FLOOR_UNMEASURED_CAMERAS never measures (empty now); a
        # camera of FLOOR_ENVELOPE_CAMERAS keeps the measurement only inside the envelope; robot_eye unchanged.
        camera = getattr(fr, "camera", None)
        fpts = pts if floor_pts is None else floor_pts
        fstats: dict = {}
        mf = None
        if camera not in FLOOR_UNMEASURED_CAMERAS:
            mf = self._measured_floor(fr, cam_p, cam_R, fpts, floor_rect=grid_rect(fpts, GRID_INSET), stats=fstats)
        valid = mf is not None and abs(mf[0] - fz) <= FLOOR_BELIEF_ERR_M
        open_over = self._open_drawers_over(fpts) if rid in C.SURFACE_REGION.values() else []
        checks = floor_envelope_checks(valid, fstats.get("edge_med_m"), fstats.get("band_iqr_m"),
                                       fstats.get("grid_frac"), open_over)
        applies = camera in FLOOR_ENVELOPE_CAMERAS
        keep = all(checks.values()) if applies else valid
        envelope = {"version": FLOOR_ENVELOPE_VERSION, "camera": camera, "applies": bool(applies),
                    "unmeasured_camera": camera in FLOOR_UNMEASURED_CAMERAS,
                    "edge_med_m": None if fstats.get("edge_med_m") is None else round(fstats["edge_med_m"], 5),
                    "band_iqr_m": None if fstats.get("band_iqr_m") is None else round(fstats["band_iqr_m"], 5),
                    "grid_frac": None if fstats.get("grid_frac") is None else round(fstats["grid_frac"], 4),
                    "n_in": fstats.get("n_in"), "n_pts": fstats.get("n_pts"), "n_band": fstats.get("n_band"),
                    "checks": checks, "open_drawers_over": list(open_over), "keep": bool(keep)}
        if valid and keep:
            f_used, f_err, measured, support = mf[0], FLOOR_MEASURED_ERR_M, True, mf[1]
        else:
            f_used, f_err, measured, support = fz, FLOOR_BELIEF_ERR_M, False, (0 if mf is None else mf[1])
        ext_a, ext_b = (na - 1) * pa, (nb - 1) * pb
        fa, fb = pa / 4.0, pb / 4.0  # fine lattice of the top-face samples; placement centres every 2 fine steps
        ia0, ib0 = -int(math.ceil(GRID_INSET / fa - 1e-9)), -int(math.ceil(GRID_INSET / fb - 1e-9))
        ni = int(math.floor((ext_a + GRID_INSET) / fa + 1e-9)) - ia0 + 1
        nj = int(math.floor((ext_b + GRID_INSET) / fb + 1e-9)) - ib0 + 1
        FA, FB = np.meshgrid((np.arange(ni) + ia0) * fa, (np.arange(nj) + ib0) * fb, indexing="ij")
        fxy = o + np.outer(FA.ravel(), ea) + np.outer(FB.ravel(), eb)
        modes = placement_modes(target)
        objects = self._resting_objects(detections, rid, f_used, f_err, modes)
        out_modes = {}
        for mode, (L, W, H) in modes.items():
            if mode == "upright" and mode in PLACEMENT_MODES_ACTIVE:
                # K1 only: the face-aware upright test of perception.upright_faces on this view's floor (f_used, f_err)
                # and resting objects, margin PLACEMENT_MARGIN_UPRIGHT; same lattice and record layout as below. An
                # error raises: the caller records an evidence error (the region is not cleared).
                rec = dict(_upright_faces().upright_evidence(sys.modules[__name__], fr, cam_p, cam_R, pts, grid,
                                                             f_used, f_err, objects, PLACEMENT_MARGIN_UPRIGHT))
                rec["active"] = True
                if surface is not None:
                    rec = _surface_mask_record(rec, W, grid, surface)
                if deep_end is not None:
                    rec = _deep_end_mask_record(rec, W, grid, deep_end)
                out_modes[mode] = rec
                continue
            # Placement centres (multiples of half a pitch) where some yaw could fit.
            ka = np.arange(int(math.ceil((-GRID_INSET + 0.5 * W) / (pa / 2) - 1e-9)),
                           int(math.floor((ext_a + GRID_INSET - 0.5 * W) / (pa / 2) + 1e-9)) + 1)
            kb = np.arange(int(math.ceil((-GRID_INSET + 0.5 * W) / (pb / 2) - 1e-9)),
                           int(math.floor((ext_b + GRID_INSET - 0.5 * W) / (pb / 2) + 1e-9)) + 1)
            if len(ka) == 0 or len(kb) == 0:
                out_modes[mode] = {"active": mode in PLACEMENT_MODES_ACTIVE, "k0": [0, 0], "n": [0, 0],
                                   "step": [pa / 2, pb / 2], "code": ""}
                continue
            KA, KB = np.meshgrid(ka, kb, indexing="ij")
            ac, bc = KA.ravel() * (pa / 2), KB.ravel() * (pb / 2)
            cxy = o + np.outer(ac, ea) + np.outer(bc, eb)
            ci, cj = 2 * KA.ravel() - ia0, 2 * KB.ravel() - ib0
            n_c = len(ac)
            fits = np.zeros((n_c, len(PLACEMENT_YAWS_DEG)), bool)
            # per level, over the fitting yaws: the smallest share (visible top pixels / box silhouette) and visible top
            # pixels
            min_share = {x: np.full(n_c, np.inf) for x in PLACEMENT_TEST_LEVELS}
            min_px = {x: np.full(n_c, np.inf) for x in PLACEMENT_TEST_LEVELS}
            for iy, ydeg in enumerate(PLACEMENT_YAWS_DEG):
                yaw = math.radians(ydeg)
                c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
                xa, xb = 0.5 * L * c + 0.5 * W * s, 0.5 * L * s + 0.5 * W * c
                fit = ((ac - xa >= -GRID_INSET - 1e-9) & (ac + xa <= ext_a + GRID_INSET + 1e-9)
                       & (bc - xb >= -GRID_INSET - 1e-9) & (bc + xb <= ext_b + GRID_INSET + 1e-9))
                fits[:, iy] = fit
            any_fit = fits.any(axis=1)
            lv = {"low": f_used - f_err + H - PLACEMENT_SETTLE_M, "high": f_used + f_err + H}
            for lname in PLACEMENT_TEST_LEVELS:
                z_top = lv[lname]
                P = np.column_stack([fxy, np.full(len(fxy), z_top)])
                u, v, z, ok = self.project(fr, cam_p, cam_R, P)
                idx = np.flatnonzero(ok)
                vw = np.zeros(len(P))
                if len(idx):
                    meas = fr.depth[v[idx].astype(int), u[idx].astype(int)]
                    rz = cam_p[2] - P[idx, 2]
                    tol = PLACEMENT_DEPTH_TOL_M + 2.5 * depth_sigma(z[idx])
                    need = (z[idx] + PLACEMENT_BEYOND_SIGMAS * depth_sigma(z[idx]) if lname == "low"
                            else z[idx] - tol)  # low_beyond_1
                    shown = (meas > 0) & (meas >= need) & (rz > 0)
                    vw[idx] = np.where(shown, fa * fb * fr.fx * fr.fy * np.maximum(rz, 0.0) / z[idx] ** 3, 0.0)
                VW = vw.reshape(ni, nj)
                for iy, ydeg in enumerate(PLACEMENT_YAWS_DEG):
                    sel = np.flatnonzero(fits[:, iy])
                    if len(sel) == 0:
                        continue
                    yaw = math.radians(ydeg)
                    di, dj = _rect_taps(L, W, yaw, fa, fb)
                    II, JJ = ci[sel, None] + di[None, :], cj[sel, None] + dj[None, :]
                    inr = (II >= 0) & (II < ni) & (JJ >= 0) & (JJ < nj)
                    px = np.where(inr, VW[np.clip(II, 0, ni - 1), np.clip(JJ, 0, nj - 1)], 0.0).sum(axis=1)
                    ul = np.tile(math.cos(yaw) * ea + math.sin(yaw) * eb, (len(sel), 1))
                    uw = np.tile(-math.sin(yaw) * ea + math.cos(yaw) * eb, (len(sel), 1))
                    sil = box_silhouette_px(fr, cam_p, cam_R, cxy[sel], ul, uw, L, W, z_top, H)
                    min_px[lname][sel] = np.minimum(min_px[lname][sel], px)
                    min_share[lname][sel] = np.minimum(min_share[lname][sel], np.where(
                        np.isfinite(sil) & (sil > 0), px / np.where(sil > 0, sil, 1.0), 0.0))
            visible = any_fit.copy()
            for lname in PLACEMENT_TEST_LEVELS:
                visible &= (min_px[lname] >= PLACEMENT_MARGIN * PLACEMENT_MIN_PX) & (
                    min_share[lname] >= PLACEMENT_MARGIN * PLACEMENT_MIN_FRAC)
            excl = placement_excluded(cxy, objects, mode, 0.5 * W) & any_fit
            uc, vc, zc, okc = self.project(fr, cam_p, cam_R, np.column_stack([cxy, np.full(n_c, f_used)]))
            code = np.where(~any_fit, " ", np.where(excl, "x", np.where(visible, "v", np.where(okc, "-", "."))))
            n_masked = n_deep = None
            if surface is not None:
                code, n_masked = _surface_beyond(ac, bc, code, W, grid, surface)
            if deep_end is not None:
                code, n_deep = _deep_end_beyond(ac, code, W, grid, deep_end)
            out_modes[mode] = {"active": mode in PLACEMENT_MODES_ACTIVE, "k0": [int(ka[0]), int(kb[0])],
                               "n": [int(len(ka)), int(len(kb))], "step": [pa / 2, pb / 2], "r_in": 0.5 * W,
                               # records: per level and placement, the smallest share (percent) and visible top pixels
                               # over the fitting yaws (-1: no fitting yaw)
                               "share_pct": {x: [int(math.floor(100 * min(v, 9.99))) if f else -1 for v, f in zip(
                                   min_share[x], any_fit)] for x in PLACEMENT_TEST_LEVELS},
                               "px": {x: [int(min(v, 99999)) if f else -1 for v, f in zip(min_px[x], any_fit)]
                                      for x in PLACEMENT_TEST_LEVELS},
                               "code": "".join(code)}
            if n_masked is not None:
                out_modes[mode]["beyond_margin"] = n_masked  # records: placements no longer candidates (were not 'v')
            if n_deep is not None:
                out_modes[mode]["beyond_deep_end"] = n_deep  # records: the same, beyond the deep-end extension
        out = {"version": SCOPE_COVER_VERSION, "category": target, "region": rid,
               "frame": [round(float(x), 6) for x in (o[0], o[1], ea[0], ea[1], eb[0], eb[1])],
               "floor_estimate": {"believed_z": round(fz, 5), "used_z": round(float(f_used), 5), "measured": measured,
                                  "support": int(support), "err_m": f_err, "envelope": envelope},
               "modes": out_modes, "objects": objects, "low_rule": PLACEMENT_LOW_RULE_VERSION}
        if surface is not None:
            out["module_frame"] = dict(surface["block"])
        if deep_end is not None:
            out["deep_end"] = {"version": DRAWER_DEEP_END_VERSION, "e_m": float(deep_end["margin"]),
                               "rows": int(deep_end["k"])}
        return out

    # ------------------------------------------------------------ the main call
    def observe(self, camera: str) -> Observation:
        self.step += 1
        step = self.step
        fr = self.sensor.capture(camera)
        if self.keep_frames:
            self.frames.append((step, camera, fr))
        cam_p, cam_R = self.camera_pose(camera)
        b = self.robot.belief
        prov = Provenance(camera, step, fr.t_s, [float(x) for x in cam_p], [float(x) for x in -cam_R[:, 2]],
                          [float(x) for x in self.robot.q_cmd],
                          {k: v for k, v in b.to_dict().items()})  # robot_q_cmd: commanded joints
        self._n_obs += 1
        obs = Observation(f"obs-{self._n_obs:03d}", prov)
        used: set = set()
        seen_objects = []
        module_track = {}  # surface-frame-1: index in fr.segments -> the module track it was associated with
        seg_index = {id(s): k for k, s in enumerate(fr.segments)}
        # Modules and handles first (they define the regions), then objects.
        order = sorted(fr.segments, key=lambda s: {"module": 0, "handle": 1, "object": 2}[s.kind])
        for s in order:
            if len(s.depth) < 10:
                continue
            pts = self._backproject(fr, s.uv, s.depth, cam_p, cam_R)
            d_med = float(np.median(s.depth))
            if s.kind == "module":
                pose, bearing, slot = self._module_pose(pts)
                t = self._associate(s.category, "module", np.array(pose.position_m), np.array(pose.sigma_m), used, step)
                self._update_track(t, pose, step, 0.0, s.at_image_border)
                obs.modules.append(ModuleObservation(t.track_id, s.category, pose, bearing, slot,
                                                     None, s.n_pixels, step, camera))
                module_track[seg_index[id(s)]] = t
            elif s.kind == "handle":
                pose, pull, vis = self._handle_pose(s.category, pts, s.n_pixels, d_med, fr, (cam_p, cam_R))
                if s.at_image_border:
                    pose.sigma_m = [x + 0.03 for x in pose.sigma_m]
                t = self._associate(s.category, "handle", np.array(pose.position_m), np.array(pose.sigma_m), used, step)
                old_sigma = float(t.sigma[0])
                if self._update_track(t, pose, step, vis, s.at_image_border):
                    t.pull = pull
                    if t.closed_pos is not None and not s.at_image_border:
                        self._drawer_view(t, old_sigma, step)
                pose = Pose([float(x) for x in t.pos], [float(x) for x in t.sigma], yaw_rad=t.yaw)
                pull = t.pull
                obs.handles.append(HandleObservation(t.track_id, s.category, pose, [float(x) for x in pull],
                                                     None, vis, s.n_pixels, step, camera,
                                                     self._emitted_source(t, step), t.pose_step))
            else:
                pose, note, vis = self._object_pose(s.category, pts, s.n_pixels, d_med, fr, (cam_p, cam_R))
                if s.at_image_border:
                    pose.sigma_m = [x + 0.03 for x in pose.sigma_m]
                    note = (note + "; " if note else "") + "cut by the image edge"
                seen_objects.append((s, pose, note, vis))
        self._units()
        self._update_drawer_index()
        for ho in obs.handles:
            ho.unit_track = self.tracks[ho.track_id].unit
        # surface-frame-1: the surface outlines are fitted before the regions of this view are built. Port fix:
        #  every module segment of the frame, in frame order (as the prototype's refine_view), each into
        # the union of the first track of its category
        module_hits = [(k, s, module_track.get(k)) for k, s in enumerate(fr.segments) if s.kind == "module"]
        self._fit_outlines(fr, cam_p, cam_R, module_hits, step, obs.obs_id)
        geoms = self.region_geometries(with_grid=True)
        pose_frame = None  # pick-pose-3: this view's depth, kept for the pickable tracks whose pose it sets
        for s, pose, note, vis in seen_objects:
            p = np.array(pose.position_m)
            t = self._associate(s.category, "object", p, np.array(pose.sigma_m), used, step)
            if not self._update_track(t, pose, step, vis, s.at_image_border):
                pose = Pose([float(x) for x in t.pos], [float(x) for x in t.sigma], yaw_rad=t.yaw,
                            sigma_yaw_rad=pose.sigma_yaw_rad, quat_wxyz=pose.quat_wxyz)
                p = t.pos
            elif s.category in PICKABLE_CATEGORIES:
                if pose_frame is None:
                    pose_frame = self._pose_frame(fr, step, camera, obs.obs_id, cam_p, cam_R)
                if pose_frame:
                    self.pose_frames[t.track_id] = pose_frame
                else:  # a frame without a depth image: no pose frame (overhead_check: not evaluated)
                    self.pose_frames.pop(t.track_id, None)
            t.region = self.region_of(p, geoms)
            obs.detections.append(Detection(t.track_id, s.category, pose, vis, s.n_pixels,
                                            t.region, step, camera, note, self._emitted_source(t, step),
                                            t.pose_step))
        # Floor points under an object detected in this view (its catalog footprint, inscribed circle, at its estimated
        # centre) are not hiding places (declared assumption: a target is not underneath a perceived object RESTING on
        # that floor). scope-occluder-1: each footprint carries the object's observed top and catalog height; _coverage
        # codes "o" only where the room under it is less than OCCLUDER_MIN_GAP_M (module docstring).
        objects = []
        for d in obs.detections:
            sh = C.OBJECT_SHAPES.get(d.category)
            if sh is None:
                continue
            lying = d.orientation_note.startswith("lying")
            fa, fb = sh["lying_footprint"] if lying else sh["footprint"]
            top_z, height = occluder_top_and_height(d.category, float(d.pose.position_m[2]), lying)
            # Shrunk by the pose uncertainty: only points surely under the object are excluded. A
            # lying spray's centre is uncertain along its length (head bias), not across it.
            sg = float(d.pose.sigma_m[0])
            m_long, m_side = (LYING_SPRAY_SIGMA_XY, 0.01) if lying else (min(sg, 0.03), min(sg, 0.03))
            if d.pose.yaw_rad is None or not sh["yaw"]:
                objects.append((d.pose.position_m, None, max(0.5 * min(fa, fb) - m_side, 0.005), 0.0,
                                top_z, height, d.track_id, d.category))
            else:
                objects.append((d.pose.position_m, float(d.pose.yaw_rad), max(0.5 * fa - m_long, 0.005),
                                max(0.5 * fb - m_side, 0.005), top_z, height, d.track_id, d.category))
        surfaces = self._surface_frames()  # the frames geoms were built on
        for rid, (pts, _inside, grid) in geoms.items():
            rewritten: list = []
            cov = self._coverage(fr, cam_p, cam_R, pts, objects, rewritten)
            if cov is None:
                continue
            if rewritten:  # records only
                self.cover_events.append({"step": step, "obs_id": obs.obs_id, "region": rid, "camera": camera,
                                          **rewritten[0]})
            sf = surfaces.get(rid)
            try:  # scope-placement-1: read by the site model (sitemodel.could_hide)
                if sf is None and DRAWER_DEEP_END_ACTIVE:  # deep-end extension (off)
                    pts_x, grid_x, k = self._drawer_deep_lattice(pts, grid, DRAWER_DEEP_END_M)
                    ev = self._placement_evidence(fr, cam_p, cam_R, pts_x, grid_x, rid, obs.detections, floor_pts=pts,
                                                  deep_end={"k": k, "margin": DRAWER_DEEP_END_M})
                elif sf is None:  # catalog placement grid, floor band on the interior grid
                    ev = self._placement_evidence(fr, cam_p, cam_R, pts, grid, rid, obs.detections,
                                                  floor_pts=self.drawer_floor_pts.get(rid))
                else:  # surface-frame-1: extended lattice, floor measured on the region's grid, 3 cm margin
                    pts_x, grid_x, k = self._surface_lattice(sf, SURFACE_EVIDENCE_EXTENSION_M)
                    ev = self._placement_evidence(fr, cam_p, cam_R, pts_x, grid_x, rid, obs.detections, floor_pts=pts,
                                                  surface={"k": k, "margin": SURFACE_PLACEMENT_MARGIN_M,
                                                           "block": self._frame_block(sf)})
            except Exception as e:  # noqa: BLE001 (never absence: an evidence error leaves the region not cleared)
                ev = {"version": SCOPE_COVER_VERSION, "category": OCCLUDER_TARGET_CATEGORY, "region": rid,
                      "error": f"{type(e).__name__}: {e}"[:200]}
                if sf is not None:
                    ev["module_frame"] = self._frame_block(sf)
            self.placement_evidence[(obs.obs_id, rid)] = ev
            seen = [d for d in obs.detections if d.region == rid]
            obs.regions.append(InspectedRegion(rid, step, camera, cov[0], cov[1],
                                               [d.track_id for d in seen], sorted({d.category for d in seen}),
                                               grid_shape=list(grid[0]), grid_pitch_m=list(grid[1]),
                                               grid_seen=cov[2], grid_seen_top=cov[3]))
        if self.model is not None:
            self.model.integrate(obs, self)
        return obs

    def _update_track(self, t: Track, pose: Pose, step: int, vis: float, truncated: bool = False) -> bool:
        """Refresh a track with a new estimate. A view cut by the image edge does not replace a
        pose taken from a whole view (it confirms the track only). Returns True if the pose was
        replaced."""
        new = t.first_step == step and t.last_step == step and t.history == []
        t.last_step = step
        t.status = OBSERVED
        t.history.append(step)
        if truncated and not new and not t.truncated:
            return False
        t.pos = np.array(pose.position_m, float)
        t.sigma = np.array(pose.sigma_m, float)
        t.yaw = pose.yaw_rad
        t.vis_estimate = vis
        t.truncated = truncated
        t.pose_source, t.pose_step = "this_view", step
        return True

    @staticmethod
    def _emitted_source(t: Track, step: int) -> str:
        """Provenance of the pose a record carries at this step."""
        if t.pose_step == step and t.pose_source == "this_view":
            return "this_view"
        return "dead_reckoned_after_action" if t.pose_source == "dead_reckoned_after_action" else "earlier_view"

    # ------------------------------------------------------------ drawer closed reference
    @staticmethod
    def sigma_along_pull(t: Track) -> float:
        """1-sigma uncertainty of a handle track's position along its pull direction (from its xy sigma)."""
        u = np.asarray(t.pull[:2], float)
        n = float(np.linalg.norm(u))
        if not n > 0.0:
            return float("inf")
        u = u / n
        return float(math.hypot(u[0] * float(t.sigma[0]), u[1] * float(t.sigma[1])))

    def _drawer_view(self, t: Track, old_sigma: float, step: int):
        """drawer-reference-2 (module docstring): a whole view (not cut by the image edge) has just
        replaced the pose of drawer handle t; old_sigma is the sigma of the pose it replaced. What the view does to the
        drawer's opening and to its closed reference:
        - opening unknown: a view whose handle sigma along the pull is at most UNKNOWN_CLEAR_SIGMA_MAX_M measures the
          opening and clears the flag; a poorer view keeps the flag (any whole view cleared it);
        - drawer believed closed (|moved_by_robot| < 5 mm): a handle more than CLOSED_REBASE_MAX_M outward of the closed
          reference (closed_pos, or the innermost reference closed_inner_pos when that is further in) is a measured
          opening, and the reference is not replaced (replaced whatever the offset); the reference is then the
          further-in of the two. Otherwise a view no less precise than the last one refines the reference, as before,
          and an inward one also moves closed_inner_pos;
        - drawer the robot believes open: nothing, as before.
        A view of a drawer whose opening is unknown measures from the further-in reference
        too (closed_pos goes back to closed_inner_pos when it walked out of it, event "unknown_view_from_inner"):
        before, an outward walk of closed_pos (at most CLOSED_REBASE_MAX_M, recorded at most 1.3 cm) was subtracted
        from the opening measured after a failed gesture or a short close, so a drawer 3 to 5 cm out could read under
        OPEN_BELIEF_M (agent.host) and calibrate run."""
        if t.closed_inner_pos is None:  # a track built without _new_track: its reference so far is the innermost
            t.closed_inner_pos, t.closed_inner_step = np.array(t.closed_pos, float), t.closed_step
        if t.opening_unknown:
            out_of_inner = float((t.closed_pos - t.closed_inner_pos)[:2] @ t.pull[:2])
            if out_of_inner > 0.0:
                self.reference_events.append({"step": step, "track_id": t.track_id, "rule": "unknown_view_from_inner",
                                              "closed_out_of_inner_m": out_of_inner})
                t.closed_pos, t.closed_step = np.array(t.closed_inner_pos, float), t.closed_inner_step
            sig = self.sigma_along_pull(t)
            if sig <= UNKNOWN_CLEAR_SIGMA_MAX_M:
                # This view measures how far the drawer really is open.
                t.opening_unknown = False
                t.moved_by_robot = self.opening(t.track_id)
            else:
                self.reference_events.append({"step": step, "track_id": t.track_id, "rule": "poor_view_keeps_unknown",
                                              "sigma_along_pull_m": sig, "offset_m": self.opening(t.track_id)})
            return
        if abs(t.moved_by_robot) >= 0.005:
            return
        inner = t.closed_inner_pos
        off = self.opening(t.track_id)
        off_inner = float((t.pos - inner)[:2] @ t.pull[:2])
        if max(off, off_inner) > CLOSED_REBASE_MAX_M:
            # Not a better view of where a closed drawer rests: the drawer is out by what this view measures.
            if off_inner > off:
                t.closed_pos, t.closed_step = np.array(inner, float), t.closed_inner_step
            t.moved_by_robot = self.opening(t.track_id)
            self.reference_events.append({"step": step, "track_id": t.track_id, "rule": "outward_offset_is_opening",
                                          "offset_m": off, "offset_from_inner_m": off_inner,
                                          "opening_m": t.moved_by_robot})
            return
        if float(t.sigma[0]) <= old_sigma + 1e-9:
            t.closed_pos = t.pos.copy()  # still closed: a better view of where it rests
            t.closed_step = step
            if off_inner < 0.0:
                t.closed_inner_pos, t.closed_inner_step = t.pos.copy(), step

    # ------------------------------------------------------------ actions the robot made
    def note_drawer_motion(self, handle_track: str, delta_m: float, action: str, reason: str | None = None):
        """The robot moved a drawer by delta_m along its pull direction (open > 0, close < 0):
        the handle and what was seen inside move with it; what was seen inside becomes stale. contact-invalidates-1:
        reason, the skill's result reason, reaches the site model (detail "reason")."""
        self.step += 1
        h = self.tracks[handle_track]
        rid = self.drawer_region_id(handle_track)
        pull = np.array([h.pull[0], h.pull[1], 0.0])
        for t in self.tracks.values():
            if t.kind == "object" and t.region == rid and t.status != HELD:
                t.pos = t.pos + delta_m * pull
                t.status = STALE
                if delta_m != 0.0:
                    t.pose_source = "dead_reckoned_after_action"
        h.pos = h.pos + delta_m * pull
        h.status = STALE
        if delta_m != 0.0:
            h.pose_source = "dead_reckoned_after_action"
        if action.endswith("_failed"):
            h.opening_unknown = True
        h.moved_by_robot = max(0.0, h.moved_by_robot + delta_m)
        if action == "close_drawer":
            h.moved_by_robot = 0.0
            h.pos = h.closed_pos.copy()
            h.pose_source, h.pose_step = "earlier_view", h.closed_step
        held = self._holding_during()
        if self.model is not None:
            self.model.note_action(self.step, action, rid, {"moved_m": float(delta_m), "holding": held,
                                                            **_reason_detail(reason)})

    def note_pick(self, track_id: str | None, ok: bool, reason: str | None = None):
        """A pick the robot made. Success: the track is held. Failure: the target and every
        object seen in the same region may have been pushed, so they become stale (as in the site model) until seen
        again. Reason (the skill's result reason) reaches the site model."""
        self.step += 1
        region = self.tracks[track_id].region if track_id else None
        if track_id and ok:
            self.tracks[track_id].status = HELD
            self.tracks[track_id].region = "gripper"
        elif track_id:
            for t in self.tracks.values():
                if t.kind == "object" and t.status != HELD and (t.track_id == track_id or t.region == region):
                    t.status = STALE
        held = self._holding_during()
        if self.model is not None:
            self.model.note_action(self.step, "pick", region, {"track_id": track_id, "ok": bool(ok), "holding": held,
                                                               **_reason_detail(reason)})

    def _holding(self) -> bool:
        h = getattr(self.robot, "holding", None)
        try:
            return bool(h()) if callable(h) else False
        except Exception:  # noqa: BLE001 (a robot without the reading: not holding)
            return False

    def _holding_during(self) -> bool:
        """Whether the hand held something at the start or at the end of the action just noted:
        the grip read after the previous noted action (nothing moves the arm between two noted actions) or now. An
        item dropped during the action therefore counts (the bench uses its own finger contacts over the whole call,
        a stricter record)."""
        now = self._holding()
        held = bool(now or self._held_after_last)
        self._held_after_last = now
        return held

    def note_action(self, action: str, detail: dict | None = None, reason: str | None = None):
        """Any other robot action (calibrate, look pose, move, scrub, place...): advances the step
        counter. The site model is told whether the hand held something at the start or the end of the action.
        contact-invalidates-1: reason, the skill's result reason (a look stopped by a collision, a gesture that felt
        a contact), reaches the site model as detail "reason"."""
        self.step += 1
        held = self._holding_during()
        if self.model is not None:
            self.model.note_action(self.step, action, None, {**(detail or {}), **_reason_detail(reason), "holding": held})

    def opening(self, handle_track: str) -> float:
        """Believed opening of a drawer: its handle's offset along the pull axis from where it
        was first seen (assumed closed then)."""
        h = self.tracks[handle_track]
        return float((h.pos - h.closed_pos)[:2] @ h.pull[:2])

    def tracks_of(self, category: str, statuses=(OBSERVED, STALE, DECLARED)):
        return [t for t in self.tracks.values() if t.category == category and t.status in statuses]

    # ------------------------------------------------------------ overhead pre-check (pick-pose-3)
    @staticmethod
    def _pose_frame(fr, step: int, camera: str, obs_id: str, cam_p, cam_R) -> dict:
        """The record kept for the pickable tracks whose pose this view sets: its depth (float32) and believed camera
        pose; {} for a frame without a depth image."""
        depth = getattr(fr, "depth", None)
        if depth is None or np.ndim(depth) != 2:
            return {}
        return {"step": step, "camera": camera, "obs_id": obs_id, "depth": np.asarray(depth, dtype=np.float32),
                "intrinsics": (float(fr.fx), float(fr.fy), float(fr.cx), float(fr.cy)),
                "cam_p": np.array(cam_p, float), "cam_R": np.array(cam_R, float)}

    def overhead_check(self, track, pick_pos, kind: str | None = None, yaw: float | None = None) -> dict:
        """pick-pose-3: would a top grasp at pick_pos hit something
        over the target? perception.overhead.overhead_check on the depth of the view that set the pose of `track` (a
        track id or a Track; kept in pose_frames), with the believed camera pose of that view: rule B active, rule F
        recorded. Returns {"refuse", "code", "n_points", "rule", ...}; no refusal when no pose frame is kept (recorded
        "not_evaluated", as the measured prototype); an error in rule B's evaluation raises (the callers refuse an
        unreadable check). Called by host.pick_precondition and SiteSearch._pick before any motion. Robot side only: the
        kept depth, the believed camera pose and base, the catalogs, the robot's own arm model."""
        t = self.tracks.get(track) if isinstance(track, str) else track
        pos = np.asarray(pick_pos, float).reshape(3)
        kind = kind or (t.category if t is not None else None)
        pf = self.pose_frames.get(t.track_id) if t is not None else None
        if t is None or pf is None or pf["step"] != t.pose_step or kind not in C.OBJECT_SHAPES:
            why = ("no_track" if t is None else "no_pose_frame" if pf is None else
                   "pose_frame_not_current" if pf["step"] != t.pose_step else "no_catalog_shape")
            res = OH.not_evaluated(why)
        else:  # an error here raises: the callers refuse an unreadable check (search.overhead_reading, fail-closed)
            res = self._overhead_on_pose_frame(pf, pos, kind, yaw)
        res["track"] = None if t is None else t.track_id
        res["kind"] = kind
        self.overhead_events.append({k: v for k, v in res.items()})
        return res

    def _overhead_on_pose_frame(self, pf: dict, pos, kind: str, yaw) -> dict:
        """overhead_check's evaluation on a kept pose frame (rule B; rule F with the robot's own finger boxes)."""
        sh = C.OBJECT_SHAPES[kind]
        top_z = float(pos[2]) + float(sh["top_above_centre"])
        r_in = 0.5 * float(min(sh["footprint"])) - OH.OVERHEAD_DISK_SHRINK_M
        top_dz, open_w = OH.skill_top_grasp(kind)
        if yaw is None:
            yaw = OH.default_yaw(pos, self.robot.belief.t[:2])
        tcp, hand_o, hand_R = OH.grasp_frame(pos, yaw, top_dz, TCP_OFFSET, hand_down)
        above_z = OH.approach_height(self.robot.worktop_z, tcp[2])
        try:
            boxes = OH.hand_boxes(open_w)
        except Exception:  # noqa: BLE001
            if OH.RULE_F_ACTIVE:  # An active rule F that cannot be read fails closed
                raise
            boxes = None  # rule F recorded only: without the boxes it is not evaluated
        res = OH.overhead_check(pf["depth"], pf["intrinsics"], pf["cam_p"], pf["cam_R"], pos, top_z, r_in, above_z,
                                self_depth_m=OH.OVERHEAD_SELF_DEPTH_M if pf["camera"] == "wrist_cam" else None,
                                finger_boxes=boxes, hand_origin=hand_o, hand_R=hand_R,
                                sweep_m=float(above_z - tcp[2]), tcp_z=float(tcp[2]))
        res["frame"] = {"step": pf["step"], "camera": pf["camera"], "obs_id": pf["obs_id"]}
        return res


# Checked once the module is complete (perception.upright_faces may import names from here).
configuration_check()
