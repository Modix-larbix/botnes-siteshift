"""Face-aware upright placement test (configuration K1): can a view show a sponge standing on its long edge?

A sponge on its long edge at the back of a drawer, under the unit top, shows little of its top face, so the top-only
placement test never clears that place. A tilted view shows its side face instead. This module counts the side faces
that face the camera, so a tilted view can show the place; a straight-down view sees those faces at a grazing angle,
which the angle cut excludes.

The rule, per region view (upright_evidence), for each placement centre and each fitting yaw, at both test levels:
- the top face is sampled as perception.estimate's flat test does (with the low-level "beyond" rule);
- each side face whose outward normal faces the camera is sampled along the face and in height over the band that
  exists for every floor within its error; a sample counts only under FACE_MAX_ANGLE_DEG from the camera ray and is
  shown only when the measured depth lies UPRIGHT_SIDE_BEYOND_SIGMAS depth sigmas beyond it;
- the placement is shown ('v') only when, at every fitting yaw and both levels, the shown pixels reach the margin
  (PLACEMENT_MARGIN_UPRIGHT) times the recognizer's own thresholds.
Robot side only: the frame's depth, the believed camera pose, the region lattice, the floor and its declared error,
the resting objects and the catalog; never the simulator. The thresholds are declared calibration constants of the
simulated sensor. This module imports nothing from siteagent: estimate passes itself in as E.
"""
from __future__ import annotations

import math

import numpy as np

UPRIGHT_FACES_VERSION = "upright-faces-1-v2side"  # the face-aware test with the side-face 'beyond' rule (records only)
FACE_MAX_ANGLE_DEG = 70.0  # a side-face sample counts only if its normal and the ray to the camera are under 70 degrees
FACE_DZ_M = 0.005  # side faces are sampled in height on cells of at most 5 mm
SIDE_BATCH = 400  # placements per vectorised batch of side-face samples (memory only: no effect on any number)
# The k of the low-level "beyond" rule for the upright TOP-face samples only (upright_evidence and upright_at_pose).
# Estimate's flat test keeps E.PLACEMENT_BEYOND_SIGMAS (2.5); the high level is unchanged; the side faces use the
# side-face rule below. Declared development tuning, chosen on synthetic corners and stored frames in simulation: it
# closes the single-view leaks of a bare on-edge sponge at the declared corner (measured floor about 6.6 mm over the
# truth). A merged-view residual remains: with the side-face rule it read 0 of 4,000 synthetic merges at that corner
# (in-memory emulation), and elsewhere it is smaller but not zero (16 of 725 canonical six-view merges, 52 of 1,781
# random four-view merges). It stays a declared limit.
UPRIGHT_LOW_BEYOND_SIGMAS = 2.75
# Side-face 'beyond' rule (declared development tuning, chosen after the measurements): a SIDE-face sample counts as
# shown only when the measured depth lies at least UPRIGHT_SIDE_BEYOND_SIGMAS depth sigmas BEYOND it (side_pixels),
# instead of reaching it within E.PLACEMENT_DEPTH_TOL_M + 2.5 depth sigma. The side pixels feed BOTH test levels (top +
# sides), so the high level's sum tightens too. Cost: no claim changed; upright 'v' samples were lost on the same
# views, 97 % of them in sink_basin. It closes the side-face merge path; a yaw / lattice path remains (declared). Per
# sample, the shown set is a subset of the plain reach test's (TOL > 0, sigma > 0).
UPRIGHT_SIDE_BEYOND_SIGMAS = 2.5


def top_levels(E, f_used: float, f_err: float, H: float) -> dict:
    """The two heights of the hypothetical top face (estimate's levels): "low" the strictest under a cover, "high" the
    strictest under an overhang."""
    return {"low": f_used - f_err + H - E.PLACEMENT_SETTLE_M, "high": f_used + f_err + H}


def side_band(E, f_used: float, f_err: float, H: float) -> tuple[float, float]:
    """The height band of the side faces that exists for every floor within its error: [floor + error, floor - error
    + height - settle]."""
    return f_used + f_err, f_used - f_err + H - E.PLACEMENT_SETTLE_M


def side_pixels(E, fr, cam_p, cam_R, centres, ul, uw, L: float, W: float, zb0: float, zb1: float,
                step: float) -> np.ndarray:
    """Shown pixels of the camera-facing side faces of upright boxes: centres (N, 2) world xy, long-axis unit vector ul
    (2,) and short-axis unit vector uw (2,) (the same for every box), footprint L x W, height band [zb0, zb1].
    Samples every `step` along each face and on at most FACE_DZ_M in height; a face counts when its outward
    horizontal normal points toward the camera; a sample counts when its angle to the ray is under FACE_MAX_ANGLE_DEG
    and the measured depth lies at least UPRIGHT_SIDE_BEYOND_SIGMAS depth sigmas beyond it (the side-face 'beyond'
    rule, stricter than reaching it within E.PLACEMENT_DEPTH_TOL_M + 2.5 depth sigma). Returns (N,) pixels."""
    centres = np.asarray(centres, float)
    N = len(centres)
    out = np.zeros(N)
    if N == 0 or zb1 <= zb0 + 1e-6:
        return out
    m = max(1, int(math.ceil((zb1 - zb0) / FACE_DZ_M - 1e-9)))
    dz = (zb1 - zb0) / m
    zs = zb0 + (np.arange(m) + 0.5) * dz
    cos_min = math.cos(math.radians(FACE_MAX_ANGLE_DEG))
    ul, uw = np.asarray(ul, float), np.asarray(uw, float)
    # (outward normal, offset of the face centre from the box centre, direction along the face, face length)
    faces = ((ul, 0.5 * L * ul, uw, W), (-ul, -0.5 * L * ul, uw, W), (uw, 0.5 * W * uw, ul, L),
             (-uw, -0.5 * W * uw, ul, L))
    cxy = np.asarray(cam_p[:2], float)
    for n2, off, t2, length in faces:
        nt = max(1, int(math.ceil(length / step - 1e-9)))
        dt = length / nt
        ts = -0.5 * length + (np.arange(nt) + 0.5) * dt
        dA = dt * dz
        fc = centres + off
        facing = (cxy - fc) @ n2 > 0
        idx_all = np.flatnonzero(facing)
        for s0 in range(0, len(idx_all), SIDE_BATCH):
            idx = idx_all[s0:s0 + SIDE_BATCH]
            k = len(idx)
            xy = fc[idx][:, None, :] + ts[None, :, None] * t2[None, None, :]  # (k, nt, 2)
            P = np.empty((k, nt, m, 3))
            P[..., 0] = xy[:, :, None, 0]
            P[..., 1] = xy[:, :, None, 1]
            P[..., 2] = zs[None, None, :]
            P = P.reshape(-1, 3)
            u, v, z, ok = E.Perception.project(fr, cam_p, cam_R, P)
            w = np.zeros(len(P))
            ii = np.flatnonzero(ok)
            if len(ii):
                meas = fr.depth[v[ii].astype(int), u[ii].astype(int)]
                need = z[ii] + UPRIGHT_SIDE_BEYOND_SIGMAS * E.depth_sigma(z[ii])  # side faces 'beyond'
                vec = cam_p - P[ii]
                ndot = vec[:, 0] * n2[0] + vec[:, 1] * n2[1]
                cosang = ndot / np.maximum(np.linalg.norm(vec, axis=1), 1e-9)
                shown = (meas > 0) & (meas >= need) & (cosang > cos_min)
                w[ii] = np.where(shown, fr.fx * fr.fy * np.maximum(ndot, 0.0) * dA / z[ii] ** 3, 0.0)
            out[idx] += w.reshape(k, nt * m).sum(axis=1)
    return out


def upright_evidence(E, fr, cam_p, cam_R, pts, grid, f_used: float, f_err: float, objects, margin: float) -> dict:
    """The upright mode record of one region view (module docstring), on estimate's lattice: E the estimate module,
    fr the frame (depth, fx, fy, cx, cy, width, height), cam_p / cam_R the believed camera pose, pts / grid the region's
    floor grid (as Perception._placement_evidence receives them), f_used / f_err the floor used in this view and its
    declared error, objects the region's perceived objects (Perception._resting_objects), margin the factor on both
    recogniser thresholds. Returns {active, face_aware, margin, face_max_angle_deg, low_rule, k0, n, step, r_in,
    share_pct, px, band_m, n_side_evals, code}. Errors raise (the caller records an evidence error: the region is not
    cleared)."""
    target = E.OCCLUDER_TARGET_CATEGORY
    (na, nb), (pa, pb) = grid
    o = pts[0, :2].astype(float)
    ea = (pts[nb, :2] - pts[0, :2]) / pa
    eb = (pts[1, :2] - pts[0, :2]) / pb
    ext_a, ext_b = (na - 1) * pa, (nb - 1) * pb
    fa, fb = pa / 4.0, pb / 4.0  # estimate's fine lattice of top-face samples; placement centres every 2 fine steps
    ia0, ib0 = -int(math.ceil(E.GRID_INSET / fa - 1e-9)), -int(math.ceil(E.GRID_INSET / fb - 1e-9))
    ni = int(math.floor((ext_a + E.GRID_INSET) / fa + 1e-9)) - ia0 + 1
    nj = int(math.floor((ext_b + E.GRID_INSET) / fb + 1e-9)) - ib0 + 1
    FA, FB = np.meshgrid((np.arange(ni) + ia0) * fa, (np.arange(nj) + ib0) * fb, indexing="ij")
    fxy = o + np.outer(FA.ravel(), ea) + np.outer(FB.ravel(), eb)
    L, W, H = E.placement_modes(target)["upright"]
    ka = np.arange(int(math.ceil((-E.GRID_INSET + 0.5 * W) / (pa / 2) - 1e-9)),
                   int(math.floor((ext_a + E.GRID_INSET - 0.5 * W) / (pa / 2) + 1e-9)) + 1)
    kb = np.arange(int(math.ceil((-E.GRID_INSET + 0.5 * W) / (pb / 2) - 1e-9)),
                   int(math.floor((ext_b + E.GRID_INSET - 0.5 * W) / (pb / 2) + 1e-9)) + 1)
    base = {"active": True, "face_aware": UPRIGHT_FACES_VERSION, "margin": margin,
            "face_max_angle_deg": FACE_MAX_ANGLE_DEG, "low_rule": E.PLACEMENT_LOW_RULE_VERSION}
    if len(ka) == 0 or len(kb) == 0:
        return dict(base, k0=[0, 0], n=[0, 0], step=[pa / 2, pb / 2], code="")
    KA, KB = np.meshgrid(ka, kb, indexing="ij")
    ac, bc = KA.ravel() * (pa / 2), KB.ravel() * (pb / 2)
    cxy = o + np.outer(ac, ea) + np.outer(bc, eb)
    ci, cj = 2 * KA.ravel() - ia0, 2 * KB.ravel() - ib0
    n_c = len(ac)
    yaws = E.PLACEMENT_YAWS_DEG
    ny = len(yaws)
    fits = np.zeros((n_c, ny), bool)
    for iy, ydeg in enumerate(yaws):
        yaw = math.radians(ydeg)
        c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
        xa, xb = 0.5 * L * c + 0.5 * W * s, 0.5 * L * s + 0.5 * W * c
        fits[:, iy] = ((ac - xa >= -E.GRID_INSET - 1e-9) & (ac + xa <= ext_a + E.GRID_INSET + 1e-9)
                       & (bc - xb >= -E.GRID_INSET - 1e-9) & (bc + xb <= ext_b + E.GRID_INSET + 1e-9))
    any_fit = fits.any(axis=1)
    lv = top_levels(E, f_used, f_err, H)
    zb0, zb1 = side_band(E, f_used, f_err, H)
    # Top face, per level and fitting yaw: shown pixels and the whole box silhouette (inf where no fit).
    top_px = {x: np.zeros((n_c, ny)) for x in E.PLACEMENT_TEST_LEVELS}
    sil = {x: np.full((n_c, ny), np.inf) for x in E.PLACEMENT_TEST_LEVELS}
    for lname in E.PLACEMENT_TEST_LEVELS:
        z_top = lv[lname]
        P = np.column_stack([fxy, np.full(len(fxy), z_top)])
        u, v, z, ok = E.Perception.project(fr, cam_p, cam_R, P)
        idx = np.flatnonzero(ok)
        vw = np.zeros(len(P))
        if len(idx):
            meas = fr.depth[v[idx].astype(int), u[idx].astype(int)]
            rz = cam_p[2] - P[idx, 2]
            tol = E.PLACEMENT_DEPTH_TOL_M + 2.5 * E.depth_sigma(z[idx])
            need = (z[idx] + UPRIGHT_LOW_BEYOND_SIGMAS * E.depth_sigma(z[idx]) if lname == "low"
                    else z[idx] - tol)  # low_beyond_1
            shown = (meas > 0) & (meas >= need) & (rz > 0)
            vw[idx] = np.where(shown, fa * fb * fr.fx * fr.fy * np.maximum(rz, 0.0) / z[idx] ** 3, 0.0)
        VW = vw.reshape(ni, nj)
        for iy, ydeg in enumerate(yaws):
            sel = np.flatnonzero(fits[:, iy])
            if len(sel) == 0:
                continue
            yaw = math.radians(ydeg)
            di, dj = E._rect_taps(L, W, yaw, fa, fb)
            II, JJ = ci[sel, None] + di[None, :], cj[sel, None] + dj[None, :]
            inr = (II >= 0) & (II < ni) & (JJ >= 0) & (JJ < nj)
            top_px[lname][sel, iy] = np.where(inr, VW[np.clip(II, 0, ni - 1), np.clip(JJ, 0, nj - 1)], 0.0).sum(axis=1)
            ul = np.tile(math.cos(yaw) * ea + math.sin(yaw) * eb, (len(sel), 1))
            uw = np.tile(-math.sin(yaw) * ea + math.cos(yaw) * eb, (len(sel), 1))
            sil[lname][sel, iy] = E.box_silhouette_px(fr, cam_p, cam_R, cxy[sel], ul, uw, L, W, z_top, H)
    thr_px, thr_sh = margin * E.PLACEMENT_MIN_PX, margin * E.PLACEMENT_MIN_FRAC

    def passes(px, sl):
        return (px >= thr_px) & np.isfinite(sl) & (sl > 0) & (px >= thr_sh * np.where(np.isfinite(sl), sl, np.inf))

    # Side faces where they can change the code (module docstring, "Records"); 'v' needs every fitting yaw.
    alive = any_fit.copy()
    side = np.zeros((n_c, ny))
    n_side = 0
    step = min(fa, fb)
    for iy, ydeg in enumerate(yaws):
        sel = fits[:, iy] & alive
        if not sel.any():
            continue
        top_ok = np.ones(n_c, bool)
        for lname in E.PLACEMENT_TEST_LEVELS:
            top_ok &= passes(top_px[lname][:, iy], sil[lname][:, iy])
        need = np.flatnonzero(sel & ~top_ok)
        if len(need):
            yaw = math.radians(ydeg)
            ul = math.cos(yaw) * ea + math.sin(yaw) * eb
            uw = -math.sin(yaw) * ea + math.cos(yaw) * eb
            side[need, iy] = side_pixels(E, fr, cam_p, cam_R, cxy[need], ul, uw, L, W, zb0, zb1, step)
            n_side += len(need)
        ok_all = np.ones(n_c, bool)
        for lname in E.PLACEMENT_TEST_LEVELS:
            ok_all &= passes(top_px[lname][:, iy] + side[:, iy], sil[lname][:, iy])
        alive &= ~(sel & ~ok_all)
    visible = any_fit & alive
    min_share, min_px = {}, {}
    for lname in E.PLACEMENT_TEST_LEVELS:
        tot = top_px[lname] + side
        sh = np.where(np.isfinite(sil[lname]) & (sil[lname] > 0), tot / np.where(sil[lname] > 0, sil[lname], 1.0), 0.0)
        min_share[lname] = np.where(fits, sh, np.inf).min(axis=1)
        min_px[lname] = np.where(fits, tot, np.inf).min(axis=1)
    excl = E.placement_excluded(cxy, objects, "upright", 0.5 * W) & any_fit
    uc, vc, zc, okc = E.Perception.project(fr, cam_p, cam_R, np.column_stack([cxy, np.full(n_c, f_used)]))
    code = np.where(~any_fit, " ", np.where(excl, "x", np.where(visible, "v", np.where(okc, "-", "."))))
    return dict(base, k0=[int(ka[0]), int(kb[0])], n=[int(len(ka)), int(len(kb))], step=[pa / 2, pb / 2], r_in=0.5 * W,
                share_pct={x: [int(math.floor(100 * min(v_, 9.99))) if f else -1 for v_, f in zip(min_share[x], any_fit)]
                           for x in E.PLACEMENT_TEST_LEVELS},
                px={x: [int(min(v_, 99999)) if f else -1 for v_, f in zip(min_px[x], any_fit)]
                    for x in E.PLACEMENT_TEST_LEVELS},
                band_m=[round(float(zb0), 5), round(float(zb1), 5)], n_side_evals=int(n_side), code="".join(code))


def upright_at_pose(E, fr, cam_p, cam_R, cxy, ul, f_used: float, f_err: float, step_a: float, step_b: float) -> dict:
    """The face-aware numbers of ONE upright box (for scoring and calibration; nothing robot-side calls it): centre cxy
    (2,) world xy, long axis ul (2,), with the view's floor (f_used, f_err) and fine steps (step_a, step_b). Per level
    {top_px, side_px, px, sil, share}, plus band_m. The top face is sampled on an L x W cell grid of about the fine steps,
    centred on the pose; the side faces as in upright_evidence (every facing face, no pruning)."""
    L, W, H = E.placement_modes(E.OCCLUDER_TARGET_CATEGORY)["upright"]
    ul = np.asarray(ul, float)[:2]
    ul = ul / max(float(np.linalg.norm(ul)), 1e-12)
    uw = np.array([-ul[1], ul[0]])
    cxy = np.asarray(cxy, float)[:2]
    lv = top_levels(E, f_used, f_err, H)
    zb0, zb1 = side_band(E, f_used, f_err, H)
    na_, nb_ = max(1, int(round(L / step_a))), max(1, int(round(W / step_b)))
    ta = -0.5 * L + (np.arange(na_) + 0.5) * (L / na_)
    tb = -0.5 * W + (np.arange(nb_) + 0.5) * (W / nb_)
    TA, TB = np.meshgrid(ta, tb, indexing="ij")
    xy = cxy + np.outer(TA.ravel(), ul) + np.outer(TB.ravel(), uw)
    dA = (L / na_) * (W / nb_)
    side = float(side_pixels(E, fr, cam_p, cam_R, cxy[None, :], ul, uw, L, W, zb0, zb1, min(step_a, step_b))[0])
    out = {"band_m": [round(float(zb0), 5), round(float(zb1), 5)]}
    for lname in E.PLACEMENT_TEST_LEVELS:
        z_top = lv[lname]
        P = np.column_stack([xy, np.full(len(xy), z_top)])
        u, v, z, ok = E.Perception.project(fr, cam_p, cam_R, P)
        top = 0.0
        idx = np.flatnonzero(ok)
        if len(idx):
            meas = fr.depth[v[idx].astype(int), u[idx].astype(int)]
            rz = cam_p[2] - P[idx, 2]
            tol = E.PLACEMENT_DEPTH_TOL_M + 2.5 * E.depth_sigma(z[idx])
            need = (z[idx] + UPRIGHT_LOW_BEYOND_SIGMAS * E.depth_sigma(z[idx]) if lname == "low"
                    else z[idx] - tol)  # low_beyond_1
            shown = (meas > 0) & (meas >= need) & (rz > 0)
            top = float(np.where(shown, dA * fr.fx * fr.fy * np.maximum(rz, 0.0) / z[idx] ** 3, 0.0).sum())
        sl = float(E.box_silhouette_px(fr, cam_p, cam_R, cxy[None, :], ul[None, :], uw[None, :], L, W, z_top, H)[0])
        px = top + side
        out[lname] = {"top_px": round(top, 2), "side_px": round(side, 2), "px": round(px, 2),
                      "sil": None if not math.isfinite(sl) else round(sl, 2),
                      "share": 0.0 if not (math.isfinite(sl) and sl > 0) else round(px / sl, 5)}
    return out
