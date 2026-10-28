"""Simulated depth camera with ideal semantic segmentation. Simulated perception, declared as such.

The robot's own cameras (robot_eye on the mast, wrist_cam on the hand) rendered by MuJoCo. This module is on the
simulator side: it reads simulator truth, but only to synthesize the images. Its output (CameraFrame) is what a depth
camera plus an ideal instance segmenter would give:
- a noisy depth image (meters): per-pixel Gaussian noise sigma = 1 mm + 1.5 mm * z^2, 2 % of pixels dropped;
- one segment per recognized visible thing: its catalog category and its pixels with their noisy depths. Recognition
  is ideal (not a trained vision model). A thing out of view or behind something gives no pixel, so no segment;
- systematic errors fixed per episode from noise_seed and unknown to the robot side: each camera is rendered 3 mm and
  0.3 deg off its design pose, and depth is scaled by 1 +- 0.5 %.
The visible fraction (an oracle quantity from a second, truth-side render) is used only inside this sensor to decide
which segments it reports (objects need >= 40 visible pixels and a fraction >= 0.15; rail handles >= 20 and >= 0.25;
modules >= 150 pixels); it is not in the output. Images are 640 x 480.

Not modeled: misclassification, mixed pixels at depth edges, reflections, transparency. It gives no names, simulator
ids, camera pose, identity across frames or object poses (the robot side computes those).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from .catalog import MODULE_KIND_TO_CATEGORY

DEPTH_SIGMA_A, DEPTH_SIGMA_B = 0.001, 0.0015
PIXEL_SIGMA = 0.3
DROPOUT = 0.02
MIN_PX = {"object": 40, "handle": 20, "module": 150}
MIN_FRAC = {"object": 0.15, "handle": 0.25, "module": 0.0}
CAMERAS = {"robot_eye": "robot_eye", "wrist_cam": "panda/wrist_cam"}


def depth_sigma(z):
    return DEPTH_SIGMA_A + DEPTH_SIGMA_B * np.asarray(z) ** 2


@dataclass
class Segment:
    category: str
    kind: str  # object | handle | module
    uv: np.ndarray  # (N, 2) float pixel coordinates (u right, v down), jittered
    depth: np.ndarray  # (N,) noisy z-depth (m), invalid pixels removed
    n_pixels: int  # pixels of the segment in this image (before depth dropout)
    at_image_border: bool = False  # some of its pixels touch the image edge: it may be cut by the frame


@dataclass
class CameraFrame:
    camera: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    t_s: float
    depth: np.ndarray  # (H, W) noisy z-depth, 0 = invalid
    segments: list[Segment] = field(default_factory=list)


class SimulatedCamera:
    """Simulator-side sensor. capture(camera) -> CameraFrame. See the module docstring."""

    SYS_POS_M, SYS_ROT_DEG, SYS_DEPTH_SCALE = 0.003, 0.3, 0.005

    def __init__(self, sim, width: int = 640, height: int = 480, noise_seed: int = 0, systematic: bool = True,
                 noise_scale: float = 1.0):
        self.sim = sim
        self.width, self.height = width, height
        self.rng = np.random.default_rng(noise_seed)
        self.noise_scale = float(noise_scale)
        self.depth_sigma_scale = self.noise_scale
        self.pixel_sigma = PIXEL_SIGMA * self.noise_scale
        self.dropout = min(0.2, DROPOUT * (2 * self.noise_scale - 1)) if self.noise_scale > 1 else DROPOUT
        self._sys = {}
        srng = np.random.default_rng([int(noise_seed), 4242])
        for cam in CAMERAS:  # drawn in a fixed order: the same seed gives the same errors
            d = srng.normal(size=3)
            ax = srng.normal(size=3)
            sign = 1.0 if srng.random() < 0.5 else -1.0
            if systematic:
                k = self.noise_scale
                dp = d / np.linalg.norm(d) * self.SYS_POS_M * k
                ang = math.radians(self.SYS_ROT_DEG) * k
                ax = ax / np.linalg.norm(ax)
                dq = np.array([math.cos(ang / 2), *(math.sin(ang / 2) * ax)])
                self._sys[cam] = (dp, dq, 1.0 + sign * self.SYS_DEPTH_SCALE * k)
            else:
                self._sys[cam] = (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), 1.0)
        self._r = mujoco.Renderer(sim.model, height, width)
        self._build_entities()

    def systematic_errors(self) -> dict:
        """Bench side only (reports): the per-camera systematic errors of this episode."""
        return {c: {"pos_offset_m": [round(float(x), 5) for x in dp],
                    "rot_offset_deg": round(math.degrees(2 * math.acos(min(1.0, abs(float(dq[0]))))), 4),
                    "depth_scale": round(float(sc), 5)} for c, (dp, dq, sc) in self._sys.items()}

    def _offset_camera(self, camera: str):
        """Render-time pose of the camera: design pose moved by this episode's systematic offset
        (in the camera frame). Returns the saved model values to restore."""
        m, d = self.sim.model, self.sim.data
        cid = m.camera(CAMERAS[camera]).id
        saved = (m.cam_pos[cid].copy(), m.cam_quat[cid].copy())
        dp, dq, _ = self._sys[camera]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, saved[1])
        m.cam_pos[cid] = saved[0] + R.reshape(3, 3) @ dp
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, saved[1], dq)
        m.cam_quat[cid] = q
        mujoco.mj_camlight(m, d)
        return cid, saved

    def _restore_camera(self, cid, saved):
        m = self.sim.model
        m.cam_pos[cid], m.cam_quat[cid] = saved
        mujoco.mj_camlight(m, self.sim.data)

    def close(self):
        if self._r is not None:
            self._r.close()
            self._r = None

    # ------------------------------------------------------------ truth side (synthesis only)
    def _build_entities(self):
        m = self.sim.model
        items = {it.name: it.kind for it in self.sim.site.items}

        def root(b):
            while m.body_parentid[b] != 0:
                b = m.body_parentid[b]
            return b
        keys: dict = {}
        self._ent_kind, self._ent_cat = [], []
        ent_of_geom = np.full(m.ngeom, -1, dtype=np.int64)
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            rname = m.body(root(b)).name
            gname = m.geom(g).name
            if rname in items:
                key, kind, cat = ("o", rname), "object", items[rname]
            elif gname.endswith("_handle") and (gname.startswith("drawer_") or gname == "dw_handle"):
                key, kind = ("h", gname), "handle"
                cat = "drawer_handle" if gname.startswith("drawer_") else "dishwasher_handle"
            elif rname in MODULE_KIND_TO_CATEGORY:
                key, kind, cat = ("m", rname), "module", MODULE_KIND_TO_CATEGORY[rname]
            else:
                continue
            if key not in keys:
                keys[key] = len(self._ent_kind)
                self._ent_kind.append(kind)
                self._ent_cat.append(cat)
            ent_of_geom[g] = keys[key]
        self._ent_of_geom = ent_of_geom
        self._ent_geoms = [np.flatnonzero(ent_of_geom == e) for e in range(len(self._ent_kind))]

    def _seg(self, cam):
        r = self._r
        r.enable_segmentation_rendering()
        r.update_scene(self.sim.data, camera=cam)
        seg = r.render().copy()
        r.disable_segmentation_rendering()
        ids = seg[:, :, 0]
        ok = (seg[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM) & (ids >= 0)
        ent = np.full(ids.shape, -1, dtype=np.int64)
        ent[ok] = self._ent_of_geom[ids[ok]]
        return ent

    def _oracle_visible_fraction(self, cam, e, n: int) -> float:
        """Truth side (sensor-internal gating only, never output): visible pixels n over the pixels
        the entity would cover with nothing in front."""
        return n / max(self._unoccluded_px(cam, e), n)

    def _unoccluded_px(self, cam, e) -> int:
        m = self.sim.model
        saved = m.geom_group.copy()
        try:
            m.geom_group[:] = 5  # hidden group (default MjvOption shows groups 0-2)
            gs = self._ent_geoms[e]
            m.geom_group[gs] = saved[gs]
            ent = self._seg(cam)
        finally:
            m.geom_group[:] = saved
        return int((ent == e).sum())

    # ------------------------------------------------------------ the sensor output
    def capture(self, camera: str) -> CameraFrame:
        if camera not in CAMERAS:
            raise ValueError(f"camera must be one of {sorted(CAMERAS)}")
        cam = CAMERAS[camera]
        m = self.sim.model
        W, H = self.width, self.height
        fovy = float(m.cam_fovy[m.camera(cam).id])
        fy = (H / 2) / math.tan(math.radians(fovy) / 2)
        fx = fy
        r = self._r
        cid, saved = self._offset_camera(camera)
        try:
            r.enable_depth_rendering()
            r.update_scene(self.sim.data, camera=cam)
            depth = r.render().copy().astype(np.float64)
            r.disable_depth_rendering()
            ent = self._seg(cam)
            valid = (depth > 0.02) & (depth < 6.0)
            noisy = depth * self._sys[camera][2]
            noisy = noisy + self.rng.normal(0.0, 1.0, depth.shape) * depth_sigma(depth) * self.depth_sigma_scale
            noisy[self.rng.random(depth.shape) < self.dropout] = 0.0
            noisy[~valid] = 0.0
            segs = []
            for e in np.unique(ent[ent >= 0]):
                kind = self._ent_kind[e]
                mask = ent == e
                n = int(mask.sum())
                if n < MIN_PX[kind]:
                    continue
                if self._oracle_visible_fraction(cam, e, n) < MIN_FRAC[kind]:  # gating only
                    continue
                vb, ub = np.nonzero(mask)
                border = bool(vb.min() <= 1 or ub.min() <= 1 or vb.max() >= H - 2 or ub.max() >= W - 2)
                v, u = np.nonzero(mask & (noisy > 0))
                d = noisy[v, u]
                uv = np.column_stack([u + 0.5, v + 0.5]).astype(float)
                uv += self.rng.normal(0.0, self.pixel_sigma, uv.shape)
                segs.append(Segment(self._ent_cat[e], kind, uv, d, n, border))
        finally:
            self._restore_camera(cid, saved)
        # Order carries no truth: by category, then left to right in the image.
        segs.sort(key=lambda s: (s.category, float(np.mean(s.uv[:, 0])) if len(s.uv) else 0.0))
        return CameraFrame(camera, W, H, fx, fy, W / 2, H / 2, float(self.sim.t), noisy, segs)
