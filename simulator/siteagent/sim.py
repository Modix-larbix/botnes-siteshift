"""Simulation wrapper for one site: build, reset, step at the control rate, read sensors, record.

The robot side (siteagent.robot) only uses arm joint positions, its own commands, gripper width, the IMU and contact
forces on robot links expressed in the robot's own base frame (like joint-torque based force estimation plus a wrist
force sensor). Everything else here (bodies, item poses) is for the bench and the checks.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from .robot.kinematics import ARM_JOINTS, HOME
from .world.build import build
from .world.spec import SiteSpec

CTRL_DT = 0.02  # 50 Hz control
GRIP_OPEN, GRIP_CLOSED = 255.0, 0.0
IMU_NOISE = 0.05  # m/s^2, white noise per accelerometer sample


class Sim:
    def __init__(self, site: SiteSpec, record=None, record_every: int = 4,
                 size: tuple[int, int] = (320, 240), noise_seed: int = 0, world_kw: dict | None = None):
        """world_kw: build options (world.build), e.g. {"sponge_dish_friction": None} for the
        earlier world (comparison runs only)."""
        self.site = site
        self.model, self.info = build(site, **(world_kw or {}))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.nsub = int(round(CTRL_DT / m.opt.timestep))
        self.qadr = np.array([m.joint(f"panda/{j}").qposadr[0] for j in ARM_JOINTS])
        self.dadr = np.array([m.joint(f"panda/{j}").dofadr[0] for j in ARM_JOINTS])
        self.fadr = np.array([m.joint(f"panda/finger_joint{i}").qposadr[0] for i in (1, 2)])
        self.robot_bodies = {b for b in range(m.nbody) if m.body(b).name.startswith("panda/")}
        self.robot_geom = np.array([m.geom_bodyid[g] in self.robot_bodies for g in range(m.ngeom)])
        self.finger_bodies = {m.body("panda/left_finger").id, m.body("panda/right_finger").id}
        self.hand_bodies = self.finger_bodies | {m.body("panda/hand").id}
        self.link_bodies = {m.body(f"panda/link{i}").id for i in range(8)}
        self.base_body = m.body("panda/link0").id
        self.imu_adr = m.sensor("imu_acc").adr[0]
        self.imu_site = m.site("imu").id
        self.imu_body = int(m.site_bodyid[self.imu_site])
        self.rng = np.random.default_rng(noise_seed)
        self.record_cam = record
        self.record_every = record_every
        self.frames: list[np.ndarray] = []
        self._renderer = None
        if record:
            self._renderer = mujoco.Renderer(m, height=size[1], width=size[0])
        self.ticks = 0
        self._arm_target = None
        self.flags: dict = {}  # robot-side switches (e.g. the spray trigger)
        self.monitors: list = []  # bench-side callbacks run after every tick (ground truth logging)
        self.max_env_force = 0.0  # largest robot-environment contact force seen (diagnostic)
        self.reset()

    # ------------------------------------------------------------ time
    @property
    def t(self) -> float:
        return float(self.data.time)

    def reset(self, q=HOME, settle_s: float = 0.5):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[self.qadr] = q
        d.ctrl[:7] = q
        d.qpos[self.fadr] = 0.04
        d.ctrl[7] = GRIP_OPEN
        self._arm_target = None
        mujoco.mj_forward(m, d)
        self.frames.clear()
        self.ticks = 0
        self.tick(int(settle_s / CTRL_DT))

    def tick(self, n: int = 1):
        """Advance n control periods (each = nsub physics steps). The arm target is ramped
        linearly over the substeps (a 500 Hz servo between 50 Hz commands, as a real low-level
        controller interpolates); a step change every 20 ms would shake the hand."""
        m, d = self.model, self.data
        for _ in range(n):
            if self._arm_target is not None:
                q0 = d.ctrl[:7].copy()
                q1 = self._arm_target
                for j in range(1, self.nsub + 1):
                    d.ctrl[:7] = q0 + (q1 - q0) * (j / self.nsub)
                    mujoco.mj_step(m, d)
                self._arm_target = None
            else:
                mujoco.mj_step(m, d, nstep=self.nsub)
            if self._renderer is not None and self.ticks % self.record_every == 0:
                self._renderer.update_scene(d, camera=self.record_cam)
                self.frames.append(self._renderer.render().copy())
            self.ticks += 1
            for mon in self.monitors:
                mon(self)

    # ------------------------------------------------------------ actuation
    def set_arm(self, q):
        self._arm_target = np.asarray(q, float).copy()

    def set_gripper(self, u: float):
        self.data.ctrl[7] = u

    def gripper_cmd(self) -> float:
        """The gripper command last sent (the robot's own command readback)."""
        return float(self.data.ctrl[7])

    # ------------------------------------------------------------ proprioception and sensors
    def arm_q(self) -> np.ndarray:
        return self.data.qpos[self.qadr].copy()

    def arm_qd(self) -> np.ndarray:
        return self.data.qvel[self.dadr].copy()

    def gripper_width(self) -> float:
        return float(self.data.qpos[self.fadr].sum())

    def imu(self) -> np.ndarray:
        """Accelerometer at the mount (site frame, specific force: reads +g up when static), with noise.

        MuJoCo 3.14 returns zero for acceleration sensors (and mj_objectAcceleration) on a body
        welded to the world. The mount is welded, so the reading is computed from the body's
        spatial acceleration (cacc, which carries gravity as a world acceleration) rotated into
        the site frame: the same quantity a real accelerometer measures on a static mount.
"""
        m, d = self.model, self.data
        mujoco.mj_rnePostConstraint(m, d)
        lin = d.cacc[self.imu_body][3:]
        R = d.site_xmat[self.imu_site].reshape(3, 3)
        return R.T @ lin + self.rng.normal(0, IMU_NOISE, 3)

    def robot_contacts(self):
        """Contacts between robot geoms and anything else: (robot body id, other geom id, force vector in world)."""
        m, d = self.model, self.data
        out = []
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            r1, r2 = self.robot_geom[g1], self.robot_geom[g2]
            if r1 == r2:
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            frame = c.frame.reshape(3, 3)
            fw = frame.T @ f6[:3]  # force on geom2 from geom1, world frame
            if r1:  # force on the robot geom (g1) is -fw
                out.append((int(m.geom_bodyid[g1]), int(g2), -fw))
            else:
                out.append((int(m.geom_bodyid[g2]), int(g1), fw))
        return out

    def env_force(self, bodies=None) -> np.ndarray:
        """Sum of contact forces on robot bodies (optionally a subset) from the environment, world
        frame (bench diagnostics only; the robot reads env_force_base)."""
        tot = np.zeros(3)
        for b, _g, f in self.robot_contacts():
            if bodies is None or b in bodies:
                tot += f
        n = float(np.linalg.norm(tot))
        self.max_env_force = max(self.max_env_force, n)
        return tot

    def _base_R(self) -> np.ndarray:
        return self.data.xmat[self.base_body].reshape(3, 3)

    def env_force_base(self, bodies=None) -> np.ndarray:
        """Like env_force, expressed in the robot's own base frame (what its sensors measure)."""
        return self._base_R().T @ self.env_force(bodies)

    def link_forces_base(self) -> dict:
        """Net environment contact force on each robot body (body id -> 3-vector, base frame).
        Stands in for the external force estimate a Franka computes from its joint torques."""
        out: dict = {}
        for b, _g, f in self.robot_contacts():
            out[b] = out.get(b, np.zeros(3)) + f
        Rt = self._base_R().T
        return {b: Rt @ f for b, f in out.items()}

    def finger_touch(self) -> tuple[bool, bool]:
        """Whether each finger touches something that is not the robot."""
        lf, rf = self.model.body("panda/left_finger").id, self.model.body("panda/right_finger").id
        hit = {b for b, _g, _f in self.robot_contacts()}
        return lf in hit, rf in hit

    # ------------------------------------------------------------ ground truth (bench and checks only)
    def body_pos(self, name: str) -> np.ndarray:
        return self.data.body(name).xpos.copy()

    def body_mat(self, name: str) -> np.ndarray:
        return self.data.body(name).xmat.reshape(3, 3).copy()

    def joint_pos(self, name: str) -> float:
        return float(self.data.qpos[self.model.joint(name).qposadr[0]])

    def base_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """True pose of the arm base (panda/link0) in the world."""
        return self.body_pos("panda/link0"), self.body_mat("panda/link0")

    # ------------------------------------------------------------ recording
    def snapshot(self, camera: str = "overview", size=(640, 480)) -> np.ndarray:
        with mujoco.Renderer(self.model, height=size[1], width=size[0]) as r:
            r.update_scene(self.data, camera=camera)
            return r.render().copy()

    def save_gif(self, path, fps: int | None = None, max_frames: int = 400):
        from PIL import Image
        if not self.frames:
            return None
        frames = self.frames
        if len(frames) > max_frames:
            idx = np.linspace(0, len(frames) - 1, max_frames).astype(int)
            frames = [frames[i] for i in idx]
        fps = fps or int(round(1.0 / (CTRL_DT * self.record_every)))
        imgs = [Image.fromarray(f) for f in frames]
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0, optimize=True)
        return path

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
