"""Arm kinematics in the robot's own base frame (Panda link0 at the origin).

The robot's self-knowledge: a standalone copy of the arm model with no site in it. IK is solved here, so a wrong
belief about where the base is in the world makes the real hand miss (the base pose never enters this module).
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from ..world.build import PANDA_XML

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
# Folded home: hand above the base front, pointing down, out of the way of the worktop.
HOME = np.array([0.0, -0.55, 0.0, -2.45, 0.0, 1.95, 0.785])
# Posture bias used by the IK null space (elbow up, wrist mid-range).
REST = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 1.9, 0.785])
TCP_OFFSET = 0.1034  # tcp site, hand frame z (between the finger pads)


def quat_from_mat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).flatten())
    return q


def mat_from_quat(q) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def hand_down(yaw: float) -> np.ndarray:
    """Hand frame pointing down (hand z = -z), finger closing axis (hand y) at angle `yaw`."""
    y = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    z = np.array([0, 0, -1.0])
    x = np.cross(y, z)
    return np.column_stack([x, y, z])


def hand_horizontal(approach: np.ndarray, closing_vertical: bool = False) -> np.ndarray:
    """Hand pointing along a horizontal `approach` direction.

    closing_vertical=False: fingers close horizontally (side grasp of an upright object).
    closing_vertical=True: fingers close along the vertical (pinch a horizontal bar).
"""
    z = np.array([approach[0], approach[1], 0.0])
    z /= np.linalg.norm(z)
    if closing_vertical:
        y = np.array([0, 0, 1.0])
    else:
        y = np.cross([0, 0, 1.0], z)
    x = np.cross(y, z)
    return np.column_stack([x, y, z])


def rot_about(axis, angle) -> np.ndarray:
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, axis, angle)
    return mat_from_quat(q)


class ArmKinematics:
    """Standalone Panda (no site, base at the origin): FK, Jacobian, damped least squares IK."""

    def __init__(self):
        spec = mujoco.MjSpec.from_file(str(PANDA_XML))
        for k in list(spec.keys):
            spec.delete(k)
        spec.body("hand").add_site(name="tcp", pos=[0, 0, TCP_OFFSET], size=[0.008, 0, 0])
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        self.qadr = np.array([self.m.joint(j).qposadr[0] for j in ARM_JOINTS])
        self.dadr = np.array([self.m.joint(j).dofadr[0] for j in ARM_JOINTS])
        ids = [self.m.joint(j).id for j in ARM_JOINTS]
        self.lo = self.m.jnt_range[ids, 0].copy()
        self.hi = self.m.jnt_range[ids, 1].copy()
        self.sid = self.m.site("tcp").id
        self._jp = np.zeros((3, self.m.nv))
        self._jr = np.zeros((3, self.m.nv))
        # Collision geoms of the moving links (link3 .. hand): a 3x3x3 grid on each geom's bounding
        # box (geom frame), for the mast check. Points on the link axis alone miss the forearm
        # mesh, which sits up to 10 cm off the axis (otherwise link5 hit the mast).
        bodies = {self.m.body(n).id for n in ("link3", "link4", "link5", "link6", "link7", "hand")}
        grid = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], float)
        self._box_geoms, self._box_local = [], []
        for g in range(self.m.ngeom):
            if self.m.geom_bodyid[g] in bodies and (self.m.geom_contype[g] or self.m.geom_conaffinity[g]):
                c, h = self.m.geom_aabb[g, :3], self.m.geom_aabb[g, 3:]
                self._box_geoms.append(g)
                self._box_local.append(c + grid * h)
        self._box_local = np.array(self._box_local)  # (n_geoms, 27, 3)
        # Self-collision: the arm model's own collision geoms (contacts reported within 1 cm).
        self.m.geom_margin[:] = 0.01
        self._fadr = np.array([self.m.joint(j).qposadr[0] for j in ("finger_joint1", "finger_joint2")])
        fingers = {self.m.body("left_finger").id, self.m.body("right_finger").id}
        self._finger_pair = lambda b1, b2: b1 in fingers and b2 in fingers

    def fk(self, q) -> tuple[np.ndarray, np.ndarray]:
        """TCP position and rotation matrix in the base frame."""
        d = self.d
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, d)
        return d.site_xpos[self.sid].copy(), d.site_xmat[self.sid].reshape(3, 3).copy()

    def body_points(self, q) -> np.ndarray:
        """Origins of links 3..7 and hand (for simple self checks)."""
        self.fk(q)
        names = ["link3", "link4", "link5", "link6", "link7", "hand"]
        return np.array([self.d.xpos[self.m.body(n).id] for n in names])

    def ik(self, pos, R, q_init, iters=200, tol_pos=5e-4, tol_rot=5e-3, rot_weight=1.0,
           free_yaw_axis=None, posture=0.05) -> tuple[np.ndarray, float, float]:
        """Solve for joints putting the TCP at (pos, R) in the base frame.

        Returns (q, position error m, rotation error rad). Joint limits are enforced with a
        margin. A posture term in the null space keeps the elbow up.
        free_yaw_axis: if given (a unit vector), rotation about that axis is not constrained.
        posture: weight of the null-space pull toward REST (0 when tracking a path step by step
        from the previous configuration: no drift, so consecutive solutions stay continuous).
"""
        m, d = self.m, self.d
        q = np.clip(np.asarray(q_init, float).copy(), self.lo + 0.02, self.hi - 0.02)
        target_q = quat_from_mat(R)
        cur, neg, dq_quat, err_r = np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(3)
        ep = er = 1.0
        for it in range(iters):
            d.qpos[self.qadr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            err_p = pos - d.site_xpos[self.sid]
            mujoco.mju_mat2Quat(cur, d.site_xmat[self.sid])
            mujoco.mju_negQuat(neg, cur)
            mujoco.mju_mulQuat(dq_quat, target_q, neg)
            mujoco.mju_quat2Vel(err_r, dq_quat, 1.0)
            if free_yaw_axis is not None:
                a = np.asarray(free_yaw_axis, float)
                err_r = err_r - a * (a @ err_r)
            ep, er = float(np.linalg.norm(err_p)), float(np.linalg.norm(err_r))
            if ep < tol_pos and er < tol_rot:
                break
            mujoco.mj_jacSite(m, d, self._jp, self._jr, self.sid)
            Jr = self._jr[:, self.dadr]
            if free_yaw_axis is not None:
                Jr = Jr - np.outer(a, a @ Jr)
            J = np.vstack([self._jp[:, self.dadr], rot_weight * Jr])
            e = np.concatenate([err_p, rot_weight * err_r])
            lam = 1e-4 + 1e-2 * min(1.0, ep + er)
            JJt = J @ J.T + lam * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, e)
            # Null-space posture bias.
            N = np.eye(7) - J.T @ np.linalg.solve(JJt, J)
            if posture:
                dq += N @ (posture * (REST - q))
            # Push away from joint limits.
            dq = np.clip(dq, -0.25, 0.25)
            q = np.clip(q + dq, self.lo + 0.01, self.hi - 0.01)
        return q, ep, er

    # The robot's own camera mast stands behind the base (mount plate frame x = -0.30 m, base
    # frame z from 0 to 1.05 m). Self knowledge: configurations that bring a link near it are
    # rejected.
    MAST_XY = np.array([-0.30, 0.0])
    MAST_CLEAR = 0.11
    MAST_BOX_CLEAR = 0.035  # mast half width 0.012 m plus a 2.3 cm margin, for link bounding boxes

    def link_points(self, q) -> np.ndarray:
        """Points along links 3..7 and the hand (base frame), for self-clearance checks."""
        self.fk(q)
        names = ["link2", "link3", "link4", "link5", "link6", "link7", "hand"]
        P = np.array([self.d.xpos[self.m.body(n).id] for n in names])
        mids = 0.5 * (P[1:] + P[:-1])
        # Extra points along the forearm (link4 -> link5 origin, 0.38 m long).
        fq = np.array([P[2] + f * (P[3] - P[2]) for f in (0.25, 0.75)])
        return np.vstack([P, mids, self.d.site_xpos[self.sid][None], fq])

    def link_box_points(self, q) -> np.ndarray:
        """Grid points on the bounding boxes of the moving links' collision geoms (base frame)."""
        self.fk(q)
        d = self.d
        R = d.geom_xmat[self._box_geoms].reshape(-1, 3, 3)
        x = d.geom_xpos[self._box_geoms]
        return (np.einsum("gij,gkj->gki", R, self._box_local) + x[:, None, :]).reshape(-1, 3)

    SELF_MARGIN = 0.005

    def self_collides(self, q) -> bool:
        """Any two links of the arm (fingers open) closer than SELF_MARGIN (the old look
        pose pressed a finger into link 1; a joint move from a twisted posture pressed link 5
        into link 2)."""
        m, d = self.m, self.d
        d.qpos[self.qadr] = q
        d.qpos[self._fadr] = 0.04
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_collision(m, d)
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist < self.SELF_MARGIN and not self._finger_pair(m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]):
                return True
        return False

    def clear_of_mast(self, q) -> bool:
        P = self.link_points(q)
        d = np.linalg.norm(P[:, :2] - self.MAST_XY, axis=1)
        if not np.all((d > self.MAST_CLEAR) | (P[:, 2] > 1.10)):
            return False
        B = self.link_box_points(q)
        d = np.linalg.norm(B[:, :2] - self.MAST_XY, axis=1)
        return bool(np.all((d > self.MAST_BOX_CLEAR) | (B[:, 2] > 1.10)))

    def posture_cost(self, q, azimuth: float) -> float:
        """Prefer the usual elbow-up posture: upper arm not twisted, shoulder facing the target."""
        da = (q[0] - azimuth + math.pi) % (2 * math.pi) - math.pi
        margin = float(min(np.min(q - self.lo), np.min(self.hi - q)))
        return float(abs(q[2]) + 0.3 * abs(q[4]) + 0.5 * abs(da) + 5.0 * max(0.0, 0.15 - margin))

    def untwist(self, q, pos, R, max_step=0.02, gain=0.15):
        """Move q in the null space of the pose task toward a relaxed posture (upper arm not
        twisted, shoulder facing the target), then re-converge the pose. Used every control tick
        while tracking a path, so the arm drifts back to a normal posture as it travels."""
        m, d = self.m, self.d
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_jacSite(m, d, self._jp, self._jr, self.sid)
        J = np.vstack([self._jp[:, self.dadr], self._jr[:, self.dadr]])
        pref = q.copy()
        pref[0] = math.atan2(pos[1], pos[0])
        pref[2] = 0.0
        pref[4] = 0.0 if abs(q[4]) < 1.5 else q[4]
        N = np.eye(7) - np.linalg.pinv(J, rcond=1e-3) @ J
        dq = np.clip(N @ (gain * (pref - q)), -max_step, max_step)
        q2 = np.clip(q + dq, self.lo + 0.01, self.hi - 0.01)
        q2, ep, er = self.ik(pos, R, q2, iters=10)
        return q2, ep, er

    def solve(self, pos, R, seeds, free_yaw_axis=None, tol_pos=2e-3, tol_rot=0.02, q_ref=None):
        """Try all seeds; among converged, mast-clear solutions return the one with the best
        posture (plus a small cost for moving away from q_ref). Returns (q, ok, pos err, rot err)."""
        az = math.atan2(pos[1], pos[0])
        best, fallback = None, None
        for s in seeds:
            q, ep, er = self.ik(pos, R, s, free_yaw_axis=free_yaw_axis)
            if ep < tol_pos and er < tol_rot:
                if not self.clear_of_mast(q):
                    continue
                c = self.posture_cost(q, az)
                if q_ref is not None:
                    c += 0.05 * float(np.abs(q - q_ref).sum())
                if best is None or c < best[0]:
                    best = (c, q, ep, er)
            else:
                score = ep + 0.1 * er
                if fallback is None or score < fallback[0]:
                    fallback = (score, q, ep, er)
        if best is not None:
            _, q, ep, er = best
            return q, True, ep, er
        if fallback is None:
            return np.asarray(seeds[0], float), False, 1.0, 1.0
        _, q, ep, er = fallback
        return q, False, ep, er
