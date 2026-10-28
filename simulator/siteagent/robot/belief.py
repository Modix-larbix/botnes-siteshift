"""What the robot believes about its own base pose, and how it measures it.

Frames: world is the site frame (z up, origin on the floor below the nominal robot position); base is the Panda link0
frame, where IK is solved. The belief is a base pose in the world (x, y, z and roll, pitch, yaw in degrees, ZYX). A
world target is converted to the base frame with the believed pose; if the belief is wrong, the real hand misses.

calibrate() (robot.skills) measures roll and pitch from the mount accelerometer, the base height by a guarded touch of
the worktop, and yaw and x, y by guarded touches of known worktop edges from the site plan. It uses only the IMU, arm
joint positions and robot contact forces.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

NOMINAL_WORKTOP_Z = 0.90
NOMINAL_HEIGHT_OFFSET = 0.05


def rpy_to_mat(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    r, p, y = (math.radians(a) for a in (roll_deg, pitch_deg, yaw_deg))
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def mat_to_rpy(R: np.ndarray) -> tuple[float, float, float]:
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


@dataclass
class BaseBelief:
    x: float = 0.0
    y: float = 0.0
    z: float = NOMINAL_WORKTOP_Z + NOMINAL_HEIGHT_OFFSET
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    calibrated: bool = False

    @staticmethod
    def nominal(worktop_z: float = NOMINAL_WORKTOP_Z) -> "BaseBelief":
        return BaseBelief(z=worktop_z + NOMINAL_HEIGHT_OFFSET)

    # ------------------------------------------------------------ transforms
    @property
    def R(self) -> np.ndarray:
        return rpy_to_mat(self.roll_deg, self.pitch_deg, self.yaw_deg)

    @property
    def t(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def world_to_base(self, p_world) -> np.ndarray:
        return self.R.T @ (np.asarray(p_world, float) - self.t)

    def base_to_world(self, p_base) -> np.ndarray:
        return self.R @ np.asarray(p_base, float) + self.t

    def rot_world_to_base(self, R_world) -> np.ndarray:
        return self.R.T @ np.asarray(R_world, float)

    def rot_base_to_world(self, R_base) -> np.ndarray:
        return self.R @ np.asarray(R_base, float)

    def gravity_in_base(self) -> np.ndarray:
        """Unit 'down' vector expressed in the base frame."""
        return self.R.T @ np.array([0, 0, -1.0])

    def to_dict(self) -> dict:
        return asdict(self)


def tilt_from_accel(a: np.ndarray) -> tuple[float, float]:
    """Roll and pitch (deg) of a static frame from its accelerometer reading (specific force, +g up)."""
    ax, ay, az = a
    roll = math.degrees(math.atan2(ay, az))
    pitch = math.degrees(math.atan2(-ax, math.hypot(ay, az)))
    return roll, pitch


def belief_error(b: BaseBelief, true_pos: np.ndarray, true_R: np.ndarray) -> dict:
    """Errors of a belief vs the true base pose (bench only: needs ground truth)."""
    tr, tp, ty = mat_to_rpy(true_R)
    dyaw = (b.yaw_deg - ty + 180) % 360 - 180
    return {
        "roll_err_deg": abs(b.roll_deg - tr), "pitch_err_deg": abs(b.pitch_deg - tp),
        "yaw_err_deg": abs(dyaw), "z_err_m": abs(b.z - true_pos[2]),
        "xy_err_m": float(np.hypot(b.x - true_pos[0], b.y - true_pos[1])),
        "true": {"x": float(true_pos[0]), "y": float(true_pos[1]), "z": float(true_pos[2]),
                 "roll_deg": tr, "pitch_deg": tp, "yaw_deg": ty},
    }
