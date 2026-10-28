"""Robot design geometry the robot knows about itself: where its cameras are.

Camera placements are part of the robot's design, so the robot knows them relative to its base (hand-eye
extrinsics), never through the simulator. world.build imports these constants to place the cameras in the simulated
robot; robot-side code imports them from here, so it has no import path into the kitchen builder. robot_eye sits on
the mast; wrist_cam is a depth camera on the hand looking along the hand axis, with the fingers out of its image.
"""
from __future__ import annotations

import mujoco
import numpy as np

EYE_POS_PLATE = (-0.30, 0.0, 1.08)
EYE_TARGET_PLATE = (0.42, 0.0, -0.10)
EYE_FOVY = 80.0
LINK0_ON_PLATE_Z = 0.015  # panda/link0 origin above the mount plate centre
WRIST_CAM_POS_HAND = (0.06, 0.0, 0.03)
WRIST_CAM_QUAT_HAND = (0.0, 1.0, 0.0, 0.0)  # 180 deg about hand x: camera -z = hand +z
WRIST_CAM_FOVY = 58.0


def look_at_quat(pos, target, up=(0, 0, 1)) -> list[float]:
    fwd = np.asarray(target, float) - np.asarray(pos, float)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, fwd)
    mat = np.column_stack([right, cam_up, -fwd]).flatten()
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, mat)
    return list(q)


def robot_eye_in_base() -> tuple[np.ndarray, np.ndarray]:
    """robot_eye pose in the panda/link0 (base) frame: position, rotation (camera looks along -z,
    image up = +y). Pure robot design geometry (no site, no simulator state)."""
    p = np.array(EYE_POS_PLATE, float) - np.array([0.0, 0.0, LINK0_ON_PLATE_Z])
    q = np.array(look_at_quat(EYE_POS_PLATE, EYE_TARGET_PLATE))
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    return p, R.reshape(3, 3)


def wrist_cam_in_hand() -> tuple[np.ndarray, np.ndarray]:
    """wrist_cam pose in the hand frame (robot design geometry)."""
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.array(WRIST_CAM_QUAT_HAND, float))
    return np.array(WRIST_CAM_POS_HAND, float), R.reshape(3, 3)
