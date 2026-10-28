"""Observation contract: what perception gives the robot's program, and nothing more.

Every record is plain data (JSON serializable with to_dict()). It never carries a simulator name, a body, geom or
joint index, a generator seed or an item name: things are named by an anonymous track id ("trk-007", in the order the
robot first perceived them) and a category from the public catalog. Region ids come from the dossier vocabulary
("drawer_unit:1", "sink_basin").

Frames and units: positions in meters, in the frame named by Pose.frame. "world_belief" is the site frame as the robot
believes it (camera measurement -> robot base frame -> world through the robot's BaseBelief), never computed with the
true base pose.

Site knowledge statuses (sitemodel): declared (stated by the dossier, not checked), observed (perceived, and no later
robot action could have changed it), unknown (never inspected: a thing not seen is unknown, not absent), stale
(observed, then a robot action could have changed it), held (in the robot's gripper after its own pick).

Pose provenance: every object or handle pose says where its numbers come from: "this_view", "earlier_view" (this view
was cut by the image edge, so an earlier whole view's pose is kept) or "dead_reckoned_after_action" (an earlier pose
shifted by the robot's own drawer motion); pose_step names the view. visibility_estimate is robot side: the segment's
pixel count over the catalog silhouette projected at the estimated pose, clipped to [0, 1].
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, is_dataclass

DECLARED, OBSERVED, UNKNOWN, STALE, HELD = "declared", "observed", "unknown", "stale", "held"
STATUSES = (DECLARED, OBSERVED, UNKNOWN, STALE, HELD)
WORLD_BELIEF = "world_belief"
SIM_LABEL = "simulation, perception simulee (simulated perception)"


def _round(x):
    if isinstance(x, float):
        return round(x, 4) if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_round(v) for v in x]
    return x


def to_dict(obj) -> dict:
    """Plain, rounded dict of a contract record (floats to 0.1 mm)."""
    if is_dataclass(obj):
        obj = asdict(obj)
    return _round(obj)


def to_json(obj, **kw) -> str:
    return json.dumps(to_dict(obj), **kw)


@dataclass
class Pose:
    position_m: list[float]
    sigma_m: list[float]  # 1-sigma uncertainty per axis (heuristic model, see perception.estimate)
    frame: str = WORLD_BELIEF
    yaw_rad: float | None = None  # long axis of an elongated object (mod pi), or rail axis
    sigma_yaw_rad: float | None = None
    quat_wxyz: list[float] | None = None  # rotation about z by yaw_rad (object upright on a surface)


@dataclass
class Provenance:
    sensor: str  # "robot_eye" (mast camera) or "wrist_cam" (hand camera); both simulated
    step: int  # the robot program's step counter (observations and actions)
    t_s: float  # robot clock (simulated time), seconds
    camera_position_m: list[float]  # believed camera pose, world_belief frame
    camera_forward: list[float]
    robot_q_cmd: list[float]  # commanded arm joints (rad); the wrist camera pose itself is computed
    # from the joint encoders (measured joints), not from this command
    base_belief: dict  # x, y, z, roll_deg, pitch_deg, yaw_deg, calibrated


@dataclass
class Detection:
    track_id: str
    category: str
    pose: Pose
    visibility_estimate: float  # robot side: n_pixels / catalog silhouette pixels (see top)
    n_pixels: int
    region: str  # region id it lies in (robot-side geometry), or "other"
    step: int  # step of the observation that carries this record
    sensor: str
    orientation_note: str = ""  # e.g. "lying: yaw only", "round: no yaw"
    pose_source: str = "this_view"  # this_view | earlier_view | dead_reckoned_after_action
    pose_step: int | None = None  # step of the view the pose numbers come from


@dataclass
class HandleObservation:
    track_id: str
    category: str  # drawer_handle | dishwasher_handle
    pose: Pose  # rail centre
    pull_dir: list[float]  # horizontal unit vector toward the robot, perpendicular to the rail
    unit_track: str | None
    visibility_estimate: float  # robot side (see top)
    n_pixels: int
    step: int
    sensor: str
    pose_source: str = "this_view"  # this_view | earlier_view | dead_reckoned_after_action
    pose_step: int | None = None


@dataclass
class ModuleObservation:
    track_id: str
    category: str
    centre: Pose  # rough: median of visible module points near the worktop
    bearing_deg: float  # from the site origin (nominal robot position), world_belief frame
    slot_estimate: str | None  # nearest slot of the site plan by bearing
    visibility_estimate: float | None  # None: no silhouette estimate for a module (fixed sigma)
    n_pixels: int
    step: int
    sensor: str


@dataclass
class InspectedRegion:
    region: str  # e.g. "drawer_unit:1", "sink_basin", "landing_tray"
    step: int
    sensor: str
    coverage: float  # share of the region's floor grid whose line of sight reached the floor
    n_points: int
    seen_tracks: list[str]
    seen_categories: list[str]
    note: str = ""
    # Floor grid of the region (rows along its first axis, e.g. the drawer axis away from the
    # handle; columns across), and per grid point, row-major: "1" seen (line of sight reached the
    # floor), "o" covered by a perceived object (its pixels), "0" not seen. Lets the site model
    # say whether an object of a given size could still be hidden in the unseen part.
    grid_shape: list[int] = field(default_factory=list)
    grid_pitch_m: list[float] = field(default_factory=list)
    grid_seen: str = ""
    # Per grid point: "1" if the point 3 cm above the floor (catalog VISIBILITY_LAYER_H) was in
    # direct view (an object at least that tall standing there would have shown), else "0".
    grid_seen_top: str = ""


@dataclass
class Observation:
    obs_id: str
    provenance: Provenance
    detections: list[Detection] = field(default_factory=list)
    handles: list[HandleObservation] = field(default_factory=list)
    modules: list[ModuleObservation] = field(default_factory=list)
    regions: list[InspectedRegion] = field(default_factory=list)
    label: str = SIM_LABEL
