"""The host side of robot programs: the only door between a program and the robot.

GuardedHost is what the interpreter calls (in process, or through the sandbox pipe). It re-checks every request
against the skill API (the program side is not trusted), enforces the episode limits before each call (skill calls,
observations, readings, simulated time; shared by all corrections of one episode, never reset), records every
request and returns plain data only. A backend does the work:
- StubKitchen: a toy kitchen in plain Python for unit tests of the loop and memory; it proves nothing about the robot;
- FacadeBackend: the wiring to the skill facade, perception and site model (programs.facade, perception.Perception,
  sitemodel.SiteModel). Its perception bookkeeping after each gesture follows the scripted chain (search.SiteSearch).

Robot calls. A composite skill can make several facade calls (a look tries up to 15 arm poses). A skill starts only
if its worst case (WORST_ROBOT_CALLS) fits in the remaining skill-call budget, so a program that respects its cap
cannot end above the bench's limit. Simulated time is checked before each start only (a started skill may end above
it; the evaluator then reports budget_exceeded).

Drawer pulls are bounded by the robot's own belief: on a handle the robot tracks, open_drawer pulls at most the travel
left (catalog DRAWER_MAX_OPENING minus the believed opening); with less than OPEN_LEFT_MIN_M left it does not move and
returns "already_open". Re-pulling a drawer that is already out was the main cause of drawer contact incidents.

Pick precondition (GuardedHost.pick_precondition), checked before any backend motion for every pick, new or stored
program: the target must be associated with exactly one non-held track of its category (within assoc_m), seen in a
view this host delivered with pose_source "this_view" and no relevant action since (pose_moving_records: a drawer
gesture that may have moved its drawer, a failed pick, a hold, a grip loss, any contact incident), with a horizontal
sigma under the category's bound and the commanded position within max_offset_xy_m / max_offset_z_m of the accepted
estimate. Otherwise the pick is refused without motion (pick_target_not_perceived, pick_target_ambiguous,
pick_pose_stale, pick_pose_uncertain, pick_target_offset). A refusal is not a contact incident, not a grip event and
not a robot call, but it counts against the program's skill-call cap so a program cannot loop on refusals. Only the
sponge is gated (PICK_POSE_POLICY); sigma_m is a heuristic and the 1 cm bound was chosen on development episodes.

Before a top pick the host also checks the unit-top clearance (pick-pose-3) and the overhead check of the pose frame
(perception.overhead): something over the target refuses the pick before any motion.

Reuse mode: when the loop runs a stored program it sets reuse_entry, and before every skill the host checks that
gesture's critical precondition at the moment of use (agent.memory.precondition).

Robot-side contracts recorded with their versions (TASK_EVIDENCE_VERSION, the drawer booking rule, the stop rule):
task_evidence() says whether this episode's own records show the task done (a pick and a scrub that returned ok on
the bowl pose observed before the scrub, no contested evidence); drawer notes record what the perception booked after
each drawer gesture (an opposed close whose travel did not reach the end stop stays open or unknown).

SCENE STATE (information profile s5-info-1.0 only): render_scene_state() turns the robot-side readings (search scope,
tracks, zone labels, budget, hand) into the text the model sees. Under the default profile s5-info-0.9 nothing of it is
sent. Zone labels come from the robot's own readings, never from the simulator.

Simulation, simulated perception; robot side only.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from . import api as _api
from .api import API, OBJECT_CATEGORIES, ArgError, check_args
from .interp import LimitReached, to_plain
from .limits import EpisodeLimits, ProgramLimits
from .memory import precondition


# Worst-case facade calls of one host skill (FacadeBackend): SiteSearch.look_down_at tries 3 heights x 5 yaws;
# a failed open_drawer / close_drawer is followed by the release clearance (open_gripper + move_to).
WORST_ROBOT_CALLS = {"look_down_at": 15, "look_at_region": 15, "open_drawer": 3, "close_drawer": 3}
# The fewest robot calls any skill of the API reserves (WORST_ROBOT_CALLS.get(name, 1)): 1. Once skill_calls +
# MIN_SKILL_WORST > max_skill_calls no skill can start (GuardedHost.physical_end).
MIN_SKILL_WORST = min(WORST_ROBOT_CALLS.get(n, 1) for n, e in API.items() if e["kind"] == "skill")
# Robot-side state (prompt 0.2): a drawer the robot believes pulled out by 3 cm or more is reported as open.
OPEN_BELIEF_M = 0.03
# open_drawer pulls at most the travel left (catalog DRAWER_MAX_OPENING = 0.24 m, robot side, the same constant the
# scripted chain uses; a test checks they match); below 3 cm left (search.SiteSearch._pull) it does not move and returns
# "already_open".
DRAWER_TRAVEL_M = 0.24
OPEN_LEFT_MIN_M = 0.03
SKILL_MIN_PULL_M = 0.02  # open_drawer's own lower bound on distance (agent.api)
# already_open is idempotent. In development, programs gave up after ok False / already_open. The protection
# stays (no motion, no new observation, nothing learned about the contents); the result now says the state asked for is
# reached: ok True, reason "already_open", moved 0.0, arm_moved False, already_open True, believed_opening_m. Only a
# drawer whose handle the robot tracks with a known opening can be already_open: an unknown opening (after a failed pull
# or push) or a drawer it believes stuck is never reported open. Each such call still counts as a skill call (episode
# and program skill caps bound repetitions). Same convention for the three models and every agent backend
# (ALREADY_OPEN_CONVENTION).
ALREADY_OPEN_CONVENTION = "already-open-1 (ok True, no motion)"
# Pick precondition (see the module docstring). Per category: association distance (3D), horizontal sigma bound, and the
# largest offset of the commanded pos from the accepted estimate (xy, z).
PICK_POSE_POLICY = {"sponge": {"assoc_m": 0.03, "max_sigma_xy_m": 0.010, "max_offset_xy_m": 0.010,
                               "max_offset_z_m": 0.020}}
# pick-pose-3: the unit-top clearance of the scripted chain (search.SiteSearch.MIN_TOP_CLEARANCE, a test checks they
# match) is read and recorded on every checked pick; it refuses only when PICK_UNIT_TOP_ACTIVE is set. Rule B of the
# overhead check is always active (Perception.overhead_check). The switch was set only after the recorded runs of both
# perception configurations reached each of the 180 recorded successful picks without refusing one (smallest clearance
# 0.122 m against 0.03 m); flipping it is this one line (the tests cover both states).
PICK_UNIT_TOP_ACTIVE = True
PICK_UNIT_TOP_MIN_M = 0.03
PICK_PRECONDITION_VERSION = ("pick-pose-3 (pick-pose-2: sponge, declared or perceived at pos: 3 cm association, 1 cm "
                             "sigma_xy, fresh view, pos within 1 cm xy and 2 cm z of the estimate; then overhead rule B "
                             "on the pose frame, 20 points; unit-top 3 cm "
                             + ("active)" if PICK_UNIT_TOP_ACTIVE else "recorded)"))
PICK_REFUSAL_CODES = ("pick_target_not_perceived", "pick_target_ambiguous", "pick_pose_uncertain", "pick_pose_stale",
                      "pick_target_offset", "pick_overhead_obstruction", "pick_under_unit_top")
# refusals about a target the robot DOES perceive near pos (loop.help_review_note: a help after one of them is final)
PICK_PERCEIVED_REFUSALS = ("pick_target_ambiguous", "pick_pose_uncertain", "pick_pose_stale", "pick_target_offset",
                           "pick_overhead_obstruction", "pick_under_unit_top")
# Neutral hints (what the robot found, never a help kind)
PICK_OVERHEAD_HINT = "something lies over the target: a top grasp would hit it"
PICK_UNIT_TOP_HINT = "the target is too close to the unit top for a top grasp"
PICK_REFUSED_STATUS = "precondition_refused"
PICK_HINT = 'look_at_region(region) then observe("wrist_cam"), then pick the new detection'
# The offset hint matches PICK_HINT (a new delivered view first), so following it can never lead to a new pick
# without a new observation
PICK_OFFSET_HINT = ('look_at_region(region) then observe("wrist_cam"), then pick at the position_m of the new detection, '
                    "never at an older one")
BOUND_EPS_M = 1e-9  # the bounds are inclusive; 1e-9 m absorbs float rounding of a difference (exactly 3 cm passes)
SURFACE_REGIONS = ("sink_basin", "landing_tray", "drying_mat")  # a drawer motion does not reach them
# A FAILED drawer gesture moved the drawer only if it reports a travel above 5 mm (the same 5 mm as the perception
# bookkeeping of a failed push, FacadeBackend.skill) or failed after the handle was pinched (robot.skills: pull_/push_
# forms, lost_handle, stuck, partial, no_end_stop, not_closed). An approach that closed on nothing (no_grasp_handle,
# no_handle, unreachable, no_path, hand_busy) moved no drawer; a contact on the way is a contact incident and counts on
# its own.
DRAWER_MOVED_MIN_M = 0.005
DRAWER_PINCHED_REASONS = ("lost_handle", "stuck", "partial", "no_end_stop", "not_closed")
# Failures robot.skills returns before the rail is pinched (drawer_gesture_before_pinch)
DRAWER_BEFORE_PINCH_REASONS = ("hand_busy", "bad_arg", "budget_exceeded", "no_grasp_handle", "collision_on_approach",
                               "unreachable", "no_path")
# What FacadeBackend booked in the perception for a drawer gesture (records-only drawer_note); drawer-booking-2:
# close_drawer_short (an ok opposed close whose travel did not reach the need: residual noted, opening unknown) and
# close_drawer_unverified (sufficient travel on an opening already unknown)
DRAWER_NOTES = ("open_drawer", "close_drawer", "close_drawer_not_opposed", "open_drawer_failed", "close_drawer_failed",
                "close_drawer_short", "close_drawer_unverified")
CLOSE_PUSH_OPPOSED_COS = -0.7  # a close books the drawer closed only when push_dir opposes the handle's pull
# Drawer booking: an ok opposed close books the drawer closed only when the hand's travel along the closing axis reaches
# need - CLOSE_TRAVEL_TOL_M (close_travel_check). In 616 recorded closes the travel exceeded the believed opening by at
# least 6 mm, and blocked MuJoCo closes overshoot by 1.5 to 1.7 cm, so a stop leaving about 2.6 cm or more out is
# detected (OPEN_BELIEF_M is 3 cm). A heuristic, not a proof of closing (declared).
# open-full-1 (configuration K1 only): when the robot tracks the handle with a known opening and the program asks
# OPEN_FULL_MIN_ASK_M or more but less than the travel left, the host pulls the travel left (to the end stop);
# min_distance is the program's own, else the asked distance minus OPEN_FULL_MIN_DISTANCE_MARGIN_M. Shorter requests,
# an unknown opening, the bound cap and already_open are unchanged. The full pull ships only with the face-aware upright
# test: OPEN_FULL_ACTIVE goes with perception.estimate.SCOPE_CONFIGURATION "K1" (a test pairs the two), and
# open_full_active() also requires the upright mode to be active in perception at the call (else no rewrite).
# drawer-booking-3 = drawer-booking-2 + open-full-1. Records only: host record key "pull" {rule, version, asked_m,
# pulled_m, min_distance_m, ...}. If perception ran K2 with OPEN_FULL_ACTIVE set, open_full_active() would stay False
# (no request rewritten) and the pairing test would fail until both switches say K1.
OPEN_FULL_ACTIVE = True
OPEN_FULL_MIN_ASK_M = 0.20
OPEN_FULL_MIN_DISTANCE_MARGIN_M = 0.02
# See the module docstring: drawer-booking-4 = drawer-booking-3 + handle-reref-1 (DRAWER_REREF_VERSION) + bound-known-1
# (DRAWER_BOUND_VERSION). The two new rules apply in both configurations; open-full-1 stays K1 only, so K2 reads
# "drawer-booking-4 (open-full-1 off)" (before: drawer-booking-2).
_OPEN_FULL_RULE, _DRAWER_BOOKING_FULL = "open-full-1", "drawer-booking-4"
OPEN_FULL_VERSION = _OPEN_FULL_RULE if OPEN_FULL_ACTIVE else None
DRAWER_REREF_VERSION = "handle-reref-1"
DRAWER_BOUND_VERSION = "bound-known-1"
# Implicitly drawer-booking-1 (any ok opposed close booked closed)
DRAWER_BOOKING_VERSION = _DRAWER_BOOKING_FULL if OPEN_FULL_ACTIVE else _DRAWER_BOOKING_FULL + " (open-full-1 off)"
CLOSE_TRAVEL_TOL_M = 0.01
MANUAL_HOLD = "grasped_by_hand"  # pose_moving_records: what close_gripper grasped (data.grasped True) outside a pick
# See the module docstring: versions of the two robot-side contracts of this module.
INTERNAL_ACTIONS_VERSION = "internal-actions-1"
# task-evidence-1 -> task-evidence-2 (each scrub judged on the bowl pose observed before it, rule
# EVIDENCE_TEMPORAL_RULE); agent.memory.TASK_EVIDENCE_VERSION_REQUIRED must name the same version (tested)
TASK_EVIDENCE_VERSION = "task-evidence-2"
EVIDENCE_TEMPORAL_RULE = "bowl-pose-at-scrub-1"
# The host side of the loop's stop rule (agent.loop STOP_RULE_VERSION names the same id): GuardedHost.physical_end,
# PHYSICAL_END_CODES its two codes (the episode refusal codes of call() that leave no skill)
PHYSICAL_END_RULE = "episode-physical-end-1"
PHYSICAL_END_CODES = ("episode_max_sim_s", "episode_max_skill_calls")
# a skill whose backend raised after the skill counter: recorded with this status (and reason), then re-raised
BACKEND_RAISED_STATUS = "backend_raised"
RAN_STATUSES = ("called", BACKEND_RAISED_STATUS)  # skill records that reached the backend (may have moved)
# calibrate preconditions: statuses distinct from the pick refusal (agent.loop reads that one)
CAL_HAND_REFUSED_STATUS = "hand_precondition_refused"  # reason "hand_busy"
CAL_DRAWER_REFUSED_STATUS = "drawer_precondition_refused"  # reason "drawer_open"
# origins of internal entries: the drawer release clearance (FacadeBackend._release), a composite look's arm poses
ORIGIN_CLEARANCE, ORIGIN_LOOK = "release_clearance", "look_attempt"
INTERNAL_MAX = 32  # entries kept per record (a composite look makes at most 15 facade calls); more are unitemized
# task-evidence-1: the public target (feedback.public_task, criteria 1.4 target and scrub_request) and its defaults
EVIDENCE_TARGET_DEFAULT = {"category": "bowl", "region": "landing_tray"}
EVIDENCE_SCRUB_S_DEFAULT = 3.0
EVIDENCE_TARGET_XY_M = 0.03  # a scrub matches a bowl track when its target_pos lies within 3 cm (xy) of a detection
SCRUB_CONTACT_REASONS = ("weak_contact", "scrub_cut_short")  # failed scrubs that touched something
# task-evidence-2: a measured detection of another object within EVIDENCE_CONTEST_XY_M (xy) of the bowl's latest pose
# contests it (3 cm, the evidence tolerance); never the task's tool (EVIDENCE_TOOL, the one sponge, after E1)
# nor an object seen more than its own height above the bowl's top (a held or carried object). EVIDENCE_OBJECT_SHAPES:
# (top above centre, height) per category, the values of perception.catalog.OBJECT_SHAPES (this module imports nothing
# of the robot; a test checks they match); a category not listed uses EVIDENCE_HEIGHT_DEFAULT_M (the tallest, so fewer
# detections are excluded).
EVIDENCE_CONTEST_XY_M = 0.03
EVIDENCE_TOOL = "sponge"
EVIDENCE_OBJECT_SHAPES = {"cup": (0.046, 0.09), "bowl": (0.023, 0.044), "fork": (0.006, 0.012), "spoon": (0.006, 0.012),
                          "sponge": (0.022, 0.036), "spray_bottle": (0.115, 0.23), "towel": (0.0125, 0.025)}
EVIDENCE_HEIGHT_DEFAULT_M = 0.23
EVIDENCE_SCRUBS_MAX = 10  # entries kept in task_evidence["scrubs"] (records only)
POSE_STATES = ("unseen", "unmeasured", "stale", "contested", "superseded", "valid")  # pose_before, in order of checking
# (see the module docstring). The information profiles (agent.prompt.INFO_PROFILES names the same two; the integration
# test checks they match).
INFO_PROFILE_V0_9 = "s5-info-0.9"
INFO_PROFILE_V1_0 = "s5-info-1.0"
INFO_PROFILES = (INFO_PROFILE_V0_9, INFO_PROFILE_V1_0)
SCOPE_NOTE_CATEGORY = "sponge"  # The 1.0 note override applies to search_scope("sponge") only
SCENE_STATE_VERSION = "scene-state-1"
SCENE_MAX_CHARS = 2500  # the target: optional lines are shed past it (SCENE_SHED_ORDER), never a not-cleared row
SCENE_HEADER_PREFIX = "SCENE STATE NOW"
# one reason per not-cleared row, in this precedence (render_scene_state, _scene_reason)
SCENE_REASONS = ("never_viewed", "low_coverage", "evidence_invalidated", "no_floor_grid", "module_pose_not_measured",
                 "unseen_part_could_hide", "reason_unknown")
# where a not-cleared region's open places are (robot side, FacadeBackend.search_detail and _scene_places)
SCENE_PLACES = ("under the unit top", "near the front wall", "in the open part", "on the surface", "outline not fitted",
                "grid points only")
# drawer tags (robot_state and host records only; never the truth, never the verifier's 0.05 m). "open" and
# "open (end stop)" are printed with the believed opening: "open X.XX m", "open X.XX m (end stop)".
SCENE_DRAWER_TAGS = ("open", "open (end stop)", "opening unknown", "not moved by the robot", "not believed open")
# shedding order past SCENE_MAX_CHARS: the objects line, the outside list, the obs ids of the cleared row (its count
# kept), then (declared, beyond the plan's three) the cleared row's region names (its count kept). The header, the
# legend, the hand, the outcome, the tracks and every not-cleared row with its annotations are never shed: a text whose
# never-shed part alone exceeds the target is kept whole (only with more than the generator's 3 drawers + 3 surfaces).
SCENE_SHED_ORDER = ("objects", "outside", "cleared_obs", "cleared_names")
SCENE_OBJECT_CATEGORIES = ("cup", "bowl", "fork", "spoon", "spray_bottle", "towel")
SCENE_MIN_COVER = 0.3  # sitemodel.MIN_COVER (this module imports nothing of the robot at import time; a test checks)
# field caps of the text (characters): region names as agent.loop._not_cleared, outcome, outside items, ids (obs and
# track ids), codes (actions, contact and refusal reasons: no_earlier_view_of_region has 25 characters,
# pull_collision_on_approach 26)
SCENE_REGION_MAX, SCENE_OUTCOME_MAX, SCENE_OUTSIDE_ITEM_MAX, SCENE_ID_MAX, SCENE_CODE_MAX = 24, 40, 32, 12, 32
SCENE_ROWS_MAX, SCENE_TRACKS_MAX, SCENE_OUTSIDE_MAX, SCENE_OBJECTS_MAX = 10, 3, 4, 8
SCENE_UNAVAILABLE = "reading unavailable (nothing is cleared)"
SCENE_HELD_EMPTY = "sponge track held but grip reads empty (grip event)"  # the fixed text
SCENE_ACTED_SINCE = "robot acted since"  # a track or object whose status is stale: the robot acted after its last view
# D-a: a region of the scope that the reading clears while the robot sees the sponge there (a found track observed in
# it, or its own last view listing the category) is printed on the line "<category> seen here k of n: ..." (at most
# SCENE_SEEN_MAX names, then "k more"), never in the cleared row: the cleared gloss ("not found in the part inspected")
# would be false there. Never shed; no obs id (the region's last view need not show the sponge: on the integration dry
# run's edge_back episode it was seen in obs-002 only).
SCENE_SEEN_HERE = "seen here"
SCENE_SEEN_MAX = 3
# "seen here" can be historical. The scope row's seen flag reads the region's last record, which the site model merges
# over views with no robot action on the region in between (sitemodel.SiteModel.integrate), and a found track keeps
# status "observed" after later views that do not show it. So a seen region is printed "seen here" only when the
# episode's latest delivered view (the view the header names) itself detects the category there or lists it in that
# region's own seen_categories, and no found track in the region says the robot acted since (status stale or held);
# every other seen region is printed on the line "<category> seen earlier, fresh location not guaranteed, k of n: ...".
# Both lines are never shed, carry no obs id (none is invented: the earlier sighting's view is not named) and never
# print a seen region as cleared; row.seen is kept (a region whose only sign is its merged row reads "seen earlier"). At
# most SCENE_SEEN_MAX names in all, the fresh ones first, each line then "k more". SCENE_SEEN_RULE is recorded in the
# Rendering record (records only).
SCENE_SEEN_EARLIER = "seen earlier, fresh location not guaranteed"
SCENE_SEEN_RULE = "seen-fresh-1"


def delivered_obs_id(rec: dict) -> str | None:
    """Review of the marker: the obs_id of a host log record that delivered a view (name "observe",
    status "called", obs_id a non-empty str), else None. The one record test of GuardedHost.delivered_observations;
    the analysis applies the same test to the saved host_records (a test checks they agree). The marker also counts
    an obs_id once; FacadeBackend (perception.estimate.Perception) and StubKitchen number their views obs-001,
    obs-002, ... per episode, never twice, so that dedupe never applies to them and the two counts coincide (tested)."""
    oid = rec.get("obs_id")
    if rec.get("name") == "observe" and rec.get("status") == "called" and type(oid) is str and oid:
        return oid
    return None


def _flat_plain(d: dict) -> dict:
    """The plain entries of a records-only detail (str keys; None, bool, int, float, str, or a list of
    those); anything else is dropped."""
    def plain(v):
        return v is None or type(v) in (bool, int, float, str)
    return {k: v for k, v in d.items() if isinstance(k, str) and (plain(v) or (type(v) is list and all(map(plain, v))))}


def ok_already_open(believed_opening_m: float) -> dict:
    return {"ok": True, "reason": "already_open",
            "data": {"moved": 0.0, "arm_moved": False, "already_open": True, "believed_opening_m": believed_opening_m}}


def open_full_active() -> bool:
    """open-full-1, K1 only: the full pull runs only when OPEN_FULL_ACTIVE is set AND the
    perception's upright mode is active (the face-aware upright test): never one without the other. Robot side
    (perception imported at the call: this module imports no perception at import time)."""
    if not OPEN_FULL_ACTIVE:
        return False
    from ..perception import estimate as _E  # noqa: PLC0415 (robot side)
    return "upright" in tuple(getattr(_E, "PLACEMENT_MODES_ACTIVE", ()))


def open_full_rewrite(a: dict, bound: dict | None):
    """(open-full-1), pure: (new open_drawer args, records-only pull record) when the
    rule rewrites the request, else None. Applies only when the robot tracks the handle with a known opening (bound not
    None, from FacadeBackend.drawer_pull_bound) and the program asks OPEN_FULL_MIN_ASK_M <= distance < the travel left:
    distance becomes the travel left (the end stop); min_distance is the program's own, else the asked distance minus
    OPEN_FULL_MIN_DISTANCE_MARGIN_M. A shorter request, an unknown opening, a request at or beyond the travel left (the
    bound cap) and already_open (handled before) are unchanged."""
    if bound is None:
        return None
    asked = a.get("distance")
    if isinstance(asked, bool) or not isinstance(asked, (int, float)):
        return None
    asked, left = float(asked), float(bound["left_m"])
    if not (asked >= OPEN_FULL_MIN_ASK_M and asked < left):
        return None
    md = a.get("min_distance")
    md2 = float(md) if md is not None else asked - OPEN_FULL_MIN_DISTANCE_MARGIN_M
    rec = {"rule": _OPEN_FULL_RULE, "version": _DRAWER_BOOKING_FULL, "asked_m": round(asked, 4),
           "pulled_m": round(left, 4), "min_distance_m": round(md2, 4),
           "program_min_distance_m": None if md is None else round(float(md), 4),
           "believed_opening_m": round(float(bound["believed_opening_m"]), 4)}
    return {**a, "distance": left, "min_distance": md2}, rec


@dataclass
class EpisodeCounters:
    skill_calls: int = 0  # robot (facade) calls, composite skills counted by their real calls
    observations: int = 0
    queries: int = 0
    refused: int = 0  # requests refused before the backend (arguments, budgets, reuse preconditions); meaning kept
    precondition_refusals: int = 0  # reuse preconditions (agent.memory) only, as before (bench column)
    pick_precondition_refusals: int = 0  # refused picks only (no motion, no robot call), counted apart

    def __post_init__(self):
        # Calibrate refusals (no motion, no robot call), counted apart. Plain attributes, not dataclass fields: the
        # fields are the episode's budget counters, pinned unchanged by a test (a refusal tally is no budget).
        self.hand_precondition_refusals = 0  # the hand possibly full (grip reading True or unreadable, or open_hold)
        self.drawer_precondition_refusals = 0  # a drawer believed open or of unknown opening


class GuardedHost:
    def __init__(self, backend, ep: EpisodeLimits | None = None, prog: ProgramLimits | None = None,
                 counters: EpisodeCounters | None = None, info_profile: str = INFO_PROFILE_V0_9):
        """info_profile is "s5-info-0.9" (default: the earlier information, byte for byte) or
        "s5-info-1.0" (the search_scope("sponge") note override, call()); any other value raises ValueError."""
        if info_profile not in INFO_PROFILES:
            raise ValueError(f"unknown information profile {str(info_profile)[:40]!r}")
        self.info_profile = info_profile
        self._backend = backend
        self._ep = ep or EpisodeLimits()
        self._prog = prog or ProgramLimits()
        self.counters = counters or EpisodeCounters()
        self.records: list[dict] = []
        self.reuse_entry: dict | None = None  # set by the loop while a stored program runs
        # obs_ids of the observations DELIVERED in this episode (delivered_observations), apart from the budget counter
        # counters.observations
        self.delivered_obs_ids: list[str] = []

    @property
    def delivered_observations(self) -> int:
        """The number of observations DELIVERED in this episode: an
        observe call whose backend returned a view that passed to_plain, was recorded in the host log (record status
        "called" with its obs_id, a non-empty string: delivered_obs_id) and was returned by call(); an obs_id is counted
        once. Distinct
        from counters.observations, the BUDGET counter, which is charged before the backend call (an attempt costs its
        observation and its nominal simulated time even when the camera fails) and is unchanged. Robot side, not
        counted against any budget. The loop's progress precondition of a reprise (agent.loop.reprise_has_progress)
        reads this marker: an observation that raised (no view returned) is no progress.
        Declared limit (isolated runs): call() returning is the host's delivery; the sandbox parent then writes the
        reply to the child. A failed write ends the program as sandbox_error (not in correct_on); a reply not written
        because the program's wall time ran out ends it as limit max_wall_s, the view then being in the host log and in
        the next request's observations."""
        return len(self.delivered_obs_ids)

    def _sim_s(self) -> float:
        """Robot clock + a NOMINAL simulated time per observation (observations
        cost time too; value provisional in agent_limits_s5_v0.json, to be fixed identically for B and C)."""
        return float(self._backend.sim_time_used()) + self.counters.observations * self._ep.observation_sim_s_nominal

    def budget(self) -> dict:
        return {"sim_s_used": round(self._sim_s(), 1), "max_sim_s": self._ep.max_sim_s,
                "observation_sim_s_nominal": self._ep.observation_sim_s_nominal,
                "skill_calls": self.counters.skill_calls, "max_skill_calls": self._ep.max_skill_calls,
                "observations": self.counters.observations, "max_observations": self._ep.max_observations}

    def _refuse(self, name, args, code, msg):
        self.counters.refused += 1
        self.records.append({"name": name, "args": args, "status": code, "message": msg})

    # ------------------------------------------------------------ episode gate, physical end
    def _sim_time_used_up(self, sim_s: float) -> bool:
        """The gate's own test of the simulated time (call(), skills and observations): sim_s (_sim_s(), the
        host clock) at or above max_sim_s. One expression for call(), episode_gate and physical_end."""
        return sim_s >= self._ep.max_sim_s

    def _skill_calls_exceeded(self, worst: int) -> bool:
        """The gate's own test of the skill-call budget for a skill that may make `worst` robot calls."""
        return self.counters.skill_calls + worst > self._ep.max_skill_calls

    def episode_gate(self, name, kind: str | None = None) -> str | None:
        """The episode-limit refusal code call(name, ...) would raise NOW, in call()'s order:
        the simulated time (skill, observe), the skill calls at the skill's worst case (WORST_ROBOT_CALLS), the
        observations (observe), the readings (query); None when no episode limit refuses it. kind defaults to the API's
        kind of name; a name outside the API or a terminal gets None (call() refuses those for another reason). Pure:
        no record, no count, no backend call except the clock (backend.sim_time_used, as call() reads it). Argument
        checks and the reuse, pick and calibrate preconditions are not episode limits and are not read."""
        if kind is None:
            kind = API[name]["kind"] if isinstance(name, str) and name in API else None
        if kind in ("skill", "observe") and self._sim_time_used_up(self._sim_s()):
            return "episode_max_sim_s"
        if kind == "skill" and self._skill_calls_exceeded(WORST_ROBOT_CALLS.get(name, 1)):
            return "episode_max_skill_calls"
        if kind == "observe" and self.counters.observations >= self._ep.max_observations:
            return "episode_max_observations"
        if kind == "query" and self.counters.queries >= self._ep.max_queries:
            return "episode_max_queries"
        return None

    def physical_end(self) -> dict | None:
        """agent.loop's stop rule PHYSICAL_END_RULE: the episode's physical
        end, or None. The episode is physically ended when the host's own gate refuses every skill of the API: the
        simulated time is used (code "episode_max_sim_s": _sim_time_used_up, the expression of call()) or no skill can
        start any more (code "episode_max_skill_calls": skill_calls + MIN_SKILL_WORST > max_skill_calls). By
        construction physical_end() is not None exactly when episode_gate(name) refuses a 1-call API skill (go_home,
        look_pose), so the loop and the gate agree exactly. Views from the current pose and readings may still be
        possible after a skill-call end (declared); a recoverable limit (a program's own limit, a composite or drawer
        gesture refused while a call is left, observations, readings) is not a physical end. Pure: host counters and
        the host clock only (never budget(), whose second rounding the gate does not use, the run report, the
        verifier or the facade's sticky exceeded flag); no record, no count. The clock is read once: a backend whose
        clock raises raises here as in call(). Returns {rule, code, sim_s (the unrounded host value), max_sim_s,
        skill_calls, max_skill_calls, observations, max_observations, queries, max_queries}, plain values."""
        sim_s = float(self._sim_s())
        if self._sim_time_used_up(sim_s):
            code = "episode_max_sim_s"
        elif self._skill_calls_exceeded(MIN_SKILL_WORST):
            code = "episode_max_skill_calls"
        else:
            return None
        c, ep = self.counters, self._ep
        return {"rule": PHYSICAL_END_RULE, "code": code, "sim_s": sim_s, "max_sim_s": ep.max_sim_s,
                "skill_calls": c.skill_calls, "max_skill_calls": ep.max_skill_calls, "observations": c.observations,
                "max_observations": ep.max_observations, "queries": c.queries, "max_queries": ep.max_queries}

    def robot_budget_exceeded(self) -> str | None:
        """The robot's own verdict
        on its episode budget (backend.robot_budget_exceeded: FacadeBackend reads programs.facade.SkillFacade.
        budget_state()["exceeded"], the robot clock unrounded against the facade's t0, or a skill call refused past
        max_skill_calls), or None, also for a backend without that reading (the toy kitchens). agent.loop refuses a
        done() that would otherwise be accepted while it says exceeded (refuse_done_over_budget): such a claim can
        never be a success, the bench counting the same clock from the same instant against the same budget. Not
        physical_end: the host clock adds a nominal second per observation and keeps the facade's 0.1 s rounding, so
        it would refuse true successes under the robot's own 600 s and still leave a gap. A reading: no record, no
        count."""
        fn = getattr(self._backend, "robot_budget_exceeded", None)
        return fn() if callable(fn) else None

    def call(self, name, args):
        if not isinstance(name, str) or name not in API:
            self._refuse(str(name)[:40], None, "unknown_name", "not in the skill API")
            raise ArgError(f"{str(name)[:40]!r} is not in the skill API")
        kind = API[name]["kind"]
        if kind == "terminal":
            self._refuse(name, None, "terminal_at_host", "terminals end the program in the interpreter")
            raise ArgError("terminals are not host calls")
        if not isinstance(args, dict):
            self._refuse(name, None, "bad_arg", "arguments must be a dict")
            raise ArgError("arguments must be named")
        try:
            checked = check_args(name, [], args)
        except ArgError as e:
            self._refuse(name, None, "bad_arg", str(e))
            raise
        if kind in ("skill", "observe"):
            if self._sim_time_used_up(self._sim_s()):  # the expression physical_end reads
                self._refuse(name, checked, "episode_max_sim_s", "simulated time budget used")
                raise LimitReached("episode_max_sim_s", f"simulated time budget ({self._ep.max_sim_s} s) used")
        if kind == "skill":
            worst = WORST_ROBOT_CALLS.get(name, 1)
            if self._skill_calls_exceeded(worst):
                self._refuse(name, checked, "episode_max_skill_calls",
                             f"skill-call budget: {name} may use {worst} robot calls, "
                             f"{self._ep.max_skill_calls - self.counters.skill_calls} left")
                raise LimitReached("episode_max_skill_calls",
                                   f"skill-call budget ({self._ep.max_skill_calls}) would be exceeded by {name} "
                                   f"(worst case {worst} robot calls)")
            if self.reuse_entry is not None:
                pc = precondition(self.reuse_entry, name, checked, self.observations(), self.records)
                if not pc["ok"]:
                    self.counters.precondition_refusals += 1
                    code = "reuse_precondition_" + pc["status"]
                    self._refuse(name, checked, code, pc["why"])
                    raise LimitReached(code, f"stored program not used for {name}: {pc['why']}"[:300])
            pick_check, drawer_region, sent = None, None, checked
            if name == "pick":  # robot-side pose precondition, before any motion (new and stored programs)
                refusal, pick_check = self.pick_precondition(checked)
                if refusal is not None:
                    # Counted apart only (refused and precondition_refusals keep their earlier meaning)
                    self.counters.pick_precondition_refusals += 1
                    refusal = to_plain(refusal, self._prog)
                    self.records.append({"name": name, "kind": kind, "args": checked, "status": PICK_REFUSED_STATUS,
                                         "ok": False, "reason": refusal["reason"], "data": refusal["data"],
                                         "message": str(refusal["data"].get("why", ""))[:200], "robot_calls": 0})
                    return refusal
                if pick_check is not None and checked.get("yaw") is None and pick_check.get("track_yaw_rad") is not None:
                    # A pick without yaw used to close the fingers about 12 deg off the sponge's narrow axis (one finger
                    # 9-15 N, the other 2-3 N), and the sponge slid out during the carry; the scripted chain S passes
                    # the perceived yaw. The host fills yaw from the robot's own accepted track, exactly like S (track
                    # yaw + pi/2, fingers across the long axis); a yaw given by the program is never changed. Robot side
                    # only. Only the robot gets the filled yaw (sent); the record's args stay the program's call (yaw
                    # None), so the run report of the next correction prompt never shows a yaw the program did not
                    # write. The yaw sent is pick_check.yaw_filled_from_track (records only). Declared limits: yaw_rad
                    # is used whatever its uncertainty (tracks do not expose sigma_yaw_rad; as S); the bench never yaws
                    # a sponge in its drawer or sink (fill error <= 1.44 deg over 297 picks). Trade-off kept: the
                    # aligned grasp is narrower (54.8 vs 56.4 mm), so the relaxed squeeze of scrub gesture v2.2 is
                    # lower, and the grasp yaw is kept through carry and scrub.
                    fill =(pick_check["track_yaw_rad"] + math.pi / 2 + math.pi) % (2 * math.pi) - math.pi
                    sent = {**checked, "yaw": round(fill, 6)}
                    pick_check = {**pick_check, "yaw_filled_from_track": round(fill, 6)}
            elif name == "calibrate":  # hand and drawer preconditions, before any motion and any count
                refusal = self.calibrate_precondition(checked)
                if refusal is not None:
                    return refusal
            elif name in ("open_drawer", "close_drawer"):
                drawer_region = self.drawer_region_at(checked["handle_pos"])  # before the motion (robot side)
            self.counters.skill_calls += 1
            hb = self.holding_reading()  # the grip before the skill (robot side, not counted)
            self._clear_call_detail()
            try:
                # sent is checked except a pick with the host's yaw fill
                r = to_plain(self._backend.skill(name, sent), self._prog)
            except BaseException as e:
                # The robot may have moved; the record exists before the exception goes on
                self._record_raised(name, kind, checked, e, hb, pick_check, drawer_region)
                raise
            extra = int(getattr(self._backend, "last_robot_calls", 1)) - 1
            if extra > 0:
                self.counters.skill_calls += extra
        elif kind == "observe":
            if self.counters.observations >= self._ep.max_observations:
                self._refuse(name, checked, "episode_max_observations", "observation budget used")
                raise LimitReached("episode_max_observations",
                                   f"observation budget ({self._ep.max_observations}) used")
            self.counters.observations += 1
            r = self._backend.observe(checked["camera"])
        else:
            if self.counters.queries >= self._ep.max_queries:  # 0.2: readings are capped per episode too
                self._refuse(name, checked, "episode_max_queries", "reading budget used")
                raise LimitReached("episode_max_queries", f"reading budget ({self._ep.max_queries}) used")
            self.counters.queries += 1
            r = self.budget() if name == "budget" else self._backend.query(name, checked)
        if kind != "skill":  # a skill result was made plain inside its try
            r = to_plain(r, self._prog)
        if (name == "search_scope" and self.info_profile == INFO_PROFILE_V1_0 and isinstance(r, dict)
                and checked.get("category") == SCOPE_NOTE_CATEGORY):
            # The 1.0 claim note; the rest of the reading and the record are unchanged
            r = {**r, "note": shared_phrase("SCOPE_NOTE_1_0")}
        rec = {"name": name, "kind": kind, "args": checked, "status": "called"}
        if kind == "skill":
            if isinstance(r, dict):
                rec.update(ok=r.get("ok"), reason=r.get("reason"), data=r.get("data"),
                           robot_calls=int(getattr(self._backend, "last_robot_calls", 1)))
                if pick_check is not None:
                    rec["pick_check"] = pick_check  # what the precondition accepted (records only, not the prompt)
                if name in ("open_drawer", "close_drawer"):
                    rec["drawer_region"] = drawer_region  # region the motion can reach (None: unknown, all drawers)
            rec.update(self._internal_fields(hb))  # after the keys above (records only, not the run report)
        elif kind == "observe" and isinstance(r, dict):
            rec.update(obs_id=r.get("obs_id"), n_detections=len(r.get("detections", [])))
        elif name == "holding" and type(r) is bool:
            rec.update(value=r)  # the robot's own grip reading (memory: a False reading ends a hold)
        self.records.append(rec)
        oid = delivered_obs_id(rec)
        if oid is not None and oid not in self.delivered_obs_ids:  # a delivered view (marker)
            self.delivered_obs_ids.append(oid)
        return r

    def observations(self) -> list[dict]:
        """Observations returned in this episode (for the prompt and the memory), as plain data."""
        return list(getattr(self._backend, "observation_log", []))

    def robot_state(self) -> dict:
        """The robot-side state given with every model request (prompt 0.2): the backend's own readings
        (grip, drawers the robot believes open, from its perception of the handles it moved) and the
        contact incidents of this episode, read from the host's own records of skill results. Robot
        side only: never the simulator's truth, the verifier, its thresholds or its verdict. Reading it
        moves nothing and counts nothing (no skill, observation or reading budget is used).
        Limit (declared): a contact the robot's own skills did not report is not listed; the verifier
        may still see it."""
        st = {"source": "robot side (own readings and own skill results)", "contains_oracle_information": False}
        fn = getattr(self._backend, "robot_state", None)
        if callable(fn):
            try:
                st.update(to_plain(fn(), self._prog))
            except Exception as e:  # noqa: BLE001 (a reading that fails is reported, never fatal)
                st["state_unavailable"] = type(e).__name__
        st["contact_incidents"] = contact_incidents(self.records)
        st["grip_events"] = grip_events(self.records)  # drops and releases of the held sponge, robot side
        return st

    # ------------------------------------------------------------ readings and records
    def holding_reading(self) -> bool | None:
        """The robot's own grip reading now (backend.holding_reading: FacadeBackend reads facade.holding(), a
        reading, not a skill), else the holding of backend.robot_state(). True / False, or None when unreadable (an
        exception, a non-bool value, no reading). Not counted, not recorded apart (the skill records carry it as
        holding_before / holding_after). Robot side only."""
        for attr in ("holding_reading", "robot_state"):
            fn = getattr(self._backend, attr, None)
            if not callable(fn):
                continue
            try:
                v = fn()
            except Exception:  # noqa: BLE001 (an unreadable sensor is unknown, never a value)
                return None
            if attr == "robot_state":
                v = v.get("holding") if isinstance(v, dict) else None
            return v if type(v) is bool else None
        return None

    def _clear_call_detail(self):
        """Forget the previous call's detail before a skill (a backend that exposes last_call_detail sets it on
        every call; FacadeBackend does on every return path)."""
        if getattr(self._backend, "last_call_detail", None) is not None:
            try:
                self._backend.last_call_detail = None
            except Exception:  # noqa: BLE001 (a read-only backend keeps its own reset)
                pass

    def _internal_fields(self, hb) -> dict:
        """The record fields of item 2 for the skill that just ran (or raised): holding_before, holding_after,
        and from backend.last_call_detail the internal entries (when any), internal_unitemized (when above 0) and
        clearance (when the backend reports one). Plain values only (internal_entry)."""
        out = {"holding_before": hb, "holding_after": self.holding_reading()}
        d = getattr(self._backend, "last_call_detail", None)
        if not isinstance(d, dict):
            return out
        raw = d.get("internal") if isinstance(d.get("internal"), list) else []
        ents = [internal_entry(e.get("skill"), e.get("ok"), e.get("reason"), e, e.get("t_s"), e.get("origin"))
                for e in raw if isinstance(e, dict)]
        un = d.get("unitemized")
        un = un if type(un) is int and un > 0 else 0
        un += len(raw) - len(ents) + max(0, len(ents) - INTERNAL_MAX)  # malformed or beyond the cap: not itemized
        if ents:
            out["internal"] = ents[:INTERNAL_MAX]
        if un > 0:
            out["internal_unitemized"] = un
        cl = d.get("clearance")
        if isinstance(cl, dict):
            out["clearance"] = {k: v for k, v in cl.items() if k in ("ran", "ok", "skipped")
                                and (v is None or type(v) in (bool, str))}
        if d.get("drawer_note") in DRAWER_NOTES:  # what the perception booked for a drawer gesture
            out["drawer_note"] = d["drawer_note"]
        if isinstance(d.get("close_check"), dict):  # records only, never the run report
            out["close_check"] = {k: v for k, v in d["close_check"].items()
                                  if isinstance(k, str) and (v is None or type(v) in (bool, int, float, str))}
        if isinstance(d.get("look"), dict):  # look-unseen-2: records only, never the run report
            out["look"] = {k: v for k, v in d["look"].items() if isinstance(k, str) and (
                v is None or type(v) in (bool, int, float, str)
                or (type(v) is list and all(x is None or type(x) in (bool, int, float, str) for x in v)))}
        if isinstance(d.get("pull"), dict):  # open-full-1: records only, never the report
            out["pull"] = {k: v for k, v in d["pull"].items()
                           if isinstance(k, str) and (v is None or type(v) in (bool, int, float, str))}
        # handle-reref-1, bound-known-1, tilt-retreat-1; records only, never the report
        for key in ("reref", "drawer_bound"):
            if isinstance(d.get(key), dict):
                out[key] = _flat_plain(d[key])
        if isinstance(d.get("retreat"), list):
            out["retreat"] = [_flat_plain(e) for e in d["retreat"][:INTERNAL_MAX] if isinstance(e, dict)]
        return out

    def _record_raised(self, name, kind, checked, e, hb, pick_check, drawer_region):
        """A backend that raised after the skill counter may have moved
        the arm (a FacadeBackend skill's facade calls are in the facade log already). Record it before the exception
        goes on: status and reason BACKEND_RAISED_STATUS, the exception's type name only (its text could name
        simulator objects), the robot calls the backend reports (also added to counters.skill_calls, so the episode
        reconciliation of task_evidence holds), its internal slice and the grip readings. It counts as moving
        (pose_moving_records). The exception's type name is kept under the records-only key error_type (not under
        message, which prompt.run_report copies into the next correction prompt); the prompt
        sees status, ok False and reason backend_raised only, the declared values. When the backend reports the
        result of the gesture itself (last_call_detail["gesture"]: its facade call returned before the backend
        raised), it is kept under the records-only key gesture (gesture_result) and the robot-side readers use it
        (skill_result); task_evidence voids any episode with such a record ("backend_raised")."""
        try:
            calls = max(1, int(getattr(self._backend, "last_robot_calls", 1)))
        except Exception:  # noqa: BLE001 (an unreadable count: the one call counted before the backend)
            calls = 1
        self.counters.skill_calls += calls - 1
        rec = {"name": name, "kind": kind, "args": checked, "status": BACKEND_RAISED_STATUS, "ok": False,
               "reason": BACKEND_RAISED_STATUS, "error_type": type(e).__name__[:60], "robot_calls": calls}
        if pick_check is not None:
            rec["pick_check"] = pick_check
        if name in ("open_drawer", "close_drawer"):
            rec["drawer_region"] = drawer_region
        try:
            d = getattr(self._backend, "last_call_detail", None)
            if isinstance(d, dict) and isinstance(d.get("gesture"), dict):
                rec["gesture"] = gesture_result(d["gesture"])
        except Exception:  # noqa: BLE001 (no gesture result: the record still voids the evidence)
            pass
        try:
            rec.update(self._internal_fields(hb))
        except Exception:  # noqa: BLE001 (the record must exist even when the detail cannot be read)
            rec.update(holding_before=hb, holding_after=None)
        self.records.append(rec)

    def calibrate_precondition(self, checked: dict) -> dict | None:
        """Calibrate preconditions. calibrate closes the fingers at full force, touches the worktop and its edges with
        the hand and opens it at the end (robot.skills); with the sponge held a test run hit 110 N and moved the base
        belief by 9.6 mm, and a drawer left open was hit at 208 N. Refused with no motion and no robot call, before
        counters.skill_calls (like the pick precondition), when
        (a) the grip reading is True or unreadable (holding_reading None), or (b) the host's own records show a hold
            that
        has not ended (open_hold: an ok pick, or a hold by hand, with no release or loss since; any object, a stricter
        reading, declared): status CAL_HAND_REFUSED_STATUS, reason "hand_busy", counters.hand_precondition_refusals;
        else (c) when backend.robot_state() lists a drawer believed open or of unknown opening (_drawers_believed_open:
        an unknown opening under OPEN_BELIEF_M counts unless every host gesture on that drawer since the last close
        booked closed failed before the pinch), or cannot be read: status CAL_DRAWER_REFUSED_STATUS, reason
        "drawer_open", counters.drawer_precondition_refusals. A backend without robot_state has no drawer belief to read
        (test doubles): (c) does not apply. The program receives {"ok": False, "reason": ..., "data": {"arm_moved":
        False}}; no new data key. Same path for model, correction and stored programs. Returns that result, or None.
        (c) also refuses when the host's own records show an open_drawer whose handle the robot could not resolve
        (drawer_region None: no tracked handle within 5 cm, so the perception's drawer belief never heard of it) and
        that may have moved its drawer (untracked_drawer_open: a drawer opened from the dossier's handle position
        before the handle was ever perceived, for the rest of the episode; a close_drawer of an unresolved handle
        too). It refused nothing in the recorded runs. The refusal's free text is kept under the records-only key why
        (not under message, which prompt.run_report copies into the next correction prompt); the prompt sees status,
        ok, reason and data only."""
        hb = self.holding_reading()
        held = open_hold(self.records)
        if hb is not False or held is not None:
            status, reason = CAL_HAND_REFUSED_STATUS, "hand_busy"
            why = ("the grip reading is unavailable" if hb is None else "the grip reading says the hand holds something"
                   if hb else "the hand still holds what it took (no release since)")
            self.counters.hand_precondition_refusals += 1
        else:
            drawers = self._drawers_believed_open()
            untracked = untracked_drawer_open(self.records)
            if drawers == [] and untracked is None:
                return None
            status, reason = CAL_DRAWER_REFUSED_STATUS, "drawer_open"
            why = ("the drawer state could not be read" if drawers is None
                   else f"the robot believes {str(drawers[0])[:24]} open or does not know its opening" if drawers
                   else f"the robot moved a drawer whose handle it does not track (record {untracked}); it may still "
                        "stick out")
            self.counters.drawer_precondition_refusals += 1
        result = {"ok": False, "reason": reason, "data": {"arm_moved": False}}
        self.records.append({"name": "calibrate", "kind": "skill", "args": checked, "status": status, "ok": False,
                             "reason": reason, "data": {"arm_moved": False},
                             "why": f"calibrate refused before any motion: {why}"[:200], "robot_calls": 0,
                             "holding_before": hb})
        return result

    def _drawers_believed_open(self) -> list | None:
        """The regions of the drawers backend.robot_state() lists as believed open or of unknown opening ([] when
        none, or when the backend has no robot_state), None when the state cannot be read. Replay E: one entry does
        not count, an unknown opening under OPEN_BELIEF_M whose every gesture since the last ok one on that drawer
        failed BEFORE the rail was pinched (drawer_gesture_before_pinch: the drawer cannot have moved). Without that
        exception it refused a recorded true success and a development calibrate, both after an open_drawer
        no_grasp_handle on a closed drawer. The believed opening alone does not tell: a failed pull is noted with no
        motion (FacadeBackend.skill), a failed push with the hand's travel, so a drawer pulled out 21 cm can read 0.0
        m unknown; any other failure (partial, stuck, lost_handle, pull_ / push_ forms, a collision or blocked abort
        during the stroke, a raised gesture) still counts, as does an entry without a readable opening or without a
        gesture of the host on its region."""
        fn = getattr(self._backend, "robot_state", None)
        if not callable(fn):
            return []
        try:
            st = to_plain(fn(), self._prog)
        except Exception:  # noqa: BLE001 (unreadable: refused, the conservative case)
            return None
        lst = st.get("drawers_believed_open") if isinstance(st, dict) else None
        if not isinstance(lst, list):
            return None

        def counts(d) -> bool:
            if not isinstance(d, dict):
                return True
            op = d.get("believed_opening_m")
            if not isinstance(op, (int, float)) or isinstance(op, bool) or op >= OPEN_BELIEF_M \
                    or d.get("opening_unknown") is not True:
                return True
            return not drawer_unmoved_since_last_ok(self.records, d.get("region"))

        return [d.get("region") if isinstance(d, dict) else d for d in lst if counts(d)]

    def task_evidence(self, task: dict, span=None, attempts: list | None = None) -> dict:
        """task-evidence-1: the pure task_evidence over this host's records, the observations it
        DELIVERED in this episode (delivered_obs_ids; a backend reused across episodes keeps older views in its log),
        the grip reading now (holding_reading), the robot's base position (query base_xy) for the target rule, and the
        episode reconciliation: counters.skill_calls and, for a facade with a call log, the facade calls no record
        itemized (backend.facade_log_unaccounted). attempts: the loop's attempt list (source, host_records_span), for
        the provenance of the pick and the scrub (the host does not know the attempts; the loop passes them); span: the
        final attempt's [n0, n1). Robot side only, not counted, not recorded; never the verifier or its thresholds."""
        ds = set(self.delivered_obs_ids)
        views = [o for o in self.observations() if isinstance(o, dict) and o.get("obs_id") in ds]
        fn = getattr(self._backend, "facade_log_unaccounted", None)
        try:
            unacc = fn() if callable(fn) else None
        except Exception:  # noqa: BLE001 (the log cannot be read: nothing proves it was all itemized)
            unacc = -1
        return task_evidence(self.records, views, list(self.delivered_obs_ids), self.holding_reading(), task, span,
                             base_xy=self._base_xy(), skill_calls=self.counters.skill_calls,
                             facade_log_unaccounted=unacc, attempts=attempts)

    def _base_xy(self) -> list | None:
        """The robot's own base position (query base_xy: its belief), not counted, not recorded; None if unread."""
        try:
            return finite_vec(to_plain(self._backend.query("base_xy", {}), self._prog), 2)
        except Exception:  # noqa: BLE001 (unknown: the target rule then needs a single bowl on the tray)
            return None

    def scope_now(self, category: str) -> dict | None:
        """The robot-side search scope of a category now (the same reading as the program's search_scope()),
        for the loop's help review. Like robot_state(): not counted, not recorded, never the simulator's truth.
        None if the backend cannot answer."""
        try:
            r = to_plain(self._backend.query("search_scope", {"category": category}), self._prog)
        except Exception:  # noqa: BLE001 (a reading that fails only disables the review)
            return None
        return r if isinstance(r, dict) else None

    # ------------------------------------------------------------ SCENE STATE readings
    def scene_readings(self, category: str = "sponge") -> dict:
        """The robot-side readings SCENE STATE is rendered from, as plain data. Not
        counted, not recorded, no side effect: the same readings as robot_state() and scope_now() (the backend's
        robot_state, its search_scope and tracks queries), FacadeBackend.search_detail when the backend has it (the
        toy kitchens do not: detail None, the renderer degrades), the host's own records and the observations it
        DELIVERED in this episode (delivered_obs_ids; a reused backend log can hold older views). Never backend.query
        ("handle_for_region") (handle-reref-1 forgets a moved handle on that query). A reading that fails gives a
        marker (None, detail_error, objects_unreadable, robot_state.state_unavailable), never an exception and never
        a clearance. Keys: version, profile, category, scope (scope_now verbatim), detail, detail_error,
        unseen_look_refusals {region: reason}, objects [{category, track_id, region, status, last_obs}] (latest
        first), objects_unreadable, last_obs_id, robot_state, open_full_active, drawer_gestures {regions,
        unresolved};, latest_view (scene_latest_view: the latest delivered view's obs id and the regions where that
        view itself shows the category; None when unknown)."""
        out = {"version": SCENE_STATE_VERSION, "profile": self.info_profile, "category": str(category)[:SCENE_CODE_MAX]}
        out["scope"] = self.scope_now(category)
        det, err = None, None
        fn = getattr(self._backend, "search_detail", None)
        if callable(fn):
            try:
                det = to_plain(fn(category), self._prog)
            except Exception as e:  # noqa: BLE001 (a reading that fails: no detail, the renderer degrades)
                det, err = None, type(e).__name__[:60]
        out["detail"] = det if isinstance(det, dict) else None
        out["detail_error"] = err
        try:
            views = self._scene_views()
        except Exception:  # noqa: BLE001 (an unreadable log: no view known)
            views = []
        try:
            out["unseen_look_refusals"] = scene_unseen_look_refusals(self.records, views)
        except Exception:  # noqa: BLE001
            out["unseen_look_refusals"] = {}
        out["objects"], out["objects_unreadable"] = self._scene_objects(views)
        ids = [o for o in self.delivered_obs_ids if isinstance(o, str)]
        out["last_obs_id"] = ids[-1] if ids else None
        try:  # where the latest delivered view itself shows the category (records the view's id)
            out["latest_view"] = scene_latest_view(views, category)
        except Exception:  # noqa: BLE001 (unknown: every sighting then reads "seen earlier")
            out["latest_view"] = None
        try:
            out["robot_state"] = self.robot_state()
        except Exception as e:  # noqa: BLE001 (robot_state() already reports a failing backend; this is a backstop)
            out["robot_state"] = {"state_unavailable": type(e).__name__[:60]}
        try:
            out["open_full_active"] = bool(open_full_active())
        except Exception:  # noqa: BLE001 (unknown: no end-stop tag)
            out["open_full_active"] = False
        try:
            out["drawer_gestures"] = scene_drawer_gestures(self.records)
        except Exception:  # noqa: BLE001 (unknown: every drawer not believed open reads "not believed open")
            out["drawer_gestures"] = {"regions": [], "unresolved": True}
        return out

    def scene_state(self, category: str = "sponge") -> dict:
        """{"text": the SCENE STATE text, "record": the rendering record with chars and sha256}. Always a text: an
        unreadable reading renders "reading unavailable (nothing is cleared)". Not counted, not recorded."""
        out = self._scene_render(category)
        text = out["text"]
        rec = dict(out["record"], chars=len(text), sha256=hashlib.sha256(text.encode("utf-8")).hexdigest())
        return {"text": text, "record": rec}

    def scene_reading_record(self, category: str = "sponge") -> dict:
        """The rendering record of the same rendering, without the text (the control arm records it)."""
        return dict(self._scene_render(category)["record"])

    def _scene_render(self, category: str) -> dict:
        rd = self.scene_readings(category)
        try:
            return render_scene_state(rd, rd.get("robot_state"), bool(rd.get("open_full_active")))
        except Exception as e:  # noqa: BLE001 (a renderer defect on a strange reading: the unavailable text)
            out = render_scene_state({"version": SCENE_STATE_VERSION, "profile": rd.get("profile"),
                                      "category": rd.get("category"), "scope": None,
                                      "last_obs_id": rd.get("last_obs_id")}, rd.get("robot_state"), False)
            out["record"]["render_error"] = type(e).__name__[:60]
            return out

    def _scene_views(self) -> list[dict]:
        """The observations delivered in this episode, in delivery order (delivered_obs_ids over the backend's log)."""
        by_id = {}
        for o in self.observations():
            if isinstance(o, dict) and isinstance(o.get("obs_id"), str):
                by_id[o["obs_id"]] = o
        return [by_id[i] for i in self.delivered_obs_ids if i in by_id]

    def _scene_objects(self, views: list[dict]) -> tuple[list[dict], list[str]]:
        """The tracks of SCENE_OBJECT_CATEGORIES (_read_tracks, not counted): [{category, track_id, region, status,
        last_obs}], last_obs = this episode's latest delivered view that detected the track (None if none), latest
        first (stable); and the categories whose tracks could not be read."""
        seen_at = {}
        for o in views:
            dets = o.get("detections")
            for d in dets if isinstance(dets, list) else []:
                if isinstance(d, dict) and d.get("track_id") is not None:
                    seen_at[str(d["track_id"])] = o.get("obs_id")
        order = {oid: i for i, oid in enumerate(self.delivered_obs_ids)}
        objs, bad = [], []
        for c in SCENE_OBJECT_CATEGORIES:
            tracks = self._read_tracks(c)
            if tracks is None:
                bad.append(c)
                continue
            for t in tracks:
                if not isinstance(t, dict) or t.get("category", c) != c:
                    continue
                tid = str(t.get("track_id"))
                reg = t.get("region")
                objs.append({"category": c, "track_id": tid[:SCENE_REGION_MAX],
                             "region": reg[:SCENE_REGION_MAX] if isinstance(reg, str) else None,
                             "status": str(t.get("status"))[:16], "last_obs": seen_at.get(tid)})
        objs.sort(key=lambda x: -order.get(x["last_obs"], -1))
        return objs, bad

    def target_perceived(self, category: str) -> bool | None:
        """Has the robot perceived category in this episode? True: a detection of it, or a region view
        listing it in seen_categories, in the episode's observation log, or a track of it (any status but declared: a
        track exists only after a view). False: none. None: a reading is unknown (malformed log entry, tracks
        unreadable) and nothing positive was found. Robot side, not counted, not recorded (like scope_now)."""
        unknown = False
        for o in self.observations():
            dets = o.get("detections") if isinstance(o, dict) else None
            if not isinstance(dets, list) or not all(isinstance(d, dict) for d in dets):
                unknown = True
                continue
            if any(d.get("category") == category for d in dets):
                return True
            regs = o.get("regions")
            if isinstance(regs, list) and any(isinstance(r, dict) and isinstance(r.get("seen_categories"), list)
                                              and category in r["seen_categories"] for r in regs):
                return True
        tracks = self._read_tracks(category)
        if tracks is None or not all(isinstance(t, dict) for t in tracks):
            return None
        if any(t.get("category", category) == category and t.get("status") != "declared" for t in tracks):
            return True
        return None if unknown else False

    def perceived_not_picked(self, category: str = "sponge") -> list[dict]:
        """The robot's own tracks of category that it perceives and has not picked in this episode:
        [] when a pick of that category returned ok in this episode (records: kind declared, or the pick
        precondition's category) or when a track of it is held; else the non-held tracks with a finite position_m
        that are fresh (status observed, pose_source this_view) or were detected in a view of this episode
        (observation log, by track_id), fresh first, at most 3, each {track_id, region, position_m (3 decimals),
        fresh}. Robot side (backend tracks, observation log, host records), not counted, not recorded; never the
        simulator's truth."""
        for r in self.records:
            if r.get("kind") == "skill" and r.get("status") == "called" and r.get("name") == "pick" and r.get("ok") is True:
                args = r.get("args") if isinstance(r.get("args"), dict) else {}
                if args.get("kind") == category or (r.get("pick_check") or {}).get("category") == category:
                    return []
        tracks = self._read_tracks(category)
        if not tracks:
            return []
        if any(isinstance(t, dict) and t.get("status") == "held" for t in tracks):
            return []
        seen = set()
        for o in self.observations():
            dets = o.get("detections") if isinstance(o, dict) else None
            for d in dets if isinstance(dets, list) else []:
                if isinstance(d, dict) and d.get("category") == category:
                    seen.add(str(d.get("track_id")))
        out = []
        for t in tracks:
            if not isinstance(t, dict) or t.get("category", category) != category:
                continue
            pos = finite_vec(t.get("position_m"), 3)
            fresh = t.get("status") == "observed" and t.get("pose_source") == "this_view"
            if pos is None or not (fresh or str(t.get("track_id")) in seen):
                continue
            region = t.get("region")
            out.append({"track_id": str(t.get("track_id"))[:24], "region": region[:24] if isinstance(region, str) else None,
                        "position_m": [round(x, 3) for x in pos], "fresh": fresh})
        out.sort(key=lambda x: not x["fresh"])
        return out[:3]

    def drawer_region_at(self, handle_pos) -> str | None:
        """The drawer region whose handle the robot perceives at handle_pos (backend.drawer_region_at, robot side),
        read before a drawer motion; None when unknown (then the motion counts as reaching every drawer region)."""
        fn = getattr(self._backend, "drawer_region_at", None)
        if not callable(fn):
            return None
        try:
            rid = fn(handle_pos)
        except Exception:  # noqa: BLE001 (a reading that fails is unknown, the conservative case)
            return None
        return rid if is_drawer_region(rid) else None

    def pick_precondition(self, args: dict) -> tuple[dict | None, dict | None]:
        """(refusal result or None, check summary for the record or None). See the module docstring. Reads the
        backend's tracks (not counted, not recorded, like scope_now), its observation log and the host's records. A
        pick declared of another kind is checked by the policy of a category the robot perceives within its assoc_m
        of pos (a sponge picked as kind "cup" is still a sponge pick), and the commanded pos must lie within
        max_offset_xy_m / max_offset_z_m of the accepted track's position_m (pick_target_offset)."""
        declared = args.get("kind")
        pos = [float(x) for x in args["pos"]]
        cat = declared if declared in PICK_POSE_POLICY else self._policy_category_near(pos)
        if cat is None:
            return None, None  # no policy category declared, none perceived at pos: unchecked (declared limit)
        pol = PICK_POSE_POLICY[cat]
        base = {"arm_moved": False, "category": cat, "assoc_m": pol["assoc_m"], "max_sigma_xy_m": pol["max_sigma_xy_m"],
                "max_offset_xy_m": pol["max_offset_xy_m"], "max_offset_z_m": pol["max_offset_z_m"]}
        if cat != declared:
            base["declared_kind"] = str(declared)[:20]

        def refuse(code, why, hint=PICK_HINT, **data):
            d = {**base, "nearest_m": None, "sigma_xy_m": None, **data, "why": why[:200], "hint": hint}
            return {"ok": False, "reason": code, "data": d}, None

        tracks = self._read_tracks(cat)
        if tracks is None:
            return refuse("pick_target_not_perceived", "the robot's tracks could not be read")
        near, nearest = {}, None
        for t in tracks:
            if not isinstance(t, dict) or t.get("status") == "held" or t.get("category", cat) != cat:
                continue  # a held track is in the hand, not a pick target
            p = finite_vec(t.get("position_m"), 3)
            if p is None:
                continue
            d = math.dist(p, pos)
            nearest = d if nearest is None else min(nearest, d)
            if d <= pol["assoc_m"] + BOUND_EPS_M:
                tid = str(t.get("track_id"))
                if tid not in near or d < near[tid][0]:
                    near[tid] = (d, t)
        nm = None if nearest is None else round(nearest, 4)
        if not near:
            return refuse("pick_target_not_perceived", f"no perceived {cat} within {pol['assoc_m']} m of pos",
                          nearest_m=nm)
        if len(near) > 1:
            return refuse("pick_target_ambiguous", f"{len(near)} perceived {cat} tracks within {pol['assoc_m']} m of pos",
                          nearest_m=nm, track_ids=sorted(near)[:5])
        d, t = next(iter(near.values()))
        tid, region = str(t.get("track_id")), t.get("region")
        info = {"nearest_m": round(d, 4), "track_id": tid, "region": region if isinstance(region, str) else None,
                "pose_source": t.get("pose_source"), "pose_step": t.get("pose_step")}
        view_i, view = self._view_of(t)
        sig = sigma_xy(t.get("sigma_m"))
        if view is not None:  # _view_of guarantees a list of detections
            det = next((x for x in view["detections"] if isinstance(x, dict) and x.get("track_id") == tid), None)
            ds = sigma_xy(det.get("sigma_m")) if det is not None and "sigma_m" in det else sig
            sig = None if sig is None or ds is None else max(sig, ds)
        sg = None if sig is None else round(sig, 4)
        if t.get("status") != "observed":
            return refuse("pick_pose_stale", f"track status {str(t.get('status'))[:20]}: the robot acted since its last "
                          "view of it", sigma_xy_m=sg, **info)
        if t.get("pose_source") != "this_view":
            return refuse("pick_pose_stale", f"pose not measured in a view ({str(t.get('pose_source'))[:30]})",
                          sigma_xy_m=sg, **info)
        if view is None:
            return refuse("pick_pose_stale", "no view recorded in this episode measured this pose", sigma_xy_m=sg, **info)
        after = [i for i in pose_moving_records(self.records, info["region"]) if i > view_i]
        if after:
            r = self.records[after[0]]
            return refuse("pick_pose_stale", f"pose measured before the robot's own {r.get('name')} "
                          f"{str(r.get('reason') or '')[:30]}".strip(), sigma_xy_m=sg, **info)
        if sig is None:
            return refuse("pick_pose_uncertain", "no finite sigma_m for this pose", **info)
        if sig > pol["max_sigma_xy_m"] + BOUND_EPS_M:
            return refuse("pick_pose_uncertain", f"horizontal sigma {sg} m above {pol['max_sigma_xy_m']} m",
                          sigma_xy_m=sg, **info)
        # 3 cm only associates; the pick must go to the estimate the checks above accepted (a stale side-view pos 2.9
        # cm from a fresh top view ended in collision_on_approach at 162 N)
        tp = finite_vec(t.get("position_m"), 3)
        off = {"offset_xy_m": round(math.dist(tp[:2], pos[:2]), 4), "offset_z_m": round(abs(pos[2] - tp[2]), 4)}
        if math.dist(tp[:2], pos[:2]) > pol["max_offset_xy_m"] + BOUND_EPS_M \
                or abs(pos[2] - tp[2]) > pol["max_offset_z_m"] + BOUND_EPS_M:
            return refuse("pick_target_offset", f"pos is {off['offset_xy_m']} m (horizontal), {off['offset_z_m']} m "
                          f"(vertical) from the robot's estimate of this {cat}", hint=PICK_OFFSET_HINT, sigma_xy_m=sg,
                          track_position_m=[round(x, 4) for x in tp], **off, **info)
        # pick-pose-3, before any motion: the unit-top clearance (recorded; refuses only when PICK_UNIT_TOP_ACTIVE),
        # then the overhead check on the pose frame (rule B, always active)
        clear, clear_ok = self._unit_top_clearance(tid)
        top = {"unit_top_clearance_m": None if clear is None else round(clear, 4),
               "unit_top_active": PICK_UNIT_TOP_ACTIVE}
        if PICK_UNIT_TOP_ACTIVE and (not clear_ok or (clear is not None
                                                      and clear < PICK_UNIT_TOP_MIN_M - BOUND_EPS_M)):
            why = ("the unit-top clearance could not be read" if not clear_ok else
                   f"target centre {round(clear, 4)} m in front of the unit top's edge, under {PICK_UNIT_TOP_MIN_M} m")
            return refuse("pick_under_unit_top", why, hint=PICK_UNIT_TOP_HINT, sigma_xy_m=sg,
                          clearance_m=None if clear is None else round(clear, 4), **info)
        ty = t.get("yaw_rad")
        track_yaw = float(ty) if isinstance(ty, (int, float)) and not isinstance(ty, bool) and math.isfinite(ty) else None
        ay = args.get("yaw")
        # the finger yaw the pick will use (rule F reads it, records only): the program's, else call()'s yaw fill
        grasp_yaw = (float(ay) if isinstance(ay, (int, float)) and not isinstance(ay, bool) else None if track_yaw is None
                     else (track_yaw + math.pi / 2 + math.pi) % (2 * math.pi) - math.pi)
        oc = self._overhead_reading(tid, pos, cat, grasp_yaw)
        if oc is not None and oc.get("refuse") is not False:
            n = oc.get("n_points")  # rule B's count (perception.overhead)
            if oc.get("rule") == "F" and type(oc.get("rule_f_points")) is int:  # rule F refused alone
                n = oc["rule_f_points"]
            why = ("the overhead check could not be read" if oc.get("code") == "overhead_unreadable" else
                   f"{n} depth points over the target in the view that measured its pose")
            return refuse("pick_overhead_obstruction", why, hint=PICK_OVERHEAD_HINT, sigma_xy_m=sg,
                          overhead_points=n if type(n) is int else None, **info)
        return None, {"version": PICK_PRECONDITION_VERSION, "category": cat, **({"declared_kind": base["declared_kind"]}
                      if "declared_kind" in base else {}), "sigma_xy_m": sg, "view_obs_id": view.get("obs_id"), **off,
                      **info, "track_yaw_rad": track_yaw, **top, "overhead": oc}

    def _unit_top_clearance(self, track_id) -> tuple[float | None, bool]:
        """pick-pose-3: (the target centre's clearance in m in front of its drawer's unit top edge, or None when
        it is not in a drawer or the handle's closed position is unknown; readable). The backend's reading
        (FacadeBackend.unit_top_clearance: the scripted chain's own rule, search.SiteSearch._top_clearance); a backend
        without it: (None, True), unchecked. Not counted, not recorded."""
        fn = getattr(self._backend, "unit_top_clearance", None)
        if not callable(fn):
            return None, True
        try:
            v = fn(track_id)
        except Exception:  # noqa: BLE001 (unreadable: refuses only when the rule is active)
            return None, False
        if v is None:
            return None, True
        v = _finite_float(v)
        return v, v is not None

    def _overhead_reading(self, track_id, pos, kind=None, yaw=None) -> dict | None:
        """pick-pose-3: the backend's overhead check of the target track at the commanded pos
        (FacadeBackend.overhead_check: Perception.overhead_check, rule B active, rule F as RULE_F_ACTIVE says) as plain
        values {refuse, code, n_points, rule, rule_f_points (search.overhead_reading), ...}; None when the backend has
        no such check (the toy kitchens: unchecked, declared). Fail-closed: a check that raises or answers without a
        boolean refuse reads as a refusal (code overhead_unreadable). Not counted, not recorded (the pick record keeps
        it in pick_check)."""
        fn = getattr(self._backend, "overhead_check", None)
        if not callable(fn):
            return None
        try:
            r = to_plain(fn(track_id, [float(x) for x in pos], kind=kind, yaw=yaw), self._prog)
        except Exception as e:  # noqa: BLE001 (never a blind pick)
            return {"refuse": True, "code": "overhead_unreadable", "error_type": type(e).__name__[:60]}
        if r is None:
            return None
        if not isinstance(r, dict) or type(r.get("refuse")) is not bool:
            return {"refuse": True, "code": "overhead_unreadable"}
        return {k: v for k, v in r.items() if isinstance(k, str) and (v is None or type(v) in (bool, int, float, str))}

    def _read_tracks(self, cat: str) -> list | None:
        """The backend's tracks of category cat (robot side, not counted, not recorded), or None if unreadable."""
        try:
            tracks = to_plain(self._backend.query("tracks", {"category": cat}), self._prog)
        except Exception:  # noqa: BLE001 (no robot-side reading: nothing proves a perceived target)
            return None
        return tracks if isinstance(tracks, list) else None

    def _policy_category_near(self, pos: list) -> str | None:
        """The first PICK_POSE_POLICY category with a non-held track within its assoc_m of pos, else None
        (an unreadable reading: None, the declared category then stays unchecked)."""
        for c, pol in PICK_POSE_POLICY.items():
            for t in self._read_tracks(c) or []:
                p = finite_vec(t.get("position_m"), 3) if isinstance(t, dict) and t.get("status") != "held" else None
                if p is not None and t.get("category", c) == c and math.dist(p, pos) <= pol["assoc_m"] + BOUND_EPS_M:
                    return c
        return None

    def _view_of(self, t: dict) -> tuple[int, dict | None]:
        """The observation that measured track t (its pose_step), recorded by this host, in which t's detection carries
        pose_source this_view: (index of its observe record, observation), or (-1, None). An observation whose
        detections are not a list is no view (None crashed the check)."""
        step = t.get("pose_step")
        if type(step) is not int:
            return -1, None
        obs = next((o for o in reversed(self.observations()) if isinstance(o, dict) and o.get("step") == step), None)
        if obs is None or not isinstance(obs.get("detections"), list):
            return -1, None
        det = next((x for x in obs["detections"] if isinstance(x, dict)
                    and x.get("track_id") == t.get("track_id")), None)
        if det is None or det.get("pose_source", "this_view") != "this_view":
            return -1, None
        idx = next((i for i, r in enumerate(self.records) if r.get("kind") == "observe" and r.get("status") == "called"
                    and r.get("obs_id") == obs.get("obs_id")), None)
        return (-1, None) if idx is None else (idx, obs)


def finite_vec(v, n: int) -> list | None:
    """v as a list of n finite floats (at least n numbers), or None."""
    if not isinstance(v, (list, tuple)) or len(v) < n:
        return None
    out = []
    for x in v[:n]:
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
            return None
        out.append(float(x))
    return out


def sigma_xy(sigma) -> float | None:
    """max(sigma_x, sigma_y) of a sigma_m vector, or None when missing or not finite (never negative)."""
    v = finite_vec(sigma, 2)
    return None if v is None or min(v) < 0 else max(v)


def is_drawer_region(rid) -> bool:
    return isinstance(rid, str) and rid.startswith("drawer_unit:") and rid[len("drawer_unit:"):].isdigit()


def _drawer_motion_reaches(motion_region, obj_region) -> bool:
    """Can a drawer motion (region motion_region, None = unknown) move an object of obj_region? Not an object on
    a surface (sink, tray, mat); not an object in ANOTHER known drawer; anything else yes (conservative)."""
    if obj_region in SURFACE_REGIONS:
        return False
    if is_drawer_region(obj_region) and is_drawer_region(motion_region) and motion_region != obj_region:
        return False
    return True


def drawer_gesture_moved(ok, reason: str, data: dict) -> bool:
    """Could this open_drawer / close_drawer result have moved its drawer? ok (or an unknown ok) yes,
    except already_open (no motion); a failure only with a reported travel above DRAWER_MOVED_MIN_M (an unreadable
    travel counts) or a reason given after the handle was pinched (DRAWER_PINCHED_REASONS, pull_ / push_ forms). In
    recorded runs two successful picks followed a close_drawer that ended no_grasp_handle (guarded approach, fingers
    closed on nothing, no drawer motion); counting it would have refused both."""
    if ok is not False:
        return not (ok is True and reason == "already_open")
    for k in ("moved", "moved_hand"):
        v = data.get(k)
        if v is None:
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) \
                or abs(float(v)) > DRAWER_MOVED_MIN_M:
            return True
    return reason in DRAWER_PINCHED_REASONS or reason.startswith(("pull_", "push_"))


def pose_moving_records(records: list[dict], region: str | None) -> list[int]:
    """Indexes of the host records (robot side, skills that ran) after which a pose measured earlier of an object in
    `region` is no longer valid for a pick: a drawer gesture that may have moved its drawer (drawer_gesture_moved)
    and can reach the region (a drawer whose handle the robot could not resolve reaches every drawer region), a
    failed pick, a place, a turn_over, an open_gripper while the robot believed it held something, a grip loss
    (GRIP_LOSS_CODES) and any contact incident (is_contact_incident; not in the reviewer's list, added because a
    contact may push what lies near it). A close_gripper that grasped something outside a pick (data.grasped True)
    starts a hold by hand and counts, and so does EVERY gesture while that hold lasts (a drawer pulled by hand with
    move_to, an object carried) until a release or a loss. A skill whose backend raised (BACKEND_RAISED_STATUS)
    counts; so does an internal contact incident (internal_incident, e.g. a collision of the release clearance hidden
    behind the parent's reason), an internal release of a held object (internal_release; it ends the hold) and a
    picked object no longer felt after a gesture (holding_after False on a later record; it ends the hold). A hold by
    hand is never ended by holding_after: the robot's holding() reads False for it (robot.held is set only by pick).
    A raised record is read through its gesture's result when it kept one (skill_result)."""
    out, held = [], None
    for i, r in enumerate(records):
        if r.get("kind") != "skill" or r.get("status") not in RAN_STATUSES:
            continue  # refusals (reuse, pick or calibrate preconditions, budgets) made no motion
        name = r.get("name")
        ok, reason, data = skill_result(r)
        args = r.get("args") if isinstance(r.get("args"), dict) else {}
        moves = r.get("status") == BACKEND_RAISED_STATUS  # it may have moved before raising
        before = held
        if name in ("open_drawer", "close_drawer"):
            moves = moves or (drawer_gesture_moved(ok, reason, data)
                              and _drawer_motion_reaches(r.get("drawer_region"), region))
        elif name == "pick":
            if ok is True:
                held = str(args.get("kind") or "unknown")
            else:
                moves = True
        elif name in ("place", "turn_over"):
            moves = True
            if name == "place" and ok is True:
                held = None
        elif name == "open_gripper" and held is not None:
            moves = True
            if ok is True:
                held = None
        elif name == "close_gripper" and held is None and ok is True and data.get("grasped") is True:
            moves, held = True, MANUAL_HOLD
        elif held == MANUAL_HOLD:
            moves = True
        if reason in GRIP_LOSS_CODES:
            moves, held = True, None
        if ok is False and is_contact_incident(reason, data):
            moves = True
        # Internal actions and the grip reading after the skill
        if internal_incident(r) is not None:
            moves = True
        if held is not None and before is not None and internal_release(r):
            moves, held = True, None
        if held not in (None, MANUAL_HOLD) and held == before and r.get("holding_after") is False:
            moves, held = True, None
        if moves:
            out.append(i)
    return out


# Where the wrist camera looks. Only a successful look_at_region or look_down_at aims it; any other skill that moved the
# arm (data.arm_moved not False) leaves it where that gesture ended; the fingers (open_gripper, close_gripper) and a
# skill that reports arm_moved False do not move it. Robot side (the host's own records).
LOOK_SKILLS = ("look_at_region", "look_down_at")
FINGER_SKILLS = ("open_gripper", "close_gripper")
WRIST_VIEW_NOTE = ('observe("wrist_cam") came after {after} with no look since: the wrist camera looks where that gesture '
                   "left the hand, so it may not show the target; aim it first with look_at_region(region). The "
                   'bowl pose from robot_eye or tracks("bowl") is enough for scrub.')


def unaimed_wrist_views(records: list[dict], start: int = 0) -> list[dict]:
    """The wrist_cam observations of records[start:] taken while the hand was not aimed by a successful look since
    the last arm motion (the state is read from the whole episode's records, the episode starts not aimed): [{obs_id,
    after (the last motion or "the episode start"), record_index}], at most 10. Robot side only."""
    out, aimed, last = [], False, "the episode start"
    for i, r in enumerate(records):
        if r.get("status") != "called":
            continue  # refusals made no motion and no view
        name = r.get("name")
        if r.get("kind") == "skill":
            data = r.get("data") if isinstance(r.get("data"), dict) else {}
            if name in FINGER_SKILLS or data.get("arm_moved") is False:
                continue
            aimed = name in LOOK_SKILLS and r.get("ok") is True
            last = str(name)[:24] if r.get("ok") is not False else f"{str(name)[:24]} ({str(r.get('reason'))[:22]})"
        elif r.get("kind") == "observe" and (r.get("args") or {}).get("camera") == "wrist_cam" and not aimed \
                and i >= start:
            out.append({"obs_id": r.get("obs_id"), "after": last, "record_index": i})
    return out[:10]


def wrist_view_notes(records: list[dict], start: int = 0) -> list[dict]:
    """The observable notes of unaimed_wrist_views for a run report (code, obs_id, message <= 300)."""
    return [{"code": "wrist_view_not_aimed", "obs_id": v["obs_id"],
             "message": WRIST_VIEW_NOTE.format(after=v["after"])[:300]} for v in unaimed_wrist_views(records, start)]


# ====================================================================== SCENE STATE (scene-state-1)
def shared_phrase(name: str) -> str:
    """A model-visible phrase shared with the texts, read from agent.api (which defines
    SCENE_LEGEND_1_0, SCOPE_NOTE_1_0 and RULE_CLEARED there; this module never redefines them). Read at the call,
    so profile 0.9 never needs them; a missing one raises (a build error, never a silent default)."""
    v = getattr(_api, name, None)
    if not isinstance(v, str) or not v:
        raise RuntimeError(f"agent.api.{name} is missing (shared phrase I5, role TEXTS)")
    return v


def scene_unseen_look_refusals(records: list[dict], views: list[dict]) -> dict:
    """{region: reason} of the last look_at_region(region, "unseen") that returned ok False and was recorded after
    the delivery record of that region's last view (the latest delivered observation listing the region at coverage
    SCENE_MIN_COVER or more: the view the site model keeps as the region's last); with no such view, the whole episode.
    Robot side: the host's own records and the observations it delivered. Pure."""
    last_view = {}
    for o in views:
        regs = o.get("regions") if isinstance(o, dict) else None
        for r in regs if isinstance(regs, list) else []:
            cov = r.get("coverage") if isinstance(r, dict) else None
            if isinstance(r, dict) and isinstance(r.get("region"), str) and _finite_float(cov) is not None \
                    and float(cov) >= SCENE_MIN_COVER:
                last_view[r["region"]] = o.get("obs_id")
    delivered_at = {}
    for i, rec in enumerate(records):
        oid = delivered_obs_id(rec)
        if oid is not None and oid not in delivered_at:
            delivered_at[oid] = i
    out = {}
    for i, rec in enumerate(records):
        if rec.get("name") != "look_at_region" or rec.get("status") != "called" or rec.get("ok") is not False:
            continue
        a = rec.get("args") if isinstance(rec.get("args"), dict) else {}
        rid = a.get("region")
        if a.get("part") != "unseen" or not isinstance(rid, str):
            continue
        if i > delivered_at.get(last_view.get(rid), -1):
            out[rid[:SCENE_REGION_MAX]] = _scene_txt(rec.get("reason") or "unknown", SCENE_CODE_MAX)
    return out


def scene_drawer_gestures(records: list[dict]) -> dict:
    """{"regions": the drawer regions an open_drawer or close_drawer record of this episode targets
    (drawer_region, read before the motion), "unresolved": a drawer gesture that reached the backend with no resolved
    region (it may have moved any drawer)}. Robot side (host records). Pure."""
    regions, unresolved = [], False
    for r in records:
        if r.get("name") not in ("open_drawer", "close_drawer"):
            continue
        reg = r.get("drawer_region")
        if is_drawer_region(reg):
            if reg not in regions:
                regions.append(reg)
        elif r.get("status") in RAN_STATUSES:
            unresolved = True
    return {"regions": regions, "unresolved": unresolved}


def scene_latest_view(views: list[dict], category: str) -> dict | None:
    """{"obs_id", "regions"} of the episode's latest delivered view (the last of views, the
    delivered observations in delivery order): the regions, in first-mention order, where that view itself detects
    category (a detection's region) or lists it among that region's own seen_categories (the view's, never the site
    model's merge over views). None when no view was delivered or the last one has no obs id. Robot side (the views the
    host delivered). Pure."""
    if not isinstance(views, list) or not views or not isinstance(views[-1], dict):
        return None
    o = views[-1]
    if not isinstance(o.get("obs_id"), str):
        return None
    regions = []
    dets = o.get("detections")
    for d in dets if isinstance(dets, list) else []:
        if isinstance(d, dict) and d.get("category") == category and isinstance(d.get("region"), str) \
                and d["region"] not in regions:
            regions.append(d["region"])
    regs = o.get("regions")
    for r in regs if isinstance(regs, list) else []:
        if isinstance(r, dict) and isinstance(r.get("region"), str) and isinstance(r.get("seen_categories"), list) \
                and category in r["seen_categories"] and r["region"] not in regions:
            regions.append(r["region"])
    return {"obs_id": o["obs_id"], "regions": regions}


def _scene_txt(v, n: int) -> str:
    """A model-visible fragment: printable ASCII (anything else becomes '?'), whitespace collapsed, at most n chars."""
    s = "".join(c if 32 <= ord(c) < 127 else "?" for c in " ".join(str(v).split()))
    return s[:n]


def _scene_num(v) -> str | None:
    """A number with 2 decimals, rounded half up from its shortest repr (0.215, robot_state's 3-decimal opening, reads
    0.22, never 0.21 as binary rounding of f"{0.215:.2f}" gives); None when not a finite number."""
    f = _finite_float(v)
    if f is None:
        return None
    from decimal import ROUND_HALF_UP, Decimal  # noqa: PLC0415
    q = Decimal(repr(f)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return f"{q + 0:.2f}" if q != 0 else "0.00"


def _scene_scope(scope) -> dict | None:
    """The scope reading as the renderer may use it, or None (unavailable). Well formed: a dict with a non-empty
    outcome, scope rows (dicts with a str region, regions unique), not_cleared a list of str, and the two
    consistent: not_cleared is exactly the rows whose cleared is not True (a disagreement is malformed: nothing is
    then cleared)."""
    if not isinstance(scope, dict):
        return None
    oc, rows, nc = scope.get("outcome"), scope.get("scope"), scope.get("not_cleared")
    if not isinstance(oc, str) or not oc or not isinstance(rows, list) or not isinstance(nc, list):
        return None
    if not all(isinstance(r, dict) and isinstance(r.get("region"), str) for r in rows):
        return None
    regions = [r["region"] for r in rows]
    if len(set(regions)) != len(regions) or not all(isinstance(x, str) for x in nc) or len(set(nc)) != len(nc):
        return None
    if set(nc) != {r["region"] for r in rows if r.get("cleared") is not True}:
        return None
    return scope


def _scene_reason(row: dict | None, det: dict | None, has_detail: bool) -> tuple[str, str | None]:
    """(reason code of SCENE_REASONS, its arguments or None) of a not-cleared row, in SCENE_REASONS precedence:
    step None and no view never_viewed; step None with views low_coverage (best x.xx, n view(s)) (a view at 0
    coverage is still a view); no detail (toy backend) and no view at SCENE_MIN_COVER low_coverage (no view reached
    0.3); evidence_invalidated (<action>[ <reason>] after <obs>); could_hide None no_floor_grid (coverage, obs);
    frame_accepted False module_pose_not_measured (obs); could_hide True unseen_part_could_hide (coverage, obs); else
    reason_unknown (also a region not cleared without a scope row)."""
    if not isinstance(row, dict):
        return "reason_unknown", None
    det = det if isinstance(det, dict) and "error" not in det else None
    obs = _scene_txt(det["last_obs"], SCENE_ID_MAX) if det and isinstance(det.get("last_obs"), str) else None
    cov = _scene_num(row.get("coverage"))
    if row.get("step") is None:
        if det is not None:
            views = det.get("views")
            views = views if type(views) is int and views >= 0 else 0
            if views == 0:
                return "never_viewed", None
            best = _scene_num(det.get("best_low_coverage"))
            return "low_coverage", (f"best {best}, " if best is not None else "") + f"{views} view(s)"
        f = _finite_float(row.get("coverage"))
        if f is None or f < SCENE_MIN_COVER or has_detail or "step" in row:
            return "low_coverage", f"no view reached {SCENE_MIN_COVER}"
        # a toy row viewed at SCENE_MIN_COVER or more (no step, no detail): read on below
    if row.get("evidence_invalidated_by_later_action") is True:
        ib = det.get("invalidated_by") if det else None
        parts = []
        if isinstance(ib, dict):
            parts.append(_scene_txt(ib.get("action") or "action", SCENE_CODE_MAX))
            if isinstance(ib.get("reason"), str) and ib["reason"]:
                parts.append(_scene_txt(ib["reason"], SCENE_CODE_MAX))
        else:
            parts.append("a later action")
        return "evidence_invalidated", " ".join(parts) + (f" after {obs}" if obs else "")
    hide = row.get("unseen_part_could_hide_it")
    cov_obs = (f"coverage {cov}" if cov is not None else "") + (", " if cov is not None and obs else "") + (obs or "")
    if hide is None:
        return "no_floor_grid", cov_obs or None
    if det is not None and det.get("frame_accepted") is False:
        return "module_pose_not_measured", obs
    if hide is True:
        return "unseen_part_could_hide", cov_obs or None
    return "reason_unknown", None


def _scene_places(row: dict, det: dict | None, region: str) -> list[str]:
    """The open-place classes of a not-cleared row whose reason is module_pose_not_measured or unseen_part_could_hide,
    in SCENE_PLACES order: a surface whose outline was not fitted: outline not fitted; else the classes the detail gives
    for its open placements (drawer: under the unit top / near the front wall / in the open part; surface: on the
    surface); else grid points only (could_hide True with no open placement, or no detail)."""
    det = det if isinstance(det, dict) and "error" not in det else None
    if det is not None and det.get("frame_accepted") is False:
        return ["outline not fitted"]
    got = det.get("places") if det is not None else None
    cls = [c for c in SCENE_PLACES if isinstance(got, list) and c in got]
    if cls:
        return cls
    return ["grid points only"] if row.get("unseen_part_could_hide_it") is True else []


def _scene_drawer_tag(region: str, st: dict, gestures: dict, open_full: bool) -> str:
    """The drawer tag (SCENE_DRAWER_TAGS) of a drawer region, from robot_state and the host's records only: believed
    open (drawers_believed_open, opening known) "open X.XX m", with " (end stop)" under K1 (open_full) when less than
    OPEN_LEFT_MIN_M of DRAWER_TRAVEL_M is left (the already_open rule); opening unknown (also when the drawer state
    cannot be read); otherwise "not moved by the robot" when no drawer gesture of this episode targeted it, else "not
    believed open" (also after a drawer gesture whose region the robot could not resolve)."""
    lst = st.get("drawers_believed_open") if isinstance(st, dict) else None
    if not isinstance(lst, list):
        return "opening unknown"
    e = next((d for d in lst if isinstance(d, dict) and d.get("region") == region), None)
    if e is None and any(not isinstance(d, dict) for d in lst):
        return "opening unknown"
    if e is not None:
        op = _finite_float(e.get("believed_opening_m"))
        if e.get("opening_unknown") is not False or op is None:
            return "opening unknown"
        end = bool(open_full) and DRAWER_TRAVEL_M - op < OPEN_LEFT_MIN_M
        return f"open {_scene_num(op)} m" + (" (end stop)" if end else "")
    g = gestures if isinstance(gestures, dict) else {"regions": [], "unresolved": True}
    if region in (g.get("regions") or []) or g.get("unresolved") is not False:
        return "not believed open"
    return "not moved by the robot"


def _scene_track_item(t) -> tuple[bool, str]:
    """(observed, text) of a found track of the scope: {track_id, status, region} (the toy gives bare ids)."""
    if not isinstance(t, dict):
        return False, _scene_txt(t, SCENE_ID_MAX)
    tid = _scene_txt(t.get("track_id"), SCENE_ID_MAX)
    reg = _scene_txt(t.get("region") or "region unknown", SCENE_REGION_MAX)
    s = t.get("status")
    if s == "observed":
        return True, f"{tid} observed in {reg}"
    if s == "held":
        return False, f"{tid} held, picked from {reg}"
    if s == "stale":
        return False, f"{tid} seen in {reg}, {SCENE_ACTED_SINCE}"
    return False, f"{tid} in {reg}"


def _scene_object_item(o: dict) -> str:
    cat = _scene_txt(o.get("category"), SCENE_CODE_MAX)
    tid = _scene_txt(o.get("track_id"), SCENE_ID_MAX)
    if o.get("status") == "held":
        return f"{cat} {tid} held"
    reg = _scene_txt(o.get("region") or "region unknown", SCENE_REGION_MAX)
    obs = f" {_scene_txt(o['last_obs'], SCENE_ID_MAX)}" if isinstance(o.get("last_obs"), str) else ""
    return f"{cat} {tid} {reg}{obs}" + (f" ({SCENE_ACTED_SINCE})" if o.get("status") == "stale" else "")


def render_scene_state(readings: dict, robot_state: dict, open_full: bool) -> dict:
    """{"text", "record"} of SCENE STATE scene-state-1, a pure function of the
    robot-side readings (GuardedHost.scene_readings), the robot state and open_full (open_full_active() at reading
    time).
    Text lines, in this order (absent optional lines are left out):
      1. header "SCENE STATE NOW (robot side, read when this request was built, after obs-NNN; scene-state-1):";
      2. the legend agent.api.SCENE_LEGEND_1_0;
      3. "hand: holding yes|no|unknown; contact incidents k; grip events k";
      4. "search for sponge: <outcome verbatim, 40>" ("reading unavailable (nothing is cleared)" when the scope reading
         is None or malformed: then no region line at all);
      5. "sponge tracks: ..." (found: at most 3, observed first), and SCENE_HELD_EMPTY when a found track is held while
         the grip reads empty;
      6. held collapse (a found track held AND robot_state holding True): "sponge: held (trk, picked from r); region
         lines omitted" and the drawers believed open or of unknown opening; otherwise "not cleared k of n:" and one row
         per not-cleared region in the scope's order "- <region>: <reason> (<args>)[; open places: <classes>][; look at
         its unseen part since: <reason>][; drawer <tag>]" (open places on module_pose_not_measured and
         unseen_part_could_hide rows; the nothing_unseen refusal only on evidence_invalidated rows, any other refusal
         on any row);
      6b. Only when some region is concerned: "<category> seen here k of n: <region>[ (drawer
         <tag>)], ..." (at most SCENE_SEEN_MAX names, then "k more"; never shed; no obs id): the scope's cleared regions
         where the robot sees the sponge, i.e. a found track with status "observed" in that region, or the region's
         scope row reading seen True (its own last view lists the category). They are left out of line 7 and of its
         count, never printed as cleared. Since (SCENE_SEEN_RULE seen-fresh-1) line 6b holds only the fresh ones:
         readings["latest_view"] (scene_latest_view) names the header's view and that view itself shows the category in
         the region, and no found track in it is stale or held; the others are on
      6c. "<category> seen earlier, fresh location not guaranteed, k of n: <region>[ (drawer <tag>)], ..." (never
         shed, no obs id); lines 6b and 6c show at most SCENE_SEEN_MAX names in all, the fresh first, each then
         "k more";
      7. "cleared k of n (<RULE_CLEARED>): <region> <obs>[ (drawer <tag>)], ..." (the other cleared regions);
      8. "outside the scope, never inspected: ..." (at most 4, "k more");
      9. "objects seen (latest first): ..." (at most 8, "k more").
    No position, no stale word, numbers with 2 decimals, printable ASCII. Past SCENE_MAX_CHARS the optional parts are
    shed in SCENE_SHED_ORDER (the record lists what was). Record: version, profile, built_after_obs, outcome,
    not_cleared, cleared (None when unavailable; the scope's cleared rows as read, seen-here regions included, as the
    guard reads them), seen_here (20.1: those of them printed on a seen line, never as cleared: since 21.5 lines 6b
    and 6c together; None when unavailable, also recorded when the text collapses), seen_fresh and seen_earlier
    (21.5: the part of seen_here on line 6b, and on line 6c; None when unavailable), seen_rule (SCENE_SEEN_RULE),
    reasons, places, tags, unseen_look_refusals, held, grip_empty_with_held_track, shed, unavailable, edge_unknown
    (drawer regions whose unit-top edge is unknown)."""
    rd = readings if isinstance(readings, dict) else {}
    st = robot_state if isinstance(robot_state, dict) else {}
    legend, gloss = shared_phrase("SCENE_LEGEND_1_0"), shared_phrase("RULE_CLEARED")
    cat = _scene_txt(rd.get("category") or "sponge", SCENE_CODE_MAX)
    last_obs = _scene_txt(rd["last_obs_id"], SCENE_ID_MAX) if isinstance(rd.get("last_obs_id"), str) else None
    header = (f"{SCENE_HEADER_PREFIX} (robot side, read when this request was built, "
              f"{'after ' + last_obs if last_obs else 'before any view'}; {SCENE_STATE_VERSION}):")
    hold = st.get("holding")
    n_inc = len(st["contact_incidents"]) if isinstance(st.get("contact_incidents"), list) else None
    n_grip = len(st["grip_events"]) if isinstance(st.get("grip_events"), list) else None
    hand = (f"hand: holding {'yes' if hold is True else 'no' if hold is False else 'unknown'}; contact incidents "
            f"{'unknown' if n_inc is None else n_inc}; grip events {'unknown' if n_grip is None else n_grip}")
    looks = rd.get("unseen_look_refusals") if isinstance(rd.get("unseen_look_refusals"), dict) else {}
    looks = {_scene_txt(k, SCENE_REGION_MAX): _scene_txt(v, SCENE_CODE_MAX) for k, v in looks.items()
             if isinstance(k, str)}
    rec = {"version": SCENE_STATE_VERSION, "profile": rd.get("profile"), "built_after_obs": last_obs, "outcome": None,
           "not_cleared": None, "cleared": None, "seen_here": None, "seen_fresh": None, "seen_earlier": None,
           "seen_rule": SCENE_SEEN_RULE, "reasons": {}, "places": {}, "tags": {}, "unseen_look_refusals": looks,
           "held": False, "grip_empty_with_held_track": False, "shed": [], "unavailable": False, "edge_unknown": []}
    objs = [o for o in rd.get("objects") or [] if isinstance(o, dict)] if isinstance(rd.get("objects"), list) else []
    obj_items = [_scene_object_item(o) for o in objs]
    objects_line = None
    if obj_items:
        more = len(obj_items) - SCENE_OBJECTS_MAX
        objects_line = ("objects seen (latest first): " + " | ".join(obj_items[:SCENE_OBJECTS_MAX])
                        + (f" | {more} more" if more > 0 else ""))
    scope = _scene_scope(rd.get("scope"))
    if scope is None:
        rec["unavailable"] = True
        lines = [header, legend, hand, f"search for {cat}: {SCENE_UNAVAILABLE}"]
        return _scene_finish(lines, [], None, None, objects_line, rec)
    outcome = _scene_txt(scope["outcome"], SCENE_OUTCOME_MAX)
    rec["outcome"] = outcome
    rows = scope["scope"]
    by_region = {r["region"]: r for r in rows}
    nc = list(scope["not_cleared"])
    nc_set = set(nc)
    cleared = [r["region"] for r in rows if r["region"] not in nc_set]
    detail = rd.get("detail") if isinstance(rd.get("detail"), dict) else None
    dregs = detail.get("regions") if detail is not None and isinstance(detail.get("regions"), dict) else None
    has_detail = dregs is not None
    gestures = rd.get("drawer_gestures")
    tags = {r: _scene_drawer_tag(r, st, gestures, open_full) for r in [x["region"] for x in rows]
            if is_drawer_region(r)}
    rec["not_cleared"] = [_scene_txt(r, SCENE_REGION_MAX) for r in nc]
    rec["cleared"] = [_scene_txt(r, SCENE_REGION_MAX) for r in cleared]
    rec["tags"] = {_scene_txt(r, SCENE_REGION_MAX): t for r, t in tags.items()}
    found = scope.get("found_tracks") if isinstance(scope.get("found_tracks"), list) else []
    # A cleared region where the robot sees the sponge (an observed found track in it, or its scope row's seen True) is
    # printed "<cat> seen here", never as cleared (line 6b); the guard's reading is unchanged
    seen_regions = {t.get("region") for t in found if isinstance(t, dict) and t.get("status") == "observed"}
    seen_here = [r for r in cleared if r in seen_regions or by_region[r].get("seen") is True]
    rec["seen_here"] = [_scene_txt(r, SCENE_REGION_MAX) for r in seen_here]
    # seen-fresh-1: "seen here" only for a sighting in the latest delivered view (the header's), with no found track in
    # that region stale or held (the robot acted since); every other one reads "seen earlier"
    lv = rd.get("latest_view")
    latest = (set(x for x in lv["regions"] if isinstance(x, str))
              if isinstance(lv, dict) and last_obs is not None and lv.get("obs_id") == rd.get("last_obs_id")
              and isinstance(lv.get("regions"), list) else set())
    acted = {t.get("region") for t in found if isinstance(t, dict) and t.get("status") in ("stale", "held")}
    seen_fresh = [r for r in seen_here if r in latest and r not in acted]
    seen_earlier = [r for r in seen_here if r not in seen_fresh]
    rec["seen_fresh"] = [_scene_txt(r, SCENE_REGION_MAX) for r in seen_fresh]
    rec["seen_earlier"] = [_scene_txt(r, SCENE_REGION_MAX) for r in seen_earlier]
    held_tracks = [t for t in found if isinstance(t, dict) and t.get("status") == "held"]
    rec["held"] = bool(held_tracks) and hold is True
    rec["grip_empty_with_held_track"] = bool(held_tracks) and hold is False
    lines = [header, legend, hand, f"search for {cat}: {outcome}"]
    if rec["held"]:
        t = held_tracks[0]
        lines.append(f"{cat}: held ({_scene_txt(t.get('track_id'), SCENE_ID_MAX)}, picked from "
                     f"{_scene_txt(t.get('region') or 'region unknown', SCENE_REGION_MAX)}); region lines omitted")
        lst = st.get("drawers_believed_open")
        if isinstance(lst, list):
            dl = [f"{_scene_txt(d.get('region'), SCENE_REGION_MAX)} "
                  f"({_scene_drawer_tag(d.get('region'), st, gestures, open_full)})"
                  for d in lst if isinstance(d, dict)]
            lines.append("drawers believed open or of unknown opening: " + (", ".join(dl) if dl else "none"))
        else:
            lines.append("drawers believed open or of unknown opening: unknown (drawer state unreadable)")
    else:
        if found:
            items = [_scene_track_item(t) for t in found]
            items = [x for ok, x in items if ok] + [x for ok, x in items if not ok]
            lines.append(f"{cat} tracks: " + " | ".join(items[:SCENE_TRACKS_MAX]))
        if rec["grip_empty_with_held_track"]:
            lines.append(SCENE_HELD_EMPTY)
    # reasons, places and edges of every not-cleared region (recorded also when the text collapses)
    nc_rows = []
    for r in nc:
        row = by_region.get(r)
        det = dregs.get(r) if has_detail else None
        code, args = _scene_reason(row, det, has_detail)
        name = _scene_txt(r, SCENE_REGION_MAX)
        rec["reasons"][name] = code
        txt = f"- {name}: {code}" + (f" ({args})" if args else "")
        if code in ("module_pose_not_measured", "unseen_part_could_hide"):
            pl = _scene_places(row, det, r)
            rec["places"][name] = pl
            if pl:
                txt += "; open places: " + ", ".join(pl)
        lk = looks.get(name)
        if lk and (lk != "nothing_unseen" or code == "evidence_invalidated"):
            txt += f"; look at its unseen part since: {lk}"
        if r in tags:
            txt += f"; drawer {tags[r]}"
        nc_rows.append(txt)
    if has_detail:
        rec["edge_unknown"] = [_scene_txt(r, SCENE_REGION_MAX) for r in nc + cleared if is_drawer_region(r)
                               and isinstance(dregs.get(r), dict) and dregs[r].get("edge_unknown") is True]
    if rec["held"]:
        return _scene_finish(lines, [], None, None, objects_line, rec)
    lines.append(f"not cleared {len(nc)} of {len(rows)}:" + ("" if nc else " none"))
    lines += nc_rows[:SCENE_ROWS_MAX]
    if len(nc_rows) > SCENE_ROWS_MAX:
        lines.append(f"- and {len(nc_rows) - SCENE_ROWS_MAX} more not cleared")
    left = SCENE_SEEN_MAX  # lines 6b (20.1) and 6c (21.5): never shed, no obs id, at most SCENE_SEEN_MAX names in all
    for label, regs in ((SCENE_SEEN_HERE, seen_fresh), (SCENE_SEEN_EARLIER + ",", seen_earlier)):
        if not regs:
            continue
        items = [_scene_txt(r, SCENE_REGION_MAX) + (f" (drawer {tags[r]})" if r in tags else "") for r in regs]
        shown = items[:left]
        left -= len(shown)
        more = len(items) - len(shown)
        lines.append(f"{cat} {label} {len(regs)} of {len(rows)}: "
                     + ", ".join(shown + ([f"{more} more"] if more > 0 else [])))
    shown_cleared = [r for r in cleared if r not in seen_here]
    cl_items = []
    for r in shown_cleared:
        det = dregs.get(r) if has_detail and isinstance(dregs.get(r), dict) else {}
        obs = det.get("last_obs") if isinstance(det.get("last_obs"), str) else None
        cl_items.append((_scene_txt(r, SCENE_REGION_MAX), None if obs is None else _scene_txt(obs, SCENE_ID_MAX),
                         tags.get(r)))
    cleared_head = f"cleared {len(shown_cleared)} of {len(rows)} ({gloss}):"
    out = scope.get("not_inspected_outside_scope")
    outside = ([_scene_txt(x, SCENE_OUTSIDE_ITEM_MAX) for x in out if isinstance(x, str)] if isinstance(out, list)
               else [])
    outside_line = None
    if outside:
        more = len(outside) - SCENE_OUTSIDE_MAX
        outside_line = ("outside the scope, never inspected: " + ", ".join(outside[:SCENE_OUTSIDE_MAX])
                        + (f", {more} more" if more > 0 else ""))
    return _scene_finish(lines, cl_items, cleared_head, outside_line, objects_line, rec)


def _scene_finish(lines: list[str], cl_items: list, cleared_head: str | None, outside_line: str | None,
                  objects_line: str | None, rec: dict) -> dict:
    """Join the lines (the never-shed part, then the cleared row, the outside list and the objects), shedding in
    SCENE_SHED_ORDER while the text is longer than SCENE_MAX_CHARS. Records what was shed."""
    def cleared_line(obs: bool, names: bool) -> str | None:
        if cleared_head is None:
            return None
        if not cl_items:
            return cleared_head + " none"
        if not names:
            return cleared_head + " the other regions of the scope (names shed)"
        parts = []
        for name, oid, tag in cl_items:
            parts.append(name + (f" {oid}" if obs and oid else "") + (f" (drawer {tag})" if tag else ""))
        return cleared_head + " " + ", ".join(parts)

    def build(shed: list[str]) -> str:
        cl = cleared_line("cleared_obs" not in shed, "cleared_names" not in shed)
        extra = [cl, None if "outside" in shed else outside_line, None if "objects" in shed else objects_line]
        return "\n".join(lines + [x for x in extra if x])

    shed: list[str] = []
    text = build(shed)
    for step in SCENE_SHED_ORDER:
        if len(text) <= SCENE_MAX_CHARS:
            break
        present = {"objects": objects_line is not None, "outside": outside_line is not None,
                   "cleared_obs": any(oid for _n, oid, _t in cl_items), "cleared_names": bool(cl_items)}[step]
        if present:
            shed.append(step)
            text = build(shed)
    rec["shed"] = shed
    return {"text": text, "record": rec}


# Contact incidents (prompt 0.2): an EXPLICIT list of robot-side reason codes, the same list the prompt's
# TERMINALS rule names. Before, any failed skill carrying force_peak counted (a weak_contact scrub, a press with no
# contact at 0 N), which told the model to ask for help after a recovered low-force event. The verifier's thresholds are
# not used here: these are the robot's own reflex and motion-stop reasons.
INCIDENT_REASONS = ("collision", "blocked", "stuck", "force_limit", "collision_on_approach")
INCIDENT_PREFIXES = ("", "pull_", "push_", "turn_over_", "turn_over_pick_")
INCIDENT_CODES = frozenset(p + r for p in INCIDENT_PREFIXES for r in INCIDENT_REASONS)


def is_contact_incident(reason: str, data: dict) -> bool:
    """A failed skill's reason is a contact incident: one of INCIDENT_CODES, or lift_blocked whose lift stopped
    on a collision or a blocked motion (data.lift)."""
    if reason in INCIDENT_CODES:
        return True
    return reason == "lift_blocked" and str(data.get("lift") or "") in INCIDENT_REASONS


def contact_incidents(records: list[dict]) -> list[dict]:
    """Skill calls of this episode that stopped on a contact (is_contact_incident; robot side, the skill's own
    result). force_peak_n (or the reflex's force_n) is attached when the result carries it. At most one entry per
    record. The parent's own incident as before; else the first INTERNAL entry of the record (internal_incident: the
    release clearance's move_to, a composite look's pose) that stopped on a contact, whatever the parent's result (a
    clearance collision behind hand_busy or stuck, a look that ended ok after a blocked pose): {call: the parent
    skill, reason: the internal reason, record_index, force_peak_n}, the existing entry shape (the internal origin
    stays in the host record). A skill whose backend raised after its gesture returned is read through that gesture's
    result skill_result: a go_home collision followed by an exception is listed like the same collision without one."""
    out = []
    for i, r in enumerate(records):
        if r.get("kind") != "skill":
            continue
        ok, reason, data = skill_result(r)
        if ok is False and is_contact_incident(reason, data):
            e = {"call": r.get("name"), "reason": reason[:60], "record_index": i}
            fp = data.get("force_peak", data.get("force_n"))
        else:
            ie = internal_incident(r)
            if ie is None:
                continue
            e = {"call": r.get("name"), "reason": str(ie.get("reason") or "")[:60], "record_index": i}
            fp = ie.get("force_n")
        if isinstance(fp, (int, float)) and not isinstance(fp, bool) and math.isfinite(fp):
            e["force_peak_n"] = round(float(fp), 1)
        out.append(e)
    return out[-20:]


def internal_entry(skill, ok, reason, data, t_s, origin) -> dict:
    """internal-actions-1: one internal facade call, plain values only: {skill, ok, reason, arm_moved, force_n,
    t_s, origin}. data: the call's robot-side data (programs.facade log, or a SkillResult's data); no "name" key (a
    prompt FORBIDDEN_KEYS key), no free text."""
    data = data if isinstance(data, dict) else {}

    def num(v, nd):
        ok_ = isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))
        return round(float(v), nd) if ok_ else None
    am = data.get("arm_moved")
    return {"skill": str(skill)[:24], "ok": ok if type(ok) is bool else None, "reason": str(reason or "")[:60],
            "arm_moved": am if type(am) is bool else None, "force_n": num(data.get("force_n", data.get("force_peak")), 1),
            "t_s": num(t_s, 2), "origin": str(origin)[:24] if origin is not None else None}


def gesture_result(g) -> dict:
    """The plain result {ok, reason, data} of a gesture whose backend raised
    after the gesture's own facade call returned (FacadeBackend._account, GuardedHost._record_raised): ok a bool or
    None, the reason code (60 characters), data the robot-side values of _SKILL_DATA_KEYS only (never the free-text
    error). Records only: the key "gesture" of a BACKEND_RAISED_STATUS record, which prompt.run_report does not copy."""
    g = g if isinstance(g, dict) else {}
    ok = g.get("ok")
    ok = bool(ok) if type(ok).__name__ in ("bool", "bool_") else None  # a numpy bool from a SkillResult
    raw = g.get("data") if isinstance(g.get("data"), dict) else {}
    data = {}
    for k in _SKILL_DATA_KEYS:
        v = raw.get(k)
        if type(v).__name__ in ("bool", "bool_"):
            data[k] = bool(v)
        elif isinstance(v, int):
            data[k] = int(v)
        elif isinstance(v, float) and math.isfinite(v):
            data[k] = float(v)
        elif isinstance(v, str):
            data[k] = v[:60]
    return {"ok": ok, "reason": str(g.get("reason") or "")[:60], "data": data}


def skill_result(r: dict) -> tuple:
    """(ok, reason, data) of a skill record as the robot-side readers see it
    (pose_moving_records, contact_incidents, grip_events, open_hold, scrub_reached_contact, untracked_drawer_open): the
    record's own result, except for a BACKEND_RAISED_STATUS record that kept the result of its gesture (key "gesture":
    the facade had returned it before the backend raised in its own bookkeeping): that gesture's result, so a collision,
    a grip loss or a contact scrub behind the exception enters the existing lists like any other. data is a dict."""
    src = r.get("gesture") if r.get("status") == BACKEND_RAISED_STATUS and isinstance(r.get("gesture"), dict) else r
    data = src.get("data") if isinstance(src.get("data"), dict) else {}
    return src.get("ok"), str(src.get("reason") or ""), data


def internal_entries(r: dict) -> list[dict]:
    """The internal entries of a host record (none for older records or for backends without a call detail)."""
    v = r.get("internal")
    return [e for e in v if isinstance(e, dict)] if isinstance(v, list) else []


def internal_incident(r: dict) -> dict | None:
    """The first internal entry of record r that stopped on a contact (is_contact_incident), else None."""
    return next((e for e in internal_entries(r) if e.get("ok") is False
                 and is_contact_incident(str(e.get("reason") or ""), {})), None)


def internal_release(r: dict) -> bool:
    """Did record r's skill open the fingers or place by itself (an internal open_gripper or place that returned
    ok, e.g. the drawer release clearance)?"""
    return any(e.get("skill") in RELEASE_SKILLS and e.get("ok") is True for e in internal_entries(r))


# Grip events. In a recorded episode scrub returned "dropped", a later pick and scrub returned ok and the program
# claimed done; the evaluator rejected no_drop, and the library would have stored that program as valid in the
# robot_side condition, because the harness rule above and the library only looked at contact incidents. A grip event is
# either
# - a grip loss the robot's own skill reported: reason dropped or slipped (also turn_over_dropped and the
#   turn_over_pick_ forms), read from the robot's grip switches and gripper width (robot.skills); never a
#   verifier output;
# - an intentional release of the sponge after it was picked: place or open_gripper that returned ok while the
#   robot believed it held the sponge (its own successful pick of kind "sponge"). The public task asks to pick the
#   sponge and scrub with it: the hold must last until done().
# Not grip events: not_holding / not_holding_sponge before any grasp (nothing was held), a failed release (the
# object may still be in the hand), releasing another object. A later ok gesture never erases a grip event.
# Both kinds void done() for the rest of the episode (agent.loop.refuse_done_after_incident) and refuse storing a
# program from the episode in both memory conditions (agent.memory.ProgramLibrary.store).
# Limit (declared): the finger switches cannot tell the sponge from a dish rim, so a loss they do not feel is not
# listed (the verifier may still see it); this list does not promise that every drop is detected.
GRIP_LOSS_CODES = frozenset({"dropped", "slipped", "turn_over_dropped", "turn_over_pick_dropped",
                             "turn_over_pick_slipped"})
RELEASE_SKILLS = ("place", "open_gripper")


def grip_events(records: list[dict]) -> list[dict]:
    """Grip events of this episode from the host's own records (robot side): see GRIP_LOSS_CODES above.
    At most one event per record, in this order: (1) a grip loss the skill reported; (2) a pick that returned ok sets
    what is held; (3) a release by the program (unchanged); (4) the sponge held and an INTERNAL open_gripper or place
    that returned ok (internal_release: the drawer clearance of 41008 D): event "internal_release", reason
    "released_by_robot_clearance"; (5) the sponge held and the grip not felt after a later gesture (holding_after
    False: close_gripper wide open on it, a loss while carrying during a look or go_home, a scrub that ended on
    force_limit, a reflex recovery that cleared the hold): event "grip_loss", reason "grip_not_felt_after_gesture".
    holding_after None (unreadable) is unknown, never a loss. Skills whose backend raised are read too (through their
    gesture's result when they kept one, skill_result). Entry keys unchanged."""
    out, held = [], None  # held: the kind the robot believes it holds, from its own successful pick
    for i, r in enumerate(records):
        if r.get("kind") != "skill" or r.get("status") not in RAN_STATUSES:
            continue
        name = r.get("name")
        ok, reason, _ = skill_result(r)
        args = r.get("args") if isinstance(r.get("args"), dict) else {}
        if ok is False and reason in GRIP_LOSS_CODES:
            out.append({"call": name, "reason": reason[:60], "event": "grip_loss", "object": held or "unknown",
                        "record_index": i})
            held = None
        elif name == "pick" and ok is True:
            held = str(args.get("kind") or "unknown")
        elif name in RELEASE_SKILLS and ok is True and held is not None:
            if held == "sponge":
                out.append({"call": name, "reason": "released_by_program", "event": "intentional_release",
                            "object": held, "record_index": i})
            held = None
        elif held is not None and internal_release(r):
            if held == "sponge":
                out.append({"call": name, "reason": "released_by_robot_clearance", "event": "internal_release",
                            "object": held, "record_index": i})
            held = None
        elif held is not None and r.get("holding_after") is False:
            if held == "sponge":
                out.append({"call": name, "reason": "grip_not_felt_after_gesture", "event": "grip_loss",
                            "object": held, "record_index": i})
            held = None
    return out[-20:]


def open_hold(records: list[dict]) -> str | None:
    """What the host's own records say the hand still holds after the last record: the kind of
    an ok pick, or MANUAL_HOLD (close_gripper grasped something outside a pick), with no release (place or
    open_gripper that returned ok, an internal release) and no loss (a grip loss code; holding_after False after a
    later gesture, for a picked object only: holding() reads False for a hold by hand) since; else None. Robot side.
    A raised record is read through its gesture's result when it kept one (skill_result)."""
    held = None
    for r in records:
        if r.get("kind") != "skill" or r.get("status") not in RAN_STATUSES:
            continue
        name = r.get("name")
        ok, reason, data = skill_result(r)
        args = r.get("args") if isinstance(r.get("args"), dict) else {}
        before = held
        if ok is False and reason in GRIP_LOSS_CODES:
            held = None
        elif name == "pick" and ok is True:
            held = str(args.get("kind") or "unknown")
        elif name in RELEASE_SKILLS and ok is True:
            held = None
        elif name == "close_gripper" and held is None and ok is True and data.get("grasped") is True:
            held = MANUAL_HOLD
        if held is not None and before is not None and internal_release(r):
            held = None
        if held not in (None, MANUAL_HOLD) and held == before and r.get("holding_after") is False:
            held = None
    return held


def drawer_gesture_before_pinch(ok, reason: str, data: dict) -> bool:
    """Did this failed open_drawer / close_drawer end before its fingers pinched the
    rail, so that its drawer cannot have moved? robot.skills returns these reasons only from the approach of the
    first stroke or before any motion (hand_busy, bad_arg, budget_exceeded; _pinch_rail: no_grasp_handle,
    collision_on_approach, unreachable, no_path; a later stroke's failed pinch ends as partial). No reported travel
    above DRAWER_MOVED_MIN_M either. A collision, blocked or internal_error abort can come during the stroke: never
    exempt."""
    if ok is not False or reason not in DRAWER_BEFORE_PINCH_REASONS:
        return False
    for k in ("moved", "moved_hand"):
        v = (data or {}).get(k)
        if v is None:
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) \
                or abs(float(v)) > DRAWER_MOVED_MIN_M:
            return False
    return True


def close_push_opposes_pull(push_dir, pull, cosine: bool = False):
    """Does a close_drawer's push_dir oppose the tracked handle's pull direction (horizontal
    cosine at most CLOSE_PUSH_OPPOSED_COS)? With cosine=True, the cosine itself (0.0 when unreadable). robot.skills
    pushes along push_dir whatever the drawer's axis, so a push along the pull pulls the drawer out."""
    try:
        p = [float(push_dir[0]), float(push_dir[1])]
        q = [float(pull[0]), float(pull[1])]
        n = math.hypot(*p) * math.hypot(*q)
        c = (p[0] * q[0] + p[1] * q[1]) / n if n > 1e-9 and math.isfinite(n) else None
    except (TypeError, ValueError, IndexError):
        c = None
    if cosine:
        return 0.0 if c is None or not math.isfinite(c) else c
    return c is not None and math.isfinite(c) and c <= CLOSE_PUSH_OPPOSED_COS


def _finite_float(v) -> float | None:
    """v as a finite float (a Python or numpy number), else None (None, a bool, a string, NaN, inf)."""
    if v is None or isinstance(v, (bool, str, bytes)) or type(v).__name__ == "bool_":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def close_travel_check(believed_opening, closed_pos, pull, args: dict, data: dict,
                       opening_unknown_before: bool = False) -> dict:
    """drawer-booking-2: did an ok close_drawer whose
    push_dir opposes the tracked pull travel far enough to have closed the drawer? robot.skills returns ok on the first
    resistance of its 25 N guard whatever the travel when the program passes no expected (skills.close_drawer), so an
    obstruction mid-travel reads as the end stop (6.69 cm after a 22 cm opening, booked closed, calibrate then hit the
    drawer at 309 N reported by the robot). Pure, robot-side quantities only, never the simulator's joint and never the
    program's expected:
    - believed: the perception's believed opening of the tracked handle (perception.opening; a lower bound when its
      opening is unknown);
    - pinch: the opening where the rail was pinched, (handle_pos - closed_pos) projected on the unit pull (an ok pinch
      holds the rail between fingers opened to 4.5 cm, about +-1.5 cm); None when unreadable;
    - need = max(believed, pinch);
    - travel = data.moved x -cos(push_dir, pull) (the hand's travel along the closing axis); None when moved is
      missing, not finite or negative;
    - sufficient: travel >= need - CLOSE_TRAVEL_TOL_M (BOUND_EPS_M absorbs float rounding); an unreadable travel is
      insufficient; residual = max(0, need - travel), a lower bound of what is left open (need when travel is None).
    Returns the records-only close_check {version, believed_opening_m, pinch_opening_m, need_m, travel_m, tol_m,
    sufficient, residual_m, opening_unknown_before} (floats rounded to 0.1 mm), plus residual_raw_m, the unrounded
    residual the booking uses (FacadeBackend removes it before the record). A heuristic, not a proof of closing: a stop
    that leaves less than about 2.6 cm out is still booked closed (declared; a later whole view measures it,
    perception.estimate drawer-reference-2), and a hand travel is not always a drawer travel."""
    believed = _finite_float(believed_opening)
    believed = 0.0 if believed is None else believed
    pinch = None
    try:
        q = [float(pull[0]), float(pull[1])]
        n = math.hypot(*q)
        hp, cp = args.get("handle_pos"), closed_pos
        if n > 1e-9 and math.isfinite(n) and hp is not None and cp is not None:
            v = ((float(hp[0]) - float(cp[0])) * q[0] + (float(hp[1]) - float(cp[1])) * q[1]) / n
            pinch = v if math.isfinite(v) else None
    except (TypeError, ValueError, IndexError, AttributeError):
        pinch = None
    need = believed if pinch is None else max(believed, pinch)
    cos = close_push_opposes_pull(args.get("push_dir"), pull, cosine=True)
    mv = _finite_float((data or {}).get("moved"))
    travel = None if mv is None or mv < 0.0 else mv * -cos
    sufficient = travel is not None and travel >= need - CLOSE_TRAVEL_TOL_M - BOUND_EPS_M
    residual = max(0.0, need - (travel or 0.0))
    return {"version": DRAWER_BOOKING_VERSION, "believed_opening_m": round(believed, 4),
            "pinch_opening_m": None if pinch is None else round(pinch, 4), "need_m": round(need, 4),
            "travel_m": None if travel is None else round(travel, 4), "tol_m": CLOSE_TRAVEL_TOL_M,
            "sufficient": bool(sufficient), "residual_m": round(residual, 4),
            "opening_unknown_before": bool(opening_unknown_before), "residual_raw_m": residual}


def drawer_unmoved_since_last_ok(records: list[dict], region) -> bool:
    """Replay E rounds 2 and 3: True when the host's own gestures on drawer `region` since the last close the
    perception booked closed (record drawer_note "close_drawer": ok, push_dir opposing the pull) all failed before
    the rail was pinched (drawer_gesture_before_pinch), with at least one such failure; False otherwise (no gesture
    of the host on that region, any ok open, a close not booked closed, a failure that may have moved the drawer, a
    raised gesture, a record without drawer_note). An ok open must not restart the list while
    the unknown opening survives it (only a view clears it), otherwise a lost_handle, then an ok open of 2 cm, then a
    no_grasp_handle would let calibrate run with the drawer up to 6 cm out. Used by calibrate precondition (c) for an
    unknown opening under OPEN_BELIEF_M only. Robot side (records). drawer-booking-2; code unchanged: only
    drawer_note "close_drawer" restarts the list; an ok close noted close_drawer_short (travel short of the need) or
    close_drawer_unverified (the opening was unknown before it) is an ok gesture that may have moved the drawer, so
    it never exempts the entry (a restart on any ok close by name would let calibrate run there)."""
    if not isinstance(region, str) or not region:
        return False
    since = []
    for r in records:
        if r.get("kind") != "skill" or r.get("name") not in ("open_drawer", "close_drawer") \
                or r.get("drawer_region") != region or r.get("status") not in RAN_STATUSES:
            continue
        ok, reason, data = skill_result(r)
        if r.get("status") != BACKEND_RAISED_STATUS and ok is True and r.get("drawer_note") == "close_drawer":
            since = []
            continue
        since.append(r.get("status") != BACKEND_RAISED_STATUS and drawer_gesture_before_pinch(ok, reason, data))
    return bool(since) and all(since)


def untracked_drawer_open(records: list[dict]) -> int | None:
    """The index of the last open_drawer of the host's own
    records whose handle the robot could not resolve (drawer_region None: no tracked handle within 5 cm, so
    robot_state's drawers_believed_open never lists that drawer) and that may have moved its drawer: any result
    except a failure before the rail was pinched (drawer_gesture_before_pinch; drawer_gesture_moved missed a
    collision or blocked abort during the stroke), a raised gesture included. Never cleared within the episode (a
    close of another unresolved handle, already closed, returned ok and cancelled a pending opening; nothing robot
    side ties a close to that drawer). A close_drawer of an unresolved handle counts
    as well, on the same terms (with no tracked pull, nothing shows that its push_dir opposed the drawer's pull; a
    push along the pull pulls the drawer out and returns ok). None when there is no such gesture. Robot side."""
    last = None
    for i, r in enumerate(records):
        if r.get("kind") != "skill" or r.get("status") not in RAN_STATUSES or r.get("drawer_region") is not None \
                or r.get("name") not in ("open_drawer", "close_drawer"):
            continue
        ok, reason, data = skill_result(r)
        if r.get("status") == BACKEND_RAISED_STATUS or not drawer_gesture_before_pinch(ok, reason, data):
            last = i
    return last


# ====================================================================== task evidence
def evidence_target(task) -> dict:
    """The public target of the task (feedback.public_task: target_description {category, region}, scrub_request
    seconds; criteria 1.4 bowl, landing_tray, 3.0 s), with the same values as defaults when the task dict has no such
    field (test tasks). Never a taskcheck threshold, never the simulator's name of the target."""
    t = task if isinstance(task, dict) else {}
    td = t.get("target_description") if isinstance(t.get("target_description"), dict) else {}
    sr = t.get("scrub_request") if isinstance(t.get("scrub_request"), dict) else {}
    cat, reg, sec = td.get("category"), td.get("region"), sr.get("seconds")
    ok_sec = isinstance(sec, (int, float)) and not isinstance(sec, bool) and math.isfinite(float(sec))
    return {"category": cat if isinstance(cat, str) and cat else EVIDENCE_TARGET_DEFAULT["category"],
            "region": reg if isinstance(reg, str) and reg else EVIDENCE_TARGET_DEFAULT["region"],
            "seconds": float(sec) if ok_sec else EVIDENCE_SCRUB_S_DEFAULT}


def scrub_reached_contact(r: dict) -> bool:
    """Did this scrub record touch something? ok, a reason of SCRUB_CONTACT_REASONS, or a felt time or
    fraction above 0 in its data (a raised record through its gesture's result, skill_result)."""
    ok, reason, data = skill_result(r)
    if ok is True or reason in SCRUB_CONTACT_REASONS:
        return True
    for k in ("felt_s", "felt_fraction"):
        v = data.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)) and v > 0:
            return True
    return False


def _ran(r: dict) -> bool:
    return isinstance(r, dict) and r.get("kind") == "skill" and r.get("status") in RAN_STATUSES


def _attempt_of(i, attempts):
    """(attempt index, source) of the attempt whose host_records_span [n0, n1) holds record i, else (None, None)."""
    if i is None or not isinstance(attempts, list):
        return None, None
    for k, a in enumerate(attempts):
        sp = a.get("host_records_span") if isinstance(a, dict) else None
        if isinstance(sp, (list, tuple)) and len(sp) == 2 and all(type(x) is int for x in sp) and sp[0] <= i < sp[1]:
            src = a.get("source")
            return k, src if isinstance(src, str) else None
    return None, None


def bowl_moving_records(records: list[dict], j: int, region) -> list[int]:
    """bowl-pose-at-scrub-1: the indexes (< j) of the host records after which a pose measured
    earlier of an object on `region` (the PUBLIC target region, never the flickering label of the latest view) no
    longer holds for the bowl evidence: pose_moving_records(records[:j], region), except a failed pick (no contact
    incident, no internal incident, no grip-loss code) whose pick_check names another known region
    (perception.estimate.note_pick's own convention: a failed pick makes stale the objects of its own region).
    Without that exception, pose_moving_records refused two recorded successes (a failed no_grasp pick in sink_basin,
    88 cm from the bowl). Robot side (records)."""
    out = []
    for i in pose_moving_records(records[:j], region):
        r = records[i]
        ok, reason, data = skill_result(r)
        if r.get("name") == "pick" and r.get("status") == "called" and ok is False \
                and not is_contact_incident(reason, data) and internal_incident(r) is None \
                and reason not in GRIP_LOSS_CODES:
            pc = r.get("pick_check")
            pr = pc.get("region") if isinstance(pc, dict) else None
            if isinstance(pr, str) and pr and isinstance(region, str) and pr != region:
                continue
        out.append(i)
    return out


def task_evidence(records: list[dict], observations: list[dict], delivered_obs_ids: list, holding_now, task: dict,
                  span=None, *, base_xy=None, skill_calls: int | None = None, facade_log_unaccounted: int | None = None,
                  attempts: list | None = None) -> dict:
    """Positive robot-side evidence that the public task was done in THIS episode (task-evidence-2;
    E2 and E2' below). Pure: it reads the host's records, the observations DELIVERED in this episode (only views whose
    obs_id is in delivered_obs_ids and was delivered by a record: a view of an earlier episode, or not delivered, never
    counts), the grip reading at the guard (holding_now), the public target (evidence_target) and the robot's base
    position (base_xy, for the public rule "several bowls on the tray: the one nearest the base"). No taskcheck
    threshold, no oracle name.
    - checked sponge pick: the episode's FIRST skill record pick with ok True and args.kind "sponge". Its pick_check
      must say category sponge (declared_kind absent or sponge) and name a track_id that is a sponge detection of its
      view_obs_id, a view delivered before the pick. Else missing "no_checked_sponge_pick".
    - Target: the bowl tracks (target category) seen on the target region (landing_tray) in delivered views; several:
      the one nearest base_xy (xy of its latest tray detection; ties by track id); several and no base reading: missing
      "target_bowl_ambiguous". A tray track superseded at evaluation (see pose_before) is not a candidate. The same rule
      over the views delivered before scrub j gives the target that judges scrub j (E2 and E2' below); the evaluation's
      target (over every delivered view) is the one reported and the one a proof needs.
    A scrub is never judged on a view delivered after it, nor on a pose that a newer view contradicts (temporal rule
    EVIDENCE_TEMPORAL_RULE "bowl-pose-at-scrub-1"): each scrub at record j is judged on pose_before(track, j), the state
    of a tray bowl track's pose as the robot knew it from the views DELIVERED BEFORE j only, checked in this order
    (POSE_STATES):
      unseen no detection of the track (target category) before j; unmeasured detections, none measured in its view
      (pose_source this_view); stale a record of bowl_moving_records lies after the latest view that MEASURED the track
      (the staleness
                  clock starts at the measuring view, not at an earlier_view sighting);
      contested a measured detection of another object, in a view from that measuring view up to j, lies within
                  EVIDENCE_CONTEST_XY_M (xy) of the track's latest pose; never the picked sponge track, the task's tool
                  (EVIDENCE_TOOL) after E1, an object seen more than its own height above the bowl's top (a held or
                  carried object: the perception gives a held object seen again a NEW track labelled by its position),
                  or a detection in region "gripper" (declared: never reached with perception.estimate);
      superseded a view delivered after the track's latest detection and before j measured ANOTHER bowl track on the
                  tray without detecting it (the perception re-tracks a bowl moved more than its 6 cm gate); that other
                  track must be a re-track, first detected after the track's latest detection: a bowl seen together with
                  it earlier is a second bowl, and a narrower view that misses the track (field of view, occlusion)
                  leaves it valid;
      valid else; the pose is the xy of the track's LATEST detection before j (a newer view always replaces an
                  older one, whatever the distance).
      The target states of scrub j are those of the target that judges it (the target rule over the views delivered
      before j; ambiguous before j: of every candidate then; no tray track before j: of the evaluation's target,
      unseen); the other tray tracks' states are kept for provenance.
    - target scrub: the first scrub after E1 with ok True, args.seconds at least the public scrub seconds (minus 1e-9),
      the target known and seen on the tray in a view delivered before it (identity-based: the latest region may
      flicker), the target judging it equal to the evaluation's target, the target's state before j valid and target_pos
      within EVIDENCE_TARGET_XY_M (xy only, no sigma bound, no z) of that latest pose. Else missing
      "no_target_scrub_after_pick".
    - EVERY scrub after E1 that reached contact (scrub_reached_contact), when a tray bowl track is known at evaluation:
      on target when a valid state of the target judging it lies within 3 cm (xy), but voided "scrub_target_unconfirmed"
      when that target is not the evaluation's (a later view changed the target: it can void a past scrub, never repair
      it); else voided "scrub_target_unconfirmed" when target_pos lies within 3 cm of the pose of a tray track whose
      state is not valid (a stale, contested or superseded pose, such as the ghost of a re-tracked bowl) or when no
      target state is valid (the target unseen before j included); else voided "scrub_off_target" (a valid target
      farther than 3 cm, or target_pos unreadable). A void covers the whole episode (conservative): a contact on an
      unconfirmed target may have touched another dish (the verifier's no_other_dish_scrubbed), and no later view can
      repair a past scrub. No tray bowl track at all: E2 is missing and E2' voids nothing. Neither the target judging
      scrub j nor its poses and states come from a view delivered after j, so a later view can never turn an off-target
      scrub into the proof. Declared: a bowl re-tracked under a new id after the proving scrub (moved more than 6 cm)
      changes the evaluation's target and voids the scrub (one tray track in all 176 recorded claims). Records only:
      "temporal_rule" and "scrubs" (at most EVIDENCE_SCRUBS_MAX entries {record_index, contact, ok, judged (on_target,
      off_target, unconfirmed, no_contact, no_target_known), distance_m, proves, target_track (the target judging it,
      None when ambiguous or unknown before it), target_states, other_tray_states when any}).
    - continuity from E1 INCLUSIVE (readings before E1 are ignored): voided by another ok pick after it
      ("second_pick_after_sponge_pick"), a skill with holding_after False ("grip_not_felt_after_gesture"), a recorded
      holding() query that read False after it ("holding_reading_false"), a skill with internal_unitemized above 0
      ("unitemized_robot_calls"). Episode reconciliation (any record): counters.skill_calls (skill_calls) must equal the
      sum of robot_calls of the skills that ran, and a facade with a call log must have no call left unitemized
      (facade_log_unaccounted 0; already_open makes no facade call and none is expected), else voided
      "unrecorded_robot_calls". None skips a check (older records, unit tests). Any skill record of the episode whose
      backend raised (BACKEND_RAISED_STATUS), before or after E1, voids it too "backend_raised": what the backend did
      after its gesture (perception bookkeeping, the clearance) is not recorded, so its records cannot prove the task.
      No recorded run had a host exception.
    - holding_now: True needed. False voids after E1 ("hand_empty_at_done"; missing without E1); None (unreadable) is
      missing "grip_unreadable". Asymmetry declared: holding_after None is unknown, never a loss.
    Contact incidents and grip events are checked before this by agent.loop.refuse_done_after_incident. verdict:
    "voided" if anything voids, else "not_proven" if anything is missing, else "proven". Provenance: attempts (the
    loop's attempt list) gives the attempt index and source (model or memory) of the pick and of the scrub; span (the
    final attempt's [n0, n1)) gives pick_in_span and scrub_in_span. Plain values only."""
    tgt = evidence_target(task)
    eps = 1e-9
    ds = {o for o in (delivered_obs_ids or []) if type(o) is str and o}
    at = {}  # obs_id -> index of the record that delivered it
    for i, r in enumerate(records):
        oid = delivered_obs_id(r) if isinstance(r, dict) else None
        if oid is not None and oid in ds and oid not in at:
            at[oid] = i
    vs = {}  # obs_id -> the latest observation of the log with that id (a reused backend may repeat an id)
    for o in observations or []:
        if isinstance(o, dict) and o.get("obs_id") in at and isinstance(o.get("detections"), list):
            vs[o["obs_id"]] = o
    seq = sorted(((at[k], [d for d in o["detections"] if isinstance(d, dict)], k) for k, o in vs.items()),
                 key=lambda x: x[0])  # (record index, detections, obs_id) in delivery order
    missing, voided = [], []

    def add(lst, code):
        if code not in lst:
            lst.append(code)

    # The first ok sponge pick and its check
    pick_i = next((i for i, r in enumerate(records) if _ran(r) and r.get("name") == "pick" and r.get("ok") is True
                   and (r.get("args") if isinstance(r.get("args"), dict) else {}).get("kind") == "sponge"), None)
    pick_track, pick_checked = None, False
    if pick_i is not None:
        pc = records[pick_i].get("pick_check")
        if isinstance(pc, dict):
            tid, vid = pc.get("track_id"), pc.get("view_obs_id")
            pick_track = tid if isinstance(tid, str) else None
            view = vs.get(vid) if isinstance(vid, str) and at.get(vid, len(records)) < pick_i else None
            pick_checked = (pc.get("category") == "sponge" and pc.get("declared_kind") in (None, "sponge")
                            and pick_track is not None and view is not None
                            and any(isinstance(d, dict) and d.get("track_id") == pick_track
                                    and d.get("category") == "sponge" for d in view["detections"]))
    if not pick_checked:
        add(missing, "no_checked_sponge_pick")

    # task-evidence-2: the state of a tray bowl track's pose before record j (pose_before)
    tcat, treg = tgt["category"], tgt["region"]
    shape = EVIDENCE_OBJECT_SHAPES.get(tcat)
    top_above = shape[0] if shape else 0.0  # the target's top above its centre (the detections' reference point)
    moving_cache = {}

    def moving_before(j):
        if j not in moving_cache:
            moving_cache[j] = bowl_moving_records(records, j, treg)
        return moving_cache[j]

    def supersedes(d, tid) -> bool:
        """a measured detection of ANOTHER track of the target category on the target region"""
        return (isinstance(d.get("track_id"), str) and d.get("track_id") != tid and d.get("category") == tcat
                and d.get("region") == treg and d.get("pose_source", "this_view") == "this_view"
                and finite_vec(d.get("position_m"), 2) is not None)

    def last_detection(tid, j):
        """record index of the latest view delivered before j that detected tid (target category), else None"""
        return next((oi for oi, dets, _ in reversed(seq) if oi < j and any(
            d.get("track_id") == tid and d.get("category") == tcat for d in dets)), None)

    first_cache = {}

    def first_detection(tid):
        """record index of the first delivered view that detected tid (target category), else None"""
        if tid not in first_cache:
            first_cache[tid] = next((oi for oi, dets, _ in seq if any(
                d.get("track_id") == tid and d.get("category") == tcat for d in dets)), None)
        return first_cache[tid]

    def superseded_by(tid, j):
        """(view obs_id, other track) of the first view after tid's latest detection and before j that measured
        another tray bowl track without detecting tid, else None. Only a RE-TRACK supersedes, a track first detected
        after tid's latest detection; a bowl seen together with tid in an earlier view is a second bowl (a narrower
        view that misses tid does not make it superseded)."""
        last = last_detection(tid, j)
        if last is None:
            return None
        for oi, dets, oid in seq:
            if last < oi < j:
                other = next((d for d in dets if supersedes(d, tid) and first_detection(d["track_id"]) is not None
                              and first_detection(d["track_id"]) > last), None)
                if other is not None:
                    return oid, other["track_id"]
        return None

    def contests(d, oi, tid, p, p3) -> bool:
        """does detection d (in the view delivered at record oi) contest track tid's pose p (xy; p3 xyz or None)?"""
        q, cat = finite_vec(d.get("position_m"), 2), d.get("category")
        if q is None or d.get("track_id") == tid or cat == tcat or d.get("pose_source", "this_view") != "this_view":
            return False
        if (pick_track is not None and d.get("track_id") == pick_track) or d.get("region") == "gripper":
            return False
        if cat == EVIDENCE_TOOL and pick_i is not None and oi > pick_i:
            return False  # the task's one sponge after E1: in the hand, never another dish (held-sponge views)
        q3 = finite_vec(d.get("position_m"), 3)
        dh = EVIDENCE_OBJECT_SHAPES[cat][1] if cat in EVIDENCE_OBJECT_SHAPES else EVIDENCE_HEIGHT_DEFAULT_M
        if p3 is not None and q3 is not None and q3[2] > p3[2] + top_above + dh + eps:
            return False  # more than its own height above the bowl's top: held or carried, not on the tray
        return math.dist(q, p) <= EVIDENCE_CONTEST_XY_M + eps

    def pose_before(tid, j):
        """(state dict for the records, unrounded xy or None) of tid's pose as the robot knew it just before record j"""
        dets = [(oi, d, oid) for oi, dd, oid in seq if oi < j for d in dd
                if d.get("track_id") == tid and d.get("category") == tcat
                and finite_vec(d.get("position_m"), 2) is not None]
        if not dets:
            return {"state": "unseen", "track": str(tid)[:24]}, None
        last_oi, last, last_oid = dets[-1]
        p, p3 = finite_vec(last.get("position_m"), 2), finite_vec(last.get("position_m"), 3)
        ps, rg = last.get("pose_source"), last.get("region")
        st = {"track": str(tid)[:24], "pos": [round(x, 4) for x in p], "view_obs_id": last_oid, "view_record": last_oi,
              "pose_source": ps if isinstance(ps, str) else None, "region": rg if isinstance(rg, str) else None}
        meas = [(oi, oid) for oi, d, oid in dets if d.get("pose_source", "this_view") == "this_view"]
        if not meas:
            return {"state": "unmeasured", **st}, p
        m_oi, m_oid = meas[-1]
        st.update(measured_obs_id=m_oid, measured_record=m_oi)
        mv = [i for i in moving_before(j) if i > m_oi]
        if mv:
            return {"state": "stale", **st, "moved_by_record": mv[0],
                    "moved_by": str(records[mv[0]].get("name"))[:24]}, p
        for oi, dd, oid in seq:
            if m_oi <= oi < j:
                d = next((x for x in dd if contests(x, oi, tid, p, p3)), None)
                if d is not None:
                    return {"state": "contested", **st, "contested_by": str(d.get("category"))[:20],
                            "contested_in": oid}, p
        sup = superseded_by(tid, j)
        if sup is not None:
            return {"state": "superseded", **st, "superseded_in": sup[0], "superseded_by": str(sup[1])[:24]}, p
        return {"state": "valid", **st}, p

    # Target: bowl tracks seen on the tray (delivered views), nearest the base if several (not superseded)
    def tray_tracks(j):
        """track_id -> xy of its latest tray detection in the views delivered before record j"""
        out = {}
        for oi, dets, _ in seq:
            if oi >= j:
                break
            for d in dets:
                p = finite_vec(d.get("position_m"), 2)
                if d.get("category") == tcat and d.get("region") == treg and p is not None \
                        and isinstance(d.get("track_id"), str):
                    out[d["track_id"]] = p
        return out

    bxy = finite_vec(base_xy, 2)

    def choose_target(trays, j):
        """Candidates, target or None, ambiguous: the tray tracks not superseded before j (never empty when trays is
        not: the track detected last is never superseded), the one nearest the base if several"""
        cs = {t: xy for t, xy in trays.items() if superseded_by(t, j) is None}
        if len(cs) == 1:
            return cs, next(iter(cs)), False
        if len(cs) > 1:
            if bxy is None:
                return cs, None, True
            return cs, min(sorted(cs), key=lambda t: math.dist(cs[t], bxy)), False
        return cs, None, False

    end = len(records)  # every delivered view lies before it
    tray = tray_tracks(end)
    cands, target, ambiguous = choose_target(tray, end)  # the evaluation's target (reported, and needed by a proof)

    def tray_sightings(tid, before):
        return sum(1 for oi, dets, _ in seq if oi < before and any(d.get("track_id") == tid and d.get("region") == treg
                                                                  and d.get("category") == tcat for d in dets))

    # The scrubs after E1, each judged on the views delivered before it
    tol = EVIDENCE_TARGET_XY_M + eps
    scrub_i, bowl_track, sightings, scrubs = None, None, None, []
    if pick_i is not None:
        for j in range(pick_i + 1, len(records)):
            r = records[j]
            if not (_ran(r) and r.get("name") == "scrub"):
                continue
            args = r.get("args") if isinstance(r.get("args"), dict) else {}
            tp = finite_vec(args.get("target_pos"), 2)
            contact = scrub_reached_contact(r)
            # The target that judges scrub j is chosen from the views delivered before j;
            # nothing known before j: the evaluation's (its state unseen)
            cands_j, target_j, _amb_j = choose_target(tray_tracks(j), j)
            if target_j is not None:
                tids = [target_j]
            elif cands_j:
                tids = sorted(cands_j)
            else:
                tids = ([target] if target is not None else sorted(cands)) if tray else []
            # a later view that changed the episode's target never proves scrub j (and never repairs it)
            changed = target is not None and target_j is not None and target_j != target
            states = [pose_before(t, j) for t in tids]
            others = [pose_before(t, j) for t in sorted(tray) if t not in tids]
            valid = [p for s, p in states if s["state"] == "valid"]
            dist = min(math.dist(p, tp) for p in valid) if valid and tp is not None else None
            if contact and tray:
                unconfirmed = [p for s, p in states + others if s["state"] != "valid" and p is not None]
                if tp is None:
                    judged = "off_target"
                elif dist is not None and dist <= tol:
                    judged = "unconfirmed" if changed else "on_target"
                elif any(math.dist(p, tp) <= tol for p in unconfirmed) or not valid:
                    judged = "unconfirmed"
                else:
                    judged = "off_target"
                if judged == "off_target":
                    add(voided, "scrub_off_target")
                elif judged == "unconfirmed":
                    add(voided, "scrub_target_unconfirmed")
            else:
                judged = "no_contact" if not contact else "no_target_known"
            sec = args.get("seconds")
            long_enough = isinstance(sec, (int, float)) and not isinstance(sec, bool) and math.isfinite(float(sec)) \
                and float(sec) >= tgt["seconds"] - eps
            ts = states[0] if target_j is not None and states else None  # tids == [target_j]
            proves = (scrub_i is None and r.get("ok") is True and long_enough and target is not None
                      and target_j == target and tp is not None and tray_sightings(target, j) > 0 and ts is not None
                      and ts[0]["state"] == "valid" and math.dist(ts[1], tp) <= tol)
            if proves:
                scrub_i, bowl_track, sightings = j, target, tray_sightings(target, j)
            if len(scrubs) < EVIDENCE_SCRUBS_MAX:
                e = {"record_index": j, "contact": contact, "ok": r.get("ok") if type(r.get("ok")) is bool else None,
                     "judged": judged, "distance_m": None if dist is None else round(dist, 4), "proves": proves,
                     "target_track": target_j, "target_states": [s for s, _ in states]}
                if others:
                    e["other_tray_states"] = [s for s, _ in others]
                scrubs.append(e)
    if scrub_i is None:
        add(missing, "no_target_scrub_after_pick")
        if ambiguous:
            add(missing, "target_bowl_ambiguous")

    # Continuity from E1 inclusive, and the episode reconciliation
    if pick_i is not None:
        for i in range(pick_i, len(records)):
            r = records[i]
            if _ran(r):
                if i > pick_i and r.get("name") == "pick" and skill_result(r)[0] is True:
                    add(voided, "second_pick_after_sponge_pick")
                if r.get("holding_after") is False:
                    add(voided, "grip_not_felt_after_gesture")
                un = r.get("internal_unitemized")
                if type(un) is int and un > 0:
                    add(voided, "unitemized_robot_calls")
            elif i > pick_i and isinstance(r, dict) and r.get("name") == "holding" and r.get("status") == "called" \
                    and r.get("value") is False:
                add(voided, "holding_reading_false")
    recorded = 0
    for r in records:
        if _ran(r):
            rc = r.get("robot_calls")
            recorded += rc if type(rc) is int else 0
    if (skill_calls is not None and skill_calls != recorded) \
            or (facade_log_unaccounted is not None and facade_log_unaccounted != 0):
        add(voided, "unrecorded_robot_calls")
    if any(_ran(r) and r.get("status") == BACKEND_RAISED_STATUS for r in records):
        add(voided, "backend_raised")  # anywhere in the episode

    # The grip reading at the guard
    hn = holding_now if type(holding_now) is bool else None
    if hn is None:
        add(missing, "grip_unreadable")
    elif hn is False:
        add(voided if pick_i is not None else missing, "hand_empty_at_done")

    sp = [span[0], span[1]] if isinstance(span, (list, tuple)) and len(span) == 2 \
        and all(type(x) is int for x in span) else None

    def in_span(i):
        return None if i is None or sp is None else sp[0] <= i < sp[1]
    pa, ps = _attempt_of(pick_i, attempts)
    sa, ss = _attempt_of(scrub_i, attempts)
    return {"version": TASK_EVIDENCE_VERSION, "verdict": "voided" if voided else "not_proven" if missing else "proven",
            "missing": missing, "voided_by": voided, "pick_index": pick_i, "pick_track": pick_track,
            "pick_checked": pick_checked, "scrub_index": scrub_i, "bowl_track": bowl_track,
            "bowl_tray_sightings": sightings, "holding_now": hn, "span": sp, "pick_in_span": in_span(pick_i),
            "scrub_in_span": in_span(scrub_i), "pick_attempt": pa, "pick_source": ps, "scrub_attempt": sa,
            "scrub_source": ss, "target": {**tgt, "track": target, "n_tray_tracks": len(tray)},
            "reconciliation": {"skill_calls": skill_calls, "recorded_robot_calls": recorded,
                               "facade_log_unaccounted": facade_log_unaccounted},
            "temporal_rule": EVIDENCE_TEMPORAL_RULE, "scrubs": scrubs}


# ====================================================================== toy kitchen (tests only)
def _d(a, b):
    return math.dist(a, b)


class StubKitchen:
    """A toy kitchen in plain Python for the agent-loop tests. Host-side: it knows the truth (where
    the sponge is); the program only gets what observe/queries return. Not the simulator."""

    OPEN_MIN_TO_SEE = 0.15

    def __init__(self, sponge_region: str | None = "drawer_unit:0", n_drawers: int = 3,
                 dossier_sponge_region: str = "drawer_unit:0", fail_first_pick: bool = False,
                 skill_s: float = 5.0, observe_s: float = 0.0):
        self.wz = 0.90
        self.drawers = {}
        for k in range(n_drawers):
            h = [-0.30 + 0.17 * k, 0.45, 1.03]
            self.drawers[f"drawer_unit:{k}"] = {"handle_closed": h, "pull": [0.0, -1.0, 0.0], "opening": 0.0}
        self.tray = [0.56, -0.14, 0.91]
        self.sink = [0.30, 0.55, 0.92]
        self.bowl = [0.56, -0.14, 0.931]
        self.sponge_region = sponge_region  # truth (host side)
        self.held = None
        self.hand = [0.0, 0.0, 1.3]
        self.t = 0.0
        self.skill_s, self.observe_s = skill_s, observe_s
        self.fail_first_pick = fail_first_pick
        self.stuck_drawers: set = set()  # the next open_drawer on these regions fails with "stuck"
        self.drop_on_scrub = 0  # the next N scrubs return "dropped"; the sponge falls into the sink basin
        # tests (mirroring perception.estimate): sigma_m of the next sponge detections, and the number of next views in
        # which a sponge already tracked is cut by the image edge (pose, sigma, pose_source, pose_step kept, status back
        # to observed, as estimate._update_track does with a truncated view)
        self.sponge_sigma_m = [0.01, 0.01, 0.01]
        self.truncated_views = 0
        self.close_grasps = 0  # the next N close_gripper report grasped True (a hold by hand)
        self._ids: dict[str, str] = {}
        self._tracks: dict[str, dict] = {}
        self._inspected: dict[str, dict] = {}
        self.observation_log: list[dict] = []
        self.n_obs = 0
        self.dossier_d = {"version": "stub-dossier-1", "worktop_height_m": self.wz,
                          "modules": [{"kind": "drawers", "slot": "left", "n_drawers": n_drawers},
                                      {"kind": "sink", "slot": "back"}, {"kind": "landing", "slot": "right"}],
                          "stored_items": [{"category": "sponge", "region": dossier_sponge_region}],
                          "conventions": {"tool_storage": "tools are stored in drawers or in the sink basin"}}

    # ------------------------------------------------------------ geometry
    def handle_pos(self, rid):
        d = self.drawers[rid]
        return [d["handle_closed"][i] + d["opening"] * d["pull"][i] for i in range(3)]

    def interior(self, rid):
        h = self.handle_pos(rid)
        return [h[0], h[1] + 0.16, h[2] - 0.108 + 0.022]

    def sponge_pos(self):
        r = self.sponge_region
        if r is None or r == "held":
            return None
        if r.startswith("drawer_unit:"):
            return self.interior(r)
        return [self.sink[0], self.sink[1], self.sink[2] + 0.022] if r == "sink_basin" else None

    def _tid(self, key):
        if key not in self._ids:
            self._ids[key] = f"trk-{len(self._ids) + 1:03d}"
        return self._ids[key]

    # ------------------------------------------------------------ backend interface
    def sim_time_used(self) -> float:
        return self.t

    def skill(self, name, a):
        self.t += self.skill_s + (float(a["seconds"]) if name == "scrub" else 0.0)
        ok = lambda **data: {"ok": True, "reason": "ok", "data": data}  # noqa: E731
        fail = lambda reason, **data: {"ok": False, "reason": reason, "data": data}  # noqa: E731
        if name == "move_to":
            self.hand = list(a["pos"])
            return ok()
        if name == "look_down_at":
            self.hand = [a["xy"][0], a["xy"][1], a["z_floor"] + a["height"]]
            return ok()
        if name == "look_at_region":
            rid = a["region"]
            if rid in self.drawers:
                c = self.interior(rid)
                self.hand = [c[0], c[1], c[2] + 0.42]
            elif rid == "sink_basin":
                self.hand = [self.sink[0], self.sink[1], self.sink[2] + 0.55]
            else:
                self.hand = [self.tray[0], self.tray[1], self.tray[2] + 0.55]
            return ok(where=rid)
        if name in ("open_drawer", "close_drawer"):
            for rid, d in self.drawers.items():
                if _d(self.handle_pos(rid), a["handle_pos"]) <= 0.03:
                    before = d["opening"]
                    if name == "open_drawer" and DRAWER_TRAVEL_M - before < OPEN_LEFT_MIN_M:
                        return ok_already_open(round(before, 4))  # same rule as FacadeBackend (ok True)
                    if name == "open_drawer" and rid in self.stuck_drawers:
                        self.stuck_drawers.discard(rid)
                        return fail("stuck", moved=0.04, force_peak=88.0, arm_moved=True)  # an incident
                    d["opening"] = min(DRAWER_TRAVEL_M, before + a["distance"]) if name == "open_drawer" else 0.0
                    for t in self._tracks.values():  # the robot's own drawer motion: contents become stale
                        if t["region"] == rid:
                            t["status"] = "stale"
                            if d["opening"] != before:  # provenance as perception.note_drawer_motion (toy: no shift)
                                t["pose_source"] = "dead_reckoned_after_action"
                    return ok(moved=round(abs(d["opening"] - before), 4))
            return fail("no_handle", arm_moved=True)
        if name == "pick":
            sp = self.sponge_pos()
            d = self.drawers.get(self.sponge_region or "", {"opening": 1.0})
            if self.fail_first_pick:
                self.fail_first_pick = False
                return fail("missed", grasped=False)
            if a["kind"] == "sponge" and sp is not None and d["opening"] >= self.OPEN_MIN_TO_SEE \
                    and _d(sp, a["pos"]) <= 0.03:
                self.held, self.sponge_region = "sponge", "held"
                for t in self._tracks.values():
                    if t["category"] == "sponge":
                        t["status"] = "held"
                return ok(grasped=True, width=0.05)
            return fail("missed", grasped=False)
        if name == "scrub":
            if self.held != "sponge":
                return fail("not_holding")
            if self.drop_on_scrub > 0:
                self.drop_on_scrub -= 1
                self.held, self.sponge_region = None, "sink_basin"
                return fail("dropped", stage="scrub")
            top = [self.bowl[0], self.bowl[1], self.bowl[2] + 0.023]
            if _d(top, a["target_pos"]) > 0.03:
                return fail("no_contact", force_mean=0.0)
            return ok(force_mean=a["force"], felt_fraction=0.95, strokes=3)
        if name == "place":
            if self.held is None:
                return fail("not_holding")
            self.held = None
            self.sponge_region = "other"
            return ok()
        if name == "close_gripper" and self.close_grasps > 0:  # something is held by hand
            self.close_grasps -= 1
            return ok(grasped=True, width=0.02)
        if name == "open_gripper" and self.held == "sponge":  # the sponge falls into the sink basin
            self.held, self.sponge_region = None, "sink_basin"
            return ok()
        return ok()

    def observe(self, camera):
        self.t += self.observe_s
        self.n_obs += 1
        dets, handles, regions = [], [], []

        cut = self.truncated_views > 0
        self.truncated_views = max(0, self.truncated_views - 1)

        def det(key, cat, pos, region):
            tid = self._tid(key)
            old = self._tracks.get(tid)
            if cut and cat == "sponge" and old is not None and old["status"] != "held":
                # Test hook: a view cut by the image edge confirms the track but keeps its old pose (estimate.py)
                old["status"] = "observed"
                src = "dead_reckoned_after_action" if old["pose_source"] == "dead_reckoned_after_action" else "earlier_view"
                dets.append({k: v for k, v in old.items() if k != "status"} | {"pose_source": src, "n_pixels": 120})
                return
            sig = list(self.sponge_sigma_m) if cat == "sponge" else [0.01, 0.01, 0.01]
            self._tracks[tid] = {"track_id": tid, "category": cat, "status": "observed", "region": region,
                                 "position_m": [round(x, 4) for x in pos], "sigma_m": sig,
                                 "yaw_rad": 0.0 if cat == "sponge" else None, "pose_source": "this_view",
                                 "pose_step": self.n_obs}
            dets.append({k: v for k, v in self._tracks[tid].items() if k != "status"} | {"n_pixels": 400})

        if camera == "robot_eye":
            for rid in self.drawers:
                tid = self._tid("handle:" + rid)
                hp = self.handle_pos(rid)
                self._tracks[tid] = {"track_id": tid, "category": "drawer_handle", "status": "observed",
                                     "region": rid, "position_m": hp, "sigma_m": [0.005] * 3, "yaw_rad": None,
                                     "pose_source": "this_view", "pose_step": self.n_obs}
                handles.append({"track_id": tid, "category": "drawer_handle", "position_m": hp,
                                "pull_dir": self.drawers[rid]["pull"]})
            det("bowl", "bowl", self.bowl, "landing_tray")
            regions.append(self._inspect("landing_tray", 0.8, ["bowl"]))
            seen_sink = []
            if self.sponge_region == "sink_basin":
                det("sponge", "sponge", self.sponge_pos(), "sink_basin")
                seen_sink = ["sponge"]
            regions.append(self._inspect("sink_basin", 0.7, seen_sink))
        else:
            for rid in self.drawers:
                c = self.interior(rid)
                above = _d(c[:2], self.hand[:2]) <= 0.12 and 0.2 <= self.hand[2] - c[2] <= 0.7
                if above and self.drawers[rid]["opening"] >= self.OPEN_MIN_TO_SEE:
                    seen = []
                    if self.sponge_region == rid:
                        det("sponge", "sponge", self.sponge_pos(), rid)
                        seen = ["sponge"]
                    regions.append(self._inspect(rid, 0.9, seen))
        obs = {"obs_id": f"obs-{self.n_obs:03d}", "camera": camera, "step": self.n_obs, "t_s": round(self.t, 2),
               "detections": dets, "handles": handles, "regions": regions}
        self.observation_log.append(obs)
        return obs

    def _inspect(self, rid, cov, seen):
        # toy rule: a view of 60 % or more leaves no hiding place (the real backend computes it from the grid)
        r = {"region": rid, "coverage": cov, "seen_categories": list(seen),
             "could_hide": {c: cov < 0.6 for c in OBJECT_CATEGORIES}}
        self._inspected[rid] = r
        return r

    def drawer_region_at(self, handle_pos) -> str | None:
        """The drawer whose handle is at handle_pos (3 cm, the same rule as open_drawer here), else None."""
        for rid in self.drawers:
            if _d(self.handle_pos(rid), handle_pos) <= 0.03:
                return rid
        return None

    def robot_state(self) -> dict:
        """Toy robot-side state (tests): grip, and the drawers (with a seen handle) pulled out by the robot."""
        seen = [rid for rid in self.drawers if self._ids.get("handle:" + rid) is not None]
        return {"holding": self.held is not None, "drawer_handles_perceived": len(seen),
                "drawers_believed_open": [{"region": rid, "believed_opening_m": round(self.drawers[rid]["opening"], 3),
                                           "opening_unknown": False} for rid in seen
                                          if self.drawers[rid]["opening"] >= OPEN_BELIEF_M]}

    def holding_reading(self) -> bool:
        """The toy grip reading (the same as the holding() query), not counted. The toy has no clearance and no
        internal calls (no last_call_detail): tests that need them subclass it (tests/siteagent/_v47_fakes.py)."""
        return self.held is not None

    def query(self, name, a):
        if name == "tracks":
            return [dict(t) for t in self._tracks.values() if t["category"] == a["category"]
                    and t["status"] in ("observed", "stale", "held")]
        if name == "handle_for_region":
            tid = self._ids.get("handle:" + a["region"])
            if tid is None or a["region"] not in self.drawers:
                return None
            return {"track_id": tid, "position_m": self.handle_pos(a["region"]), "sigma_m": [0.005] * 3,
                    "pull_dir": self.drawers[a["region"]]["pull"], "opening_m": self.drawers[a["region"]]["opening"],
                    "truncated": False, "visibility_estimate": 1.0}
        if name == "search_scope":
            cands = list(self.drawers) + ["sink_basin", "landing_tray"]
            rows = []
            for r in cands:
                ins = self._inspected.get(r)
                seen = bool(ins and a["category"] in ins["seen_categories"])
                rows.append({"region": r, "coverage": ins["coverage"] if ins else 0.0, "seen": seen,
                             "cleared": bool(ins and ins["coverage"] >= 0.6 and not seen)})
            found = [t["track_id"] for t in self._tracks.values() if t["category"] == a["category"]]
            outcome = "found" if found else ("not_found_in_inspected_scope" if all(r["cleared"] for r in rows)
                                             else "not_found_scope_incomplete")
            return {"outcome": outcome, "found_tracks": found, "scope": rows,
                    "not_cleared": [r["region"] for r in rows if not r["cleared"]]}
        if name == "difference_report":
            entries = []
            for it in self.dossier_d["stored_items"]:
                ins = self._inspected.get(it["region"])
                typ = ("declared_location_not_inspected" if ins is None else
                       "confirmed" if it["category"] in ins["seen_categories"] else "not_seen_at_declared_location")
                elsewhere = [r for r, i in self._inspected.items() if r != it["region"]
                             and it["category"] in i["seen_categories"]]
                entries.append({"category": it["category"], "declared_region": it["region"], "type": typ,
                                "observed_in": elsewhere})
            return {"entries": entries, "n_differences_or_unknowns": sum(e["type"] != "confirmed" for e in entries)}
        if name == "dossier":
            return self.dossier_d
        if name == "holding":
            return self.held is not None
        if name == "worktop_z":
            return self.wz
        if name == "base_xy":
            return [0.0, 0.0]
        raise ArgError(f"unknown query {name}")


# ====================================================================== real wiring (proposal)
_DET_KEYS = ("track_id", "category", "region", "pose_source", "pose_step", "n_pixels")
# programs.facade.DATA_KEYS minus "error" (free text, see FacadeBackend.skill); a test checks the two
# lists stay equal (this module imports nothing of the robot, so the list is repeated here).
_SKILL_DATA_KEYS = ("moved", "asked", "strokes", "width", "grasped", "arm_moved", "where", "force_n", "stage",
                    "end_stop", "expected", "felt_fraction", "force_mean", "t_start", "t_end", "stop_dz", "lift",
                    "force_peak", "felt_s", "gesture")
_FREE_TEXT_DATA_KEYS = ("error",)  # never passed to a program (an exception text could name simulator objects)


def region_could_hide(r: dict) -> dict:
    """Per object category, could an object of that category lie unseen in this region view?
    True / False from the view's own floor grid (sitemodel.could_hide, robot side: what the camera reached), None
    when the view carries no grid (unknown). A partial view therefore never reads as absence: absence needs an
    explicit False."""
    from ..sitemodel import could_hide  # robot side, numpy; imported only where observations are built
    if not (r.get("grid_seen") and r.get("grid_shape") and r.get("grid_pitch_m")):
        return {c: None for c in OBJECT_CATEGORIES}
    out = {}
    for c in OBJECT_CATEGORIES:
        try:
            out[c] = could_hide(r, c)
        except (ValueError, IndexError, TypeError):  # a malformed grid: unknown, never absence
            out[c] = None
    return out


def plain_observation(o: dict) -> dict:
    """Observation (perception.contract.to_dict) -> what a program receives. Whitelist only: the
    provenance details and the floor grids are not passed; each region carries instead, per object category, whether
    its unseen part could hide such an object (region_could_hide, skill API 0.2). The
    sensor does not output the oracle visible fraction and sigma_m is computed from the robot's own visibility
    estimate (perception.estimate, ORACLE_INPUTS = ()); a test checks it end to end."""
    return {"obs_id": o["obs_id"], "camera": o["provenance"]["sensor"], "step": o["provenance"]["step"],
            "t_s": o["provenance"]["t_s"],
            "detections": [{**{k: d[k] for k in _DET_KEYS if k in d}, "position_m": d["pose"]["position_m"],
                            "sigma_m": d["pose"]["sigma_m"], "yaw_rad": d["pose"].get("yaw_rad")}
                           for d in o["detections"]],
            "handles": [{"track_id": h["track_id"], "category": h["category"], "position_m": h["pose"]["position_m"],
                         "pull_dir": h["pull_dir"]} for h in o["handles"]],
            "regions": [{"region": r["region"], "coverage": r["coverage"], "seen_categories": r["seen_categories"],
                         "could_hide": region_could_hide(r)} for r in o["regions"]]}


class FacadeBackend:
    """Backend over the skill facade + perception + site model. Holds the robot-side objects;
    the program never receives them: only the plain dicts returned by skill/observe/query cross
    to the interpreter. dossier: sitemodel.SiteDossier (to_dict for the program)."""

    def __init__(self, facade, perception, site_model, dossier, to_dict):
        """to_dict: perception.contract.to_dict (passed in so this module imports nothing of the robot)."""
        self._f, self._p, self._m, self._dossier, self._to_dict = facade, perception, site_model, dossier, to_dict
        self.observation_log: list[dict] = []
        self.last_robot_calls = 1
        self._looker = None
        # internal-actions-1: what the last skill() did besides its own gesture (set on every return path), the facade
        # log length when this backend was built and the facade calls skill() itemized since (reconciliation)
        self.last_call_detail = {"internal": [], "unitemized": 0, "clearance": None}
        log = self._facade_log()
        self._log_start = None if log is None else len(log)
        self._log_itemized = 0
        self._direct: list[tuple[str, object]] = []  # the clearance's own results, for a facade without a call log
        self._clearance = None
        # handle-reref-1, the handle tracks of the drawers the robot moved since the program's last handle_for_region
        # returned them; bound-known-1, handle track -> the believed opening before a failed drawer gesture that
        # reported no travel and no contact (a lower bound on the opening, plus 5 mm); the records-only details of the
        # current skill (reref, drawer_bound, retreat)
        self._moved_since_query: set[str] = set()
        self._notravel: dict[str, float] = {}
        self._reref = self._drawer_bound = None
        self._retreat: list[dict] = []
        self._retreat_n0 = 0

    def sim_time_used(self) -> float:
        return float(self._f.budget_state()["sim_s_used"])

    def robot_budget_exceeded(self) -> str | None:
        """GuardedHost.robot_budget_exceeded: the facade's own budget
        verdict, programs.facade.SkillFacade.budget_state()["exceeded"] ("max_sim_s" when the robot clock minus the
        facade's t0 is over max_sim_s, unrounded; "max_skill_calls" once a call was refused past the call budget), else
        None; None for a facade double without budget_state. The facade's budget is the task criteria's, the one its own
        skill refusals ("budget_exceeded") apply; taskcheck.BenchEvidence counts the same clock from the same instant
        (bench_task.run_task_episode builds it just before the facade). A reading: not a skill call, not logged."""
        fn = getattr(self._f, "budget_state", None)
        if not callable(fn):
            return None
        st = fn()
        ex = st.get("exceeded") if isinstance(st, dict) else None
        return str(ex) if ex else None

    def _facade_log(self) -> list | None:
        """The facade's own call log (programs.facade.SkillFacade.log: one entry per facade call, robot side),
        or None when the facade keeps none (test doubles) or it cannot be read."""
        try:
            log = self._f.log
        except Exception:  # noqa: BLE001 (AttributeError on a double without a log)
            return None
        return log if isinstance(log, list) else None

    def holding_reading(self) -> bool | None:
        """The robot's grip reading (facade.holding(), a reading: not a skill call, not logged, not counted), or
        None when it raises or is not a bool."""
        try:
            v = self._f.holding()
        except Exception:  # noqa: BLE001 (an unreadable sensor is unknown)
            return None
        if type(v) is bool:
            return v
        try:
            import numpy as np  # robot side only
            return bool(v) if isinstance(v, np.bool_) else None
        except ImportError:
            return None

    def facade_log_unaccounted(self) -> int | None:
        """Facade calls in the facade's log since this backend was built that no skill() itemized
        (as the parent gesture or an internal entry); 0 when every robot call is accounted for. None for a facade
        without a call log; -1 when the log can no longer be read."""
        if self._log_start is None:
            return None
        log = self._facade_log()
        if log is None:
            return -1
        return len(log) - self._log_start - self._log_itemized

    def _n_facade_calls(self) -> int:
        try:
            return int(self._f.budget_state()["skill_calls"])
        except Exception:  # noqa: BLE001 (a test double without the counter)
            return 0

    def _nearest_track(self, kind_cats, pos, max_m=0.05):
        best, bd = None, max_m
        for t in self._p.tracks.values():
            if t.category in kind_cats:
                d = math.dist([float(x) for x in t.pos[:3]], pos)
                if d <= bd:
                    best, bd = t.track_id, d
        return best

    def _search(self):
        """The scripted chain's look helpers (same rule for C and S), robot side only."""
        if self._looker is None:
            from ..search import SiteSearch
            self._looker = SiteSearch(self._f, self._p, self._dossier)
        return self._looker

    def _release(self):
        """Same clearance as the scripted chain after a pull or push stopped on a fault: open the
        fingers and back up 8 cm (search.SiteSearch._release). Its two results are no longer thrown away: they are
        itemized in the parent record (from the facade log, or directly for a facade without one) and the clearance's
        fate is recorded (last_call_detail)."""
        f = self._f
        self._clearance = {"ran": True, "ok": None}
        r1 = f.open_gripper()
        self._direct.append(("open_gripper", r1))
        p, _ = f.tcp_pose(f.q_cmd)
        r2 = f.move_to([p[0], p[1], p[2] + 0.08], orient=f.tcp_pose(f.q_cmd)[1], speed=0.1)
        self._direct.append(("move_to", r2))
        self._clearance["ok"] = bool(r1.ok and r2.ok)
        from ..search import reason_detail, retreat_records  # contact-invalidates-1: the failing call's reason
        self._fix_state("_retreat", list).extend(retreat_records(r1) + retreat_records(r2))  # records only
        self._p.note_action("release", {"ok": bool(r1.ok and r2.ok), **reason_detail(r2 if not r2.ok else r1)})

    def _look_point(self, region: str, part: str):
        """(xy, z_floor, height, min_tcp_z) or a failure reason. Drawer: centre of its interior from
        the perceived handle; surface: from the perceived module (sink: look forward, TCP above the faucet). part
        "unseen": the mean of the floor points not seen in the latest view. Part "centre" only (part "unseen" follows
        search.SiteSearch.unseen_aim, see _composite); a surface is looked at from its fitted centre when perception
        accepted its outline (SiteSearch.surface_centre), else from its track as before. part "unseen" returns the
        aim record of unseen_aim."""
        import numpy as np
        from ..perception import catalog as C
        s = self._search()
        if part == "unseen":
            return s.unseen_aim(region)
        if region.startswith("drawer_unit:"):
            h = self._p.handle_for_region(region)
            if h is None:
                return "handle_not_perceived"
            pull = np.array([h.pull[0], h.pull[1]], float)
            z = float(h.pos[2]) - C.DRAWER_FLOOR_BELOW_HANDLE
            xy, height, min_tcp = h.pos[:2] - 0.16 * pull, 0.42, None
        else:
            cat = next((c for c, r in C.SURFACE_REGION.items() if r == region), None)
            mod = next((t for t in self._p.tracks.values() if t.category == cat), None)
            if mod is None:
                return "module_not_perceived"
            z = float(self._f.worktop_z) + C.MODULE_SURFACES[cat]["z"]
            xy, height, min_tcp = s._look_xy(region, s.surface_centre(cat, mod)), 0.55, s._min_tcp_z(region)
        return [float(xy[0]), float(xy[1])], float(z), float(height), min_tcp

    def _composite(self, name, a):
        """The two composite looks. look_at_region(region, "unseen"): the aim of
        search.SiteSearch.unseen_aim, the scripted chain's own rule: nothing_unseen only when could_hide is False (the
        invariant), the T50 tilted look under the unit top for a drawer (SiteSearch.look_under_top), a straight-down or
        whole-module look for a surface. The aim and the pose are kept as the records-only key look (self._look)."""
        from ..robot.skills import SkillResult
        from ..search import plain_record
        s = self._search()
        if name == "look_down_at":
            r = s.look_down_at(list(a["xy"]), float(a["z_floor"]), height=float(a["height"]),
                               min_tcp_z=a.get("min_tcp_z"))
        elif a["part"] == "unseen":
            aim = self._look_point(a["region"], "unseen")
            self._look = plain_record(aim)
            if aim["pose"] is None:
                return SkillResult(False, aim["reason"], {"arm_moved": False})
            if aim["pose"].startswith("T"):
                lk = s.look_under_top(a["region"])
                self._look = plain_record({**aim, **{k: v for k, v in lk.items() if k not in ("aim", "region")}})
                r = lk["result"]
                if r is None:
                    return SkillResult(False, lk["reason"], {"arm_moved": False})
            else:
                r = s.look_down_at(aim["xy"], aim["z_floor"], height=aim["height_m"], min_tcp_z=aim["min_tcp_z"])
        else:
            lp = self._look_point(a["region"], a["part"])
            if isinstance(lp, str):
                return SkillResult(False, lp, {"arm_moved": False})
            xy, z, height, min_tcp = lp
            r = s.look_down_at(xy, z, height=height, min_tcp_z=min_tcp)
        return r if r is not None else SkillResult(False, "no_pose", {"arm_moved": False})

    def overhead_check(self, track_id, pos, kind: str | None = None, yaw: float | None = None) -> dict | None:
        """pick-pose-3: the robot-side overhead check of the tracked target at pos (Perception.overhead_check
        through search.overhead_reading, the scripted chain's own call: rule B active on the view that set the
        target's pose, rule F as RULE_F_ACTIVE says), plain values {refuse, code, n_points, rule, ...}; a check that
        raises or answers malformed reads as a refusal (overhead_unreadable). None when the perception has no such
        check (test doubles). A reading: no motion, not a skill call, not logged."""
        from ..search import overhead_reading
        if not callable(getattr(self._p, "overhead_check", None)):
            return None
        _refuse, rec = overhead_reading(self._p, self._p.tracks[track_id], pos, kind=kind, yaw=yaw)
        return rec

    def unit_top_clearance(self, track_id) -> float | None:
        """pick-pose-3: the clearance in m of the tracked target's centre in front of its drawer's unit top edge
        (the scripted chain's rule, search.top_clearance: the handle's closed view and the catalog), or None when the
        target is not in a drawer or the handle's closed position is unknown. A reading, robot side. With the
        unit-top check active, a perception that tracks no drawer handles (no handle_for_region: the test doubles;
        the robot's Perception always has it) reads None, unchecked, as a perception without an overhead check is; a
        reading that raises still refuses (GuardedHost._unit_top_clearance)."""
        from ..search import top_clearance
        t = self._p.tracks.get(track_id)
        if t is None or not isinstance(getattr(t, "region", None), str) or not t.region.startswith("drawer_unit:"):
            return None
        if not callable(getattr(self._p, "handle_for_region", None)):
            return None
        _h, clear = top_clearance(self._p, t)
        return None if clear is None else float(clear)

    def drawer_region_at(self, handle_pos) -> str | None:
        """The drawer region of the handle track the robot perceives within 5 cm of handle_pos (the same rule as
        the drawer bookkeeping in skill()), robot side; None when no handle is near or its drawer index is unknown."""
        h = self._nearest_track(("drawer_handle",), [float(x) for x in handle_pos])
        if h is None:
            return None
        rid = str(self._p.drawer_region_id(h))
        return rid if is_drawer_region(rid) else None

    def drawer_pull_bound(self, handle_pos) -> dict | None:
        """What is left of the travel of the drawer whose tracked handle is at handle_pos (5 cm),
        from the robot's own belief (perception.opening, catalog travel). None when no handle track is near or
        its opening is unknown (a failed pull or push): then nothing is bounded."""
        h = self._nearest_track(("drawer_handle",), handle_pos)
        if h is None:
            return None
        t = self._p.tracks[h]
        if bool(getattr(t, "opening_unknown", False)):
            return None
        op = float(self._p.opening(h))
        return {"track": h, "believed_opening_m": op, "left_m": DRAWER_TRAVEL_M - op}

    def skill(self, name, a):
        """One host skill on the facade. internal-actions-1: on EVERY return path, the exception path included,
        last_robot_calls (facade calls of this skill, at least 1) and last_call_detail are set from the facade's own
        call log (_account): {internal: the facade calls other than the gesture itself, unitemized: facade calls that
        could not be itemized, clearance: the drawer clearance's fate or None}. already_open makes no facade call (an
        empty slice, an empty detail)."""
        self.last_call_detail = {"internal": [], "unitemized": 0, "clearance": None}
        self._direct, self._clearance, self._drawer_note, self._close_check = [], None, None, None
        self._look = None  # the look record of a look_at_region (records only, _account)
        self._pull = None  # the open-full-1 record of an open_drawer (records only, _account)
        # handle-reref-1, bound-known-1 and tilt-retreat-1 records of this skill (records only, _account); the scripted
        # chain's look helpers keep the retreats of the composite looks (SiteSearch.retreats)
        self._reref = self._drawer_bound = None
        self._retreat = []
        lk = getattr(self, "_looker", None)
        self._retreat_n0 = len(lk.retreats) if isinstance(getattr(lk, "retreats", None), list) else 0
        n0, log0 = self._n_facade_calls(), self._facade_log()
        # parent: did the gesture itself reach the facade (for a facade without a call log); result: the gesture's own
        # result once its facade call returned; raised: the skill ends on an exception (_account)
        state = {"parent": False, "result": None, "raised": False}
        try:
            return self._skill(name, a, state)
        except BaseException:
            state["raised"] = True
            raise
        finally:
            self._account(name, n0, None if log0 is None else len(log0), state)

    def _account(self, name, n0, l0, state):
        """last_robot_calls and last_call_detail of the skill that just ended (or raised). With a facade call log,
        the slice log[l0:] is this skill's: for a plain gesture its first entry is the gesture and the rest the
        release clearance (origin ORIGIN_CLEARANCE); for a composite look every entry is one of its arm poses
        (ORIGIN_LOOK); unitemized = facade calls counted but not logged (a facade call that raised). Without a log
        (test doubles): the clearance's own results are itemized and unitemized = counted calls minus the gesture and
        those; a composite look is not itemized there and reports 0 (declared). When the skill raised AFTER its
        gesture's facade call returned (in the perception bookkeeping, the clearance, the data whitelist), the
        gesture's own result is kept as last_call_detail["gesture"] ({ok, reason, data}: the returned result, else
        the gesture's log entry), which GuardedHost._record_raised copies into the backend_raised record. Before, the
        slice counted that log entry as itemized while no record held it, so a collision or a contact scrub behind
        the exception was invisible to contact_incidents, grip_events and E2' with the reconciliation still at 0."""
        parent = state["parent"]
        delta = max(0, self._n_facade_calls() - n0)
        self.last_robot_calls = max(1, delta)  # nothing moved for already_open or a refused look: still 1
        composite = name in LOOK_SKILLS
        log = self._facade_log() if l0 is not None else None
        logged_gesture = None
        if log is not None and len(log) >= l0:
            sl = log[l0:]
            self._log_itemized += len(sl)
            gesture = not composite and sl and isinstance(sl[0], dict) and sl[0].get("skill") == name
            rest = sl[1:] if gesture else sl
            logged_gesture = sl[0] if gesture else None
            origin = ORIGIN_LOOK if composite else ORIGIN_CLEARANCE
            internal = [internal_entry(e.get("skill"), e.get("ok"), e.get("reason"), e.get("data"), e.get("t_s"),
                                       origin) for e in rest if isinstance(e, dict)]
            unitemized = max(0, delta - len(sl)) + sum(1 for e in rest if not isinstance(e, dict))
        else:
            internal = [internal_entry(n, getattr(r, "ok", None), getattr(r, "reason", None), getattr(r, "data", None),
                                       None, ORIGIN_CLEARANCE) for n, r in self._direct]
            unitemized = 0 if composite else max(0, delta - int(parent) - len(internal))
        self.last_call_detail = {"internal": internal, "unitemized": unitemized,
                                 "clearance": None if self._clearance is None else dict(self._clearance)}
        if getattr(self, "_drawer_note", None):
            self.last_call_detail["drawer_note"] = self._drawer_note
        if getattr(self, "_close_check", None):  # records only
            self.last_call_detail["close_check"] = dict(self._close_check)
        if isinstance(getattr(self, "_look", None), dict):  # look-unseen-2: records only
            self.last_call_detail["look"] = dict(self._look)
        if isinstance(getattr(self, "_pull", None), dict):  # open-full-1: records only
            self.last_call_detail["pull"] = dict(self._pull)
        # handle-reref-1, bound-known-1, tilt-retreat-1 (records only)
        if isinstance(getattr(self, "_reref", None), dict):
            self.last_call_detail["reref"] = dict(self._reref)
        if isinstance(getattr(self, "_drawer_bound", None), dict):
            self.last_call_detail["drawer_bound"] = dict(self._drawer_bound)
        ret = list(getattr(self, "_retreat", None) or [])
        lk = getattr(self, "_looker", None)
        if isinstance(getattr(lk, "retreats", None), list):
            ret += lk.retreats[getattr(self, "_retreat_n0", 0):]
        if ret:
            self.last_call_detail["retreat"] = [dict(e) for e in ret if isinstance(e, dict)]
        if state["raised"]:
            g = state["result"]
            if g is not None:
                g = {"ok": getattr(g, "ok", None), "reason": getattr(g, "reason", None),
                     "data": getattr(g, "data", None)}
            elif logged_gesture is not None:  # raised after the facade logged the gesture but before it returned
                g = {k: logged_gesture.get(k) for k in ("ok", "reason", "data")}
            if g is not None:
                self.last_call_detail["gesture"] = gesture_result(g)

    def _skill(self, name, a, state):
        import numpy as np  # robot side only
        from ..search import note_skill_reason, reason_detail, retreat_records  # robot side
        pre = None  # bound-known-1: (handle track, its believed opening before the gesture or None if unknown)
        tracks = getattr(getattr(self, "_p", None), "tracks", None)  # (a backend without a perception: as before)
        if name in ("open_drawer", "close_drawer") and tracks is not None:
            a = self._reref_handle(name, a)  # handle-reref-1 (the record keeps the program's argument)
            hb = self._nearest_track(("drawer_handle",), [float(x) for x in a["handle_pos"]])
            if hb is not None:
                if name == "open_drawer":
                    refused = self._bound_known_refusal(hb)  # bound-known-1: no motion
                    if refused is not None:
                        return refused
                t = self._p.tracks[hb]
                pre = (hb, None if bool(getattr(t, "opening_unknown", False)) else float(self._p.opening(hb)))
        if name == "open_drawer":
            bound = self.drawer_pull_bound(a["handle_pos"])
            if bound is not None and bound["left_m"] < OPEN_LEFT_MIN_M:
                # the robot believes this drawer fully open: no motion (scripted chain rule), not a contact
                # (nothing moved: perception and site model are not told of an action; last_robot_calls 1, _account)
                return ok_already_open(round(float(bound["believed_opening_m"]), 4))
            full = open_full_rewrite(a, bound) if open_full_active() else None
            if full is not None:  # open-full-1, K1 only: pull to the end stop
                a, self._pull = full
            elif bound is not None and a["distance"] > bound["left_m"]:
                dist = max(SKILL_MIN_PULL_M, float(bound["left_m"]))
                md = a.get("min_distance")
                a = {**a, "distance": dist, "min_distance": None if md is None else min(float(md), dist)}
        if name in ("look_down_at", "look_at_region"):
            r = self._composite(name, a)  # SiteSearch.look_down_at notes the action itself
        else:
            kw = {k: (np.array(v, dtype=float) if isinstance(v, list) else v) for k, v in a.items()}
            state["parent"] = True
            r = getattr(self._f, name)(**kw)
            state["result"] = r  # kept if the bookkeeping below raises (_account)
            self._fix_state("_retreat", list).extend(retreat_records(r))  # tilt-retreat-1 (records only)
            if isinstance(getattr(self, "_pull", None), dict):  # open-full-1: records only
                mv = r.data.get("moved")
                self._pull.update(ok=bool(r.ok), reason=str(r.reason),
                                  moved=round(float(mv), 4) if isinstance(mv, (int, float)) and not isinstance(mv, bool)
                                  else None)
            # Perception bookkeeping after the robot's own actions (same rule as the scripted chain).
            if name in ("open_drawer", "close_drawer"):
                h = self._nearest_track(("drawer_handle",), a["handle_pos"])
                if h is not None and drawer_gesture_moved(bool(r.ok), str(r.reason), r.data or {}):
                    self._fix_state("_moved_since_query", set).add(h)  # handle-reref-1: moved since the query
                if not r.ok and r.data.get("arm_moved") is False:
                    # Refused before any motion (hand_busy while the robot holds, bad_arg, budget_exceeded): no
                    # clearance, no drawer note, no action note. The drawer's opening stays known, its contents and
                    # region are not made stale, and no holding=True action reaches the site model (its global
                    # invalidation of cleared regions); a missing arm_moved counts as moved.
                    self._clearance = {"ran": False, "skipped": "refused_before_motion"}
                elif h is None:
                    self._p.note_action(name, {"ok": bool(r.ok), **reason_detail(r)})  # contact reason
                    if not r.ok:  # recorded only; the 5 cm handle gate is unchanged
                        self._clearance = {"ran": False, "skipped": "no_tracked_handle_near"}
                elif r.ok and name == "open_drawer":
                    self._p.note_drawer_motion(h, float(r.data.get("moved", 0.0)), "open_drawer")
                    self._drawer_note = "open_drawer"
                elif r.ok and close_push_opposes_pull(a.get("push_dir"), self._p.tracks[h].pull):
                    # drawer-booking-2: an end stop felt after a travel short of the robot's own need (an obstruction)
                    # is not a proof of closing; the result the program receives is unchanged
                    t = self._p.tracks[h]
                    was_unknown = bool(getattr(t, "opening_unknown", False))
                    chk = close_travel_check(self._p.opening(h), getattr(t, "closed_pos", None), t.pull, a, r.data,
                                             was_unknown)
                    residual = chk.pop("residual_raw_m")
                    self._close_check = chk
                    if not chk["sufficient"]:
                        # the residual stays out, opening unknown (estimate: a *_failed note), until a view measures it
                        self._p.note_drawer_motion(h, residual - self._p.opening(h), "close_drawer_failed")
                        self._drawer_note = "close_drawer_short"
                    else:
                        self._p.note_drawer_motion(h, -self._p.opening(h), "close_drawer")
                        # The unknown flag survives a close (estimate.note_drawer_motion) and the record does not
                        # restart drawer_unmoved_since_last_ok (believed was only a lower bound)
                        self._drawer_note = "close_drawer_unverified" if was_unknown else "close_drawer"
                elif r.ok:
                    # robot.skills does not check push_dir; a push_dir that does not oppose the
                    # handle's pull pulls the drawer out and the end stop there reads as closed (recorded once: ok,
                    # moved 0.249, the drawer 0.240 m out in the robot's next view).
                    # Not booked closed: the hand's travel along the pull, opening unknown.
                    cos = close_push_opposes_pull(a.get("push_dir"), self._p.tracks[h].pull, cosine=True)
                    moved = float(r.data.get("moved", 0.0) or 0.0)
                    self._p.note_drawer_motion(h, max(0.0, moved * cos), "close_drawer_failed")
                    self._drawer_note = "close_drawer_not_opposed"
                else:
                    moved = float(r.data.get("moved", 0.0) or 0.0)
                    delta = 0.0 if name == "open_drawer" or moved <= 0.005 else -min(moved, self._p.opening(h))
                    self._p.note_drawer_motion(h, delta, name + "_failed")
                    note_skill_reason(self._p, r)  # contact-invalidates-1: before the release's own note
                    self._drawer_note = name + "_failed"
                    self._release()
            elif name == "pick":
                self._p.note_pick(self._nearest_track((a["kind"],), a["pos"]), bool(r.ok))
                note_skill_reason(self._p, r)  # a pick invalidates anyway; the reason is recorded
            else:
                self._p.note_action(name, {"ok": bool(r.ok), **reason_detail(r)})  # contact reason
        data = {}  # whitelist; data["error"] (exception text for internal_error) is dropped: the reason code stays
        for k in _SKILL_DATA_KEYS:
            v = r.data.get(k)
            if isinstance(v, bool):  # numpy scalars are cast to plain Python values
                data[k] = bool(v)
            elif isinstance(v, int):
                data[k] = int(v)
            elif isinstance(v, float):
                data[k] = float(v)
            elif isinstance(v, str):
                data[k] = str(v)
        if pre is not None:  # bound-known-1, on the result as the program receives it
            self._book_no_travel(pre, bool(r.ok), str(r.reason), data)
        return {"ok": bool(r.ok), "reason": str(r.reason), "data": data}

    # ------------------------------------------------------------ drawer gestures
    def _fix_state(self, attr: str, kind):
        """The backend's state of the corrections (set in __init__; created here for a backend built otherwise)."""
        v = getattr(self, attr, None)
        if not isinstance(v, kind):
            v = kind()
            setattr(self, attr, v)
        return v

    def _reref_handle(self, name, a):
        """handle-reref-1: before a
        close_drawer / open_drawer, when the handle track the 5 cm gate selects (nearest to the program's handle_pos)
        belongs to a drawer the robot moved since the program's last handle_for_region returned that track, and the
        track's believed position lies more than DRAWER_MOVED_MIN_M (horizontally) from handle_pos, the gesture is
        commanded at the track's x, y (the program's z kept). Not when the track's opening is unknown (after a failed
        pull or push its position is a guess): the program's handle, as live (a declared narrowing that
        keeps the drawer-booking-2 scenarios). The host record keeps the program's argument; the change is
        recorded (records-only key reref). Robot side only: the robot's own handle track and its own gesture records."""
        import numpy as np  # robot side only
        pos = [float(x) for x in a["handle_pos"]]
        h = self._nearest_track(("drawer_handle",), pos)
        if h is None or h not in self._fix_state("_moved_since_query", set):
            return a
        if bool(getattr(self._p.tracks[h], "opening_unknown", False)):
            return a  # the robot does not know where that handle is: the program's own handle, as live (declared)
        tp = np.asarray(self._p.tracks[h].pos, float)
        d = float(np.linalg.norm(tp[:2] - np.asarray(pos[:2], float)))
        if d <= DRAWER_MOVED_MIN_M:
            return a
        self._reref = {"rule": DRAWER_REREF_VERSION, "skill": name, "track": str(h),
                       "from": [round(x, 4) for x in pos], "to": [round(float(tp[0]), 4), round(float(tp[1]), 4),
                                                                 round(pos[2], 4)],
                       "d_m": round(d, 4), "pose_source": getattr(self._p.tracks[h], "pose_source", None)}
        return {**a, "handle_pos": [float(tp[0]), float(tp[1]), pos[2]]}

    def _bound_known_refusal(self, h) -> dict | None:
        """bound-known-1 ("boundknown"): an open_drawer on a handle whose
        opening is unknown only after a failed drawer gesture that reported no travel (_book_no_travel) reads the
        lower bound (the believed opening before that gesture minus DRAWER_MOVED_MIN_M); when the travel left above
        it is under OPEN_LEFT_MIN_M, the live already_open rule answers (no motion, believed_opening_m the bound).
        Otherwise nothing changes (the opening stays unknown, the request is not bounded). Records only:
        drawer_bound."""
        t = self._p.tracks[h]
        notravel = self._fix_state("_notravel", dict)
        if not bool(getattr(t, "opening_unknown", False)) or h not in notravel:
            return None
        op = float(notravel[h]) - DRAWER_MOVED_MIN_M
        left = DRAWER_TRAVEL_M - op
        refuse = left < OPEN_LEFT_MIN_M
        self._drawer_bound = {"rule": DRAWER_BOUND_VERSION, "track": str(h), "bound_opening_lower_m": round(op, 4),
                              "bound_left_upper_m": round(left, 4), "bound_already_open": refuse}
        return ok_already_open(round(op, 4)) if refuse else None

    def _book_no_travel(self, pre, ok: bool, reason: str, data: dict):
        """bound-known-1, after an open_drawer / close_drawer that reached the backend (not already_open): a failure whose
        reported travel ("moved") is at most DRAWER_MOVED_MIN_M, that is no contact incident, on a handle whose opening
        was known before it, leaves that opening as the bound's base; any other result except already_open clears it.
        Perception is not told (its opening stays unknown as before)."""
        h, known_before = pre
        mv = data.get("moved")
        notravel = self._fix_state("_notravel", dict)
        rec = getattr(self, "_drawer_bound", None)
        rec = rec if isinstance(rec, dict) else {"rule": DRAWER_BOUND_VERSION, "track": str(h)}
        if (ok is False and known_before is not None and isinstance(mv, (int, float)) and not isinstance(mv, bool)
                and abs(float(mv)) <= DRAWER_MOVED_MIN_M and not is_contact_incident(reason, data)):
            notravel[h] = float(known_before)
            rec.update(marked=True, mark_reason=reason, mark_moved=round(float(mv), 4),
                       mark_opening_before_m=round(float(known_before), 4))
            self._drawer_bound = rec
        elif reason != "already_open" and notravel.pop(h, None) is not None:
            rec.update(cleared=True, clear_reason=reason)
            self._drawer_bound = rec

    def robot_state(self) -> dict:
        """Robot-side state for the prompt (0.2): the grip reading (the same reading as the holding() query,
        not counted), the number of drawer handles it perceives, and the drawers it believes open (opening of
        OPEN_BELIEF_M or more from its own drawer motions, perception.opening: the handle offset from where it
        was first seen, assumed closed then; or opening_unknown after a failed pull or push). Only drawers
        the robot itself moved can appear: no region is named before the robot acted on it. Moves nothing,
        counts nothing."""
        handles = sorted((t for t in self._p.tracks.values() if t.category == "drawer_handle"), key=lambda x: x.track_id)
        open_ = []
        for t in handles:
            op = float(self._p.opening(t.track_id))
            unknown = bool(getattr(t, "opening_unknown", False))
            if unknown or op >= OPEN_BELIEF_M:
                open_.append({"region": str(self._p.drawer_region_id(t.track_id)), "believed_opening_m": round(op, 3),
                              "opening_unknown": unknown})
        return {"holding": bool(self._f.holding()), "drawer_handles_perceived": len(handles),
                "drawers_believed_open": open_}

    # ------------------------------------------------------------ SCENE STATE detail
    def search_detail(self, category: str = "sponge") -> dict:
        """Host only, NOT an API name (a program
        that calls it gets unknown_name from GuardedHost.call). Reads the site model and the perception read only, as
        plain values: {version, category, regions: {region: {views, best_low_coverage, last_obs, last_step,
        invalidated_by, frame_accepted, places, n_open, edge_unknown} | {error}}} for every region the site model
        knows. views = len(inspections); best_low_coverage = the best coverage among the inspections below
        sitemodel.MIN_COVER (None without one); last_obs / last_step = the region's last record (the view the scope
        reads); invalidated_by = {action, reason, step, holding} of the first action after it at a step of
        SiteModel._invalidating_steps() (the method is called, the rule is not re-implemented: contacts included);
        frame_accepted = sitemodel.frame_accepted on the last record's placement evidence (surfaces only, else None);
        places = the classes of sitemodel.open_placements(last) (the set placement_could_hide tests), robot side:
        drawer: under the unit top (behind the unit-top edge, the handle's closed position minus
        catalog.DRAWER_TOP_EDGE_BEHIND_CLOSED_HANDLE along the pull, the scripted chain's top_clearance rule), else near
        the front wall (within half the category's catalog length of the front of the drawer floor,
        catalog.DRAWER_FLOOR_BEHIND_HANDLE[0] behind the handle as it stood at that view: the handle track minus the
        robot's own drawer motions noted since), else in the open part; a drawer whose closed handle position is
        unknown: in the open part, edge_unknown True; surface: on the surface. Not counted, not recorded, moves
        nothing; never the simulator, the sensor or the truth; Perception.handle_for_region (pure), never the
        handle_for_region query (handle-reref-1)."""
        from ..perception import catalog as C  # noqa: PLC0415 (robot side)
        from ..sitemodel import MIN_COVER, frame_accepted, open_placements  # noqa: PLC0415 (robot side)
        m = self._m
        inv = sorted({s for s in m._invalidating_steps() if type(s) is int})
        surfaces = set(C.SURFACE_REGION.values())
        regions = {}
        for rid, reg in m.regions.items():
            try:
                regions[str(rid)] = self._region_detail(str(rid), reg, inv, surfaces, category, MIN_COVER,
                                                        frame_accepted, open_placements, C)
            except Exception as e:  # noqa: BLE001 (a region that cannot be read: a marker, never a clearance)
                regions[str(rid)] = {"error": type(e).__name__[:60]}
        return {"version": SCENE_STATE_VERSION, "category": str(category), "regions": regions}

    def _region_detail(self, rid, reg, inv, surfaces, category, min_cover, frame_accepted, open_placements, C) -> dict:
        ins = [r for r in reg.get("inspections") or [] if isinstance(r, dict)]
        low = [float(r["coverage"]) for r in ins if _finite_float(r.get("coverage")) is not None
               and float(r["coverage"]) < min_cover]
        last = reg.get("last") if isinstance(reg.get("last"), dict) else None
        d = {"views": len(ins), "best_low_coverage": round(max(low), 4) if low else None,
             "last_obs": None if last is None else str(last.get("obs_id")), "last_step": None,
             "invalidated_by": None, "frame_accepted": None, "places": [], "n_open": 0, "edge_unknown": False}
        if last is None:
            return d
        s0 = last.get("step")
        d["last_step"] = int(s0) if type(s0) is int else None
        after = [s for s in inv if type(s0) is int and s > s0]
        if after:
            d["invalidated_by"] = self._invalidating_action(after[0])
        ev = last.get("placement")
        if rid in surfaces:
            fa = frame_accepted(ev)
            d["frame_accepted"] = None if fa is None else bool(fa)
        if isinstance(ev, dict) and ev.get("category") == category:
            xy = open_placements(last)
            d["n_open"] = int(len(xy))
            if len(xy):
                if rid in surfaces:
                    d["places"] = ["on the surface"]
                elif is_drawer_region(rid):
                    d["places"], d["edge_unknown"] = self._drawer_place_classes(rid, xy, s0, category, C)
        return d

    def _invalidating_action(self, step: int) -> dict:
        """The action of the site model at an invalidating step: the first action of that step that
        SiteModel._invalidating_steps() itself counts (the method is called on that one action), else the first."""
        acts = [a for a in self._m.actions if isinstance(a, dict) and a.get("step") == step]
        method = getattr(type(self._m), "_invalidating_steps", None)

        class _One:  # the site model's own rule, applied to a single action
            def __init__(self, a):
                self.actions = [a]
        a = None
        for x in acts:
            try:
                if callable(method) and method(_One(x)):
                    a = x
                    break
            except Exception:  # noqa: BLE001 (an unreadable action: not this one)
                continue
        a = a if a is not None else (acts[0] if acts else {})
        reason = a.get("reason")
        return {"action": str(a.get("action") or "action")[:SCENE_CODE_MAX], "step": int(step),
                "reason": reason[:SCENE_CODE_MAX] if isinstance(reason, str) and reason else None,
                "holding": bool(a.get("holding"))}

    def _drawer_place_classes(self, rid, xy, s0, category, C) -> tuple[list[str], bool]:
        """(classes in SCENE_PLACES order, edge_unknown) of a drawer's open placements xy (world, as the view put
        them)."""
        import numpy as np  # robot side only
        h = self._p.handle_for_region(rid)
        if h is None or getattr(h, "closed_pos", None) is None:
            return ["in the open part"], True
        pull = np.asarray(h.pull[:2], float)
        n = float(np.linalg.norm(pull))
        if not n > 1e-9:
            return ["in the open part"], True
        pull = pull / n
        shift = 0.0  # the robot's own drawer motions noted on this drawer after the view (note_drawer_motion)
        for a in self._m.actions:
            if isinstance(a, dict) and a.get("region") == rid and type(a.get("step")) is int and a["step"] > s0 \
                    and str(a.get("action", "")).startswith(("open_drawer", "close_drawer")):
                mv = _finite_float(a.get("moved_m"))
                shift += 0.0 if mv is None else mv
        edge = np.asarray(h.closed_pos[:2], float) - C.DRAWER_TOP_EDGE_BEHIND_CLOSED_HANDLE * pull
        front = np.asarray(h.pos[:2], float) - shift * pull - C.DRAWER_FLOOR_BEHIND_HANDLE[0] * pull
        shape = C.OBJECT_SHAPES.get(category) or C.OBJECT_SHAPES["sponge"]
        half = 0.5 * max(shape["footprint"])
        P = np.asarray(xy, float).reshape(-1, 2)
        under = (P - edge) @ pull < 0.0
        near = ~under & ((front - P) @ pull <= half)
        rest = ~under & ~near
        return [c for c, mk in zip(SCENE_PLACES[:3], (under, near, rest)) if bool(mk.any())], False

    def observe(self, camera):
        o = plain_observation(self._to_dict(self._p.observe(camera)))
        self.observation_log.append(o)
        return o

    def query(self, name, a):
        if name == "tracks":
            return [{"track_id": t.track_id, "category": t.category, "status": t.status, "region": t.region,
                     "position_m": [float(x) for x in t.pos], "sigma_m": [float(x) for x in t.sigma],
                     "yaw_rad": None if t.yaw is None else float(t.yaw), "pose_source": t.pose_source,
                     "pose_step": None if t.pose_step is None else int(t.pose_step)}
                    for t in self._p.tracks_of(a["category"], ("observed", "stale", "held"))]
        if name == "handle_for_region":
            t = self._p.handle_for_region(a["region"])
            if t is None:
                return None
            # handle-reref-1: the program now holds this track's believed position
            getattr(self, "_moved_since_query", set()).discard(t.track_id)
            return {"track_id": t.track_id, "position_m": [float(x) for x in t.pos],
                    "sigma_m": [float(x) for x in t.sigma], "pull_dir": [float(x) for x in t.pull],
                    "opening_m": float(self._p.opening(t.track_id)),
                    # robot-side view quality (the scripted chain's refine rule uses the same two;
                    # visibility estimate from pixels, never the oracle fraction)
                    "truncated": bool(t.truncated), "visibility_estimate": float(t.vis_estimate)}
        if name == "search_scope":
            return self._m.search_scope(a["category"])
        if name == "difference_report":
            return self._m.diff_report()
        if name == "dossier":
            return self._dossier.to_dict()
        if name == "holding":
            return bool(self._f.holding())
        if name == "worktop_z":
            return float(self._f.worktop_z)
        if name == "base_xy":
            return [float(x) for x in self._f.belief.t[:2]]
        raise ArgError(f"unknown query {name}")
