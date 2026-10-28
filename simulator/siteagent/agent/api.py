"""Skill API offered to robot programs: names, arguments, argument checks and the text that describes them.

A program can call ONLY the names listed here (plus the pure builtins of agent.interp). Each name is one of:
- "skill": a robot gesture of the skill facade (programs.facade), same name and argument ranges as robot.skills;
  counted against the program's and the episode's skill-call limits, with the simulated-time limit checked first;
- "observe": a capture by a simulated camera, counted against max_observations;
- "query": a robot-side reading that returns plain data (tracks, handle of a region, site knowledge, dossier,
  budget, holding, worktop height, believed base position);
- "terminal": ends the program with the program's own claim (the evaluator decides, never the program).

Arguments are checked here, before the host is asked to do anything: types (3-vectors of finite numbers, numbers in
the skill's own range, enumerations) and no other value kind. Nothing in this module touches the robot, and it
imports nothing of the robot code (the sandboxed child imports it).

The S5 skill profile keeps only the gestures the task needs: the looks (look_pose, look_down_at, look_at_region),
the drawers (open_drawer, close_drawer), pick, scrub, the fingers (open_gripper, close_gripper), go_home and
calibrate. Removed names (move_to, place, turn_over, press, spray) are refused by the language, by the host and by
the sandbox parent. look_down_at stays a program-directed motion within the API ranges.

Several versions of the API description are kept because recorded requests were built with them and their sha256
values are pinned (describe, describe_v0_7, describe_v0_9). API_V0_9 has the same names, kinds and parameters as API;
only the documentation lines differ, and it is reached only through the information profile s5-info-1.0. Shared
phrases (RULE_*, CLAIM_TEXT, SCENE_LEGEND_1_0, SCOPE_NOTE_1_0) appear word for word in the system prompt and in the
description; the host's SCENE STATE renderer imports them from here. CLAIM_TEXT is generated from CLAIM_PARAMS, a
mirror of the perception and robot constants it states. Every model-visible text is ASCII.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

SKILL_API_VERSION = "skill-api-s5-0.8"  # a sponge pick without yaw: perceived long axis + pi/2
SKILL_API_VERSION_V0_7 = "skill-api-s5-0.7"  # kept: earlier recorded requests name it
SKILL_API_VERSION_V0_6 = "skill-api-s5-0.6"  # kept: earlier recorded requests name it
SKILL_API_VERSION_V0_5 = "skill-api-s5-0.5"
SKILL_API_VERSION_V0_4 = "skill-api-s5-0.4"
# API versions. Recorded requests name the version they were built with, so the older tables stay loadable.
# - 0.6: move_to is no longer offered to programs: the language refuses it (unknown_function), the host refuses the
#   name (unknown_name) and the sandbox parent stops a child that asks for it (call_not_in_api). Direct move_to calls
#   caused collisions; the facade's own gestures (looks, pick, scrub, drawers) still move the arm internally. Stored
#   programs of an older API are never reused (agent.memory filters on this version).
#   help(why, kind): kind is required, written as a word of HELP_KINDS in quotes; agent.lang refuses every certain
#   argument error of help(). A reached help() always ends the program: a why that is not readable at run time becomes
#   HELP_WHY_UNREADABLE (check_help), never a runtime error, so a stop can never turn into a correction and more
#   gestures. Only search_incomplete can be reviewed (agent.loop.help_review_trigger), also when it ends a stored
#   program; every other kind is final. not_found("inspected_scope") is accepted only when the robot's own
#   search_scope("sponge") says not_found_in_inspected_scope at that moment (agent.loop.not_found_guard);
#   not_found("scope_incomplete") claims no absence. scrub's line says the gesture brings the held sponge onto
#   target_pos itself. The 0.5 entries are kept below (MOVE_TO_ENTRY_V0_5, HELP_DOC_V0_5, SCRUB_DOC_V0_5,
#   NOT_FOUND_DOC_V0_5) because API_DESCRIPTION_V0_5_SHA256 pins the 0.5 description.
# - 0.5: only the help() line changed (a help without kind could be reviewed once while the robot had not perceived the
#   sponge). The 0.4 help text is HELP_DOC_V0_4 and its description hash API_DESCRIPTION_V0_4_SHA256.
# - 0.4: help(why, kind) with a structured reason travelling apart from the claim (interpreter outcome "help_kind",
#   checked by the sandbox parent); a pick of a category with a pose policy (agent.host.PICK_POSE_POLICY) may return
#   ok False without any motion (agent.host.PICK_REFUSAL_CODES), also when pos is not the accepted estimate
#   (pick_target_offset).
# - 0.3: open_drawer on a drawer the robot believes fully open returns ok True with reason already_open and no motion
#   (agent.host.ALREADY_OPEN_CONVENTION); the why text of help / give_up is cleaned instead of refused (any character
#   other than a letter, a digit, space, _ - . becomes a space, spaces collapse, cut to 80 characters), because a why
#   holding ":" or a quote failed at run time after the robot had already acted. An empty text is still refused.
# - 0.2: two composite looks shared with the scripted chain (look_down_at, look_at_region, the same rule as
#   search.SiteSearch); observe() regions carry could_hide per category (a partial view is not absence);
#   handle_for_region() returns sigma_m and the robot-side view quality.
TWO_PI = 2 * math.pi
# Public catalog words (perception.catalog.OBJECT_CATEGORIES; a test checks they match).
OBJECT_CATEGORIES = ("cup", "bowl", "fork", "spoon", "sponge", "spray_bottle", "towel")
OTHER_CATEGORIES = ("drawer_handle", "dishwasher_handle", "sink", "dishwasher", "drawer_unit", "landing_tray",
                    "drying_mat", "bins")
CAMERAS = ("robot_eye", "wrist_cam")
# 0.4: structured reason of help(). search_incomplete = the target was not found and plausible places remain (reviewed
# once by the loop); safety, perception, cannot_continue, other = final. 0.5: a help WITHOUT kind (outcome help_kind
# "other") is reviewed once while the robot has not perceived the sponge (agent.loop.help_review_trigger). 0.6: kind
# required; no help without kind can run; the 0.5 review of a help without kind is removed.
HELP_KINDS = ("search_incomplete", "safety", "perception", "cannot_continue", "other")
MAX_COORD_M = 3.0  # sanity bound on any position given to a skill (the site is about 2 m across)


class ArgError(ValueError):
    """An argument does not satisfy the API (the call is not made)."""


@dataclass(frozen=True)
class P:
    name: str
    type: str  # vec3 | vec2 | num | bool | enum | text | orient | region | track_id
    required: bool = True
    default: object = None
    lo: float | None = None
    hi: float | None = None
    choices: tuple = ()
    nullable: bool = False
    doc: str = ""
    fallback: object = None  # enum only: the value an unknown choice is read as, instead of an ArgError


def _e(kind, params, returns, doc):
    return {"kind": kind, "params": tuple(params), "returns": returns, "doc": doc}


SKILL_RESULT = "{ok: bool, reason: str, data: {scalar robot-side readings, e.g. moved, width, force_mean}}"
# 0.6: the texts of the lines that changed (the 0.5 texts are kept below). Rules shared word for word with SYSTEM 0.8
# (agent.prompt; a test checks both carry them).
RULE_HELP_KIND = "kind is required"
# The review applies to a stored program's search_incomplete too; a kind is a word in quotes;
# a reached help always ends the program; the guard covers the absence claim only.
RULE_REVIEW_V0_6 = ("search_incomplete, from a model or a stored program, is reviewed once per episode while the "
                    "robot's own search_scope lists regions not cleared")  # 0.6 text, RULE_REVIEW below
RULE_FINAL = "every other help is final, also when it ends a stored program"
RULE_KIND_WORD = "written as one of these words in quotes"
RULE_HELP_ENDS = "a help() that is reached always ends the program"
RULE_NOT_FOUND = "accepted only when the robot's own search_scope says not_found_in_inspected_scope"
RULE_SCRUB = "the gesture itself brings the held sponge onto target_pos"
HELP_WHY_UNREADABLE = "reason_unreadable"  # the why kept when a reached help's why is not a readable text
# 0.6 texts of the lines 0.7 changed, verbatim (describe() of api_v0_6() hashes to API_DESCRIPTION_V0_6_SHA256)
SCRUB_DOC_V0_6 = ("scrub the surface at target_pos with the held sponge; the gesture itself brings the held sponge onto "
                  "target_pos (target_pos = the top of the bowl, from its perceived position_m) under its preconditions: "
                  "call it right after a pick that returned ok, no positioning step is needed")
HELP_DOC_V0_6 = ("ask for help (claim 'help_requested:<why>'); kind is required, written as one of these words in quotes: "
                 "search_incomplete (the target was not found and plausible places remain), safety, perception, "
                 "cannot_continue or other; a program with a help() without kind, with another kind, or with other "
                 "arguments than why and kind is refused before it runs; a help() that is reached always ends the "
                 "program (a why that is not a readable text is kept as reason_unreadable); search_incomplete, from a "
                 "model or a stored program, is reviewed once per episode while the robot's own search_scope lists "
                 "regions not cleared, no contact or grip event, the budget allows one more look and no pick was refused "
                 "in that program; every other help is final, also when it ends a stored program: no new program, no "
                 "motion after it; a second help is final")
NOT_FOUND_DOC_V0_6 = ("claim the target was not found ('not_found_in_<scope>'); not_found(\"inspected_scope\") claims an "
                      "absence: accepted only when the robot's own search_scope says not_found_in_inspected_scope at "
                      "that moment, else refused (harness_rule not_found_refused_scope_incomplete, regions not cleared "
                      "listed): a third refusal, or a refusal after a contact or grip event or with no look left, ends "
                      "the episode as help_requested:not_found_unsupported; not_found(\"scope_incomplete\") claims no "
                      "absence, ends the episode and is never reviewed; not seen is not absent")
OBSERVE_DOC_V0_6 = ("capture with a simulated camera; counted against max_observations. could_hide[c] is false only when no "
                    "unseen part of that region could hold an object of category c (not seen is not absent)")
# 0.7: rules shared word for word with SYSTEM 0.9 (agent.prompt; a test checks both carry them). The review
# (agent.loop.help_review_trigger) also fires on a perceived sponge not picked, and the active limits
# (agent-limits-s5-0.8) set no number of reviews or of not_found refusals: a help of another kind stops.
RULE_REVIEW = ("search_incomplete, from a model or a stored program, is reviewed while the robot's own search_scope "
               "lists regions not cleared or the robot perceives a sponge it has not picked")
RULE_STOP_KIND = "to stop, write a help of another kind: it stands"
RULE_PICK_PERCEIVED = "a perceived sponge should be picked (after a top view) before asking for help about the search"
RULE_NOT_FOUND_STOP = ("a refusal after a contact or grip event or with no look left ends the episode as "
                       "help_requested:not_found_unsupported")
RULE_BOWL_POSE = 'the bowl pose from robot_eye or tracks("bowl") is enough for scrub'
RULE_WRIST_AIM = ('observe("wrist_cam") sees where the last gesture left the hand: aim it first with look_at_region or '
                  "look_down_at")
# The first 0.7 text, 'a wrist view of the bowl needs look_at_region("landing_tray") first', pointed to a pattern
# that failed in recorded episodes: an aimed wrist view taken while the hand held the sponge showed no bowl.
RULE_BOWL_WRIST = ('a wrist view while the hand holds the sponge may not show the bowl, even after look_at_region: take '
                   'the bowl pose from robot_eye or tracks("bowl")')
# The progress precondition of limits 0.8 (agent.loop.reprise_has_progress)
RULE_NEW_OBSERVATION = ("after a first review of search_incomplete or refusal of not_found, the next one needs a new "
                        "observation since the previous one")
RULE_PROFILE = "no other gesture exists in this skill profile, and each gesture moves the arm itself"
SCRUB_DOC = (f"scrub the surface at target_pos with the held sponge; {RULE_SCRUB} (target_pos = the top of the bowl, "
             "from its perceived position_m) under its preconditions: call it right after a pick that returned ok, "
             f"no positioning step is needed; {RULE_BOWL_POSE}")
HELP_DOC = (f"ask for help (claim 'help_requested:<why>'); {RULE_HELP_KIND}, {RULE_KIND_WORD}: search_incomplete "
            "(the target was not found and plausible places remain), safety, perception, cannot_continue or other; a "
            "program with a help() without kind, with another kind, or with other arguments than why and kind is "
            f"refused before it runs; {RULE_HELP_ENDS} (a why that is not a readable text is kept as "
            f"{HELP_WHY_UNREADABLE}); {RULE_REVIEW}, no contact or grip event, the budget allows one more look and no "
            f"pick was refused in that program; {RULE_NEW_OBSERVATION}, else the help stands; {RULE_PICK_PERCEIVED}; "
            f"{RULE_STOP_KIND}; {RULE_FINAL}: no new program, no motion after it")
NOT_FOUND_DOC = ("claim the target was not found ('not_found_in_<scope>'); not_found(\"inspected_scope\") claims an "
                 f"absence: {RULE_NOT_FOUND} at that moment, else refused (harness_rule "
                 "not_found_refused_scope_incomplete, regions not cleared listed) and a corrected program may continue; "
                 f"{RULE_NOT_FOUND_STOP}, and so does a refusal without a new observation ({RULE_NEW_OBSERVATION}); "
                 "not_found(\"scope_incomplete\") claims no absence, ends the episode and is never reviewed; not seen is "
                 "not absent")
OBSERVE_DOC = (f"{OBSERVE_DOC_V0_6}; {RULE_WRIST_AIM}; {RULE_BOWL_WRIST}")
# 0.8: only the pick line changed. A sponge pick accepted by the pick precondition without a yaw takes yaw = the
# accepted track's yaw_rad + pi/2 (agent.host.GuardedHost.call, pick_check.yaw_filled_from_track); a yaw written by the
# program is never changed. PICK_DOC_V0_7 is the pick line of 0.4 to 0.7, verbatim (api_v0_7() and every older table
# use it; describe_v0_7() hashes to API_DESCRIPTION_V0_7_SHA256). The 0.8 text says that yaw_rad is the LONG axis
# (perception contract: long axis of an elongated object, mod pi) and that the robot adds pi/2: read literally,
# pick(..., yaw=d["yaw_rad"]) closes the fingers along the long axis, and such picks collided on approach in
# simulation (median pick force about 270 N).
RULE_PICK_YAW = ("for a sponge leave yaw out: the robot closes its fingers across the long axis it perceives, at "
                 "yaw_rad + pi/2; yaw_rad is a long axis, never a finger axis")
PICK_DOC_V0_7 = ("grasp an object of category kind at pos (its centre); yaw = finger closing axis. For a sponge, or at a "
                 "perceived sponge, the robot first checks its own perception: one perceived sponge within 3 cm of pos, "
                 "pos within 1 cm horizontally and 2 cm vertically of its position_m, horizontal sigma_m at most 0.01 m, "
                 "measured in a view taken after the last action that could move it; else ok False, no motion, reason "
                 "pick_target_not_perceived, pick_target_ambiguous, pick_pose_uncertain, pick_pose_stale or "
                 "pick_target_offset")
PICK_DOC = (f"grasp an object of category kind at pos (its centre); yaw = finger closing axis ({RULE_PICK_YAW}). For a "
            "sponge, or at a perceived sponge, the robot first checks its own perception: one perceived sponge within "
            "3 cm of pos, pos within 1 cm horizontally and 2 cm vertically of its position_m, horizontal sigma_m at "
            "most 0.01 m, measured in a view taken after the last action that could move it; else ok False, no motion, "
            "reason pick_target_not_perceived, pick_target_ambiguous, pick_pose_uncertain, pick_pose_stale or "
            "pick_target_offset")
API = {
    # ------------------------------------------------------------- skills (skill facade)
    "calibrate": _e("skill", [], SKILL_RESULT, "touch the worktop edges and correct the believed base pose"),
    "go_home": _e("skill", [], SKILL_RESULT, "arm to its home posture"),
    "look_pose": _e("skill", [], SKILL_RESULT, "arm out of the mast camera's view"),
    "look_down_at": _e("skill", [P("xy", "vec2"), P("z_floor", "num", lo=0.0, hi=2.0),
                                 P("height", "num", False, 0.42, 0.2, 0.8),
                                 P("min_tcp_z", "num", False, None, 0.0, 2.5, nullable=True)],
                       SKILL_RESULT, "move the hand so the wrist camera looks straight down at the floor point (xy, "
                                     "z_floor) from `height` above it (several arm angles are tried; a collision stops "
                                     "it; each arm motion counts as a skill call); then call observe('wrist_cam')"),
    "look_at_region": _e("skill", [P("region", "region"), P("part", "enum", False, "centre", choices=("centre", "unseen"))],
                         SKILL_RESULT, "look_down_at the floor of a region, computed robot side: a drawer's interior "
                                       "from its perceived handle (open the drawer first), or a surface (sink_basin, "
                                       "landing_tray, drying_mat) from the perceived module; part 'unseen' aims at "
                                       "the floor the latest view of that region did not reach; then observe('wrist_cam')"),
    "open_gripper": _e("skill", [P("width", "num", False, 0.08, 0.0, 0.08)], SKILL_RESULT, "open the fingers"),
    "close_gripper": _e("skill", [P("width", "num", False, 0.0, 0.0, 0.08)], SKILL_RESULT, "close the fingers"),
    "pick": _e("skill", [P("pos", "vec3"), P("kind", "enum", choices=OBJECT_CATEGORIES),
                         P("grasp", "enum", False, "top", choices=("top", "side")),
                         P("yaw", "num", False, None, -TWO_PI, TWO_PI, nullable=True)],
               SKILL_RESULT, PICK_DOC),  # 0.8: + the yaw fill (PICK_DOC_V0_7 = the 0.4 to 0.7 line)
    # Skill API 0.7: place and turn_over are not in the S5 profile (PLACE_ENTRY_V0_6, TURN_OVER_ENTRY_V0_6)
    "open_drawer": _e("skill", [P("handle_pos", "vec3"), P("pull_dir", "vec3"),
                                P("distance", "num", False, 0.22, 0.02, 0.30),
                                P("min_distance", "num", False, None, 0.0, 0.30, nullable=True)],
                      SKILL_RESULT, "grip a drawer's rail handle and pull it along pull_dir; data.moved = hand travel (m)"),
    "close_drawer": _e("skill", [P("handle_pos", "vec3"), P("push_dir", "vec3"),
                                 P("expected", "num", False, None, 0.0, 0.30, nullable=True),
                                 P("max_distance", "num", False, 0.28, 0.02, 0.30)],
                       SKILL_RESULT, "push a drawer closed"),
    # Skill API 0.7: press and spray are not in the S5 profile (PRESS_ENTRY_V0_6, SPRAY_ENTRY_V0_6)
    "scrub": _e("skill", [P("target_pos", "vec3"), P("seconds", "num", False, 3.0, 0.2, 30.0),
                          P("force", "num", False, 4.0, 0.5, 15.0)],
                SKILL_RESULT, SCRUB_DOC),
    # ------------------------------------------------------------- observation (counted)
    "observe": _e("observe", [P("camera", "enum", choices=CAMERAS)],
                  "{obs_id, camera, step, t_s, detections: [{track_id, category, region, position_m, sigma_m, yaw_rad, "
                  "pose_source, pose_step, n_pixels}], handles: [{track_id, category, position_m, pull_dir}], "
                  "regions: [{region, coverage, seen_categories, could_hide: {category: true|false|null}}]}",
                  OBSERVE_DOC),  # + where a wrist view looks
    # ------------------------------------------------------------- robot-side readings
    "tracks": _e("query", [P("category", "enum", choices=OBJECT_CATEGORIES + OTHER_CATEGORIES)],
                 "[{track_id, category, status, region, position_m, sigma_m, yaw_rad, pose_source, pose_step}]",
                 "what the robot currently believes about things of this category (observed, stale or held)"),
    "handle_for_region": _e("query", [P("region", "region")],
                            "{track_id, position_m, sigma_m, pull_dir, opening_m, truncated, visibility_estimate} or None",
                            "the perceived handle of a drawer region ('drawer_unit:k', k as in the dossier)"),
    "search_scope": _e("query", [P("category", "enum", choices=OBJECT_CATEGORIES)],
                       "{outcome, found_tracks, scope: [{region, status, coverage, seen, unseen_part_could_hide_it, "
                       "cleared}], not_cleared}", "the site model's account of where this category was looked for"),
    "difference_report": _e("query", [], "{entries: [...], n_differences_or_unknowns}",
                            "differences between the dossier and the robot's own observations"),
    "dossier": _e("query", [], "{version, worktop_height_m, modules, stored_items, conventions}",
                  "the site dossier given to the robot (may be out of date)"),
    "budget": _e("query", [], "{sim_s_used, max_sim_s, skill_calls, max_skill_calls, observations, max_observations}",
                 "the episode budget as the robot counts it"),
    "holding": _e("query", [], "bool", "whether the gripper holds something"),
    "worktop_z": _e("query", [], "number (m)", "worktop height from the dossier"),
    "base_xy": _e("query", [], "[x, y] (m)", "where the robot believes its base is"),
    # ------------------------------------------------------------- terminals
    "done": _e("terminal", [], "ends the program", "claim the task is done (claim 'task_done')"),
    "help": _e("terminal", [P("why", "text"), P("kind", "enum", choices=HELP_KINDS)], "ends the program", HELP_DOC),
    "give_up": _e("terminal", [P("why", "text")], "ends the program", "stop on a failure (claim 'failed:<why>')"),
    "not_found": _e("terminal", [P("scope", "enum", choices=("inspected_scope", "scope_incomplete"))],
                    "ends the program", NOT_FOUND_DOC),
}
# 0.5 entries, for the record: describe_api(api_v0_5(), SKILL_API_VERSION_V0_5) hashes to API_DESCRIPTION_V0_5_SHA256 (=
# manifest v4_3_super api_description_sha256).
MOVE_TO_ENTRY_V0_5 = _e("skill", [P("pos", "vec3"), P("orient", "orient", False, "down"),
                                  P("speed", "num", False, 0.25, 0.01, 0.5), P("transit", "bool", False, False)],
                        SKILL_RESULT, "move the gripper centre to pos (world_belief frame, m); orient 'down' or a yaw (rad)")
HELP_PARAMS_V0_5 = (P("why", "text"), P("kind", "enum", False, "other", choices=HELP_KINDS, fallback="other"))
HELP_DOC_V0_5 = ("ask for help (claim 'help_requested:<why>'); kind search_incomplete only when the target was not found "
                 "and plausible places remain; without kind: reviewed once like search_incomplete while the robot has "
                 "not perceived the sponge in this episode, plausible places remain not cleared, the budget allows one "
                 "more look and no pick was refused in that program; safety, perception, cannot_continue or other "
                 "written in the call: final; any other kind is read as other (final); a second help is final")
SCRUB_DOC_V0_5 = "scrub the surface at target_pos with the held sponge (target_pos = top surface)"
NOT_FOUND_DOC_V0_5 = ("claim the target was not found ('not_found_in_<scope>'); only for the inspected scope: not seen "
                      "is not absent")
API_DESCRIPTION_V0_5_SHA256 = "c3f75de5838bf8a4726f0a071b14c023e74f1929863ecdef2e66cd33e6c3a0d6"


# 0.6 entries removed from the S5 profile in 0.7, verbatim, and the 0.6 order of the table (for the record only).
PLACE_ENTRY_V0_6 = _e("skill", [P("pos", "vec3"), P("orientation", "enum", False, "upright",
                                                    choices=("upright", "upside_down")),
                                P("yaw", "num", False, None, -TWO_PI, TWO_PI, nullable=True)],
                      SKILL_RESULT, "put the held object down at pos")
TURN_OVER_ENTRY_V0_6 = _e("skill", [P("regrasp", "bool", False, True)], SKILL_RESULT, "turn the held object upside down")
PRESS_ENTRY_V0_6 = _e("skill", [P("button_pos", "vec3")], SKILL_RESULT, "press a button")
SPRAY_ENTRY_V0_6 = _e("skill", [P("target_pos", "vec3"), P("seconds", "num", False, 2.0, 0.1, 20.0),
                                P("standoff", "num", False, 0.30, 0.10, 0.60),
                                P("tilt_deg", "num", False, 50.0, 10.0, 80.0)],
                      SKILL_RESULT, "spray toward target_pos with the held spray bottle")
REMOVED_FROM_S5_PROFILE = ("place", "turn_over", "press", "spray")  # (0.7); move_to went in 0.6
API_DESCRIPTION_V0_6_SHA256 = "8a9235fbbbd5771cb448548927797f3a73f85813f8a06c358068dbda1148ba6e"
# describe_v0_7(): the 0.7 description, kept because earlier recorded requests were built with it
API_DESCRIPTION_V0_7_SHA256 = "ecb45dde679c37e4871bbf62254d9d0fe8cfeea6a44bfd928596f769ee2021ce"


def api_v0_7() -> dict:
    """The 0.7 API (for the record only: never used to check or run a program). The current
    table with the 0.7 pick line (PICK_DOC_V0_7); the same names, arguments and order."""
    return {n: ({**e, "doc": PICK_DOC_V0_7} if n == "pick" else e) for n, e in API.items()}


def api_v0_6() -> dict:
    """The 0.6 API (for the record only: never used to check or run a program). Built on the
    0.7 table (the pick line of 0.6 is PICK_DOC_V0_7)."""
    out = {}
    for n, e in api_v0_7().items():
        if n == "scrub":
            e = {**e, "doc": SCRUB_DOC_V0_6}
        elif n == "help":
            e = {**e, "doc": HELP_DOC_V0_6}
        elif n == "not_found":
            e = {**e, "doc": NOT_FOUND_DOC_V0_6}
        elif n == "observe":
            e = {**e, "doc": OBSERVE_DOC_V0_6}
        if n == "close_drawer":
            out[n] = e
            out["press"], out["spray"] = PRESS_ENTRY_V0_6, SPRAY_ENTRY_V0_6
            continue
        out[n] = e
        if n == "pick":
            out["place"], out["turn_over"] = PLACE_ENTRY_V0_6, TURN_OVER_ENTRY_V0_6
    return out


def api_v0_5() -> dict:
    """The 0.5 API (for the record only: never used to check or run a program). Built on the 0.6 table."""
    out = {}
    for n, e in api_v0_6().items():
        if n == "scrub":
            e = {**e, "doc": SCRUB_DOC_V0_5}
        elif n == "help":
            e = {**e, "params": HELP_PARAMS_V0_5, "doc": HELP_DOC_V0_5}
        elif n == "not_found":
            e = {**e, "doc": NOT_FOUND_DOC_V0_5}
        out[n] = e
        if n == "look_at_region":
            out["move_to"] = MOVE_TO_ENTRY_V0_5
    return out


# 0.4 help() text, for the record; describe() with it and SKILL_API_VERSION_V0_4 hashes to API_DESCRIPTION_V0_4_SHA256
HELP_DOC_V0_4 = ("ask for help (claim 'help_requested:<why>'); kind search_incomplete only when the target was not "
                 "found and plausible places remain; safety, perception, cannot_continue or other (default): final; "
                 "any other kind is read as other")
API_DESCRIPTION_V0_4_SHA256 = "d613a825405bb6c8d5f21f49b24d3ec9bd4213ca6de2fda0412f40f188990f21"
SKILLS = tuple(k for k, v in API.items() if v["kind"] == "skill")
COMPOSITE_SKILLS = ("look_down_at", "look_at_region")  # robot side, built on the facade (agent.host.FacadeBackend)
TERMINALS = tuple(k for k, v in API.items() if v["kind"] == "terminal")


MAX_INT_ABS = 10 ** 15  # same bound as ProgramLimits.max_int_abs; a larger int is refused before float()


def _float(name: str, v) -> float:
    """float(v) for a checked int or float; a huge int (only a hostile child can send one) is an
    ArgError, never an OverflowError."""
    if type(v) is int and abs(v) > MAX_INT_ABS:
        raise ArgError(f"{name}: integer too large")
    try:
        return float(v)
    except OverflowError:  # pragma: no cover (guarded above)
        raise ArgError(f"{name}: number too large") from None


def _num(p: P, v):
    if v is None:
        if p.nullable:
            return None
        raise ArgError(f"{p.name} must be a number, got None")
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ArgError(f"{p.name} must be a number, got {type(v).__name__}")
    x = _float(p.name, v)
    if not math.isfinite(x):
        raise ArgError(f"{p.name} must be finite")
    if (p.lo is not None and x < p.lo) or (p.hi is not None and x > p.hi):
        raise ArgError(f"{p.name} must be in [{p.lo}, {p.hi}], got {x:g}")
    return x


def _vec(p: P, v, n):
    if not isinstance(v, list) or len(v) != n:
        raise ArgError(f"{p.name} must be a list of {n} numbers")
    out = []
    for x in v:
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(_float(p.name, x)):
            raise ArgError(f"{p.name} must be a list of {n} finite numbers")
        if abs(float(x)) > MAX_COORD_M:
            raise ArgError(f"{p.name}: |coordinate| must be <= {MAX_COORD_M} m")
        out.append(float(x))
    return out


TEXT_EXTRA_CHARS = "_-. "
MAX_RAW_TEXT_CHARS = 400  # 0.3: longer raw texts are refused (bounded work); the kept text is at most 80 characters


def clean_text(s: str) -> str:
    """0.3: a help / give_up why as kept in the claim: other characters become spaces, spaces collapse, 80 max."""
    kept = "".join(c if (c.isalnum() or c in TEXT_EXTRA_CHARS) else " " for c in s)
    out = " ".join(kept.split())[:80].strip()
    return out if any(c.isalnum() for c in out) else ""  # at least one letter or digit (punctuation alone is refused)


def text_bad_chars(s: str) -> str:
    """Characters of s that a 'text' argument (help/give_up why) does not accept (same rule as _check)."""
    return "".join(c for c in s if not (c.isalnum() or c in TEXT_EXTRA_CHARS))


def _check(p: P, v):
    t = p.type
    if t == "vec3":
        return _vec(p, v, 3)
    if t == "vec2":
        return _vec(p, v, 2)
    if t == "num":
        return _num(p, v)
    if t == "bool":
        if not isinstance(v, bool):
            raise ArgError(f"{p.name} must be True or False")
        return v
    if t == "enum":
        if not isinstance(v, str) or v not in p.choices:
            if p.fallback is not None:
                return p.fallback  # help kind: a terminal must always end the program
            raise ArgError(f"{p.name} must be one of {list(p.choices)}")
        return v
    if t == "orient":
        if v == "down":
            return v
        return _num(P(p.name, "num", lo=-TWO_PI, hi=TWO_PI), v)
    if t == "text":
        cleaned = clean_text(v) if isinstance(v, str) and len(v) <= MAX_RAW_TEXT_CHARS else ""
        if not cleaned:
            raise ArgError(f"{p.name} must be a text with at least one letter or digit (at most {MAX_RAW_TEXT_CHARS} "
                           "characters before cleaning)")
        return cleaned.replace(" ", "_")
    if t == "region":
        ok = isinstance(v, str) and (v in ("sink_basin", "landing_tray", "drying_mat") or (
            v.startswith("drawer_unit:") and v[len("drawer_unit:"):].isdigit() and len(v) <= 16))
        if not ok:
            raise ArgError(f"{p.name} must be a region id such as 'drawer_unit:k' (k a number) or 'sink_basin'")
        return v
    raise ArgError(f"unknown parameter type {t}")  # pragma: no cover (spec error)


def check_args(name: str, args: list, kwargs: dict) -> dict:
    """Map positional and keyword arguments onto the API parameters, check each, fill defaults.
    Raises ArgError (no call is made)."""
    if name not in API:
        raise ArgError(f"{name!r} is not in the skill API")
    params = API[name]["params"]
    if len(args) > len(params):
        raise ArgError(f"{name}() takes at most {len(params)} arguments, got {len(args)}")
    names = [p.name for p in params]
    got = dict(zip(names, args))
    for k, v in kwargs.items():
        if k not in names:
            raise ArgError(f"{name}() has no argument {k!r}")
        if k in got:
            raise ArgError(f"{name}() got {k!r} twice")
        got[k] = v
    out = {}
    for p in params:
        if p.name in got:
            out[p.name] = _check(p, got[p.name])
        elif p.required:
            raise ArgError(f"{name}() is missing argument {p.name!r}")
        else:
            out[p.name] = p.default
    return out


def check_help(why, kind) -> tuple[dict, str | None]:
    """help()'s arguments once the call is
    reached (agent.interp). The kind is checked without fallback (ArgError outside HELP_KINDS: only when agent.lang was
    bypassed, it refuses any kind not written as a word of HELP_KINDS). A why that is not a readable text (not a str,
    nothing left after cleaning, longer than MAX_RAW_TEXT_CHARS, or None when it could not be computed) never fails the
    help: it becomes HELP_WHY_UNREADABLE. Returns (checked arguments, why it was replaced or None)."""
    params = {p.name: p for p in API["help"]["params"]}
    k = _check(params["kind"], kind)
    try:
        return {"why": _check(params["why"], why), "kind": k}, None
    except ArgError as x:
        return {"why": HELP_WHY_UNREADABLE, "kind": k}, str(x)[:160]


def describe() -> str:
    """Plain-text API description for the prompt (stable order). 0.7: ends with the S5 skill profile line."""
    return describe_api(API, SKILL_API_VERSION) + _profile_line(API)


def describe_v0_7() -> str:
    """The 0.7 description as sent in earlier recorded requests (hashes to API_DESCRIPTION_V0_7_SHA256);
    for the record only."""
    return describe_api(api_v0_7(), SKILL_API_VERSION_V0_7) + _profile_line(api_v0_7())


def _profile_line(api: dict) -> str:
    """0.7: the S5 skill profile line that describe() appends (the skills of the given table)."""
    skills = tuple(k for k, v in api.items() if v["kind"] == "skill")
    return (f"\n\nSkill profile S5 (find the sponge, pick it, scrub the bowl): the robot skills above are its gestures "
            f"({', '.join(skills)}); {RULE_PROFILE}.")


def describe_api(api: dict, version: str) -> str:
    """describe() of a given API table and version (0.6: so that a frozen earlier table can be hashed again; 0.7: without
    the profile line, which describe() adds for the current version only)."""
    lines = [f"Skill API {version}. Positions in metres, world_belief frame (the site frame as the robot "
             "believes it), z up. Every call returns plain data."]
    for kind, title in (("skill", "Robot skills (each counted; the time budget is checked first)"),
                        ("observe", "Observation (counted against max_observations)"),
                        ("query", "Robot-side readings (no motion)"),
                        ("terminal", "Terminals (end the program)")):
        lines.append(f"\n{title}:")
        for n, e in api.items():
            if e["kind"] != kind:
                continue
            ps = []
            for p in e["params"]:
                s = p.name
                if p.type in ("num",) and p.lo is not None:
                    s += f" in [{p.lo:g},{p.hi:g}]"
                elif p.type == "enum":
                    s += " in {" + ",".join(p.choices) + "}"
                elif p.type in ("vec3", "vec2"):
                    s += f" ({p.type})"
                if not p.required:
                    s += f"={p.default!r}"
                ps.append(s)
            lines.append(f"- {n}({', '.join(ps)}) -> {e['returns']}: {e['doc']}")
    return "\n".join(lines)


# ====================================================================================================== API 0.9
# Profile s5-info-1.0 only (agent.prompt.INFO_PROFILES); the defaults above are unchanged.
SKILL_API_VERSION_V0_9 = "skill-api-s5-0.9"
# Shared phrases, word for word (whitespace-normalised) in SYSTEM 1.0 and describe_v0_9().
RULE_DRAWER_HAND = "drawer gestures need an empty hand: while holding they return hand_busy and nothing moves"
RULE_HANDLE_MOVES = "the handle moves with the drawer: read handle_for_region again after each drawer motion"
RULE_PUSH_DIR = "push_dir = vscale(pull_dir, -1)"
RULE_LOOK_OBSERVE = 'look_at_region only aims the hand: nothing is seen until observe("wrist_cam")'
RULE_SURFACES = "the open surfaces the SITE DOSSIER lists"
RULE_CLOSE_OPTIONAL = "closing a drawer is not required by this task"
RULE_CLEARED = "not found in the part inspected, not absent"
RULE_CLAIM_INTRO = 'For the sponge, not_found("inspected_scope") means: '
# The per-view could_hide of observe() is agent.host.region_could_hide, i.e. sitemodel.could_hide on a perception region
# record that carries no placement evidence: the floor grid of that view alone (sitemodel module docstring)
RULE_GRID_VIEW = "could_hide in observe() is the floor grid of that view alone"
# Fixed for configuration K1; the host renderer imports these two and never redefines them
SCENE_LEGEND_1_0 = ("legend: cleared means not found in the part inspected, not absent: every place where the sponge "
                    "could lie flat or stand on its long edge was shown since the robot's last pick, scrub, action while "
                    "holding something or gesture that reported a contact; the floor under an object resting on it is not "
                    "checked, under a raised object (towel, board) it is")
SCOPE_NOTE_1_0 = ("not seen is not absent: the statement holds for the inspected scope only, and only for a sponge of the "
                  "catalog size (9 x 6 x 3.6 cm) lying flat or standing on its long edge, as the robot measured each region")

# The K1 claim text. CLAIM_PARAMS mirrors the constants the text states; a test checks each against its source:
# perception.estimate PLACEMENT_MIN_PX, PLACEMENT_MIN_FRAC, PLACEMENT_MARGIN_UPRIGHT, SURFACE_PLACEMENT_MARGIN_M,
# DRAWER_DEEP_END_ACTIVE, DRAWER_DEEP_END_M; perception.catalog OBJECT_SHAPES["sponge"] (footprint and height);
# robot.skills LINK_FORCE_LIMIT and HAND_FORCE_LIMIT. Variants of the text were written in advance for the declared
# alternatives of these constants (a margin of 1.5 reads "50 % more", a margin of 4 cm reads "4 cm", an active deep end
# adds a sentence after "a drawer from its tracked handle"), each with a golden text in the test. A change of a source
# constant fails that test until its mirrored value here is changed, which selects the pre-written variant. Only K1 is
# written: a K2 fallback (estimate.SCOPE_CONFIGURATION "K2" or agent.host.OPEN_FULL_ACTIVE False) fails the pairing
# test and needs a declared text revision.
CLAIM_CONFIGURATION = "K1"
CLAIM_PARAMS = {"configuration": CLAIM_CONFIGURATION, "min_px": 40, "min_frac": 0.15, "margin_upright": 1.25,
                "surface_margin_m": 0.03, "deep_end_active": False, "deep_end_m": 0.020,
                "sponge_m": (0.09, 0.06, 0.036), "link_force_n": 20.0, "hand_force_n": 60.0}
# The robot actions whose invalidation the claim names, as sitemodel words: INVALIDATING_ACTIONS (pick, place, turn_over,
# scrub, spray, press), any action while holding, and a contact reason (sitemodel.CONTACT_INVALIDATING_REASONS); T10
# checks the list equals sitemodel's.
CLAIM_INVALIDATIONS = ("pick", "place", "turn-over", "scrub", "spray", "press", "action while holding something",
                       "gesture that reported a contact")


def _claim_num(x) -> str:
    """A number as the claim text writes it: at most 2 decimals, no trailing zero (3.6, 9, 15, 25)."""
    return f"{round(float(x), 2):.2f}".rstrip("0").rstrip(".")


def claim_text(params: dict | None = None) -> str:
    """The K1 claim text for the given parameters (CLAIM_PARAMS by default). Pure; ASCII."""
    p = dict(CLAIM_PARAMS if params is None else params)
    if p.get("configuration") != "K1":
        raise ValueError("only the K1 claim text is written: a K2 configuration needs a declared text revision")
    size = " x ".join(_claim_num(100 * float(v)) for v in p["sponge_m"])
    since = ", ".join(CLAIM_INVALIDATIONS[:-1]) + ", or " + CLAIM_INVALIDATIONS[-1]
    deep = (f" (places at the back of a drawer count as shown only if they would also show "
            f"{_claim_num(1000 * float(p['deep_end_m']))} mm deeper)") if p["deep_end_active"] else ""
    return ("Not found in the inspected scope (simulation, simulated perception). In every candidate region of the site "
            f"dossier, the robot's views taken since its last {since}, showed every place where a catalog sponge "
            f"({size} cm) could lie flat or stand on its long edge. \"Showed\" means at the recogniser's own thresholds: "
            f"at least {_claim_num(p['min_px'])} pixels and {_claim_num(100 * float(p['min_frac']))} % of the sponge's "
            "outline, counted on its top for a sponge lying flat, and on its top and the sides facing the camera, with "
            f"{_claim_num(100 * (float(p['margin_upright']) - 1.0))} % more, for a sponge on its edge. A region is its "
            f"floor as the robot measured it: a drawer from its tracked handle{deep}; a sink, tray or mat from the module "
            "outline fitted to the robot's own depth images, extended by "
            f"{_claim_num(100 * float(p['surface_margin_m']))} cm on every side. A surface whose outline could not be "
            "fitted is never cleared. Not covered: a sponge under or on an object that the robot saw resting on that "
            "floor; a sponge in another pose (leaning, crushed); a sponge moved by a contact too weak to trigger the "
            f"robot's reflex ({_claim_num(p['link_force_n'])} N on the arm, {_claim_num(p['hand_force_n'])} N on the "
            "hand); a sponge outside the dossier's candidate regions.")


CLAIM_TEXT = claim_text()

# 0.9 documentation lines (configuration K1)
LOOK_AT_REGION_DOC_1_0 = (
    "look_down_at the floor of a region, computed robot side: a drawer's interior from its perceived handle (open the "
    f"drawer first), or one of {RULE_SURFACES}, from the perceived module; part 'unseen' looks at the places of that "
    "region its views have not shown yet: on a drawer a tilted look under the unit top's edge (straight down when that "
    "edge is unknown), on a sink, tray or mat straight down at them, or the whole module while its outline is not "
    "fitted yet; nothing_unseen means no such place is left in the region's latest view (if search_scope still lists "
    "the region as not cleared, look_at_region(region) and observe(\"wrist_cam\") again); "
    f"{RULE_LOOK_OBSERVE}. Other refusals without motion: no_earlier_view_of_region (no view of it yet: look at its "
    "centre first), handle_not_perceived or module_not_perceived (observe(\"robot_eye\") first); unreachable or "
    "no_path: no arm pose reached the look")
# "it opens and closes the fingers itself" and "after no_grasp_handle, observe(...) and read handle_for_region again
# before any retry" removed here: SYSTEM 1.0 DRAWERS states both, for both drawer gestures
OPEN_DRAWER_DOC_1_0 = (
    "grip a drawer's rail handle and pull it along pull_dir; data.moved = hand travel (m); while the robot knows the "
    "drawer's opening, 0.20 or more pulls it to its end stop (about 0.24 m) and less pulls that far; with "
    f"opening_unknown the pull is not bounded and can reach the end stop; {RULE_DRAWER_HAND}")
# RULE_DRAWER_HAND (already in the open_drawer line, and "drawer gestures" in SYSTEM 1.0 DRAWERS) and "handle_pos from
# handle_for_region read after that drawer's last motion" (= RULE_HANDLE_MOVES, in the handle_for_region line and SYSTEM
# 1.0 DRAWERS, next to "to close one, push_dir = ...") removed here
CLOSE_DRAWER_DOC_1_0 = f"push a drawer closed; {RULE_CLOSE_OPTIONAL}; {RULE_PUSH_DIR}"
HANDLE_DOC_1_0 = f"the perceived handle of a drawer region ('drawer_unit:k', k as in the dossier); {RULE_HANDLE_MOVES}"
# Declared: the phrase "it changes only after observe()" is not true of the robot's tracks (a drawer
# motion makes the tracks inside stale and moves them with the drawer, a pick makes the sponge's track held:
# perception.estimate note_drawer_motion / note_pick); what holds is that only observe() adds detections
TRACKS_DOC_1_0 = ("what the robot currently believes about things of this category (observed, stale or held); new "
                  "detections come only from observe() (a look adds none), and the robot's own gestures can make a "
                  "track stale or held")
SEARCH_SCOPE_RETURNS_1_0 = ("{category, outcome, found_tracks: [{track_id, status, region}], scope: [{region, status, "
                            "coverage, step, seen, unseen_part_could_hide_it, evidence_invalidated_by_later_action, "
                            "cleared}], min_coverage, not_cleared, not_inspected_outside_scope, note}")
# The "({RULE_GRID_VIEW})" of the search_scope line removed (the observe line states it, next to "search_scope merges
# the views and is the reading that clears a region")
SEARCH_SCOPE_DOC_1_0 = ("the robot's merged account of where this category was looked for, the reading its not_found "
                        "check uses; for the sponge, unseen_part_could_hide_it means some place where it could lie flat "
                        f"or stand on its long edge was not shown; cleared = {RULE_CLEARED}")
# The second full copy of the claim text: the not_found line names the claim and points to its one full copy, SYSTEM 1.0
# TERMINALS, where the same words RULE_CLAIM_INTRO are followed by CLAIM_TEXT
CLAIM_POINTER_1_0 = "the claim stated after these same words in TERMINALS"
NOT_FOUND_DOC_1_0 = f"{NOT_FOUND_DOC}. {RULE_CLAIM_INTRO}{CLAIM_POINTER_1_0}"
OBSERVE_DOC_1_0 = ("capture with a simulated camera; counted against max_observations. could_hide[c] is false only when "
                   "no unseen part of that region could hold an object of category c (not seen is not absent; "
                   f"{RULE_GRID_VIEW}; search_scope merges the views and is the reading that clears a region); "
                   f"{RULE_WRIST_AIM}; {RULE_BOWL_WRIST}")
# The meanings of the two new refusals and "do not retry the same pick" removed here (SYSTEM 1.0 PICK PRECONDITION
# states them); the line keeps both codes and points there
PICK_DOC_1_0 = f"{PICK_DOC}; then pick_under_unit_top or pick_overhead_obstruction (see PICK PRECONDITION)"
_DOCS_1_0 = {"look_at_region": LOOK_AT_REGION_DOC_1_0, "open_drawer": OPEN_DRAWER_DOC_1_0,
             "close_drawer": CLOSE_DRAWER_DOC_1_0, "handle_for_region": HANDLE_DOC_1_0, "tracks": TRACKS_DOC_1_0,
             "search_scope": SEARCH_SCOPE_DOC_1_0, "not_found": NOT_FOUND_DOC_1_0, "observe": OBSERVE_DOC_1_0,
             "pick": PICK_DOC_1_0}
_RETURNS_1_0 = {"search_scope": SEARCH_SCOPE_RETURNS_1_0}
# Same names, order, kinds and parameter objects as API; only doc and returns differ (tested).
API_V0_9 = {n: {**e, "doc": _DOCS_1_0.get(n, e["doc"]), "returns": _RETURNS_1_0.get(n, e["returns"])}
            for n, e in API.items()}


def describe_v0_9() -> str:
    """Profile s5-info-1.0: the 0.9 description (API_V0_9, SKILL_API_VERSION_V0_9) with the S5 skill profile
    line. Its sha256 is computed, not pinned, until the freeze."""
    return describe_api(API_V0_9, SKILL_API_VERSION_V0_9) + _profile_line(API_V0_9)
