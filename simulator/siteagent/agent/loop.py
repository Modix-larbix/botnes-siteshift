"""The agent loop: observe, prompt, parse, check, run, correct, remember.

One episode:
0. an optional prelude run by the loop (calibrate, look pose), the same for every attempt;
1. an optional orientation view taken by the loop itself (counted like any observation);
2. memory: if the site library holds a program for this dossier, skill API and criteria that no evidence of the
   episode contradicts, run it without a model call, in reuse mode (the host checks each critical precondition at the
   moment of use, agent.memory.precondition);
3. otherwise ask the model client (prompt.build_messages: observable inputs only), parse its answer, then refuse or
   run the program (sandbox.run_isolated by default, in process for tests);
4. on an observable failure (parse error, refusal, runtime error, limit, give_up) and while the limits allow it, ask
   for a correction with the previous program and its observable run report.

The loop never sees the evaluator: it returns the program's own claim and, for the library, assumptions and
robot-side evidence of the final run. An isolated run requires the OS sandbox by default (require_os_sandbox): without
sandbox-exec the episode is refused before any model call. Every limit is checked before the step it bounds, and the
episode's call count, committed cost and wall time are checked before every reservation, technical retry included.

Harness rules (code, not prompt), the same for model and stored programs:
- done() is refused when the robot felt a contact incident earlier in the episode, when its own records do not prove
  the task (agent.host.task_evidence: a checked sponge pick, a scrub of the target bowl after it, the hold continuous
  since the pick), or when the robot's own budget reading says the episode is over. A final task_done always needs a
  proven task_evidence verdict (end-of-episode invariant);
- not_found guard: a not_found claim stands only when the robot's own search scope, read at that moment, says
  not_found_in_inspected_scope; otherwise it is refused, and a later reprise must bring a new observation;
- help review: a help(kind="search_incomplete") is reviewed (the program is asked to continue) only while the robot's
  budget allows one more look and regions are still not cleared, or when the robot perceives a sponge it has not
  picked; the other help kinds (safety, perception, cannot_continue, other) are final;
- the episode ends, with no further reservation, model call or program run, when its physical budget is used up
  (simulated time or skill calls: episode-physical-end-1), when its reading budget is used up
  (episode-query-budget-end-1), or when a run is stopped again by an operation already known to be impossible and
  brought no new observation and no useful state change (episode-non-progress-end-1). Each end has its own status
  and a claim that asserts neither success nor absence.

Every proposal, refusal and execution is kept in the episode record, failures included. Under the information
profile s5-info-1.0 every request also carries the robot's SCENE STATE text (agent.host.render_scene_state).
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import time
from decimal import Decimal
from pathlib import Path

from tf_guardrails import GuardrailError, prompt_token_bound

from .api import HELP_KINDS, HELP_WHY_UNREADABLE, SKILL_API_VERSION  # noqa: F401 (the profile gives the versions)
from .host import (DRAWER_BOOKING_VERSION, INTERNAL_ACTIONS_VERSION, PHYSICAL_END_RULE, PICK_REFUSED_STATUS,
                   RAN_STATUSES, TASK_EVIDENCE_VERSION, WORST_ROBOT_CALLS, EpisodeCounters, GuardedHost,
                   contact_incidents, delivered_obs_id, grip_events, wrist_view_notes)
from .interp import Interpreter, utf8_safe
from .lang import Refused, compile_program
from .ledger import RESERVATION_USD, RESERVED_PROMPT_TOKENS, BudgetStop, _entry_cost
from .limits import Limits, load_limits, reprise_limit, reprise_needs_new_observation
from .llm import (MODEL_ID, ClientNotReady, LLMRefused, LLMResponse, TransientLLMError, build_request,
                  request_fingerprint)
from .memory import (LIBRARY_SCHEMA, STORE_CONTRACT, ProgramLibrary, check, contract_views, extract_assumptions,
                     robot_side_evidence, store_contract_fields)
from .prompt import (INFO_PROFILE_DEFAULT, PROMPT_VERSION, LeakError, ResponseError, build_messages,  # noqa: F401
                     info_profile_of, parse_response, run_report)
from .sandbox import child_command, run_isolated


class SandboxUnavailable(RuntimeError):
    """An isolated run was required but the OS sandbox layer is missing: nothing is started."""


DONE_REFUSED = "done_refused"
DONE_REFUSED_CLAIM = "help_requested:done_refused_contact_incident"
DONE_REFUSED_GRIP_CLAIM = "help_requested:done_refused_grip_event"  # a drop or a release of the sponge
# done() without positive robot-side evidence (agent.host.task_evidence, TASK_EVIDENCE_VERSION). The claims keep the
# help_requested: prefix (taskcheck: search_unsuccessful), so earlier analyses read them as before.
DONE_REFUSED_TASK_CODE = "done_refused_task_not_proven"
DONE_REFUSED_HOLD_CODE = "done_refused_hold_not_proven"
DONE_REFUSED_TASK_CLAIM = "help_requested:" + DONE_REFUSED_TASK_CODE
DONE_REFUSED_HOLD_CLAIM = "help_requested:" + DONE_REFUSED_HOLD_CODE
# done() claimed after the robot's own episode budget ran out
# (agent.host.GuardedHost.robot_budget_exceeded; refuse_done_over_budget). taskcheck classes such an episode
# budget_exceeded whatever the claim, so the done() could only be a false success
DONE_REFUSED_BUDGET_CODE = "done_refused_episode_budget_exceeded"
DONE_REFUSED_BUDGET_CLAIM = "help_requested:" + DONE_REFUSED_BUDGET_CODE
# the task_evidence codes about the hold only (the episode reconciliation): a refusal whose codes are all here is
# done_refused_hold_not_proven; any other missing or void element (the pick, the target scrub) is the task's.
# "backend_raised" (a skill whose backend raised, agent.host) is a reconciliation code
HOLD_EVIDENCE_CODES = ("second_pick_after_sponge_pick", "grip_not_felt_after_gesture", "holding_reading_false",
                       "unitemized_robot_calls", "unrecorded_robot_calls", "hand_empty_at_done", "grip_unreadable",
                       "backend_raised")
HARNESS_INVARIANT_RULE = "final_task_done_needs_proven_task_evidence"  # end-of-episode invariant (run_episode)
# See the module docstring: the episode's physical end is final. The stop rule is the host's
# (agent.host.PHYSICAL_END_RULE, GuardedHost.physical_end); 
STOP_RULE_VERSION = PHYSICAL_END_RULE  # "episode-physical-end-1"
PHYSICAL_END_STATUS = "stopped_physical_end"  # a new status (never stopped_budget, which stops a real stage)
PHYSICAL_END_CLAIM = "help_requested:episode_physical_budget_exhausted"  # taskcheck: search_unsuccessful, not money
PHYSICAL_END_REASON = "episode_physical_end"  # the final's reason is "episode_physical_end:<code>"
# where the loop recognised the end (rec["physical_end"]["recognised_at"]); "at_final": an episode that ended otherwise
# (a reached terminal, a final refusal) while the end holds, records only
PHYSICAL_END_CHECKPOINTS = ("before_reservation", "before_stored_run", "after_run", "memory_after_run",
                            "before_received_run", "at_final")
# See the module docstring: the episode's reading budget used up ends the episode. QUERY_BUDGET_CODE is the host's own
# refusal code (agent.host.GuardedHost.call and episode_gate).
QUERY_BUDGET_RULE = "episode-query-budget-end-1"
QUERY_BUDGET_CODE = "episode_max_queries"
QUERY_BUDGET_STATUS = "stopped_query_budget"  # an episode stop (never stopped_budget, which stops a real stage)
QUERY_BUDGET_CLAIM = "help_requested:query_budget_exhausted"  # taskcheck: search_unsuccessful
QUERY_BUDGET_REASON = "episode_query_budget_end"  # the final's reason is "episode_query_budget_end:episode_max_queries"
QUERY_BUDGET_CHECKPOINTS = ("after_run", "memory_after_run")  # rec["query_budget_end"]["recognised_at"]
# See the module docstring: a correction branch that cannot progress ends. NON_PROGRESS_CODES are the host's own refusal
# codes of a budget that never comes back in the episode (agent.host.GuardedHost.call and episode_gate); the reading
# budget has its own end (19.2), the simulated time and the last skill call are the physical end.
NON_PROGRESS_RULE = "episode-non-progress-end-1"
NON_PROGRESS_CODES = ("episode_max_observations", "episode_max_skill_calls")
NON_PROGRESS_STATUS = "stopped_non_progress"  # an episode stop (never stopped_budget, which stops a real stage)
NON_PROGRESS_CLAIM = "help_requested:repeated_impossible_operation"  # taskcheck: search_unsuccessful
NON_PROGRESS_REASON = "episode_non_progress_end"  # the final's reason is "episode_non_progress_end:<code>:<operation>"
NON_PROGRESS_CHECKPOINTS = ("after_run", "memory_after_run")  # rec["non_progress_end"]["recognised_at"]
# skills that move only the arm or the camera: without a view they change nothing the next plan can use (a contact they
# report still counts, through _progress_state's contact incidents)
VIEW_ONLY_SKILLS = ("go_home", "look_pose", "look_down_at", "look_at_region")
PROGRESS_STATE_KEYS = ("holding", "drawers", "contact_incidents", "grip_events", "scope")


def drawer_reference_version() -> str:
    """perception.estimate.DRAWER_REFERENCE_VERSION ("drawer-reference-2"), for
    rec["system_versions"]. Imported when called: this module imports nothing of the robot or of MuJoCo at import time
    (the perception module is the one the robot-side backend uses; the shared perception group of the S chain)."""
    from ..perception.estimate import DRAWER_REFERENCE_VERSION
    return DRAWER_REFERENCE_VERSION


def scope_cover_version() -> str:
    """perception.estimate.SCOPE_COVER_VERSION ("scope-occluder-1": an unseen floor point under a
    perceived object counts as an exclusion only when that object rests on the floor), for rec["system_versions"].
    Imported when called, like drawer_reference_version (the shared perception group of the S chain)."""
    from ..perception.estimate import SCOPE_COVER_VERSION
    return SCOPE_COVER_VERSION


def physical_end_why(pe: dict) -> str:
    """The text _why_no_call gives for a physical end (records only: a BudgetStop reason; the final's reason
    is "episode_physical_end:<code>")."""
    return (f"{PHYSICAL_END_REASON}: {pe.get('code')} (sim_s {pe.get('sim_s')} of {pe.get('max_sim_s')}, "
            f"skill_calls {pe.get('skill_calls')} of {pe.get('max_skill_calls')})")


def query_budget_stop(outcome: dict | None, span: list[dict] | None) -> dict | None:
    """The run that just ended stopped on the episode's reading
    budget, or None. Both must hold: the outcome is the limit QUERY_BUDGET_CODE (the interpreter's, or the sandbox
    child's as the parent checked it), and span (the host records of that run) holds the host's own refusal of a reading
    for that code (agent.host.GuardedHost.call: _refuse(name, args, "episode_max_queries", ...) before LimitReached).
    An outcome that names the code without that robot-side record is not believed (None).
    Returns {"code", "refused_reading" (the refused query's name), "line" (the program line)}. Pure."""
    o = outcome if isinstance(outcome, dict) else {}
    err = o.get("error") if isinstance(o.get("error"), dict) else {}
    if o.get("status") != "limit" or err.get("code") != QUERY_BUDGET_CODE:
        return None
    refused = [r for r in (span or []) if isinstance(r, dict) and r.get("status") == QUERY_BUDGET_CODE]
    if not refused:
        return None
    line = err.get("line")
    return {"code": QUERY_BUDGET_CODE, "refused_reading": str(refused[-1].get("name"))[:40],
            "line": line if type(line) is int else None}


def impossible_refusal(outcome: dict | None, span: list[dict] | None, gate=None) -> dict | None:
    """The operation the run that just ended was stopped on, when
    the robot refused it for good in this episode, or None. All must hold: the outcome is a limit whose code is one
    of NON_PROGRESS_CODES (the interpreter's, or the sandbox child's as the parent checked it); span (the host
    records of that run) holds the host's own refusal record for that code (agent.host.GuardedHost.call:
    _refuse(name, args, code, ...) before LimitReached), whose name is the operation; and gate
    (GuardedHost.episode_gate; None: not checked, for the pure tests and the record readers) still refuses that
    operation for that code now. Those budgets only shrink (observations used, skill calls used against the
    operation's worst case), so the refusal holds for the rest of the episode. An outcome that names the code without
    that robot-side record is not believed (None: corrected as before). Returns {"code", "operation", "line"}. Pure
    but for the gate's reading (no record, no count)."""
    o = outcome if isinstance(outcome, dict) else {}
    err = o.get("error") if isinstance(o.get("error"), dict) else {}
    code = err.get("code")
    if o.get("status") != "limit" or code not in NON_PROGRESS_CODES:
        return None
    refused = [r for r in (span or []) if isinstance(r, dict) and r.get("status") == code]
    if not refused or not isinstance(refused[-1].get("name"), str):
        return None
    op = refused[-1]["name"]
    if gate is not None:
        try:
            if gate(op) != code:
                return None
        except Exception:  # noqa: BLE001 (a gate that cannot answer proves nothing: corrected as before)
            return None
    line = err.get("line")
    return {"code": code, "operation": op[:40], "line": line if type(line) is int else None}


def run_progress(span: list[dict] | None, before: dict | None, after: dict | None) -> dict:
    """What a run brought, robot side: {"new_observations" (views delivered in its span,
    agent.host.delivered_obs_id), "state_gestures" (skills outside VIEW_ONLY_SKILLS that reached the robot,
    RAN_STATUSES, ok or not: they may have moved something), "state_changed" (the PROGRESS_STATE_KEYS whose
    robot-side reading differs between before and after, the loop's _progress_state at the start and the end of the
    run; ["unread"] when either is missing: no proof of no change), "progress"} where progress is any of the three. A
    refused gesture (a precondition, a budget) reached nothing. Pure."""
    recs = [r for r in (span or []) if isinstance(r, dict)]
    views = [delivered_obs_id(r) for r in recs]
    gestures = [str(r.get("name"))[:40] for r in recs if r.get("kind") == "skill" and r.get("status") in RAN_STATUSES
                and r.get("name") not in VIEW_ONLY_SKILLS]
    if not isinstance(before, dict) or not isinstance(after, dict):
        changed = ["unread"]
    else:
        changed = [k for k in PROGRESS_STATE_KEYS if before.get(k) != after.get(k)]
    out = {"new_observations": sum(1 for v in views if v is not None), "state_gestures": gestures[:8],
           "state_changed": changed}
    out["progress"] = bool(out["new_observations"] or gestures or changed)
    return out


def claims_done(outcome: dict | None) -> bool:
    """The trigger of the done() guards (the same as refuse_done_after_incident's): the claim task_done, whatever
    the terminal or the status, or the terminal done."""
    o = outcome or {}
    return o.get("claim") == "task_done" or (o.get("status") == "terminal" and o.get("terminal") == "done")


def refuse_done_after_incident(outcome: dict, records: list[dict]) -> dict | None:
    """Harness rule. If the program ended with done() and the robot felt a contact incident earlier in
    the episode (agent.host.contact_incidents over the host's own records, robot side only), return the outcome
    rewritten as status "done_refused" with an observable error; else None. The incidents are named in the message
    (skill, reason, force read by the robot), never the verifier's thresholds or verdict. The rule
    also fires on the claim task_done whatever the terminal or the status say (a forged sandbox-child
    outcome could pair terminal "help" with claim "task_done"; the sandbox now refuses such a pair too,
    sandbox.claim_matches_terminal; this is the second barrier)."""
    claims_done = outcome.get("claim") == "task_done"
    if not claims_done and (outcome.get("status") != "terminal" or outcome.get("terminal") != "done"):
        return None
    inc = contact_incidents(records)
    grips = grip_events(records)  # drops and releases of the sponge too
    if not inc and not grips:
        return None
    if inc:
        first = inc[0]
        force = f", {first['force_peak_n']} N read by the robot" if "force_peak_n" in first else ""
        head = (f"done() refused by the robot: {len(inc)} contact incident(s) in this episode (first: {first['call']} "
                f"{first['reason']}{force}). The task can no longer be claimed done in this episode;")
        tail = ' end with help("contact incident", kind="safety").'  # a kind (a help without kind is refused)
        code = "done_refused_contact_incident"
    else:
        first = grips[0]
        # An internal release (agent.host: the drawer clearance opened the hand) is not the program's
        what = ("the robot's grip reading lost the held object" if first["event"] == "grip_loss"
                else "the robot's own clearance opened the hand on the sponge" if first["event"] == "internal_release"
                else "the program released the sponge")
        head = (f"done() refused by the robot: {len(grips)} grip event(s) in this episode (first: {first['call']} "
                f"{first['reason']}, {what}). The sponge must stay in the hand from its pick until done(); the task "
                "can no longer be claimed done in this episode;")
        tail = ' end with help("sponge dropped", kind="cannot_continue").'
        code = "done_refused_grip_event"
    msg = head[:300 - len(tail)] + tail  # the suggested call is never cut by the 300-character bound
    return {**outcome, "status": DONE_REFUSED, "terminal": None, "claim": None,
            "refused_terminal": outcome.get("terminal") or "done",
            "error": {"kind": "harness_rule", "code": code, "line": 0, "message": msg}}


def refuse_done_without_evidence(outcome: dict, ev: dict | None) -> dict | None:
    """If the outcome claims done (claims_done) and ev (agent.host.GuardedHost.task_evidence, TASK_EVIDENCE_VERSION:
    task-evidence-2, whose scrub_target_unconfirmed is a task code) does not say "proven" under that version, return the
    outcome rewritten as status "done_refused" with an observable error, else None. Code done_refused_hold_not_proven
    when every missing or void element is about the hold (HOLD_EVIDENCE_CODES: the grip not felt after a gesture, a
    negative grip reading, a second pick, robot calls not itemized, the grip empty or unreadable at done()), else
    done_refused_task_not_proven. The message (at most 300 characters, the suggested help() with a kind last) names the
    robot-side codes only, never the verifier. Called after refuse_done_after_incident (which keeps its codes and
    messages). Under limits 0.8 done_refused is not in correct_on: on the model path the refusal is final and its text
    reaches no prompt."""
    if not claims_done(outcome):
        return None
    e = ev if isinstance(ev, dict) else {}
    if e.get("verdict") == "proven" and e.get("version") == TASK_EVIDENCE_VERSION:
        return None
    codes = [str(c)[:40] for c in list(e.get("voided_by") or []) + list(e.get("missing") or []) if isinstance(c, str)]
    hold_only = bool(codes) and all(c in HOLD_EVIDENCE_CODES for c in codes)
    if hold_only:
        what = "the hold of the sponge from its pick until done() is not proven by its own readings"
        tail = ' end with help("sponge hold not proven", kind="cannot_continue").'
    else:
        what = "its own records do not prove the task in this episode"
        tail = ' end with help("task not proven", kind="cannot_continue").'
    head = (f"done() refused by the robot: {what} ({str(e.get('verdict') or 'no evidence')[:12]}: "
            f"{', '.join(codes[:4]) or 'none'}). The task can no longer be claimed done in this episode;")
    msg = head[:300 - len(tail)] + tail  # the suggested call is never cut by the 300-character bound
    return {**outcome, "status": DONE_REFUSED, "terminal": None, "claim": None,
            "refused_terminal": outcome.get("terminal") or "done",
            "error": {"kind": "harness_rule", "code": DONE_REFUSED_HOLD_CODE if hold_only else DONE_REFUSED_TASK_CODE,
                      "line": 0, "message": msg}}


def refuse_done_over_budget(outcome: dict, exceeded) -> dict | None:
    """If the outcome claims done (claims_done) and `exceeded` (agent.host.GuardedHost.robot_budget_exceeded: the
    robot's own, unrounded episode budget verdict, "max_sim_s" or "max_skill_calls") is set, return the outcome
    rewritten as status "done_refused" with code DONE_REFUSED_BUDGET_CODE (claim DONE_REFUSED_BUDGET_CLAIM through
    done_refused_claim), else None. Why: a program whose last gesture started before the gate closed and ended past
    600 s of robot time, then called done(), would otherwise be accepted (task_done) while taskcheck classes the
    episode budget_exceeded (a false success claim). The loop applies it, model and stored programs alike, after
    refuse_done_after_incident and refuse_done_without_evidence (their codes are kept), only to a done() they would
    accept. The message (at most 300 characters, the suggested help() last) names no evaluator value and, under
    limits 0.8 (done_refused not in correct_on), reaches no prompt."""
    if not exceeded or not claims_done(outcome):
        return None
    tail = ' end with help("episode budget used", kind="cannot_continue").'
    head = (f"done() refused by the robot: its own episode budget was used before done() ({str(exceeded)[:20]}: the "
            "robot clock or its skill calls). The task can no longer be claimed done in this episode;")
    msg = head[:300 - len(tail)] + tail  # the suggested call is never cut by the 300-character bound
    return {**outcome, "status": DONE_REFUSED, "terminal": None, "claim": None,
            "refused_terminal": outcome.get("terminal") or "done",
            "error": {"kind": "harness_rule", "code": DONE_REFUSED_BUDGET_CODE, "line": 0, "message": msg}}


# Help review. A model program may end with help() right after the dossier's drawer although the sponge is elsewhere,
# while the scripted chain searches every candidate region and succeeds. Harness rule: when a program ends with
# help(kind="search_incomplete"), the robot felt no contact incident and no grip event, no pick of a perceived target
# was refused in that program, the robot-side budget allows one more look, and the robot's own search scope for the
# sponge says "not found, scope incomplete", the help is not accepted at once: the program is asked to continue, with
# the regions not cleared, within the episode caps and the profile's review limits. It also fires when the robot
# perceives a sponge it has not picked. A help stands when no call is allowed. The review reads the structured kind
# only, never the text of why: a help after a refused pick of a perceived target is about perception or safety and
# stays final, as are the kinds safety, perception, cannot_continue and other. search_incomplete of a stored program
# is reviewed under the same conditions.
HELP_REVIEW = "help_review"
HELP_REVIEW_CATEGORY = "sponge"  # the task's tool (public task: "Find the sponge, pick it, and scrub the bowl")
HELP_REVIEW_KIND = "search_incomplete"  # the only reviewable kind (agent.api.HELP_KINDS)
HELP_DEFAULT_KIND = "other"  # how a help outcome without help_kind is read (final; never reviewed)
# which rule reviewed a help (attempt and episode record "help_review_trigger"); one rule left; a perceived sponge not
# picked (help_review_trigger)
HELP_REVIEW_TRIGGERS = ("kind_search_incomplete", "kind_search_incomplete_target_perceived")
# not_found guard (see the module docstring), robot side, model and stored programs alike
NOT_FOUND_REFUSED = "not_found_refused"
NOT_FOUND_REFUSED_CODE = "not_found_refused_scope_incomplete"
NOT_FOUND_UNSUPPORTED_CLAIM = "help_requested:not_found_unsupported"
NOT_FOUND_ACCEPTED_SCOPE = "not_found_in_inspected_scope"
NOT_FOUND_NO_ABSENCE_CLAIM = "not_found_scope_incomplete"  # claims no absence, never guarded
NOT_FOUND_GUARD_CATEGORY = "sponge"  # the task's target (same reading as the help review)
MAX_NOT_FOUND_REFUSALS = 2  # per episode; the next refused not_found ends it with NOT_FOUND_UNSUPPORTED_CLAIM
# These two counts are profile limits (agent.limits.AgentLimitsV08 max_not_found_refusals, max_help_reviews; None = no
# count, limits 0.8; limits.reprise_limit); the constants are the earlier values (limits.REPRISE_DEFAULTS)
MAX_HELP_REVIEWS = 1
# The episode record's mark of the last reprise, in DELIVERED observations (GuardedHost.delivered_observations); the
# older key reprise_observation_mark holds the budget counter
REPRISE_MARK = "reprise_delivered_observation_mark"


def help_kind_of(outcome: dict | None) -> str | None:
    """The structured kind of a help() outcome ("other" when a help outcome carries none), None if not a help."""
    o = outcome or {}
    if o.get("terminal") != "help":
        return None
    hk = o.get("help_kind")
    return hk if isinstance(hk, str) and hk in HELP_KINDS else HELP_DEFAULT_KIND


def help_kind_can_be_written(program, kind: str) -> bool:
    """Defence in depth: True when a help() call of program can give kind:
    kind passed as that string literal, or as an expression that is not a literal (a name, a call, an operation).
    False otherwise, for *args or **kwargs (refused by the language), or for a text that is not a program. Structure
    only (the syntax tree, never the text of why): a sandbox child reporting search_incomplete for a program that cannot
    write it is not believed (the help stays final)."""
    if not isinstance(program, str):
        return False
    try:
        tree = ast.parse(program, mode="exec")
    except (SyntaxError, ValueError, RecursionError, MemoryError, OverflowError):
        return False
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "help"):
            continue
        if any(kw.arg is None for kw in node.keywords) or any(isinstance(a, ast.Starred) for a in node.args):
            return False
        for v in [kw.value for kw in node.keywords if kw.arg == "kind"] + list(node.args[1:2]):
            if not isinstance(v, ast.Constant) or (type(v.value) is str and v.value == kind):
                return True
    return False


def robot_budget_allows_a_look(budget) -> bool:
    """An exhausted budget is a stop: the robot-side budget
    (agent.host.GuardedHost.budget) leaves one more look: an observation, the worst case of look_at_region within the
    skill calls (WORST_ROBOT_CALLS, the host's own start rule) and simulated time. False for a missing or malformed
    reading."""
    if not isinstance(budget, dict):
        return False
    keys = ("observations", "max_observations", "skill_calls", "max_skill_calls", "sim_s_used", "max_sim_s")
    vals = [budget.get(k) for k in keys]
    if not all(type(v) in (int, float) and v == v for v in vals):
        return False
    obs, max_obs, calls, max_calls, sim_s, max_sim_s = vals
    return obs < max_obs and calls + WORST_ROBOT_CALLS["look_at_region"] <= max_calls and sim_s < max_sim_s


def help_review_trigger(outcome: dict, records: list[dict], scope: dict | None,
                        program_records: list[dict] | None = None, budget: dict | None = None,
                        kind_written: bool = False, perceived: list[dict] | None = None) -> str | None:
    """"kind_search_incomplete" when this help is reviewed once, else None (it stands). Reviewed only
    when ALL hold: the outcome is a help whose own help_kind is "search_incomplete" (a missing key is final);
    kind_written is True (help_kind_can_be_written: the program can write that kind; default False, the stricter
    reading); budget (agent.host.GuardedHost.budget) allows one more look (robot_budget_allows_a_look; None is
    final); no contact incident and no grip event in records (the episode's host records); no pick refused by the
    robot's pick precondition in program_records (those of the program that ended on help; None: the episode's,
    stricter), whatever the refusal code; the robot-side scope says not_found_scope_incomplete and lists regions not
    cleared. The caller applies the other conditions (the profile's review limits): a call allowed,
    the profile's number of reviews (reprise_limit; None: no count) and, under a profile that needs it, the progress
    precondition (reprise_has_progress). With the same other conditions, "kind_search_incomplete_target_perceived"
    when perceived (agent.host GuardedHost.perceived_not_picked: a sponge the robot perceives and has not picked in
    this episode) is not empty, whatever the scope says (it says found then), under every profile (0.7 and 0.6
    included: agent code, not profile)."""
    if outcome.get("status") != "terminal" or outcome.get("terminal") != "help":
        return None
    if outcome.get("help_kind") != HELP_REVIEW_KIND or kind_written is not True:
        return None
    if not robot_budget_allows_a_look(budget):
        return None
    if contact_incidents(records) or grip_events(records):
        return None
    if any(r.get("status") == PICK_REFUSED_STATUS for r in (records if program_records is None else program_records)):
        return None
    if _perceived(perceived):  # a perceived sponge not picked
        return HELP_REVIEW_TRIGGERS[1]
    if not isinstance(scope, dict) or scope.get("outcome") != "not_found_scope_incomplete":
        return None
    return HELP_REVIEW_TRIGGERS[0] if _not_cleared(scope) else None


def _not_cleared(scope: dict) -> list[str]:
    return [str(r)[:24] for r in (scope.get("not_cleared") or []) if isinstance(r, str)][:10]


def _perceived(perceived) -> list[dict]:
    """The well-formed entries of GuardedHost.perceived_not_picked (a finite position_m of 3 numbers)."""
    out = []
    for t in perceived if isinstance(perceived, list) else []:
        pos = t.get("position_m") if isinstance(t, dict) else None
        if isinstance(pos, list) and len(pos) == 3 and all(type(x) in (int, float) and x == x and abs(x) < 1e6
                                                           for x in pos):
            out.append(t)
    return out


# The text versions of the two harness notes. review-note-1 is the text (the regions last, cut at 300 characters);
# review-note-2 (profile s5-info-1.0 only) lists every region not cleared before the advice, with no cut (bounded by
# _not_cleared: at most 10 x 24 characters). The decisions, codes, triggers and stop reasons are the same under both
# versions: only the message text differs (tested over a grid).
NOTE_VERSIONS = ("review-note-1", "review-note-2")


def _note_version(note_version) -> str:
    if note_version not in NOTE_VERSIONS:
        raise ValueError(f"unknown note version {str(note_version)[:40]!r}: one of {NOTE_VERSIONS}")
    return note_version


def help_review_note(outcome: dict, records: list[dict], scope: dict | None,
                     program_records: list[dict] | None = None, budget: dict | None = None,
                     kind_written: bool = False, perceived: list[dict] | None = None, review: int = 1,
                     max_reviews: int | None = MAX_HELP_REVIEWS, needs_new_observation: bool = False,
                     note_version: str = "review-note-1") -> str | None:
    """The observable note for one more model call, or None when the help stands (help_review_trigger). It offers
    both ways, without imposing an action: search the regions not cleared if the preconditions allow, or stop now
    with help(why, kind); the regions come last, so that a long list is what the 300-character cut shortens. Review
    is this review's number in the episode, max_reviews the profile's (None: no count). With max_reviews 1 the
    earlier text is unchanged ("once per episode", "any next help stands"); otherwise the stop is a help of another
    kind (a later search_incomplete may be reviewed again). A perceived sponge not picked gives its region and
    position_m (the robot's own track) and offers to pick it after a top view if the preconditions allow.
    needs_new_observation (limits 0.8) adds that another review needs a new observation. note_version "review-note-2"
    (profile s5-info-1.0): the regions trigger lists the robot's own not-cleared regions in full BEFORE the advice,
    with no 300-character cut, and points to SCENE STATE for the reasons; tag and stop are computed as in
    review-note-1; the perceived-sponge note is unchanged (with its cut)."""
    note_version = _note_version(note_version)
    trigger = help_review_trigger(outcome, records, scope, program_records, budget, kind_written, perceived)
    if trigger is None:
        return None
    last = max_reviews is not None and review >= max_reviews
    tag = "once per episode" if max_reviews == 1 else f"review {review}" + (f" of {max_reviews}" if max_reviews else "")
    stop = ("stop now with help(why, kind): any next help stands" if last else
            "stop now with a help of another kind, which stands" + ("; another review needs a new observation"
                                                                    if needs_new_observation else ""))
    if trigger == HELP_REVIEW_TRIGGERS[1]:
        t = _perceived(perceived)[0]
        where = f"{str(t.get('region') or 'region unknown')[:24]} at position_m {[round(x, 3) for x in t['position_m']]}"
        return (f'help(kind="search_incomplete") not accepted yet ({tag}): the robot perceives a sponge it has not '
                f"picked, {where} (its track). Pick it after a top view if the preconditions allow, or {stop}.")[:300]
    left = _not_cleared(scope)
    if note_version == NOTE_VERSIONS[1]:
        return (f'help(kind="search_incomplete") not accepted yet ({tag}). The robot\'s own '
                f'search_scope("{HELP_REVIEW_CATEGORY}") lists {len(left)} region(s) not cleared: {", ".join(left)} '
                f"(reasons in SCENE STATE). Search them if the preconditions allow, or {stop}.")
    return (f'help(kind="search_incomplete") not accepted yet ({tag}). Search the regions not cleared if '
            f"the preconditions allow, or {stop}. The robot's own "
            f"search_scope lists {len(left)} region(s) not cleared: {', '.join(left)}")[:300]


def not_found_guard(outcome: dict, scope: dict | None, note_version: str = "review-note-1") -> dict | None:
    """None when a not_found() outcome is accepted (the robot's own scope, read now, says
    not_found_in_inspected_scope), else the observable error of the refusal (kind harness_rule, code
    not_found_refused_scope_incomplete, the scope's outcome and the regions not cleared; never an incident). A
    missing, malformed or other reading refuses. Robot side only: the reading of GuardedHost.scope_now, no oracle.
    The claim not_found_scope_incomplete (not_found("scope_incomplete")) claims no absence and is never refused (it
    was accepted only when the scope was complete, an inversion). note_version "review-note-2" (profile s5-info-1.0)
    changes the message only: the regions not cleared come before the advice, in full, with no 300-character cut
    (bounded by _not_cleared); the "found" message is unchanged. The decision and the code are those of
    review-note-1."""
    note_version = _note_version(note_version)
    if (outcome or {}).get("claim") == NOT_FOUND_NO_ABSENCE_CLAIM:
        return None
    so = scope.get("outcome") if isinstance(scope, dict) else None
    if so == NOT_FOUND_ACCEPTED_SCOPE:
        return None
    left = _not_cleared(scope) if isinstance(scope, dict) else []
    said = f"says {str(so)[:40]}" if isinstance(so, str) else "gives no reading"
    head = (f"not_found() refused by the robot: its own search_scope(\"{NOT_FOUND_GUARD_CATEGORY}\") {said}, not "
            f"{NOT_FOUND_ACCEPTED_SCOPE}.")
    if note_version == NOTE_VERSIONS[1] and so != "found":
        msg = (f"{head} Not cleared ({len(left)}): {', '.join(left)} (reasons in SCENE STATE). Search them if the "
               "preconditions allow, or stop with help(why, kind)." if left else
               f"{head} Search the regions not cleared if the preconditions allow, or stop with help(why, kind).")
        return {"kind": "harness_rule", "code": NOT_FOUND_REFUSED_CODE, "line": 0, "message": msg}
    then = ("The robot has a track of it: use it if the preconditions allow, or stop with help(why, kind)." if so ==
            "found" else "Search the regions not cleared if the preconditions allow, or stop with help(why, kind)."
            + (f" Not cleared ({len(left)}): {', '.join(left)}" if left else ""))
    msg = f"{head} {then}"
    return {"kind": "harness_rule", "code": NOT_FOUND_REFUSED_CODE, "line": 0, "message": msg[:300]}


def reprise_has_progress(mark: int | None, observations: int) -> bool:
    """True when a reprise may happen under the progress precondition: no earlier
    reprise in the episode (mark None), or at least one robot-side observation since it (observations > mark; a
    malformed reading is no progress). The loop passes GuardedHost.delivered_observations (views returned with an
    obs_id), no longer the budget counter EpisodeCounters.observations."""
    if mark is None:
        return True
    return type(mark) is int and type(observations) is int and observations > mark


def not_found_stop_reason(refusals: int, records: list[dict], budget,
                          max_refusals: int | None = MAX_NOT_FOUND_REFUSALS, progress: bool = True) -> str | None:
    """Why a refused not_found ends the episode at once (no absence claimed:
    NOT_FOUND_UNSUPPORTED_CLAIM), else None (a correction may follow): more than MAX_NOT_FOUND_REFUSALS refusals, a
    contact incident or a grip event of the episode (robot side), or a robot-side budget that allows no more look (a
    relaunch only under budget, never after an incident). max_refusals is the profile's
    AgentLimits.max_not_found_refusals; None (limits 0.8): no count, the other reasons and the episode wall time
    still stop. Progress False (reprise_has_progress under a profile that needs it: no new observation since the
    previous reprise) ends it too ("no_new_observation")."""
    if max_refusals is not None and refusals > max_refusals:
        return "max_refusals"
    if contact_incidents(records):
        return "contact_incident"
    if grip_events(records):
        return "grip_event"
    if not robot_budget_allows_a_look(budget):
        return "no_look_left"
    if not progress:
        return "no_new_observation"
    return None


def ended_by_pick_refusal(span: list[dict]) -> bool:
    """A program's records (its span) hold a pick refused by the robot's pick precondition, and after the
    LAST such refusal no pick reached the robot and no skill failed (the program stopped on the refusal, by its own
    terminal, a limit or an error of its own). Robot side, the host's records only. A skill whose backend raised
    after the refusal (agent.host.BACKEND_RAISED_STATUS, ok False) reached the robot and failed, like a called one:
    the refusal did not stop the program."""
    last = max((i for i, r in enumerate(span) if r.get("status") == PICK_REFUSED_STATUS), default=None)
    if last is None:
        return False
    return not any(r.get("kind") == "skill" and r.get("status") in RAN_STATUSES
                   and (r.get("name") == "pick" or r.get("ok") is False) for r in span[last + 1:])


def done_refused_claim(outcome: dict | None) -> str:
    """The episode's claim after a refused done(): names the first blocking cause (contact incident first).
    One claim per code (contact incident, grip event, task not proven, hold not proven); a refusal without a known
    code keeps the contact-incident claim."""
    code = ((outcome or {}).get("error") or {}).get("code")
    return {"done_refused_grip_event": DONE_REFUSED_GRIP_CLAIM, DONE_REFUSED_TASK_CODE: DONE_REFUSED_TASK_CLAIM,
            DONE_REFUSED_HOLD_CODE: DONE_REFUSED_HOLD_CLAIM,
            DONE_REFUSED_BUDGET_CODE: DONE_REFUSED_BUDGET_CLAIM}.get(code, DONE_REFUSED_CLAIM)  # round


class TechnicalRetries:
    """At most N technical retries for the whole run; None means no count cap (limits 0.7).
    state_dir (real batches): the retries are shared by every stage of a BATCH, across processes: each retry taken is
    a marker file created exclusively and never erased, and the count is read back from the folder, so a stage
    started later sees the retries already used."""

    def __init__(self, n: int | None, state_dir=None):
        if n is not None and (type(n) is not int or n < 0):
            raise ValueError("retry cap must be a non-negative integer or None")
        self.n = n
        self.state_dir = None if state_dir is None else Path(state_dir)
        if self.state_dir is not None:
            self.state_dir.mkdir(parents=True, exist_ok=True)
        self.used_here = 0
        self.left = None if self.n is None else max(0, self.n - self._used_total())
        self.used = 0

    def _used_total(self) -> int:
        if self.state_dir is None:
            return self.used_here
        return len(list(self.state_dir.glob("technical_retry_*.used.json")))

    def take(self, *, deadline: float | None = None) -> bool:
        """Atomically claim one retry. A racing process causes a recount, never a fictitious exhausted cap.
        The loop supplies its perf_counter deadline so contention cannot outlive the episode."""
        while True:
            if deadline is not None and time.perf_counter() >= deadline:
                return False
            total = self._used_total()
            if self.n is not None and self.n - total <= 0:
                self.left = 0
                return False
            if self.state_dir is None:
                break
            k = total + 1
            try:
                fd = os.open(self.state_dir / f"technical_retry_{k:03d}.used.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                             0o644)
            except FileExistsError:
                continue
            with os.fdopen(fd, "w") as f:
                json.dump({"retry": k, "of": self.n, "t": time.strftime("%Y-%m-%dT%H:%M:%S")}, f)
                f.flush()
                os.fsync(f.fileno())
            break
        self.used_here += 1
        self.used = self.used_here
        self.left = None if self.n is None else max(0, self.n - self._used_total())
        return True


class AgentLoop:
    def __init__(self, llm, ledger, limits: Limits | None = None, isolate: bool = True,
                 retries: TechnicalRetries | None = None, criteria_version: str = "unset",
                 require_os_sandbox: bool | None = None, info_profile: str = INFO_PROFILE_DEFAULT,
                 record_scene_reading: bool = False):
        """require_os_sandbox: None (default) = required whenever isolate is True. With it, an episode
        on a machine without sandbox-exec is refused before any model call (S5 result runs must keep the default;
        False only for an explicitly labelled run on another machine). info_profile, one of
        agent.prompt.INFO_PROFILES (ValueError otherwise; default s5-info-0.9, the earlier information byte for
        byte); record_scene_reading (default False, so the records of every earlier harness stay byte-identical):
        each model attempt records att["scene_reading"] (GuardedHost.scene_reading_record, both profiles, records
        only) and the episode rec["scope_at_end"]."""
        self.info_profile = info_profile_of(info_profile)["name"]
        self._profile = info_profile_of(info_profile)
        self.record_scene_reading = bool(record_scene_reading)
        self.llm, self.ledger = llm, ledger
        self.lim = limits or load_limits()
        self.isolate = isolate
        self.retries = retries or TechnicalRetries(self.lim.agent.max_technical_retries_per_run)
        self.criteria_version = criteria_version
        self.require_os_sandbox = isolate if require_os_sandbox is None else bool(require_os_sandbox)
        # The GuardedHost of the episode running now (run_episode sets it and clears it when it returns or raises); None
        # outside an episode: _why_no_call then reads no physical end (two older tests call the loop's pieces
        # on a fresh AgentLoop)
        self._host: GuardedHost | None = None

    # ------------------------------------------------------------ pieces
    def _check_isolation(self):
        if not self.require_os_sandbox:
            return
        if not self.isolate:
            raise SandboxUnavailable("require_os_sandbox needs isolate=True")
        if child_command()[1] == "none":
            raise SandboxUnavailable("sandbox-exec is not available: isolated runs refused")

    def _run_program(self, src: str, host: GuardedHost) -> dict:
        if self.isolate:
            out = run_isolated(src, host, self.lim.program)
            iso = out.get("isolation") or {}
            if self.require_os_sandbox and iso.get("process") == "separate" and iso.get("os_sandbox") == "none":
                raise SandboxUnavailable("a program ran without the OS sandbox layer")  # pragma: no cover
            return out
        out = Interpreter(host, self.lim.program).run(src)
        out["isolation"] = {"process": "same (development mode)", "os_sandbox": "none"}
        return out

    @staticmethod
    def _report(outcome: dict, host: GuardedHost, n0: int) -> dict:
        """The observable run report (prompt.run_report); with the robot-side wrist view notes of this run (the
        three-argument call when there are none, as before)."""
        notes = wrist_view_notes(host.records, n0)
        if notes:
            return run_report(outcome, host.records[n0:], host.budget(), notes=notes)
        return run_report(outcome, host.records[n0:], host.budget())

    def _reviews_left(self, rec: dict) -> bool:
        """May one more help review happen in this episode (AgentLimits.max_help_reviews; None: no count)."""
        n = reprise_limit(self.lim.agent, "max_help_reviews")
        return n is None or rec.get("help_reviews", 0) < n

    def _progress(self, rec: dict, host: GuardedHost) -> bool:
        """The progress precondition of the next reprise (always True for a profile without
        reprise_needs_new_observation). Read on the DELIVERED observations (GuardedHost.delivered_observations), no
        longer on the budget counter (an observe that raised is no progress)."""
        if not reprise_needs_new_observation(self.lim.agent):
            return True
        return reprise_has_progress(rec.get(REPRISE_MARK), host.delivered_observations)

    @staticmethod
    def stop_claim(rec: dict, previous: dict | None, reason: str, physical_end: dict | None = None) -> dict:
        """The final record of an episode stopped before the next program (a budget stop, the episode wall time, or a
        program received after the deadline): after a refused done() its claim (never task_done), after a review the
        reviewed help stands (help_kind, final_help), after a refused not_found no absence is claimed; else
        help_requested:agent_budget. One rule for the three stops (the wall-time stops always gave
        help_requested:agent_budget). When physical_end (agent.host.GuardedHost.physical_end) is given, the stop is
        the episode's physical end. The status is then PHYSICAL_END_STATUS in EVERY branch (stopped_budget would stop
        a whole real stage, siteagent.scripts.run_agent_real.stage_stop_checker), the reason
        "episode_physical_end:<code>", the source "loop", physical_end is attached, and the claim follows the same
        precedence: the refused done()'s, the reviewed help's (help_kind, final_help), no absence after a refused
        not_found, else PHYSICAL_END_CLAIM. Without physical_end the reason is kept as given."""
        status = ((previous or {}).get("report") or {}).get("status")
        fin = {"status": "stopped_budget", "claim": "help_requested:agent_budget", "reason": reason}
        if physical_end is not None:
            fin = {"status": PHYSICAL_END_STATUS, "claim": PHYSICAL_END_CLAIM,
                   "reason": f"{PHYSICAL_END_REASON}:{physical_end.get('code')}", "source": "loop",
                   "physical_end": dict(physical_end)}
        if status == DONE_REFUSED:
            fin["claim"] = done_refused_claim(previous.get("report"))
        elif status == HELP_REVIEW and rec.get("help_review_claim"):
            fin.update(claim=rec["help_review_claim"], help_kind=rec.get("help_review_kind"), final_help=True)
        elif status == NOT_FOUND_REFUSED:
            fin["claim"] = NOT_FOUND_UNSUPPORTED_CLAIM
        return fin

    def _reservation(self) -> Decimal:
        """The next reservation: the ledger's own (its model's bounded worst case), else the default model's."""
        return Decimal(str(getattr(self.ledger, "reservation_usd", RESERVATION_USD)))

    def _episode_committed(self, rec: dict) -> Decimal:
        """Cost committed by this episode's reservations: for each, the larger of the reservation and what
        the ledger keeps (settled cost or uncertain amount). Conservative by construction."""
        entries = getattr(self.ledger, "entries", None)
        total = Decimal(0)
        for tok in rec.get("reservation_ids", []):
            try:
                e = entries[tok]
                res = Decimal(e["reserved_usd"])
                total += max(res, _entry_cost(e, res))
            except (TypeError, KeyError, IndexError, ValueError):
                total += self._reservation()
        return total

    def _physical_end(self) -> dict | None:
        """The current episode's physical end (agent.host.GuardedHost.physical_end: host counters
        and the host clock, the gate's own expressions), or None: none, or no episode running (self._host None). Pure
        reading: no record, no count. A host clock that raises raises here, as in the host's gate."""
        h = getattr(self, "_host", None)
        return None if h is None else h.physical_end()

    def _physical_stop(self, rec: dict, previous: dict | None, pe: dict, recognised_at: str,
                       outcome: dict | None = None, source: str | None = None, host: GuardedHost | None = None,
                       task: dict | None = None) -> dict:
        """the final of an episode the loop ends at a recognised physical end
        pe (recognised_at in PHYSICAL_END_CHECKPOINTS), and rec["physical_end"]. outcome: the run that just ended
        (after a run); its status and error stand for the previous report of stop_claim's precedence; a give_up keeps
        its own claim failed:<why> and the program's source. Records only, never a prompt: rec["physical_end"] = pe +
        recognised_at, the attempt index, the rule, and two readings at the stop (the robot-side scope outcome of the
        task's tool, host.scope_now; the verdict GuardedHost.task_evidence would give now, over the last run's span):
        they let the analysis count stops where a terminal could have been accepted; the loop never claims one."""
        mark = {**pe, "recognised_at": recognised_at, "attempt": len(rec.get("attempts") or []) - 1,
                "rule": STOP_RULE_VERSION}
        if host is not None:
            try:
                scope = host.scope_now(NOT_FOUND_GUARD_CATEGORY)
                mark["scope_outcome_at_stop"] = scope.get("outcome") if isinstance(scope, dict) else None
            except Exception as e:  # noqa: BLE001 (a reading that fails is recorded, never fatal)
                mark["scope_outcome_at_stop"] = f"unreadable:{type(e).__name__}"
            last = next((a for a in reversed(rec.get("attempts") or []) if a.get("host_records_span")), None)
            try:
                ev = host.task_evidence(task or {}, span=(last or {}).get("host_records_span"),
                                        attempts=rec.get("attempts"))
                mark["task_evidence_verdict_at_stop"] = ev.get("verdict") if isinstance(ev, dict) else None
            except Exception as e:  # noqa: BLE001
                mark["task_evidence_verdict_at_stop"] = f"unreadable:{type(e).__name__}"
        rec["physical_end"] = mark
        o = outcome or {}
        if o.get("status") == "terminal" and o.get("terminal") == "give_up":
            fin = {"status": PHYSICAL_END_STATUS, "claim": o.get("claim") or "failed:agent:give_up",
                   "reason": f"{PHYSICAL_END_REASON}:{pe.get('code')}", "source": source, "physical_end": dict(mark)}
        else:
            prev = previous if outcome is None else {"report": {"status": o.get("status"), "error": o.get("error")}}
            fin = self.stop_claim(rec, prev, physical_end_why(pe), physical_end=mark)
        return fin

    def _mark_physical_end_at_final(self, rec: dict):
        """Critique: mark every final where the end is set: records only. When the episode
        ended otherwise (a reached terminal, a final refusal, a failed prelude, a wall-time stop) while its physical end
        holds, rec["physical_end"] = the host reading + recognised_at "at_final", the attempt, the rule and the final's
        status; an end the loop recognised itself (_physical_stop) is kept as recorded."""
        if "physical_end" in rec:
            return
        pe = self._physical_end()
        if pe is not None:
            rec["physical_end"] = {**pe, "recognised_at": "at_final", "attempt": len(rec.get("attempts") or []) - 1,
                                   "rule": STOP_RULE_VERSION, "final_status": (rec.get("final") or {}).get("status")}

    def _query_budget_stop(self, rec: dict, qb: dict, recognised_at: str, source: str | None, host: GuardedHost,
                           task: dict | None = None) -> dict:
        """The final of an episode whose last run stopped on the
        episode's reading budget (query_budget_stop gave qb; recognised_at in QUERY_BUDGET_CHECKPOINTS), and
        rec["query_budget_end"]. Records only, never a prompt: qb, the rule, the readings used and allowed (host
        counters, episode limits), the attempt index, the program's source, and two readings at the stop as
        _physical_stop takes them (the robot-side scope outcome of the task's tool, host.scope_now, uncounted; the
        verdict GuardedHost.task_evidence would give now over the last run's span): nothing is claimed on the model's
        behalf."""
        mark = {"rule": QUERY_BUDGET_RULE, **qb, "recognised_at": recognised_at,
                "attempt": len(rec.get("attempts") or []) - 1, "source": source,
                "queries": host.counters.queries, "max_queries": self.lim.episode.max_queries}
        try:
            scope = host.scope_now(NOT_FOUND_GUARD_CATEGORY)
            mark["scope_outcome_at_stop"] = scope.get("outcome") if isinstance(scope, dict) else None
        except Exception as e:  # noqa: BLE001 (a reading that fails is recorded, never fatal)
            mark["scope_outcome_at_stop"] = f"unreadable:{type(e).__name__}"
        last = next((a for a in reversed(rec.get("attempts") or []) if a.get("host_records_span")), None)
        try:
            ev = host.task_evidence(task or {}, span=(last or {}).get("host_records_span"), attempts=rec.get("attempts"))
            mark["task_evidence_verdict_at_stop"] = ev.get("verdict") if isinstance(ev, dict) else None
        except Exception as e:  # noqa: BLE001
            mark["task_evidence_verdict_at_stop"] = f"unreadable:{type(e).__name__}"
        rec["query_budget_end"] = mark
        return {"status": QUERY_BUDGET_STATUS, "claim": QUERY_BUDGET_CLAIM,
                "reason": f"{QUERY_BUDGET_REASON}:{QUERY_BUDGET_CODE}", "source": "loop", "query_budget_end": dict(mark)}

    @staticmethod
    def _progress_state(host: GuardedHost) -> dict:
        """The robot-side readings run_progress compares at the start and the end of a run
        PROGRESS_STATE_KEYS: the grip reading (holding_reading), the drawers believed open, the episode's contact
        incidents and grip events (robot_state, from the host's own records), and the search scope of the task's tool
        (scope_now: outcome, not cleared, found tracks). Not counted, not recorded, moves nothing (the readings a
        model request already makes); a reading that fails is kept as its error type (a change of it counts as a
        change)."""
        out = {}
        try:
            out["holding"] = host.holding_reading()
        except Exception as e:  # noqa: BLE001
            out["holding"] = f"unreadable:{type(e).__name__}"
        try:
            st = host.robot_state()
            out["drawers"] = json.dumps(st.get("drawers_believed_open"), sort_keys=True, default=str)
            out["contact_incidents"] = len(st.get("contact_incidents") or [])
            out["grip_events"] = len(st.get("grip_events") or [])
        except Exception as e:  # noqa: BLE001
            out["drawers"] = out["contact_incidents"] = out["grip_events"] = f"unreadable:{type(e).__name__}"
        try:
            sc = host.scope_now(NOT_FOUND_GUARD_CATEGORY)
            out["scope"] = (json.dumps({k: sc.get(k) for k in ("outcome", "not_cleared", "found_tracks")},
                                       sort_keys=True, default=str) if isinstance(sc, dict) else None)
        except Exception as e:  # noqa: BLE001
            out["scope"] = f"unreadable:{type(e).__name__}"
        return out

    @staticmethod
    def _non_progress_check(np_state: dict, outcome: dict, host: GuardedHost, n0: int, before: dict | None,
                            attempt: int, source: str | None) -> dict | None:
        """None when the run that just ended may be corrected as before, else the stop: the
        run was stopped by the refusal of an operation already known to be impossible in this episode (impossible_refusal,
        a key (code, operation) of np_state["known"], set by an earlier run) and brought no progress (run_progress over
        its span, before = _progress_state at its start). The first refusal of an operation only makes it known.
        np_state (the loop's, never the record): {"known": {(code, operation): attempt}, "checks": [...]}."""
        imp = impossible_refusal(outcome, host.records[n0:], host.episode_gate)
        if imp is None:
            return None
        key = (imp["code"], imp["operation"])
        if key not in np_state["known"]:
            np_state["known"][key] = attempt
            return None
        prog = run_progress(host.records[n0:], before, AgentLoop._progress_state(host))
        check_ = {"attempt": attempt, "source": source, "code": imp["code"], "operation": imp["operation"],
                  "known_since_attempt": np_state["known"][key], **prog}
        np_state["checks"].append(check_)
        return None if prog["progress"] else {**imp, "known_since_attempt": np_state["known"][key], "progress": prog}

    def _non_progress_stop(self, rec: dict, np_: dict, np_state: dict, recognised_at: str, source: str | None,
                           host: GuardedHost, task: dict | None = None) -> dict:
        """The final of an episode whose correction branch cannot progress (_non_progress_check
        gave np_; recognised_at in NON_PROGRESS_CHECKPOINTS), and rec["non_progress_end"]. Records only, never a
        prompt: the rule, the refusal (code, operation, program line), the attempt where the operation became known,
        the stopping run's progress reading, every operation known and every progress check of the episode, the
        attempt index, the program's source, the counters, and two readings at the stop as _physical_stop takes them
        (the robot-side scope outcome of the task's tool, uncounted; the verdict GuardedHost.task_evidence would give
        now over the last run's span): nothing is claimed on the model's behalf."""
        c, ep = host.counters, self.lim.episode
        mark = {"rule": NON_PROGRESS_RULE, **np_, "recognised_at": recognised_at,
                "attempt": len(rec.get("attempts") or []) - 1, "source": source,
                "known": [{"code": k[0], "operation": k[1], "attempt": a} for k, a in np_state["known"].items()],
                "checks": [dict(x) for x in np_state["checks"]],
                "observations": c.observations, "max_observations": ep.max_observations,
                "skill_calls": c.skill_calls, "max_skill_calls": ep.max_skill_calls}
        try:
            scope = host.scope_now(NOT_FOUND_GUARD_CATEGORY)
            mark["scope_outcome_at_stop"] = scope.get("outcome") if isinstance(scope, dict) else None
        except Exception as e:  # noqa: BLE001 (a reading that fails is recorded, never fatal)
            mark["scope_outcome_at_stop"] = f"unreadable:{type(e).__name__}"
        last = next((a for a in reversed(rec.get("attempts") or []) if a.get("host_records_span")), None)
        try:
            ev = host.task_evidence(task or {}, span=(last or {}).get("host_records_span"), attempts=rec.get("attempts"))
            mark["task_evidence_verdict_at_stop"] = ev.get("verdict") if isinstance(ev, dict) else None
        except Exception as e:  # noqa: BLE001
            mark["task_evidence_verdict_at_stop"] = f"unreadable:{type(e).__name__}"
        rec["non_progress_end"] = mark
        return {"status": NON_PROGRESS_STATUS, "claim": NON_PROGRESS_CLAIM,
                "reason": f"{NON_PROGRESS_REASON}:{np_['code']}:{np_['operation']}", "source": "loop",
                "non_progress_end": dict(mark)}

    def _why_no_call(self, rec: dict) -> str | None:
        """Why the next reservation may not be made (None: it may). Checked before EVERY reservation,
        the technical retry included: episode calls, episode committed cost, episode wall time, then the run envelope
        if the ledger has one. FIRST the episode's physical end (_physical_end; physical_end_why), so no reservation
        and no model call follow a physical end on any path (first call, correction, call after a stored program,
        technical retry, help review gate, stored give_up gate). Outside an episode (self._host None) it is not read."""
        ag = self.lim.agent
        pe = self._physical_end()
        if pe is not None:
            return physical_end_why(pe)
        if ag.max_model_calls is not None and rec["model_calls"] >= ag.max_model_calls:
            return f"max_model_calls ({ag.max_model_calls}, technical retry included) reached for this episode"
        if (ag.max_episode_cost_usd is not None
                and self._episode_committed(rec) + self._reservation() > Decimal(str(ag.max_episode_cost_usd))):
            return f"max_episode_cost_usd ({ag.max_episode_cost_usd}) would be exceeded by the next reservation"
        if time.perf_counter() - self._t0 >= ag.max_episode_wall_s:
            return f"max_episode_wall_s ({ag.max_episode_wall_s} s) reached"
        envelope_check = getattr(self.ledger, "would_refuse", None)
        return envelope_check() if callable(envelope_check) else None

    def _ask_model(self, rec: dict, task, dossier, observations, previous, memory_note,
                   activity: str | None = None, robot_state: dict | None = None) -> tuple[str | None, dict]:
        """One planned model call plus the allowed technical retries. Returns (program or None, attempt).
        activity: journal label of the reservation (default: program, or correction when previous is set)."""
        ag = self.lim.agent
        why = self._why_no_call(rec)
        if why is not None:
            raise BudgetStop(why)
        # Under s5-info-1.0 the robot's readings now (GuardedHost.scene_state: uncounted, unrecorded, no side effect)
        # become the SCENE STATE text of this request; build_messages refuses 1.0 without it. Read once per request: a
        # technical retry resends the same request, hence the same readings.
        prof, host = self._profile, self._host
        scene = host.scene_state(HELP_REVIEW_CATEGORY) if prof["scene_version"] is not None else None
        reading = host.scene_reading_record(HELP_REVIEW_CATEGORY) if self.record_scene_reading else None
        p = build_messages(task, dossier, observations, previous, ag.max_prompt_chars, memory_note, robot_state,
                           info_profile=prof["name"], scene_text=None if scene is None else scene["text"])
        req = build_request(p["messages"], ag.max_completion_tokens, ag.temperature,
                            model=getattr(self.llm, "request_model", None) or MODEL_ID,
                            reasoning_effort=getattr(ag, "reasoning_effort", None))
        try:
            bound = prompt_token_bound(req["messages"])
        except GuardrailError:
            bound = RESERVED_PROMPT_TOKENS + 1
        if bound > RESERVED_PROMPT_TOKENS:  # the reservation covers at most this many prompt tokens
            raise BudgetStop(f"max_prompt_tokens_reserved: prompt bound {bound} > {RESERVED_PROMPT_TOKENS} tokens; "
                             "nothing reserved, nothing sent")
        att = {"source": "model", "prompt_version": p["prompt_version"], "skill_api_version": p["skill_api_version"],
               "prompt_sha256": p["sha256"], "request_sha256": request_fingerprint(req), "prompt_chars": p["n_chars"],
               "omitted_observations": p["omitted_observations"], "correction_of": None if previous is None else
               hashlib.sha256(previous["program"].encode()).hexdigest()[:16], "technical_retries": 0, "calls": [],
               "max_completion_tokens": ag.max_completion_tokens}
        att["info_profile"] = prof["name"]  # records only
        if scene is not None:  # the rendering record of the text sent (records only, never a prompt)
            att["scene_state"] = {**dict(scene.get("record") or {}), "chars": p["scene_chars"],
                                  "sha256": p["scene_sha256"]}
        if reading is not None:
            att["scene_reading"] = reading
        activity = activity or ("S5_agent_program" if previous is None else "S5_agent_correction")
        att["activity"] = activity
        try:
            return self._ask_model_calls(rec, req, p, activity, att)
        except BudgetStop as e:
            # A stop after a call was made (retry refused, or usage outside the reviewed bounds): the attempt is kept in
            # the episode record, like the ledger keeps its row.
            if att["calls"]:
                att["error"] = {"kind": "budget_stop", "code": "budget_stop", "line": 0, "message": str(e)[:200]}
                rec["attempts"].append(att)
            raise
        except BaseException as e:  # any other stop after a paid call (Ctrl-C included): the attempt is kept
            if att["calls"]:
                att["error"] = {"kind": "interrupted", "code": type(e).__name__, "line": 0, "message": ""}
                rec["attempts"].append(att)
            raise

    def _uncertain(self, token) -> dict:
        try:
            e = self.ledger.entries[token]
            amt = _entry_cost(e, Decimal(e["reserved_usd"]))
        except (AttributeError, TypeError, KeyError, IndexError, ValueError):
            amt = self._reservation()
        return {"reservation_id": token, "cost_status": "uncertain", "uncertain_usd": str(amt)}

    def _ask_model_calls(self, rec, req, p, activity, att):
        binds = bool(getattr(self.llm, "binds_reservations", False))
        while True:
            why = self._why_no_call(rec)  # before EVERY reservation, the retry included
            if why is not None:
                raise BudgetStop(why)
            token = self.ledger.reserve(activity)
            rec["model_calls"] += 1
            rec.setdefault("reservation_ids", []).append(token)
            t_send = time.perf_counter()
            try:
                resp = self.llm.complete(req, reservation=token) if binds else self.llm.complete(req)
            except TransientLLMError as e:
                self.ledger.keep_uncertain(token, f"transient: {type(e).__name__}")
                att["calls"].append({"status": "transient_error", "latency_s": round(time.perf_counter() - t_send, 3),
                                     **self._uncertain(token)})
                # the next reservation is checked first: the run's single retry is not spent when no call may follow
                why = self._why_no_call(rec)
                if why is None:
                    if self.retries.take(deadline=self._t0 + self.lim.agent.max_episode_wall_s):
                        att["technical_retries"] += 1
                        continue
                    why = self._why_no_call(rec)  # a marker race may have reached the wall-time deadline
                att["error"] = {"kind": "model_error", "line": 0,
                                "code": "transient_no_retry_left" if why is None else "transient_no_call_allowed",
                                "message": "the request failed and no technical retry is left" if why is None
                                else f"the request failed and no further call is allowed: {why}"[:200]}
                return None, att
            except ClientNotReady as e:
                # A precondition of the real client is missing (preflight, authorization, envelope, reservation):
                # nothing was sent. The reservation is closed as uncertain (conservative) and the episode ends with
                # help_requested:model_unavailable (it escaped run_episode before).
                self.ledger.keep_uncertain(token, "client_not_ready")
                att["calls"].append({"status": "client_not_ready", "nothing_sent": True, **self._uncertain(token)})
                att["error"] = {"kind": "model_error", "code": "client_not_ready", "line": 0, "message": str(e)[:200]}
                return None, att
            except LLMRefused as e:
                if getattr(e, "usage", None) is not None:  # an answer came back (billed) but is unusable
                    return self._answered_unusable(token, e, p, att, round(time.perf_counter() - t_send, 3))
                self.ledger.keep_uncertain(token, "refused_or_empty")
                att["calls"].append({"status": "refused_or_empty", "latency_s": round(time.perf_counter() - t_send, 3),
                                     **self._uncertain(token)})
                att["error"] = {"kind": "model_error", "code": "refused", "line": 0, "message": str(e)[:200]}
                return None, att
            except BaseException as e:
                # An interrupted send may have been billed. Never lose its reservation or swallow cancellation.
                self.ledger.keep_uncertain(token, "interrupted_or_unexpected_transport_failure")
                att["calls"].append({"status": "interrupted", "error_type": type(e).__name__,
                                     "latency_s": round(time.perf_counter() - t_send, 3), **self._uncertain(token)})
                raise
            call = {"status": "answered", "model_id": resp.model_id, "usage": resp.usage,
                    "usage_source": resp.usage_source, "latency_s": resp.latency_s,
                    "finish_reason": resp.finish_reason, "reservation_id": token,
                    "reasoning_chars": len(getattr(resp, "reasoning", None) or "")}
            # A decoded answer may hold a lone surrogate (JSON "\ud800"); kept as an escape so that every hash, prompt
            # and record can be encoded as UTF-8
            att["response_text"] = utf8_safe(resp.text) if isinstance(resp.text, str) else resp.text
            if getattr(resp, "reasoning", None):
                att["reasoning_text"] = utf8_safe(str(resp.reasoning))  # bench side only (calls.jsonl), never run
            try:
                cost = self.ledger.settle(token, resp, p["prompt_version"])
            except BudgetStop:
                att["calls"].append({**call, "status": "answered_settlement_refused", **self._uncertain(token)})
                raise
            except BaseException as e:
                self.ledger.keep_uncertain(token, "interrupted_settlement")
                att["calls"].append({**call, "status": "answered_settlement_interrupted",
                                     "error_type": type(e).__name__, **self._uncertain(token)})
                raise
            att["calls"].append({**call, "computed_cost_usd": str(cost), "cost_status": "settled"})
            if resp.finish_reason != "stop":
                cut = resp.finish_reason == "length"
                att["truncated_at_output_limit"] = cut  # 0.2: detectable in calls.jsonl and in the stage gate
                att["error"] = {"kind": "parse_error", "code": "truncated", "line": 0,
                                "message": (f"the answer was cut off at the output limit of {req['max_completion_tokens']} "
                                            "tokens (reasoning included) before a complete program (finish_reason="
                                            "length)" if cut else
                                            f"the answer stopped with finish_reason={resp.finish_reason}")}
                return None, att
            try:
                return parse_response(att["response_text"]), att
            except ResponseError as e:
                att["error"] = e.as_dict()
                return None, att

    def _answered_unusable(self, token, e, p, att, latency):
        """An answer came back but is unusable (real_client.LLMAnsweredRefused): keep the provider's usage
        and finish_reason. Empty answer of the reviewed model: settled on that usage, then a parse error
        (a correction may follow within the caps). Another model named: reservation kept as uncertain,
        help_requested:model_unexpected."""
        call = {"status": "answered_" + e.code, "model_id": e.model_id, "usage": e.usage, "usage_source": e.usage_source,
                "latency_s": e.latency_s if e.latency_s is not None else latency, "finish_reason": e.finish_reason,
                "reservation_id": token, "reasoning_chars": e.reasoning_chars}
        if e.code == "empty_answer":
            att["response_text"] = ""
            resp = LLMResponse("", str(e.finish_reason), e.usage, e.model_id, call["latency_s"], e.usage_source)
            try:
                cost = self.ledger.settle(token, resp, p["prompt_version"])
            except BudgetStop:
                att["calls"].append({**call, "status": "answered_settlement_refused", **self._uncertain(token)})
                raise
            except BaseException as interrupted:
                self.ledger.keep_uncertain(token, "interrupted_settlement")
                att["calls"].append({**call, "status": "answered_settlement_interrupted",
                                     "error_type": type(interrupted).__name__, **self._uncertain(token)})
                raise
            att["calls"].append({**call, "computed_cost_usd": str(cost), "cost_status": "settled"})
            trunc = e.finish_reason == "length"
            att["truncated_at_output_limit"] = trunc
            att["error"] = {"kind": "parse_error", "code": "empty_answer_truncated" if trunc else "empty_answer",
                            "line": 0, "message": ("the answer was empty: the output limit of "
                                                   f"{att.get('max_completion_tokens')} tokens (reasoning included) was "
                                                   "reached before any program (finish_reason=length); reason briefly "
                                                   "and write the program first" if trunc else
                                                   f"the answer was empty (finish_reason={e.finish_reason})")}
            return None, att
        self.ledger.keep_uncertain(token, f"answered_{e.code}")
        att["calls"].append({**call, **self._uncertain(token)})
        att["error"] = {"kind": "model_error", "code": e.code, "line": 0, "message": str(e)[:200]}
        return None, att

    # ------------------------------------------------------------ episode
    def run_episode(self, task: dict, backend, dossier: dict, library: ProgramLibrary | None = None,
                    orientation: bool = True, prelude: tuple = ()) -> dict:
        """prelude: skills the loop runs before the orientation view, the same for every condition
        that uses it (real robot: ("calibrate", "look_pose"), like the first steps of the scripted chain). A failed
        prelude skill ends the episode before any model call. The episode's host is self._host while it runs
        (_physical_end, _why_no_call), cleared when the episode returns or raises (a loop reused across episodes
        never reads a previous episode's host)."""
        self._host = None
        try:
            return self._run_episode(task, backend, dossier, library, orientation, prelude)
        finally:
            self._host = None

    def _run_episode(self, task: dict, backend, dossier: dict, library: ProgramLibrary | None,
                     orientation: bool, prelude: tuple) -> dict:
        """The episode (run_episode)."""
        ag = self.lim.agent
        self._check_isolation()  # before any observation or model call
        counters = EpisodeCounters()
        prof = self._profile
        # The host gets the profile (its search_scope("sponge") note under 1.0, its scene readings). Under the default
        # s5-info-0.9 the host is built exactly as before: its own default profile is 0.9.
        hkw = {} if prof["name"] == INFO_PROFILE_DEFAULT else {"info_profile": prof["name"]}
        host = GuardedHost(backend, self.lim.episode, self.lim.program, counters, **hkw)
        self._host = host  # read by _physical_end before every reservation
        rec = {"label": "simulation, perception simulee", "prompt_version": prof["prompt_version"],
               "skill_api_version": prof["skill_api_version"], "info_profile": prof["name"],
               "limits_version": self.lim.version,
               "criteria_version": self.criteria_version, "model_calls": 0, "corrections": 0,
               "attempts": [], "memory": None, "final": None, "prelude": [],
               # The robot-side contracts this record was made under (records only, never a prompt); and the stop rule,
               # the drawer booking (agent.host) and the closed drawer reference (the shared perception); the shared
               # perception's occluder rule (scope-occluder-1)
               "system_versions": {"internal_actions": INTERNAL_ACTIONS_VERSION, "task_evidence": TASK_EVIDENCE_VERSION,
                                   "store_contract": STORE_CONTRACT, "library_schema": LIBRARY_SCHEMA,
                                   "stop_rule": STOP_RULE_VERSION, "drawer_booking": DRAWER_BOOKING_VERSION,
                                   "drawer_reference": drawer_reference_version(),
                                   "scope_cover": scope_cover_version(),
                                   # The information profile of this record (records only)
                                   "information": {"profile": prof["name"], "note_version": prof["note_version"],
                                                   "scene_version": prof["scene_version"],
                                                   "scope_note_version": prof["scope_note_version"]}}}
        self.current_rec = rec  # live reference: the bench writes a partial record if the episode is interrupted
        t0 = time.perf_counter()
        self._t0 = t0
        for name in prelude:
            try:
                r = host.call(name, {})
            except Exception as e:  # noqa: BLE001 (budget refusal at the very start is recorded)
                r = {"ok": False, "reason": type(e).__name__}
            rec["prelude"].append({"skill": name, "ok": bool(r.get("ok")), "reason": r.get("reason")})
            if not r.get("ok"):
                rec["final"] = {"status": "prelude_failed", "claim": f"failed:{name}:{r.get('reason')}", "source": "loop"}
                self._mark_physical_end_at_final(rec)  # records only
                return self._close(rec, host, t0)
        if orientation:
            try:
                host.call("observe", {"camera": "robot_eye"})
            except Exception as e:  # noqa: BLE001 (budget refusal at the very start is recorded)
                rec["orientation_error"] = f"{type(e).__name__}: {e}"
        # store-contract-1 S2: the episode opening is the records before this index (prelude, orientation view)
        rec["opening_end"] = len(host.records)
        previous, memory_note, entry = None, None, None
        if library is not None:
            # The profile's skill API version (0.8 entries are never served to 1.0, and the reverse)
            entry, checks = library.lookup(dossier, prof["skill_api_version"], self.criteria_version,
                                           host.observations(), host.records)
            rec["memory"] = {"site_id": library.site_id, "validation": library.validation, "checks": checks,
                             "reused_key": None if entry is None else entry["key"],
                             "reused_entry_id": None if entry is None else entry.get("entry_id"), "events": []}
            if entry is None and any(c["result"] == "contradicted" for c in checks):
                memory_note = "A stored program was invalidated: an observation contradicts what it relied on."
        program, source, after_memory = None, None, False
        # The operations known to be impossible in this episode and the progress checks of the runs refused again on one
        # of them; kept here, written to the record only when the rule ends the episode
        np_state = {"known": {}, "checks": []}
        while True:
            np_before = None  # 21.4: _progress_state at this run's start, read once an operation is known
            if time.perf_counter() - t0 >= ag.max_episode_wall_s:
                rec["final"] = self.stop_claim(rec, previous, f"max_episode_wall_s ({ag.max_episode_wall_s} s) reached")
                break
            if entry is not None:
                pe = self._physical_end()
                if pe is not None:
                    # (checkpoint before_stored_run; the initialisation path: the prelude or the orientation view used
                    # the episode up): the stored program is neither run nor invalidated, no model call
                    rec["final"] = self._physical_stop(rec, previous, pe, "before_stored_run", host=host, task=task)
                    rec["final"]["stored_entry_not_run"] = entry.get("entry_id")
                    rec["memory"]["events"].append({"key": entry["key"], "event": "not_run_physical_end",
                                                    "why": str(pe.get("code"))})
                    break
                program, source = entry["program"], "memory"
                att = {"source": "memory", "library_key": entry["key"], "library_entry_id": entry.get("entry_id"),
                       "calls": []}
            else:
                try:
                    program, att = self._ask_model(rec, task, dossier, host.observations(), previous, memory_note,
                                                   activity="S5_agent_after_memory" if after_memory else None,
                                                   robot_state=host.robot_state())
                except BudgetStop as e:
                    # after a refused done() the claim says so (never task_done); after a review the reviewed help
                    # stands; after a refused not_found no absence is claimed; else the budget stop checkpoint
                    # before_reservation: _why_no_call refused on the episode's physical end read first: the physical
                    # final, same precedence (stop_claim)
                    pe = self._physical_end()
                    if pe is not None:
                        rec["final"] = self._physical_stop(rec, previous, pe, "before_reservation", host=host,
                                                           task=task)
                    else:
                        rec["final"] = self.stop_claim(rec, previous, str(e))
                    break
                source, after_memory = "model", False
            rec["attempts"].append(att)
            n0 = len(host.records)
            if program is None:  # a parse error drives a correction; a model/transport error does not
                # (a correction call after a failed request would be a hidden retry: the run's single technical retry is
                # the only one)
                kind = (att.get("error") or {}).get("kind")
                code = (att.get("error") or {}).get("code")
                outcome = {"status": "model_error" if kind == "model_error" else "parse_error", "error": att.get("error"),
                           "claim": ("help_requested:model_unexpected" if code == "unexpected_model"
                                     else "help_requested:model_unavailable") if kind == "model_error" else None,
                           "log": [], "calls": []}
            else:
                att["program"] = program
                att["program_sha256"] = hashlib.sha256(program.encode()).hexdigest()
                if time.perf_counter() - t0 >= ag.max_episode_wall_s:
                    # A slow answer is accounted for but cannot start a program after the episode deadline.
                    rec["final"] = self.stop_claim(rec, previous,
                                                   f"max_episode_wall_s ({ag.max_episode_wall_s} s) reached")
                    break
                pe = self._physical_end()
                if pe is not None:
                    # (checkpoint before_received_run, a hard stop; no terminal-only window): the answer is
                    # accounted for, the program is not run. Nothing physical happens during a model call, so this fires
                    # only if the state changed outside the loop
                    att["not_run"] = PHYSICAL_END_REASON
                    rec["final"] = self._physical_stop(
                        rec, previous, pe, "before_received_run" if source == "model" else "before_stored_run",
                        host=host, task=task)
                    break
                host.reuse_entry = entry if source == "memory" else None
                if np_state["known"]:  # 21.4: the robot-side readings this run may change (uncounted)
                    np_before = self._progress_state(host)
                try:
                    outcome = self._run_program(program, host)
                finally:
                    host.reuse_entry = None
                att["host_records_span"] = [n0, len(host.records)]
                att["outcome"] = outcome
                if help_kind_of(outcome) is not None:
                    att["help_kind"] = help_kind_of(outcome)  # the program's structured help reason
                    # The reached help's why was not a readable text (agent.interp, api.check_help)
                    att["help_why_unreadable"] = outcome.get("claim") == f"help_requested:{HELP_WHY_UNREADABLE}"
                try:
                    att["run_report"] = self._report(outcome, host, n0)
                except (LeakError, ValueError) as e:
                    # An outcome that cannot enter a prompt (e.g. a hostile child's extra key): recorded as a sandbox
                    # error, never an exception out of the episode.
                    outcome = {"status": "sandbox_error", "terminal": None, "claim": None,
                               "error": {"kind": "runtime_error", "code": "report_refused", "line": 0,
                                         "message": f"the run report was refused ({type(e).__name__})"},
                               "log": [], "calls": [], "counters": {}, "isolation": outcome.get("isolation")}
                    att["outcome_refused"] = {"error_type": type(e).__name__, "message": str(e)[:200]}
                    att["outcome"] = outcome
                    att["run_report"] = self._report(outcome, host, n0)
                if outcome.get("status") == "terminal":  # the program's own terminal, before any harness rule
                    att["program_terminal"] = {"terminal": outcome.get("terminal"), "claim": outcome.get("claim")}
                refused = refuse_done_after_incident(outcome, host.records)
                if claims_done(outcome):
                    # Positive robot-side evidence at episode level (records only; never in a prompt)
                    att["task_evidence"] = host.task_evidence(task, span=att["host_records_span"],
                                                              attempts=rec["attempts"])
                    if refused is None and att["task_evidence"].get("verdict") != "proven" and (
                            source == "model" or att["task_evidence"].get("verdict") != "not_proven"):
                        # a model's done() is refused (final); a stored one too when voided (refused_stops below);
                        # a stored done() only not_proven keeps its outcome: _after_reuse reads the verdict ("failed")
                        refused = refuse_done_without_evidence(outcome, att["task_evidence"])
                    if refused is None and att["task_evidence"].get("verdict") == "proven":
                        # A done() the guards above accept is still refused when the robot's own budget is
                        # used (model and stored programs; the stored one stops through refused_stops below): the bench
                        # classes such an episode budget_exceeded whatever its claim
                        refused = refuse_done_over_budget(outcome, host.robot_budget_exceeded())
                if refused is not None:  # Harness rule: robot-side incidents void done()
                    outcome = refused
                    att["outcome"] = outcome
                    att["done_refused"] = True
                    att["run_report"] = self._report(outcome, host, n0)
                    rec["done_refusals"] = rec.get("done_refusals", 0) + 1
                elif outcome.get("status") == "terminal" and outcome.get("terminal") == "not_found":
                    # not_found guard, before every terminal branch, model and stored programs alike
                    scope = host.scope_now(NOT_FOUND_GUARD_CATEGORY)
                    if prof["note_version"] == NOTE_VERSIONS[0]:
                        # Under review-note-1 (profile 0.9) the earlier two-argument call, so that a replay
                        # harness's wrapper taking (outcome, scope) only keeps working
                        err = not_found_guard(outcome, scope)
                    else:
                        err = not_found_guard(outcome, scope, note_version=prof["note_version"])  # text only
                    decision = {"attempt": len(rec["attempts"]) - 1, "source": source,
                                "raw_claim": outcome.get("claim"),
                                "absence_claim": outcome.get("claim") != NOT_FOUND_NO_ABSENCE_CLAIM,  # review
                                "scope_outcome": scope.get("outcome") if isinstance(scope, dict) else None,
                                "not_cleared": _not_cleared(scope) if isinstance(scope, dict) else [],
                                "decision": "accepted" if err is None else "refused"}
                    if err is not None:
                        rec["not_found_refusals"] = rec.get("not_found_refusals", 0) + 1
                        progress = self._progress(rec, host)
                        stop = not_found_stop_reason(rec["not_found_refusals"], host.records, host.budget(),
                                                     reprise_limit(ag, "max_not_found_refusals"), progress)
                        decision.update(refusal=rec["not_found_refusals"], ends_episode=stop is not None,
                                        ends_because=stop, new_observation=progress)
                        rec[REPRISE_MARK] = host.delivered_observations  # delivered, not the budget counter
                        if stop == "no_new_observation":
                            rec["reprises_without_new_observation"] = rec.get("reprises_without_new_observation", 0) + 1
                        outcome = {**outcome, "status": NOT_FOUND_REFUSED, "terminal": None, "claim": None,
                                   "refused_terminal": "not_found", "refused_claim": outcome.get("claim"), "error": err}
                        att["outcome"] = outcome
                        att["run_report"] = self._report(outcome, host, n0)
                    att["not_found_guard"] = decision
                    rec.setdefault("not_found_guard", []).append(decision)
                elif self._reviews_left(rec) and self._why_no_call(rec) is None \
                        and outcome.get("status") == "terminal" and outcome.get("help_kind") == HELP_REVIEW_KIND:
                    # Only search_incomplete can be reviewed; every other help is final. From a model or a stored
                    # program alike (the stored run's own records, its own text for kind_written). As many reviews as
                    # the profile allows (limits 0.8: no count), also on a perceived sponge not picked
                    scope = host.scope_now(HELP_REVIEW_CATEGORY)
                    budget = host.budget()  # an exhausted robot-side budget keeps the help final
                    review_args = {"program_records": host.records[n0:], "budget": budget,
                                   "kind_written": help_kind_can_be_written(program, HELP_REVIEW_KIND),
                                   "perceived": host.perceived_not_picked(HELP_REVIEW_CATEGORY)}
                    trigger = help_review_trigger(outcome, host.records, scope, **review_args)
                    progress = self._progress(rec, host)  # a later review needs a new observation
                    att["help_review_check"] = {  # why it was reviewed or not (robot-side readings only)
                        "kind_written": review_args["kind_written"], "scope_outcome": (scope or {}).get("outcome"),
                        "budget_allows_a_look": robot_budget_allows_a_look(budget),
                        "pick_refused_in_program": any(r.get("status") == PICK_REFUSED_STATUS
                                                       for r in host.records[n0:]), "trigger": trigger,
                        "perceived_not_picked": len(_perceived(review_args["perceived"])),
                        "new_observation": progress}
                    if trigger is not None and not progress:
                        rec["reprises_without_new_observation"] = rec.get("reprises_without_new_observation", 0) + 1
                    if trigger is not None and progress:  # one more call: regions not cleared or the perceived sponge
                        rec[REPRISE_MARK] = host.delivered_observations  # delivered, not the budget counter
                        rec["help_reviews"] = rec.get("help_reviews", 0) + 1
                        note = help_review_note(outcome, host.records, scope, **review_args, review=rec["help_reviews"],
                                                max_reviews=reprise_limit(ag, "max_help_reviews"),
                                                needs_new_observation=reprise_needs_new_observation(ag),
                                                note_version=prof["note_version"])  # text only
                        rec["help_review_claim"] = outcome.get("claim")
                        rec["help_review_kind"] = HELP_REVIEW_KIND
                        rec["help_review_trigger"] = trigger
                        rec.setdefault("help_review_triggers", []).append(trigger)  # one per review
                        rec["help_review_source"] = source  # model or memory
                        outcome = {**outcome, "status": HELP_REVIEW, "terminal": None, "claim": None,
                                   "refused_terminal": "help",
                                   "error": {"kind": "harness_rule", "code": HELP_REVIEW, "line": 0, "message": note}}
                        att["outcome"] = outcome
                        att["help_review"] = True
                        att["help_review_trigger"] = trigger
                        att["run_report"] = self._report(outcome, host, n0)
            status = outcome["status"]
            # The refused not_found past MAX_NOT_FOUND_REFUSALS ends the episode (no absence claimed); so does one after
            # a contact or grip event, or with no look left (not_found_stop_reason)
            nf_ends = status == NOT_FOUND_REFUSED and bool((att.get("not_found_guard") or {}).get("ends_episode"))
            if source == "memory":
                verdict = self._after_reuse(library, entry, outcome, host, n0, att, rec)
                if status == "terminal" and outcome.get("terminal") == "help":
                    # A stored program's help is final (unless search_incomplete was reviewed above, its status then
                    # help_review): no model call, no gesture after it; the entry is invalidated
                    att["returned_to_model"] = False
                    att["final_help"] = True
                    rec["final"] = {"status": "terminal", "claim": outcome["claim"], "source": source,
                                    "help_kind": help_kind_of(outcome), "library_key": entry["key"],
                                    "library_entry_id": entry.get("entry_id"), "returned_to_model": False,
                                    "final_help": True, "memory_verdict": verdict}
                    break
                no_call = self._why_no_call(rec) if verdict == "ended_without_task" else None
                if no_call is not None:
                    # the stored program's own give_up stands when no model call is allowed (caps)
                    att["returned_to_model"] = False
                    att["no_call_reason"] = no_call
                    rec["final"] = {"status": status, "claim": outcome.get("claim") or f"failed:agent:{status}",
                                    "source": source, "library_key": entry["key"],
                                    "library_entry_id": entry.get("entry_id"), "returned_to_model": False}
                    pe = self._physical_end()
                    if pe is not None:
                        # checkpoint memory_after_run: at a physical end the stored give_up keeps its own claim;
                        # the status says the physical end, physical_end is attached
                        rec["final"] = {**self._physical_stop(rec, previous, pe, "memory_after_run", outcome=outcome,
                                                              source=source, host=host, task=task),
                                        "library_key": entry["key"], "library_entry_id": entry.get("entry_id"),
                                        "returned_to_model": False, "memory_verdict": verdict}
                    if help_kind_of(outcome) is not None:
                        rec["final"]["help_kind"] = help_kind_of(outcome)
                    break
                # limits 0.5: a stored program whose done() the robot refused is invalidated (above) and, like a model's
                # program, ends the episode at once when done_refused is not in correct_on (the next call could only end
                # on help or give_up)
                refused_stops = status == DONE_REFUSED and DONE_REFUSED not in ag.correct_on
                if verdict != "accepted" and not refused_stops and not nf_ends:
                    pe = self._physical_end()
                    if pe is not None:
                        # Checkpoint memory_after_run: the stored program used the episode up: no return to the model,
                        # no S5_agent_after_memory reservation; the entry keeps what _after_reuse decided (its library
                        # reason names the physical code)
                        att["returned_to_model"] = False
                        rec["final"] = {**self._physical_stop(rec, previous, pe, "memory_after_run", outcome=outcome,
                                                              source=source, host=host, task=task),
                                        "library_key": entry["key"], "library_entry_id": entry.get("entry_id"),
                                        "returned_to_model": False, "memory_verdict": verdict}
                        break
                    qb = query_budget_stop(outcome, host.records[n0:])
                    if qb is not None:
                        # Checkpoint memory_after_run: the stored program used the episode's readings up: no return to
                        # the model; the entry keeps what _after_reuse decided
                        att["returned_to_model"] = False
                        rec["final"] = {**self._query_budget_stop(rec, qb, "memory_after_run", source, host, task),
                                        "library_key": entry["key"], "library_entry_id": entry.get("entry_id"),
                                        "returned_to_model": False, "memory_verdict": verdict}
                        break
                    np_ = self._non_progress_check(np_state, outcome, host, n0, np_before, len(rec["attempts"]) - 1,
                                                   source)
                    if np_ is not None:
                        # Checkpoint memory_after_run: the stored program repeated an operation known to be impossible,
                        # with no progress: no return to the model; the entry keeps what _after_reuse decided
                        att["returned_to_model"] = False
                        rec["final"] = {**self._non_progress_stop(rec, np_, np_state, "memory_after_run", source, host,
                                                                  task),
                                        "library_key": entry["key"], "library_entry_id": entry.get("entry_id"),
                                        "returned_to_model": False, "memory_verdict": verdict}
                        break
                    memory_note = {"contradicted": "The stored program for this site no longer matches what the robot "
                                                   "observes; it is invalidated.",
                                   "unverified": "The stored program for this site could not be used: a condition it "
                                                 "relies on was not observed at the moment of use.",
                                   "failed": "The stored program for this site failed on reuse; it is invalidated.",
                                   "ended_without_task": "The stored program for this site ended with give_up() "
                                                         "on reuse (see its report); it is invalidated. "
                                                         "Continue from the robot's current state.",
                                   # The guard refused a stored program's not_found (not an explicit help)
                                   "not_found_refused": "The stored program for this site ended with not_found(), "
                                                        "which the robot refused (see its report); it is invalidated. "
                                                        "Continue from the robot's current state.",
                                   # A stored search_incomplete, reviewed once like a model program's
                                   "help_review": "The stored program for this site ended with "
                                                  'help(kind="search_incomplete") on reuse (see its report); it is '
                                                  "invalidated. Continue from the robot's current state."}[verdict]
                    if verdict in ("ended_without_task", "not_found_refused", "help_review"):
                        att["returned_to_model"] = True
                    entry = None
                    previous = {"program": program, "report": att["run_report"]}
                    after_memory = True  # the next model call is not a correction (journal label)
                    continue  # ask the model (counted in max_model_calls, not a correction)
            if status == "terminal" and outcome.get("terminal") != "give_up":
                rec["final"] = {"status": "terminal", "claim": outcome["claim"], "source": source}
                if help_kind_of(outcome) is not None:
                    rec["final"]["help_kind"] = help_kind_of(outcome)  # the claim keeps its format
                    rec["final"]["final_help"] = True  # nothing runs after it in this episode
                if source == "memory":  # external_validation only: the bench reports the verifier's bit back
                    rec["final"]["library_key"] = entry["key"]
                    rec["final"]["library_entry_id"] = entry.get("entry_id")
                break
            reason = "give_up" if status == "terminal" else status
            # The review is one more call; so is a refused not_found, up to MAX_NOT_FOUND_REFUSALS (the profile's
            # max_not_found_refusals, None in limits 0.8)
            correctable = reason in ag.correct_on or reason == HELP_REVIEW or (reason == NOT_FOUND_REFUSED
                                                                              and not nf_ends)
            corrections_exhausted = ag.max_corrections is not None and rec["corrections"] >= ag.max_corrections
            if not correctable or corrections_exhausted:
                if status == DONE_REFUSED:
                    claim = done_refused_claim(outcome)
                elif status == NOT_FOUND_REFUSED:  # no absence claimed
                    claim = NOT_FOUND_UNSUPPORTED_CLAIM
                else:
                    claim = (rec.get("help_review_claim") if status == HELP_REVIEW else None) or (
                        outcome.get("claim") or f"failed:agent:{reason}")
                rec["final"] = {"status": status, "claim": claim,
                                "source": source, "corrections_exhausted": corrections_exhausted}
                if status == HELP_REVIEW and rec.get("help_review_claim"):
                    rec["final"]["help_kind"] = rec.get("help_review_kind")  # the reviewed help stands
                    rec["final"]["final_help"] = True
                if status == NOT_FOUND_REFUSED:  # the program's raw claim, kept apart from the final claim
                    rec["final"].update(raw_claim=outcome.get("refused_claim"),
                                        not_found_refusals=rec.get("not_found_refusals", 0))
                break
            pe = self._physical_end()
            if pe is not None:
                # Checkpoint after_run: a correctable outcome, but the episode is physically ended (the simulated time
                # used, or no skill can start): no correction, no reservation. A give_up keeps its claim failed:<why>
                # limit, runtime_error, refused and parse_error get the physical final
                rec["final"] = self._physical_stop(rec, previous, pe, "after_run", outcome=outcome, source=source,
                                                   host=host, task=task)
                break
            qb = query_budget_stop(outcome, host.records[n0:])
            if qb is not None:
                # Checkpoint after_run: a correctable limit, but the episode's readings are used up (every later program
                # would stop at its first reading): no correction, no reservation
                rec["final"] = self._query_budget_stop(rec, qb, "after_run", source, host, task)
                break
            np_ = self._non_progress_check(np_state, outcome, host, n0, np_before, len(rec["attempts"]) - 1, source)
            if np_ is not None:
                # Checkpoint after_run: a correctable limit, but the run repeated an operation the robot refused for
                # good in this episode and brought neither a new view nor a useful state change: no correction, no
                # reservation
                rec["final"] = self._non_progress_stop(rec, np_, np_state, "after_run", source, host, task)
                break
            rec["corrections"] += 1
            previous = {"program": program or (att.get("response_text") or "")[:2000],
                        "report": att.get("run_report") or {"status": status, "error": outcome.get("error"),
                                                             "contains_oracle_information": False}}
        self._mark_physical_end_at_final(rec)  # records only
        self._end_invariant(rec)  # a final task_done needs proven task evidence, whatever branch set it
        rec = self._close(rec, host, t0)
        incidents = contact_incidents(host.records)
        grips = grip_events(host.records)
        rec["contact_incidents"] = incidents  # robot side (own skill results), for the record and the bench
        rec["grip_events"] = grips  # drops and releases of the sponge, robot side
        last = next((a for a in reversed(rec["attempts"]) if a.get("program")), None)
        if rec["final"] and rec["final"].get("claim") == "task_done" and last is not None \
                and rec["final"].get("source") == "model" and last.get("host_records_span"):
            if incidents or grips or rec.get("done_refusals"):
                # Never a library candidate from an episode with a robot-felt contact incident or a refused done(); nor
                # with a grip event (cannot happen with the harness rule above; kept as a second barrier)
                rec["library_candidate_refused"] = {"contact_incidents": len(incidents), "grip_events": len(grips),
                                                    "done_refusals": rec.get("done_refusals", 0)}
            else:
                # assumptions from the records of the FINAL program run only (not from earlier failed programs)
                a0, a1 = last["host_records_span"]
                ev = robot_side_evidence(host.records[a0:a1])
                ev.update(episode_contact_incidents=len(incidents), episode_grip_events=len(grips),
                          done_refusals=rec.get("done_refusals", 0))
                # (store-contract-1, option (a)): the supporting detections come only from the views delivered in the
                # episode opening and in the final run (every view of the episode, earlier attempts included), and the
                # evidence carries the contract fields of the final attempt's task evidence;
                # agent.memory.ProgramLibrary.store decides (not_stored:<class> in the bench row)
                views, view_ids = contract_views(host.records, host.observations(), rec.get("opening_end"), [a0, a1])
                assumptions = extract_assumptions(host.records[a0:a1], views)
                ev.update(store_contract_fields(host.records, [a0, a1], rec.get("opening_end"),
                                                last.get("task_evidence"), assumptions, view_ids))
                rec["library_candidate"] = {"program": last["program"], "assumptions": assumptions,
                                            "robot_side_evidence": ev}
        return rec

    @staticmethod
    def _end_invariant(rec: dict):
        """The end-of-episode invariant. A final claim task_done needs the task
        evidence of the attempt that ended the episode (the last one) to be "proven" under TASK_EVIDENCE_VERSION;
        otherwise the final is rewritten as the task_not_proven refusal (status done_refused, the
        help_requested:done_refused_task_not_proven claim), the attempt is marked done_refused, rec["done_refusals"]
        counts it and rec["harness_invariant"] keeps the rule, the verdict and the raw final. The rewritten final
        carries no library_key (the bench reports no verifier bit for it; agent.bench_agent._update_library also checks
        harness_invariant). No current branch should reach it: it covers the non-terminal path and any future one."""
        fin = rec.get("final") or {}
        if fin.get("claim") != "task_done":
            return
        att = rec["attempts"][-1] if rec.get("attempts") else {}
        ev = att.get("task_evidence") if isinstance(att, dict) else None
        if isinstance(ev, dict) and ev.get("verdict") == "proven" and ev.get("version") == TASK_EVIDENCE_VERSION:
            return
        rec["harness_invariant"] = {"rule": HARNESS_INVARIANT_RULE, "attempt": len(rec.get("attempts") or []) - 1,
                                    "verdict": ev.get("verdict") if isinstance(ev, dict) else None,
                                    "raw_final": dict(fin)}
        rec["final"] = {"status": DONE_REFUSED, "claim": DONE_REFUSED_TASK_CLAIM, "source": fin.get("source"),
                        "harness_invariant": True}
        rec["done_refusals"] = rec.get("done_refusals", 0) + 1
        if isinstance(att, dict) and rec.get("attempts"):
            att["done_refused"] = True
            att["harness_invariant"] = True

    def _close(self, rec, host, t0):
        rec["budget"] = host.budget()
        rec["counters"] = {"skill_calls": host.counters.skill_calls, "observations": host.counters.observations,
                           "queries": host.counters.queries, "refused": host.counters.refused,
                           "precondition_refusals": host.counters.precondition_refusals,
                           "pick_precondition_refusals": host.counters.pick_precondition_refusals,
                           # Calibrate refused before any motion (agent.host.GuardedHost.calibrate_precondition)
                           "hand_precondition_refusals": getattr(host.counters, "hand_precondition_refusals", 0),
                           "drawer_precondition_refusals": getattr(host.counters, "drawer_precondition_refusals", 0)}
        rec["delivered_observations"] = host.delivered_observations  # the marker, apart from counters.observations
        if self.record_scene_reading:
            # Records only: the robot's own scope of the task's tool when the episode ends (scope_now: not counted, not
            # recorded), a reference for the last request's scene reading
            sc = host.scope_now(NOT_FOUND_GUARD_CATEGORY)
            rec["scope_at_end"] = {"outcome": sc.get("outcome") if isinstance(sc, dict) else None,
                                   "not_cleared": ([str(r) for r in sc.get("not_cleared") or []]
                                                   if isinstance(sc, dict) else None)}
        rec["host_records"] = host.records
        rec["observations"] = host.observations()
        rec["wall_s"] = round(time.perf_counter() - t0, 3)
        return rec

    def _after_reuse(self, library, entry, outcome, host, n0, att, rec) -> str:
        """After a stored program ran: "accepted", "contradicted", "unverified", "failed", "ended_without_task"
        (give_up), "ended_on_help" (help, final in the loop) or "not_found_refused" (the guard refused its
        not_found), "help_review" (its search_incomplete reviewed once). Robot side only (never the verifier). Traces
        are kept in the library (uses, events, invalidation reason); nothing is erased."""
        status, err = outcome["status"], outcome.get("error") or {}
        ref = entry.get("entry_id") or entry["key"]
        span = host.records[n0:]
        post = check(entry, host.observations(), host.records)
        att["memory_check_after_run"] = post
        code = str(err.get("code", ""))
        if code.startswith("reuse_precondition_"):
            kind = code[len("reuse_precondition_"):]
            why = str(err.get("message", ""))[:300]
            library.note_use(ref, f"precondition_{kind}")
            library.note_event(ref, {"event": "precondition_" + kind, "why": why})
            rec["memory"]["events"].append({"key": entry["key"], "event": "precondition_" + kind, "why": why})
            if kind == "contradicted" or post["result"] == "contradicted":
                library.invalidate(ref, "precondition contradicted at the moment of use: " + why, by="robot_side")
                rec["memory"]["invalidated_after_run"] = entry["key"]
                return "contradicted"
            return "unverified"  # not seen is not absent: kept valid, not used in this episode
        if status == NOT_FOUND_REFUSED:  # the guard refused the stored program's not_found: invalidated
            why = "not_found refused on reuse: " + str(err.get("message", ""))[:200]
            library.note_use(ref, f"{status}:{outcome.get('refused_claim')}")
            library.invalidate(ref, why, by="robot_side")
            rec["memory"]["invalidated_after_run"] = entry["key"]
            rec["memory"]["events"].append({"key": entry["key"], "event": "not_found_refused_on_reuse",
                                            "why": why[:120]})
            return "not_found_refused"
        if status == HELP_REVIEW:  # the stored program's search_incomplete is reviewed once: invalidated
            why = "ended on help(kind=search_incomplete) on reuse, reviewed once: " + str(rec.get("help_review_claim"))
            library.note_use(ref, f"{status}:{rec.get('help_review_claim')}")
            library.invalidate(ref, why[:300], by="robot_side")
            rec["memory"]["invalidated_after_run"] = entry["key"]
            rec["memory"]["events"].append({"key": entry["key"], "event": "ended_on_help_reviewed_on_reuse",
                                            "why": str(rec.get("help_review_claim"))[:120]})
            return "help_review"
        claim_done = status == "terminal" and outcome.get("claim") == "task_done"
        # help() is handled like give_up() (before, a stored program's help ended the episode with no model call); a
        # refused done() (status done_refused) is a failure. A stored program's help still invalidates the entry here,
        # but the loop ends the episode on it (final whatever its kind)
        ended = status == "terminal" and outcome.get("terminal") in ("give_up", "help")
        # A reused done() counts only with the episode's task evidence proven (att["task_evidence"], the same
        # agent.host.task_evidence as the guard; any pick and any scrub that returned ok in the reused span,
        # agent.memory.robot_side_evidence). A voided one was refused above (status done_refused).
        tev = att.get("task_evidence") if isinstance(att.get("task_evidence"), dict) else {}
        proven = tev.get("verdict") == "proven" and tev.get("version") == TASK_EVIDENCE_VERSION
        failed = status != "terminal" or ended or (claim_done and not proven)
        contradicted = post["result"] == "contradicted"
        library.note_use(ref, f"{status}:{outcome.get('claim')}" + (":contradicted" if contradicted else "")
                         + (":claim_without_checked_gestures" if claim_done and failed else ""))
        if contradicted and not failed and status == "terminal" and outcome.get("terminal") == "not_found":
            # an honest search outcome of the reused program stands (the bench judges it); the entry no
            # longer matches the site: invalidated, reason kept
            library.invalidate(ref, "observation during the reused run contradicts: " + "; ".join(
                f"{a.get('category', 'handle')} {a['why']}" for a in post["assumptions"] if a["status"] == "contradicted"),
                by="robot_side")
            rec["memory"]["invalidated_after_run"] = entry["key"]
            return "accepted"
        helped = status == "terminal" and outcome.get("terminal") == "help"  # final in the loop
        if failed and not contradicted and ended_by_pick_refusal(span) and not claim_done and status != DONE_REFUSED \
                and not contact_incidents(span) and not grip_events(span) and not helped:
            # The robot refused one of its picks (pose not fresh or too uncertain at the moment of use) and the stored
            # program then ended without the task, with no incident and no grip event: kept valid, not used in this
            # episode (like a reuse precondition). A refused done(), a claim without checked gestures, an incident or a
            # contradiction still invalidate it as before. Only when the refusal is what stopped it
            # (ended_by_pick_refusal); a failure of a gesture that ran after the refusal (a pick that missed, a scrub
            # without contact) invalidates it as before. Not after a help (final: nothing else runs in this episode, a
            # valid entry would end every later episode the same way).
            why = "a pick of the stored program was refused by the robot's pick precondition; it then ended with " + \
                str(outcome.get("claim") or status)
            library.note_event(ref, {"event": "precondition_pick_pose", "why": why[:300]})
            rec["memory"]["events"].append({"key": entry["key"], "event": "precondition_pick_pose", "why": why[:300]})
            return "unverified"
        if failed or contradicted:
            # The library reason of a reused done() names the task_evidence codes (library text only, never a prompt:
            # the model gets the generic "failed" note)
            codes = "; ".join(f"{k}: {', '.join(str(c)[:40] for c in (tev.get(k) or [])) or 'none'}"
                              for k in ("missing", "voided_by"))
            done_why = (f" (done() without a pick and a scrub proven robot side: {TASK_EVIDENCE_VERSION} "
                        f"{tev.get('verdict') or 'not evaluated'}; {codes})") if claim_done or (
                            status == DONE_REFUSED and str(err.get("code", "")) in (DONE_REFUSED_TASK_CODE,
                                                                                    DONE_REFUSED_HOLD_CODE)) else ""
            # A stored run that ended the episode physically says so (library text only, never a prompt)
            pe = self._physical_end() if failed else None
            pe_why = f" (episode physical end: {pe.get('code')})" if pe is not None else ""
            why = ("failed on reuse: " + str(outcome.get("claim") or status) + done_why + pe_why) \
                if failed else "observation during the reused run contradicts: " + "; ".join(
                    f"{a.get('category', 'handle')} {a['why']}" for a in post["assumptions"] if a["status"] == "contradicted")
            library.invalidate(ref, why, by="robot_side")
            rec["memory"]["invalidated_after_run"] = entry["key"]
            if ended:
                ev_ = {"key": entry["key"], "why": str(outcome.get("claim"))[:120],
                       "event": "ended_on_final_help_on_reuse" if helped else "ended_without_task_on_reuse"}
                if helped and ended_by_pick_refusal(span):
                    ev_["after_pick_refusal"] = True  # invalidated anyway
                rec["memory"]["events"].append(ev_)
                return "ended_on_help" if helped else "ended_without_task"
            return "failed" if failed else "contradicted"
        return "accepted"
