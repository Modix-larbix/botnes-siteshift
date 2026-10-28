"""The agent loop wired to the simulator (MuJoCo, Franka Panda) and scored by the bench-side evaluator.

run_agent_episode() goes through bench_task.run_task_episode with a program function, so the agent gets exactly what
the scripted chain gets (same world build, skill facade, simulated camera and perception, evaluator and budget):
  task text + dossier + robot-side observations -> prompt -> model client -> BotScript program -> isolated
  interpreter (sandbox-exec) -> GuardedHost -> FacadeBackend -> skill facade -> robot in MuJoCo;
then the evaluator (siteagent.taskcheck, simulator truth) scores the episode on the bench side. The loop and the
program only receive plain robot-side data; the verdict never reaches them.

The model client is a parameter (llm). The offline demo passes its own client (demo/demo_client.py), which serves
fixed programs and a recorded choice and calls no model. client_label() names what was actually used.

Each episode writes, next to the bench files (oracle_verdict.json, observable_feedback.json, trace.json):
agent_record.json (attempts, programs, run reports, memory checks, budgets, host records), requests.json (the exact
requests the client received) and, when a library is used, the library file of the sequence. Failures are kept like
successes. A library's site id is opaque (site_id_for: a hash, no kitchen seed); the bench identity of each stored
entry goes to a separate bench-side sidecar file.

run_sequence() runs several episodes on one kitchen and shares the client, ledger and technical-retry objects it is
given; without them it builds a mock client and an in-memory ledger per sequence (no model call, no money).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

from ..bench_task import run_task_episode
from ..feedback import public_task
from ..perception.contract import to_dict
from .. import taskcheck
from .api import SKILL_API_VERSION
from .host import BACKEND_RAISED_STATUS, FacadeBackend, internal_entries, internal_incident, internal_release
from .ledger import MemoryLedger
from .limits import load_limits
from .llm import MockLLMClient
from .loop import AgentLoop, TechnicalRetries
from .memory import ProgramLibrary

PROGRAM_LABEL = "C_mock"  # default of the mock wiring runs; see client_label()
PRELUDE = ("calibrate", "look_pose")  # the first two steps of the scripted chain S, run by the loop
DEFAULT_INFO_PROFILE = "s5-info-0.9"  # AgentLoop's default and real_runner.DEFAULT_INFO_PROFILE (tested)
DEV_WORLD_SEQUENCE = "dev_world_single"  # development-only, run with a world_dev spec only
_WORLD_DEV_SPEC = re.compile(r"[a-z0-9][a-z0-9\-]{0,40}:([a-z][a-z0-9_]{0,40})")  # "<version>:<mode>" (checked by
# the launcher through world_dev's own parser; here only its shape, for the folder name)


def profile_kwargs(info_profile: str = DEFAULT_INFO_PROFILE, record_scene_reading: bool = False) -> dict:
    """The information-profile arguments, given to AgentLoop (and from run_sequence to run_agent_episode) only
    when they differ from the defaults: a default call is then the same call as without these arguments, argument for
    argument."""
    kw = {}
    if info_profile != DEFAULT_INFO_PROFILE:
        kw["info_profile"] = info_profile
    if record_scene_reading:
        kw["record_scene_reading"] = True
    return kw


def client_label(llm) -> dict:
    """{label, client_fr, requests_note, real}: what the evidence must say about this client."""
    kind = getattr(llm, "client_kind", None)
    if kind == "mock":
        return {"label": "C_mock", "real": False, "client_fr": "client simule (programmes ecrits a la main, pas Nemotron)",
                "requests_note": "requests as received by the model client (mock: nothing left the machine)"}
    if kind == "real":
        from .real_client import FAKE_TRANSPORT_KIND, REAL_TRANSPORT_KIND
        kind_t = getattr(llm, "transport_kind", None)
        if kind_t == REAL_TRANSPORT_KIND:
            return {"label": "C_real", "real": True,
                    "client_fr": "client reel (Token Factory, Nemotron), cout calcule par jeton",
                    "requests_note": "requests sent to Token Factory by the guarded real client (no header, no key)"}
        if kind_t == FAKE_TRANSPORT_KIND:
            return {"label": "C_fake_transport", "real": False,
                    "client_fr": "client reel sur transport fictif (essai a blanc, aucun appel reseau, cout sur usage fictif)",
                    "requests_note": "requests given to a FAKE transport (dry run: nothing left the machine)"}
        return {"label": "C_unknown_transport", "real": None,
                "client_fr": "client reel sur transport non identifie (cout incertain)",
                "requests_note": "requests given to an unidentified transport (whether they left the machine is unknown)"}
    return {"label": "C_unknown", "real": False, "client_fr": "client non identifie",
            "requests_note": "requests as received by an unidentified model client"}


def call_log_rows(rec: dict, label: dict, meta: dict) -> list[dict]:
    """One row per model call of the episode (bench side). No header or key ever reaches this layer."""
    rows = []
    for i, a in enumerate(rec.get("attempts", [])):
        if a.get("source") != "model":
            continue
        outcome = a.get("outcome") or {}
        for j, c in enumerate(a.get("calls", [])):
            usage = c.get("usage") or {}
            answered = c.get("status", "").startswith("answered")
            nothing_sent = bool(c.get("nothing_sent"))
            rows.append({
                **meta, "program_label": label["label"], "client": label["client_fr"], "attempt": i, "call": j,
                "status": c.get("status"), "model_id": c.get("model_id"), "prompt_version": a.get("prompt_version"),
                "prompt_sha256": a.get("prompt_sha256"), "request_sha256": a.get("request_sha256"),
                "prompt_chars": a.get("prompt_chars"), "correction_of": a.get("correction_of"),
                "request_prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"), "usage_source": c.get("usage_source"),
                "finish_reason": c.get("finish_reason"), "latency_s": c.get("latency_s"),
                # 0.2: an answer cut at the output limit is flagged, with the limit it was sent with
                "truncated_at_output_limit": c.get("finish_reason") == "length" if answered else None,
                "max_completion_tokens": a.get("max_completion_tokens"),
                "n_language_errors": (outcome.get("error") or {}).get("n_errors") if answered else None,
                "reservation_id": c.get("reservation_id"), "computed_cost_usd": c.get("computed_cost_usd"),
                "cost_status": c.get("cost_status"), "uncertain_cost": c.get("cost_status") != "settled",
                "uncertain_usd": c.get("uncertain_usd"), "nothing_sent": nothing_sent,
                "uncertain_note": ("nothing sent, reservation kept conservatively" if nothing_sent
                                   else "reservation kept: the provider's charge is not known" if
                                   c.get("cost_status") != "settled" else None),
                "activity": a.get("activity"), "reasoning_chars": c.get("reasoning_chars"),
                # Provider's optional usage.completion_tokens_details.reasoning_tokens (recorded, never a cost)
                "reasoning_tokens_reported": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
                if isinstance(usage.get("completion_tokens_details"), dict) else None,
                "reasoning_text": a.get("reasoning_text") if answered else None,
                "response_text": a.get("response_text") if answered else None,
                "parsed_program": a.get("program") if answered else None,
                "parse_or_model_error": a.get("error") if answered or j == len(a.get("calls", [])) - 1 else None,
                "run_status": outcome.get("status") if answered else None,
                "run_error": outcome.get("error") if answered else None,
                "run_claim": outcome.get("claim") if answered else None,
                # review (stage gate measures): lines of the parent's language warnings, and the robot skill calls the
                # program made (parent count), to see programs that failed at once on a warned call
                "language_warning_lines": sorted({w.get("line") for w in outcome.get("language_warnings") or []
                                                  if isinstance(w, dict) and type(w.get("line")) is int})
                if answered else None,
                "run_skill_calls": (outcome.get("counters") or {}).get("skill_calls") if answered else None,
                # The program's own terminal claim, before the harness rules (not_found guard, help review)
                "program_claim": (a.get("program_terminal") or {}).get("claim") if answered else None})
    return rows


def v44_fields(rec: dict | None) -> dict:
    """What the analysis needs kept apart (records only): the claims the programs themselves ended on
    (attempt program_terminal), the not_found guard's decisions and refusals (agent.loop.not_found_guard), the final
    claim's raw program claim when the guard ended the episode, the final help (kind, final_help: nothing ran after
    it) and the help review (count, trigger). The review's source (model or memory), the guard's absence_claim and
    ends_because, and the attempts whose reached help kept an unreadable why. The trigger of every review and the
    number of wrist view notes (agent.host.wrist_view_notes) in the run reports. The guard's new_observation and the
    reprises refused for no new observation, so the analysis can show per episode the number of reviews
    (help_reviews) and of not_found corrections (not_found_refusals) and any runaway (the row's final_status
    stopped_budget marks a stop by the budget or the episode wall time)."""
    rec = rec or {}
    fin = rec.get("final") or {}
    return {"program_terminal_claims": [(a.get("program_terminal") or {}).get("claim")
                                        for a in rec.get("attempts", [])],
            "not_found_refusals": rec.get("not_found_refusals", 0),
            "not_found_guard": [{k: g.get(k) for k in ("attempt", "source", "raw_claim", "absence_claim",
                                                       "scope_outcome", "decision", "refusal", "ends_because",
                                                       "new_observation")}
                                for g in rec.get("not_found_guard", [])],
            "final_raw_claim": fin.get("raw_claim"), "final_help": bool(fin.get("final_help")),
            "final_help_kind": fin.get("help_kind"), "help_reviews": rec.get("help_reviews", 0),
            "help_review_trigger": rec.get("help_review_trigger"),
            "help_review_source": rec.get("help_review_source"),
            "help_why_unreadable": sum(1 for a in rec.get("attempts", []) if a.get("help_why_unreadable")),
            # Every review's trigger (no count in limits 0.8) and the robot-side wrist view notes
            "help_review_triggers": list(rec.get("help_review_triggers", [])),
            "wrist_view_notes": sum(len((a.get("run_report") or {}).get("robot_notes") or [])
                                    for a in rec.get("attempts", [])),
            # Reprises refused by the progress precondition (no new observation since the last)
            "reprises_without_new_observation": rec.get("reprises_without_new_observation", 0)}


V47_EVIDENCE_KEYS = ("verdict", "missing", "voided_by", "pick_index", "pick_track", "pick_attempt", "pick_source",
                     "scrub_index", "bowl_track", "scrub_attempt", "scrub_source", "bowl_tray_sightings", "holding_now")


def v47_fields(rec: dict | None) -> dict:
    """What the analysis needs kept apart about task evidence and internal actions (records only, robot side; never the
    verifier):
    - done_evidence: one row per attempt whose outcome claimed done (att["task_evidence"], agent.host task_evidence:
      task-evidence-1 or task-evidence-2, whose version is in system_versions): attempt, source, the refusal code
      if done() was refused (an incident or grip event first, then the evidence), and the evidence's verdict, codes,
      pick and target scrub (record index, track, attempt and source: provenance) and bowl_tray_sightings (the matched
      bowl track's landing_tray sightings before the scrub, a margin worth reporting);
    - final_evidence_source: for a final task_done, "memory" when its pick or scrub came from a stored program's run,
      else "model" (successes resting on memory evidence are reported apart);
    - internal_events: the internal robot calls itemized in the host records (agent.host internal-actions-1) by origin,
      the internal ones that stopped on a contact, the internal releases, the unitemized ones and the backend_raised
      records;
    - clearances: the drawer release clearances that ran (ok or not) and the skipped ones by reason
      (refused_before_motion, no_tracked_handle_near);
    - hand_precondition_refusals, drawer_precondition_refusals: calibrate refused before any motion;
    - harness_invariant: the end-of-episode invariant rewrote the final (agent.loop._end_invariant);
    - system_versions of the record."""
    rec = rec or {}
    rows = []
    for i, a in enumerate(rec.get("attempts", [])):
        ev = a.get("task_evidence")
        if not isinstance(ev, dict):
            continue
        err = (a.get("outcome") or {}).get("error") or {}
        rows.append({"attempt": i, "source": a.get("source"), "done_refused": bool(a.get("done_refused")),
                     "refusal_code": err.get("code") if a.get("done_refused") else None,
                     **{k: ev.get(k) for k in V47_EVIDENCE_KEYS}})
    fin = rec.get("final") or {}
    src = None
    if fin.get("claim") == "task_done" and rows:
        last = rows[-1]
        src = "memory" if "memory" in (last.get("pick_source"), last.get("scrub_source")) else "model"
    hosts = rec.get("host_records") or []
    by_origin, inc, rel, unit, raised = {}, 0, 0, 0, 0
    ran, skipped = {"ok": 0, "failed": 0}, {}
    for r in hosts:
        if not isinstance(r, dict) or r.get("kind") != "skill":
            continue
        for e in internal_entries(r):
            o = str(e.get("origin"))
            by_origin[o] = by_origin.get(o, 0) + 1
        inc += internal_incident(r) is not None
        rel += bool(internal_release(r))
        unit += r["internal_unitemized"] if type(r.get("internal_unitemized")) is int else 0
        raised += r.get("status") == BACKEND_RAISED_STATUS
        cl = r.get("clearance")
        if isinstance(cl, dict) and cl.get("ran") is True:
            ran["ok" if cl.get("ok") is True else "failed"] += 1
        elif isinstance(cl, dict) and cl.get("ran") is False:
            k = str(cl.get("skipped"))
            skipped[k] = skipped.get(k, 0) + 1
    cnt = rec.get("counters") or {}
    return {"done_evidence": rows, "final_evidence_source": src,
            "internal_events": {"entries_by_origin": by_origin, "incidents": inc, "releases": rel, "unitemized": unit,
                                "backend_raised": raised},
            "clearances": {"ran": ran, "skipped": skipped},
            "hand_precondition_refusals": cnt.get("hand_precondition_refusals", 0),
            "drawer_precondition_refusals": cnt.get("drawer_precondition_refusals", 0),
            "harness_invariant": bool(rec.get("harness_invariant")), "system_versions": rec.get("system_versions")}


PHYSICAL_END_ROW_KEYS = ("code", "recognised_at", "attempt", "sim_s", "skill_calls", "rule")
C2_DRAWER_NOTES = ("close_drawer_short", "close_drawer_unverified")  # agent.host drawer-booking-2


def v471_fields(rec: dict | None) -> dict:
    """What the analysis needs kept apart about the physical end and drawer bookings (records only, robot side; never
    the
    verifier):
    - physical_end: rec["physical_end"] as agent.loop recorded it (episode-physical-end-1), reduced to
      PHYSICAL_END_ROW_KEYS, or None. recognised_at "at_final" marks an episode that ended otherwise while its end held;
      any other value is an episode the loop ended at its physical end (final_status stopped_physical_end, or a give_up
      or stored give_up keeping its claim);
    - drawer_notes: the host records' drawer gestures counted by drawer_note (agent.host DRAWER_NOTES;
      close_drawer_short and close_drawer_unverified are also counted under c2_drawer_notes, even when 0);
    - close_checks: the records-only close_check of the ok opposed closes (agent.host.close_travel_check), counted by
      sufficient True / False / unreadable;
    - reference_events: rec["perception_reference_events"] (the episode perception's drawer-reference-2 events,
      perception.estimate.Perception.reference_events, written by run_agent_episode) counted by rule, or None when
      the record has none (a toy kitchen): what the replays count."""
    rec = rec or {}
    pe = rec.get("physical_end")
    notes, checks = {}, {"sufficient": 0, "insufficient": 0, "unreadable": 0}
    for r in rec.get("host_records") or []:
        if not isinstance(r, dict) or r.get("kind") != "skill":
            continue
        n = r.get("drawer_note")
        if isinstance(n, str):
            notes[n] = notes.get(n, 0) + 1
        cc = r.get("close_check")
        if isinstance(cc, dict):
            s = cc.get("sufficient")
            checks["sufficient" if s is True else "insufficient" if s is False else "unreadable"] += 1
    evs = rec.get("perception_reference_events")
    ref = None
    if isinstance(evs, list):
        ref = {}
        for e in evs:
            rule = e.get("rule") if isinstance(e, dict) else None
            if isinstance(rule, str):
                ref[rule] = ref.get(rule, 0) + 1
    return {"physical_end": {k: pe.get(k) for k in PHYSICAL_END_ROW_KEYS} if isinstance(pe, dict) else None,
            "drawer_notes": notes, "c2_drawer_notes": {k: notes.get(k, 0) for k in C2_DRAWER_NOTES},
            "close_checks": checks, "reference_events": ref}


def run_agent_episode(kitchen: str, variant: str, noise: int, out_dir: str | Path, llm, ledger,
                      library: ProgramLibrary | None, limits=None, retries: TechnicalRetries | None = None,
                      crit: dict | None = None, label: str | None = None, gif_path: str | None = None,
                      calls_log: str | Path | None = None, calls_meta: dict | None = None,
                      info_profile: str = DEFAULT_INFO_PROFILE, record_scene_reading: bool = False) -> dict:
    """One episode of the agent on the simulator. Returns the bench row plus agent fields.
    label: None = from the client actually used (client_label). info_profile and record_scene_reading go to
    AgentLoop."""
    who = client_label(llm)
    label = label or who["label"]
    crit = crit or taskcheck.load_criteria()
    lim = limits or load_limits()
    retries = retries or TechnicalRetries(lim.agent.max_technical_retries_per_run)
    holder: dict = {}
    n_req0 = len(getattr(llm, "received", []))

    def program_fn(facade, per, dossier, hook):
        backend = FacadeBackend(facade, per, per.model, dossier, to_dict)
        loop = AgentLoop(llm, ledger, lim, isolate=True, retries=retries, criteria_version=crit["version"],
                         **profile_kwargs(info_profile, record_scene_reading))
        holder["loop"] = loop  # its live record survives an exception or an interruption
        rec = loop.run_episode(public_task(crit), backend, dossier.to_dict(), library=library, prelude=PRELUDE)
        # drawer-reference-2's records-only events of the episode's perception
        rec["perception_reference_events"] = [dict(e) for e in getattr(per, "reference_events", [])]
        holder["rec"], holder["dossier"] = rec, dossier.to_dict()
        return {"claim": rec["final"]["claim"], "agent_final": rec["final"]}, None

    out = Path(out_dir)
    written = []

    def write_records(rec, partial: bool):
        if rec is None or written:
            return
        written.append(True)
        if partial:
            rec = {**rec, "partial": "episode interrupted after this point; record as it stood"}
        (out / "agent_record.json").write_text(json.dumps(rec, indent=1, default=str))
        reqs = getattr(llm, "received", [])[n_req0:]
        (out / "requests.json").write_text(json.dumps(
            {"label": who["requests_note"], "program_label": label, "client": who["client_fr"],
             "model_client": getattr(llm, "model_id", "?"), "requests": reqs}, indent=1))
        if calls_log is not None:
            meta = {"kitchen": kitchen, "variant": variant, "noise_seed": noise, "episode_dir": str(out),
                    "partial_record": partial,
                    # The request option agent.loop sends with these limits (None: not sent)
                    "reasoning_effort": getattr(lim.agent, "reasoning_effort", None), **(calls_meta or {})}
            with open(calls_log, "a", encoding="utf-8") as f:
                for r in call_log_rows(rec, {**who, "label": label}, meta):
                    f.write(json.dumps(r, default=str) + "\n")
                f.flush()
                os.fsync(f.fileno())

    try:
        row = run_task_episode(label, kitchen, variant, noise, out_dir=str(out), program_fn=program_fn, crit=crit,
                               gif_path=gif_path)
    except BaseException:  # Ctrl-C or a bench failure: keep what the paid calls returned, then stop
        loop_ = holder.get("loop")
        out.mkdir(parents=True, exist_ok=True)
        write_records(holder.get("rec") or getattr(loop_, "current_rec", None), holder.get("rec") is None)
        raise
    rec = holder.get("rec")
    partial = False
    if rec is None and getattr(holder.get("loop"), "current_rec", None) is not None:
        rec, partial = holder["loop"].current_rec, True  # the loop raised (bench execution_error): partial record
    lib_events = []
    if library is not None and rec is not None and not partial:
        lib_events = _update_library(library, rec, row, holder["dossier"], crit)
    write_records(rec, partial)
    fin = (rec or {}).get("final") or {}
    att = (rec or {}).get("attempts", [])
    row.update(client=who["client_fr"], client_real=who["real"], agent={
        "final_status": fin.get("status"), "final_source": fin.get("source"), "final_reason": fin.get("reason"),
        "call_statuses": [c.get("status") for a in att for c in a.get("calls", [])], "partial_record": partial,
        "model_calls": (rec or {}).get("model_calls"), "corrections": (rec or {}).get("corrections"),
        "attempt_sources": [a.get("source") for a in att],
        "attempt_claims": [(a.get("outcome") or {}).get("claim") or (a.get("error") or {}).get("code")
                           or ((a.get("outcome") or {}).get("error") or {}).get("code") for a in att],
        "reused_key": ((rec or {}).get("memory") or {}).get("reused_key"),
        "memory_checks": [c["result"] for c in ((rec or {}).get("memory") or {}).get("checks", [])],
        "precondition_refusals": ((rec or {}).get("counters") or {}).get("precondition_refusals"),
        "host_counters": (rec or {}).get("counters"), "host_budget": (rec or {}).get("budget"),
        "prelude": (rec or {}).get("prelude"), "agent_wall_s": (rec or {}).get("wall_s"),
        "library_events": lib_events, "validation": None if library is None else library.validation,
        # robot-side contact incidents, done() refused by the harness, stored program's help/give_up
        "robot_contact_incidents": len((rec or {}).get("contact_incidents") or []),
        "done_refusals": (rec or {}).get("done_refusals", 0),
        "memory_returned_to_model": [a.get("returned_to_model") for a in att if "returned_to_model" in a],
        "model_id": getattr(llm, "model_id", None), **v44_fields(rec), **v47_fields(rec), **v471_fields(rec)})
    return row


def _update_library(library: ProgramLibrary, rec: dict, row: dict, dossier: dict, crit: dict) -> list[str]:
    """Bench side, after the verifier: what the library's declared condition allows."""
    ev = []
    success = bool(row.get("success"))
    fin = rec.get("final") or {}
    cand = rec.get("library_candidate")
    api_version = rec.get("skill_api_version", SKILL_API_VERSION)  # the profile's API (a record without it: 0.8)
    counter = sum(1 for _ in _sidecar_rows(library)) + 1
    prov = {"model_id": next((c.get("model_id") for a in rec["attempts"] for c in a.get("calls", [])
                              if c.get("model_id")), None), "prompt_version": rec["prompt_version"],
            "episode_counter": counter}  # opaque; the bench identity goes to the bench-side sidecar
    if library.validation == "external_validation":
        # A final rewritten by the end-of-episode invariant carries no library_key; it is never reported either
        if fin.get("source") == "memory" and fin.get("library_key") and not rec.get("harness_invariant"):
            library.note_verdict(fin.get("library_entry_id") or fin["library_key"], success,
                                 f"external:verifier:{crit['version']}")
            ev.append(f"verdict_bit:{'validated' if success else 'rejected'}")
        if cand and success:
            try:
                e = library.store(cand["program"], dossier, api_version, crit["version"], cand["assumptions"],
                                  f"external:verifier_success:{crit['version']}", prov,
                                  evidence=cand.get("robot_side_evidence"))
                ev.append("stored:" + e["key"][:12])
            except ValueError as err:
                ev.append(f"not_stored:{type(err).__name__}")
    else:
        if cand:
            try:
                e = library.store(cand["program"], dossier, api_version, crit["version"], cand["assumptions"],
                                  "robot_side:pick_and_scrub_ok", prov, evidence=cand.get("robot_side_evidence"))
                ev.append("stored:" + e["key"][:12])
            except ValueError as err:
                ev.append(f"not_stored:{type(err).__name__}")
    stored = next((x[len("stored:"):] for x in ev if x.startswith("stored:")), None)
    _sidecar_append(library, {"episode_counter": counter, "kitchen": row["kitchen"], "variant": row["variant"],
                              "noise_seed": row["noise_seed"], "stored_entry_key12": stored,
                              "reused_entry_id": (rec.get("memory") or {}).get("reused_entry_id")})
    mem = rec.get("memory") or {}
    if mem.get("invalidated_after_run"):
        ev.append("invalidated_after_run:" + mem["invalidated_after_run"][:12])
    for c in mem.get("checks", []):
        if c["result"] == "contradicted":
            ev.append("invalidated_before_run:" + c["key"][:12])
    for e in mem.get("events", []):
        ev.append(f"{e['event']}:{e['key'][:12]}")
    return ev


def _sidecar_path(library: ProgramLibrary) -> Path:
    return library.path.with_name(f"bench_sidecar_{library.site_id}.jsonl")


def _sidecar_rows(library: ProgramLibrary):
    p = _sidecar_path(library)
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                yield json.loads(line)


def _sidecar_append(library: ProgramLibrary, row: dict):
    with open(_sidecar_path(library), "a") as f:
        f.write(json.dumps({"label": "BENCH SIDE (never read by the robot or the loop)", **row}) + "\n")


def site_id_for(seq: str, kitchen: str, world_dev: str | None = None) -> str:
    """Opaque robot-side site id of a (sequence, kitchen) library: no kitchen seed in it. With a development
    world, hashed over (sequence, world_dev, kitchen); without one, the original formula (unchanged)."""
    if world_dev is None:
        return "site-" + hashlib.sha256(f"{seq}:{kitchen}".encode()).hexdigest()[:12]
    return "site-" + hashlib.sha256(f"{seq}:{world_dev}:{kitchen}".encode()).hexdigest()[:12]


def world_dev_mode(world_dev: str) -> str:
    """The mode of a world_dev spec "<version>:<mode>" (its shape only; the launcher parses it with world_dev's
    own parser). ValueError otherwise."""
    m = _WORLD_DEV_SPEC.fullmatch(str(world_dev))
    if m is None:
        raise ValueError(f"world_dev must be '<version>:<mode>' (got {world_dev!r})")
    return m.group(1)


def sequence_folder(kitchen: str, seq: str, world_dev: str | None = None) -> Path:
    """The sequence folder under the out root: <kitchen>/<sequence>, or <kitchen>/<sequence>@<mode> with a
    development world."""
    return Path(f"{kitchen}") / (seq if world_dev is None else f"{seq}@{world_dev_mode(world_dev)}")


# ====================================================================== scenario sequences
SEQUENCES = {
    # one library per sequence (a site), episodes in order; the dossier is the same in every variant
    "memory_chain": ("correct", "correct", "moved", "moved", "absent"),
    "first_moved": ("moved",),
    "first_absent": ("absent",),
    "external_chain": ("correct", "correct"),
    # the sponge is put back after an adaptation: the stored search program finds it at its old place,
    # the precondition at the moment of the pick refuses it (it relied on the other drawer)
    "restored_chain": ("moved", "correct"),
    # Development only: one episode of a development world (world_dev modes run on variant "correct"); run_sequence and
    # the launcher run it only with a world_dev spec; no validation generator names it
    DEV_WORLD_SEQUENCE: ("correct",),
}
SEQUENCE_FR = {
    "memory_chain": "chaine memoire : juste (1re adaptation), juste (reutilisation), deplacee (changement apres "
                    "memorisation), deplacee (reutilisation du programme adapte), absente",
    "first_moved": "eponge deplacee, bibliotheque vide (premiere adaptation)",
    "first_absent": "eponge absente, bibliotheque vide",
    "external_chain": "condition declaree memoire avec validation externe : juste, juste",
    "restored_chain": "deplacee (adaptation memorisee), puis eponge remise a sa place (precondition refusee a la prise)",
    DEV_WORLD_SEQUENCE: "monde de developpement (world_dev), un episode, bibliotheque vide",
}


def run_sequence(kitchen: str, seq: str, noise0: int, out_root: str | Path, validation: str = "robot_side",
                 gifs: bool = False, llm=None, ledger=None, retries: TechnicalRetries | None = None,
                 limits=None, calls_log: str | Path | None = None, stop_check=None,
                 info_profile: str = DEFAULT_INFO_PROFILE, world_dev: str | None = None,
                 record_scene_reading: bool = False) -> list[dict]:
    """A sequence of episodes on one kitchen with one library (one site). llm / ledger / retries: the
    RUN's shared objects (agent.real_runner); when omitted, a mock client and a per-sequence MemoryLedger and
    TechnicalRetries (mock wiring runs only, no network, no money). stop_check(row) -> reason or None (real runs):
    called after each episode; a reason stops the sequence there (the remaining episodes are not run, and say so in
    rows.jsonl). info_profile and record_scene_reading go to every episode's AgentLoop. world_dev (a spec
    "<version>:<mode>") names the development world the CALLER applies around this call (the launcher, world_dev's
    applied()); here it only sets the folder <kitchen>/<sequence>@<mode>, the site id and the rows' world_dev field.
    dev_world_single runs only with it, and world_dev only with dev_world_single (ValueError, before any folder)."""
    if (seq == DEV_WORLD_SEQUENCE) != (world_dev is not None):
        raise ValueError(f"{DEV_WORLD_SEQUENCE} runs only with a world_dev spec, and world_dev only with "
                         f"{DEV_WORLD_SEQUENCE} (got sequence {seq!r}, world_dev {world_dev!r})")
    out_root = Path(out_root) / sequence_folder(kitchen, seq, world_dev)
    out_root.mkdir(parents=True, exist_ok=True)
    lim = limits or load_limits()
    shared = llm is not None
    if llm is None:
        from .mock_policy import MOCK_MODEL_ID, answer
        llm = MockLLMClient(answer, model_id=MOCK_MODEL_ID)
    ledger = ledger if ledger is not None else MemoryLedger()
    retries = retries if retries is not None else TechnicalRetries(lim.agent.max_technical_retries_per_run)
    crit = taskcheck.load_criteria()
    library = ProgramLibrary(site_id_for(seq, kitchen, world_dev), out_root, validation=validation)
    (out_root / "site_identity.json").write_text(json.dumps(
        {"label": "BENCH SIDE (never read by the robot or the loop)", "site_id": library.site_id,
         "kitchen": kitchen, "sequence": seq, **({"world_dev": world_dev} if world_dev is not None else {})}, indent=1))
    rows = []
    for i, variant in enumerate(SEQUENCES[seq]):
        t0 = time.perf_counter()
        noise = noise0 + i
        meta = {"sequence": seq, "step": i + 1, **({"world_dev": world_dev} if world_dev is not None else {})}
        try:
            ep = out_root / f"e{i + 1}_{variant}_n{noise}"
            row = run_agent_episode(kitchen, variant, noise, ep, llm, ledger, library, lim, retries, crit,
                                    gif_path=str(ep / "episode.gif") if gifs else None,
                                    **({"calls_log": calls_log, "calls_meta": meta} if calls_log is not None else {}),
                                    **profile_kwargs(info_profile, record_scene_reading))
        except Exception as e:  # noqa: BLE001 (a bench failure is kept as a row)
            row = {"program": client_label(llm)["label"], "kitchen": kitchen, "variant": variant, "noise_seed": noise,
                   "bench_exception": f"{type(e).__name__}: {e}"[:300]}
        row.update(sequence=seq, step=i + 1, wall_s_total=round(time.perf_counter() - t0, 1))
        if world_dev is not None:
            row["world_dev"] = world_dev
        rows.append(row)
        with open(out_root / "rows.jsonl", "a") as f:
            f.write(json.dumps(row, default=float) + "\n")
        why = stop_check(row) if stop_check is not None else None
        if why:
            with open(out_root / "rows.jsonl", "a") as f:
                f.write(json.dumps({"not_run": [f"e{j + 1}_{v}" for j, v in enumerate(SEQUENCES[seq]) if j > i],
                                    "stopped": why}) + "\n")
            break
    if not shared:  # a shared (run) ledger is written once by the runner
        (out_root / "ledger_mock.json").write_text(json.dumps({"summary": ledger.summary(), "entries": ledger.entries},
                                                              indent=1, default=str))
    return rows
