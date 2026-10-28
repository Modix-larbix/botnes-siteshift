"""Program library per site: reuse a stored program while the site is compatible, check preconditions at use.

One library is one site (a JSON file named after an opaque site id); nothing is shared between sites. An entry key is
the sha256 of the dossier version and content, the skill API version, the criteria version and the program's
assumptions (where the picked object and the scrubbed target were perceived, and the drawer handles it pulled).

Every library declares one validation condition, fixed for the life of its file:
- "robot_side" (main): a program is stored only on robot-side evidence of its own run (claim task_done, a pick and a
  scrub that returned ok in the final run). The evaluator's verdict never reaches the library; a stored program may
  therefore be wrong, which the bench measures;
- "external_validation": the bench stores a program only after the evaluator validated the episode and reports the
  verdict for a reused program; a declared one-bit oracle channel, reported apart.
An entry is never deleted: invalidation keeps the program, its uses, the reason and who decided.

check() and object_state() read an episode's evidence in order (the most recent evidence about a category wins): a
detection at or away from the assumed place, a region inspection that says the unseen part could not hide the object
(could_hide False and enough coverage), and the robot's own picks and releases. A missing or unknown flag is never
read as absence. precondition() is the check the host applies before each critical gesture of a reused program.
Episodes with a contact incident, a grip event or a refused done() never produce a library candidate.

The offline demo runs without a library (library=None): nothing is read or stored.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

# 0.4: unique entry_id per entry (uses, events, verdicts, invalidations by id); 0.5: entries carry store_contract
LIBRARY_SCHEMA = "program-library-0.5"
# store() refuses evidence with episode_contact_incidents or done_refusals (IncidentEpisode), and with
# episode_grip_events (agent.host.grip_events); it requires the three counts in both conditions (EPISODE_EVIDENCE_KEYS;
# ConditionError when missing). New entries carry the storage contract STORE_CONTRACT (store-contract-2, with
# task-evidence-2); entries of an earlier contract are filtered at lookup and never rewritten.
STORE_CONTRACT = "store-contract-2"
TASK_EVIDENCE_VERSION_REQUIRED = "task-evidence-2"  # = agent.host.TASK_EVIDENCE_VERSION (tested)
ASSUMPTION_SOURCES_RULE = "opening_and_final_span"  # S2 (store-contract-1): where the stored assumptions come from
STORE_CONTRACT_KEYS = ("store_contract", "task_evidence_version", "task_evidence_verdict", "pick_in_final_run",
                       "target_scrub_in_final_run", "end_holding", "episode_sponge_picks_ok",
                       "unitemized_robot_calls_since_pick", "assumption_sources", "sponge_drawer_opened_in_final_run")
RAISED_STATUS = "backend_raised"  # = agent.host.BACKEND_RAISED_STATUS (a skill whose backend raised; a test checks)
VALIDATION_MODES = ("robot_side", "external_validation")
VALIDATION_FR = {"robot_side": "memoire observable (validation cote robot)",
                 "external_validation": "memoire avec validation externe (bit du verificateur, declare)"}
CLEAR_MIN_COVERAGE = 0.3  # = sitemodel.MIN_COVER (a test checks they match)
BOWL_TOP_ABOVE_CENTRE = 0.023  # perception.catalog.OBJECT_SHAPES["bowl"]["top_above_centre"] (test checks)
DRAWER_PREFIX = "drawer_unit:"
# Robot actions that can move objects (staleness). Arm motions and looks do not.
MOVING_SKILLS = ("open_drawer", "close_drawer", "pick", "place", "turn_over", "scrub", "spray", "press")


def _h(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _r(v, q=0.01):
    return [round(round(float(x) / q) * q, 3) for x in v]


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def dossier_hash(dossier: dict) -> str:
    return _h(dossier)


# ====================================================================== evidence timeline
def timeline(observations: list[dict], records: list[dict] | None = None) -> list[tuple[str, dict]]:
    """Ordered evidence: ("obs", observation) and ("act", skill record). With records (the host's
    call records, in call order) observations are placed where they were captured; without, the observations alone,
    in the given order (no action known). A skill whose backend raised (status RAISED_STATUS) is an action too (it
    may have moved: object_state makes every region's evidence stale after it)."""
    if records is None:
        return [("obs", o) for o in observations]
    by_id = {o.get("obs_id"): o for o in observations}
    out, used = [], set()
    for r in records:
        if r.get("status") == RAISED_STATUS and r.get("kind") == "skill":
            out.append(("act", r))
            continue
        if r.get("status") != "called":
            continue
        if r.get("kind") == "observe" and r.get("obs_id") in by_id:
            out.append(("obs", by_id[r["obs_id"]]))
            used.add(r["obs_id"])
        elif r.get("kind") == "skill":
            out.append(("act", r))
        elif r.get("kind") == "query" and r.get("name") == "holding" and type(r.get("value")) is bool:
            out.append(("act", r))  # the robot's own grip reading: a negative reading ends a hold
    # observations not matched to a record (e.g. given by a caller): before everything, oldest first
    rest = [("obs", o) for o in observations if o.get("obs_id") not in used]
    return rest + out


def _region_of_nearest(evidence_pos: dict, pos) -> str | None:
    best, bd = None, 0.08
    for region, p in evidence_pos.items():
        d = math.dist(p, pos)
        if d <= bd:
            best, bd = region, d
    return best


def object_state(category: str, region: str, position_m, tol_m: float, events) -> dict:
    """State of one assumed object (category at region/position) from the ordered evidence.
    Returns {status: confirmed|contradicted|unverified, fresh: bool, why, sighting (latest at the
    assumed place, if any), held_by_robot: bool}."""
    st = {"status": "unverified", "fresh": False, "why": "not observed in this episode", "sighting": None,
          "held_by_robot": False, "moved_by_robot": False, "evidence_step": None, "own_move_sightings": []}
    last_pos = {}  # region -> latest perceived position of any object category (to place actions)
    for i, (kind, ev) in enumerate(events):
        if kind == "obs":
            dets = ev.get("detections", [])
            for d in dets:
                if d.get("region") is not None and isinstance(d.get("position_m"), list):
                    last_pos[d["region"]] = d["position_m"]
            if st["held_by_robot"]:
                continue  # in the hand: its positions are the robot's own doing (state of the pick kept)
            mine = [d for d in dets if d.get("category") == category]
            near = [d for d in mine if d.get("region") == region
                    and math.dist(d["position_m"], position_m) <= tol_m]
            if near:
                st.update(status="confirmed", fresh=True, why="perceived at the assumed place",
                          sighting=near[0]["position_m"], evidence_step=i, moved_by_robot=False, own_move_sightings=[])
                continue
            if mine and st["moved_by_robot"]:
                # fresh sightings of where the robot's own move left it: a gesture aimed at one of them relies on this
                # view, not on the stored place (agent.memory.precondition)
                st.update(status="unverified", fresh=False, sighting=None, evidence_step=i,
                          own_move_sightings=[list(d["position_m"]) for d in mine],
                          why="perceived elsewhere after the robot itself moved it (its own move, not a change of the site)")
                continue
            if mine:
                where = sorted({str(d.get("region")) for d in mine})
                why = ("perceived in the region but beyond the tolerance" if region in where
                       else f"perceived elsewhere: {where}")
                st.update(status="contradicted", fresh=True, why=why, sighting=None, evidence_step=i)
                continue
            for rg in ev.get("regions", []):
                if rg.get("region") != region or category in rg.get("seen_categories", []):
                    continue
                ch = rg.get("could_hide")
                flag = ch.get(category) if isinstance(ch, dict) else None
                if flag is False and float(rg.get("coverage", 0.0)) >= CLEAR_MIN_COVERAGE:
                    if st["moved_by_robot"]:
                        st.update(status="unverified", fresh=False, sighting=None, evidence_step=i,
                                  why="not at the assumed place after the robot itself moved it")
                    else:
                        st.update(status="contradicted", fresh=True, sighting=None, evidence_step=i,
                                  why="region inspected without it; its unseen part cannot hide it")
                # flag True / None / missing: a partial view is not absence (no evidence)
        else:
            name, a, ok = ev.get("name"), ev.get("args") or {}, bool(ev.get("ok"))
            if st["held_by_robot"]:
                # The hold ends at a SUCCESSFUL place / turn_over / open_gripper, or when the robot's own grip reading
                # says it holds nothing (holding() -> False). Before, only place / turn_over ended it,
                # so "pick ok, open_gripper, new view elsewhere" kept a stale hold and refused a new pick. Internal
                # review: a FAILED release no longer ends the hold (the object may still be in the hand; later views of
                # it would have counted as own-move sightings and let a reused pick pass); a negative holding() reading
                # ends it. Keeping the hold can only refuse a reused pick (then the model is asked), never allow one.
                # Robot side only, no truth. agent.host internal-actions-1: so do the grip reading taken after a later
                # gesture (holding_after False) and an internal open_gripper or place that returned ok (the drawer
                # clearance).
                internal = ev.get("internal") if isinstance(ev.get("internal"), list) else []
                released = (name in ("place", "turn_over", "open_gripper") and ok) or (
                    name == "holding" and ev.get("value") is False)
                what = "grip reading: holding nothing" if name == "holding" else name
                if not released and any(isinstance(e, dict) and e.get("skill") in ("place", "open_gripper")
                                        and e.get("ok") is True for e in internal):
                    released, what = True, f"internal release during {name}"
                if not released and ev.get("kind") == "skill" and ev.get("holding_after") is False:
                    released, what = True, f"grip reading after {name}: holding nothing"
                if released:
                    st.update(held_by_robot=False, moved_by_robot=True, status="unverified", fresh=False, sighting=None,
                              own_move_sightings=[],
                              why=st["why"] + f"; then moved by the robot ({what}): not perceived at the assumed place since")
                continue
            raised = ev.get("status") == RAISED_STATUS  # the backend raised, the arm may have moved
            if name not in MOVING_SKILLS and not raised:
                continue
            if name == "open_drawer" and ev.get("reason") == "already_open":
                continue  # idempotent already_open, nothing moved (agent.host.ALREADY_OPEN_CONVENTION)
            if st["own_move_sightings"]:
                st["own_move_sightings"] = []  # an own-move sighting is fresh only until the next moving action
            if name == "pick" and ok and a.get("kind") == category and st["status"] == "confirmed" \
                    and st["sighting"] is not None and isinstance(a.get("pos"), list) \
                    and math.dist(a["pos"], st["sighting"]) <= tol_m:
                st.update(held_by_robot=True, why=st["why"] + "; then picked by the robot (own move)")
                continue
            if raised:
                affected = True  # whatever the skill, everything it could have touched is no longer fresh
            elif name in ("open_drawer", "close_drawer"):
                affected = region.startswith(DRAWER_PREFIX)
            elif name in ("pick", "scrub") and isinstance(a.get("pos", a.get("target_pos")), list):
                p = a.get("pos", a.get("target_pos"))
                affected = _region_of_nearest(last_pos, p) in (region, None)
            else:
                affected = True
            if affected and st["fresh"]:
                st.update(fresh=False, why=st["why"] + f"; stale after the robot's own {name}")
    return st


# ====================================================================== assumptions
def extract_assumptions(host_records: list[dict], observations: list[dict], tol_obj: float = 0.05,
                        tol_handle: float = 0.03) -> list[dict]:
    """From the robot-side record of a run: what the successful skills relied on."""
    dets = [d for o in observations for d in o.get("detections", [])]
    out = []

    def nearest(cat, pos):
        best, bd = None, 0.06
        for d in dets:
            if d["category"] == cat:
                dd = math.dist(d["position_m"], pos)
                if dd <= bd:
                    best, bd = d, dd
        return best

    for r in host_records:
        if r.get("status") != "called" or r.get("kind") != "skill" or not r.get("ok"):
            continue
        a = r.get("args", {})
        if r["name"] == "pick":
            d = nearest(a["kind"], a["pos"])
            if d is not None:
                out.append({"type": "object_at", "category": a["kind"], "region": d["region"],
                            "position_m": _r(d["position_m"]), "tol_m": tol_obj, "used_by": "pick"})
        elif r["name"] == "scrub":
            d = nearest("bowl", [a["target_pos"][0], a["target_pos"][1], a["target_pos"][2] - BOWL_TOP_ABOVE_CENTRE])
            if d is not None:
                out.append({"type": "object_at", "category": "bowl", "region": d["region"],
                            "position_m": _r(d["position_m"]), "tol_m": tol_obj, "used_by": "scrub"})
        elif r["name"] == "open_drawer":
            out.append({"type": "handle_at", "position_m": _r(a["handle_pos"]), "tol_m": tol_handle,
                        "used_by": "open_drawer"})
    uniq, seen = [], set()
    for a in out:
        k = _h(a)
        if k not in seen:
            seen.add(k)
            uniq.append(a)
    return uniq


def robot_side_evidence(host_records: list[dict]) -> dict:
    """What the robot itself knows about a run (no verifier): a pick and a scrub that returned ok."""
    skills = [r for r in host_records if r.get("status") == "called" and r.get("kind") == "skill"]
    return {"pick_ok": any(r["name"] == "pick" and r.get("ok") for r in skills),
            "scrub_ok": any(r["name"] == "scrub" and r.get("ok") for r in skills),
            "n_skill_calls": len(skills)}


# ====================================================================== storage contract
def contract_views(records: list[dict], observations: list[dict], opening_end, span) -> tuple[list[dict], list[str]]:
    """store-contract-1: the only views the stored assumptions may come from: those
    DELIVERED (an observe record with status called and its obs_id, agent.host.delivered_obs_id) by the records before
    opening_end (prelude and orientation: agent.loop rec["opening_end"]) and inside the final run [a0, a1). Returns
    (the observations in delivery order, their obs_ids). The latest observation of the log with an id wins (a backend
    reused across episodes keeps older views). opening_end None: no opening view (the stricter reading)."""
    oe = opening_end if type(opening_end) is int and opening_end >= 0 else 0
    a0, a1 = span
    ids = []
    for r in list(records[:oe]) + list(records[a0:a1]):
        oid = r.get("obs_id")
        if r.get("name") == "observe" and r.get("status") == "called" and type(oid) is str and oid and oid not in ids:
            ids.append(oid)
    by_id = {o.get("obs_id"): o for o in observations if isinstance(o, dict) and o.get("obs_id") in ids}
    return [by_id[i] for i in ids if i in by_id], ids


def _sponge_drawers(assumptions: list[dict]) -> set:
    """S3: the drawer regions of the stored sponge/pick assumptions."""
    return {str(a.get("region")) for a in assumptions if a.get("type") == "object_at" and a.get("used_by") == "pick"
            and a.get("category") == "sponge" and str(a.get("region", "")).startswith(DRAWER_PREFIX)}


def store_contract_fields(records: list[dict], span, opening_end, task_ev: dict, assumptions: list[dict],
                          view_ids: list[str]) -> dict:
    """store-contract-1: the contract fields of a library candidate's evidence (STORE_CONTRACT_KEYS), robot side:
    from the FINAL attempt's agent.host.task_evidence (task_ev, computed at its done() with span = that attempt's run),
    the episode's host records and the assumptions extracted from contract_views (view_ids). store() checks them."""
    a0, a1 = span
    ev = task_ev if isinstance(task_ev, dict) else {}
    ran = [r for r in records if r.get("kind") == "skill" and r.get("status") in ("called", RAISED_STATUS)]
    picks = sum(1 for r in ran if r.get("name") == "pick" and r.get("ok") is True
                and (r.get("args") if isinstance(r.get("args"), dict) else {}).get("kind") == "sponge")
    pi = ev.get("pick_index")
    since = records[pi:] if type(pi) is int and 0 <= pi < len(records) else records  # no pick: the whole episode
    unitemized = sum(r["internal_unitemized"] for r in since if r.get("kind") == "skill"
                     and type(r.get("internal_unitemized")) is int and r["internal_unitemized"] > 0)
    drawers = _sponge_drawers(assumptions)
    opened = {r.get("drawer_region") for r in records[a0:a1] if r.get("kind") == "skill" and r.get("status") == "called"
              and r.get("name") == "open_drawer" and r.get("ok") is True}  # already_open returns ok True
    return {"store_contract": STORE_CONTRACT, "task_evidence_version": ev.get("version"),
            "task_evidence_verdict": ev.get("verdict"), "pick_in_final_run": ev.get("pick_in_span") is True,
            "target_scrub_in_final_run": ev.get("scrub_in_span") is True, "end_holding": ev.get("holding_now"),
            "episode_sponge_picks_ok": picks, "unitemized_robot_calls_since_pick": unitemized,
            "assumption_sources": {"rule": ASSUMPTION_SOURCES_RULE, "opening_end": opening_end,
                                   "final_span": [a0, a1], "views": list(view_ids)},
            "sponge_drawer_opened_in_final_run": bool(drawers & opened) if drawers else None}


def check(entry: dict, observations: list[dict], records: list[dict] | None = None) -> dict:
    """Compatibility of an entry with the evidence of the current episode (latest evidence wins).
    result: "contradicted" (some assumption contradicted), "consistent" (at least one object
    confirmed, none contradicted), "unconfirmed" (nothing contradicted, nothing confirmed)."""
    events = timeline(observations, records)
    handles = [h for o in observations for h in o.get("handles", [])]
    rows = []
    for a in entry["assumptions"]:
        if a["type"] == "object_at":
            s = object_state(a["category"], a["region"], a["position_m"], a["tol_m"], events)
            rows.append({**a, "status": s["status"], "fresh": s["fresh"], "why": s["why"],
                         "held_by_robot": s["held_by_robot"], "moved_by_robot": s["moved_by_robot"]})
        elif a["type"] == "handle_at":
            ok = any(math.dist(h["position_m"], a["position_m"]) <= a["tol_m"] for h in handles)
            rows.append({**a, "status": "confirmed" if ok else "unverified",
                         "why": "handle perceived at the assumed place" if ok else "not observed in this episode"})
    if any(r["status"] == "contradicted" for r in rows):
        verdict = "contradicted"
    elif any(r["type"] == "object_at" and r["status"] == "confirmed" for r in rows):
        verdict = "consistent"
    else:
        verdict = "unconfirmed"
    return {"result": verdict, "assumptions": rows}


def precondition(entry: dict, name: str, args: dict, observations: list[dict], records: list[dict]) -> dict:
    """Critical precondition of one gesture of a REUSED program, checked at the moment of use.
    Returns {"ok": bool, "status": "ok"|"unverified"|"contradicted"|"not_in_memory", "why", ...}."""
    events = timeline(observations, records)
    if name == "pick":
        cands = [a for a in entry["assumptions"] if a["type"] == "object_at" and a.get("used_by") == "pick"
                 and a["category"] == args.get("kind")]
        target = args.get("pos")
    elif name == "scrub":
        cands = [a for a in entry["assumptions"] if a["type"] == "object_at" and a.get("used_by") == "scrub"]
        tp = args.get("target_pos")
        target = None if tp is None else [tp[0], tp[1], tp[2] - BOWL_TOP_ABOVE_CENTRE]
    elif name == "open_drawer":
        # Robot-side safety only: pulling a drawer whose handle the robot perceived in this episode.
        # Opening another drawer than the stored program did is exploration, not a dependency on memory.
        seen = any(math.dist(h["position_m"], args.get("handle_pos")) <= 0.03
                   for o in observations for h in o.get("handles", []))
        return {"ok": seen, "status": "ok" if seen else "unverified",
                "why": "handle perceived in this episode" if seen else "no handle perceived there in this episode"}
    else:
        return {"ok": True, "status": "ok", "why": "no critical precondition"}
    if not cands:
        return {"ok": False, "status": "not_in_memory",
                "why": f"the stored program relied on no {args.get('kind', 'bowl')} for {name} (the reuse deviates)"}
    rows = []
    for a in cands:
        s = object_state(a["category"], a["region"], a["position_m"], a["tol_m"], events)
        if name == "pick" and s["held_by_robot"]:
            rows.append(("unverified", f"{a['category']} is held by the robot already"))
            continue
        if name == "pick" and s["moved_by_robot"] and target is not None and any(
                math.dist(target, p) <= a["tol_m"] for p in s["own_move_sightings"]):
            # The robot itself moved it (place, release) and a FRESH view shows it there; the pick is aimed at that
            # view. The site did not change; the stored place is not relied on.
            return {"ok": True, "status": "ok", "assumption": a,
                    "why": f"{a['category']} freshly perceived where the robot's own move left it"}
        if s["status"] == "confirmed" and s["fresh"]:
            if target is not None and s["sighting"] is not None and math.dist(target, s["sighting"]) <= a["tol_m"]:
                return {"ok": True, "status": "ok", "why": f"{a['category']} freshly perceived at the assumed place",
                        "assumption": a}
            rows.append(("unverified", f"{a['category']}: the gesture is not aimed at the fresh sighting"))
        elif s["status"] == "contradicted":
            rows.append(("contradicted", f"{a['category']} {s['why']}"))
        elif s["status"] == "confirmed":
            rows.append(("unverified", f"{a['category']} {s['why']}: observe again before this gesture"))
        else:
            rows.append(("unverified", f"{a['category']} {s['why']}"))
    status = "contradicted" if all(r[0] == "contradicted" for r in rows) else "unverified"
    return {"ok": False, "status": status, "why": "; ".join(r[1] for r in rows)[:300]}


# ====================================================================== library
class NoAssumption(ValueError):
    """store() refused: the program relied on no perceived object, so a later reuse could not be checked."""


class ConditionError(ValueError):
    """A call that the library's declared validation condition does not allow."""


EPISODE_EVIDENCE_KEYS = ("episode_contact_incidents", "episode_grip_events", "done_refusals")


class IncidentEpisode(ConditionError):
    """store() refused: the episode had a robot-felt contact incident or a refused done()."""


class LegacyLibrary(ConditionError):
    """A library file of another schema than LIBRARY_SCHEMA is never written (refused when the library is built to
    write; a read_only library refuses every write)."""


class StoreContractMismatch(ConditionError):
    """store() refused: the evidence was built under another storage contract or task-evidence version."""


class TaskEvidenceMissing(ConditionError):
    """store() refused: the final program's own run does not carry the robot-side task evidence (store-contract-1
    S1 and the hold: proven, pick and target scrub in the final run, grip True at done(), one sponge pick, no unitemized
    robot call since it)."""


class AssumptionContract(NoAssumption):
    """store() refused: the assumptions do not follow store-contract-1 (S2: extracted from the opening and
    final-run views only, with a sponge/pick and a bowl/scrub assumption; S3: a sponge assumed in a drawer opened in the
    final run)."""


class ProgramLibrary:
    def __init__(self, site_id: str, directory: str | Path, validation: str = "robot_side", read_only: bool = False):
        """read_only: history tools only; a file of another schema is then readable and every write is refused
        (LegacyLibrary). Without it, an existing file of another schema is refused here, before any episode runs."""
        if not site_id or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in site_id.lower()):
            raise ValueError("site_id: letters, digits, _ and - only")
        if validation not in VALIDATION_MODES:
            raise ValueError(f"validation must be one of {VALIDATION_MODES}")
        self.site_id = site_id
        self.read_only = bool(read_only)
        self.path = Path(directory) / f"library_{site_id}.json"
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
            if self.data.get("schema") != LIBRARY_SCHEMA and not self.read_only:
                # Never write into a historical file (it would mix contracts and rewrite old bytes)
                raise LegacyLibrary(f"library schema {str(self.data.get('schema'))[:40]!r} is not {LIBRARY_SCHEMA!r}: "
                                    "not written (read_only=True reads it)")
            for i, e in enumerate(self.data.get("entries", [])):  # files written before 0.4: ids added
                e.setdefault("entry_id", f"{e['key'][:16]}-{i:03d}")
            # a file written before 0.3 had no declared condition: it was the external one (store after verdict)
            declared = self.data.get("validation", "external_validation")
            if declared != validation:
                raise ConditionError(f"library declared {declared!r}; cannot be used as {validation!r}")
        else:
            self.data = {"schema": LIBRARY_SCHEMA, "site_id": site_id, "label": "simulation, perception simulee",
                         "validation": validation, "validation_fr": VALIDATION_FR[validation], "entries": []}
        if self.data.get("site_id") != site_id:
            raise ValueError("library file belongs to another site")
        self.validation = validation

    def _writable(self):
        """Refuse any write to a read_only library or to a file of another schema (nothing is changed)."""
        if self.read_only or self.data.get("schema") != LIBRARY_SCHEMA:
            raise LegacyLibrary("read-only or historical library: nothing is written")

    def _save(self):
        self._writable()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1))
        os.replace(tmp, self.path)

    def _entry(self, ref):
        """The entry with entry_id `ref`; for a key (callers before 0.4), the VALID entry with that key,
        else the latest one (a key stored again after an invalidation has two entries, and the first one used to
        receive the new entry's uses and verdicts). store-contract-2: among the entries of STORE_CONTRACT only; an
        entry of another contract (an entry of store-contract-1, which may share its key with a current one) is never
        returned, so no use, event or verdict is ever written into it (None: the callers then change nothing)."""
        mine = [e for e in self.data["entries"] if e.get("store_contract") == STORE_CONTRACT]
        by_id = next((e for e in mine if e.get("entry_id") == ref), None)
        if by_id is not None:
            return by_id
        same = [e for e in mine if e["key"] == ref]
        return next((e for e in reversed(same) if e["status"] == "valid"), same[-1] if same else None)

    @staticmethod
    def key(dossier: dict, skill_api_version: str, criteria_version: str, assumptions: list[dict]) -> str:
        return _h({"dossier_version": dossier.get("version"), "dossier_sha256": dossier_hash(dossier),
                   "skill_api_version": skill_api_version, "criteria_version": criteria_version,
                   "assumptions": sorted(_h(a) for a in assumptions)})

    def store(self, program: str, dossier: dict, skill_api_version: str, criteria_version: str,
              assumptions: list[dict], validated_by: str, provenance: dict, evidence: dict | None = None) -> dict:
        """Keep a program. robot_side: evidence = robot_side_evidence() of the final run (pick and
        scrub ok) is required; validated_by names it. external_validation: the bench calls it after the verifier
        validated the episode. provenance: model id, prompt version, episode label (no truth). (store-contract-1,
        both conditions, in this order; see the module docstring): NoAssumption, LegacyLibrary, ConditionError
        (keys), IncidentEpisode, StoreContractMismatch, TaskEvidenceMissing, AssumptionContract, the validated_by
        rules; deduplication only among valid entries of the same contract."""
        if not any(a.get("type") == "object_at" for a in assumptions):
            raise NoAssumption("no object_at assumption: the program is not stored")
        self._writable()  # LegacyLibrary (never write into a historical file)
        need = EPISODE_EVIDENCE_KEYS + STORE_CONTRACT_KEYS
        if not isinstance(evidence, dict) or any(k not in evidence for k in need):
            # Both conditions: the episode's robot-side counts must be given; and the store-contract-1 fields
            # (agent.memory.store_contract_fields)
            raise ConditionError("store needs the episode's robot-side evidence: " + ", ".join(
                k for k in need if not isinstance(evidence, dict) or k not in evidence)[:300])
        if any(evidence.get(k) for k in EPISODE_EVIDENCE_KEYS):
            # Both conditions: never a program from an episode with a contact incident felt by the robot or a refused
            # done() (such a program was stored and reused); nor with a grip event (a drop read by the robot's grip, or
            # a release of the sponge)
            raise IncidentEpisode("the episode had a robot-felt contact incident, a grip event or a refused done(): "
                                  "not stored")
        if evidence.get("store_contract") != STORE_CONTRACT \
                or evidence.get("task_evidence_version") != TASK_EVIDENCE_VERSION_REQUIRED:
            raise StoreContractMismatch(f"evidence of contract {str(evidence.get('store_contract'))[:40]!r} and task "
                                        f"evidence {str(evidence.get('task_evidence_version'))[:40]!r}: this library "
                                        f"stores {STORE_CONTRACT} with {TASK_EVIDENCE_VERSION_REQUIRED} only")
        miss = [k for k, ok in (("task_evidence_proven", evidence.get("task_evidence_verdict") == "proven"),
                                ("pick_in_final_run", evidence.get("pick_in_final_run") is True),
                                ("target_scrub_in_final_run", evidence.get("target_scrub_in_final_run") is True),
                                ("end_holding", evidence.get("end_holding") is True),
                                ("one_sponge_pick", type(evidence.get("episode_sponge_picks_ok")) is int
                                 and evidence["episode_sponge_picks_ok"] == 1),
                                ("no_unitemized_robot_call", type(evidence.get("unitemized_robot_calls_since_pick"))
                                 is int and evidence["unitemized_robot_calls_since_pick"] == 0)) if not ok]
        if miss:
            # The task and the hold: the final program's OWN run proves the task (a scrub-only correction relying on an
            # earlier attempt's pick was once stored; in another episode the hold was lost after the scrub)
            raise TaskEvidenceMissing("the final program's run does not carry the task evidence: " + ", ".join(miss))
        src = evidence.get("assumption_sources")
        bad = []
        if not isinstance(src, dict) or src.get("rule") != ASSUMPTION_SOURCES_RULE:
            bad.append("assumptions_not_from_opening_and_final_run_views")
        if not any(a.get("type") == "object_at" and a.get("used_by") == "pick" and a.get("category") == "sponge"
                   for a in assumptions):
            bad.append("no_sponge_pick_assumption")
        if not any(a.get("type") == "object_at" and a.get("used_by") == "scrub" and a.get("category") == "bowl"
                   for a in assumptions):
            bad.append("no_bowl_scrub_assumption")
        if _sponge_drawers(assumptions) and evidence.get("sponge_drawer_opened_in_final_run") is not True:
            bad.append("sponge_drawer_not_opened_in_final_run")  # S3
        if bad:
            # An entry without both assumptions can only give not_in_memory at reuse ("unverified", valid forever);
            # one relying on an earlier attempt's view or drawer failed at its first reuse
            raise AssumptionContract("the assumptions do not follow " + STORE_CONTRACT + ": " + ", ".join(bad))
        if self.validation == "robot_side":
            if not (evidence and evidence.get("pick_ok") and evidence.get("scrub_ok")):
                raise ConditionError("robot_side condition: store needs a pick and a scrub that returned ok")
            if not validated_by.startswith("robot_side"):
                raise ConditionError("robot_side condition: validated_by must name robot-side evidence")
        elif not validated_by.startswith("external"):
            raise ConditionError("external_validation condition: validated_by must name the verifier's bit")
        k = self.key(dossier, skill_api_version, criteria_version, assumptions)
        for e in self.data["entries"]:
            # Only an entry of the same contract (an entry of an earlier contract is never a candidate: validations
            # would be lost)
            if e["key"] == k and e["status"] == "valid" and e.get("store_contract") == STORE_CONTRACT:
                e["validations"].append(validated_by)
                self._save()
                return e
        e = {"entry_id": f"{k[:16]}-{len(self.data['entries']):03d}", "key": k, "status": "valid",
             "store_contract": STORE_CONTRACT, "validation": self.validation, "program": program,
             "program_sha256": hashlib.sha256(program.encode()).hexdigest(),
             "dossier_version": dossier.get("version"), "dossier_sha256": dossier_hash(dossier),
             "skill_api_version": skill_api_version, "criteria_version": criteria_version,
             "assumptions": assumptions, "validations": [validated_by], "robot_side_evidence": evidence,
             "provenance": provenance, "created": _now(), "uses": [], "events": [], "invalidated": None}
        self.data["entries"].append(e)
        self._save()
        return e

    def candidates(self, dossier: dict, skill_api_version: str, criteria_version: str,
                   store_contract: str | None = STORE_CONTRACT) -> list[dict]:
        """Valid entries of this dossier, skill API and criteria. With skill API 0.7 (S5 profile, no
        place, turn_over, press or spray) the entries stored under 0.6 or earlier are never candidates; kept as
        history. And of this storage contract (store_contract None: every contract, for history tools only)."""
        dh = dossier_hash(dossier)
        return [e for e in self.data["entries"] if e["status"] == "valid" and e["dossier_sha256"] == dh
                and e["skill_api_version"] == skill_api_version and e["criteria_version"] == criteria_version
                and (store_contract is None or e.get("store_contract") == store_contract)]

    def lookup(self, dossier: dict, skill_api_version: str, criteria_version: str,
               observations: list[dict], records: list[dict] | None = None,
               store_contract: str | None = STORE_CONTRACT) -> tuple[dict | None, list[dict]]:
        """(entry to attempt or None, checks made). A contradicted entry is invalidated here (robot
        side). An entry not contradicted may be attempted: its preconditions are checked at use. Among the entries of
        store_contract only (candidates); an entry of another contract is never checked, invalidated or rewritten."""
        checks = []
        for e in reversed(self.candidates(dossier, skill_api_version, criteria_version, store_contract)):
            c = check(e, observations, records)
            checks.append({"key": e["key"], **c})
            if c["result"] == "contradicted":
                if not self.read_only:  # a read_only (history) library is read, never changed
                    self.invalidate(e["entry_id"], "observation contradicts: " + "; ".join(
                        f"{a.get('category', 'handle')} {a['why']}" for a in c["assumptions"]
                        if a["status"] == "contradicted"), by="robot_side")
                continue
            return e, checks
        return None, checks

    def note_event(self, ref: str, event: dict):
        self._writable()  # before any change in memory
        e = self._entry(ref)
        if e is not None:
            e.setdefault("events", []).append({"t": _now(), **event})
            self._save()

    def note_verdict(self, ref: str, validated: bool, by: str):
        """external_validation only: the bench reports the verifier's bit for an episode that REUSED
        this entry. A rejection invalidates the entry (declared one-bit channel)."""
        if self.validation != "external_validation":
            raise ConditionError("note_verdict is the external validation channel; this library is robot_side")
        self._writable()
        e = self._entry(ref)
        if e is not None:
            e.setdefault("verdict_bits", []).append({"validated": bool(validated), "by": by[:120], "t": _now()})
        self._save()
        if not validated and e is not None:
            self.invalidate(e["entry_id"], f"reuse rejected by the verifier ({by[:120]})", by="external_validation")

    def note_use(self, ref: str, outcome: str):
        self._writable()
        e = self._entry(ref)
        if e is not None:
            e["uses"].append({"t": _now(), "outcome": outcome})
        self._save()

    def invalidate(self, ref: str, reason: str, by: str = "robot_side"):
        """Invalidate the entry with entry_id `ref` (or, for a key, the valid entries with that key).
        Nothing is erased: program, uses, events, reason and author stay in the file. store-contract-2: entries of
        STORE_CONTRACT only (an earlier entry is never rewritten)."""
        self._writable()
        for e in self.data["entries"]:
            if (e.get("entry_id") == ref or e["key"] == ref) and e["status"] == "valid" \
                    and e.get("store_contract") == STORE_CONTRACT:
                e["status"], e["invalidated"] = "invalidated", {"reason": reason[:300], "by": by, "t": _now()}
        self._save()
