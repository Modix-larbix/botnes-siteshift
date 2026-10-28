"""Site dossier, site model and difference report: what the robot knows about the site, and how sure it is.

SiteDossier: what the site's documentation declares (modules and slots, drawers, where tools are stored). It may be out
of date and names things by category and region ("sponge in drawer_unit:0"), never by simulator name.

SiteModel: the robot's knowledge during a session. It merges the dossier with observations (perception.contract) and
the robot's own actions, with the statuses declared, observed, unknown, stale and held.

diff_report() and search_scope() are computed only from the dossier and the observations the robot received. A thing
that was not seen is never called absent:
- a declared location never inspected stays "declared_location_not_inspected";
- an inspected declared location without the item gives "not_seen_at_declared_location", with coverage and step;
- search_scope() says "not_found_in_inspected_scope" only when every region the dossier's conventions allow was
  inspected and its unseen part cannot hold the item: no placement of the item's footprint on the region floor, clear
  of resting objects, is left with too little of it in view (HIDE_SHARE, the sensor's own threshold). For the sponge
  the placement evidence of perception.estimate must also leave no open placement (could_hide reads both); a surface
  without an accepted outline is never cleared. Otherwise the outcome is "not_found_scope_incomplete", with the
  regions still to check.
A clearance counts only if no robot action that could have put an object into the region came after the view that
cleared it (INVALIDATING_ACTIONS, and any action noted with a contact reason, CONTACT_INVALIDATING_REASONS); views on
either side of such an action are not merged. Drawers of a unit are numbered from 0, left to right as seen from the
robot.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np

from .perception import catalog as C
from .perception.contract import DECLARED, HELD, OBSERVED, SIM_LABEL, STALE, UNKNOWN, to_dict

MIN_COVER = 0.3  # a region counts as inspected when >= 30 % of its floor grid was reached
GRID_INSET = 0.008  # region floor grids stop 8 mm inside the floor edges (perception.estimate)
HIDE_MARGIN = 0.0  # grid points within the disk radius count for a placement
HIDE_SHARE = 0.15  # a placement hides the item if less than this share of its points were in view
INVALIDATING_ACTIONS = ("pick", "place", "turn_over", "scrub", "spray", "press")
# contact-invalidates-1: the robot's own contact reasons (robot.skills reflex and motion stops), the same
# list as agent.host.INCIDENT_REASONS / INCIDENT_PREFIXES (this module imports nothing of the agent; a test checks
# CONTACT_INVALIDATING_REASONS == agent.host.INCIDENT_CODES). lift_blocked counts when its lift stopped on one of
# CONTACT_REASONS (agent.host.is_contact_incident).
CONTACT_INVALIDATES_VERSION = "contact-invalidates-1"
CONTACT_REASONS = ("collision", "blocked", "stuck", "force_limit", "collision_on_approach")
CONTACT_PREFIXES = ("", "pull_", "push_", "turn_over_", "turn_over_pick_")
CONTACT_INVALIDATING_REASONS = frozenset(p + r for p in CONTACT_PREFIXES for r in CONTACT_REASONS)


def is_contact_reason(reason, lift=None) -> bool:
    """contact-invalidates-1: a skill reason that reports a contact (CONTACT_INVALIDATING_REASONS), or lift_blocked whose
    lift (the skill's data.lift) stopped on a contact. Anything else, None and non-strings included, is not."""
    if not isinstance(reason, str):
        return False
    if reason in CONTACT_INVALIDATING_REASONS:
        return True
    return reason == "lift_blocked" and isinstance(lift, str) and lift in CONTACT_REASONS


def _merge_codes(a: str, b: str) -> str:
    """Union of two views of the same floor grid: seen beats under-an-object beats not seen."""
    rank = {"1": 2, "o": 1, "0": 0}
    return "".join(x if rank[x] >= rank[y] else y for x, y in zip(a, b))


def _merge_grids(rec: dict, prev: dict) -> dict | None:
    if not (rec.get("grid_seen") and prev.get("grid_seen") and rec["grid_shape"] == prev["grid_shape"]
            and len(rec["grid_seen"]) == len(prev["grid_seen"])):
        return None
    out = {"grid_seen": _merge_codes(rec["grid_seen"], prev["grid_seen"])}
    if rec.get("grid_seen_top") and prev.get("grid_seen_top"):
        out["grid_seen_top"] = _merge_codes(rec["grid_seen_top"], prev["grid_seen_top"])
    return out


def _lattice(ev: dict):
    """The 6 numbers (ox, oy, eax, eay, ebx, eby) of an evidence record's lattice frame. surface-frame-1: the
    record may carry its frame as a mapping (the outline block: frame {accepted, ...}); the
    lattice is then its "lattice" entry, else its first entry made of 6 numbers. Unchanged for a 6-number frame."""
    f = ev["frame"]
    if isinstance(f, dict):
        if "lattice" in f:
            return f["lattice"]
        for v in f.values():
            if isinstance(v, (list, tuple)) and len(v) == 6 and all(
                    isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
                return v
        raise ValueError("evidence frame without a lattice")
    return f


def frame_accepted(ev) -> bool | None:
    """surface-frame-1: did the view's lattice rest on an accepted module outline? The flag of the
    evidence's outline block: "module_frame" {accepted, ...} (perception.estimate writes it on every surface view;
    the key "frame" keeps the lattice), or a frame mapping {accepted, lattice, ...}, else a boolean "frame_fitted".
    Only True counts as accepted (a malformed flag is not). None: the record carries no outline block (drawers;
    records written previously): no gate here (perception.estimate.outline_could_hide also gates a surface view
    without its block)."""
    if not isinstance(ev, dict):
        return None
    for b in (ev.get("module_frame"), ev.get("frame")):
        if isinstance(b, dict) and "accepted" in b:
            return b["accepted"] is True
    if "frame_fitted" in ev:
        return ev["frame_fitted"] is True
    return None


def _placement_centres(ev: dict, mode: str):
    """(world xy (N, 2), codes (N,)) of a placement-evidence record's lattice for one mode."""
    m = ev["modes"][mode]
    ox, oy, eax, eay, ebx, eby = _lattice(ev)
    na, nb = m["n"]
    if na * nb == 0:
        return np.zeros((0, 2)), np.array([], dtype="<U1")
    A, B = np.meshgrid((np.arange(na) + m["k0"][0]) * m["step"][0], (np.arange(nb) + m["k0"][1]) * m["step"][1],
                       indexing="ij")
    a, b = A.ravel(), B.ravel()
    return np.column_stack([ox + a * eax + b * ebx, oy + a * eay + b * eby]), np.array(list(m["code"]))


def _cell_shown(ev: dict, mode: str, xy: np.ndarray) -> np.ndarray:
    """Per point of xy (world, (N, 2)): did the view of ev show the target at every placement of its own lattice around
    that point (the corners of the lattice cell holding it, or the lattice point itself)? Views are compared by world
    position, never by grid index: a later view may place its grid elsewhere (the drawer or module track moved)."""
    m = ev["modes"].get(mode)
    out = np.zeros(len(xy), bool)
    if m is None or len(xy) == 0 or m["n"][0] * m["n"][1] == 0:
        return out
    ox, oy, eax, eay, ebx, eby = _lattice(ev)
    d = xy - np.array([ox, oy])
    fa = (d[:, 0] * eax + d[:, 1] * eay) / m["step"][0] - m["k0"][0]
    fb = (d[:, 0] * ebx + d[:, 1] * eby) / m["step"][1] - m["k0"][1]
    na, nb = m["n"]
    code = np.array(list(m["code"]))
    out[:] = True
    snap = 1e-3  # lattice units: a point within 5 microns of a lattice point is that point (frames are rounded)
    for ia in (np.floor(fa + snap), np.ceil(fa - snap)):
        for ib in (np.floor(fb + snap), np.ceil(fb - snap)):
            inside = (ia >= 0) & (ia < na) & (ib >= 0) & (ib < nb)
            k = (np.clip(ia, 0, na - 1) * nb + np.clip(ib, 0, nb - 1)).astype(int)
            out &= inside & (code[k] == "v")
    return out


def merge_placement(cur: dict, prev: dict | None) -> dict:
    """scope-placement-1: the placement evidence of a view (cur) merged with the record of the views
    before it (prev), when the site model merges the two (no invalidating action in between). The record keeps cur's
    lattice and codes, the earlier views' lattices and codes ("earlier", compared by world position: _cell_shown), and
    the perceived objects of the region by track, each with its LATEST reading ("objects": a later view that reads an
    object as raised removes the exclusion an earlier reading as resting gave; an object not seen again keeps its last
    reading, nothing having moved it since)."""
    if not cur or "modes" not in cur or not prev or "modes" not in prev:
        return dict(cur or {})
    out = dict(cur)
    own = {"frame": prev["frame"], "modes": {k: {x: v[x] for x in ("k0", "n", "step", "code")}
                                             for k, v in prev["modes"].items()}}
    out["earlier"] = list(prev.get("earlier", [])) + [own]
    objs = {o["track"]: o for o in prev.get("objects", [])}
    for o in cur.get("objects", []):  # the latest reading decides whether it rests; its footprints add up
        old = objs.get(o["track"])
        objs[o["track"]] = dict(o, fps=(list(old["fps"]) if old else []) + list(o["fps"]))
    out["objects"] = list(objs.values())
    return out


def _evidence_of(rec):
    """The placement evidence of a region record (its "placement" entry), or rec itself when it is the evidence."""
    if not isinstance(rec, dict):
        return None
    return rec if "modes" in rec else rec.get("placement")


def open_placements(rec, modes=None) -> np.ndarray:
    """Pure: the world xy (N, 2) of the placements of the record's LATEST
    lattice that are not 'v' (this view showed the target there), not ' ' (no fitting yaw), not excluded by a perceived
    object resting there (objects, latest reading per track) and not shown by an earlier merged view (_cell_shown, by
    world position). rec: a region record (its "placement" evidence) or the evidence itself. modes: the mode names to
    read; None = the evidence's active modes. This is exactly the set placement_could_hide tests. Empty (0, 2) when the
    record has no readable evidence: the caller must then read could_hide, which is True on an evidence error."""
    from .perception.estimate import placement_excluded  # noqa: PLC0415 (robot side; perception imports nothing here)
    ev = _evidence_of(rec)
    out = []
    if not ev or "modes" not in ev:
        return np.zeros((0, 2))
    for mode, m in ev["modes"].items():
        if (not m.get("active")) if modes is None else (mode not in modes):
            continue
        xy, code = _placement_centres(ev, mode)
        rest = (code != " ") & (code != "v")
        if not rest.any():
            continue
        idx = np.flatnonzero(rest)
        idx = idx[~placement_excluded(xy[idx], ev.get("objects", []), mode, m["r_in"])]
        for e in ev.get("earlier", []):
            if len(idx) == 0:
                break
            idx = idx[~_cell_shown(e, mode, xy[idx])]
        if len(idx):
            out.append(xy[idx])
    return np.concatenate(out, axis=0) if out else np.zeros((0, 2))


def placement_could_hide(ev: dict) -> bool:
    """scope-placement-1: could the guarded target lie at some placement of the record's lattice (any
    active mode) that no merged view showed ('v' here, or _cell_shown in an earlier view) and where no perceived
    object of the region rests (objects, latest reading per track)? True on an evidence error (never absence). The
    placements are open_placements(ev) (the same set the look aims use); and True when the lattice rests on no
    accepted module outline (surface-frame-1: a surface whose outline was not fitted is never cleared): perception's
    own reading perception.estimate.outline_could_hide (a surface view whose "module_frame" block is missing or not
    accepted), or the evidence's outline block read here (frame_accepted False)."""
    if not ev or "modes" not in ev:
        return True
    from .perception import estimate as _est  # noqa: PLC0415 (robot side; perception imports nothing here)
    outline = getattr(_est, "outline_could_hide", None)
    if (callable(outline) and outline(ev)) or frame_accepted(ev) is False:
        return True
    return len(open_placements(ev)) > 0


def could_hide(rec: dict | None, category: str) -> bool | None:
    """Could an object of this category lie unseen on the region floor, given the inspection
    record (grid of seen / under-an-object / unseen points)? None if no grid is known. Rotation-free test: a disk of
    the category's smallest footprint dimension (it lies inside the footprint whatever the yaw) placed anywhere on
    the floor (inside its edges) covers only unseen points. scope-placement-1: when the record carries placement
    evidence for this category (robot side: perception.estimate._placement_evidence, merged by SiteModel), the answer
    is True also when that evidence leaves a placement open (placement_could_hide): the evidence can only add "could
    hide", never remove it. Records without it (the bench's copies, the per-view records of observe()) read the grid
    alone, as before."""
    grid = _could_hide_grid(rec, category)
    ev = rec.get("placement") if rec else None
    if grid is None or not ev or ev.get("category") != category:
        return grid
    return bool(grid or placement_could_hide(ev))


def _could_hide_grid(rec: dict | None, category: str) -> bool | None:
    """The grid reading of could_hide (before placement evidence existed, the whole of it)."""
    if not rec or not rec.get("grid_seen") or category not in C.OBJECT_SHAPES:
        return None
    na, nb = rec["grid_shape"]
    pa, pb = rec["grid_pitch_m"]
    code = np.array(list(rec["grid_seen"]))
    occupied = code == "o"
    shape = C.OBJECT_SHAPES[category]
    if shape["height"] >= C.VISIBILITY_LAYER_H + 0.005 and rec.get("grid_seen_top"):
        in_view = np.array(list(rec["grid_seen_top"])) == "1"
    else:
        in_view = code == "1"
    r = 0.5 * min(shape["footprint"])
    ia, ib = np.meshgrid(np.arange(na) * pa, np.arange(nb) * pb, indexing="ij")
    P = np.column_stack([ia.ravel(), ib.ravel()])
    # Disk centres on a half-pitch lattice, at least r inside the floor edges (the floor extends
    # GRID_INSET beyond the outer grid points).
    lo_a, hi_a = r - GRID_INSET, (na - 1) * pa - (r - GRID_INSET)
    lo_b, hi_b = r - GRID_INSET, (nb - 1) * pb - (r - GRID_INSET)
    if hi_a < lo_a or hi_b < lo_b:
        return bool(in_view.mean() < HIDE_SHARE)
    ca = np.arange(lo_a, hi_a + 1e-9, pa / 2)
    cb = np.arange(lo_b, hi_b + 1e-9, pb / 2)
    CA, CB = np.meshgrid(ca, cb, indexing="ij")
    Cc = np.column_stack([CA.ravel(), CB.ravel()])
    inside = np.hypot(Cc[:, None, 0] - P[None, :, 0], Cc[:, None, 1] - P[None, :, 1]) <= r - HIDE_MARGIN
    n_in = inside.sum(axis=1)
    possible = (n_in > 0) & ~(inside & occupied[None, :]).any(axis=1)  # not under a perceived object
    share = (inside & in_view[None, :]).sum(axis=1) / np.maximum(n_in, 1)
    return bool((possible & (share < HIDE_SHARE)).any())


@dataclass
class SiteDossier:
    version: str
    worktop_height_m: float
    modules: list[dict] = field(default_factory=list)  # {"kind", "slot", "n_drawers"?}
    stored_items: list[dict] = field(default_factory=list)  # {"category", "region"}
    conventions: dict = field(default_factory=lambda: {
        "drawer_numbering": "drawer_unit:k, k from 0, left to right as seen from the robot",
        "tool_storage": "tools are stored in drawers or in the sink basin"})

    @staticmethod
    def from_spec(spec, version: str = "dossier-v1") -> "SiteDossier":
        """The dossier as written for this site layout (bench side: done once, before any change)."""
        mods = []
        for m in spec.modules:
            d = {"kind": m.kind, "slot": m.slot}
            if m.kind == "drawers":
                d["n_drawers"] = int(m.params.get("n", 2))
            mods.append(d)
        items = []
        for it in spec.items:
            if it.kind not in C.TOOL_CATEGORIES:
                continue
            if it.where.startswith("drawers/"):
                region = C.drawer_region(int(it.where.split("/")[1]))
            elif it.where == "sink_side":
                region = "sink_basin"
            else:
                region = "other"
            items.append({"category": it.kind, "region": region})
        items.sort(key=lambda d: (d["category"], d["region"]))
        return SiteDossier(version, float(spec.worktop_height), mods, items)

    def n_drawers(self) -> int | None:
        m = next((m for m in self.modules if m["kind"] == "drawers"), None)
        return None if m is None else int(m["n_drawers"])

    def regions(self) -> list[str]:
        out = []
        for m in self.modules:
            cat = C.MODULE_KIND_TO_CATEGORY[m["kind"]]
            if m["kind"] == "drawers":
                out += [C.drawer_region(k) for k in range(int(m["n_drawers"]))]
            elif cat in C.SURFACE_REGION:
                out.append(C.SURFACE_REGION[cat])
            elif m["kind"] in C.UNINSPECTED_REGIONS:
                out.append(C.UNINSPECTED_REGIONS[m["kind"]])
        return out

    def candidate_regions(self, category: str) -> list[str]:
        """Where the dossier's conventions allow a stored tool to be: every drawer and the open
        surfaces (sink basin, landing tray, drying mat). The declared region first."""
        declared = [d["region"] for d in self.stored_items if d["category"] == category]
        rest = [r for r in self.regions() if r.startswith("drawer_unit:")]
        rest += [r for r in self.regions() if r in ("sink_basin", "landing_tray", "drying_mat")]
        out = []
        for r in declared + rest:
            if r not in out:
                out.append(r)
        return out

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), **kw)


class SiteModel:
    """The robot's site knowledge: dossier (declared) + observations + own actions."""

    def __init__(self, dossier: SiteDossier):
        self.dossier = dossier
        self.regions: dict[str, dict] = {}
        for r in dossier.regions():
            self.regions[r] = self._region(r, declared=True)
        self.tracks: dict[str, dict] = {}
        self.modules: dict[str, dict] = {}
        self.actions: list[dict] = []
        self.obs_ids: list[str] = []

    @staticmethod
    def _region(rid, declared):
        return {"region": rid, "declared": declared, "status": UNKNOWN, "last": None,
                "inspections": [], "stale_since": None, "stale_reason": None}

    # ------------------------------------------------------------ inputs
    def integrate(self, obs, perception=None):
        self.obs_ids.append(obs.obs_id)
        for r in obs.regions:
            reg = self.regions.setdefault(r.region, self._region(r.region, declared=False))
            rec = {"obs_id": obs.obs_id, "step": r.step, "sensor": r.sensor, "coverage": r.coverage,
                   "seen_categories": list(r.seen_categories), "seen_tracks": list(r.seen_tracks),
                   "grid_shape": list(r.grid_shape), "grid_pitch_m": list(r.grid_pitch_m),
                   "grid_seen": r.grid_seen, "grid_seen_top": r.grid_seen_top}
            ev = getattr(perception, "placement_evidence", {}).get((obs.obs_id, r.region)) if perception else None
            if ev is not None:  # scope-placement-1
                rec["placement"] = ev
            reg["inspections"].append(rec)
            if r.coverage >= MIN_COVER:
                prev = reg["last"]
                if reg["status"] == OBSERVED and prev is not None and not self._invalidated_after(prev["step"], r.step):
                    # No robot action on this region since the previous inspection: the views add
                    # up (union of what was seen, point by point when both grids match), a later
                    # partial view does not erase an earlier sighting.
                    cov = max(rec["coverage"], prev["coverage"])
                    merged = _merge_grids(rec, prev) or {}
                    if merged:
                        cov = max(cov, merged["grid_seen"].count("1") / len(merged["grid_seen"]))
                    if rec.get("placement") is not None:  # per placement, by world position
                        merged["placement"] = merge_placement(rec["placement"], prev.get("placement"))
                    rec = dict(rec, coverage=cov, **merged,
                               seen_categories=sorted(set(prev["seen_categories"]) | set(rec["seen_categories"])),
                               seen_tracks=sorted(set(prev["seen_tracks"]) | set(rec["seen_tracks"])),
                               merged_steps=prev.get("merged_steps", [prev["step"]]) + [rec["step"]])
                reg["status"] = OBSERVED
                reg["last"] = rec
                reg["stale_since"] = reg["stale_reason"] = None
        for d in obs.detections:
            self.tracks[d.track_id] = {"track_id": d.track_id, "category": d.category, "status": OBSERVED,
                                       "region": d.region, "pose": to_dict(d.pose), "last_step": d.step,
                                       "visibility_estimate": d.visibility_estimate}
        for m in obs.modules:
            self.modules[m.track_id] = {"track_id": m.track_id, "category": m.category,
                                        "slot_estimate": m.slot_estimate, "bearing_deg": m.bearing_deg,
                                        "last_step": m.step}

    def note_action(self, step: int, action: str, region: str | None, detail: dict):
        self.actions.append({"step": step, "action": action, "region": region, **detail})
        if (action.startswith(("open_drawer", "close_drawer")) or action == "pick") and region:
            reg = self.regions.get(region)
            if reg is not None and reg["status"] == OBSERVED:
                reg["status"], reg["stale_since"], reg["stale_reason"] = STALE, step, action
            for t in self.tracks.values():
                if t["region"] == region and t["status"] == OBSERVED:
                    t["status"] = STALE
        if action == "pick" and detail.get("ok") and detail.get("track_id") in self.tracks:
            t = self.tracks[detail["track_id"]]
            t["picked_from"] = t["region"]
            t["status"], t["region"] = HELD, "gripper"

    def note_reason(self, step: int, reason, lift=None):
        """contact-invalidates-1: the skill reason of the robot action(s) noted at `step` (perception's drawer
        and pick notes carry none in their signature). Sets "reason" (and "lift" for a lift_blocked) on the actions
        of that step that have no reason yet; an action that already carries one keeps it. Records only; read by
        _invalidating_steps."""
        if not isinstance(reason, str):
            return
        for a in reversed(self.actions):
            if a.get("step") != step:
                break
            if a.get("reason") is None:
                a["reason"] = reason
            if a.get("reason") == reason and isinstance(lift, str) and a.get("lift") is None:
                a["lift"] = lift

    def _invalidating_steps(self) -> list[int]:
        """Steps of the actions that invalidate every earlier clearance: a pick, place, turn_over, scrub, spray
        or press (failed forms included), any action while holding, and (contact-invalidates-1) any action noted with
        a contact reason (is_contact_reason)."""
        return [a["step"] for a in self.actions
                if a["action"].split("_failed")[0] in INVALIDATING_ACTIONS or a.get("holding")
                or is_contact_reason(a.get("reason"), a.get("lift"))]

    def _invalidated_after(self, step0: int, step1: int | None = None) -> bool:
        """An action that could have put an object into a region happened after step0 (and before
        step1 if given)."""
        return any(s_ > step0 and (step1 is None or s_ < step1) for s_ in self._invalidating_steps())

    # ------------------------------------------------------------ queries
    def inspected(self, rid: str) -> bool:
        reg = self.regions.get(rid)
        return bool(reg and reg["last"] is not None)

    def could_hide(self, rid: str, category: str) -> bool | None:
        reg = self.regions.get(rid)
        return None if not reg or reg["last"] is None else could_hide(reg["last"], category)

    def seen_in(self, rid: str) -> list[str]:
        reg = self.regions.get(rid)
        return [] if not reg or reg["last"] is None else list(reg["last"]["seen_categories"])

    def tracks_of(self, category: str, statuses=(OBSERVED, STALE, HELD)) -> list[dict]:
        return [t for t in self.tracks.values() if t["category"] == category and t["status"] in statuses]

    def status_table(self) -> dict:
        """Knowledge statuses: declared items, regions, tracks."""
        items = []
        for d in self.dossier.stored_items:
            items.append({"category": d["category"], "declared_region": d["region"], "status": DECLARED,
                          "declared_region_status": self.regions.get(d["region"], {}).get("status", UNKNOWN)})
        regions = [{k: v for k, v in r.items() if k != "inspections"} | {"n_inspections": len(r["inspections"])}
                   for r in self.regions.values()]
        for r in regions:  # keep the status table short: the grid stays in the inspection records
            if r.get("last"):
                r["last"] = {k: v for k, v in r["last"].items() if k not in ("grid_seen", "grid_seen_top", "placement")}
        return {"declared_items": items, "regions": regions, "tracks": list(self.tracks.values()),
                "modules": list(self.modules.values())}

    def search_scope(self, category: str) -> dict:
        found = self.tracks_of(category)
        cands = self.dossier.candidate_regions(category)
        rows = []
        for r in cands:
            reg = self.regions.get(r)
            last = reg["last"] if reg else None
            hide = could_hide(last, category)
            stale_evidence = bool(last is not None and self._invalidated_after(last["step"]))
            rows.append({"region": r, "status": reg["status"] if reg else UNKNOWN,
                         "coverage": last["coverage"] if last else 0.0,
                         "step": last["step"] if last else None,
                         "seen": category in self.seen_in(r),
                         "unseen_part_could_hide_it": hide,
                         "evidence_invalidated_by_later_action": stale_evidence,
                         "cleared": bool(last is not None and hide is False and not stale_evidence)})
        never = [r for r, reg in self.regions.items() if reg["last"] is None and r not in cands]
        if found:
            outcome = "found"
        elif all(row["cleared"] for row in rows):
            outcome = "not_found_in_inspected_scope"
        else:
            outcome = "not_found_scope_incomplete"
        insp = [row["coverage"] for row in rows if row["step"] is not None]
        return {"category": category, "outcome": outcome,
                "found_tracks": [{"track_id": t["track_id"], "status": t["status"],
                                  "region": t.get("picked_from", t["region"])} for t in found],
                "scope": rows, "min_coverage": round(min(insp), 4) if insp else None,
                "not_cleared": [row["region"] for row in rows if not row["cleared"]],
                "not_inspected_outside_scope": never + ["other surfaces not in view"],
                "note": "not seen is not absent: the statement holds for the inspected scope only, and "
                        "only for an object at least as large as the category's catalog footprint"}

    def diff_report(self) -> dict:
        entries = []
        declared_pairs = {(d["category"], d["region"]) for d in self.dossier.stored_items}
        for d in self.dossier.stored_items:
            cat, rid = d["category"], d["region"]
            reg = self.regions.get(rid)
            elsewhere = sorted({r for r, rg in self.regions.items()
                                if r != rid and rg["last"] is not None and cat in rg["last"]["seen_categories"]})
            held = [t for t in self.tracks.values() if t["category"] == cat and t["status"] == HELD]
            if reg is None or reg["last"] is None:
                e = {"type": "declared_location_not_inspected", "status": UNKNOWN}
            elif cat in reg["last"]["seen_categories"]:
                e = {"type": "confirmed", "status": reg["status"]}
            else:
                e = {"type": "not_seen_at_declared_location", "status": reg["status"],
                     "coverage": reg["last"]["coverage"],
                     "unseen_part_could_hide_it": could_hide(reg["last"], cat)}
            if reg is not None and reg["last"] is not None:
                e["inspected_step"] = reg["last"]["step"]
                e["stale"] = reg["status"] == STALE
            entries.append({"category": cat, "declared_region": rid, **e, "observed_in": elsewhere,
                            "held_after_pick_from": [t.get("picked_from") for t in held]})
        for rid, reg in self.regions.items():
            if reg["last"] is None:
                continue
            for cat in reg["last"]["seen_categories"]:
                if cat in C.TOOL_CATEGORIES and (cat, rid) not in declared_pairs:
                    entries.append({"category": cat, "type": "observed_not_declared", "observed_region": rid,
                                    "step": reg["last"]["step"], "stale": reg["status"] == STALE,
                                    "declared_regions": sorted(r for c, r in declared_pairs if c == cat)})
        for m in self.dossier.modules:
            cat = C.MODULE_KIND_TO_CATEGORY[m["kind"]]
            obs = [x for x in self.modules.values() if x["category"] == cat]
            if not obs:
                t = {"type": "module_not_observed", "status": UNKNOWN}
            elif any(x["slot_estimate"] == m["slot"] for x in obs):
                t = {"type": "module_confirmed", "status": OBSERVED}
            else:
                t = {"type": "module_at_other_slot", "status": OBSERVED,
                     "observed_slots": sorted({str(x["slot_estimate"]) for x in obs})}
            entries.append({"category": cat, "declared_slot": m["slot"], **t})
        declared_mods = {C.MODULE_KIND_TO_CATEGORY[m["kind"]] for m in self.dossier.modules}
        for x in self.modules.values():
            if x["category"] not in declared_mods:
                entries.append({"category": x["category"], "type": "module_not_declared",
                                "observed_slot": x["slot_estimate"]})
        changed = [e for e in entries if e["type"] not in ("confirmed", "module_confirmed")]
        return {"label": SIM_LABEL, "dossier_version": self.dossier.version, "basis_observations": list(self.obs_ids),
                "entries": entries, "n_differences_or_unknowns": len(changed),
                "rule": "computed from dossier and robot observations only; not seen is not absent"}
