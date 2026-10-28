"""Paired variants of one kitchen: the hidden truth changes, the dossier stays as it was.

A variant is (hidden truth SiteSpec, dossier written for the original layout, change log). The change log is hidden
truth for the bench; the robot never receives it.
- "correct": nothing changed; the dossier is right.
- "moved": the sponge is in another drawer than the dossier says (a drawer with no item, else swapped with that
  drawer's item; a sponge declared in the sink is moved into a drawer); None when this is not possible.
- "absent": the sponge is removed.
- "moved_jittered": as "moved", with the sponge anywhere in the drawer (some places stay under the unit top).
- "moved_surface": the sponge lies in the sink basin instead of its drawer.
"""
from __future__ import annotations

import copy

import numpy as np

from .spec import DRAWER_ITEM_POS, SINK_SIDE_POS, SiteSpec

VARIANTS = ("correct", "moved", "absent")
EXTRA_VARIANTS = ("moved_jittered", "moved_surface")
ALL_VARIANTS = VARIANTS + EXTRA_VARIANTS
JITTER_ALONG = (-0.085, 0.075)
JITTER_SIDE = (-0.03, 0.03)


def _drawer_of(it) -> int | None:
    return int(it.where.split("/")[1]) if it.where.startswith("drawers/") else None


def make_variant(spec: SiteSpec, variant: str, rng_seed: int | None = None):
    """Returns (truth_spec, dossier_spec, change_log) or None if the variant does not apply."""
    if variant not in ALL_VARIANTS:
        raise ValueError(variant)
    dossier_spec = copy.deepcopy(spec)
    truth = copy.deepcopy(spec)
    sponge = next((i for i in truth.items if i.kind == "sponge"), None)
    if sponge is None:
        return None
    if variant == "correct":
        return truth, dossier_spec, []
    if variant == "absent":
        truth.items = [i for i in truth.items if i is not sponge]
        return truth, dossier_spec, [{"change": "removed", "item": "sponge", "was": sponge.where}]
    if variant == "moved_surface":
        if sponge.where == "sink_side" or not any(m.kind == "sink" for m in truth.modules):
            return None
        log = [{"change": "moved", "item": "sponge", "from": sponge.where, "to": "sink_side"}]
        sponge.where, sponge.pos = "sink_side", SINK_SIDE_POS["sponge"]
        return truth, dossier_spec, log
    # moved, moved_jittered
    n = int(truth.module("drawers").params.get("n", 2))
    rng = np.random.default_rng([spec.seed, 7331] if rng_seed is None else rng_seed)
    k0 = _drawer_of(sponge)
    occupied = {_drawer_of(i): i for i in truth.items if _drawer_of(i) is not None}
    others = [k for k in range(n) if k != k0]
    empty = [k for k in others if k not in occupied]
    log = []
    if empty:
        k = int(rng.choice(empty))
    elif k0 is not None:
        k = int(rng.choice(others))
        occ = occupied[k]
        occ.where, occ.pos = f"drawers/{k0}", DRAWER_ITEM_POS[occ.kind]
        log.append({"change": "swapped", "item": occ.kind, "from": f"drawers/{k}", "to": f"drawers/{k0}"})
    else:
        swappable = [k for k in others if occupied[k].kind in SINK_SIDE_POS]
        if not swappable:
            return None
        k = int(rng.choice(swappable))
        occ = occupied[k]
        occ.where, occ.pos = "sink_side", SINK_SIDE_POS[occ.kind]
        log.append({"change": "swapped", "item": occ.kind, "from": f"drawers/{k}", "to": "sink_side"})
    log.insert(0, {"change": "moved", "item": "sponge", "from": sponge.where, "to": f"drawers/{k}"})
    sponge.where, sponge.pos = f"drawers/{k}", DRAWER_ITEM_POS["sponge"]
    if variant == "moved_jittered":
        jr = np.random.default_rng([spec.seed, 7332])
        sponge.pos = (round(float(jr.uniform(*JITTER_ALONG)), 4), round(float(jr.uniform(*JITTER_SIDE)), 4))
        log[0]["pos_in_drawer"] = list(sponge.pos)
    return truth, dossier_spec, log
