"""Rule-based mock model for wiring tests. Not Nemotron, not a language model.

It exercises the wiring only: task text, dossier and robot-side observations -> prompt -> a BotScript program ->
parser -> isolated interpreter -> skill facade -> evaluator -> observable feedback -> correction -> memory. Its programs
were written by hand for this purpose; a mock answer says nothing about what a real model would write.

Rule (deterministic, reads only the prompt messages): the first request of an episode gets a "dossier program" that
trusts the dossier; any later request gets a "search program" that searches the regions the robot has not cleared yet
and ends with help(kind="search_incomplete") when regions remain. Not used by the offline demo.
"""
from __future__ import annotations

MOCK_MODEL_ID = "mock/rule-based-botscript-0.2"  # 0.2: handle refine and pull retry as in S (after run1)
MOCK_LABEL = "mock (programmes ecrits a la main, regle fixe) : prouve le branchement, pas Nemotron"

_COMMON_PICK_SCRUB = '''
# ---- pick the sponge from its perceived pose (top grasp, fingers across its long axis), two tries
held = False
tries = 0
while tries < 2 and not held:
    tries += 1
    y = None
    if found["yaw_rad"] != None:
        y = found["yaw_rad"] + 1.5708
    r = pick(found["position_m"], "sponge", "top", y)
    held = r["ok"]
    if not held:
        log("pick failed", r["reason"])
        r2 = look_down_at([found["position_m"][0], found["position_m"][1]], found["position_m"][2] - 0.02, 0.42)
        if r2["ok"]:
            o = observe("wrist_cam")
        fresh = None
        for t in tracks("sponge"):
            if t["status"] == "observed":
                fresh = t
        if fresh == None:
            give_up("pick_failed_not_reseen")
        found = fresh
if not held:
    give_up("pick_failed")
# ---- the bowl on the landing tray (the one nearest to the robot's base if several)
bowl = None
for attempt in range(2):
    if bowl == None:
        best = 99.0
        b0 = base_xy()
        for b in tracks("bowl"):
            if b["region"] == "landing_tray":
                dd = hypot(b["position_m"][0] - b0[0], b["position_m"][1] - b0[1])
                if dd < best:
                    best = dd
                    bowl = b
        if bowl == None and attempt == 0:
            r = look_at_region("landing_tray")
            if r["ok"]:
                o = observe("wrist_cam")
if bowl == None:
    give_up("bowl_not_seen")
r = scrub(vadd(bowl["position_m"], [0.0, 0.0, 0.023]), 3.0, 4.0)
if r["ok"]:
    done()
give_up("scrub_" + r["reason"])
'''

DOSSIER_DIRECT = '''# Dossier program (MOCK, hand-written): trust the dossier, go to the declared place of the sponge.
d = dossier()
region = None
for it in d["stored_items"]:
    if it["category"] == "sponge":
        region = it["region"]
found = None
for t in tracks("sponge"):
    if t["status"] == "observed":
        found = t
if found == None:
    if region == None:
        help("dossier_declares_no_sponge", kind="other")
    if region[0:12] == "drawer_unit:":
        h = handle_for_region(region)
        if h == None:
            give_up("declared_drawer_handle_not_seen")
        if h["sigma_m"][0] > 0.01 or h["truncated"] or h["visibility_estimate"] < 0.6:
            r = look_down_at([h["position_m"][0], h["position_m"][1]], h["position_m"][2], 0.40)
            if r["ok"]:
                o = observe("wrist_cam")
                h = handle_for_region(region)
        pulled = h["opening_m"] >= 0.15
        pulls = 0
        while not pulled and pulls < 2:
            pulls += 1
            r = open_drawer(h["position_m"], h["pull_dir"], max(0.05, 0.24 - h["opening_m"]))
            if r["ok"]:
                pulled = True
            else:
                log("pull failed", region, r["reason"])
                p = h["pull_dir"]
                r = look_down_at([h["position_m"][0] + 0.08 * p[0], h["position_m"][1] + 0.08 * p[1]], h["position_m"][2], 0.50)
                if r["ok"]:
                    o = observe("wrist_cam")
                h = handle_for_region(region)
                if h == None:
                    pulls = 2
                elif h["opening_m"] >= 0.15:
                    pulled = True
        if not pulled:
            give_up("open_declared_drawer_failed")
    r = look_at_region(region)
    if r["ok"]:
        o = observe("wrist_cam")
    for t in tracks("sponge"):
        if t["status"] == "observed":
            found = t
    if found == None:
        if region[0:12] == "drawer_unit:":
            h = handle_for_region(region)
            if h != None and h["opening_m"] > 0.02:
                r = close_drawer(h["position_m"], [0.0 - h["pull_dir"][0], 0.0 - h["pull_dir"][1], 0.0], h["opening_m"])
        give_up("sponge_not_at_declared_place")
''' + _COMMON_PICK_SCRUB

SEARCH = '''# Search program (MOCK, hand-written): search the regions not cleared yet, then pick and scrub.
d = dossier()
n_drawers = 0
for m in d["modules"]:
    if m["kind"] == "drawers":
        n_drawers = get(m, "n_drawers", 0)
if len(tracks("drawer_handle")) < n_drawers:
    units = tracks("drawer_unit")
    if len(units) > 0:
        r = look_down_at([units[0]["position_m"][0], units[0]["position_m"][1]], worktop_z(), 0.50)
        if r["ok"]:
            o = observe("wrist_cam")
found = None
for t in tracks("sponge"):
    if t["status"] == "observed":
        found = t
sc = search_scope("sponge")
n_regions = 0
for row in sc["scope"]:
    if found != None:
        break
    if row["cleared"] or n_regions >= 6:
        continue
    region = row["region"]
    n_regions += 1
    is_drawer = region[0:12] == "drawer_unit:"
    if is_drawer:
        h = handle_for_region(region)
        if h == None:
            log("no handle perceived for", region)
            continue
        if h["sigma_m"][0] > 0.01 or h["truncated"] or h["visibility_estimate"] < 0.6:
            r = look_down_at([h["position_m"][0], h["position_m"][1]], h["position_m"][2], 0.40)
            if r["ok"]:
                o = observe("wrist_cam")
                h = handle_for_region(region)
        pulled = h["opening_m"] >= 0.15
        pulls = 0
        while not pulled and pulls < 2:
            pulls += 1
            r = open_drawer(h["position_m"], h["pull_dir"], max(0.05, 0.24 - h["opening_m"]))
            if r["ok"]:
                pulled = True
            else:
                log("pull failed", region, r["reason"])
                p = h["pull_dir"]
                r = look_down_at([h["position_m"][0] + 0.08 * p[0], h["position_m"][1] + 0.08 * p[1]], h["position_m"][2], 0.50)
                if r["ok"]:
                    o = observe("wrist_cam")
                h = handle_for_region(region)
                if h == None:
                    pulls = 2
                elif h["opening_m"] >= 0.15:
                    pulled = True
        if not pulled:
            continue
    r = look_at_region(region)
    if r["ok"]:
        o = observe("wrist_cam")
        for t in tracks("sponge"):
            if t["status"] == "observed":
                found = t
        if found == None:
            hide = None
            for rg in o["regions"]:
                if rg["region"] == region:
                    hide = get(rg["could_hide"], "sponge", None)
            if hide != False:
                r = look_at_region(region, "unseen")
                if r["ok"]:
                    o = observe("wrist_cam")
                    for t in tracks("sponge"):
                        if t["status"] == "observed":
                            found = t
    if found == None and is_drawer:
        h = handle_for_region(region)
        if h != None and h["opening_m"] > 0.02:
            r = close_drawer(h["position_m"], [0.0 - h["pull_dir"][0], 0.0 - h["pull_dir"][1], 0.0], h["opening_m"])
if found == None:
    sc = search_scope("sponge")
    if sc["outcome"] == "not_found_in_inspected_scope":
        not_found("inspected_scope")
    help("sponge_not_found_scope_incomplete", kind="search_incomplete")
''' + _COMMON_PICK_SCRUB


def answer(messages: list[dict], n: int) -> str:
    """MockLLMClient answers=answer. Reads only the prompt text."""
    user = messages[-1]["content"]
    later = "PREVIOUS PROGRAM:" in user or "\nMEMORY:\n" in user
    prog = SEARCH if later else DOSSIER_DIRECT
    return "Mock answer (hand-written program, not a model output).\n```botscript\n" + prog + "```\n"
