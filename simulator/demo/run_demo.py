"""The original demo launcher: runs ONE episode of the frozen robot code with the demo client.

This is the launcher that recorded the three episodes shown on the replay page. In this package the public entry
point is demo/run_episode.py: it reuses the checks and the identity writer below (seed_whitelist_problems,
hygiene_problems, world_dev_module_name, finish) but not this command line, whose default folders belong to the
development lab the episodes were recorded in (demo_client.LAB) and are not part of this package.

What one episode fixes, all of it written into DEMO_IDENTITY.json:
- the robot loop as frozen: bench_agent.run_agent_episode with information profile s5-info-1.0, the scene reading
  recorded, limits 0.8, criteria 1.4, no program library, the demo client (demo_client.DemoClient) in place of a model
  client, and a MemoryLedger (the demo client reports zero usage; the cost of a real choice is recorded with that call
  only);
- the scenes of the three situations, all on development kitchen 31000: A4 sponge in place (variant correct), A1
  sponge moved (variant moved) and A5 sponge masked (variant correct inside the development world
  occlusion-dev-1:towel_over, the catalog towel lying on the sponge in its drawer, applied around the episode by
  world_dev.occlusion.applied() with its own refusals; the robot never lifts the towel). A2 and A3 are earlier scenes
  that the public entry point does not run;
- the kitchens and seeds: a whitelist checked before any import of the robot code and before any simulation
  (seed_whitelist_problems). Each scene runs only on its own (kitchen, noise seed) pair, and the camera seed is the
  noise seed + 1, both inside the demo range 960000-960099;
- the modes: prepare (writes the choice request when both options are valid; such an episode ends with a sentinel
  help and is never shown), replay (serves a recorded answer only on an equal request fingerprint), rule (no call
  possible);
- hygiene: refuses to start with NEBIUS_API_KEY or OPENAI_API_KEY in the environment, with bytecode writing on, or
  from a working folder inside the lab; counts __pycache__ and .pyc under the code root before and after;
- the loop's request 2 (the robot prompt, kept in requests.json) is answered by the demo client and never sent to
  Nemotron.

An episode is marked showable only when the client served no sentinel and refused nothing, no exception was raised,
no answer came from a fake transport, the code tree is the designated frozen one, the code root holds no bytecode
before or after, the copies of the recorded choice files equal the record, and the scene's seed pair is confirmed.

Output: a NEW folder <out-root>/<scene>_<mode>_<kitchen>_n<noise>_run<N>/ (never deleted, never re-created).

usage (from a working folder outside the lab):
  env -u NEBIUS_API_KEY -u OPENAI_API_KEY PYTHONDONTWRITEBYTECODE=1 <python> -B \\
      demo/run_demo.py --scene A1 --mode prepare --kitchen 31000 --noise 960000 --root <frozen code root>
--root is required and has no default, so that a run cannot pick up another code tree by mistake.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
TERRAIN = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import choice_prompt as CP  # noqa: E402
import demo_client as DC  # noqa: E402

DEMO_VERSION = "demoA-run-3"
INFO_PROFILE = "s5-info-1.0"
RECORD_SCENE_READING = True
LIMITS_VERSION = "agent-limits-s5-0.8"
CRITERIA_VERSION = "task-criteria-1.4"
A3_WORLD_DEV = "occlusion-dev-2:edge_back"
PROXY_MODE = "towel_over"  # occlusion-dev-1 mode whose refusal rules A1 and A2 are checked with (no world change)
SCENES = {
    "A1": {"variant": "moved", "world_dev": None, "situation": "moved",
           "title": "found: the sponge moved to another drawer"},
    "A2": {"variant": "absent", "world_dev": None, "situation": "sponge_removed",
           "title": "absent: the sponge removed from the kitchen"},
    "A3": {"variant": "correct", "world_dev": A3_WORLD_DEV, "situation": "on_edge_at_the_back",
           "title": "hard: the sponge on its long edge at the back of a drawer, partly under the unit top"},
}
# The demo's kitchens and noise/camera seed pairs, nothing else
DEMO_KITCHENS = ("reference", "31000")
DEMO_SEED_RANGE = (960000, 960099)  # inclusive; BOTH streams of a scene (noise n, camera n + 1) must lie inside
DEMO_SEED_PAIRS = ((960000, 960001), (960010, 960011), (960020, 960021))  # (noise, camera)
# Each scene runs on its confirmed (kitchen, noise) pair only
DEMO_SCENE_SEEDS = {"A1": ("31000", 960000), "A2": ("reference", 960010), "A3": ("reference", 960020)}
# Seed table v2: first proposed, then confirmed in review (below). The three situations of the demo, all on kitchen
# 31000: in place (A4), moved (A1, the confirmed pair above), masked (A5, occlusion-dev-1:towel_over). The new pairs
# were never run or named before (960030 and 960098 stay the tests' refused examples); both streams lie inside
# DEMO_SEED_RANGE, spaced by 10 like the confirmed pairs.
SEED_TABLE_V2_STATUS = "PROPOSED, to be confirmed in review"  # its status before the decision below (kept as written)
SEED_TABLE_V2_CONFIRMED = True  # set True only by the written review decision recorded below
# The decision confirms A4 31000/960040/960041 and A5 31000/960050/960051, A5 only with occlusion-dev-1:towel_over
# (the thick catalog towel); A1 is unchanged. The record below is the one the recorded episodes carry, with the
# time written in UTC and the review note named without its private path (the note is not part of this package).
SEED_TABLE_V2_CONFIRMATION = {
    "by": "review", "date": "2026-10-20T21:40Z",
    "path": "review note of 2026-10-20, not included in this package",
    "sha256": "35bd4e93a667229e165aa0c693a0c010313d4c1e92a77e65921710062220dbb5",  # the note as read on
    "section": "1 (answers to the four questions)",
    "scope": "A4 31000/960040 (camera 960041); A5 31000/960050 (camera 960051), only with occlusion-dev-1:towel_over"}
A5_WORLD_DEV = "occlusion-dev-1:towel_over"
SCENES_V2_PROPOSED = {
    "A4": {"variant": "correct", "world_dev": None, "situation": "in_place",
           "title": "in place: the sponge where the site file puts it"},
    "A5": {"variant": "correct", "world_dev": A5_WORLD_DEV, "situation": "masked",
           "title": "masked: the catalog towel lying on the sponge in its drawer (never lifted by the robot)"},
}
DEMO_SEED_PAIRS_V2_PROPOSED = ((960040, 960041), (960050, 960051))  # (noise, camera)
DEMO_SCENE_SEEDS_V2_PROPOSED = {"A4": ("31000", 960040), "A5": ("31000", 960050)}
ALL_SCENES = {**SCENES, **SCENES_V2_PROPOSED}
# The demo's three situations (approved framing) and the scene that shows each
SITUATIONS = {"in_place": "A4", "moved": "A1", "masked": "A5"}
S5_OUT = DC.LAB / "outputs" / "siteagent" / "s5"
LIVE_REGISTRY = DC.LAB / "siteagent" / "config" / "development_seed_registry_not_included.json"
KEY_VARS = ("NEBIUS_API_KEY", "OPENAI_API_KEY")
REQUEST_2_NOTE = ("The loop's request 2 (the designated robot prompt, kept in requests.json) is answered by the demo "
                  "client; it is never sent to Nemotron. Nemotron receives choice_request.json only (text only).")


def camera_seed(noise: int) -> int:
    """The bench's camera stream of an episode: noise + 1 (bench_task.py, SimulatedCamera(noise_seed=noise + 1))."""
    return noise + 1


def seed_whitelist_problems(kitchen, noise, scene=None) -> list[str]:
    """Why this kitchen and noise seed are refused (empty: accepted). Pure, no import of the robot code: run
    by main() before anything else. Accepted only: a kitchen of DEMO_KITCHENS, and a noise seed whose (noise, camera)
    pair is in DEMO_SEED_PAIRS with both seeds inside DEMO_SEED_RANGE. With a scene (main() always gives it), only
    that scene's confirmed (kitchen, noise) of DEMO_SCENE_SEEDS is accepted."""
    out = []
    v2 = scene is not None and scene in DEMO_SCENE_SEEDS_V2_PROPOSED  # seed table v2 (PROPOSED), scenes A4 and A5
    if scene is not None:
        want = DEMO_SCENE_SEEDS_V2_PROPOSED.get(scene) if v2 else DEMO_SCENE_SEEDS.get(scene)
        if want is None:
            out.append(f"scene {scene!r} is not a demo scene (only "
                       f"{', '.join([*DEMO_SCENE_SEEDS, *DEMO_SCENE_SEEDS_V2_PROPOSED])})")
        elif (kitchen, noise) != want or type(noise) is not int:
            src = (f"seed table v2, {seed_table_v2_status()}" if v2
                   else "seed table v1, confirmed triple")
            out.append(f"scene {scene} runs only on kitchen {want[0]} with noise {want[1]} ({src}), not kitchen "
                       f"{kitchen!r} with noise {noise!r}")
    if kitchen not in DEMO_KITCHENS:
        out.append(f"kitchen {kitchen!r} is not a demo kitchen (only {', '.join(DEMO_KITCHENS)})")
    if type(noise) is not int:
        return out + [f"noise seed {noise!r} is not an integer"]
    cam, (lo, hi) = camera_seed(noise), DEMO_SEED_RANGE
    if not (lo <= noise <= hi and lo <= cam <= hi):
        out.append(f"noise seed {noise} and camera seed {cam} are not both inside the demo range {lo}-{hi}")
    pairs = DEMO_SEED_PAIRS_V2_PROPOSED if v2 else DEMO_SEED_PAIRS
    if (noise, cam) not in pairs:
        out.append(f"noise/camera {noise}/{cam} is not a {'proposed (seed table v2)' if v2 else 'confirmed'} demo "
                   "pair (only " + ", ".join(f"{n}/{c}" for n, c in pairs) + ")")
    return out


def seed_table_v2_status() -> str:
    """The v2 table's status string: the confirmed status once SEED_TABLE_V2_CONFIRMED (see
    SEED_TABLE_V2_CONFIRMATION), else SEED_TABLE_V2_STATUS."""
    return "confirmed in review" if SEED_TABLE_V2_CONFIRMED else SEED_TABLE_V2_STATUS


def seed_table_of(scene) -> dict:
    """Which seed table a scene runs on, and whether that table is confirmed (v1: confirmed; v2: proposed until
    SEED_TABLE_V2_CONFIRMED, then confirmed by the decision of SEED_TABLE_V2_CONFIRMATION)."""
    if scene in DEMO_SCENE_SEEDS_V2_PROPOSED:
        return {"table": "v2", "status": seed_table_v2_status(),
                "confirmed": bool(SEED_TABLE_V2_CONFIRMED), "source": "seed table v2 (review note not included)",
                "decision": dict(SEED_TABLE_V2_CONFIRMATION) if SEED_TABLE_V2_CONFIRMED else None}
    return {"table": "v1", "status": "confirmed in review (2026-10-19)", "confirmed": True,
            "source": "seed table v1 (review note not included)"}


def world_dev_module_name(spec) -> str | None:
    """The frozen root's module that builds a world_dev spec (the same rule as occlusion2.module_for), for the
    identity: occlusion for occlusion-dev-1 (A5), occlusion2 for occlusion-dev-2 (A3); None without a world."""
    if spec is None:
        return None
    dev2 = str(spec).startswith("occlusion-dev-2:")
    return "siteagent.world_dev.occlusion2" if dev2 else "siteagent.world_dev.occlusion"


def file_sha(p: Path) -> str | None:
    p = Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def tree_sha(root: Path, rel: str) -> tuple[str, int]:
    """Tree rule: sha256 of canonical JSON [[relpath, sha256]] over the files, without __pycache__ (the same
    function as the code root a choice request records: demo_client.tree_sha)."""
    return DC.tree_sha(root, rel)


def bytecode_count(root: Path) -> dict:
    root = Path(root)
    return {"pycache_dirs": sum(1 for p in root.rglob("__pycache__") if p.is_dir()),
            "pyc_files": sum(1 for _ in root.rglob("*.pyc"))}


def hygiene_problems(environ, cwd: Path, dont_write_bytecode: bool) -> list[str]:
    """Why this process may not start (empty: it may)."""
    out = [f"{k} is in the environment (prepare, replay and rule run without a key)" for k in KEY_VARS
           if environ.get(k)]
    if not dont_write_bytecode or environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        out.append("run with python -B and PYTHONDONTWRITEBYTECODE=1")
    c = Path(cwd).resolve()
    if c == DC.LAB.resolve() or DC.LAB.resolve() in c.parents:
        out.append(f"the working folder is inside the lab ({c})")
    return out


NO_KEY_ENV = "env -u NEBIUS_API_KEY -u OPENAI_API_KEY PYTHONDONTWRITEBYTECODE=1"


def send_commands(choice_dir: Path, fp: str, python: str, root: Path, replay: str = "") -> dict:
    """The two send_choice.py commands handed over after prepare: the dry run never has a key in its environment;
    only the real call keeps NEBIUS_API_KEY, in a session where the operator put it there by hand. Both carry
    --root, the code root this episode was prepared with (send_choice.py refuses another one), and so does the replay
    command (replay: its --scene, --kitchen and --noise part). The replay command is complete, with no key in its
    environment, like the dry run."""
    q = lambda x: shlex.quote(str(x))  # noqa: E731
    tail = (f"{q(python)} -B {q(HERE / 'send_choice.py')} --choice-dir {q(choice_dir)} --expect-sha256 {fp} "
            f"--root {q(root)}")
    return {"fake": f"{NO_KEY_ENV} {tail} --fake-transport",
            "execute": f"env -u OPENAI_API_KEY PYTHONDONTWRITEBYTECODE=1 {tail} --execute",
            "replay": " ".join(x for x in (NO_KEY_ENV, q(python), "-B", q(HERE / "run_demo.py"), replay.strip(),
                                           "--mode replay", f"--root {q(root)}", f"--choice-dir {q(choice_dir)}")
                               if x)}


def showable_status(*, sentinel: bool, refusal, err, fake_answer: bool, same_as_designated, bytecode_before: dict,
                    bytecode_after: dict, copies_same: bool = True, seeds_confirmed: bool = True) -> tuple[bool, list]:
    """(showable, reasons). Besides a client that served no sentinel and refused nothing, no
    exception and no fake answer, the code tree must be the designated one (same_as_designated is True: never for a
    root without DESIGNATION.json) and the frozen root must hold no bytecode, before and after the episode. And the
    scene's seed pair must be confirmed (a pair of seed table v2 counts only once that table is confirmed)."""
    zero = {"pycache_dirs": 0, "pyc_files": 0}
    reasons = []
    if sentinel:
        reasons.append("sentinel help (prepare mode)")
    if refusal is not None:
        reasons.append("the demo client refused")
    if fake_answer:
        reasons.append("fake-transport answer (dry run)")
    if err is not None:
        reasons.append("exception")
    if same_as_designated is not True:
        reasons.append("the code tree is not the designated one (no DESIGNATION.json or another tree)")
    if bytecode_before != zero or bytecode_after != zero:
        reasons.append(f"bytecode under the frozen root (before {bytecode_before}, after {bytecode_after})")
    if not copies_same:
        reasons.append("a copy of the recorded choice files differs from the record")
    if seeds_confirmed is not True:
        reasons.append("seed pair proposed, not yet confirmed in review (seed table v2)")
    return not reasons, reasons


def episode_dir(out_root: Path, scene: str, mode: str, kitchen: str, noise: int) -> Path:
    """A new run folder name (never an existing one)."""
    n = 1
    while (out_root / f"{scene}_{mode}_{kitchen}_n{noise}_run{n}").exists():
        n += 1
    return out_root / f"{scene}_{mode}_{kitchen}_n{noise}_run{n}"


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True, choices=sorted(ALL_SCENES),
                    help="A1 moved, A2 removed, A3 on edge (seed table v1); A4 in place, A5 masked (seed table v2)")
    ap.add_argument("--mode", required=True, choices=DC.MODES)
    ap.add_argument("--kitchen", required=True, choices=DEMO_KITCHENS)
    ap.add_argument("--noise", required=True, type=int,
                    help="noise seed: 960000, 960010 or 960020 (seed table v1); 960040 or 960050 (seed table v2); "
                         "camera seed noise + 1")
    ap.add_argument("--root", required=True,
                    help="frozen code root, required (no default; the three situations: the DESIG-4 copy "
                         "outputs/siteagent/s5/desig4_20261020/root); carried into the send and replay commands")
    ap.add_argument("--out-root", help="default: <lab>/outputs/siteagent/s5/demoA_<yyyymmdd>")
    ap.add_argument("--programs", default=str(DC.PROGRAMS_DIR),
                    help="folder of " + ", ".join(DC.PROGRAM_FILES.values()))
    ap.add_argument("--choice-dir", help="replay: the folder of the recorded choice (prepare episode's choice/)")
    ap.add_argument("--allow-fake-answer", action="store_true",
                    help="replay a fake-transport answer (dry run only: the episode is marked not showable)")
    ap.add_argument("--registry", default=str(LIVE_REGISTRY), help="seed registry read for the refusals")
    return ap


def main(argv=None) -> int:
    a = parser().parse_args(argv)
    seeds = seed_whitelist_problems(a.kitchen, a.noise, a.scene)  # before any robot code import
    if seeds:
        print("refused (seed whitelist; robot code not imported, nothing simulated): " + "; ".join(seeds), file=sys.stderr)
        return 2
    problems = hygiene_problems(os.environ, Path.cwd(), sys.flags.dont_write_bytecode)
    if problems:
        print("refused: " + "; ".join(problems), file=sys.stderr)
        return 2
    if a.mode == "replay" and not a.choice_dir:
        print("refused: replay needs --choice-dir", file=sys.stderr)
        return 2
    if a.mode != "replay" and (a.choice_dir or a.allow_fake_answer):
        print("refused: --choice-dir and --allow-fake-answer belong to replay", file=sys.stderr)
        return 2
    scene = ALL_SCENES[a.scene]
    root = DC.use_root(a.root)
    before = bytecode_count(root)
    if before["pycache_dirs"] or before["pyc_files"]:
        print(f"refused: the frozen root already holds bytecode {before}; check the copy first", file=sys.stderr)
        return 2
    BA = DC.root_module("siteagent.agent.bench_agent", root)
    LG = DC.root_module("siteagent.agent.ledger", root)
    LM = DC.root_module("siteagent.agent.limits", root)
    TC = DC.root_module("siteagent.taskcheck", root)
    O1 = DC.root_module("siteagent.world_dev.occlusion", root)
    O2 = DC.root_module("siteagent.world_dev.occlusion2", root) if scene["world_dev"] else None
    # the scene's world module, by the frozen root's own dispatcher: occlusion (dev-1, A5) or occlusion2 (dev-2, A3)
    WD = O2.module_for(scene["world_dev"]) if scene["world_dev"] else None
    if WD is not None and WD.__name__ != world_dev_module_name(scene["world_dev"]):
        print(f"refused: world {scene['world_dev']} resolved to {WD.__name__}", file=sys.stderr)
        return 2
    lim, crit = LM.load_limits(), TC.load_criteria()
    if lim.version != LIMITS_VERSION or crit["version"] != CRITERIA_VERSION:
        print(f"refused: limits {lim.version} and criteria {crit['version']}, expected {LIMITS_VERSION} and "
              f"{CRITERIA_VERSION}", file=sys.stderr)
        return 2
    registry_path = Path(a.registry)
    reg = {"ranges": json.loads(registry_path.read_text(encoding="utf-8"))["ranges"]}
    out_root = Path(a.out_root) if a.out_root else S5_OUT / f"demoA_{datetime.now(DC.TIMEZONE):%Y%m%d}"
    ep = episode_dir(out_root, a.scene, a.mode, a.kitchen, a.noise)
    if scene["world_dev"]:
        why = WD.refusal(a.kitchen, WD.parse(scene["world_dev"]), [a.noise], reg, ep, variant=scene["variant"],
                         s5_root=S5_OUT)
    else:
        why = O1.refusal(a.kitchen, PROXY_MODE, [a.noise], reg, ep, variant=None, s5_root=S5_OUT)
    if why:
        print(why, file=sys.stderr)
        return 2
    try:
        programs = DC.load_programs(a.programs)
        recorded = DC.load_recorded(a.choice_dir, allow_fake=a.allow_fake_answer, root=root) \
            if a.mode == "replay" else None
    except (FileNotFoundError, ValueError) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    ep.mkdir(parents=True, exist_ok=False)
    client = DC.DemoClient(a.mode, programs, root=root, recorded=recorded, calls_path=ep / "demo_calls.jsonl",
                           choice_dir=ep / "choice" if a.mode == "prepare" else None)
    ledger = LG.MemoryLedger(label="demo A: no money spent by the loop; the demo client returns zero usage; the real "
                                   "cost of a choice is in choice_call.json (send_choice.py)")
    started = DC.now_text()
    row, err, world_state = None, None, None

    def run():
        return BA.run_agent_episode(a.kitchen, scene["variant"], a.noise, ep, client, ledger, None, lim, None, crit,
                                    calls_log=ep / "calls.jsonl",
                                    calls_meta={"demo": DEMO_VERSION, "scene": a.scene, "mode": a.mode},
                                    info_profile=INFO_PROFILE, record_scene_reading=RECORD_SCENE_READING)
    try:
        if scene["world_dev"]:
            with WD.applied(scene["world_dev"], kitchen=a.kitchen, registry=reg, noise_seeds=[a.noise], out_dir=ep,
                            s5_root=S5_OUT) as state:
                world_state = state  # the post-settle checks are appended to it while the episode runs
                row = run()
        else:
            row = run()
    except BaseException as e:  # kept in the identity, then raised again
        err = f"{type(e).__name__}: {str(e)[:300]}"
        raise
    finally:
        finish(a, scene, root, ep, client, ledger, row, err, world_state, recorded, before, started, programs,
               registry_path, lim, crit)
    after = bytecode_count(root)
    if after != before:
        print(f"WARNING: bytecode under the frozen root changed {before} -> {after}", file=sys.stderr)
        return 4
    if client.refusal is not None:
        print(f"the demo client refused and stopped the episode: {client.refusal['code']}", file=sys.stderr)
        return 3
    if a.mode == "prepare" and client.sentinel_served:
        cmds = send_commands(ep / "choice", client.choice["choice_request_sha256"], sys.executable, root,
                             replay=f"--scene {a.scene} --kitchen {a.kitchen} --noise {a.noise}")
        print("Prepare done: this episode is NOT showable (sentinel help).\n"
              f"Dry run first (0 USD, nothing sent, no key in its environment):\n  {cmds['fake']}\n"
              "Then, only in a session where the operator put NEBIUS_API_KEY in the environment by hand:\n"
              f"  {cmds['execute']}\n"
              f"Then replay (the same --root, no key in its environment):\n  {cmds['replay']}")
    print(json.dumps({"episode": str(ep), "final": (row or {}).get("agent", {}).get("final_status"),
                      "who_chose": (client.choice or {}).get("who_chose")}, indent=1, default=str))
    return 0


def finish(a, scene, root, ep, client, ledger, row, err, world_state, recorded, before, started, programs,
           registry_path, lim, crit):
    """Writes DEMO_IDENTITY.json, ledger_demo.json, episode_row.json, world_dev_state.json and, in replay, verified
    copies of the recorded choice files. Runs even when the episode raised."""
    if world_state is not None:
        (ep / "world_dev_state.json").write_text(json.dumps(
            {"label": "BENCH SIDE (never read by the robot or the loop): development world state, post-settle checks",
             "world_dev": scene["world_dev"], **world_state}, indent=1, default=str))
    if row is not None:
        (ep / "episode_row.json").write_text(json.dumps(row, indent=1, default=str))
    (ep / "ledger_demo.json").write_text(json.dumps({"summary": ledger.summary(), "entries": ledger.entries},
                                                    indent=1, default=str))
    copies = {}
    if recorded is not None:
        for name, sha in recorded["files"].items():
            shutil.copyfile(Path(recorded["choice_dir"]) / name, ep / name)
            copies[name] = {"sha256": file_sha(ep / name), "same_as_recorded": file_sha(ep / name) == sha}
    tree, n_files = tree_sha(root, "siteagent")
    designation = root.parent / "DESIGNATION.json"
    designated = json.loads(designation.read_text()).get("siteagent_tree_sha256") if designation.is_file() else None
    wd = None
    if scene["world_dev"]:
        wd = {"spec": scene["world_dev"],
              "applied_by": f"{world_dev_module_name(scene['world_dev'])}.applied (with its refusals)",
              "modules": {m: file_sha(root / "siteagent" / "world_dev" / m)
                          for m in ("occlusion2.py", "occlusion.py", "__init__.py")}}
    choice = client.choice or {}
    after = bytecode_count(root)
    table = seed_table_of(a.scene)
    showable, reasons = showable_status(sentinel=client.sentinel_served, refusal=client.refusal, err=err,
                                        fake_answer=recorded is not None and recorded.get("fake") is not False,
                                        same_as_designated=(designated is not None and designated == tree),
                                        bytecode_before=before, bytecode_after=after,
                                        copies_same=all(c["same_as_recorded"] for c in copies.values()),
                                        seeds_confirmed=table["confirmed"])
    identity = {
        "schema": "demoA-identity-1", "demo": DEMO_VERSION, "label": "simulation, simulated perception; demo episode",
        "started": started, "written": DC.now_text(), "episode_dir": str(ep), "exception": err,
        "showable": showable,
        "not_showable_because": None if showable else "; ".join(reasons),
        "scene": {"id": a.scene, **scene, "kitchen": a.kitchen, "noise_seed": a.noise,
                  "camera_seed": camera_seed(a.noise)},
        "seed_whitelist": {"kitchens": list(DEMO_KITCHENS), "range": list(DEMO_SEED_RANGE),
                           "pairs": [list(x) for x in DEMO_SEED_PAIRS], "checked_before_import": True,
                           "scene_seeds": {k: list(x) for k, x in DEMO_SCENE_SEEDS.items()},
                           "source": "seed table v1, confirmed in review on 2026-10-19 (review note not included)",
                           "table_v2": {"status": seed_table_v2_status(),
                                        "decision": dict(SEED_TABLE_V2_CONFIRMATION) if SEED_TABLE_V2_CONFIRMED
                                        else None,
                                        "pairs": [list(x) for x in DEMO_SEED_PAIRS_V2_PROPOSED],
                                        "scene_seeds": {k: list(x) for k, x in DEMO_SCENE_SEEDS_V2_PROPOSED.items()},
                                        "source": "seed table v2 (review note not included)"},
                           "this_scene": table},
        "situation": scene.get("situation"),
        "mode": a.mode,
        "robot": {"root": str(root), "siteagent_tree_sha256": tree, "siteagent_files": n_files,
                  "designated_tree_sha256": designated,
                  "same_as_designated": designated is not None and designated == tree,
                  "tf_guardrails_sha256": file_sha(root / "tf_guardrails.py")},
        "profile": {"information_profile": INFO_PROFILE, "record_scene_reading": RECORD_SCENE_READING,
                    "limits_version": lim.version, "criteria_version": crit["version"], "library": None,
                    "ledger": "MemoryLedger (zero usage from the demo client; never the development spending journal)"},
        "world_dev": wd,
        "registry": {"path": str(registry_path), "sha256": file_sha(registry_path)},
        "programs": {k: {"file": DC.PROGRAM_FILES[k], "sha256": DC.sha256_text(v)} for k, v in programs.items()},
        "fixed_texts": {"sentinel_sha256": DC.sha256_text(DC.SENTINEL_PROGRAM),
                        "fixed_help_sha256": DC.sha256_text(DC.FIXED_HELP_PROGRAM)},
        "code": {name: file_sha(p) for name, p in (
            ("demo_client.py", HERE / "demo_client.py"), ("choice_prompt.py", HERE / "choice_prompt.py"),
            ("run_demo.py", HERE / "run_demo.py"), ("send_choice.py", HERE / "send_choice.py"),
            ("export_episodes.py", TERRAIN / "export_episodes.py"))},
        "client": {"version": DC.CLIENT_VERSION, "model_id_reported": DC.DEMO_MODEL_ID,
                   "usage_source": DC.USAGE_SOURCE, "label_in_designated_code": "client non identifie (expected)",
                   "requests_answered": client.n, "refusal": client.refusal},
        "choice_prompt": {"version": CP.CHOICE_PROMPT_VERSION, "system_sha256": CP.SYSTEM_SHA256},
        "request_2_substitution": REQUEST_2_NOTE,
        "choice": {"who_chose": choice.get("who_chose"), "option": choice.get("option"),
                   "valid_options": (choice.get("menu") or {}).get("valid_options"),
                   "conditions": (choice.get("menu") or {}).get("conditions"),
                   "choice_request_sha256": choice.get("choice_request_sha256"),
                   "recorded_answer_sha256": choice.get("recorded_answer_sha256"),
                   "call_date": choice.get("call_date"), "computed_cost_usd": choice.get("computed_cost_usd"),
                   "cost_status": choice.get("cost_status"), "uncertain_usd": choice.get("uncertain_usd"),
                   "cost_source": "choice_call.json only (send_choice.py); the loop's MemoryLedger (ledger_demo.json) "
                                  "and calls.jsonl are never read for cost: after a client refusal they keep a "
                                  "reservation of the loop as uncertain, money never spent",
                   "fake_answer": choice.get("fake_answer"),
                   "request_code_root": (recorded or {}).get("code_root"),
                   "request_code_root_siteagent_tree_sha256": (recorded or {}).get("code_root_siteagent_tree_sha256")},
        "recorded_copies": copies,
        "import_hygiene": {"python_B": bool(sys.flags.dont_write_bytecode),
                           "PYTHONDONTWRITEBYTECODE": os.environ.get("PYTHONDONTWRITEBYTECODE"),
                           "cwd": str(Path.cwd()), "keys_in_environment": [k for k in KEY_VARS if os.environ.get(k)],
                           "frozen_root_bytecode_before": before, "frozen_root_bytecode_after": after},
        "stepwise_wrapper_3_7": None,
    }
    (ep / "DEMO_IDENTITY.json").write_text(json.dumps(identity, indent=1, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    sys.exit(main())
