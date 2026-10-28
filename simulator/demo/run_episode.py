"""Run one SiteShift situation in the simulator, offline, from this folder.

  env -u NEBIUS_API_KEY -u OPENAI_API_KEY PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B demo/run_episode.py \\
      --situation moved --out runs/moved_1

Situations (all in development kitchen 31000):
  in_place  the sponge lies where the site file puts it (scene A4, noise seed 960040, camera seed 960041)
  moved     the sponge was moved to another drawer (scene A1, noise seed 960000, camera seed 960001)
  masked    a towel lies over the sponge in its drawer (scene A5, noise seed 960050, camera seed 960051)

What runs: the robot inspects the kitchen with simulated perception, labels each zone from its own readings, then
either scrubs the bowl with the sponge or asks a person for help, and the evaluator grades the episode from the
simulator's records (which the robot never sees). The robot's two programs are fixed programs written by the team
(demo/programs/). Between them the code builds a menu of next steps from the robot's own state text. When both
options are valid (in_place, moved), the choice is the one Nemotron made when this episode was recorded: the recorded
answer is served only if the robot's request text today has exactly the recorded sha256, else the run stops with a
refusal. When only "ask for help" is valid (masked), a rule takes it. No model is called and no network is used.

Needs macOS and Python 3.12. Every run writes a NEW folder (--out must not exist). It is a local reproduction, never
one of the recorded episodes.
Exit codes: 0 the run finished and matches the recorded episode on every compared fact; 2 refused before simulating
(nothing written: another system or Python version, a file that differs from the manifest, no Panda model, an
existing --out, a model API key in the environment); 3 the demo client refused during the episode (for example a
different request text); 4 compiled Python files appeared in the code folders; 5 the run finished but differs from the
recorded episode (listed in REPRODUCTION.json); 1 an unexpected error.
"""
from __future__ import annotations

import argparse
import json
import os
import platform as platform_module
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True  # never leave compiled files next to the code, even when started without -B
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import public_adapter as PA  # noqa: E402

REGISTRY = HERE / "demo_seed_registry.json"
PROGRAMS = HERE / "programs"
STATUS = ("local reproduction: a new run of the published code, not one of the recorded episodes (those, the replay "
          "page and its scores are unchanged)")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--situation", required=True, choices=sorted(PA.SITUATIONS))
    ap.add_argument("--out", help="a new folder for this run (it must not exist yet)")
    ap.add_argument("--check-only", action="store_true",
                    help="check platform, files, Panda model and recorded choice, then stop without simulating")
    return ap


def refuse(problems) -> int:
    for p in problems:
        print(f"refused: {p}", file=sys.stderr)
    print("nothing was simulated and nothing was written", file=sys.stderr)
    return 2


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def versions() -> dict:
    import importlib.metadata  # noqa: PLC0415 (imported late: it loads the email and socket modules)

    out = {"python": platform_module.python_version(), "system": platform_module.platform(),
           "machine": platform_module.machine()}
    for name in ("mujoco", "numpy", "pillow"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


NETWORK_MODULES = {"socket", "ssl", "http.client", "urllib.request", "openai", "httpx", "requests"}


def loaded_modules(root: Path) -> dict:
    """The package's modules this process imported, with the sha256 of each file, and the third-party packages.
    Called before anything else of the final report, so that its own imports are not counted."""
    own = {}
    for name, mod in sorted(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if f and Path(f).resolve().is_relative_to(root):
            own[name] = {"file": Path(f).resolve().relative_to(root).as_posix(), "sha256": PA.sha256_file(f)}
    third = sorted({n.split(".")[0] for n in sys.modules} & {"mujoco", "numpy", "PIL", "openai"})
    # Python modules a network call would need; none is expected (the program sandbox denies network as well)
    network = sorted(set(sys.modules) & NETWORK_MODULES)
    return {"package_modules": own, "third_party_loaded": third, "network_modules_loaded": network}


def main(argv=None, platform: str | None = None, python_version: tuple | None = None) -> int:
    a = parser().parse_args(argv)
    sit = PA.SITUATIONS[a.situation]
    why = PA.platform_problem(platform) or PA.python_problem(python_version)
    if why:
        return refuse([why])
    import run_demo as RD  # noqa: PLC0415 (the original demo launcher; importing it loads no robot code)
    import demo_client as DC  # noqa: PLC0415

    root = PA.PACKAGE_ROOT
    problems = RD.seed_whitelist_problems(sit.kitchen, sit.noise, sit.scene)
    problems += RD.hygiene_problems(os.environ, Path.cwd(), sys.flags.dont_write_bytecode)
    out = None
    if not a.check_only:
        if not a.out:
            problems.append("--out is required (a new folder for this run)")
        else:
            out = Path(a.out).resolve()
            why = PA.output_problem(out)
            if why:
                problems.append(why)
    try:
        manifest = PA.load_manifest(root)
    except PA.Refused as e:
        return refuse(problems + [str(e)])
    problems += PA.verify_distribution(root, manifest)
    problems += PA.verify_panda_assets(root)
    before = PA.bytecode_count(root)
    if before["pycache_dirs"] or before["pyc_files"]:
        problems.append(f"compiled Python files in the code folders {before}: delete them, run with python -B")
    if problems:
        return refuse(problems)

    scene = RD.ALL_SCENES[sit.scene]
    DC.use_root(root)  # robot code imported from this folder only, bytecode writing off
    llm = DC.root_module("siteagent.agent.llm", root)
    recorded = None
    if sit.mode == "replay":
        try:
            recorded = PA.load_public_record(root / "recorded_choices" / sit.record, root, manifest, llm)
        except ValueError as e:
            return refuse([str(e)])
    if a.check_only:
        print(json.dumps({"check": "ok", "situation": sit.name, "files_verified": len(manifest["code_and_data"]),
                          "code_and_data_sha256": PA.code_and_data_sha256(manifest),
                          "recorded_choice": None if recorded is None else recorded["request_sha256"]}, indent=1))
        return 0

    BA = DC.root_module("siteagent.agent.bench_agent", root)
    LG = DC.root_module("siteagent.agent.ledger", root)
    LM = DC.root_module("siteagent.agent.limits", root)
    TC = DC.root_module("siteagent.taskcheck", root)
    O1 = DC.root_module("siteagent.world_dev.occlusion", root)
    O2 = DC.root_module("siteagent.world_dev.occlusion2", root) if scene["world_dev"] else None
    WD = O2.module_for(scene["world_dev"]) if scene["world_dev"] else None
    if WD is not None and WD.__name__ != RD.world_dev_module_name(scene["world_dev"]):
        return refuse([f"world {scene['world_dev']} resolved to {WD.__name__}"])
    lim, crit = LM.load_limits(), TC.load_criteria()
    if lim.version != RD.LIMITS_VERSION or crit["version"] != RD.CRITERIA_VERSION:
        return refuse([f"limits {lim.version} and criteria {crit['version']}, expected {RD.LIMITS_VERSION} and "
                       f"{RD.CRITERIA_VERSION}"])
    reg = {"ranges": json.loads(REGISTRY.read_text(encoding="utf-8"))["ranges"]}
    # the robot's own refusals (kitchen, registry, seed blocks, output folder name), with --out's parent as the base
    if scene["world_dev"]:
        why = WD.refusal(sit.kitchen, WD.parse(scene["world_dev"]), [sit.noise], reg, out, variant=scene["variant"],
                         s5_root=out.parent)
    else:
        why = O1.refusal(sit.kitchen, RD.PROXY_MODE, [sit.noise], reg, out, variant=None, s5_root=out.parent)
    if why:
        return refuse([why])
    try:
        programs = DC.load_programs(PROGRAMS)
    except (FileNotFoundError, ValueError) as e:
        return refuse([str(e)])

    out.mkdir(parents=True, exist_ok=False)
    client = DC.DemoClient(sit.mode, programs, root=root, recorded=recorded, calls_path=out / "demo_calls.jsonl")
    ledger = LG.MemoryLedger(label="local reproduction: no model call, no money spent; the demo client returns zero "
                                   "usage")
    ns = argparse.Namespace(scene=sit.scene, mode=sit.mode, kitchen=sit.kitchen, noise=sit.noise)
    started, t0 = utc_now(), time.monotonic()
    row, err, world_state = None, None, None
    print(f"running situation {sit.name} (scene {sit.scene}, kitchen {sit.kitchen}, noise seed {sit.noise}) "
          f"into {out.name}/ ...", file=sys.stderr)

    def run():
        return BA.run_agent_episode(sit.kitchen, scene["variant"], sit.noise, out, client, ledger, None, lim, None,
                                    crit, calls_log=out / "calls.jsonl",
                                    calls_meta={"demo": RD.DEMO_VERSION, "scene": sit.scene, "mode": sit.mode},
                                    info_profile=RD.INFO_PROFILE, record_scene_reading=RD.RECORD_SCENE_READING)
    try:
        if scene["world_dev"]:
            with WD.applied(scene["world_dev"], kitchen=sit.kitchen, registry=reg, noise_seeds=[sit.noise],
                            out_dir=out, s5_root=out.parent) as state:
                world_state = state
                row = run()
        else:
            row = run()
    except BaseException as e:  # kept in the identity files, then raised again
        err = f"{type(e).__name__}: {str(e)[:300]}"
        raise
    finally:
        # the original launcher's identity file (its "showable" field stays false here: that gate belonged to the
        # recorded episodes and needs the full snapshot), then this package's own reproduction record
        RD.finish(ns, scene, root, out, client, ledger, row, err, world_state, recorded, before, started, programs,
                  REGISTRY, lim, crit)
        code = write_reproduction(out, sit, manifest, client, recorded, before, started, time.monotonic() - t0, err)
    return code


def write_reproduction(out: Path, sit, manifest: dict, client, recorded, before: dict, started: str, wall_s: float,
                       err) -> int:
    root = PA.PACKAGE_ROOT
    modules = loaded_modules(root)
    after = PA.bytecode_count(root)
    facts = PA.extract_facts(out)
    recorded_facts = json.loads((root / PA.FACTS_NAME).read_text(encoding="utf-8"))["situations"][sit.name]
    differences = PA.compare_facts(facts, recorded_facts["facts"])
    choice = client.choice or {}
    if err is not None:
        code = 1
    elif client.refusal is not None:
        code = 3
    elif after != before:
        code = 4
    elif differences:
        code = 5
    else:
        code = 0
    doc = {
        "schema": "siteshift-local-reproduction-1",
        "status": STATUS,
        "situation": sit.name, "title": sit.title, "scene": sit.scene, "mode": sit.mode, "kitchen": sit.kitchen,
        "noise_seed": sit.noise, "camera_seed": sit.noise + 1,
        "started_utc": started, "finished_utc": utc_now(), "wall_s": round(wall_s, 1),
        "exit_code": code, "exception": err,
        "distribution": {"code_and_data_sha256": PA.code_and_data_sha256(manifest),
                         "files_verified": len(manifest["code_and_data"]),
                         "snapshot_tree_sha256": manifest["source_snapshot"]["siteagent_tree_sha256"]},
        "panda_assets": {"list_sha256": PA.sha256_file(root / PA.PANDA_LIST),
                         "upstream_commit": manifest.get("panda_assets", {}).get("commit")},
        "environment": versions(),
        "choice": {
            "who_chose": choice.get("who_chose"), "option": choice.get("option"),
            "model_called_during_this_run": False,
            "recorded_answer_used": recorded is not None and choice.get("recorded_answer_sha256") is not None,
            "choice_request_sha256_this_run": choice.get("choice_request_sha256"),
            "choice_request_sha256_recorded": None if recorded is None else recorded["request_sha256"],
            "same_request_text": None if recorded is None
            else choice.get("choice_request_sha256") == recorded["request_sha256"],
            "client_refusal": client.refusal},
        "bytecode_in_code_folders": {"before": before, "after": after},
        "comparison_with_recorded_episode": {"facts_file": PA.FACTS_NAME, "equal": not differences,
                                             "differences": differences},
        "result": facts,
        "modules": modules,
    }
    (out / "REPRODUCTION.json").write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    summary = {"situation": sit.name, "exit_code": code, "final_claim": facts["final"].get("claim"),
               "evaluator": {k: facts["evaluator"].get(k) for k in ("outcome_class", "success")},
               "who_chose": choice.get("who_chose"), "option": choice.get("option"),
               "same_request_text_as_recorded": doc["choice"]["same_request_text"],
               "matches_recorded_episode": not differences, "differences": differences[:10],
               "wall_s": doc["wall_s"], "out": out.name}
    print(json.dumps(summary, indent=1))
    return code


if __name__ == "__main__":
    sys.exit(main())
