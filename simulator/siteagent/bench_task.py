"""One task episode on the bench: build the task instance and its variant, run a program, score it.

The task is "find the sponge, pick it, and scrub the bowl on the landing tray with it". run_task_episode() builds the
kitchen (world.build), applies the task setup and the variant (world.task_setup, world.variants), runs a program
through the skill facade and scores the episode with the bench-side evaluator (siteagent.taskcheck).

When out_dir is given it writes:
  oracle_verdict.json        the evaluator's diagnosis with exact truth (bench evidence only)
  observable_feedback.json   robot-side record only (siteagent.feedback)
  trace.json                 robot-side trace: skill calls, perception trace, claim
A program is not given the simulator, the truth spec, the seed or the verdict as arguments, but an in-process program
is not isolated from them either; isolation is the agent's sandbox. The evaluator's evidence is bench-held
(taskcheck.BenchEvidence): skill records through a private hook of the facade, observations recorded at capture time
and a budget counted by the bench.
"""
from __future__ import annotations

import copy
import json
import time
import traceback
from pathlib import Path

import mujoco
import numpy as np

from . import taskcheck
from .feedback import observable_feedback, public_task
from .perception import Perception, SimulatedCamera
from .perception.estimate import DRAWER_REFERENCE_VERSION  # recorded in rows and oracle
from .perception.estimate import SCOPE_COVER_VERSION  # recorded in rows and oracle, next to drawer_reference
from .programs import fixed_reference, scripted_chain
from .programs.facade import SkillFacade
from .robot.skills import Robot
from .sim import Sim
from .sitemodel import SiteDossier, SiteModel
from .world import reference_site, sample_site
from .world.task_setup import task_site
from .world.variants import make_variant

PROGRAMS = {"A": fixed_reference.NAME, "S": scripted_chain.NAME}
VARIANT_FR = {"correct": "dossier juste", "moved": "éponge déplacée dans un autre tiroir, dossier inchangé",
              "absent": "éponge retirée, dossier inchangé",
              "moved_jittered": "éponge n'importe où dans un autre tiroir, dossier inchangé",
              "moved_surface": "éponge dans l'évier, dossier inchangé"}
PROGRAM_FR = {"A": "programme fixe A (positions de la cuisine de référence)",
              "S": "chaîne adaptative scriptée (recherche S3 + frotter)"}


def _font(size):
    from PIL import ImageFont
    for f in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


class TaskRecorder:
    """Bench-only GIF: robot mast camera | wrist camera | overview, the last skill call, and at
    the end the verifier's verdict (spectator information)."""

    PANEL = (320, 240)

    def __init__(self, sim, title: str, truth_text: str, every: int = 20):
        self.sim = sim
        self.r = mujoco.Renderer(sim.model, self.PANEL[1], self.PANEL[0])
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = [0.05, 0.10, sim.site.worktop_height]
        self.cam.distance, self.cam.elevation, self.cam.azimuth = 2.1, -62.0, 200.0
        self.every, self.n = every, 0
        self.frames = []
        self.title, self.truth_text = title, truth_text
        self.line = "début"
        self.verdict_line = ""
        self.font, self.font_s = _font(13), _font(11)

    def compose(self):
        from PIL import Image, ImageDraw
        W, H = self.PANEL
        img = Image.new("RGB", (3 * W, H + 66), (250, 250, 248))
        d = ImageDraw.Draw(img)
        titles = ["caméra du robot (mât)", "caméra du poignet", "vue d'ensemble (spectateur)"]
        for i, cam in enumerate(("robot_eye", "panda/wrist_cam", None)):
            self.r.update_scene(self.sim.data, camera=cam if cam else self.cam)
            img.paste(Image.fromarray(self.r.render()), (i * W, 0))
            d.rectangle([i * W, 0, i * W + 210, 15], fill=(0, 0, 0))
            d.text((i * W + 3, 1), titles[i], fill=(255, 255, 255), font=self.font_s)
        y = H + 3
        d.text((4, y), f"Simulation, perception simulée. {self.title}", fill=(0, 0, 0), font=self.font)
        d.text((4, y + 18), self.line, fill=(20, 20, 120), font=self.font)
        d.text((4, y + 38), self.verdict_line or f"Vérité cachée (spectateur) : {self.truth_text}",
               fill=(140, 30, 30), font=self.font_s)
        return img.quantize(colors=128, method=Image.Quantize.MEDIANCUT)

    def __call__(self, sim):
        self.n += 1
        if self.n % self.every == 0:
            self.frames.append(self.compose())

    def on_call(self, rec):
        res = "ok" if rec["ok"] else rec["reason"]
        self.line = f"Geste {rec['i']} ({rec['t_s']:.0f} s) : {rec['skill']} -> {res}"

    def finish(self, verdict_text: str, claim: str):
        self.line = f"Fin. Le programme annonce : {claim}"
        self.verdict_line = f"Vérificateur indépendant (vérité du simulateur) : {verdict_text}"
        for _ in range(20):
            self.frames.append(self.compose())

    def save(self, path: Path, fps: int = 8, max_frames: int = 420):
        frames = self.frames
        if len(frames) > max_frames:
            idx = np.linspace(0, len(frames) - 1, max_frames).astype(int)
            frames = [frames[i] for i in idx]
        path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
        self.r.close()
        return path


def truth_region(where: str) -> str:
    if where.startswith("drawers/"):
        return f"drawer_unit:{where.split('/')[1]}"
    return {"sink_side": "sink_basin", "landing": "landing_tray"}.get(where, "other")


def build_instance(kitchen: str, variant: str):
    """(truth spec, dossier spec, change log, setup log, target name), or None if not applicable."""
    base = reference_site() if kitchen == "reference" else sample_site(int(kitchen))
    try:
        spec, setup_log, target = task_site(base)
    except ValueError:
        return None
    v = make_variant(spec, variant)
    if v is None:
        return None
    truth, dspec, change_log = v
    return truth, dspec, change_log, setup_log, target


def run_task_episode(program: str, kitchen: str, variant: str, noise: int, out_dir: str | None = None,
                     gif_path: str | None = None, sensor_kw: dict | None = None, crit: dict | None = None,
                     program_fn=None, world_kw: dict | None = None, robot_kw: dict | None = None,
                     also_grade: dict | list | None = None) -> dict:
    """program: "A" or "S" (or any label with program_fn(facade, perception, dossier, hook) -> (claim, search|None)).
    world_kw: world build options (world.build, e.g. sponge_dish_friction); robot_kw: Robot options (e.g.
    scrub_gesture "v1" for comparison runs; held_contract "held-0" to reproduce earlier runs, the Robot default
    robot-held-1 otherwise, recorded as held_contract next to scrub_gesture in the row and oracle_verdict.json, for
    every condition: A, S and the agent, whose bench_agent passes no robot_kw; next to it drawer_reference, the
    shared perception's closed-reference rule perception.estimate DRAWER_REFERENCE_VERSION, declared for the
    condition comparisons, and in trace.json the perception's records-only reference_events as
    perception_reference_events, None without a perception; next to it scope_cover, the shared perception's occluder
    rule perception.estimate SCOPE_COVER_VERSION, declared for the condition comparisons, and in trace.json its
    records-only cover_events as perception_cover_events); also_grade: one criteria config or a list of them (e.g.
    [1.3, 1.2]) grading the SAME episode, for before/after tables (oracle_verdict_<version>.json each; the row's
    "also" is the first of them, "also_more" the others). Returns the summary row; writes the episode files into
    out_dir if given."""
    t_wall = time.perf_counter()
    crit = crit or taskcheck.load_criteria()
    sensor_kw = sensor_kw or {}
    row = {"program": program, "kitchen": kitchen, "variant": variant, "noise_seed": noise}
    inst = build_instance(kitchen, variant)
    if inst is None:
        return dict(row, applicable=False)
    truth, dspec, change_log, setup_log, target = inst
    dossier = SiteDossier.from_spec(dspec)
    sim = Sim(truth, noise_seed=noise, world_kw=world_kw)
    mon = taskcheck.TaskMonitor(sim, truth, target, crit)
    sim.monitors.append(mon)
    extra = [] if also_grade is None else (list(also_grade) if isinstance(also_grade, (list, tuple)) else [also_grade])
    mons2 = []
    for cx in extra:
        m2 = taskcheck.TaskMonitor(sim, truth, target, cx)
        sim.monitors.append(m2)
        mons2.append((cx, m2))
    rec = None
    if gif_path:
        sp = next((i for i in truth.items if i.kind == "sponge"), None)
        ttext = "éponge retirée" if sp is None else f"éponge dans {truth_region(sp.where).replace('drawer_unit:', 'tiroir ')}"
        rec = TaskRecorder(sim, f"{PROGRAM_FR.get(program, program)}. Variante : {VARIANT_FR.get(variant, variant)}",
                           ttext)
        sim.monitors.append(rec)
    robot = Robot(sim, worktop_z=dossier.worktop_height_m, **(robot_kw or {}))
    ev = taskcheck.BenchEvidence(sim, crit["budget"], scope_regions=dossier.candidate_regions("sponge"),
                                 on_call=None if rec is None else rec.on_call)
    facade = SkillFacade(robot, crit["budget"], bench=ev)
    sensor = per = model = None
    search = None
    exception = None
    claim: dict = {"claim": None}
    try:
        if program == "A" and program_fn is None:
            claim = fixed_reference.run(facade)
        else:
            model = SiteModel(dossier)
            sensor = SimulatedCamera(sim, noise_seed=noise + 1, **sensor_kw)
            per = Perception(facade, sensor, declared_drawers=dossier.n_drawers(), site_model=model)
            _observe = per.observe

            def observe(camera, _observe=_observe):  # bench-side record of every capture
                obs = _observe(camera)
                ev.observation(obs)
                return obs
            per.observe = observe
            fn = program_fn or (lambda f, p, d, h: scripted_chain.run(f, p, d, hook=h))
            claim, search = fn(facade, per, dossier, None)
    except Exception as e:  # noqa: BLE001 (the episode is classed execution_error)
        exception = f"{type(e).__name__}: {e}"
        claim = {"claim": "exception", "traceback": traceback.format_exc()[-1500:]}
    budget = ev.budget_state()
    v = taskcheck.verdict(mon, claim, ev, exception)
    v2s = []
    for cx, m2 in mons2:
        ev2 = copy.copy(ev)  # same bench records, budget counted against the other config
        ev2.max_sim_s, ev2.max_calls = float(cx["budget"]["max_sim_s"]), int(cx["budget"]["max_skill_calls"])
        v2s.append((cx, taskcheck.verdict(m2, claim, ev2, exception)))
    v2 = v2s[0][1] if v2s else None
    # The feedback is built from the bench-held copies of the robot-side records (a program cannot
    # erase its own failed calls from what the next attempt is shown).
    fb = observable_feedback(public_task(crit), PROGRAMS.get(program, program), claim, ev.records, budget,
                             holding_at_end=bool(robot.holding()), search_trace=None if search is None else search.trace,
                             site_model=model)
    if rec is not None:
        txt = v["outcome_class"] + (f" ({v['outcome_detail']})" if v["outcome_detail"] else "")
        rec.finish(txt, str(claim.get("claim")))
        rec.save(Path(gif_path))
    sp = next((i for i in truth.items if i.kind == "sponge"), None)
    if out_dir:
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        oracle = {"label": "simulation, perception simulee; ORACLE (bench only, never given to the robot)",
                  "program": program, "kitchen": kitchen, "variant": variant, "noise_seed": noise,
                  "seed": None if kitchen == "reference" else int(kitchen), "change_log": change_log,
                  "task_setup_log": setup_log, "target": target,
                  "sponge_truth_region": None if sp is None else truth_region(sp.where),
                  "truth_spec": json.loads(truth.to_json()),
                  "sensor_systematic_errors": None if sensor is None else sensor.systematic_errors(),
                  "sensor_kw": sensor_kw, "world_kw": world_kw or {}, "robot_kw": robot_kw or {},
                  "scrub_gesture": robot.scrub_gesture,  # the gesture used, robot_kw or default
                  "held_contract": robot.held_contract,  # the grip-state contract, robot_kw or default
                  "drawer_reference": DRAWER_REFERENCE_VERSION,  # the perception's rule
                  "scope_cover": SCOPE_COVER_VERSION,  # the perception's occluder rule
                  "verdict": v}
        (d / "oracle_verdict.json").write_text(json.dumps(oracle, indent=1, default=float))
        for cx, vx in v2s:
            (d / f"oracle_verdict_{cx['version']}.json").write_text(json.dumps(
                {"label": oracle["label"], "note": "same episode graded with another criteria version (comparison)",
                 "verdict": vx}, indent=1, default=float))
        (d / "observable_feedback.json").write_text(json.dumps(fb, indent=1, default=float))
        trace = {"label": "simulation, perception simulee; robot side", "program": program, "claim": claim,
                 "skill_calls": ev.records, "budget": budget, "dossier": dossier.to_dict(),
                 "search_trace": None if search is None else search.trace,
                 # drawer-reference-2's records-only events (perception.estimate)
                 "perception_reference_events": None if per is None else [dict(e) for e in getattr(
                     per, "reference_events", [])],
                 # scope-occluder-1's records-only events (perception.estimate)
                 "perception_cover_events": None if per is None else [dict(e) for e in getattr(
                     per, "cover_events", [])]}
        (d / "trace.json").write_text(json.dumps(trace, indent=1, default=float))
    if sensor is not None:
        sensor.close()
    sim.close()
    return dict(row, applicable=True, claim=claim.get("claim"), outcome_class=v["outcome_class"],
                outcome_detail=v["outcome_detail"], success=v["outcome_class"] == "success",
                false_success_claim=v["false_success_claim"], false_not_found=v.get("false_not_found", False),
                false_failure_claim=v["false_failure_claim"], help_with_sponge_present=v["help_with_sponge_present"],
                unsupported_not_found=v["unsupported_not_found"],
                stroke_path_m=None if v["best_bout"] is None else v["best_bout"].get("stroke_path_m"),
                hand_env_force_peak_outside_exempt_n=v["hand_env_force_peak_outside_exempt_n"],
                criteria_version=v["criteria_version"],
                sponge_present=sp is not None, sponge_region=None if sp is None else truth_region(sp.where),
                criteria={k: bool(x) for k, x in v["criteria"].items()},
                violations=[x["type"] for x in v["violations"]], best_bout=v["best_bout"],
                skill_calls=budget["skill_calls"], sim_s=budget["sim_s_used"],
                search_steps=claim.get("search_steps"), drawers_left_open=v["drawers_left_open"],
                change_log=change_log, task_setup_log=setup_log, exception=exception,
                world_kw=world_kw or {}, robot_kw=robot_kw or {}, scrub_gesture=robot.scrub_gesture,
                held_contract=robot.held_contract, drawer_reference=DRAWER_REFERENCE_VERSION,
                scope_cover=SCOPE_COVER_VERSION,
                also=None if v2 is None else {"criteria_version": v2["criteria_version"],
                                              "outcome_class": v2["outcome_class"], "outcome_detail": v2["outcome_detail"],
                                              "success": v2["outcome_class"] == "success",
                                              "false_success_claim": v2["false_success_claim"],
                                              "unsupported_not_found": v2["unsupported_not_found"],
                                              "violations": [x["type"] for x in v2["violations"]]},
                also_more=[{"criteria_version": vx["criteria_version"], "outcome_class": vx["outcome_class"],
                            "outcome_detail": vx["outcome_detail"], "success": vx["outcome_class"] == "success"}
                           for _cx, vx in v2s[1:]],
                wall_s=round(time.perf_counter() - t_wall, 1))
