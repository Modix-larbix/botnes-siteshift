#!/usr/bin/env python3
# Copyright 2026 Botnes (Moad Larabi)
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""SIMULATION ONLY. Botnes / SiteShift: bounded trial of the AssembleNIST pretrained policy (26/10/2026).

Runs inside ONE disposable Linux x86-64 Nebius CPU Job (job_entry.sh), never on a workstation. It imports the
third-party AssembleNIST code (Tao Sun, MIT, commit pinned below) from --upstream; this file is ours.

Question: a downloaded robot skill works in its own demo. What happens when conditions change, and which of the
robot's statements can be audited against the simulator? One task (R16), one reference condition, two predefined
variations, all fixed before any result (CONDITIONS, RULES). Every episode is kept and reported. The robot-side
report is retrospective: computed after the rollout from recorded signals, it never stops the controller.

Modes: smoke (tiny end-to-end check), main (the trial), decode (stdlib only: rebuild and verify the evidence package
from saved Job logs, see evidence_export.py; imports nothing third-party).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import io
import json
import math
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import traceback
import types
from collections import Counter
from pathlib import Path

LABEL = "SIMULATION"
SCHEMA = "botnes-assemblenist-trial-3"  # trial-3: Codex review of 26/10 evening (FIX_LOG_CODEX_20261026.md)
UPSTREAM_URL = "https://github.com/suniique/assembly-nist"
UPSTREAM_COMMIT = "eb5d15e004591d380c0d521da67e025ebee37703"
CKPT_REL = "checkpoints/e8192_it12500.pt"
CKPT_SHA256 = "eb2cd4e9aacda2d26d23c059570522559e9fcd812a64f0aef60daea83408581e"
TASK = "R16"
PART_NAME = "16 mm round peg"

# Fixed 26/10/2026 before any rollout (V1 replaced the same evening, still before any rollout: review A3,
# FIX_LOG.md). Line numbers refer to the pinned upstream commit.
CONDITIONS = (
    {"id": "REF", "short": "reference", "starts": "all_families", "friction_scale": 1.0, "seed_base": 10_000,
     "part_obs_offset_m": 0.0,
     "what": "author's evaluation starts for R16: uniform draw from the evaluation bank (5 start families), "
             "as evaluate.py lines 50-54; nominal physics; the policy gets the simulator's exact part pose"},
    {"id": "V1_OFFSET", "short": "part tracker off by 1 mm", "starts": "same_as_REF", "friction_scale": 1.0,
     "seed_base": 10_000, "part_obs_offset_m": 0.001, "offset_seed_base": 30_000,
     "what": "same starts and same policy noise stream as REF, nominal physics; the part position given to the policy "
             "(env.observe part_p, env.py line 183, its only part input with part_q: rl/obs.py lines 108-113) is offset "
             "by 1.0 mm in the horizontal plane, a fixed calibration error per episode whose direction is drawn from "
             "the episode's seed and written in plan.json before any rollout. R16 entry tolerance: radial clearance "
             "0.25 mm (hole r 8.25 mm, scene.py line 150; rod r 7.9975 mm, line 198) plus the rod's 0.5 mm chamfer "
             "(line 199). The simulator, its outcome predicate and the robot-side report are unchanged."},
    {"id": "V2_OILY", "short": "oily part (friction x0.25)", "starts": "same_as_REF", "friction_scale": 0.25,
     "seed_base": 10_000, "part_obs_offset_m": 0.0,
     "what": "same starts and same policy noise stream as REF; sliding friction x0.25 on the gripper pads "
             "(PhysicsConfig.finger_friction, scene.py line 52; pad-part contacts use it through priority 2, "
             "line 312) and on the part (part_friction, line 51; priority 1, line 369); passed to "
             "BatchedAssemblyEnv(pcfg=...), env.py lines 87-91"},
)
REMOVED_BEFORE_ANY_RUN = {
    "V1_LOOSE": "'loose' starts only. Not a change of conditions: 'loose' is one of the five start families the policy "
                "was trained and evaluated on (resets.py line 28; README line 56: 98.1 % at 12.5K iterations), so it "
                "was a slice of REF. Kept as REF's per-family breakdown (env_success_by_start_family). Review A3, "
                "26/10/2026."}

# Robot-side report rules, fixed before any rollout, never tuned on results (SI units).
RULES = {
    "force_push_n": 5.0,  # estimated push into the hole that counts as a contact (estimate, no force sensor)
    "stall_speed_mps": 0.01,  # tool speed below which the arm counts as stalled
    "descent_frac": 0.8,  # descent after first contact, as a fraction of the hole depth (9 mm)
    "persist_steps": 3,  # consecutive control steps (25 Hz) for the inserted rule
    "onset_reset_steps": 5,  # steps outside the hole column that forget a first contact
    "column_extra_m": 0.002,  # added to the largest grasp offset from the rod axis (resets.py sample_grasp)
    "near_board_m": 0.06,  # tool height above the hole bottom below which contact is considered
    "not_inserted_radial_m": 0.015,
    "not_inserted_height_margin_m": 0.010,
    "joint_jump_rad": 0.5,  # one-step joint jump read as a fault
    "hold_low_margin_m": 0.0035,  # gripper opening band around the expected rod grasp opening
    "hold_high_margin_m": 0.004,
}
ROBOT_KEYS = frozenset({"hold", "radial", "height", "speed", "push_down", "fault", "viol", "jump", "nonfinite",
                        "finger_mean", "grip_closed", "vratio"})
INPUTS_USED = ["joint positions and velocities", "gripper finger positions", "own gripper command",
               "own joint torque commands (force estimated through the kinematic Jacobian)",
               "joint-velocity limit monitor", "robot kinematic model", "board layout (hole position and axis)",
               "part catalogue (rod 16 mm x 50 mm, hole 9 mm deep)"]
NOT_LOOKED_AT = ["the part itself: no camera or part tracker is used for this report",
                 "whether the part is seated within 1 mm: inferred from tool motion and the estimated push only",
                 "the rest of the board"]
SAFETY_FAULT = "safety fault detected"
SCORED_CLAIMS = ("inserted", "not inserted")
# What the audit checks and what it does not (Codex review 26/10, point 3): never present a diagnostic as verified.
AUDIT_SCOPE = {
    "audited_predicates": {
        "main_claim": "'inserted' / 'not inserted' against the part keypoints in the simulator at the report instant "
                      "(the env's 1 mm threshold); 'still to check' is counted, never scored; 'safety fault detected' "
                      "is a consistency check, not scored",
        "part_held": "against geometric contacts of both fingers with the part (distance <= 0.5 mm) at the report "
                     "instant; this does not prove a stable grasp or a sufficient holding force"},
    "diagnostics_not_independently_audited": [
        "joint speeds within limits (the robot's own joint-velocity monitor)",
        "summed finger-joint opening and the estimated pad gap: joint readings, not a measured object width",
        "tool position: tool centre point from the kinematic model, not the part",
        "estimated axial push and descent: from commanded joint torques, not a force sensor",
        "the 'did not look at' list"]}
REPORTER_NOTE = ("retrospective: computed after the full rollout from recorded robot-side signals; it picks its own "
                 "report instant and never stops or changes the controller")
TIMING_NOTE = ("README line 108: 3200 episodes take about 1 h on 16 CPU cores (all 20 tasks, evaluate.py protocol), "
               "about 1.1 s per episode. tasks.py lines 74-78: a rod control step costs 1.0 against an all-20 mean "
               "of about 6.6 (gears and nuts on fine physics). Budget taken without that discount: 3 x 320 = 960 "
               "episodes, about 18 min if 16 vCPU = 16 cores, about 36 min if a vCPU counts as half a core.")
CAVEATS = [
    "Simulation only: MuJoCo, Franka Panda model, NIST board assets. No real robot, no transfer claim.",
    "The controller (AssembleNIST policy) receives the simulator's exact part pose every step (env.observe, "
    "env.py lines 177-185; rl/obs.py) as if a perfect tracker existed. The robot-side report does not use it.",
    "Outcome by the env's own predicate (env.py lines 225-227): keypoints within 1 mm, no simultaneous safety "
    "failure, 15 s limit. Episodes keep stepping to the horizon after their outcome is frozen (as evaluate.py line "
    "64). The robot-side report is computed afterwards from the recorded signals: " + REPORTER_NOTE + ".",
    "No force/torque or contact sensor exists in the model (panda.xml has none). Force is estimated from the "
    "commanded joint torques (PD, averaged over the control step, env.py line 146). Distances are those of the tool "
    "centre point (kinematic model), never of the part. The finger-joint opening is a joint reading, not an object "
    "width; the pad gap is an estimate from the model geometry (PAD_OFFSET, resets.py lines 29, 45-47).",
    "Robot-side thresholds were fixed before any rollout and not tuned on these results.",
    "V1_OFFSET changes only what the policy is told about the part (a 1 mm horizontal error of its exact pose); "
    "the simulated world and the robot-side report are those of REF.",
    "'safety fault detected' and 'joint speeds within limits' are not scored: the env's safety flag (vel_violation | "
    "nonfinite, env.py lines 217, 220, 225) reads the same joint-velocity monitor as the robot (env.py line 145), so a "
    "score would be true by construction. They are reported as consistency checks. Audited: the main claim when it "
    "is 'inserted' / 'not inserted' (part pose) and 'part held' (finger-part contacts). Everything else in a report "
    "is a diagnostic, not independently audited (AUDIT_SCOPE).",
]
SAFETY_CONSISTENCY_NOTE = ("consistency check, not scored: the env's safety flag is vel_violation | nonfinite (env.py "
                           "lines 217, 220, 225), the same joint-velocity monitor the robot reads; a fault on a joint "
                           "jump alone is outside that flag's definition (a mismatch of definitions, not a false claim)")
# Bank-time gate (review A4, fixed before any run): largest eval bank whose projected build fits its cap, else exit 6.
BANK_PROJECTION = {"safety_factor": 2.0, "overhead_s": 60.0, "eval_per_family_steps": (128, 64, 32)}


class BankTooSlow(Exception):
    pass


class Terminated(BaseException):  # not an Exception: no "except Exception" (a video, a timing read) may swallow a TERM
    pass


def _on_term(signum, frame):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)  # a second TERM cannot cut the trace nor the finalisation
    raise Terminated(f"signal {signum} received")


STATUS = {0: "complete", 3: "partial_time_guard", 4: "error", 5: "smoke_incomplete", 6: "bank_projection_refused",
          7: "terminated_by_signal"}
T_START = time.time()


def log(msg):
    print(f"[trial {(time.time() - T_START) / 60:6.2f} min] {msg}", flush=True)


class PhaseTimeout(Exception):
    pass


class alarm:
    def __init__(self, seconds):
        self.s = int(max(1, seconds))

    def __enter__(self):
        signal.signal(signal.SIGALRM, self._raise)
        signal.alarm(self.s)

    def __exit__(self, *exc):
        signal.alarm(0)
        return False

    @staticmethod
    def _raise(signum, frame):
        raise PhaseTimeout("phase time limit reached")


def clean(o):
    """JSON-safe copy: numpy scalars/arrays to Python, non-finite floats to None."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if hasattr(o, "tolist") and not isinstance(o, (str, bytes)):
        return clean(o.tolist())
    if isinstance(o, bool) or o is None or isinstance(o, (int, str)):
        return o
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, Path):
        return str(o)
    return str(o)


def rnd(x, nd=4):
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return round(min(1.0, 2 * tail), 6)


class Out:
    """Write-once outputs: each file is written once locally and, with a mirror (bucket), copied once."""

    def __init__(self, root, mirror=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.mirror = Path(mirror) if mirror else None
        self.written = []

    def bytes(self, rel, data):
        p = self.root / rel
        if p.exists():
            raise FileExistsError(str(p))
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".part")  # complete or absent, even if a TERM lands mid-write
        tmp.write_bytes(data)
        os.replace(tmp, p)
        self.adopt(rel)
        return p

    def json(self, rel, obj):
        return self.bytes(rel, (json.dumps(clean(obj), indent=1, allow_nan=False) + "\n").encode())

    def adopt(self, rel):
        self.written.append(rel)
        if self.mirror is None:
            return
        try:
            dst = self.mirror / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copyfile(self.root / rel, dst)
        except OSError as e:
            log(f"mirror copy failed for {rel}: {e.__class__.__name__}")


def load_evidence():
    """Our stdlib-only evidence module, injected next to this file (sys.path is not used: python -I)."""
    p = Path(__file__).resolve().with_name("evidence_export.py")
    spec = importlib.util.spec_from_file_location("botnes_evidence_export", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def send(ev, out, pkg, rels):
    """Progressive export (review T1): files already written leave through the log at once."""
    if ev is not None:
        ev.emit(pkg, [out.root / r for r in rels])


# --------------------------------------------------------------------------- upstream loading (Job only)
def load_upstream(upstream, data_dir):
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
    os.environ["ASSEMBLENIST_DATA"] = str(data_dir)
    up = str(Path(upstream).resolve())
    if up not in sys.path:
        sys.path.insert(0, up)
    import mujoco
    import numpy as np
    import torch
    from assembly_policy.evaluate import env_config
    from assembly_policy.rl.nets import ActorCritic
    from assembly_policy.rl.obs import PointTables
    from assembly_policy.sim.env import STATE_SPEC, BatchedAssemblyEnv
    from assembly_policy.sim.resets import FAMILIES, INSERT_DEPTH, PAD_OFFSET, PRISM, ResetGenerator
    from assembly_policy.sim.robot import quat_to_mat
    from assembly_policy.sim.scene import PhysicsConfig, build_xml
    from assembly_policy.sim.tasks import TASKS, resolve_tasks, scene_pose

    class TrialEnv(BatchedAssemblyEnv):
        """Upstream env unchanged; keeps the per-step work dict to read the commanded torques and the joint-velocity
        monitor (env.py lines 145-146), which a real arm also has."""

        def _work(self, item):
            self._last_out = item[3]
            return super()._work(item)

    return types.SimpleNamespace(
        np=np, torch=torch, mujoco=mujoco, env_config=env_config, ActorCritic=ActorCritic, PointTables=PointTables,
        STATE_SPEC=STATE_SPEC, TrialEnv=TrialEnv, FAMILIES=FAMILIES, INSERT_DEPTH=INSERT_DEPTH, PAD_OFFSET=PAD_OFFSET,
        PRISM=PRISM, ResetGenerator=ResetGenerator, quat_to_mat=quat_to_mat, PhysicsConfig=PhysicsConfig,
        build_xml=build_xml, TASKS=TASKS, resolve_tasks=resolve_tasks, scene_pose=scene_pose)


class SafePolicy:
    """Checkpoint actor as env -> sampled actions, like evaluate.TorchPolicy (lines 85-108), but loaded with
    torch.load(weights_only=True) after a SHA-256 check (upstream line 89 uses weights_only=False). A static opcode
    scan of the pickle (26/10) found three globals only: collections.OrderedDict, torch._utils._rebuild_tensor_v2,
    torch.FloatStorage, all in torch's weights-only allowlist. There is no fallback to full unpickling."""

    def __init__(self, U, ckpt, task):
        torch = U.torch
        d = torch.load(str(ckpt), map_location="cpu", weights_only=True)
        a = d["args"]
        self.torch, self.args, self.it = torch, a, d.get("it")
        train_tasks = U.resolve_tasks(a.get("tasks", "all20"))
        self.pts = U.PointTables(train_tasks, a["n_part"], a["n_fix"], a["n_grip"], device="cpu")
        self.remap = torch.as_tensor([train_tasks.index(task)])
        self.model = U.ActorCritic(self.pts.n_pts, max_std=a.get("max_std", 2.0), min_std=a.get("min_std", 1e-3))
        actor = {k: v for k, v in d["model"].items() if not k.startswith(("critic.", "priv_norm."))}
        res = self.model.load_state_dict(actor, strict=False)
        bad = [k for k in res.missing_keys if not k.startswith("critic.")] + list(res.unexpected_keys)
        if bad:
            raise RuntimeError(f"checkpoint and actor do not match: {bad[:5]}")
        self.model.eval()
        self.obs_offset = None  # V1_OFFSET: (N, 3) error added to the part position the policy is told, else None

    def __call__(self, env):
        torch = self.torch
        o = env.observe()

        def f(k, dt=torch.float32):
            return torch.as_tensor(o[k], dtype=dt)

        with torch.no_grad():
            part_p = f("part_p") if self.obs_offset is None else f("part_p") + self.obs_offset
            pts = self.pts(self.remap[f("task", torch.long)], f("tcp_p"), f("tcp_R"), part_p, f("part_q"),
                           f("finger"))
            return self.model.dist(pts, f("proprio")).sample().cpu().numpy()


def friction_scaled(p, scale):
    if scale == 1.0:
        return p

    def sc(s):
        v = s.split()
        v[0] = f"{float(v[0]) * scale:.6g}"
        return " ".join(v)

    return dataclasses.replace(p, finger_friction=sc(p.finger_friction), part_friction=sc(p.part_friction))


# --------------------------------------------------------------------------- bank and plan
def project_bank_s(timing, eval_pf, train_pf, workers):
    """Projected wall time (s) of build_banks.py for one task from the smoke run's per-family times (one state each):
    one ResetGenerator per (family, worker) job, train then eval split (build_banks.py lines 33-38, 45-51)."""
    cpu = 0.0
    for t in timing.values():
        for n in (train_pf, eval_pf):
            cpu += n * float(t["per_state_s"]) + min(n, workers) * float(t["setup_s"])
    return BANK_PROJECTION["safety_factor"] * cpu / max(1, workers) + BANK_PROJECTION["overhead_s"]


def choose_bank_size(timing, eval_pf, train_pf, workers, cap_s):
    """The largest eval bank (per family) whose projected build fits cap_s, from eval_pf down the fixed steps; None if
    none fits. Decided on timing only, never on outcomes, and written to plan.json before any rollout."""
    steps = sorted({e for e in BANK_PROJECTION["eval_per_family_steps"] if e <= eval_pf} | {eval_pf}, reverse=True)
    tried = []
    for e in steps:
        p = project_bank_s(timing, e, train_pf, workers)
        tried.append({"eval_per_family": e, "projected_s": round(p, 1)})
        if p <= cap_s:
            return e, tried
    return None, tried


def build_bank(U, a, ecfg, out, budget_s):
    np = U.np
    data = Path(a.data)
    if a.mode == "smoke":
        S, F, L, timing = [], [], [], {}
        for k, fam in enumerate(U.FAMILIES):
            t0 = time.time()
            g = U.ResetGenerator(a.task, U.PhysicsConfig(fine_families=ecfg.fine_families), seed=1000 + k)
            t1 = time.time()
            with alarm(min(240, budget_s)):
                s, _ = g.generate(fam, 1)
            timing[fam] = {"setup_s": round(t1 - t0, 3), "per_state_s": round(time.time() - t1, 3)}
            S.append(s)
            F.append(np.full(len(s), k))
            L.append(np.asarray(g.levels, float))
        return np.concatenate(S), np.concatenate(F), np.concatenate(L), {"source": "smoke, 1 state per family",
                                                                         "per_family_timing": timing}
    # review A4: project the build time from the smoke run before starting it
    try:
        timing = json.loads(Path(a.smoke_plan).read_text())["bank"]["per_family_timing"]
        assert set(timing) == set(U.FAMILIES)
    except Exception as e:
        raise BankTooSlow(f"no per-family timing from the smoke run ({e.__class__.__name__}): bank not projected")
    workers = max(1, min(a.bank_workers, os.cpu_count() or 1))
    chosen, tried = choose_bank_size(timing, a.eval_per_family, a.train_per_family, workers, budget_s)
    projection = {"label": LABEL, "rule": BANK_PROJECTION, "smoke_timing": timing, "workers": workers,
                  "cap_s": round(budget_s, 1), "tried": tried, "chosen_eval_per_family": chosen}
    out.json("bank/projection.json", projection)
    if chosen is None:
        raise BankTooSlow(f"projected bank build over its cap at every size: {tried} > {budget_s:.0f} s")
    if chosen != a.eval_per_family:
        log(f"eval bank reduced to {chosen} per family by the time projection ({tried})")
        a.eval_per_family = chosen
    bank_dir = data / "banks"
    cmd = [sys.executable, "-B", str(Path(a.upstream) / "scripts" / "build_banks.py"), "--tasks", a.task,
           "--per-family", str(a.train_per_family), "--eval-per-family", str(a.eval_per_family),
           "--workers", str(a.bank_workers), "--out", str(bank_dir)]
    t0 = time.time()
    logf = data / "build_banks_log.txt"
    data.mkdir(parents=True, exist_ok=True)
    with open(logf, "wb") as lf:
        p = subprocess.Popen(cmd, cwd=str(a.upstream), stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = p.wait(timeout=max(60, budget_s))
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            rc = "timeout"
        except BaseException:  # TERM or error: the bank builders do not outlive this process
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            raise
    tail = logf.read_bytes()[-4000:]
    out.bytes("bank/build_banks_log_tail.txt", tail)
    if rc != 0:
        raise RuntimeError(f"build_banks.py failed ({rc}); see bank/build_banks_log_tail.txt")
    z = np.load(bank_dir / f"{a.task}_eval.npz")  # allow_pickle stays False
    meta = {"source": "upstream scripts/build_banks.py (README line 105 recipe, this task only)",
            "command": " ".join(cmd[2:]), "wall_s": round(time.time() - t0, 1),
            "projection": {k: projection[k] for k in ("tried", "chosen_eval_per_family", "cap_s", "workers")}}
    for split in ("train", "eval"):
        jp = bank_dir / f"{a.task}_{split}.json"
        if jp.exists():
            meta[f"{split}_stats"] = json.loads(jp.read_text())
    meta["eval_npz_sha256"] = sha256_file(bank_dir / f"{a.task}_eval.npz")
    return z["states"], z["family"], z["level"], meta


def make_plan(np, nb, seeds, trials):
    """Starts drawn as evaluate.py lines 50-54 (REF); a variation 'same_as_REF' takes REF's exact starts. A tracker
    offset (V1_OFFSET) is a fixed horizontal error per episode, its direction drawn from the episode's seed."""
    plan = {}
    for c in CONDITIONS:
        if c["starts"] == "same_as_REF":
            plan[c["id"]] = {k: list(v) for k, v in plan["REF"].items() if k in ("bank_index", "seed", "trial")}
        elif c["starts"] == "all_families":
            idx, sd, tr = [], [], []
            for s in range(seeds):
                rng = np.random.default_rng(c["seed_base"] + s)
                idx += [int(x) for x in rng.integers(0, nb, trials)]
                sd += [s] * trials
                tr += list(range(trials))
            plan[c["id"]] = {"bank_index": idx, "seed": sd, "trial": tr}
        else:
            raise RuntimeError(f"unknown starts rule {c['starts']!r}")
        m = float(c.get("part_obs_offset_m") or 0.0)
        if m > 0:
            off = []
            for s in range(seeds):
                th = np.random.default_rng(c["offset_seed_base"] + s).uniform(0.0, 2.0 * math.pi, trials)
                off += [[round(m * math.cos(x), 9), round(m * math.sin(x), 9), 0.0] for x in th]
            plan[c["id"]]["part_obs_offset_m"] = off
    return plan


# --------------------------------------------------------------------------- rollout
def run_condition(U, task, policy, starts, ecfg, pcfg, abort_at, obs_offset=None):
    np, torch = U.np, U.torch
    N = len(starts)
    policy.obs_offset = None if obs_offset is None else torch.as_tensor(np.asarray(obs_offset), dtype=torch.float32)
    try:
        return _run_condition(U, task, policy, starts, ecfg, pcfg, abort_at, N)
    finally:
        policy.obs_offset = None


def _run_condition(U, task, policy, starts, ecfg, pcfg, abort_at, N):
    np, torch = U.np, U.torch
    env = U.TrialEnv([task], np.zeros(N, dtype=np.int64), ecfg, pcfg)
    env.set_states(np.arange(N), starts, np.zeros(N, dtype=np.int64))
    H = env.horizon
    R = {"S": np.empty((H + 1, N, env.nstate)), "grip": np.zeros((H + 1, N), bool), "tau": np.zeros((H + 1, N, 7)),
         "viol": np.zeros((H + 1, N), bool), "dasm": np.empty((H + 1, N)), "saf": np.zeros((H + 1, N), bool),
         "uns": np.zeros((H + 1, N), bool), "done_at": np.full(N, -1), "succ": np.zeros(N, bool),
         "sfail": np.zeros(N, bool), "tout": np.zeros(N, bool), "unst": np.zeros(N, bool)}
    R["S"][0] = env.state
    R["grip"][0] = env.grip_cmd < 0.02
    R["dasm"][0] = env.assembly_distance()
    torch.manual_seed(0)
    t0, aborted, steps = time.time(), False, 0
    for t in range(H):
        if t % 25 == 0 and time.time() > abort_at:
            aborted = True
            break
        _, done, info = env.step(policy(env))
        R["S"][t + 1] = env.state
        R["grip"][t + 1] = env.grip_cmd < 0.02
        R["tau"][t + 1] = env._last_out["tau"]
        R["viol"][t + 1] = env._last_out["viol"]
        R["dasm"][t + 1] = info["d_asm"]
        R["saf"][t + 1] = info["safety"]
        R["uns"][t + 1] = info["unstable"]
        new = (R["done_at"] < 0) & done
        R["done_at"][new] = t
        for k_out, k_in in (("succ", "success"), ("sfail", "safety"), ("tout", "timeout"), ("unst", "unstable")):
            R[k_out][new] = info[k_in][new]
        steps = t + 1
    env.close()
    R.update(H=H, steps=steps, aborted=aborted, wall_s=round(time.time() - t0, 1), env=env)
    return R


def geometry(U, env, task):
    np = U.np
    t = U.TASKS[task]
    hx, hy, L = U.PRISM[t.part_asset]
    tm = env.models[0]
    exp_open = hx - U.PAD_OFFSET + 0.0002  # resets.finger_q_for, lines 45-47
    return {"base": np.asarray(tm.goal_kp[0], float), "axis": np.asarray(tm.goal_axis, float),
            "depth": float(U.INSERT_DEPTH.get(t.fixture_asset, 0.009)), "radius": float(hx), "length": float(L),
            "expected_opening": exp_open, "pad_offset": float(U.PAD_OFFSET),
            "hold_lo": max(0.0008, exp_open - RULES["hold_low_margin_m"]),
            "hold_hi": exp_open + RULES["hold_high_margin_m"], "r_col": max(0.0, 0.012 - hx) + RULES["column_extra_m"],
            "h_out": L - 0.006 + RULES["not_inserted_height_margin_m"]}


def robot_channel(U, env, R, geo):
    """Robot-side signals only: joints, fingers, own commands, kinematic model, board layout."""
    np = U.np
    S, tau = R["S"], R["tau"]
    q, qd, fing = S[:, :, env.sq], S[:, :, env.sqd], S[:, :, env.sf]
    T1, N = q.shape[:2]
    finite = (np.isfinite(q).all(-1) & np.isfinite(qd).all(-1) & np.isfinite(tau).all(-1)
              & np.isfinite(fing).all(-1))

    def z(x):
        return np.where(finite[..., None], x, 0.0)

    qf, qdf, tf = z(q).reshape(-1, 7), z(qd).reshape(-1, 7), z(tau).reshape(-1, 7)
    fkm = env.models[0].fk
    fk = fkm.forward(qf)
    lin, _ = fkm.tcp_twist(fk, qdf)
    tcp = fk["tcp_p"]
    JT = np.concatenate([np.cross(fk["axes"], tcp[:, None, :] - fk["origins"]), fk["axes"]], -1)  # (M, 7, 6)
    J = np.swapaxes(JT, 1, 2)
    w = np.linalg.solve(J @ JT + 1e-6 * np.eye(6), J @ tf[:, :, None])[..., 0]  # wrench applied at the tool
    F = w[:, :3].reshape(T1, N, 3)
    tcp = tcp.reshape(T1, N, 3)
    rel = tcp - geo["base"]
    h = rel @ geo["axis"]
    radial = np.linalg.norm(rel - h[..., None] * geo["axis"], axis=-1)
    fm = np.where(finite, z(fing).mean(-1), 0.0)
    hold = R["grip"] & (fm >= geo["hold_lo"]) & (fm <= geo["hold_hi"])
    jump = np.zeros((T1, N), bool)
    jump[1:] = (np.abs(np.diff(z(q), axis=0)) > RULES["joint_jump_rad"]).any(-1)
    fault = R["viol"] | ~finite | jump
    fault[0] = False
    ch = {"hold": hold, "radial": radial, "height": h, "speed": np.linalg.norm(lin, axis=-1).reshape(T1, N),
          "push_down": -(F @ geo["axis"]), "fault": fault, "viol": R["viol"], "jump": jump, "nonfinite": ~finite,
          "finger_mean": fm, "grip_closed": R["grip"], "vratio": (np.abs(z(qd)) / env.vlim).max(-1)}
    return ch, {"tcp": tcp, "F": F}


def robot_report(ch, i, geo, dt):
    """Report from robot-side signals of episode i only. Retrospective: runs after the full rollout on the recorded
    signals, decides at their first decisive event, never stops the controller. Main claim and 'part held' are
    audited (AUDIT_SCOPE); every other line is a diagnostic."""
    assert set(ch) <= ROBOT_KEYS
    hold, rad, h, spd = ch["hold"][:, i], ch["radial"][:, i], ch["height"][:, i], ch["speed"][:, i]
    push, fault = ch["push_down"][:, i], ch["fault"][:, i]
    T = len(hold) - 1
    inc = hold & (rad <= geo["r_col"]) & (h >= -0.005) & (h <= RULES["near_board_m"])
    contact = inc & (push >= RULES["force_push_n"])
    onset, away, run, claim, inst = -1, 0, 0, None, T
    for t in range(1, T + 1):
        if fault[t]:
            claim, inst = SAFETY_FAULT, t
            break
        away = 0 if inc[t] else away + 1
        if away >= RULES["onset_reset_steps"]:
            onset = -1
        if onset < 0 and contact[t]:
            onset = t
        ok = (contact[t] and spd[t] <= RULES["stall_speed_mps"] and onset >= 0
              and h[onset] - h[t] >= RULES["descent_frac"] * geo["depth"])
        run = run + 1 if ok else 0
        if run >= RULES["persist_steps"]:
            claim, inst = "inserted", t
            break
    if claim is None:
        last = slice(max(0, T - 2), T + 1)
        far = bool(hold[last].all() and ((rad[last] > RULES["not_inserted_radial_m"]).all()
                                         or (h[last] > geo["h_out"]).all()))
        claim, inst = ("not inserted" if far else "still to check"), T
    ts = inst * dt
    desc = (h[onset] - h[inst]) * 1000 if onset >= 0 else None
    vmax = float(ch["vratio"][1:inst + 1, i].max()) if inst >= 1 else 0.0
    fm = float(ch["finger_mean"][inst, i])
    j_mm = 2 * fm * 1000  # summed finger-joint opening: a joint reading, not a width
    gap = 2 * (fm + geo["pad_offset"]) * 1000 if "pad_offset" in geo else None  # estimated pad gap (model geometry)
    rod = f"{geo['radius'] * 2000:.0f} mm rod"
    r_mm, h_mm = rad[inst] * 1000, h[inst] * 1000
    why = None
    if claim == SAFETY_FAULT:
        why = ("joint speed limit exceeded" if ch["viol"][inst, i] else
               "non-finite joint reading" if ch["nonfinite"][inst, i] else "joint position jump")
        main = (f"Safety fault detected at {ts:.2f} s: {why}. This retrospective reporter did not stop the controller "
                "(the rollout ran to its horizon).")
        diag = [f"joint-speed monitor: {why} at {ts:.2f} s (detection instant)"]
    else:
        main = {"inserted": f"Inserted (inferred, not seen): the {PART_NAME} is in its hole at {ts:.2f} s, from the tool "
                            "motion and the push estimated from commanded joint torques.",
                "not inserted": f"Not inserted (inferred, not seen): the {PART_NAME} is not in its hole at {ts:.2f} s, "
                                "from the tool position (distances and grasp: diagnostics, not audited).",
                "still to check": f"Cannot tell from its own signals whether the {PART_NAME} is inserted at {ts:.2f} s: "
                                  "still to check."}[claim]
        diag = [f"joint speeds within the limits until {ts:.2f} s (max {vmax:.0%} of limit), robot's own monitor"]
    if not ch["grip_closed"][inst, i]:
        held_txt = "Part not held: gripper commanded open."
        diag.append("gripper commanded open")
    else:
        held_txt = (f"Part held between both fingers (inferred from the closed gripper command and a finger-joint "
                    f"opening within the band of the {rod})." if hold[inst] else
                    f"Part not held: finger-joint opening outside the band of the {rod}.")
        diag.append(f"summed finger-joint opening {j_mm:.1f} mm" + (f", estimated pad gap {gap:.1f} mm from the model "
                    "geometry" if gap is not None else "") + " (joint readings, not a measured object width)")
    diag.append(f"tool centre point {r_mm:.1f} mm from the hole axis, {h_mm:.1f} mm above the hole bottom (kinematic "
                "model; the part is not tracked)")
    diag.append(f"estimated axial push {push[inst]:.1f} N from commanded joint torques (no force sensor); tool "
                f"{desc:.1f} mm below its height at the first estimated contact ({onset * dt:.2f} s)" if onset >= 0 else
                f"no estimated push of {RULES['force_push_n']:.0f} N or more with the tool inside the hole column")
    basis = {"onset_s": rnd(onset * dt, 3) if onset >= 0 else None, "descent_mm": rnd(desc, 2) if desc is not None else None,
             "push_estimated_n": rnd(push[inst], 2), "tool_radial_mm": rnd(r_mm, 2), "tool_height_mm": rnd(h_mm, 2),
             "tool_speed_mps": rnd(spd[inst], 4), "finger_joint_opening_mm": rnd(j_mm, 2),
             "estimated_pad_gap_mm": rnd(gap, 2) if gap is not None else None, "holding": bool(hold[inst]),
             "max_joint_speed_ratio": rnd(vmax, 3), "fault_reason": why}
    return {"claim": claim, "inst": int(inst), "report_s": rnd(ts, 3), "reporter": "retrospective",
            "main_claim": {"text": main, "audited": claim in SCORED_CLAIMS},
            "part_held_claim": {"text": held_txt, "value": bool(hold[inst]), "audited": True},
            "diagnostics_not_audited": diag, "did_not_look_at": NOT_LOOKED_AT, "inputs_used": INPUTS_USED,
            "basis": basis}


def holding_truth(U, model, data, idx, state):
    mj = U.mujoco
    mj.mj_setState(model, data, U.np.ascontiguousarray(state), U.STATE_SPEC)
    mj.mj_forward(model, data)
    touch = set()
    for c in data.contact[:data.ncon]:
        b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
        if idx.part_body in (b1, b2) and c.dist <= 5e-4:
            touch.add(b2 if b1 == idx.part_body else b1)
    return all(int(b) in touch for b in idx.finger_bodies)


def audit_episode(U, env, R, i, rep, geo, thr, mdl, dat):
    """Audits only the two predicates of AUDIT_SCOPE at the report instant: the main claim when 'inserted' / 'not
    inserted' (part keypoints, env's 1 mm threshold) and 'part held' (finger-part contacts). Safety faults are
    consistency checks, not scored (review A2); the report's diagnostics are not checked here."""
    np = U.np
    t = rep["inst"]
    d = float(R["dasm"][t, i])
    inserted = bool(d < thr)
    s = R["S"][t, i]
    pp, pq = s[env.sp[:3]], s[env.sp[3:7]]
    ax = U.quat_to_mat(pq)[:, 2]
    rel = pp - geo["base"]
    hb = float(rel @ geo["axis"])
    claim = rep["claim"]
    scored = claim in SCORED_CLAIMS
    ok = (inserted if claim == "inserted" else not inserted) if scored else None
    hold_t = holding_truth(U, mdl, dat, env.idx, s)
    held = bool(rep["basis"]["holding"])
    if claim == SAFETY_FAULT:
        why = rep["basis"].get("fault_reason")
        cons = {"fault_reason": why, "env_safety_flag_at_detection": bool(R["saf"][t, i]),
                "outside_env_flag_definition": why == "joint position jump"}
    else:
        cons = {"env_safety_flag_clear_until_report": not bool(R["saf"][1:t + 1, i].any())}
    truth = {"d_asm_mm": rnd(d * 1000, 3), "part_inserted": inserted, "part_base_height_mm": rnd(hb * 1000, 2),
             "part_radial_mm": rnd(np.linalg.norm(rel - hb * geo["axis"]) * 1000, 2),
             "part_tilt_deg": rnd(math.degrees(math.acos(max(-1.0, min(1.0, float(ax @ geo["axis"]))))), 2),
             "env_safety_at_instant": bool(R["saf"][t, i]), "part_held_by_both_fingers": bool(hold_t)}
    return {"audited": {"main_claim": {"claim": claim, "scored": scored, "true": ok},
                        "part_held": {"claim": held, "true": held == bool(hold_t)}},
            "consistency_not_scored": cons, "truth": truth}


def summarize(recs):
    n = len(recs)
    k = sum(r["env_success"] for r in recs)
    claims = Counter(r["robot_claim"] for r in recs)
    scored = [r for r in recs if r["robot_claim_true"] is not None]
    by_claim = {c: {"n": claims[c], "true": sum(1 for r in recs if r["robot_claim"] == c and r["robot_claim_true"] is True),
                    "false": sum(1 for r in recs if r["robot_claim"] == c and r["robot_claim_true"] is False)}
                for c in SCORED_CLAIMS}
    faults = [r for r in recs if r["robot_claim"] == SAFETY_FAULT]
    fams = sorted({r["start_family"] for r in recs})
    return {"label": LABEL, "n": n, "env_success": k, "env_success_rate": rnd(k / n, 4) if n else None,
            "env_success_wilson95": wilson(k, n), "env_safety_failures": sum(r["env_safety_failure"] for r in recs),
            "robot_claims": dict(claims),
            "audited": {"main_claim": {"scored": len(scored), "true": sum(1 for r in scored if r["robot_claim_true"]),
                                       "by_kind": by_claim, "still_to_check_not_scored": claims.get("still to check", 0),
                                       "still_to_check_part_inserted": sum(1 for r in recs if r["robot_claim"] == "still to check"
                                                                           and r["part_inserted_at_report"])},
                        "part_held": {"n": n, "true": sum(1 for r in recs if r["holding_claim_true"])}},
            "consistency_not_scored": {SAFETY_FAULT: {
                "n": len(faults), "env_flag_agrees": sum(1 for r in faults if r["safety_fault_env_flag_agrees"]),
                "joint_jump_only": sum(1 for r in faults if r["safety_fault_reason"] == "joint position jump")}},
            "diagnostics_not_audited": "reported in each robot report, never counted as verified (AUDIT_SCOPE)",
            "env_success_by_start_family": {f: [sum(r["env_success"] for r in recs if r["start_family"] == f),
                                                sum(1 for r in recs if r["start_family"] == f)] for f in fams}}


# --------------------------------------------------------------------------- video
def render_episode(U, task, states, path, size, lines_at, final_lines, fps=25, hold_s=1.0):
    mj, np = U.mujoco, U.np
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.load_default(size=13)
    except TypeError:
        font = ImageFont.load_default()
    W, Hh = size
    m = mj.MjModel.from_xml_string(U.build_xml(task, visual=True))
    m.vis.quality.shadowsize = 1024
    m.vis.quality.offsamples = 0
    d = mj.MjData(m)
    r = mj.Renderer(m, Hh, W)
    mj.mj_setState(m, d, np.ascontiguousarray(states[0]), U.STATE_SPEC)
    mj.mj_forward(m, d)
    part = d.xpos[mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "part")].copy()
    fix = U.scene_pose(U.TASKS[task].fixture_scene)[0]
    cam = mj.MjvCamera()  # framing as viz/success_videos.py lines 104-113
    cam.lookat[:] = 0.5 * (part + fix) + np.array([0.0, 0.0, 0.02])
    cam.distance = float(np.clip(1.3 * np.linalg.norm(part - fix) + 0.3, 0.35, 0.95))
    cam.azimuth, cam.elevation = 150.0, -35.0

    def band(img, lines, top):
        dr = ImageDraw.Draw(img)
        hgt = 6 + 16 * len(lines)
        y0 = 0 if top else img.size[1] - hgt
        dr.rectangle([0, y0, img.size[0], y0 + hgt], fill=(15, 17, 22))
        for j, s in enumerate(lines):
            dr.text((6, y0 + 3 + 16 * j), s, fill=(255, 255, 255), font=font)

    frames = []
    for k in range(len(states)):
        mj.mj_setState(m, d, np.ascontiguousarray(states[k]), U.STATE_SPEC)
        mj.mj_forward(m, d)
        r.update_scene(d, cam)
        img = Image.fromarray(r.render())
        band(img, lines_at(k), True)
        frames.append(np.asarray(img))
    img = Image.fromarray(frames[-1].copy())
    band(img, final_lines, False)
    frames += [np.asarray(img)] * int(hold_s * fps)
    imageio.mimwrite(str(path), frames, fps=fps, codec="libx264", quality=6, macro_block_size=8)
    try:
        r.close()
    except Exception:
        pass
    return len(frames)


# --------------------------------------------------------------------------- outputs (written once, sent at once)
def jsonl(rows):
    return "".join(json.dumps(clean(r)) + "\n" for r in rows).encode()


def plan_outputs(out, ev, mode, plan_doc):
    out.json("plan.json", plan_doc)  # written before any rollout
    send(ev, out, f"{mode}-plan", ["plan.json", "bank/projection.json", "bank/build_banks_log_tail.txt"])


def condition_outputs(out, ev, mode, c, recs, reps, auds, timing):
    """A finished condition's evidence, written and sent at once (review T1): a later error, time guard or TERM cannot
    take its episodes with it."""
    cid = c["id"]
    out.json(f"conditions/{cid}_results.json", {"label": LABEL, "condition": c, "summary": summarize(recs),
                                                "episodes": recs, "timing": timing})
    out.bytes(f"conditions/{cid}_robot_reports.jsonl", jsonl(reps))
    out.json(f"conditions/{cid}_audit.json", {"label": LABEL, "condition": cid, "audit_scope": AUDIT_SCOPE,
                                              "per_episode": auds})
    send(ev, out, f"{mode}-{cid}", [f"conditions/{cid}_{x}" for x in ("results.json", "robot_reports.jsonl", "audit.json")])


def final_outputs(a, out, ev, meta, recs_all, reps_all, aud_all, timings, skipped, vindex, job_t0, planned):
    """The combined files. The common finalisation (execute) sends them with everything not sent yet."""
    per_cond = {cid: summarize(recs_all[cid]) for cid in recs_all}
    paired = {}  # each variation on REF's exact starts, episode by episode
    for c in CONDITIONS:
        cid = c["id"]
        if c["starts"] != "same_as_REF" or "REF" not in recs_all or cid not in recs_all:
            continue
        pairs = list(zip(recs_all["REF"], recs_all[cid]))
        assert all(r["bank_index"] == v["bank_index"] for r, v in pairs)
        b = sum(1 for r, v in pairs if r["env_success"] and not v["env_success"])
        cc = sum(1 for r, v in pairs if v["env_success"] and not r["env_success"])
        paired[cid] = {"REF_success_variation_failure": b, "REF_failure_variation_success": cc,
                       "mcnemar_exact_p": mcnemar_exact(b, cc)}
    episodes = [r for cid in recs_all for r in recs_all[cid]]
    out.json("results.json", dict(meta, timings=timings, conditions_not_run=skipped, summary=per_cond,
                                  paired_vs_REF=paired, episodes=episodes, all_episodes_kept=True))
    out.bytes("robot_reports.jsonl", jsonl(r for cid in reps_all for r in reps_all[cid]))
    out.json("audit.json", {"label": LABEL, "audit_scope": AUDIT_SCOPE, "consistency_note": SAFETY_CONSISTENCY_NOTE,
                            "reporter": REPORTER_NOTE, "per_condition": per_cond,
                            "per_episode": [x for cid in aud_all for x in aud_all[cid]]})
    out.json("video_index.json", vindex)
    keys = ("n", "env_success", "env_success_wilson95", "env_safety_failures", "robot_claims", "audited",
            "consistency_not_scored")
    compact = {"label": LABEL, "mode": a.mode, "task": a.task, "episodes_planned": planned,
               "episodes_completed": len(episodes), "conditions_not_run": skipped, "paired_vs_REF": paired,
               "per_condition": {cid: {k: s[k] for k in keys} for cid, s in per_cond.items()},
               "audit_scope": AUDIT_SCOPE, "reporter": REPORTER_NOTE, "raw_signals": ev.raw_signals_note,
               "videos": {k: v.get("file") for k, v in vindex["videos"].items()}, "timings": timings,
               "elapsed_min": round((time.time() - job_t0) / 60, 1)}
    out.json("summary.json", compact)
    print("BOTNES_TRIAL_SUMMARY " + json.dumps(clean(compact), separators=(",", ":")), flush=True)
    return episodes


# --------------------------------------------------------------------------- main run
def run(a, ev):
    job_t0 = a.job_start_epoch or T_START
    hard = job_t0 + a.job_budget_min * 60
    rollout_abort = hard - 9 * 60
    out = Out(a.out, a.mirror)
    U = load_upstream(a.upstream, a.data)
    np, torch = U.np, U.torch
    torch.set_num_threads(a.nthread)
    smoke = a.mode == "smoke"
    if U.TASKS[a.task].family != "rod" or U.TASKS[a.task].part_asset not in U.PRISM:
        raise RuntimeError("this trial is written for one rod task (no SDF plugin)")
    ck = Path(a.upstream) / CKPT_REL
    ck_sha = sha256_file(ck)
    if ck_sha != CKPT_SHA256:
        raise RuntimeError("checkpoint SHA-256 differs from the pinned value")
    import mujoco
    versions = {"python": platform.python_version(), "mujoco": mujoco.__version__, "numpy": np.__version__,
                "torch": torch.__version__}
    policy = SafePolicy(U, ck, a.task)
    log(f"policy loaded with weights_only=True (iteration {policy.it})")
    ecfg = U.env_config(policy.args, a.nthread)
    if smoke:
        ecfg = dataclasses.replace(ecfg, horizon_s=1.2)
    pref = U.PhysicsConfig(fine_families=ecfg.fine_families, fine_sdf_initpoints=ecfg.fine_sdf_initpoints,
                           fine_sdf_iterations=ecfg.fine_sdf_iterations)  # as env.py lines 89-91
    pcfgs = {c["id"]: friction_scaled(pref, c["friction_scale"]) for c in CONDITIONS}
    states, fam, lev, bank_meta = build_bank(U, a, ecfg, out, budget_s=(rollout_abort - time.time()) * 0.5)
    log(f"bank ready: {len(fam)} start states, families {dict(Counter(int(x) for x in fam))}")
    seeds, trials = (1, 2) if smoke else (a.seeds, a.trials)
    plan = make_plan(np, len(fam), seeds, trials)
    cpu_model = ""
    try:
        cpu_model = next((ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name")), "")
    except OSError:
        pass
    meta = {"label": LABEL, "schema": SCHEMA, "mode": a.mode, "task": a.task, "part": PART_NAME,
            "upstream": {"url": UPSTREAM_URL, "commit": UPSTREAM_COMMIT, "checkpoint": CKPT_REL, "checkpoint_sha256": ck_sha,
                         "checkpoint_iteration": policy.it, "loading": "torch.load(weights_only=True)"},
            "conditions": CONDITIONS, "removed_before_any_run": REMOVED_BEFORE_ANY_RUN, "rules": RULES,
            "seeds": seeds, "trials_per_seed": trials,
            "episodes_per_condition": seeds * trials, "env_config": dataclasses.asdict(ecfg),
            "physics": {k: dataclasses.asdict(v) for k, v in pcfgs.items()},
            "bank": bank_meta, "timing_estimate": TIMING_NOTE, "caveats": CAVEATS, "audit_scope": AUDIT_SCOPE,
            "versions": versions, "host": {"platform": platform.platform(), "cpus": os.cpu_count(), "cpu_model": cpu_model},
            "policy_actions": "sampled (evaluate.py default), torch.manual_seed(0) before each condition",
            "per_family_reference": "REF's env_success_by_start_family (the 'loose' slice included)",
            "one_job_rule": "one Job; any re-run is reported together with the first, never instead of it"}
    plan_outputs(out, ev, a.mode, dict(meta, plan=plan, start_families=[int(x) for x in fam],
                                       start_levels=[rnd(x, 4) for x in lev]))
    thr = ecfg.success_threshold
    dt = ecfg.control_dt
    recs_all, reps_all, aud_all, kept, timings, skipped = {}, {}, {}, {}, {}, []
    est = None
    for c in CONDITIONS:
        cid = c["id"]
        if est is not None and time.time() + est > rollout_abort:
            skipped.append(cid)
            log(f"{cid} not started: time guard (estimate {est / 60:.1f} min)")
            continue
        P = plan[cid]
        idx = np.asarray(P["bank_index"])
        log(f"{cid}: {len(idx)} episodes, {c['short']}")
        R = run_condition(U, a.task, policy, states[idx], ecfg, pcfgs[cid], rollout_abort,
                          obs_offset=P.get("part_obs_offset_m"))
        timings[cid] = {"rollout_wall_s": R["wall_s"], "steps": R["steps"], "horizon": R["H"]}
        if R["aborted"]:
            skipped.append(cid)
            out.json(f"conditions/{cid}_ABORTED.json", {"label": LABEL, "condition": cid, "reason": "time guard during rollout",
                                                        "steps_done": R["steps"], "horizon": R["H"]})
            send(ev, out, f"{a.mode}-{cid}-aborted", [f"conditions/{cid}_ABORTED.json"])
            log(f"{cid} aborted by the time guard after {R['steps']} steps; no outcome counted")
            continue
        est = R["wall_s"] * 1.25 + 30
        env = R["env"]
        geo = geometry(U, env, a.task)
        t1 = time.time()
        ch, extra = robot_channel(U, env, R, geo)
        mdl = env.models[0].model
        dat = U.mujoco.MjData(mdl)
        recs, reps, auds = [], [], []
        for i in range(len(idx)):
            rep = robot_report(ch, i, geo, dt)  # retrospective: the rollout above is already finished
            au = audit_episode(U, env, R, i, rep, geo, thr, mdl, dat)
            am, ah = au["audited"]["main_claim"], au["audited"]["part_held"]
            da = int(R["done_at"][i])
            reason = ("success" if R["succ"][i] else "safety" if R["sfail"][i] else "timeout" if R["tout"][i]
                      else "not_finished")
            recs.append({"label": LABEL, "condition": cid, "episode": i, "seed": P["seed"][i], "trial": P["trial"][i],
                         "bank_index": int(idx[i]), "start_family": U.FAMILIES[int(fam[idx[i]])],
                         "start_level": rnd(lev[idx[i]], 4), "env_success": bool(R["succ"][i]), "env_end_reason": reason,
                         "env_end_s": rnd((da + 1) * dt, 3) if da >= 0 else None,
                         "env_safety_failure": bool(R["sfail"][i]), "env_unstable": bool(R["unst"][i]),
                         "robot_claim": rep["claim"], "robot_report_s": rep["report_s"],
                         "robot_claim_scored": am["scored"], "robot_claim_true": am["true"],
                         "part_inserted_at_report": au["truth"]["part_inserted"], "holding_claim_true": ah["true"],
                         "safety_fault_reason": rep["basis"]["fault_reason"],
                         "safety_fault_env_flag_agrees": au["consistency_not_scored"].get("env_safety_flag_at_detection"),
                         "part_obs_offset_m": P["part_obs_offset_m"][i] if "part_obs_offset_m" in P else None})
            reps.append(dict({"label": LABEL, "condition": cid, "episode": i}, **{k: v for k, v in rep.items() if k != "inst"}))
            auds.append(dict({"label": LABEL, "condition": cid, "episode": i, "report_s": rep["report_s"]}, **au))
        timings[cid]["report_audit_wall_s"] = round(time.time() - t1, 1)
        recs_all[cid], reps_all[cid], aud_all[cid] = recs, reps, auds
        kept[cid] = {"S": R["S"], "done_at": R["done_at"], "inst": [rp["report_s"] for rp in reps], "n": len(idx)}
        condition_outputs(out, ev, a.mode, c, recs, reps, auds, timings[cid])
        if a.save_signals:  # kept in the container (and a bucket copy), never sent through the logs
            buf = io.BytesIO()
            f32 = np.float32
            np.savez_compressed(buf, tcp=extra["tcp"].astype(f32), force_at_tool=extra["F"].astype(f32),
                                finger_mean=ch["finger_mean"].astype(f32), grip_closed=ch["grip_closed"],
                                joint_speed_ratio=ch["vratio"].astype(f32), joint_speed_violation=ch["viol"],
                                truth_d_asm=R["dasm"].astype(f32), truth_part_pos=R["S"][:, :, env.sp[:3]].astype(f32),
                                truth_part_quat=R["S"][:, :, env.sp[3:7]].astype(f32), truth_safety=R["saf"],
                                truth_unstable=R["uns"], dt=np.asarray(dt), label=np.asarray(LABEL))
            out.bytes(f"signals/{cid}_signals.npz", buf.getvalue())
        log(f"{cid}: env success {sum(r['env_success'] for r in recs)}/{len(recs)}, robot claims "
            f"{dict(Counter(r['robot_claim'] for r in recs))}, rollout {R['wall_s'] / 60:.1f} min")
        del R, ch, extra
    # videos: first env success and first env failure in the predefined episode order, never selected
    order = [(cid, i) for cid in recs_all for i in range(len(recs_all[cid]))]
    pick = {"first_success": next((x for x in order if recs_all[x[0]][x[1]]["env_success"]), None),
            "first_failure": next((x for x in order if not recs_all[x[0]][x[1]]["env_success"]), None)}
    vindex = {"label": LABEL, "rule": "first episode with env success and first without, in the order "
                                      + ", ".join(c["id"] for c in CONDITIONS) + " then episode index; no selection",
              "videos": {}}
    size = (320, 240) if smoke else tuple(int(v) for v in a.video_size.lower().split("x"))
    for name, x in pick.items():
        if x is None:
            vindex["videos"][name] = {"rendered": False, "reason": "no such episode"}
            continue
        if time.time() > hard - 5 * 60:
            vindex["videos"][name] = {"rendered": False, "reason": "time guard", "condition": x[0], "episode": x[1]}
            continue
        cid, i = x
        rec, rep, au = recs_all[cid][i], reps_all[cid][i], aud_all[cid][i]
        am = au["audited"]["main_claim"]
        K = kept[cid]
        end = max(int(K["done_at"][i]) + 1, int(round(rep["report_s"] / dt)))
        end = min(end, 12) if smoke else end
        cond = next(c for c in CONDITIONS if c["id"] == cid)
        top = [f"SIMULATION | AssembleNIST policy (Tao Sun, MIT) | {PART_NAME} (R16)",
               f"{cid}: {cond['short']} | episode {i + 1}/{K['n']}, seed {rec['seed']}, trial {rec['trial']}"]
        verdict = ({True: "TRUE", False: "FALSE"}[am["true"]] if am["scored"] else
                   "not scored (consistency check only)" if rep["claim"] == SAFETY_FAULT else
                   "not decidable (still to check)")
        env_line = (f"Simulator predicate: SUCCESS at {rec['env_end_s']:.2f} s" if rec["env_success"] else
                    f"Simulator predicate: FAILURE ({rec['env_end_reason']}) at {rec['env_end_s']} s")
        final = [env_line, f"Robot report (retrospective): {rep['claim']} at {rep['report_s']:.2f} s",
                 f"Audit of that claim vs part pose: {verdict}"]
        rel = f"videos/{name}_{cid}_ep{i:03d}.mp4"
        p = Path(a.out) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        t2 = time.time()
        try:
            nfr = render_episode(U, a.task, K["S"][:end + 1, i], p, size, lambda k: top + [f"t = {k * dt:5.2f} s"], final)
            out.adopt(rel)
            vindex["videos"][name] = {"rendered": True, "file": rel, "condition": cid, "episode": i, "frames": nfr,
                                      "seconds_shown": rnd(end * dt, 2), "render_wall_s": round(time.time() - t2, 1)}
        except Exception as e:  # a failed video never hides the results
            vindex["videos"][name] = {"rendered": False, "reason": f"{e.__class__.__name__}: {str(e)[:200]}",
                                      "condition": cid, "episode": i}
        log(f"video {name}: {vindex['videos'][name].get('rendered')}")
    episodes = final_outputs(a, out, ev, meta, recs_all, reps_all, aud_all, timings, skipped, vindex, job_t0,
                             len(CONDITIONS) * seeds * trials)
    if smoke:  # review A5: the results do not depend on the video, so a video problem does not block the main run
        ok = all((Path(a.out) / f).is_file() for f in ("plan.json", "results.json", "audit.json", "robot_reports.jsonl",
                                                       "summary.json"))
        ok = ok and len(episodes) == len(CONDITIONS) * seeds * trials
        video_ok = any(v.get("rendered") for v in vindex["videos"].values())
        if not video_ok:
            log("smoke: no video rendered (" + "; ".join(str(v.get("reason")) for v in vindex["videos"].values())
                + "); the results do not depend on it, the main run may go on")
        log(f"smoke check: {'pass' if ok else 'FAIL'} (video {'rendered' if video_ok else 'not rendered'})")
        return 0 if ok else 5
    return 3 if skipped else 0


def _trace(a, rel, obj):
    try:
        Out(a.out, a.mirror).json(rel, obj)
    except Exception:
        pass


def execute(a, EV):
    """One phase, then ALWAYS the common finalisation (review T1): what exists is listed in the phase manifest and
    sent, on success, any error, a refused projection, ABORTED conditions and TERM (exit 7)."""
    ev = EV.Exporter(a.evidence_root or Path(a.out).resolve().parent, a.run_id, a.evidence_ledger,
                     enabled=a.stdout_export, optional_cap_kb=a.export_cap_kb)
    prev = signal.signal(signal.SIGTERM, _on_term)
    rc, note, stamp = 4, None, int(time.time())
    try:
        try:
            rc = run(a, ev)
        finally:  # review B6: from the end of the run on, no TERM cuts the trace or the finalisation
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
    except BankTooSlow as e:  # review A4: stop at once, before the bank build
        rc, note = 6, f"bank projection: {e}"[:500]
        print("BOTNES_TRIAL_STOP " + json.dumps(note), flush=True)
        _trace(a, f"stopped_bank_projection_{stamp}.json", {"label": LABEL, "exit": 6, "reason": str(e)[:2000]})
    except BaseException as e:  # TERM (exit 7) or any error (exit 4): keep a trace
        rc = 7 if isinstance(e, Terminated) else 4
        note, tb = f"{e.__class__.__name__}: {str(e)[:500]}", traceback.format_exc()[-4000:]
        try:  # nothing here changes the exit code or skips the finalisation
            print(("BOTNES_TRIAL_TERMINATED " if rc == 7 else "BOTNES_TRIAL_ERROR ") + json.dumps(note), flush=True)
            print(tb, flush=True)
            _trace(a, f"{'terminated' if rc == 7 else 'error'}_{stamp}.json", {"label": LABEL, "exit": rc,
                                                                               "error": note, "traceback": tb})
        except BaseException:
            pass
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)  # the finalisation is not cut by a second TERM
        try:
            EV.finalize_phase(ev, a.out, a.mode, rc, STATUS.get(rc, "error"), note)
        except Exception as e:
            print("BOTNES_EVIDENCE_FINALISATION_ERROR " + json.dumps(f"{e.__class__.__name__}: {e}"[:300]), flush=True)
        if prev is not None:
            signal.signal(signal.SIGTERM, prev)
    return rc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("smoke", "main", "decode"), required=True)
    ap.add_argument("--upstream", help="pinned AssembleNIST checkout")
    ap.add_argument("--upstream-commit", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", help="local working directory for reset banks")
    ap.add_argument("--smoke-plan", help="main mode: the smoke run's plan.json (per-family bank timing, review A4)")
    ap.add_argument("--mirror", help="optional write-once copy target (mounted bucket)")
    ap.add_argument("--task", default=TASK)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--trials", type=int, default=32)
    ap.add_argument("--eval-per-family", type=int, default=128)
    ap.add_argument("--train-per-family", type=int, default=16)
    ap.add_argument("--bank-workers", type=int, default=16)
    ap.add_argument("--nthread", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--job-start-epoch", type=float, default=0.0)
    ap.add_argument("--job-budget-min", type=float, default=110.0)
    ap.add_argument("--video-size", default="480x360")
    ap.add_argument("--no-signals", dest="save_signals", action="store_false")
    ap.add_argument("--stdout-export", action="store_true", help="send the evidence through stdout (evidence_export.py)")
    ap.add_argument("--export-cap-kb", type=int, default=600, help="cap for the optional videos only; evidence is never dropped")
    ap.add_argument("--run-id", default="local")
    ap.add_argument("--evidence-root", help="job root of the evidence paths (default: parent of --out)")
    ap.add_argument("--evidence-ledger", help="ledger shared by the job's processes (default: in the evidence root)")
    ap.add_argument("--log", help="decode mode: saved Job log file")
    a = ap.parse_args()
    EV = load_evidence()
    if a.mode == "decode":
        if not a.log:
            print("decode needs --log (saved Job log file) and --out (a new folder)")
            return 2
        return EV.decode(a.log, a.out)
    if platform.system() != "Linux" or platform.machine() != "x86_64" or os.environ.get("BOTNES_DISPOSABLE_JOB") != "1":
        print("refused: smoke/main run only inside the disposable Linux x86-64 Job (BOTNES_DISPOSABLE_JOB=1)")
        return 2
    if a.upstream_commit != UPSTREAM_COMMIT or not a.upstream or not a.data:
        print("refused: --upstream, --data and the pinned --upstream-commit are required")
        return 2
    if a.task != TASK:
        print("refused: the predefined trial task is R16")
        return 2
    if a.mode == "main" and not a.smoke_plan:
        print("refused: the main run needs --smoke-plan (the smoke run's plan.json) for its bank-time projection")
        return 2
    rc = execute(a, EV)
    if rc == 7:  # TERM: leave right after the finalisation, without waiting for worker threads
        sys.stdout.flush()
        os._exit(rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
