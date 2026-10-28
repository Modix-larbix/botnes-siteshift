"""Checks around a public SiteShift run: platform, distributed files, Panda model, recorded choices, output folder.

The robot code (siteagent/ and tf_guardrails.py) is part of the frozen code snapshot that produced the three recorded
episodes, with the same code: only its comments and docstrings were rewritten for readers, and MANIFEST_SOURCE_MAP.json
proves it file by file (code_ast_sha256 below). Nothing here changes how the robot perceives, moves, chooses or is
graded. This module only decides whether a run may start, loads the recorded Nemotron choices, and describes what ran.

It replaces two things of the original demo launcher (demo/run_demo.py and demo/demo_client.py; their logic is
unchanged, and the few values edited for publication, such as private folder paths and the time zone of their time
stamps, are listed in MANIFEST_SOURCE_MAP.json):

1. Paths. The original launcher names folders of the machine it was written on. Here every path is relative to this
   package or given on the command line.
2. The replay check of a recorded choice. The original loader refuses a record unless the code folder sits at the
   absolute path recorded when the choice was prepared, holding the whole 126-file snapshot tree. A copy of a subset
   can never meet that. The relocation rule used instead:
   - every file listed in MANIFEST_SOURCE_MAP.json ("code_and_data") is present with its listed sha256, and no unlisted
     file sits in siteagent/, demo/ or recorded_choices/ (stale bytecode included);
   - every robot file is the snapshot's bytes, or a declared comment or docstring edit whose code is unchanged (same
     sha256 of its syntax tree without docstrings, code_ast_sha256 below);
   - the record names that snapshot tree and the sha256 of the snapshot's tf_guardrails.py.
   Every other check of the original loader is kept: the five record files, the request fingerprint recomputed and
   equal across the record, the answer text against its hash, the call receipt, the client's own log line, and a real
   Token Factory transport. The demo client itself still refuses a live request whose text differs from the recorded
   one (demo_client.DemoClient), so an old answer is never served on a new robot state.

Like the original, these checks guard against mistakes, not against someone who edits the code and its manifest.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_NAME = "MANIFEST_SOURCE_MAP.json"
MANIFEST_SCHEMA = "siteshift-public-sim-manifest-1"
RECORD_SCHEMA = "siteshift-public-choice-record-1"
FACTS_NAME = "recorded_choices/RECORDED_EPISODE_FACTS.json"
PANDA_LIST = "assets/PANDA_ASSETS.sha256"
PANDA_DIR = "menagerie/franka_emika_panda"
SANDBOX_EXEC = "/usr/bin/sandbox-exec"
RECORD_FILES = ("choice_request.json", "choice_request.sha256", "choice_response.json", "choice_call.json",
                "client_ops.jsonl")
CALL_AND_RESPONSE_KEYS = ("answer_sha256", "finish_reason", "model_id")
SYNTHETIC_MARK_FILE, SYNTHETIC_MARK_KEY = "SYNTHETIC_FIXTURE.txt", "synthetic_fixture"
# Output folder names the robot's world code refuses (siteagent/world_dev/occlusion.py, FORBIDDEN_OUT): checked here
# first so that the refusal comes before any simulation.
FORBIDDEN_OUT = re.compile(r"^(real|pilot|val)|^probe_format_")
# Folders scanned for files the manifest does not list.
SCANNED_DIRS = ("siteagent", "demo", "recorded_choices")
CODE_DIRS = ("siteagent", "demo")


@dataclass(frozen=True)
class Situation:
    name: str
    scene: str  # the scene id of the recorded episode
    mode: str  # replay: the recorded Nemotron answer is served; rule: only one option is valid, no model involved
    kitchen: str
    noise: int  # noise seed; the camera seed is noise + 1
    record: str | None  # folder under recorded_choices/
    title: str


SITUATIONS = {
    "in_place": Situation("in_place", "A4", "replay", "31000", 960040, "in_place_A4",
                          "the sponge lies where the site file puts it"),
    "moved": Situation("moved", "A1", "replay", "31000", 960000, "moved_A1",
                       "the sponge was moved to another drawer"),
    "masked": Situation("masked", "A5", "rule", "31000", 960050, None,
                        "a towel lies over the sponge in its drawer"),
}


class Refused(RuntimeError):
    """A run may not start; the message says why."""


# ---------------------------------------------------------------------------------------------------- hashing
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:]
    return tree


def code_ast_sha256(source: str) -> str:
    """sha256 of a Python file's syntax tree without comments, docstrings or positions (Python 3.12 ast.dump).

    Two files with the same value run the same code: only comments, blank lines, layout or docstrings differ. No
    file of the robot reads its own docstrings or source at run time, so such an edit cannot change an episode."""
    tree = _strip_docstrings(ast.parse(source))
    return sha256_bytes(ast.dump(tree, annotate_fields=True, include_attributes=False).encode("utf-8"))


# ---------------------------------------------------------------------------------------------------- platform
def platform_problem(platform: str | None = None, sandbox_exec: str = SANDBOX_EXEC) -> str | None:
    """Why this machine cannot run the simulator (None when it can).

    The robot's programs run in a separate process locked by macOS sandbox-exec (siteagent/agent/sandbox.py). On
    another system that isolation does not exist and the agent loop refuses to run unisolated programs. This launcher
    says so up front instead of switching the isolation off."""
    platform = sys.platform if platform is None else platform
    if platform != "darwin":
        return (f"this simulator needs macOS: robot programs run inside the macOS sandbox (sandbox-exec), which "
                f"this platform ({platform}) does not provide, and the robot refuses to run them without it. "
                f"Tested on macOS with Apple silicon. Other systems would need their own isolation and a new review.")
    if not Path(sandbox_exec).exists():
        return f"{sandbox_exec} is missing on this Mac: robot programs cannot be isolated, so nothing is run"
    return None


PYTHON = (3, 12)  # the Python version whose syntax trees MANIFEST_SOURCE_MAP.json records (code_ast_sha256)


def python_problem(version=None) -> str | None:
    """Why this Python cannot run the launcher (None when it can).

    The code check compares each robot file's syntax tree with the one recorded in MANIFEST_SOURCE_MAP.json, and the
    text of ast.dump changes between Python versions. On another version every robot file would look changed, so the
    launcher says which Python it needs instead."""
    found = tuple(sys.version_info[:2]) if version is None else tuple(version[:2])
    if found != PYTHON:
        return (f"this launcher needs Python {PYTHON[0]}.{PYTHON[1]} (found {'.'.join(map(str, found))}): its code "
                f"check compares Python {PYTHON[0]}.{PYTHON[1]} syntax trees. Create the environment with: "
                f"uv venv --python {PYTHON[0]}.{PYTHON[1]} .venv")
    return None


# ---------------------------------------------------------------------------------------------------- the manifest
def load_manifest(root=PACKAGE_ROOT) -> dict:
    path = Path(root) / MANIFEST_NAME
    if not path.is_file():
        raise Refused(f"{MANIFEST_NAME} is missing from {Path(root).name}/")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or doc.get("schema") != MANIFEST_SCHEMA:
        raise Refused(f"{MANIFEST_NAME}: unknown schema {doc.get('schema') if isinstance(doc, dict) else None!r}")
    return doc


def code_and_data_sha256(manifest: dict) -> str:
    """One fingerprint for the verified part of the distribution: sha256 of the canonical JSON of [path, sha256]."""
    rows = sorted([e["path"], e["sha256"]] for e in manifest["code_and_data"])
    return sha256_bytes(json.dumps(rows, separators=(",", ":")).encode("utf-8"))


def verify_distribution(root, manifest: dict) -> list[str]:
    """Problems of this copy against its manifest; no problem listed means it matches. See the module docstring."""
    root = Path(root)
    problems = []
    listed = set()
    for e in manifest.get("code_and_data", []):
        rel = e["path"]
        listed.add(rel)
        p = root / rel
        if not p.is_file():
            problems.append(f"missing file {rel}")
            continue
        data = p.read_bytes()
        if sha256_bytes(data) != e["sha256"]:
            problems.append(f"{rel} differs from the manifest")
            continue
        if e.get("origin") != "snapshot":
            continue
        change = e.get("change")
        if change == "identical":
            if e["sha256"] != e.get("source_sha256"):
                problems.append(f"{rel}: listed as identical to the snapshot but its source sha256 differs")
        elif change == "comments_or_docstrings_only":
            if not rel.endswith(".py"):
                problems.append(f"{rel}: only Python files can have a comment-only change")
            elif code_ast_sha256(data.decode("utf-8")) != e.get("source_code_ast_sha256"):
                problems.append(f"{rel}: its code differs from the snapshot (not only comments or docstrings)")
        else:
            problems.append(f"{rel}: robot file with change {change!r}; the recorded episodes need the snapshot's "
                            "code")
    for d in SCANNED_DIRS:
        base = root / d
        if base.is_dir():
            for f in sorted(base.rglob("*")):
                rel = f.relative_to(root).as_posix()
                if f.is_file() and rel not in listed and f.name != ".DS_Store":  # Finder's folder settings
                    problems.append(f"file not in the manifest: {rel}")
    for f in sorted(root.glob("*.py")):
        if f.name not in listed:
            problems.append(f"file not in the manifest: {f.name}")
    return problems


def verify_panda_assets(root=PACKAGE_ROOT) -> list[str]:
    """Problems of the Panda model files against assets/PANDA_ASSETS.sha256 (fetched by assets/fetch_panda.sh)."""
    root = Path(root)
    listing = root / PANDA_LIST
    if not listing.is_file():
        return [f"{PANDA_LIST} is missing"]
    target = root / PANDA_DIR
    if not target.is_dir():
        return [f"the Panda model is not in {PANDA_DIR}/: run assets/fetch_panda.sh first (it downloads the model "
                "from mujoco_menagerie at a pinned commit and checks every file)"]
    problems = []
    for line in listing.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        want, rel = line.split(None, 1)
        p = target / rel.strip()
        if not p.is_file():
            problems.append(f"Panda file missing: {PANDA_DIR}/{rel.strip()}")
        elif sha256_file(p) != want:
            problems.append(f"Panda file differs from the pinned one: {PANDA_DIR}/{rel.strip()}")
    return problems


def bytecode_count(root=PACKAGE_ROOT) -> dict:
    """Compiled Python files in the code folders. Python reuses such files, so a stale one could run instead of the
    source: a run starts only when there are none, and it is checked again at the end."""
    root = Path(root)
    dirs = [root / d for d in CODE_DIRS if (root / d).is_dir()]
    pyc = [p for d in dirs for p in d.rglob("*.pyc")] + list(root.glob("*.pyc"))
    caches = [p for d in dirs for p in d.rglob("__pycache__") if p.is_dir()]
    if (root / "__pycache__").is_dir():
        caches.append(root / "__pycache__")
    return {"pycache_dirs": len(caches), "pyc_files": len(pyc)}


# ---------------------------------------------------------------------------------------------------- the output
def output_problem(out) -> str | None:
    """Why --out cannot receive a run (None when it can). A run always writes into a folder that does not exist yet,
    so it can never overwrite an earlier run or a recorded episode."""
    out = Path(out).resolve()
    if out.exists():
        return f"{out.name} already exists: a run never writes into an existing folder, choose a new --out"
    if FORBIDDEN_OUT.match(out.name):
        return f"output folder name {out.name!r} is refused (names starting with real, pilot or val are reserved)"
    for d in ("siteagent", "demo", "recorded_choices", "assets", "menagerie"):
        base = (PACKAGE_ROOT / d).resolve()
        if out == base or base in out.parents:
            return f"--out must not be inside the package's {d}/ folder"
    for parent in out.parents:
        if parent.exists():
            if not parent.is_dir():
                return f"{parent} is not a folder"
            break
    return None


# ---------------------------------------------------------------------------------------------------- records
def load_public_record(choice_dir, root, manifest: dict, llm) -> dict:
    """A recorded Nemotron choice in the public record format, checked (ValueError when refused).

    llm is the robot's siteagent.agent.llm module (its request_fingerprint is the one the record was made with).
    Returns the same fields as demo_client.load_recorded, which the demo client uses."""
    d = Path(choice_dir)
    root = Path(root)
    missing = [n for n in (*RECORD_FILES, "PROVENANCE.json") if not (d / n).is_file()]
    if missing:
        raise ValueError(f"recorded choice incomplete in {d.name}: missing {missing}")
    prov = json.loads((d / "PROVENANCE.json").read_text(encoding="utf-8"))
    if not isinstance(prov, dict) or prov.get("schema") != RECORD_SCHEMA:
        raise ValueError(f"recorded choice {d.name}: PROVENANCE.json has an unknown schema")
    for name in RECORD_FILES:
        entry = (prov.get("files") or {}).get(name) or {}
        if sha256_file(d / name) != entry.get("public_sha256"):
            raise ValueError(f"recorded choice {d.name}: {name} differs from its PROVENANCE.json entry")
    req = json.loads((d / "choice_request.json").read_text(encoding="utf-8"))
    sha_file = ((d / "choice_request.sha256").read_text(encoding="utf-8").split() or [""])[0]
    resp = json.loads((d / "choice_response.json").read_text(encoding="utf-8"))
    call = json.loads((d / "choice_call.json").read_text(encoding="utf-8"))
    if not all(isinstance(x, dict) for x in (req, resp, call)):
        raise ValueError("recorded choice: a record is not a JSON object")
    if (d / SYNTHETIC_MARK_FILE).exists() or any(SYNTHETIC_MARK_KEY in x for x in (req, resp, call)):
        raise ValueError("recorded choice: a synthetic test record, never served as Nemotron's answer")
    # relocation rule (module docstring): the record's code tree is the snapshot this copy was made from
    snap = manifest.get("source_snapshot") or {}
    if req.get("code_root_siteagent_tree_sha256") != snap.get("siteagent_tree_sha256"):
        raise ValueError("recorded choice: it was prepared with another code tree than this package's snapshot")
    # tf_guardrails.py (prices and spending guard): the snapshot's file is the one the record names, and the copy here
    # is that file or a declared comment-only edit of it (checked with every other file by verify_distribution)
    guard = next((e for e in manifest.get("code_and_data", []) if e.get("path") == "tf_guardrails.py"), {})
    if guard.get("origin") != "snapshot" or guard.get("source_sha256") != req.get("code_root_tf_guardrails_sha256"):
        raise ValueError("recorded choice: tf_guardrails.py differs from the one the choice was prepared with")
    problems = verify_distribution(root, manifest)
    if problems:
        raise ValueError("recorded choice: the code does not match its manifest: " + "; ".join(problems[:5]))
    fp = llm.request_fingerprint(req.get("request"))
    shas = {"recomputed": fp, "choice_request.json": req.get("request_sha256"), "choice_request.sha256": sha_file,
            "choice_response.json": resp.get("request_sha256"), "choice_call.json": call.get("request_sha256")}
    if len(set(shas.values())) != 1:
        raise ValueError(f"recorded choice: request fingerprints differ {shas}")
    text = resp.get("answer_text")
    if not isinstance(text, str) or sha256_bytes(text.encode("utf-8")) != resp.get("answer_sha256"):
        raise ValueError("recorded choice: the answer text does not match its sha256")
    if call.get("status") != "answered":
        raise ValueError(f"recorded choice: the call did not answer (status {call.get('status')!r})")
    differ = [k for k in CALL_AND_RESPONSE_KEYS if call.get(k) is None or call.get(k) != resp.get(k)]
    if differ:
        raise ValueError(f"recorded choice: choice_response.json differs from the call receipt on {differ}")
    ops = []
    for line in (d / "client_ops.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                ops.append(json.loads(line))
            except ValueError:
                raise ValueError("recorded choice: a line of client_ops.jsonl is not JSON") from None
    if any(isinstance(o, dict) and SYNTHETIC_MARK_KEY in o for o in ops):
        raise ValueError("recorded choice: a synthetic test client log, never served as Nemotron's answer")
    answered = [o for o in ops
                if isinstance(o, dict) and o.get("status") == "answered" and o.get("request_sha256") == fp]
    if len(answered) != 1 or answered[0].get("answer_sha256") != resp["answer_sha256"]:
        raise ValueError("recorded choice: the client log has no single answered line with this request and answer")
    if call.get("transport") != "token_factory" or resp.get("usage_source") != "provider" \
            or call.get("nothing_sent") is not False:
        raise ValueError("recorded choice: not a real Token Factory answer")
    return {"choice_dir": str(d), "request_sha256": fp, "loop_request_sha256": req.get("loop_request_sha256"),
            "answer_text": text, "answer_sha256": resp["answer_sha256"], "finish_reason": resp.get("finish_reason"),
            "fake": False, "call_date": call.get("date"), "computed_cost_usd": call.get("computed_cost_usd"),
            "cost_status": call.get("cost_status"), "uncertain_usd": call.get("uncertain_usd"),
            "code_root": req.get("code_root"),
            "code_root_siteagent_tree_sha256": req.get("code_root_siteagent_tree_sha256"),
            "files": {n: sha256_file(d / n) for n in RECORD_FILES}}


# ---------------------------------------------------------------------------------------------------- episode facts
def _top_level_items(text: str) -> list[str]:
    """Split "a (x, y), b, c" on the commas outside parentheses."""
    items, depth, cur = [], 0, ""
    for ch in text:
        depth += (ch == "(") - (ch == ")")
        if ch == "," and depth == 0:
            items.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        items.append(cur.strip())
    return items


def inspection_map(scene_text: str | None) -> dict:
    """The zone labels of the robot's SCENE STATE text (its own readings, never the simulator's truth):
    seen (a sponge track there), not_found_in_checked_zone ("cleared"), still_to_check ("not cleared")."""
    out = {"search": None, "sponge_tracks": None, "seen": [], "not_found_in_checked_zone": [], "still_to_check": [],
           "outside_scope": None}
    if not scene_text:
        return out
    lines = scene_text.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("search for sponge: "):
            out["search"] = line[len("search for sponge: "):]
        elif line.startswith("sponge tracks: "):
            out["sponge_tracks"] = line[len("sponge tracks: "):]
        elif line.startswith("sponge seen "):
            out["seen"] += [x.split()[0] for x in _top_level_items(line.split(": ", 1)[1])]
        elif line.startswith("cleared "):
            out["not_found_in_checked_zone"] += [x.split()[0] for x in _top_level_items(line.split(": ", 1)[1])]
        elif line.startswith("not cleared "):
            for item in lines[i + 1:]:
                if not item.startswith("- "):
                    break
                out["still_to_check"].append(item[2:].split(": ", 1)[0])
        elif line.startswith("outside the scope"):
            out["outside_scope"] = line.split(": ", 1)[1]
    return out


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def extract_facts(ep) -> dict:
    """Facts of one episode folder that a reproduction must share with the recorded episode: the robot's request
    texts (their fingerprints), its zone labels, the choice, the gestures, its final claim and the evaluator's
    verdict. Wall-clock times are left out. Used both for the recorded episodes (RECORDED_EPISODE_FACTS.json) and
    for every new run, so the two are read the same way."""
    import demo_client as DC  # noqa: PLC0415 (the demo layer reads the robot's request texts)

    ep = Path(ep)
    calls = [json.loads(x) for x in (ep / "demo_calls.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()] \
        if (ep / "demo_calls.jsonl").is_file() else []
    requests = (_json(ep / "requests.json") or {}).get("requests") or []
    rec = _json(ep / "agent_record.json") or {}
    row = _json(ep / "episode_row.json") or {}
    oracle = _json(ep / "oracle_verdict.json")
    world = _json(ep / "world_dev_state.json")
    identity_choice = (_json(ep / "DEMO_IDENTITY.json") or {}).get("choice") or {}
    scene_texts = []
    for r in requests:
        try:
            scene_texts.append(DC.read_loop_request(r)["scene_text"])
        except ValueError:
            scene_texts.append(None)
    choice_rows = [c for c in calls if c.get("n") == 2]
    choice = choice_rows[0] if choice_rows else {}
    host = rec.get("host_records") or []
    picks = [h for h in host if h.get("name") == "pick"]
    scrubs = [h for h in host if h.get("name") == "scrub"]
    final = rec.get("final") or {}
    keep_row = ("claim", "outcome_class", "success", "criteria", "violations", "sponge_present", "sponge_region",
                "false_success_claim", "false_not_found", "false_failure_claim", "help_with_sponge_present",
                "unsupported_not_found", "stroke_path_m", "skill_calls", "sim_s", "drawers_left_open", "change_log",
                "task_setup_log", "criteria_version")
    return {
        "loop_requests": len(requests),
        "loop_request_sha256": [c.get("loop_request_sha256") for c in calls],
        "programs_served": [c.get("program_file") or c.get("origin") for c in calls],
        "scene_state_sha256": [None if t is None else sha256_bytes(t.encode("utf-8")) for t in scene_texts],
        "inspection_map_at_choice": inspection_map(scene_texts[1] if len(scene_texts) > 1 else None),
        "choice": {"who_chose": identity_choice.get("who_chose"), "option": identity_choice.get("option"),
                   "origin": choice.get("origin"), "program": choice.get("program_file"),
                   "valid_options": (choice.get("menu") or {}).get("valid_options"),
                   "menu_conditions": (choice.get("menu") or {}).get("conditions"),
                   "choice_request_sha256": choice.get("choice_request_sha256"),
                   "recorded_answer_sha256": choice.get("recorded_answer_sha256")},
        "pick": [{"ok": h.get("ok"), "reason": h.get("reason"), "kind": (h.get("args") or {}).get("kind"),
                  "pos_m": [round(float(x), 4) for x in (h.get("args") or {}).get("pos") or []]} for h in picks],
        "scrub": [{"ok": h.get("ok"), "reason": h.get("reason"),
                   "felt_s": (h.get("data") or {}).get("felt_s")} for h in scrubs],
        "final": {k: final.get(k) for k in ("status", "claim", "source", "help_kind", "final_help")},
        "attempt_claims": (row.get("agent") or {}).get("attempt_claims"),
        "contact_incidents": len(rec.get("contact_incidents") or []),
        "grip_events": len(rec.get("grip_events") or []),
        "scope_at_end": rec.get("scope_at_end"),
        "evaluator": {k: row.get(k) for k in keep_row},
        "evaluator_verdict_sha256": None if oracle is None else sha256_bytes(
            json.dumps(oracle.get("verdict"), sort_keys=True, separators=(",", ":")).encode("utf-8")),
        "world_dev_checks": None if world is None else world.get("checks"),
    }


def compare_facts(new, recorded, path: str = "") -> list[str]:
    """Paths where two fact trees differ (exact comparison: the same code, seeds and libraries give the same bytes)."""
    if isinstance(new, dict) and isinstance(recorded, dict):
        out = []
        for k in sorted(set(new) | set(recorded)):
            out += compare_facts(new.get(k), recorded.get(k), f"{path}.{k}" if path else k)
        return out
    if isinstance(new, list) and isinstance(recorded, list) and len(new) == len(recorded):
        out = []
        for i, (a, b) in enumerate(zip(new, recorded)):
            out += compare_facts(a, b, f"{path}[{i}]")
        return out
    return [] if new == recorded else [f"{path}: run {json.dumps(new)[:120]} / recorded {json.dumps(recorded)[:120]}"]
