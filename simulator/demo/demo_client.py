"""Demo client, passed to the agent loop in place of a model client. It calls no model and uses no network.

Every answer it gives the loop is a fixed program written by the team, a rule's choice, or an answer Nemotron gave
earlier to a separate, frozen choice request. It replaces nothing in the robot code: it goes through
run_agent_episode's public llm parameter, and demo_calls.jsonl says where each answer came from.

Requests of one episode (information profile s5-info-1.0, each carrying the robot's SCENE STATE text):
  1   the inspection program (programs/inspect_v2.bs);
  2   only when it is the robot's help review: the menu conditions are read from the SCENE STATE text of this request,
      never from the host (menu_conditions). scrub_bowl is valid only when the hand holds nothing, there are no
      contact incidents or grip events, the search found the sponge, a sponge track is not held and the review note
      says the robot perceives a sponge it has not picked. ask_help is always valid.
      - only ask_help valid: a rule chooses, no call (programs/help_v1.bs);
      - both valid, mode "replay": the recorded answer is served only when the recomputed choice request and the loop
        request have the recorded fingerprints; otherwise the client refuses and the episode stops (ClientNotReady).
        The answer must pass choice_prompt.parse_answer and its option's condition, else the robot asks for help;
      - both valid, mode "prepare": the choice request is written for a separate send and the loop gets a sentinel
        help; that episode is never shown;
  3+  a fixed help(kind="cannot_continue"), no call.
Usage returned to the loop is zero (usage_source "demoA_no_model_call") and the loop's ledger is in memory.

load_recorded() checks a recorded choice: the request fingerprint recomputed and equal across its files, the answer
against its hash, the call receipt, the client's own log line, and the code root the request was prepared with.
The public launcher replaces only the code-root part of that check (demo/public_adapter.py). These checks guard against
mistakes, not against someone who edits the files and their hashes together.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import choice_prompt as CP  # noqa: E402

# The development lab the original launcher ran in. It is not part of this package: the default folders below and in
# run_demo.py (S5_OUT, LIVE_REGISTRY) point under this placeholder and do not exist here. demo/run_episode.py, the
# public entry point, does not use them.
LAB = HERE.parent / "lab_not_included"
DESIG2_ROOT = LAB / "outputs" / "siteagent" / "s5" / "desig2_20261019" / "root"
PROGRAMS_DIR = HERE / "programs"
TIMEZONE = ZoneInfo("UTC")  # time stamps written by the demo client and the launcher

CLIENT_VERSION = "demoA-client-2"
CALLS_SCHEMA = "demoA-calls-1"
DEMO_MODEL_ID = "demoA/fixed-programs-no-model"
USAGE_SOURCE = "demoA_no_model_call"
ZERO_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
MODES = ("prepare", "replay", "rule")
# Version 2 of the inspect and scrub programs; the version 1 files stay unchanged
PROGRAM_FILES = {"inspect": "inspect_v2.bs", "scrub": "scrub_v2.bs", "help": "help_v1.bs"}
OPTION_PROGRAM = {"scrub_bowl": "scrub", "ask_help": "help"}
# Who chose and the demo_calls.jsonl origin of a recorded answer served in replay, by (fake answer, answer accepted). A
# fake-transport answer is never labeled Nemotron.
WHO_CHOSE = {(False, True): "nemotron", (False, False): "rule_after_refused_answer",
             (True, True): "fake_transport_dry_run", (True, False): "rule_after_refused_fake_answer"}
ORIGIN_CHOSEN = {False: "nemotron_recorded_choice", True: "fake_transport_dry_run_choice"}
ORIGIN_REFUSED = {False: "nemotron_answer_refused:", True: "fake_transport_dry_run_answer_refused:"}
# why texts kept to letters and spaces (agent.api.clean_text keeps them as written, 80 characters at most)
SENTINEL_PROGRAM = 'help("demo prepare mode choice request written episode never shown", kind="cannot_continue")'
FIXED_HELP_PROGRAM = 'help("demo client no further program after the chosen one", kind="cannot_continue")'
CHOICE_MAX_COMPLETION_TOKENS = 16384  # the agent's value (limits 0.8)
CHOICE_TEMPERATURE = 0.0

# Texts of the loop request (agent.prompt.build_messages, agent.host.render_scene_state, agent.loop.help_review_note)
TASK_HEAD = "TASK (public statement):\n"
REPORT_HEAD = "\n\nOBSERVABLE REPORT OF THE PREVIOUS RUN (robot side only):\n"
SCENE_HEAD = "\n\nSCENE STATE NOW ("
REMINDER_HEAD = "\n\nTASK REMINDER (the same public statement as at the top): "
HELP_REVIEW_CODE = "help_review"
NOTE_HEAD = 'help(kind="search_incomplete") not accepted yet ('
NOTE_PERCEIVED = "the robot perceives a sponge it has not picked"
NOTE_REGIONS = "region(s) not cleared"
HAND_RE = re.compile(r"^hand: holding (yes|no|unknown); contact incidents (\d+|unknown); grip events (\d+|unknown)$")
SEARCH_RE = re.compile(r"^search for sponge: (.+)$")
TRACKS_PREFIX = "sponge tracks: "
HELD_LINE_PREFIX = "sponge: held ("
HELD_EMPTY_LINE = "sponge track held but grip reads empty (grip event)"
TRACK_ITEM_RES = (("held", re.compile(r"^(\S+) held, picked from (.+)$")),
                  ("stale", re.compile(r"^(\S+) seen in (.+), robot acted since$")),
                  ("observed", re.compile(r"^(\S+) observed in (.+)$")),
                  ("other", re.compile(r"^(\S+) in (.+)$")))


class RootError(RuntimeError):
    """The frozen root cannot be used (missing, or another siteagent already imported)."""


def now_text() -> str:
    return datetime.now(TIMEZONE).isoformat(timespec="seconds")


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _inside(path, root: Path) -> bool:
    p = Path(path).resolve()
    return p == root or root in p.parents


def use_root(root=None) -> Path:
    """Put the frozen code root first on sys.path, bytecode writing off, and import its agent.llm.
    Refuses a root without the agent code, and a siteagent or tf_guardrails already imported from elsewhere."""
    root = Path(root or DESIG2_ROOT).resolve()
    if not (root / "siteagent" / "agent" / "llm.py").is_file() or not (root / "tf_guardrails.py").is_file():
        raise RootError(f"not a frozen root with siteagent/agent/llm.py and tf_guardrails.py: {root}")
    sys.dont_write_bytecode = True  # importing the frozen copy must never write __pycache__ into it
    for name in ("siteagent", "tf_guardrails"):
        mod = sys.modules.get(name)
        if mod is not None and not _inside(getattr(mod, "__file__", "") or "", root):
            raise RootError(f"{name} is already imported from another place ({getattr(mod, '__file__', None)})")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    llm = importlib.import_module("siteagent.agent.llm")
    if not _inside(llm.__file__, root):
        raise RootError(f"siteagent.agent.llm resolved outside the root: {llm.__file__}")
    return root


def root_module(name: str, root: Path):
    mod = importlib.import_module(name)
    if not _inside(mod.__file__, root):
        raise RootError(f"{name} resolved outside the root: {mod.__file__}")
    return mod


def tree_sha(root, rel: str = "siteagent") -> tuple[str, int]:
    """Tree rule (the designated tree of DESIGNATION.json, run_demo.tree_sha): sha256 of the canonical JSON
    [[relpath, sha256]] over the files under root/rel, without __pycache__; and the number of files."""
    root = Path(root)
    rows = [[str(f.relative_to(root)), hashlib.sha256(f.read_bytes()).hexdigest()]
            for f in sorted((root / rel).rglob("*")) if f.is_file() and "__pycache__" not in f.parts]
    blob = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest(), len(rows)


def code_root_of(root) -> dict:
    """What a prepared choice request records of its code root: the folder and its siteagent tree; and the sha256 of
    its tf_guardrails.py, the send's price and journal guard."""
    root = Path(root).resolve()
    tree, n = tree_sha(root)
    return {"code_root": str(root), "code_root_siteagent_tree_sha256": tree, "code_root_siteagent_files": n,
            "code_root_tf_guardrails_sha256": sha256_file(root / "tf_guardrails.py")}


def code_root_problem(doc, root) -> str | None:
    """Why a prepared choice request (choice_request.json) may not be sent or replayed with this code root; None when
    it may. The request names the root it was prepared with, and the root given to send_choice.py (--root) or to the
    replay must be that same folder, holding that same siteagent tree (the same tree rule). A request with no
    code root (prepared by demoA-client-1) is refused: prepare it again. The root's tf_guardrails.py must also be the
    one the request was prepared with (its sha256); a request that does not record it is refused:
    prepare it again. A check against mistakes, not against tampering: these keys are outside the request's
    fingerprint."""
    want = doc.get("code_root") if isinstance(doc, dict) else None
    want_tree = doc.get("code_root_siteagent_tree_sha256") if isinstance(doc, dict) else None
    want_guard = doc.get("code_root_tf_guardrails_sha256") if isinstance(doc, dict) else None
    if not isinstance(want, str) or not want or not isinstance(want_tree, str) or not want_tree:
        return "the prepared request names no code root (an earlier client prepared it): prepare it again"
    if not isinstance(want_guard, str) or not want_guard:
        return ("the prepared request does not record the sha256 of its root's tf_guardrails.py (an earlier "
                "client prepared it): prepare it again")
    have = Path(root).resolve()
    if Path(want).resolve() != have:
        return f"the prepared request's code root {want} differs from the root given {have}"
    tree, _ = tree_sha(have)
    if tree != want_tree:
        return (f"the siteagent tree under {have} ({tree[:16]}) differs from the one the request was prepared with "
                f"({want_tree[:16]})")
    guard_file = have / "tf_guardrails.py"
    guard = sha256_file(guard_file) if guard_file.is_file() else None
    if guard != want_guard:
        return (f"tf_guardrails.py under {have} ({str(guard)[:16]}) differs from the one the request was prepared with "
                f"({want_guard[:16]})")
    return None


# ------------------------------------------------------------------------------------------------ reading a request
def loop_user_text(request) -> str:
    """The user message of a loop request (exactly [system, user], text only)."""
    msgs = request.get("messages") if isinstance(request, dict) else None
    if not isinstance(msgs, list) or len(msgs) != 2 or not all(isinstance(m, dict) for m in msgs):
        raise ValueError("a loop request has exactly two messages")
    if [m.get("role") for m in msgs] != ["system", "user"] or not all(isinstance(m.get("content"), str) for m in msgs):
        raise ValueError("a loop request has a system and a user message, both text")
    return msgs[1]["content"]


def review_of(report) -> dict:
    """Whether the previous run report is the robot's help review, and which trigger its note names."""
    rep = report if isinstance(report, dict) else {}
    err = rep.get("error") if isinstance(rep.get("error"), dict) else {}
    msg = err.get("message") if isinstance(err.get("message"), str) else ""
    is_review = (rep.get("status") == HELP_REVIEW_CODE and err.get("kind") == "harness_rule"
                 and err.get("code") == HELP_REVIEW_CODE and msg.startswith(NOTE_HEAD))
    trigger = None
    if is_review:
        trigger = ("perceived_sponge" if NOTE_PERCEIVED in msg else "regions_not_cleared" if NOTE_REGIONS in msg
                   else "unknown")
    return {"is_review": is_review, "trigger": trigger, "report_status": rep.get("status"),
            "report_error_code": err.get("code"), "note": msg[:300] if is_review else None}


def read_loop_request(request) -> dict:
    """{task, task_text, report, review, scene_text} of a loop request, read from its user text only."""
    u = loop_user_text(request)
    task = None
    if u.startswith(TASK_HEAD):
        try:
            task = json.loads(u[len(TASK_HEAD):].split("\n", 1)[0])
        except ValueError:
            task = None
    task_text = task.get("text") if isinstance(task, dict) and isinstance(task.get("text"), str) else None
    report = None
    i = u.rfind(REPORT_HEAD)
    if i >= 0:
        try:
            report = json.loads(u[i + len(REPORT_HEAD):].split("\n", 1)[0])
        except ValueError:
            report = None
    scene_text = None
    r = u.rfind(REMINDER_HEAD)
    s = u.rfind(SCENE_HEAD, 0, r) if r >= 0 else -1
    if s >= 0:
        scene_text = u[s + 2:r]
    return {"task": task, "task_text": task_text, "has_report": i >= 0, "report": report,
            "review": review_of(report) if i >= 0 else review_of(None), "scene_text": scene_text}


def parse_scene_state(text) -> dict:
    """The lines of a SCENE STATE text that the menu reads (agent.host.render_scene_state, scene-state-1)."""
    out = {"header_ok": False, "hand": None, "hand_lines": 0, "search_outcome": None, "search_lines": 0,
           "tracks": [], "tracks_lines": 0, "held_line": False, "held_empty_line": False, "problems": []}
    if not isinstance(text, str) or not text.startswith(CP.SCENE_PREFIX + " (robot side, read when this request was "
                                                                          "built, "):
        out["problems"].append("no SCENE STATE header")
        return out
    out["header_ok"] = True
    for line in text.split("\n"):
        m = HAND_RE.match(line)
        if m:
            out["hand_lines"] += 1
            out["hand"] = {"holding": m.group(1), "contact_incidents": m.group(2), "grip_events": m.group(3)}
            continue
        m = SEARCH_RE.match(line)
        if m:
            out["search_lines"] += 1
            out["search_outcome"] = m.group(1)
            continue
        if line.startswith(TRACKS_PREFIX):
            out["tracks_lines"] += 1
            for item in line[len(TRACKS_PREFIX):].split(" | "):
                for status, rx in TRACK_ITEM_RES:
                    mm = rx.match(item)
                    if mm:
                        out["tracks"].append({"track_id": mm.group(1), "status": status, "region": mm.group(2)})
                        break
                else:
                    out["problems"].append("unreadable track item")
            continue
        if line.startswith(HELD_LINE_PREFIX):
            out["held_line"] = True
        elif line == HELD_EMPTY_LINE:
            out["held_empty_line"] = True
    for key, n in (("hand line", out["hand_lines"]), ("search line", out["search_lines"])):
        if n != 1:
            out["problems"].append(f"{n} {key}s (exactly 1 expected)")
    if out["tracks_lines"] > 1:
        out["problems"].append(f"{out['tracks_lines']} sponge tracks lines")
    return out


def menu_conditions(scene: dict, review: dict) -> dict:
    """The menu of request 2, from the robot's own texts only: the conditions, the valid options, who chooses."""
    hand = scene.get("hand") or {}
    cond = {
        "scene_state_readable": bool(scene.get("header_ok")) and not scene.get("problems"),
        "hand_holding_no": hand.get("holding") == "no",
        "contact_incidents_0": hand.get("contact_incidents") == "0",
        "grip_events_0": hand.get("grip_events") == "0",
        "search_found": scene.get("search_outcome") == "found",
        "sponge_track_not_held": any(t.get("status") != "held" for t in scene.get("tracks") or []),
        "no_held_line": not scene.get("held_line") and not scene.get("held_empty_line"),
        "review_note_perceived_sponge": review.get("trigger") == "perceived_sponge",
    }
    scrub = all(cond.values())
    valid = list(CP.OPTIONS) if scrub else ["ask_help"]
    return {"conditions": cond, "valid_options": valid, "call_needed": scrub,
            "who_chooses": "nemotron_from_menu_of_two" if scrub else "rule_only_one_valid_option",
            "tracks": scene.get("tracks") or [], "search_outcome": scene.get("search_outcome"),
            "hand": scene.get("hand"), "review_trigger": review.get("trigger")}


def answer_text(program: str) -> str:
    """A loop answer: exactly one botscript block (agent.prompt.parse_response)."""
    return "```botscript\n" + program.strip("\n") + "\n```"


def load_programs(directory=None) -> dict:
    """{name: text} of the fixed programs (PROGRAM_FILES: programs/inspect_v2.bs, scrub_v2.bs, help_v1.bs)."""
    d = Path(directory or PROGRAMS_DIR)
    progs = {}
    for name, fn in PROGRAM_FILES.items():
        p = d / fn
        if not p.is_file():
            raise FileNotFoundError(f"fixed program missing: {p}")
        progs[name] = p.read_text(encoding="utf-8")
    return progs


# ------------------------------------------------------------------------------------------------ recorded answers
RECORD_FILES = ("choice_request.json", "choice_request.sha256", "choice_response.json", "choice_call.json",
                "client_ops.jsonl")
CALL_AND_RESPONSE_KEYS = ("answer_sha256", "finish_reason", "model_id")


SYNTHETIC_MARK_FILE, SYNTHETIC_MARK_KEY = "SYNTHETIC_FIXTURE.txt", "synthetic_fixture"  # marks a synthetic test record


def load_recorded(choice_dir, allow_fake: bool = False, root=None, allow_synthetic: bool = False) -> dict:
    """A recorded choice answer (send_choice.py), checked: the request's fingerprint equals its sha file, the
    response's and the call record's; the answer's sha256 equals its text; the call answered on the real transport (a
    fake-transport answer only with allow_fake: dry runs, never shown). Never a SYNTHETIC fixture's stand-in record
    (a folder holding SYNTHETIC_FIXTURE.txt, or any record or client_ops.jsonl row marked synthetic_fixture) unless
    allow_synthetic: only the test fixture builder passes it. The response is the one the
    call recorded: choice_call.json and choice_response.json carry the same answer_sha256, finish_reason and
    model_id, and the guarded client's own log (client_ops.jsonl, written by the client before send_choice.py writes
    either file) holds exactly one "answered" line with this request's fingerprint and that same answer_sha256.
    ValueError otherwise. The cost comes from choice_call.json only (computed_cost_usd, cost_status, uncertain_usd),
    never from the loop's ledger. The request's code root must be the root of this replay, holding the same siteagent
    tree (code_root_problem), else ValueError."""
    root = use_root(root)
    llm = root_module("siteagent.agent.llm", root)
    d = Path(choice_dir)
    missing = [n for n in RECORD_FILES if not (d / n).is_file()]
    if missing:
        raise ValueError(f"recorded choice incomplete in {d}: missing {missing}")
    req = json.loads((d / "choice_request.json").read_text(encoding="utf-8"))
    sha_file = ((d / "choice_request.sha256").read_text(encoding="utf-8").split() or [""])[0]
    resp = json.loads((d / "choice_response.json").read_text(encoding="utf-8"))
    call = json.loads((d / "choice_call.json").read_text(encoding="utf-8"))
    if not all(isinstance(x, dict) for x in (req, resp, call)):
        raise ValueError("recorded choice: a record is not a JSON object")
    if not allow_synthetic and ((d / SYNTHETIC_MARK_FILE).exists()
                                or any(SYNTHETIC_MARK_KEY in x for x in (req, resp, call))):
        raise ValueError("recorded choice: a SYNTHETIC fixture's stand-in record, never served as Nemotron's answer")
    why = code_root_problem(req, root)
    if why:
        raise ValueError(f"recorded choice: {why}")
    fp = llm.request_fingerprint(req.get("request"))
    shas = {"recomputed": fp, "choice_request.json": req.get("request_sha256"), "choice_request.sha256": sha_file,
            "choice_response.json": resp.get("request_sha256"), "choice_call.json": call.get("request_sha256")}
    if len(set(shas.values())) != 1:
        raise ValueError(f"recorded choice: request fingerprints differ {shas}")
    text = resp.get("answer_text")
    if not isinstance(text, str) or sha256_text(text) != resp.get("answer_sha256"):
        raise ValueError("recorded choice: the answer text does not match its sha256")
    if call.get("status") != "answered":
        raise ValueError(f"recorded choice: the call did not answer (status {call.get('status')!r})")
    differ = [k for k in CALL_AND_RESPONSE_KEYS if call.get(k) is None or call.get(k) != resp.get(k)]
    if differ:
        raise ValueError(f"recorded choice: choice_response.json differs from the call record on {differ}")
    ops = []
    for line in (d / "client_ops.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                ops.append(json.loads(line))
            except ValueError:
                raise ValueError("recorded choice: a line of client_ops.jsonl is not JSON") from None
    if not allow_synthetic and any(isinstance(o, dict) and SYNTHETIC_MARK_KEY in o for o in ops):
        raise ValueError("recorded choice: a SYNTHETIC fixture's client_ops.jsonl, never served as Nemotron's answer")
    answered = [o for o in ops
                if isinstance(o, dict) and o.get("status") == "answered" and o.get("request_sha256") == fp]
    if len(answered) != 1 or answered[0].get("answer_sha256") != resp["answer_sha256"]:
        raise ValueError("recorded choice: the guarded client's log has no single answered line with this request and "
                         "this answer")
    fake = call.get("transport") != "token_factory" or resp.get("usage_source") != "provider"
    if fake and not (allow_fake and call.get("transport") == "fake_dry_run" and call.get("nothing_sent") is True):
        raise ValueError("recorded choice: not a real Token Factory answer (a fake one needs allow_fake: dry run only)")
    return {"choice_dir": str(d), "request_sha256": fp, "loop_request_sha256": req.get("loop_request_sha256"),
            "answer_text": text, "answer_sha256": resp["answer_sha256"], "finish_reason": resp.get("finish_reason"),
            "fake": fake, "call_date": call.get("date"), "computed_cost_usd": call.get("computed_cost_usd"),
            "cost_status": call.get("cost_status"), "uncertain_usd": call.get("uncertain_usd"),
            "code_root": req.get("code_root"),
            "code_root_siteagent_tree_sha256": req.get("code_root_siteagent_tree_sha256"),
            "files": {n: sha256_file(d / n) for n in RECORD_FILES}}


# ------------------------------------------------------------------------------------------------ the client
class DemoClient:
    """complete(request) -> LLMResponse for agent.loop.AgentLoop (no reservation binding, no network)."""

    binds_reservations = False

    def __init__(self, mode: str, programs: dict, *, root=None, choice_dir=None, recorded: dict | None = None,
                 calls_path=None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode == "prepare" and choice_dir is None:
            raise ValueError("prepare mode needs choice_dir")
        if mode == "replay" and not isinstance(recorded, dict):
            raise ValueError("replay mode needs a recorded answer (load_recorded)")
        if mode != "replay" and recorded is not None:
            raise ValueError("a recorded answer is used in replay mode only")
        for name in PROGRAM_FILES:
            t = programs.get(name) if isinstance(programs, dict) else None
            if not isinstance(t, str) or not t.strip() or "```" in t or "~~~" in t:
                raise ValueError(f"fixed program {name!r}: a non-empty text with no fence is required")
        self.root = use_root(root)
        self._llm = root_module("siteagent.agent.llm", self.root)
        self.mode, self.recorded = mode, recorded
        self.programs = {k: programs[k] for k in PROGRAM_FILES}
        self.program_sha256 = {k: sha256_text(v) for k, v in self.programs.items()}
        self.choice_dir = None if choice_dir is None else Path(choice_dir)
        self.calls_path = None if calls_path is None else Path(calls_path)
        self.model_id = DEMO_MODEL_ID
        self.received: list[dict] = []  # requests.json (bench_agent): the loop requests, never sent to Nemotron
        self.rows: list[dict] = []
        self.n = 0
        self.refusal: dict | None = None
        self.choice: dict | None = None  # what happened at the menu (request 2), for DEMO_IDENTITY.json
        self.sentinel_served = False

    # -------------------------------------------------------------------------------------------- helpers
    def _row(self, row: dict):
        row = {"schema": CALLS_SCHEMA, "client": CLIENT_VERSION, "mode": self.mode, "n": self.n, "t": now_text(),
               "loop_request_sent_to_nemotron": False, "usage_returned": dict(ZERO_USAGE),
               "usage_source": USAGE_SOURCE, **row}
        self.rows.append(row)
        if self.calls_path is not None:
            self.calls_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.calls_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
                f.flush()
                os.fsync(f.fileno())

    def _serve(self, origin: str, loop_fp: str, program: str | None = None, text: str | None = None, **extra):
        prog = self.programs[program] if program is not None else text
        out = answer_text(prog)
        self._row({"origin": origin, "loop_request_sha256": loop_fp, "program": program,
                   "program_file": PROGRAM_FILES.get(program) if program else None,
                   "program_sha256": sha256_text(prog), "answer_sha256": sha256_text(out), **extra})
        return self._llm.LLMResponse(text=out, finish_reason="stop", usage=dict(ZERO_USAGE), model_id=DEMO_MODEL_ID,
                                     latency_s=0.0, usage_source=USAGE_SOURCE)

    def _refuse(self, code: str, why: str, loop_fp: str, **extra):
        self.refusal = {"code": code, "why": why, "request_n": self.n, "t": now_text(), **extra}
        self._row({"origin": "refused", "loop_request_sha256": loop_fp, "program": None, "refusal": self.refusal})
        raise self._llm.ClientNotReady(f"demo client refused ({code}): {why}; nothing served, the episode stops")

    # -------------------------------------------------------------------------------------------- the loop's calls
    def complete(self, request: dict):
        self.n += 1
        self.received.append(json.loads(json.dumps(request)))
        loop_fp = self._llm.request_fingerprint(request)
        if self.refusal is not None:  # stopped for good; the first refusal stays the recorded one
            self._row({"origin": "refused_after_stop", "loop_request_sha256": loop_fp, "program": None})
            raise self._llm.ClientNotReady(f"demo client stopped earlier ({self.refusal['code']}); nothing served")
        try:
            info = read_loop_request(request)
        except ValueError as e:
            self._refuse("malformed_loop_request", str(e)[:200], loop_fp)
        if self.n == 1:
            if info["scene_text"] is None:
                self._refuse("no_scene_state", "request 1 carries no SCENE STATE (profile s5-info-1.0 expected)",
                             loop_fp)
            return self._serve("fixed_program_inspection", loop_fp, program="inspect")
        if self.n == 2:
            return self._request_2(info, loop_fp)
        return self._serve("fixed_help_later_request", loop_fp, text=FIXED_HELP_PROGRAM)

    def _request_2(self, info: dict, loop_fp: str):
        review = info["review"]
        if not review["is_review"]:
            if self.mode == "replay":
                self._refuse("not_a_help_review", "request 2 is not the help review the prepared episode had",
                             loop_fp, report_error_code=review.get("report_error_code"))
            return self._serve("fixed_help_request_2_not_a_help_review", loop_fp, text=FIXED_HELP_PROGRAM,
                               report_error_code=review.get("report_error_code"))
        scene = parse_scene_state(info["scene_text"])
        menu = menu_conditions(scene, review)
        self.choice = {"menu": menu, "review": review, "scene_state_sha256":
                       None if info["scene_text"] is None else sha256_text(info["scene_text"])}
        if not menu["call_needed"]:
            if self.mode == "replay":
                self._refuse("no_call_needed_in_replay", "the prepared episode needed a choice; this request does not",
                             loop_fp, menu=menu)
            self.choice.update(who_chose="rule", option="ask_help")
            return self._serve("rule_only_ask_help_valid", loop_fp, program="help", menu=menu, who_chose="rule")
        if info["task_text"] is None:
            self._refuse("no_task_text", "the loop request's TASK part cannot be read", loop_fp)
        try:
            messages = CP.build_choice_messages(info["task_text"], info["scene_text"])
        except CP.ChoiceTextError as e:
            self._refuse("choice_text", str(e)[:200], loop_fp)
        creq = self._llm.build_request(messages, CHOICE_MAX_COMPLETION_TOKENS, CHOICE_TEMPERATURE,
                                       model=self._llm.MODEL_ID, reasoning_effort=None)
        fp = self._llm.request_fingerprint(creq)
        self.choice.update(choice_request_sha256=fp)
        if self.mode == "rule":
            self._refuse("choice_needed", "both options are valid: run prepare, send_choice.py, then replay", loop_fp,
                         choice_request_sha256=fp)
        if self.mode == "prepare":
            self._write_choice_request(creq, fp, menu, loop_fp, info["scene_text"])
            self.sentinel_served = True
            self.choice.update(who_chose="nobody_yet_prepare_mode", option=None)
            return self._serve("sentinel_prepare_mode", loop_fp, text=SENTINEL_PROGRAM, menu=menu,
                               choice_request_sha256=fp)
        rec = self.recorded
        if fp != rec["request_sha256"]:
            self._refuse("choice_fingerprint_mismatch", f"live choice request {fp[:16]} differs from the recorded "
                         f"{str(rec['request_sha256'])[:16]}", loop_fp, choice_request_sha256=fp)
        if rec.get("loop_request_sha256") != loop_fp:
            self._refuse("loop_fingerprint_mismatch", f"live loop request {loop_fp[:16]} differs from the prepared "
                         f"{str(rec.get('loop_request_sha256'))[:16]}", loop_fp, choice_request_sha256=fp)
        verdict = CP.parse_answer(rec["answer_text"], rec["finish_reason"])
        if verdict["ok"] and verdict["option"] not in menu["valid_options"]:
            verdict = {**verdict, "ok": False, "refusal": "condition_not_met"}
        option = verdict["option"] if verdict["ok"] else "ask_help"
        fake = rec.get("fake") is not False  # Nemotron's labels only for an answer recorded as real
        self.choice.update(who_chose=WHO_CHOSE[(fake, bool(verdict["ok"]))], option=option,
                           answer=verdict, recorded_answer_sha256=rec["answer_sha256"], fake_answer=fake,
                           call_date=rec.get("call_date"), computed_cost_usd=rec.get("computed_cost_usd"),
                           cost_status=rec.get("cost_status"), uncertain_usd=rec.get("uncertain_usd"))
        origin = ORIGIN_CHOSEN[fake] if verdict["ok"] else ORIGIN_REFUSED[fake] + str(verdict["refusal"])
        return self._serve(origin, loop_fp, program=OPTION_PROGRAM[option], menu=menu, choice_request_sha256=fp,
                           answer=verdict, recorded_answer_sha256=rec["answer_sha256"], fake_answer=fake)

    def _write_choice_request(self, creq: dict, fp: str, menu: dict, loop_fp: str, scene_text: str):
        d = self.choice_dir
        d.mkdir(parents=True, exist_ok=True)
        doc = {"schema": CP.SCHEMA, "client": CLIENT_VERSION, "created": now_text(),
               "label": ("Demo A choice request: the exact request a real call sends to Nemotron (text only, no "
                         "images). Written by the demo client in prepare mode; the demo client sends nothing."),
               "choice_prompt_version": CP.CHOICE_PROMPT_VERSION, "system_sha256": CP.SYSTEM_SHA256,
               "text_only": True, "model": creq["model"], "max_completion_tokens": creq["max_completion_tokens"],
               "request": creq, "request_sha256": fp,
               "menu": {"options": list(CP.OPTIONS), "valid_options": menu["valid_options"],
                        "conditions": menu["conditions"], "review_trigger": menu["review_trigger"]},
               "scene_state_sha256": sha256_text(scene_text), "loop_request_sha256": loop_fp,
               "loop_request_note": ("the loop's request 2 (the designated robot prompt) is answered by the demo "
                                     "client and never sent to Nemotron"),
               # The root this request was prepared with; send_choice.py --root and the replay must match
               **code_root_of(self.root)}
        for name, data in (("choice_request.json", json.dumps(doc, indent=1, sort_keys=True) + "\n"),
                           ("choice_request.sha256", fp + "\n")):
            try:
                fd = os.open(d / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            except FileExistsError:
                self._refuse("choice_request_exists", f"{d / name} already exists (never overwritten)", loop_fp)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(data)
