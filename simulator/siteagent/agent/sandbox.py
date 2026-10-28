"""Run a robot program in a separate, locked-down process.

run_isolated(src, host, lim) starts agent/sandbox_child.py with:
- an empty environment and python -E -s -S -B (no inherited variable, so no credential, no site packages);
- on macOS, /usr/bin/sandbox-exec with a deny-by-default profile (build_profile, SANDBOX_VERSION): no network, no file
  write, no file read outside the Python installation, the system libraries and the few agent files the child
  imports, no fork, no exec, no Mach service lookup, no signal to another process. When sandbox-exec is missing the
  run says os_sandbox="none" and agent.loop refuses it by default;
- in the child, RLIMIT_CPU and RLIMIT_FSIZE=0.
The child holds only the interpreter and the program text. The simulator, robot, bench, evaluator, seeds and truth stay
in this process behind GuardedHost, which re-checks every request coming up the pipe.

The parent does not trust the child: it bounds the child's compute time, the total wall time and the size of every
message, decodes every message as strict UTF-8, counts skill calls, observations and readings itself, stops the child
at once on a call outside the API (call_not_in_api), turns an unexpected host exception into a sandbox error, checks
the final outcome against the program's own bounds (outcome_problem) and rebuilds it from a whitelist with its own
counters (rebuild_outcome). The parent and the child compile the same program text with the same lang.py.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import selectors
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .api import API, HELP_KINDS, ArgError, clean_text
from .interp import HostError, LimitReached
from .lang import Refused, compile_program
from .limits import ProgramLimits

CHILD = Path(__file__).with_name("sandbox_child.py")
AGENT_DIR = CHILD.parent
PKG_DIR = AGENT_DIR.parent
LAB_DIR = PKG_DIR.parent
SANDBOX_EXEC = "/usr/bin/sandbox-exec"
# Files the child may read in the project: the modules it imports and the limits file, nothing else
# (not the robot, the bench, the verifier, the seeds, the criteria, the outputs, the ledger).
CHILD_FILES = ("__init__.py", "api.py", "interp.py", "lang.py", "limits.py", "sandbox_child.py",
               "agent_limits_s5_v0.json", "agent_limits_s5_v0_2.json",
               "agent_limits_s5_v0_3.json")
SANDBOX_VERSION = "sandbox-exec-deny-default-0.2"


def _q(path) -> str:
    return json.dumps(str(path))  # a quoted SBPL string literal (paths contain no quote)


def build_profile() -> str:
    """sandbox-exec profile, DENY BY DEFAULT (0.2; 0.1 was
    allow-by-default with a short deny list, see S5_PREP_NOTES section 3). Allowed:
    - exec of this Python interpreter only (the venv link and its real file);
    - reading file METADATA anywhere (stat, needed to resolve paths; no content);
    - reading the Python installation (sys.base_prefix), the system libraries, the root
      directory listing (the dynamic loader needs it), /dev/null and /dev/urandom, the venv's
      pyvenv.cfg;
    - in the project: listing the lab root, siteagent/ and siteagent/agent/ (names only, the
      import system needs it) and reading only CHILD_FILES;
    - writing /dev/null; sysctl reads; signals to itself.
    Everything else is denied: network, any other file read or write, fork, exec of any other
    program, Mach service lookups (launchservices, pasteboard...), signals to other processes."""
    exe_link = Path(sys.executable)
    exe_real = exe_link.resolve()
    base = Path(sys.base_prefix).resolve()
    lines = ["(version 1)", "(deny default)",
             f"(allow process-exec (literal {_q(exe_link)}) (literal {_q(exe_real)}))",
             "(allow file-read-metadata)",
             "(allow file-read*",
             f"  (subpath {_q(base)})",
             '  (subpath "/System/Library") (subpath "/System/Volumes/Preboot/Cryptexes") (subpath "/usr/lib")',
             '  (subpath "/private/var/db/dyld")',
             '  (literal "/") (literal "/dev/null") (literal "/dev/urandom") (literal "/dev/random")',
             f"  (literal {_q(exe_link.parent.parent / 'pyvenv.cfg')})",
             f"  (literal {_q(LAB_DIR)}) (literal {_q(PKG_DIR)}) (literal {_q(AGENT_DIR)})",
             f"  (literal {_q(PKG_DIR / '__init__.py')})"]
    lines += [f"  (literal {_q(AGENT_DIR / f)})" for f in CHILD_FILES]
    lines[-1] += ")"
    lines += ['(allow file-write-data (literal "/dev/null"))',
              "(allow sysctl-read)",
              "(allow signal (target self))"]
    return "\n".join(lines) + "\n"


PROFILE = build_profile()
MAX_MSG = 4_000_000  # largest message either way (the child reads at most this per line)
# Child -> parent messages are decoded with bounds (a 20 kB JSON nested 10 000 levels deep raised RecursionError out of
# run_isolated). An honest child sends calls (name + a few short arguments) and one outcome (at most 5000 short call
# records, 200 log lines).
MAX_CHILD_MSG = 1_000_000
MAX_CHILD_JSON_DEPTH = 32
STARTUP_ALLOWANCE_S = 5.0
WRITE_TIMEOUT_S = 5.0  # a reply the child does not read within this time kills it


def child_command(extra_args: list[str] | None = None, use_os_sandbox: bool = True) -> tuple[list[str], str]:
    # -S: no site module (the child needs nothing outside the standard library and CHILD_FILES).
    cmd = [sys.executable, "-E", "-s", "-S", "-B", str(CHILD)] + (extra_args or [])
    if use_os_sandbox and sys.platform == "darwin" and shutil.which(SANDBOX_EXEC):
        return [SANDBOX_EXEC, "-p", PROFILE] + cmd, SANDBOX_VERSION
    return cmd, "none"


def _spawn(cmd):
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            env={}, cwd=str(CHILD.parent), close_fds=True)


class _Lines:
    """Bounded line reader over the child's stdout (stderr drained and kept short)."""

    def __init__(self, proc, max_line: int = MAX_MSG):
        self.p = proc
        self.max_line = max_line
        self.sel = selectors.DefaultSelector()
        self.sel.register(proc.stdout, selectors.EVENT_READ, "out")
        self.sel.register(proc.stderr, selectors.EVENT_READ, "err")
        self.buf = b""
        self.err = b""
        self.open = 2

    def readline(self, timeout: float):
        deadline = time.monotonic() + timeout
        while b"\n" not in self.buf:
            left = deadline - time.monotonic()
            if left <= 0 or self.open == 0:
                return None
            for key, _ in self.sel.select(left):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    self.sel.unregister(key.fileobj)
                    self.open -= 1
                elif key.data == "out":
                    self.buf += chunk
                    if len(self.buf) > self.max_line:
                        raise ValueError("message too large")
                else:
                    self.err = (self.err + chunk)[-4000:]
        line, self.buf = self.buf.split(b"\n", 1)
        return line


def _write(proc, data: bytes, timeout: float) -> bool:
    """Write all of data to the child's stdin within timeout seconds (non-blocking writes + select).
    False if the child did not take it in time (it is then killed by the caller)."""
    fd = proc.stdin.fileno()
    os.set_blocking(fd, False)
    sel = selectors.DefaultSelector()
    sel.register(fd, selectors.EVENT_WRITE)
    deadline = time.monotonic() + timeout
    view = memoryview(data)
    try:
        while view:
            left = deadline - time.monotonic()
            if left <= 0:
                return False
            if not sel.select(left):
                continue
            try:
                n = os.write(fd, view)
            except BlockingIOError:
                continue
            view = view[n:]
        return True
    finally:
        sel.close()


def json_depth_ok(line: bytes, max_depth: int = MAX_CHILD_JSON_DEPTH) -> bool:
    """True if the JSON text never nests arrays/objects deeper than max_depth (brackets inside
    strings ignored). Checked BEFORE json.loads, which recurses once per level."""
    if line.count(b"[") + line.count(b"{") <= max_depth:
        return True
    depth, in_str, esc = 0, False, False
    for ch in line:
        if in_str:
            if esc:
                esc = False
            elif ch == 0x5C:  # backslash
                esc = True
            elif ch == 0x22:  # quote
                in_str = False
        elif ch == 0x22:
            in_str = True
        elif ch in (0x5B, 0x7B):
            depth += 1
            if depth > max_depth:
                return False
        elif ch in (0x5D, 0x7D):
            depth -= 1
    return True


class DecodeRefused(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def decode_child_message(line: bytes) -> dict:
    """Bounded decoding of one child message: size, nesting depth, then json.loads with every
    decoding error (ValueError, RecursionError, MemoryError, UnicodeError, TypeError) turned into
    DecodeRefused, never an exception out of the parent. The message must be an object with a
    string "t"."""
    if len(line) > MAX_CHILD_MSG:
        raise DecodeRefused("message_too_large", f"child message over {MAX_CHILD_MSG} bytes")
    try:  # strict UTF-8 first: json.loads(bytes) would auto-detect UTF-16/32 and desynchronise the depth check
        text = bytes(line).decode("utf-8")
    except (UnicodeDecodeError, TypeError):
        raise DecodeRefused("protocol", "child message is not UTF-8") from None
    if not json_depth_ok(line):  # on valid UTF-8 bytes, quote and backslash bytes never occur inside a character
        raise DecodeRefused("message_too_deep", f"child message nested deeper than {MAX_CHILD_JSON_DEPTH}")
    try:
        msg = json.loads(text)
    except (ValueError, RecursionError, MemoryError, UnicodeError, TypeError) as e:
        raise DecodeRefused("protocol", f"malformed message from the child ({type(e).__name__})") from None
    if not isinstance(msg, dict) or not isinstance(msg.get("t"), str):
        raise DecodeRefused("protocol", "malformed message from the child")
    return msg


TERMINAL_NAMES = ("done", "help", "give_up", "not_found")
OUTCOME_STATUSES = ("terminal", "refused", "runtime_error", "limit", "ended_without_terminal")
NOT_FOUND_CLAIMS = ("not_found_in_inspected_scope", "not_found_scope_incomplete")


def claim_matches_terminal(terminal, claim) -> bool:
    """The claim must follow from the terminal exactly as the interpreter writes it (interp._api_call): done ->
    "task_done", help -> "help_requested:<why>", give_up -> "failed:<why>", not_found -> one of NOT_FOUND_CLAIMS; no
    terminal -> no claim. A forged child outcome {terminal: help, claim: task_done}
    after a robot-felt incident went past loop.refuse_done_after_incident, which read the terminal only."""
    if terminal is None:
        return claim is None
    if not isinstance(claim, str):
        return False
    if terminal == "done":
        return claim == "task_done"
    if terminal == "help":
        return claim.startswith("help_requested:") and _clean_why(claim[len("help_requested:"):])
    if terminal == "give_up":
        return claim.startswith("failed:") and _clean_why(claim[len("failed:"):])
    if terminal == "not_found":
        return claim in NOT_FOUND_CLAIMS
    return False


def _clean_why(s: str) -> bool:
    """the why of a help/give_up claim exactly as the API writes it (api.clean_text, spaces as _):
    1 to 80 letters, digits and _ - . with at least one letter or digit; no control, bidi or surrogate character."""
    return 0 < len(s) <= 80 and " " not in s and clean_text(s) == s


def _valid_outcome(out) -> bool:
    """Types of the child's final outcome (its values are the program's own claims; the verifier
    decides the task). The claim must follow from the terminal (claim_matches_terminal), else the parent records a
    protocol error (sandbox_error)."""
    if not isinstance(out, dict) or out.get("status") not in OUTCOME_STATUSES:
        return False
    if out.get("terminal") is not None and out.get("terminal") not in TERMINAL_NAMES:
        return False
    if (out.get("status") == "terminal") != (out.get("terminal") is not None):
        return False
    if out.get("claim") is not None and (not isinstance(out["claim"], str) or len(out["claim"]) > 120):
        return False
    if not claim_matches_terminal(out.get("terminal"), out.get("claim")):  # claim follows terminal
        return False
    err = out.get("error")
    if err is not None and not (isinstance(err, dict) and all(isinstance(err.get(k), str) for k in ("kind", "code"))
                                and isinstance(err.get("message", ""), str) and type(err.get("line", 0)) is int):
        return False
    log = out.get("log", [])
    if not (isinstance(log, list) and len(log) <= 200 and all(isinstance(x, str) and len(x) <= 300 for x in log)):
        return False
    calls = out.get("calls", [])
    if not (isinstance(calls, list) and len(calls) <= 5000 and all(isinstance(c, dict) for c in calls)):
        return False
    c = out.get("counters", {})
    return isinstance(c, dict) and all(isinstance(k, str) and type(v) in (int, float) for k, v in c.items())


MAX_ERROR_MESSAGE = 300
OUTCOME_KEYS = frozenset(("status", "terminal", "claim", "error", "counters", "log", "calls", "language_warnings",
                          "help_kind"))  # help_kind (help() only, checked in outcome_problem)
COUNTER_KEYS = frozenset(("steps", "skill_calls", "queries", "interpreter_s", "host_s"))
MAX_CHILD_COUNT = 10 ** 6  # the child's own skill_calls / queries: checked, never used (the parent keeps its count)
_ERROR_CODE = re.compile(r"^[a-z0-9_]{1,80}$")


def _utf8(s: str) -> bool:
    try:
        s.encode("utf-8")
        return True
    except UnicodeEncodeError:
        return False


def outcome_problem(out, lim: ProgramLimits, n_lines: int) -> str | None:
    """what is wrong with a child's final outcome for THIS program and these limits, or None. Every
    value the parent keeps or shows is bounded here; an error dict's extra keys and a long message are still dropped
    and cut by rebuild_outcome (a test)."""
    try:
        if not _valid_outcome(out):
            return "types, or a claim that does not follow from the terminal"
    except (RecursionError, ValueError, TypeError):  # bounded by the depth check; defence in depth
        return "types"
    extra = set(out) - OUTCOME_KEYS
    if extra:
        return f"unknown key {sorted(extra)[0][:40]!r}"
    st = out["status"]
    if st == "refused":
        return "the child refused a program the parent accepted"
    if "help_kind" in out:  # only with terminal help, one of api.HELP_KINDS (an exact str, nothing else)
        hk = out["help_kind"]
        if out.get("terminal") != "help" or type(hk) is not str or hk not in HELP_KINDS:
            return "help_kind"
    c = out.get("counters", {})
    if set(c) - COUNTER_KEYS:
        return f"unknown counter {sorted(set(c) - COUNTER_KEYS)[0][:40]!r}"
    if "steps" in c and not (type(c["steps"]) is int and 0 <= c["steps"] <= lim.max_steps + 1):
        return "counters.steps out of range"
    for k in ("skill_calls", "queries"):
        if k in c and not (type(c[k]) is int and 0 <= c[k] <= MAX_CHILD_COUNT):
            return f"counters.{k} out of range"
    for k in ("interpreter_s", "host_s"):  # (math.isfinite of a huge int would raise OverflowError: type first)
        v = c.get(k, 0)
        if not ((type(v) is float and math.isfinite(v) or type(v) is int) and 0 <= v <= MAX_CHILD_COUNT):
            return f"counters.{k} out of range"
    err = out.get("error")
    if st == "terminal":
        if err is not None:
            return "a terminal outcome with an error"
    else:
        if err is None:
            return f"status {st} without an error"
        if err["kind"] != ("limit" if st == "limit" else "runtime_error"):
            return "error.kind does not match the status"
        if not _ERROR_CODE.match(err["code"]) or (st == "ended_without_terminal" and err["code"] != "no_terminal"):
            return "error.code"
        if not 0 <= err.get("line", 0) <= n_lines + 1:
            return "error.line outside the program"
        if not _utf8(err.get("message", "")):
            return "error.message is not UTF-8 text"
    log = out.get("log", [])
    if len(log) > lim.max_log_lines or not all(_utf8(x) for x in log):
        return "log"
    if len(out.get("calls", [])) > lim.max_skill_calls + lim.max_queries + 2:
        return "more call records than the program limits allow"
    lw = out.get("language_warnings")
    if lw is not None and not (isinstance(lw, list) and len(lw) <= 25 and all(isinstance(w, dict) for w in lw)):
        return "language_warnings"
    return None


def rebuild_outcome(out: dict, n_kind: dict) -> dict:
    """The outcome the parent keeps, rebuilt from a whitelist of a VALID child outcome (_valid_outcome
    first): no extra key, a bounded error, the parent's own call counts. The child's steps count is kept
    (deterministic); its timing and its list of calls are not."""
    err = out.get("error")
    error = None if err is None else {"kind": err["kind"][:40], "code": err["code"][:80],
                                      "message": str(err.get("message", ""))[:MAX_ERROR_MESSAGE],
                                      "line": int(err.get("line", 0))}
    steps = (out.get("counters") or {}).get("steps")
    kept = {"status": out["status"], "terminal": out.get("terminal"), "claim": out.get("claim"), "error": error,
            "log": [str(x) for x in out.get("log", [])], "calls": [],
            "counters": {"steps": steps if type(steps) is int else None, "skill_calls": n_kind["skill"],
                         "queries": n_kind["query"], "source": "parent count (steps reported by the child)"},
            "child_reported": {"calls": len(out.get("calls", [])),
                               "interpreter_s": (out.get("counters") or {}).get("interpreter_s")}}
    hk = out.get("help_kind")
    if out.get("terminal") == "help" and type(hk) is str and hk in HELP_KINDS:  # checked by outcome_problem
        kept["help_kind"] = hk  # an absent key stays absent (read as other, never reviewed)
    return kept


def _kill(proc):
    try:
        proc.kill()
    except OSError:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def _fail(status, code, message, os_sandbox, stderr=b""):
    return {"status": status, "terminal": None, "claim": None,
            "error": {"kind": status if status != "sandbox_error" else "runtime_error", "code": code, "line": 0,
                      "message": message},
            "counters": {}, "log": [], "calls": [], "isolation": {"process": "separate", "os_sandbox": os_sandbox,
                                                                   "stderr_tail": stderr.decode("utf-8", "replace")[-500:]}}


def run_isolated(src: str, host, lim: ProgramLimits | None = None, use_os_sandbox: bool = True) -> dict:
    """Same outcome dict as interp.Interpreter.run, plus "isolation". host: agent.host.GuardedHost."""
    lim = lim or ProgramLimits()
    try:  # refuse in the parent first: nothing starts for a refused program
        warnings = compile_program(src, lim).get("warnings") or []  # parent side (botscript-0.2), never the child's
    except Refused as e:
        return {"status": "refused", "terminal": None, "claim": None, "error": e.as_dict(), "counters": {},
                "log": [], "calls": [], "isolation": {"process": "not_started"}}
    except Exception as x:  # noqa: BLE001 (compile_program raises only Refused; defence in depth)
        e = Refused("checker_internal", f"the program could not be checked (internal {type(x).__name__}); nothing ran")
        return {"status": "refused", "terminal": None, "claim": None, "error": e.as_dict(), "counters": {},
                "log": [], "calls": [], "isolation": {"process": "not_started"}}
    n_lines = src.count("\n") + 1
    cmd, os_sb = child_command(use_os_sandbox=use_os_sandbox)
    proc = _spawn(cmd)
    rd = _Lines(proc, MAX_CHILD_MSG)
    t_start = time.monotonic()
    child_s = 0.0
    n_calls = 0
    n_kind = {"skill": 0, "query": 0}  # the parent's own count (skill + observe share the program limit)
    try:
        if not _write(proc, (json.dumps({"t": "start", "src": src, "limits": dataclasses.asdict(lim)}) + "\n").encode(),
                      WRITE_TIMEOUT_S):
            _kill(proc)
            return _fail("sandbox_error", "write_timeout", "the child did not read its start message", os_sb)
        allowance = STARTUP_ALLOWANCE_S
        while True:
            budget_child = lim.max_interpreter_cpu_s + allowance - child_s
            budget_wall = lim.max_wall_s - (time.monotonic() - t_start)
            if budget_wall <= 0:
                _kill(proc)
                return _fail("limit", "max_wall_s", "the program ran longer than the wall-time limit", os_sb)
            t0 = time.monotonic()
            try:
                line = rd.readline(max(0.0, min(budget_child, budget_wall)))
            except ValueError:
                _kill(proc)
                return _fail("sandbox_error", "message_too_large", "the child sent too large a message", os_sb)
            child_s += time.monotonic() - t0
            if line is None:
                _kill(proc)
                if proc.returncode not in (None, 0) and rd.open == 0:
                    return _fail("sandbox_error", "child_died", f"the child stopped (code {proc.returncode})",
                                 os_sb, rd.err)
                return _fail("limit", "interpreter_time", "the program computed for too long without a robot call",
                             os_sb, rd.err)
            try:
                msg = decode_child_message(line)
                kind = msg["t"]
            except DecodeRefused as e:
                _kill(proc)
                return _fail("sandbox_error", e.code, e.message, os_sb, rd.err)
            if kind == "end":
                out = msg.get("outcome")
                problem = outcome_problem(out, lim, n_lines)  # types AND the program's own bounds
                if problem is not None:
                    _kill(proc)
                    return _fail("sandbox_error", "protocol", f"malformed outcome from the child: {problem}", os_sb,
                                 rd.err)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    _kill(proc)
                # rebuilt from a whitelist, counters from the parent's own count (the child's figures apart)
                out = rebuild_outcome(out, n_kind)
                out["isolation"] = {"process": "separate", "os_sandbox": os_sb, "host_calls": n_calls,
                                    "child_compute_s": round(child_s, 3)}
                if warnings:
                    out["language_warnings"] = [dict(w) for w in warnings]
                return out
            if kind != "call" or not isinstance(msg.get("name"), str) or not isinstance(msg.get("args"), dict):
                _kill(proc)
                return _fail("sandbox_error", "protocol", "unexpected message from the child", os_sb, rd.err)
            allowance = 0.5  # startup done; afterwards only a small allowance per reply
            n_calls += 1
            # the program limits, mirrored here: a child past its own interpreter's limit is not honest
            k = API.get(msg["name"], {}).get("kind")
            if k not in ("skill", "observe", "query"):  # an honest child asks only for these (terminals end it)
                _kill(proc)
                return _fail("sandbox_error", "call_not_in_api", f"the child asked for {msg['name'][:40]!r}, which is "
                             "not a skill, an observation or a reading of the API", os_sb)
            bucket = "skill" if k in ("skill", "observe") else "query"
            cap = lim.max_skill_calls if bucket == "skill" else lim.max_queries
            if n_kind[bucket] >= cap:
                _kill(proc)
                return _fail("sandbox_error", "parent_program_limit",
                             f"the child asked for more {bucket} calls than the program limit ({cap})", os_sb)
            n_kind[bucket] += 1
            try:
                reply = {"t": "result", "value": host.call(msg["name"], msg["args"])}
            except LimitReached as e:
                reply = {"t": "limit", "code": e.code, "message": e.message}
            except ArgError as e:
                reply = {"t": "arg_error", "message": str(e)}
            except HostError as e:
                _kill(proc)
                return _fail("sandbox_error", "host_error", f"host returned non-plain data: {e}", os_sb)
            except Exception as e:  # noqa: BLE001 (any other host exception: the child is stopped, nothing escapes)
                _kill(proc)
                return _fail("sandbox_error", "host_exception", f"the host raised {type(e).__name__}", os_sb)
            data = (json.dumps(reply, allow_nan=False) + "\n").encode()
            if len(data) > MAX_MSG:
                _kill(proc)
                return _fail("sandbox_error", "reply_too_large",
                             f"a host reply exceeded {MAX_MSG} bytes (the child reads at most that)", os_sb)
            budget_wall = lim.max_wall_s - (time.monotonic() - t_start)
            if budget_wall <= 0:
                _kill(proc)
                return _fail("limit", "max_wall_s", "the program ran longer than the wall-time limit", os_sb)
            if not _write(proc, data, min(WRITE_TIMEOUT_S, budget_wall)):
                _kill(proc)
                return _fail("sandbox_error", "write_timeout", "the child did not read a reply in time", os_sb)
    except (BrokenPipeError, OSError) as e:
        _kill(proc)
        return _fail("sandbox_error", "pipe", f"pipe error: {type(e).__name__}", os_sb, rd.err)
    finally:
        if proc.poll() is None:
            _kill(proc)
        for f in (proc.stdin, proc.stdout, proc.stderr):
            try:
                f.close()
            except OSError:
                pass


def selftest(decoy_path: str, use_os_sandbox: bool = True, reads: dict | None = None, write_path: str | None = None,
             try_exec: bool = False) -> dict:
    """Run the child's OS-layer self test (tests only). reads: {label: path} files the child tries
    to read (lengths reported, never contents)."""
    spec = {"decoy": decoy_path, "reads": reads or {}, "try_exec": try_exec}
    if write_path:
        spec["write"] = write_path
    cmd, os_sb = child_command(["--selftest", json.dumps(spec)], use_os_sandbox)
    p = subprocess.run(cmd, env={}, capture_output=True, timeout=30, cwd=str(CHILD.parent))
    lines = [json.loads(x) for x in p.stdout.decode().strip().splitlines() if x.startswith("{")]
    rep = next((x.get("report", {}) for x in lines if x.get("t") == "selftest"), {})
    ex = next((x.get("result") for x in lines if x.get("t") == "exec"), None)
    if try_exec:
        rep["exec_other_program"] = "EXEC_OK" if b"EXEC_OK" in p.stdout else (ex or "no_output")
    return {"os_sandbox": os_sb, "returncode": p.returncode, **rep}
