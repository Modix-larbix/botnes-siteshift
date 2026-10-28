"""Interpreter of BotScript programs: runs the checked form built by agent.lang.

It only knows plain values (None, bool, int, float, str, list, dict with str keys), never mutated in place, so sharing
is harmless. The program reaches the outside world through one door: host.call(name, args) for a name of the skill API
(agent.api), with checked, named arguments; whatever the host returns is copied into new plain containers (to_plain).

Limits are checked before each step: instructions, loop iterations, skill calls and readings per run, interpreter
time (host time excluded), value sizes (lengths, total items, depth), integer magnitude and non-finite numbers. A limit
stops the program with an observable "limit" outcome; a type error, unknown variable or bad argument stops it with
"runtime_error". Neither is an exception the program can catch (the language has no try).

log() and str() build their text piece by piece up to a bound, so one call cannot produce an unbounded string; error
messages are cut and made UTF-8 safe. A help() that is reached always ends the program: a why that cannot be read is
replaced by api.HELP_WHY_UNREADABLE and the substitution is logged. Methods (lang.METHODS) check their receiver's type
first and never look up attributes; user functions get their own frame and a bounded call depth.
"""
from __future__ import annotations

import json
import math
import time

from .api import API, HELP_KINDS, HELP_WHY_UNREADABLE, ArgError, check_args, check_help
from .lang import METHODS, PowerError, Refused, compile_program, power
from .limits import ProgramLimits

PLAIN_SCALARS = (type(None), bool, int, float, str)
LOG_MAX_ARGS = 10
LOG_LINE_CHARS = 300
LOG_DUMP_MAX_ITEMS = 200  # a list or dict with more items (all levels) is logged as a placeholder
MAX_ERROR_MESSAGE = 300  # every outcome's error message is cut here (as the sandbox parent does)


def json_text(v, cap: int, cut: bool = False):
    """the JSON text of v built piece by piece (JSONEncoder.iterencode), stopped as soon as it is
    longer than cap: None then (or, with cut, its first cap characters). Before, json.dumps built the whole text
    first: about 480 MB for a legal value listing one large dict 9 times."""
    parts, n = [], 0
    for chunk in json.JSONEncoder().iterencode(v):
        parts.append(chunk)
        n += len(chunk)
        if n > cap:
            return "".join(parts)[:cap] if cut else None
    return "".join(parts)


def short_repr(s, n: int = 60) -> str:
    """repr() of a program value for a message, never longer than about n characters."""
    r = repr(s[:n] if type(s) is str else s)
    return r if len(r) <= n else r[:n] + "..."


def utf8_safe(s: str) -> str:
    """The same text with any lone surrogate written as an escape, so it can always be encoded as UTF-8."""
    return s.encode("utf-8", "backslashreplace").decode("utf-8")


class ProgramError(Exception):
    def __init__(self, code: str, message: str, line: int = 0):
        super().__init__(message)
        self.code, self.message, self.line = code, message, line


class LimitReached(Exception):
    def __init__(self, code: str, message: str = "", line: int = 0):
        super().__init__(message or code)
        self.code, self.message, self.line = code, message or code, line


class HostError(Exception):
    """The host returned something that is not plain data (a bench bug, never the program's)."""


class _Terminal(Exception):
    def __init__(self, name: str, claim: str, help_kind: str | None = None):
        super().__init__(claim)
        self.name, self.claim, self.help_kind = name, claim, help_kind  # help_kind: help() only


class _Break(Exception):
    pass


class _Continue(Exception):
    pass


class _Return(Exception):
    def __init__(self, value):
        super().__init__("return")
        self.value = value


def to_plain(v, lim: ProgramLimits, depth: int = 0, budget: list | None = None):
    """Deep copy into plain JSON-like values; refuses any other type (HostError) and over-size values."""
    budget = budget if budget is not None else [lim.max_value_items]
    budget[0] -= 1
    if budget[0] < 0:
        raise HostError("value has too many items")
    if depth > lim.max_value_depth:
        raise HostError("value nested too deeply")
    if v is None or isinstance(v, bool) or type(v) is int or type(v) is str:
        if type(v) is str and len(v) > lim.max_str_len:
            return v[:lim.max_str_len]
        if type(v) is int and abs(v) > lim.max_int_abs:
            raise HostError("integer too large")
        return v
    if type(v) is float:
        return v if math.isfinite(v) else None
    if type(v) in (list, tuple):
        if len(v) > lim.max_list_len:
            raise HostError("list too long")
        return [to_plain(x, lim, depth + 1, budget) for x in v]
    if type(v) is dict:
        out = {}
        for k, x in v.items():
            if type(k) is not str:
                raise HostError("dict key is not a string")
            out[k] = to_plain(x, lim, depth + 1, budget)
        return out
    raise HostError(f"non-plain value of type {type(v).__name__}")


def _type(v) -> str:
    return {type(None): "None", bool: "bool", int: "int", float: "float", str: "str", list: "list",
            dict: "dict"}.get(type(v), type(v).__name__)


def _is_num(v) -> bool:
    return type(v) in (int, float)


class Interpreter:
    def __init__(self, host, lim: ProgramLimits | None = None):
        self.host = host
        self.lim = lim or ProgramLimits()
        self.env: dict = {}
        self.steps = 0
        self.skill_calls = 0
        self.queries = 0
        self.log: list[str] = []
        self.calls: list[dict] = []
        self._host_s = 0.0
        self._t0 = 0.0
        self._line = 0
        # 0.3: functions and the current call (frame None at the top level)
        self.functions: dict = {}
        self.frame: dict | None = None
        self.flocals: frozenset = frozenset()
        self.fname: str | None = None
        self.depth = 0

    # ------------------------------------------------------------ limits
    def _tick(self, line: int = 0):
        self.steps += 1
        if line:
            self._line = line
        if self.steps > self.lim.max_steps:
            raise LimitReached("max_steps", f"more than {self.lim.max_steps} instructions", self._line)
        if self.steps % 128 == 0:
            if time.perf_counter() - self._t0 - self._host_s > self.lim.max_interpreter_cpu_s:
                raise LimitReached("interpreter_time", f"more than {self.lim.max_interpreter_cpu_s} s of "
                                                       "interpreter time", self._line)

    def _num(self, x):
        if type(x) is int:
            if abs(x) > self.lim.max_int_abs:
                raise ProgramError("value_too_large", "integer too large", self._line)
            return x
        if type(x) is float and not math.isfinite(x):
            raise ProgramError("non_finite", "a computation gave an infinite or undefined number", self._line)
        return x

    def _size(self, v, depth=0) -> int:
        if depth > self.lim.max_value_depth:
            raise ProgramError("value_too_large", "value nested too deeply", self._line)
        if type(v) is list:
            if len(v) > self.lim.max_list_len:
                raise ProgramError("value_too_large", f"list longer than {self.lim.max_list_len}", self._line)
            n = 1 + sum(self._size(x, depth + 1) for x in v)
        elif type(v) is dict:  # each entry counts (its key as one item), keys are bounded too
            if any(type(k) is str and len(k) > self.lim.max_str_len for k in v):
                raise ProgramError("value_too_large", f"dict key longer than {self.lim.max_str_len}", self._line)
            n = 1 + sum(1 + self._size(x, depth + 1) for x in v.values())
        elif type(v) is str:
            if len(v) > self.lim.max_str_len:
                raise ProgramError("value_too_large", f"string longer than {self.lim.max_str_len}", self._line)
            n = 1
        else:
            n = 1
        if n > self.lim.max_value_items:
            raise ProgramError("value_too_large", f"value with more than {self.lim.max_value_items} items", self._line)
        return n

    def _new(self, v):
        self._size(v)
        return v

    # ------------------------------------------------------------ run
    def run(self, program) -> dict:
        """program: text or the output of lang.compile_program. Returns the outcome (plain dict)."""
        self._t0 = time.perf_counter()
        try:
            ir = program if isinstance(program, dict) else compile_program(program, self.lim)
        except Refused as e:
            return self._outcome("refused", error=e.as_dict())
        except Exception as x:  # noqa: BLE001 (compile_program raises only Refused; defence in depth)
            return self._outcome("refused", error=Refused("checker_internal", "the program could not be checked "
                                                          f"(internal {type(x).__name__}); nothing ran").as_dict())
        self.functions = {name: (frozenset(f["locals"]), f) for name, f in (ir.get("functions") or {}).items()}
        out = self._run_ir(ir)
        if ir.get("warnings"):  # botscript-0.2: calls that will fail if reached, announced before the run
            out["language_warnings"] = [dict(w) for w in ir["warnings"]]
        return out

    def _run_ir(self, ir) -> dict:
        try:
            self._block(ir["body"])
            return self._outcome("ended_without_terminal", error={
                "kind": "runtime_error", "code": "no_terminal", "line": self._line,
                "message": "the program ended without done(), help(why, kind), give_up(why) or not_found(scope)"})
        except _Terminal as t:
            return self._outcome("terminal", terminal=t.name, claim=t.claim, help_kind=t.help_kind)
        except LimitReached as e:
            return self._outcome("limit", error={"kind": "limit", "code": e.code, "line": e.line or self._line,
                                                 "message": e.message})
        except ProgramError as e:
            return self._outcome("runtime_error", error={"kind": "runtime_error", "code": e.code,
                                                         "line": e.line or self._line, "message": e.message})
        except (_Break, _Continue):  # refused before running since 0.3; kept as a bounded outcome
            return self._outcome("runtime_error", error={"kind": "runtime_error", "code": "bad_break",
                                                         "line": self._line, "message": "break/continue outside a loop"})
        except _Return:  # refused before running (lang); kept as a bounded outcome
            return self._outcome("runtime_error", error={"kind": "runtime_error", "code": "bad_return",
                                                         "line": self._line, "message": "return outside a function"})
        except RecursionError:  # cannot happen within max_nesting_depth; kept as a bounded outcome
            return self._outcome("limit", error={"kind": "limit", "code": "recursion", "line": self._line,
                                                 "message": "evaluation too deep"})
        except HostError:
            raise  # a bench bug, never the program's outcome
        except (ArithmeticError, LookupError, ValueError, TypeError) as e:
            # An unexpected Python error inside the interpreter (a defect of this interpreter, e.g. an overflow in
            # dist()): reported as an observable runtime error with the exception type only,
            # so the episode can be corrected and never crashes.
            return self._outcome("runtime_error", error={"kind": "runtime_error", "code": "interpreter_internal",
                                                         "line": self._line,
                                                         "message": f"internal interpreter error ({type(e).__name__})"})

    def _outcome(self, status, terminal=None, claim=None, error=None, help_kind=None) -> dict:
        if isinstance(error, dict) and isinstance(error.get("message"), str):  # bounded, UTF-8 safe
            error = dict(error, message=utf8_safe(error["message"][:MAX_ERROR_MESSAGE]))
        out = {"status": status, "terminal": terminal, "claim": claim, "error": error,
               "counters": {"steps": self.steps, "skill_calls": self.skill_calls, "queries": self.queries,
                            "interpreter_s": round(max(0.0, time.perf_counter() - self._t0 - self._host_s), 4),
                            "host_s": round(self._host_s, 4)},
               "log": list(self.log), "calls": list(self.calls)}
        if terminal == "help":  # the structured reason of help() (api.HELP_KINDS); absent for any other outcome
            out["help_kind"] = help_kind if help_kind in HELP_KINDS else "other"  # always a checked kind here
        return out

    # ------------------------------------------------------------ statements
    def _block(self, stmts):
        for s in stmts:
            self._stmt(s)

    # ------------------------------------------------------------ variables (0.3: a call's locals, then the globals)
    def _get(self, name):
        if self.frame is not None and name in self.flocals:
            if name in self.frame:
                return self.frame[name]
            raise ProgramError("unbound_local", f"{name!r} is assigned in function {self.fname}(), so it is local there, "
                               "and it is read before it gets a value; a function cannot change a global: pass the "
                               "value as a parameter and return the new value", self._line)
        if name not in self.env:
            raise ProgramError("unknown_name", f"{name!r} is not defined", self._line)
        return self.env[name]

    def _set(self, name, v):
        if self.frame is None:
            self.env[name] = v
        elif name in self.flocals:
            self.frame[name] = v
        else:  # pragma: no cover (lang puts every name a body assigns in its locals)
            raise ProgramError("internal", f"assignment of a non-local {name!r} in {self.fname}()", self._line)

    def _stmt(self, s):
        op, line = s[0], s[1]
        self._tick(line)
        if op == "assign":
            self._set(s[2], self._eval(s[3]))
        elif op == "augassign":
            self._set(s[2], self._binop(s[3], self._get(s[2]), self._eval(s[4])))
        elif op == "expr":
            self._eval(s[2])
        elif op == "if":
            self._block(s[3] if self._truth(self._eval(s[2])) else s[4])
        elif op == "while":
            n = 0
            while self._truth(self._eval(s[2])):
                n += 1
                if n > self.lim.max_loop_iterations:
                    raise LimitReached("max_loop_iterations", f"a loop ran more than {self.lim.max_loop_iterations} "
                                                              "times", line)
                try:
                    self._block(s[3])
                except _Break:
                    break
                except _Continue:
                    continue
        elif op == "for":
            seq = self._eval(s[3])
            if type(seq) is dict:
                seq = list(seq.keys())
            elif type(seq) is str:
                seq = list(seq)
            elif type(seq) is not list:
                raise ProgramError("type_error", f"cannot loop over {_type(seq)}", line)
            for n, x in enumerate(seq):
                if n + 1 > self.lim.max_loop_iterations:
                    raise LimitReached("max_loop_iterations", f"a loop ran more than {self.lim.max_loop_iterations} "
                                                              "times", line)
                self._set(s[2], x)
                try:
                    self._block(s[4])
                except _Break:
                    break
                except _Continue:
                    continue
        elif op == "break":
            raise _Break()
        elif op == "continue":
            raise _Continue()
        elif op == "pass":
            pass
        elif op == "return":
            raise _Return(None if s[2] is None else self._eval(s[2]))
        else:  # pragma: no cover (compile_program never produces it)
            raise ProgramError("internal", f"unknown statement {op}", line)

    @staticmethod
    def _truth(v) -> bool:
        return bool(v)  # plain values only: no user-defined __bool__ exists

    # ------------------------------------------------------------ expressions
    def _eval(self, e):
        self._tick()
        op = e[0]
        if op == "const":
            return e[1]
        if op == "name":
            return self._get(e[1])
        if op == "list":
            return self._new([self._eval(x) for x in e[1]])
        if op == "dict":
            out = {}
            for k, v in e[1]:
                kk = self._eval(k)
                if type(kk) is not str:
                    raise ProgramError("type_error", "dict keys must be strings", self._line)
                out[kk] = self._eval(v)
            return self._new(out)
        if op == "bin":
            return self._binop(e[1], self._eval(e[2]), self._eval(e[3]))
        if op == "unary":
            v = self._eval(e[2])
            if e[1] == "not":
                return not self._truth(v)
            if not _is_num(v) or type(v) is bool:
                raise ProgramError("type_error", f"unary {e[1]} needs a number, got {_type(v)}", self._line)
            return self._num(-v if e[1] == "neg" else +v)
        if op == "bool":
            if e[1] == "and":
                v = True
                for x in e[2]:
                    v = self._eval(x)
                    if not self._truth(v):
                        return v
                return v
            v = False
            for x in e[2]:
                v = self._eval(x)
                if self._truth(v):
                    return v
            return v
        if op == "cmp":
            left = self._eval(e[1])
            for cop, rexpr in e[2]:
                right = self._eval(rexpr)
                if not self._compare(cop, left, right):
                    return False
                left = right
            return True
        if op == "ifexp":
            return self._eval(e[2]) if self._truth(self._eval(e[1])) else self._eval(e[3])
        if op == "sub":
            return self._index(self._eval(e[1]), self._eval(e[2]))
        if op == "slice":
            v = self._eval(e[1])
            lo = None if e[2] is None else self._eval(e[2])
            hi = None if e[3] is None else self._eval(e[3])
            if type(v) not in (list, str) or any(x is not None and type(x) is not int for x in (lo, hi)):
                raise ProgramError("type_error", "slices need a list or string and integer bounds", self._line)
            return v[lo:hi]
        if op == "call":
            return self._call(e)
        if op == "method":
            return self._method(e)
        if op == "ucall":
            return self._ucall(e)
        raise ProgramError("internal", f"unknown expression {op}", self._line)  # pragma: no cover

    def _binop(self, op, a, b):
        if type(a) is bool or type(b) is bool:
            raise ProgramError("type_error", f"operator {op} does not take True/False", self._line)
        if op == "**":  # 0.3: lang.power (|exponent| <= 8, finite, within max_int_abs), same rule as the checker
            if not (_is_num(a) and _is_num(b)):
                raise ProgramError("type_error", f"operator ** cannot combine {_type(a)} and {_type(b)}", self._line)
            try:
                return self._num(power(a, b, self.lim.max_int_abs))
            except PowerError as x:
                raise ProgramError(x.code, f"x ** y: {x.message}", self._line) from None
        if _is_num(a) and _is_num(b):
            try:
                if op == "+":
                    r = a + b
                elif op == "-":
                    r = a - b
                elif op == "*":
                    r = a * b
                elif op == "/":
                    r = a / b
                elif op == "//":
                    r = a // b
                else:
                    r = a % b
            except ZeroDivisionError:
                raise ProgramError("division_by_zero", "division by zero", self._line) from None
            except OverflowError:
                raise ProgramError("non_finite", "number too large", self._line) from None
            return self._num(r)
        if op == "+" and type(a) is type(b) and type(a) in (str, list):
            if len(a) + len(b) > (self.lim.max_str_len if type(a) is str else self.lim.max_list_len):
                raise ProgramError("value_too_large", f"{_type(a)} would be too long", self._line)
            return self._new(a + b)
        raise ProgramError("type_error", f"operator {op} cannot combine {_type(a)} and {_type(b)}", self._line)

    def _compare(self, op, a, b) -> bool:
        if op == "==":
            return a == b
        if op == "!=":
            return a != b
        if op in ("in", "not in"):
            if type(b) not in (list, str, dict):
                raise ProgramError("type_error", f"'in' needs a list, string or dict, got {_type(b)}", self._line)
            if type(b) in (str, dict) and type(a) is not str:
                raise ProgramError("type_error", f"'in' a {_type(b)} needs a string", self._line)
            r = a in b
            return r if op == "in" else not r
        ok = (_is_num(a) and _is_num(b) and type(a) is not bool and type(b) is not bool) or (
            type(a) is str and type(b) is str)
        if not ok:
            raise ProgramError("type_error", f"cannot order {_type(a)} and {_type(b)}", self._line)
        return {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}[op]

    def _index(self, v, i):
        if type(v) in (list, str):
            if type(i) is not int:
                raise ProgramError("type_error", "list and string indices must be integers", self._line)
            if not -len(v) <= i < len(v):
                raise ProgramError("index_error", f"index {i} out of range (length {len(v)})", self._line)
            return v[i]
        if type(v) is dict:
            if type(i) is not str:
                raise ProgramError("type_error", "dict keys are strings", self._line)
            if i not in v:
                raise ProgramError("key_error", f"no key {short_repr(i)} (use get(d, key, default))", self._line)
            return v[i]
        raise ProgramError("type_error", f"cannot index {_type(v)}", self._line)

    # ------------------------------------------------------------ calls
    def _call(self, e):
        line, name = e[1], e[2]
        self._line = line
        if name == "help" and len(e[3]) <= 2 and all(k in ("why", "kind") for k, _ in e[4]):
            return self._help_call(e)  # a reached help always ends the program
        args = [self._eval(a) for a in e[3]]
        kwargs = {k: self._eval(v) for k, v in e[4]}
        if name in API:
            return self._api_call(name, args, kwargs, line)
        if kwargs and name not in ("get", "round"):
            raise ProgramError("bad_arg", f"{name}() takes no keyword arguments", line)
        return self._builtin(name, args, kwargs, line)

    def _help_call(self, e):
        """help(why, kind) once reached. The kind is read as written (agent.lang: a word of
        api.HELP_KINDS in quotes; anything else is evaluated and checked, a runtime bad_arg as before). The why is
        computed; an error while computing it (a ProgramError or a LimitReached of this program) or a value that is not
        a readable text gives why api.HELP_WHY_UNREADABLE (api.check_help) and the help still ends the program."""
        line, pos, kws = e[1], list(e[3]), dict(e[4])
        why_node = pos[0] if pos else kws.get("why")
        kind_node = pos[1] if len(pos) > 1 else kws.get("kind")
        if kind_node is not None and kind_node[0] == "const":
            kind = kind_node[1]
        else:  # pragma: no cover (agent.lang refuses a kind that is not a word in quotes)
            kind = None if kind_node is None else self._eval(kind_node)
        why, failed = None, None
        try:
            why = None if why_node is None else self._eval(why_node)
        except (ProgramError, LimitReached) as x:
            failed = f"computing why: {x.code}"
        self._line = line
        try:
            checked, replaced = check_help(why, kind)
        except ArgError as err:  # a kind outside api.HELP_KINDS (agent.lang bypassed): a runtime error, never "other"
            self.calls.append({"line": line, "name": "help", "status": "bad_arg", "message": str(err)})
            raise ProgramError("bad_arg", f"help(): {err}", line) from None
        replaced = failed or replaced
        rec = {"line": line, "name": "help", "status": "terminal", "args": checked}
        if replaced:
            rec["why_substituted"] = replaced[:160]
            if len(self.log) < self.lim.max_log_lines:
                self.log.append(utf8_safe(f"help(): why not a readable text ({replaced}); kept as "
                                          f"{HELP_WHY_UNREADABLE}")[:LOG_LINE_CHARS])
        self.calls.append(rec)
        raise _Terminal("help", f"help_requested:{checked['why']}", checked["kind"])

    def _method(self, e):
        """0.3: recv.m(args), m in lang.METHODS: the receiver's type, then the builtin m(recv, *args)."""
        line, m = e[1], e[2]
        self._line = line
        recv = self._eval(e[3])
        args = [self._eval(a) for a in e[4]]
        self._line = line
        spec = METHODS.get(m)
        if spec is None:  # pragma: no cover (lang compiles only the listed methods)
            raise ProgramError("internal", f"unknown method {m!r}", line)
        want, lo, hi = spec
        if (want == "dict" and type(recv) is not dict) or (want == "str" and type(recv) is not str):
            kind = "a dict" if want == "dict" else "a string"
            raise ProgramError("type_error", f".{m}() is a method of {kind}, but the value before the dot is "
                                             f"{_type(recv)}", line)
        if not lo <= len(args) <= hi:  # pragma: no cover (lang refuses other counts)
            raise ProgramError("bad_arg", f".{m}() takes {lo}..{hi} arguments, got {len(args)}", line)
        return self._builtin(m, [recv] + args, {}, line)

    def _ucall(self, e):
        """0.3: a call of a top-level def: bind the arguments, run the body in a new frame, give its return value."""
        line, name = e[1], e[2]
        self._line = line
        entry = self.functions.get(name)
        if entry is None:  # pragma: no cover (lang compiles calls of defined functions only)
            raise ProgramError("unknown_function", f"{name!r} is not defined", line)
        flocals, f = entry
        args = [self._eval(a) for a in e[3]]
        kwargs = [(k, self._eval(v)) for k, v in e[4]]
        self._line = line
        params = f["params"]
        if len(args) > len(params):
            raise ProgramError("bad_arg", f"{name}() takes at most {len(params)} arguments, got {len(args)}", line)
        frame = dict(zip(params, args))
        for k, v in kwargs:
            if k not in params or k in frame:
                raise ProgramError("bad_arg", f"{name}(): unknown or repeated argument {k!r}", line)
            frame[k] = v
        for p in params:
            if p not in frame:
                if p not in f["defaults"]:
                    raise ProgramError("bad_arg", f"{name}() is missing argument {p!r}", line)
                frame[p] = f["defaults"][p]  # a constant (number, string, True, False, None): never shared state
        if self.depth >= self.lim.max_call_depth:
            raise LimitReached("max_call_depth", f"more than {self.lim.max_call_depth} nested function calls", line)
        self._tick(line)  # the call itself is a step (on top of its expression's)
        saved = (self.frame, self.flocals, self.fname)
        self.frame, self.flocals, self.fname = frame, flocals, name
        self.depth += 1
        try:
            self._block(f["body"])
            result = None
        except _Return as r:
            result = r.value
        except (_Break, _Continue):  # pragma: no cover (lang refuses break/continue outside a loop)
            raise ProgramError("bad_break", f"break/continue outside a loop in {name}()", self._line) from None
        finally:
            self.frame, self.flocals, self.fname = saved
            self.depth -= 1
        self._line = line
        return result

    def _api_call(self, name, args, kwargs, line):
        kind = API[name]["kind"]
        try:
            checked = check_args(name, args, kwargs)
        except ArgError as err:
            self.calls.append({"line": line, "name": name, "status": "bad_arg", "message": str(err)})
            raise ProgramError("bad_arg", f"{name}(): {err}", line) from None
        if kind == "terminal":
            self.calls.append({"line": line, "name": name, "status": "terminal", "args": checked})
            claim = {"done": "task_done", "help": f"help_requested:{checked.get('why')}",
                     "give_up": f"failed:{checked.get('why')}",
                     "not_found": f"not_found_{'in_inspected_scope' if checked.get('scope') == 'inspected_scope' else 'scope_incomplete'}"}[name]
            raise _Terminal(name, claim, checked.get("kind") if name == "help" else None)
        if kind == "skill" or kind == "observe":
            if self.skill_calls >= self.lim.max_skill_calls:
                raise LimitReached("program_max_skill_calls", f"more than {self.lim.max_skill_calls} skill calls "
                                                              "and observations in one program", line)
            self.skill_calls += 1
        else:
            if self.queries >= self.lim.max_queries:
                raise LimitReached("program_max_queries", f"more than {self.lim.max_queries} readings", line)
            self.queries += 1
        t = time.perf_counter()
        try:
            r = self.host.call(name, checked)
        except ArgError as err:  # the host re-checks; a refusal there is the program's bad argument
            self.calls.append({"line": line, "name": name, "status": "bad_arg", "message": str(err)})
            raise ProgramError("bad_arg", f"{name}(): {err}", line) from None
        finally:
            self._host_s += time.perf_counter() - t
        r = to_plain(r, self.lim)  # HostError propagates: a bench bug, not a program outcome
        self.calls.append({"line": line, "name": name, "status": "called",
                           "ok": r.get("ok") if isinstance(r, dict) and kind == "skill" else None})
        return r

    def _builtin(self, name, a, kw, line):
        def need(n_min, n_max=None):
            n_max = n_min if n_max is None else n_max
            if not n_min <= len(a) <= n_max:
                raise ProgramError("bad_arg", f"{name}() takes {n_min}..{n_max} arguments, got {len(a)}", line)

        def nums(xs, what="arguments"):
            if any(not _is_num(x) or type(x) is bool for x in xs):
                raise ProgramError("type_error", f"{name}() {what} must be numbers", line)
            return xs

        def vec(x):
            if type(x) is not list or not x or len(x) > 16:
                raise ProgramError("type_error", f"{name}() needs vectors (lists of 1 to 16 numbers)", line)
            return nums(x, "vector entries")

        if name == "len":
            need(1)
            if type(a[0]) not in (list, str, dict):
                raise ProgramError("type_error", f"len() of {_type(a[0])}", line)
            return len(a[0])
        if name in ("abs", "sqrt", "sin", "cos", "floor"):
            need(1)
            x = nums(a)[0]
            if name == "sqrt" and x < 0:
                raise ProgramError("math_error", "sqrt of a negative number", line)
            if name == "floor":
                return self._num(math.floor(x))
            return self._num({"abs": abs, "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos}[name](x))
        if name in ("atan2", "hypot"):
            need(2)
            x, y = nums(a)
            return self._num(math.atan2(x, y) if name == "atan2" else math.hypot(x, y))
        if name in ("min", "max"):
            xs = a[0] if len(a) == 1 and type(a[0]) is list else a
            if not xs:
                raise ProgramError("bad_arg", f"{name}() of an empty list", line)
            if not (all(_is_num(x) and type(x) is not bool for x in xs) or all(type(x) is str for x in xs)):
                raise ProgramError("type_error", f"{name}() needs numbers or strings", line)
            return min(xs) if name == "min" else max(xs)
        if name == "round":
            need(1, 2)
            if set(kw) - {"ndigits"}:
                raise ProgramError("bad_arg", "round(x, ndigits)", line)
            if kw and len(a) > 1:
                raise ProgramError("bad_arg", "round() got ndigits twice", line)
            nd = kw.get("ndigits", a[1] if len(a) > 1 else None)
            x = nums(a[:1])[0]
            if nd is None:  # Python 3, round(x) is an int (round(2.6) == 3, round(2.5) == 2)
                return self._num(round(x))
            if type(nd) is not int or not 0 <= nd <= 6:
                raise ProgramError("bad_arg", "round() digits must be an integer 0..6", line)
            return self._num(round(x, nd))
        if name == "int":
            need(1)
            x = nums(a)[0]
            return self._num(int(x))
        if name == "float":
            need(1)
            return float(nums(a)[0])
        if name == "str":
            need(1)
            x = a[0]
            if type(x) is str:
                return x
            if x is None or type(x) in (bool, int, float):  # Python's text: True, False, None, 3
                s = str(x)
            else:  # a list or dict: JSON text, built piece by piece and stopped past the limit
                s = json_text(x, self.lim.max_str_len)
            if s is None or len(s) > self.lim.max_str_len:
                raise ProgramError("value_too_large", f"str() would give a string longer than {self.lim.max_str_len}",
                                   line)
            return s
        if name == "range":
            need(1, 3)
            xs = a
            if any(type(x) is not int for x in xs):
                raise ProgramError("type_error", "range() needs integers", line)
            if len(xs) == 3 and xs[2] == 0:
                raise ProgramError("bad_arg", "range() step must not be 0", line)
            r = range(*xs)
            if len(r) > self.lim.max_list_len:
                raise ProgramError("value_too_large", f"range longer than {self.lim.max_list_len}", line)
            return list(r)
        if name == "sum":
            need(1)
            if type(a[0]) is not list:
                raise ProgramError("type_error", "sum() needs a list of numbers", line)
            return self._num(sum(nums(a[0], "list entries")))
        if name == "append":
            need(2)
            if type(a[0]) is not list:
                raise ProgramError("type_error", "append(list, value)", line)
            if len(a[0]) + 1 > self.lim.max_list_len:
                raise ProgramError("value_too_large", "list would be too long", line)
            return self._new(a[0] + [a[1]])
        if name == "set_key":
            need(3)
            if type(a[0]) is not dict or type(a[1]) is not str:
                raise ProgramError("type_error", "set_key(dict, string key, value)", line)
            d = dict(a[0])
            d[a[1]] = a[2]
            return self._new(d)
        if name == "get":
            if len(a) < 2 or not 2 <= len(a) + len(kw) <= 3 or (kw and set(kw) != {"default"}):
                raise ProgramError("bad_arg", "get(container, key, default=None)", line)
            default = kw.get("default", a[2] if len(a) > 2 else None)
            c, k = a[0], a[1]
            if type(c) is dict and type(k) is str:
                return c.get(k, default)
            if type(c) is list and type(k) is int:
                return c[k] if -len(c) <= k < len(c) else default
            if c is None:
                return default
            raise ProgramError("type_error", "get() needs a dict with a string key or a list with an integer", line)
        if name in ("keys", "values"):
            need(1)
            if type(a[0]) is not dict:
                raise ProgramError("type_error", f"{name}() needs a dict", line)
            if len(a[0]) > self.lim.max_list_len:  # 0.3: checked before the list is built
                raise ProgramError("value_too_large", f"{name}() would give a list longer than {self.lim.max_list_len}",
                                   line)
            return self._new(list(a[0].keys()) if name == "keys" else list(a[0].values()))
        if name in ("lower", "upper", "strip"):  # 0.3
            need(1)
            if type(a[0]) is not str:
                raise ProgramError("type_error", f"{name}() needs a string, got {_type(a[0])}", line)
            s = a[0]
            return self._new(s.lower() if name == "lower" else s.upper() if name == "upper" else s.strip())
        if name in ("startswith", "endswith"):  # 0.3
            need(2)
            s, p = a
            if type(s) is not str:
                raise ProgramError("type_error", f"{name}() needs a string, got {_type(s)}", line)
            if type(p) is list and all(type(x) is str for x in p):
                p = tuple(p)  # Python's tuple form: True if any of them matches
            elif type(p) is not str:
                raise ProgramError("type_error", f"{name}() takes a string or a list of strings", line)
            return s.startswith(p) if name == "startswith" else s.endswith(p)
        if name == "replace":  # 0.3: the result's length is computed before it is built
            need(3)
            if any(type(x) is not str for x in a):
                raise ProgramError("type_error", "replace(s, old, new) needs three strings", line)
            s, old, new = a
            size = len(s) + s.count(old) * (len(new) - len(old))
            if size > self.lim.max_str_len:
                raise ProgramError("value_too_large", f"replace() would give a string of {size} characters (limit "
                                                      f"{self.lim.max_str_len})", line)
            return s.replace(old, new)
        if name == "split":  # 0.3: the number of parts is bounded before the list is built
            need(1, 2)
            s = a[0]
            if type(s) is not str:
                raise ProgramError("type_error", f"split() needs a string, got {_type(s)}", line)
            if len(a) == 2:
                sep = a[1]
                if type(sep) is not str or not sep:
                    raise ProgramError("bad_arg", "split() separator must be a non-empty string", line)
                if s.count(sep) + 1 > self.lim.max_list_len:
                    raise ProgramError("value_too_large", f"split() would give more than {self.lim.max_list_len} parts",
                                       line)
                return self._new(s.split(sep))
            parts = s.split(None, self.lim.max_list_len)  # at most max_list_len + 1 parts
            if len(parts) > self.lim.max_list_len:
                raise ProgramError("value_too_large", f"split() would give more than {self.lim.max_list_len} parts",
                                   line)
            return self._new(parts)
        if name == "join":  # 0.3: join(sep, xs) = sep.join(xs); the length is computed before the text is built
            need(2)
            sep, xs = a
            if type(sep) is not str:
                raise ProgramError("type_error", f"join() separator must be a string, got {_type(sep)}", line)
            if type(xs) is not list or any(type(x) is not str for x in xs):
                raise ProgramError("type_error", "join() needs a list of strings", line)
            size = sum(len(x) for x in xs) + len(sep) * max(0, len(xs) - 1)
            if size > self.lim.max_str_len:
                raise ProgramError("value_too_large", f"join() would give a string of {size} characters (limit "
                                                      f"{self.lim.max_str_len})", line)
            return sep.join(xs)
        if name == "has":
            need(2)
            if type(a[0]) is not dict or type(a[1]) is not str:
                raise ProgramError("type_error", "has(dict, string key)", line)
            return a[1] in a[0]
        if name in ("vadd", "vsub", "dist"):
            need(2)
            u, v = vec(a[0]), vec(a[1])
            if len(u) != len(v):
                raise ProgramError("type_error", f"{name}() vectors of different lengths", line)
            if name == "dist":  # math.dist scales internally: no OverflowError
                return self._num(math.dist([float(x) for x in u], [float(y) for y in v]))
            return [self._num(x + y if name == "vadd" else x - y) for x, y in zip(u, v)]
        if name == "vscale":
            need(2)
            u = vec(a[0])
            s = nums([a[1]])[0]
            return [self._num(x * s) for x in u]
        if name == "vnorm":
            need(1)
            return self._num(math.sqrt(sum(x * x for x in vec(a[0]))))
        if name == "log":
            if len(self.log) >= self.lim.max_log_lines:
                raise LimitReached("max_log_lines", f"more than {self.lim.max_log_lines} log lines", line)
            if kw or len(a) > LOG_MAX_ARGS:
                raise ProgramError("bad_arg", f"log() takes 0 to {LOG_MAX_ARGS} values and no keyword", line)
            parts, n = [], 0
            for x in a:  # piece by piece, never more than LOG_LINE_CHARS (memory)
                if n >= LOG_LINE_CHARS:
                    break
                if type(x) is str:
                    piece = x[:LOG_LINE_CHARS - n]
                elif type(x) in (list, dict) and self._size(x) > LOG_DUMP_MAX_ITEMS:
                    piece = f"<{type(x).__name__} of {len(x)} entries>"
                else:  # never more JSON text than the line can still hold
                    piece = json_text(x, LOG_LINE_CHARS - n, cut=True)
                parts.append(piece)
                n += len(piece) + 1
            self.log.append(utf8_safe(" ".join(parts)[:LOG_LINE_CHARS]))
            return None
        raise ProgramError("unknown_function", f"{name!r} is not a builtin", line)  # pragma: no cover


def run_program(program, host, lim: ProgramLimits | None = None) -> dict:
    """In-process run (development). For agent runs prefer agent.sandbox.run_isolated."""
    return Interpreter(host, lim).run(program)
