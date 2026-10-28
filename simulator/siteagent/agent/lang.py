"""BotScript, the restricted language of robot programs: program text to a checked plain form.

A program is text in a small Python-looking language. It is never compiled or executed by Python: the text is parsed
by ast.parse (which only builds a syntax tree), every node is checked against an explicit whitelist, and the program
is translated into a small intermediate form of plain tuples that agent.interp runs.

Allowed: assignments (also += -= *= /=), calls, lst.append(e), if/elif/else, while, for, break, continue, pass,
top-level def with constant defaults and return; numbers, strings, True/False/None, lists, dicts with str keys,
arithmetic (** bounded by lang.power), comparisons (including "is None"), and/or/not, conditional expressions,
indexing and slicing, calls of a whitelisted name or of a function defined in the program, and the methods of METHODS.

Refused (nothing runs): import, attribute access other than METHODS calls (so no dunder can be reached), lambda,
class, nested def, recursion, with/try/raise/assert/del/global/nonlocal, comprehensions, f-strings, set literals, star
arguments, walrus, match, names starting with "_", a function used as a value, bytes or complex constants, and
programs that are too long, too large, too deeply nested or define too many functions.

All language errors of a program are reported at once (Refused.errors: line, construct, message and a one-line fix
hint each), so a correction can fix them together. Certain argument errors of help() and gestures outside the S5
skill profile are refused before the run; code that can never run after a terminal on every path is reported.
"""
from __future__ import annotations

import ast
import io
import math
import re
import tokenize
from collections import deque

from .api import API, HELP_KINDS, MAX_RAW_TEXT_CHARS, ArgError, P, _check, clean_text
from .limits import ProgramLimits

LANGUAGE = "botscript-0.3"
BUILTINS = ("len", "abs", "min", "max", "round", "sqrt", "sin", "cos", "atan2", "hypot", "floor", "int", "float",
            "str", "range", "sum", "append", "set_key", "get", "keys", "has", "vadd", "vsub", "vscale", "vnorm",
            "dist", "log",
            # 0.3: also the methods' builtins, callable as functions with the receiver first
            "values", "startswith", "endswith", "replace", "lower", "upper", "strip", "split", "join")
BUILTIN_KEYWORDS = {"get": ("default",), "round": ("ndigits",)}
CALLABLE = set(API) | set(BUILTINS)
# 0.3: the ONLY methods. name -> (receiver type, min positional arguments, max positional arguments). A method call
# compiles to ("method", line, name, receiver, args); the interpreter checks the receiver type, then runs the
# builtin of the same name with the receiver first. No other attribute is ever looked up.
METHODS = {"get": ("dict", 1, 2), "keys": ("dict", 0, 0), "values": ("dict", 0, 0),
           "startswith": ("str", 1, 1), "endswith": ("str", 1, 1), "replace": ("str", 2, 2),
           "lower": ("str", 0, 0), "upper": ("str", 0, 0), "strip": ("str", 0, 0), "split": ("str", 0, 1),
           "join": ("str", 1, 1)}
METHOD_FORMS = {"get": "d.get(key) or d.get(key, default)", "keys": "d.keys()", "values": "d.values()",
                "startswith": "s.startswith(p) (p a string or a list of strings)",
                "endswith": "s.endswith(p) (p a string or a list of strings)", "replace": "s.replace(old, new)",
                "lower": "s.lower()", "upper": "s.upper()", "strip": "s.strip()", "split": "s.split() or s.split(sep)",
                "join": "sep.join(list_of_strings)", "append": "lst.append(x) on its own line"}
# 0.3: a defined function may not carry a Python builtin name either (no def getattr / type / eval ...).
PYTHON_BUILTIN_NAMES = frozenset((
    "abs", "aiter", "all", "anext", "any", "ascii", "bin", "bool", "breakpoint", "bytearray", "bytes", "callable",
    "chr", "classmethod", "compile", "complex", "copyright", "credits", "delattr", "dict", "dir", "divmod",
    "enumerate", "eval", "exec", "exit", "filter", "float", "format", "frozenset", "getattr", "globals", "hasattr",
    "hash", "help", "hex", "id", "input", "int", "isinstance", "issubclass", "iter", "len", "license", "list",
    "locals", "map", "max", "memoryview", "min", "next", "object", "oct", "open", "ord", "pow", "print", "property",
    "quit", "range", "repr", "reversed", "round", "set", "setattr", "slice", "sorted", "staticmethod", "str", "sum",
    "super", "tuple", "type", "vars", "zip"))
MAX_POWER_EXPONENT = 8
MAX_REPORTED_ERRORS = 25
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_SURROGATE = re.compile("[\ud800-\udfff]")
_BIN = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.FloorDiv: "//", ast.Mod: "%", ast.Pow: "**"}
_UN = {ast.USub: "neg", ast.UAdd: "pos", ast.Not: "not"}
_CMP = {ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.In: "in",
        ast.NotIn: "not in"}
_FATAL = ("too_many_nodes", "too_deep", "too_long")

# One-line fixes (shown to the model in the observable feedback). Generic: no kitchen, seed or variant.
# 0.3: the methods of METHODS are allowed (a method name without a call gets METHOD_FORMS as its hint).
_METHODS_LIST = ("allowed methods: d.get d.keys d.values s.startswith s.endswith s.replace s.lower s.upper s.strip "
                 "s.split sep.join, and lst.append(x) as a statement")
_METHOD_HINTS = {
    "append": "write lst.append(x) on its own line (lst a plain name), or lst = append(lst, x)",
    "items": "loop over keys: for k in d.keys(): v = d[k]",
    "sort": "no sorting: keep the best item while looping",
    "index": "loop with for i in range(len(xs)) and compare xs[i]",
    "copy": "values are never shared: just reuse the name",
    "update": "write d = set_key(d, \"key\", value)",
    "setdefault": "write v = d.get(k, default) (and d = set_key(d, k, v) to store it)",
    "pop": "rebuild the list without the item, or keep an index",
    "remove": "rebuild the list without the item",
    "insert": "write lst = lst[0:i] + [x] + lst[i:]",
    "extend": "write lst = lst + other",
    "lstrip": "use s.strip() (both sides)",
    "rstrip": "use s.strip() (both sides)",
    "find": "use p in s, or s.startswith(p)",
    "count": "count in a for loop",
    "format": "build text with + and str(x)",
    "title": "use s.upper() or s.lower()",
    "capitalize": "use s.upper() or s.lower()",
}
_FUNCTION_HINTS = {
    "print": "use log(...)",
    "sorted": "no sorting: keep the best item while looping",
    "isinstance": "no type checks: compare with None, or check len(x)",
    "type": "no type checks: compare with None, or check len(x)",
    "enumerate": "write for i in range(len(xs)): x = xs[i]",
    "zip": "write for i in range(len(xs)): a = xs[i]; b = ys[i]",
    "list": "write a list literal [ ... ] or build it with append()",
    "dict": "write a dict literal {\"key\": value} or build it with set_key()",
    "any": "use a for loop and a flag variable",
    "all": "use a for loop and a flag variable",
    "math": "sqrt sin cos atan2 hypot floor are builtins: call them by name",
    "pow": "write x ** n (n between -8 and 8)",
    "reversed": "write for i in range(len(xs) - 1, -1, -1): x = xs[i]",
    "bool": "compare explicitly (== True, != 0, len(x) > 0)",
    "getattr": "no attribute access: use d[\"key\"] or d.get(\"key\", default)",
    "hasattr": "use has(d, \"key\") or \"key\" in d",
    # Skill API 0.6: no free arm motion; each gesture moves the arm itself
    "move_to": ("move_to does not exist: look_at_region or look_down_at to look, pick to grasp, scrub after the pick "
                "(it brings the held sponge onto target_pos itself)"),
}
# Skill API 0.7: gestures outside the S5 skill profile; the hint names the profile's gestures
_S5_PROFILE_HINT = ("the S5 profile's gestures: look_at_region, look_down_at, look_pose, open_drawer, close_drawer, pick, "
                    "scrub, open_gripper, close_gripper, go_home, calibrate")
_FUNCTION_HINTS.update({n: _S5_PROFILE_HINT for n in ("place", "turn_over", "press", "spray")})
_UNREACHABLE_HINT = "move that code before it, or put the terminal inside an if branch"
# help() needs an explicit kind (api.HELP_KINDS), refused before running otherwise
_HELP_KIND_HINT = ('write help("why", kind="...") with kind one of ' + ", ".join(HELP_KINDS)
                   + ' (search_incomplete: target not found, plausible places remain)')
_DEF_TOP_HINT = "define functions at the top level of the program (not inside a def, if, while or for), then call them"
_STMT_HINTS = {
    "FunctionDef": ("def", _DEF_TOP_HINT),
    "AsyncFunctionDef": ("async def", "write a plain def name(a, b): at the top level"),
    "ClassDef": ("class", "no class: use dicts {\"key\": value} and top-level def functions"),
    "Import": ("import", "no import: only the API functions and the builtins listed exist"),
    "ImportFrom": ("from ... import", "no import: only the API functions and the builtins listed exist"),
    "Try": ("try", "no try: check r[\"ok\"], or use d.get(\"key\", default)"),
    "TryStar": ("try", "no try: check r[\"ok\"], or use d.get(\"key\", default)"),
    "With": ("with", "no with: write the calls directly"),
    "AsyncWith": ("with", "no with: write the calls directly"),
    "Raise": ("raise", "end with give_up(\"why\") or help(\"why\", kind=\"cannot_continue\")"),  # a kind
    "Assert": ("assert", "use if ...: give_up(\"why\")"),
    "Delete": ("del", "no del: assign a new value"),
    "Global": ("global", "a function cannot change a global: pass it as a parameter and return the new value"),
    "Nonlocal": ("nonlocal", "no nonlocal (functions are only at the top level): return the new value instead"),
    "Match": ("match", "use if / elif / else"),
    "AsyncFor": ("async for", "use for x in xs:"),
    "Return": ("return", "return is allowed only inside a def; end the program with done(), help(why, kind), "
                         "give_up(why) or not_found(scope)"),
}
_EXPR_HINTS = {
    "ListComp": ("list comprehension", "write a for loop with lst.append(x)"),
    "SetComp": ("set comprehension", "write a for loop with lst.append(x)"),
    "DictComp": ("dict comprehension", "write a for loop with d = set_key(d, k, v)"),
    "GeneratorExp": ("generator expression", "write a for loop with lst.append(x)"),
    "Lambda": ("lambda", "no lambda: write a def at the top level, or the expression inline"),
    "JoinedStr": ("f-string", "build text with + and str(x)"),
    "Set": ("set literal {a, b}", "use a list [a, b]"),
    "NamedExpr": ("walrus :=", "assign on its own line first"),
    "Await": ("await", "not available"),
    "Yield": ("yield", "no generators: build a list and return it"),
    "YieldFrom": ("yield", "no generators: build a list and return it"),
    "Starred": ("*expression", "pass the items one by one"),
}
_POWER_HINT = "x ** n needs numbers, n between -8 and 8, and a result within the number limits (or use x * x, sqrt())"


class PowerError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def power(a, b, max_int_abs: int):
    """a ** b under the 0.3 rule (a and b int or float, not bool: checked by the caller). Used by the checker on
    two constants and by the interpreter. Raises PowerError(code, message); never returns a complex number."""
    if abs(b) > MAX_POWER_EXPONENT:  # the exponent is shown only when it is small (no int to text)
        shown = repr(b) if type(b) is float or abs(b) <= 10 ** 6 else "too large"
        raise PowerError("bad_power", f"exponent {shown} is outside -{MAX_POWER_EXPONENT}..{MAX_POWER_EXPONENT}")
    if a < 0 and type(b) is float and not b.is_integer():
        raise PowerError("math_error", "a negative number to a non-integer power")
    if a == 0 and b < 0:
        raise PowerError("division_by_zero", "zero to a negative power")
    try:
        r = a ** b  # |b| <= 8 and |a| <= max_int_abs (or a finite float): a small computation
    except OverflowError:
        raise PowerError("non_finite", "number too large") from None
    except ZeroDivisionError:  # pragma: no cover (0 ** negative is checked above)
        raise PowerError("division_by_zero", "zero to a negative power") from None
    if type(r) is int:
        if abs(r) > max_int_abs:
            raise PowerError("value_too_large", "integer too large")
        return r
    if type(r) is not float or not math.isfinite(r):
        raise PowerError("non_finite", "number too large or undefined")
    return r


class Refused(Exception):
    """The program is refused before it runs. code, message, line: the FIRST error (0.1 fields, kept);
    errors: every error found (0.2), each {line, code, construct, message, hint}; n_errors: their count."""

    def __init__(self, code: str, message: str, line: int = 0, errors: list | None = None, n_errors: int | None = None,
                 warnings: list | None = None):
        super().__init__(message)
        self.code, self.message, self.line = code, message, line
        self.errors = list(errors) if errors else [{"line": line, "code": code, "construct": "", "message": message,
                                                     "hint": ""}]
        self.n_errors = len(self.errors) if n_errors is None else int(n_errors)
        self.warnings = list(warnings or [])

    def as_dict(self) -> dict:
        d = {"kind": "refused", "code": self.code, "line": self.line, "message": self.message,
             "n_errors": self.n_errors, "errors": [dict(e) for e in self.errors]}
        if self.warnings:
            d["warnings"] = [dict(w) for w in self.warnings]
        return d


class _Fatal(Exception):
    def __init__(self, code, construct, message, hint, line):
        super().__init__(message)
        self.err = {"line": line, "code": code, "construct": construct, "message": message, "hint": hint}


_NOCONST = object()


def _const_value(node, max_int_abs: int = 10 ** 15):
    """The value of a literal argument (number, string, bool, None, -number, list of those), else _NOCONST. Review
    an integer above max_int_abs is not a constant here (it is refused as a literal anyway, and such an int must
    never reach a message: Python refuses to turn an int of more than 4300 digits into text)."""
    if isinstance(node, ast.Constant) and (node.value is None or isinstance(node.value, (bool, int, float, str))):
        v = node.value
        return _NOCONST if type(v) is int and abs(v) > max_int_abs else v
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)) and isinstance(node.operand, ast.Constant) \
            and isinstance(node.operand.value, (int, float)) and not isinstance(node.operand.value, bool):
        v = node.operand.value
        if type(v) is int and abs(v) > max_int_abs:
            return _NOCONST
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, (ast.List, ast.Tuple)) and len(node.elts) <= 16:
        vals = [_const_value(x, max_int_abs) for x in node.elts]
        return _NOCONST if any(v is _NOCONST for v in vals) else vals
    return _NOCONST


def _str_pieces(node) -> list[str] | None:
    """Constant string pieces of a + expression whose result, if it is text at all, contains them all."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return (_str_pieces(node.left) or []) + (_str_pieces(node.right) or [])
    return None


def _is_number(v) -> bool:
    return type(v) in (int, float)


def _assigned_names(body) -> set:
    """Names a function body assigns (0.3, Python's rule for locals): assignment, augmented assignment and for
    targets, anywhere in the body. ast.walk is iterative (no recursion)."""
    out = set()
    for s in body:
        for n in ast.walk(s):
            targets = (n.targets if isinstance(n, ast.Assign) else [n.target] if isinstance(n, (ast.AugAssign, ast.For))
                       else [])
            out.update(t.id for t in targets if isinstance(t, ast.Name))
    return out


class _Func:
    """A top-level def known to the checker (0.3): its parameters, how many have defaults, its node."""

    def __init__(self, node):
        self.node, self.line = node, node.lineno
        self.params = [a.arg for a in node.args.args]
        self.n_defaults = len(node.args.defaults)

    def signature(self) -> str:
        req = len(self.params) - self.n_defaults
        parts = [p if i < req else f"{p}=..." for i, p in enumerate(self.params)]
        return f"{self.node.name}({', '.join(parts)})"


class _Ctx:
    """The function whose body is being checked (0.3). name None: the body of a refused def (checked for the other
    errors only, no call-graph edge)."""

    def __init__(self, name, params, assigned):
        self.name, self.params, self.assigned = name, set(params), set(assigned)


_NO_DESCENT = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
_TERMINALS = tuple(n for n, e in API.items() if e["kind"] == "terminal")  # done, help, give_up, not_found


def _inert(s, top: bool) -> bool:
    """Dead code that could do nothing if it ran (not flagged as unreachable): a statement with no call at all
    (pass, break, continue, return, return x, x = 1, a string line, an if of such lines), or a top-level def (hoisted,
    callable from before). Every gesture, observation, reading, log or terminal is a call. Generated programs
    often have such lines after a help or an if/else of terminals (help(...) then return); they are harmless."""
    if top and isinstance(s, ast.FunctionDef):
        return True
    return not any(isinstance(n, ast.Call) for n in _walk_scope(s))


def _walk_scope(node):
    """Node and its descendants, never entering a nested def, class or lambda (iterative)."""
    stack = [node]
    while stack:
        x = stack.pop()
        yield x
        if x is node or not isinstance(x, _NO_DESCENT):
            stack.extend(ast.iter_child_nodes(x))
_KEEPING_ARGS = {"append": (1,), "set_key": (2,), "get": (2,)}  # builtin -> argument positions it stores or gives back
_KEEPING_KWS = {"get": ("default",)}
_GIVING_BACK = ("get", "min", "max")  # builtins whose result may be one of the objects given to them


def _scope_nodes(stmts) -> list:
    """(node, parent) pairs of one scope's statements, never entering a nested def, class or lambda (iterative)."""
    out, stack = [], [(s, None) for s in reversed(stmts)]
    while stack:
        node, parent = stack.pop()
        out.append((node, parent))
        if not isinstance(node, _NO_DESCENT):
            stack.extend((ch, node) for ch in reversed(list(ast.iter_child_nodes(node))))
    return out


class _Scope:
    def __init__(self, name, params, nodes):
        self.name, self.params, self.nodes = name, list(params), nodes
        self.locals = set(params)
        self.bad_bind: dict = {}  # name -> (line, why): a binding that may give it a list held elsewhere
        self.unsafe: dict = {}  # name -> (line, why): a read that may keep or give on the list itself
        self.returned: set = set()  # local names returned (functions)
        self.listy: set = set()  # names bound somewhere to a list literal or an append(...) result


class _Sharing:
    """may 'n.append(x)' (or 'n += [..]') run with Python's meaning? In Python it changes the list
    object, so every other name, list or dict holding that same object sees the change; in BotScript values are never
    shared, so only the name changes. The two agree when the list in n is certainly held by n alone:
    - every binding of n in its scope makes a NEW value: a literal, an expression that builds a value (+, a slice,
      builtins and API calls that return new data), or a call of a defined function whose returns are all new
      lists; never a copy of another name, an element x[k] / d["k"], get / min / max (they give back an object they
      were given), a parameter or a for-loop variable (both receive an object held elsewhere);
    - every read of the bare name n is a read that cannot keep or give on the object: an index, a slice, len(), an
      operator, a comparison, a test, a loop over it, a method call on it, an argument of a builtin or API function
      that neither stores nor gives it back, an argument of a defined function whose parameter is itself such a
      parameter, 'return n' for a local of a function. Anything else (b = n, [n], {"k": n}, set_key(d, k, n),
      append(lst, n), lst.append(n), n or [], a function that returns or stores its parameter, 'return n' of a global)
      makes it shared.
    Flow-insensitive (the whole scope counts, before and after the line) and conservative: a refusal comes with the
    form that is always right, n = append(n, x). The scope of a name follows Python: the top level, or the body of the
    function that assigns it (a global read inside a function counts for the top level)."""

    def __init__(self, top_stmts, defs: dict, refine: bool):
        self.defs = defs
        self.top = _Scope(None, (), _scope_nodes([s for s in top_stmts if not isinstance(s, _NO_DESCENT)]))
        self.scopes = {}
        for name, d in defs.items():
            sc = _Scope(name, [a.arg for a in d.args.args], _scope_nodes(d.body))
            for node, _ in sc.nodes:
                targets = (node.targets if isinstance(node, ast.Assign) else [node.target]
                           if isinstance(node, (ast.AugAssign, ast.For)) else [])
                sc.locals.update(t.id for t in targets if isinstance(t, ast.Name))
            self.scopes[name] = sc
        self.fresh_return = {n: refine for n in defs}
        self.param_safe = {n: [refine] * len(sc.params) for n, sc in self.scopes.items()}
        rounds = len(defs) + sum(len(sc.params) for sc in self.scopes.values()) + 2
        for _ in range(rounds if refine else 1):  # monotone (True -> False only): stops within `rounds` passes
            self._pass()
            fr = {n: self._returns_fresh(n) for n in defs}
            ps = {n: [p not in sc.unsafe and p not in sc.returned for p in sc.params] for n, sc in self.scopes.items()}
            if not refine or (fr == self.fresh_return and ps == self.param_safe):
                return
            self.fresh_return, self.param_safe = fr, ps
        self.fresh_return = {n: False for n in defs}  # pragma: no cover (did not settle: the conservative answer)
        self.param_safe = {n: [False] * len(sc.params) for n, sc in self.scopes.items()}
        self._pass()

    # ---------------------------------------------------------------- queries
    def problem(self, fname, n):
        """None if n.append(x) has Python's meaning in this scope (fname None: the top level), else (line, why)."""
        sc = self.top if fname is None else self.scopes.get(fname)
        if sc is None:
            return None
        return sc.bad_bind.get(n) or sc.unsafe.get(n)

    def listy(self, fname, n) -> bool:
        sc = self.top if fname is None else self.scopes.get(fname)
        return sc is not None and n in sc.listy

    # ---------------------------------------------------------------- analysis
    def _pass(self):
        for sc in [self.top] + list(self.scopes.values()):
            sc.bad_bind, sc.unsafe, sc.returned, sc.listy = {}, {}, set(), set()
        for sc in [self.top] + list(self.scopes.values()):
            if sc.name is not None:
                for p in sc.params:
                    sc.bad_bind.setdefault(p, (self.defs[sc.name].lineno, "a parameter receives its caller's value"))
            parents = {id(node): parent for node, parent in sc.nodes}
            for node, parent in sc.nodes:
                if isinstance(node, ast.Assign):
                    for t in node.targets:
                        if isinstance(t, ast.Name):
                            if not self._fresh(node.value, t.id):
                                sc.bad_bind.setdefault(t.id, (node.lineno, "it is assigned a value held elsewhere"))
                            if self._is_listy(node.value):
                                sc.listy.add(t.id)
                elif isinstance(node, ast.For) and isinstance(node.target, ast.Name):
                    sc.bad_bind.setdefault(node.target.id, (node.lineno, "a for-loop variable holds an element"))
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    why = self._unsafe_read(node, parent, parents, sc)
                    if why:
                        owner = sc if sc.name is None or node.id in sc.locals else self.top
                        owner.unsafe.setdefault(node.id, (node.lineno, why))

    @staticmethod
    def _is_listy(e) -> bool:
        if isinstance(e, (ast.List, ast.Tuple)):
            return True
        if isinstance(e, ast.Call) and isinstance(e.func, ast.Name) and e.func.id == "append":
            return True
        return isinstance(e, ast.BinOp) and isinstance(e.op, ast.Add) and (
            isinstance(e.left, (ast.List, ast.Tuple)) or isinstance(e.right, (ast.List, ast.Tuple)))

    def _fresh(self, e, self_name=None, local_scope=None) -> bool:
        """True if the value of e is certainly a NEW object (or an immutable one). local_scope: a function whose
        unshared locals count as new (for its returns)."""
        stack = [e]
        while stack:
            x = stack.pop()
            if isinstance(x, ast.Name):
                if x.id == self_name:
                    continue
                sc = local_scope
                if sc is None or x.id not in sc.locals or x.id in sc.params or x.id in sc.bad_bind or x.id in sc.unsafe:
                    return False
            elif isinstance(x, ast.Subscript):
                if not isinstance(x.slice, ast.Slice):
                    return False
            elif isinstance(x, ast.IfExp):
                stack += [x.body, x.orelse]
            elif isinstance(x, ast.BoolOp):
                stack += x.values
            elif isinstance(x, ast.Call):
                f = x.func
                if isinstance(f, ast.Name):
                    if f.id in self.defs:
                        if not self.fresh_return.get(f.id, False):
                            return False
                    elif f.id in _GIVING_BACK or f.id not in CALLABLE:
                        return False
                elif not (isinstance(f, ast.Attribute) and f.attr in METHODS and f.attr != "get"):
                    return False
            elif isinstance(x, (ast.Starred, ast.NamedExpr, ast.Await, ast.Yield, ast.YieldFrom, ast.Attribute,
                                ast.Lambda)):
                return False
        return True

    def _returns_fresh(self, fname) -> bool:
        sc = self.scopes[fname]
        return all(self._fresh(node.value, None, sc) for node, _ in sc.nodes
                   if isinstance(node, ast.Return) and node.value is not None)

    def _unsafe_read(self, node, p, parents, sc):
        """Why this read of a bare name may keep or give on its object; None for a read that cannot."""
        if p is None or isinstance(p, (ast.Attribute, ast.Subscript, ast.Slice, ast.BinOp, ast.Compare, ast.UnaryOp,
                                       ast.Expr, ast.FormattedValue, ast.AugAssign, ast.Assert)):
            return None  # method receiver, index, operator, comparison, a += n (copies the items), refused forms
        if isinstance(p, (ast.If, ast.While)) or (isinstance(p, ast.IfExp) and node is p.test):
            return None
        if isinstance(p, ast.For) and node is p.iter:
            return None
        if isinstance(p, ast.BoolOp):  # "a and b" only tested for its truth is a test
            child, q = p, parents.get(id(p))
            while isinstance(q, ast.BoolOp):
                child, q = q, parents.get(id(q))
            if (isinstance(q, (ast.If, ast.While, ast.IfExp)) and child is q.test) or isinstance(
                    q, (ast.UnaryOp, ast.Compare, ast.Assert)):
                return None
            return "it is given on by and/or"
        if isinstance(p, ast.Return):
            if sc.name is not None and node.id in sc.locals:
                sc.returned.add(node.id)
                return None
            return "a function returns it"
        if isinstance(p, ast.keyword):
            return self._arg_unsafe(parents.get(id(p)), None, p.arg)
        if isinstance(p, ast.Call):
            if node is p.func:
                return None
            idx = next((i for i, a in enumerate(p.args) if a is node), None)
            return self._arg_unsafe(p, idx, None)
        if isinstance(p, ast.Assign) and all(isinstance(t, ast.Name) and t.id == node.id for t in p.targets):
            return None  # n = n
        if isinstance(p, ast.Assign):
            return "it is copied to another name"
        if isinstance(p, (ast.List, ast.Tuple, ast.Dict, ast.Set)):
            return "it is stored in a list or dict"
        return "it is given on"

    def _arg_unsafe(self, call, idx, kw):
        if not isinstance(call, ast.Call):
            return "it is given on"
        f = call.func
        if isinstance(f, ast.Name):
            fn = f.id
            if fn in self.defs:
                params = self.scopes[fn].params
                i = idx if kw is None else (params.index(kw) if kw in params else None)
                if i is None or i >= len(params) or not self.param_safe[fn][i]:
                    return f"it is given to {fn}(), which may keep or return it"
                return None
            if fn in API:
                return None  # copied into the request, never kept
            if fn in ("min", "max"):  # min(xs) gives back an element of xs; min(a, b) gives back a or b itself
                if kw is not None:
                    return f"{fn}() may give it back" if kw == "default" else None
                return f"{fn}() may give it back" if len(call.args) > 1 else None
            if (kw is None and idx in _KEEPING_ARGS.get(fn, ())) or (kw is not None and kw in _KEEPING_KWS.get(fn, ())):
                return f"{fn}() stores it or gives it back"
            return None if fn in CALLABLE else f"it is given to {fn}()"
        if isinstance(f, ast.Attribute):
            if f.attr == "append":
                return "it is stored by .append()"
            if f.attr == "get" and idx == 1:
                return ".get() may give it back"
            return None
        return "it is given to a call"


class _Checker:
    def __init__(self, lim: ProgramLimits):
        self.lim = lim
        self.n_nodes = 0
        self.errors: list[dict] = []
        self.warnings: list[dict] = []
        self._seen: set = set()
        self.funcs: dict[str, _Func] = {}  # 0.3: the top-level defs, known before any body is checked
        self.fir: dict[str, dict] = {}  # 0.3: their compiled form
        self.graph: dict[str, dict[str, int]] = {}  # 0.3: caller -> {callee: first call line}
        self.ctx: _Ctx | None = None
        self.loop_depth = 0
        self.sharing: _Sharing | None = None  # computed when the program has an append or a +=
        self._ends_memo: dict[str, bool | None] = {}  # defined functions that always end the program

    # ------------------------------------------------------------ errors and warnings
    def err(self, code: str, construct: str, message: str, hint: str, line: int, warning: bool = False):
        key = (line, code, construct, message, warning)
        if key in self._seen:
            return
        self._seen.add(key)
        (self.warnings if warning else self.errors).append({
            "line": int(line or 0), "code": code, "construct": construct[:60], "message": message[:200],
            "hint": hint[:160]})

    def warn(self, code: str, construct: str, message: str, hint: str, line: int):
        """A call that will fail if it is reached (0.2, point 4): reported, the program still runs."""
        self.err(code, construct, message + " (fails if this line is reached)", hint, line, warning=True)

    def _count(self, node, depth):
        self.n_nodes += 1
        if self.n_nodes > self.lim.max_ast_nodes:
            raise _Fatal("too_many_nodes", "program size", f"the program has more than {self.lim.max_ast_nodes} syntax "
                         "nodes", "write a shorter program", getattr(node, "lineno", 0))
        if depth > self.lim.max_nesting_depth:
            raise _Fatal("too_deep", "nesting", f"nesting deeper than {self.lim.max_nesting_depth}",
                         "flatten the nesting", getattr(node, "lineno", 0))

    def _name(self, s: str, line: int) -> str:
        if not _NAME.match(s):
            self.err("bad_name", f"name {s}", f"name {s!r} is not allowed (letters, digits, _; no leading _)",
                     "start names with a letter, e.g. i or unused", line)
        return s

    def _shared(self, name: str, line: int, construct: str, fix: str) -> bool:
        """an error (bad_append) and True when the list in name may be held elsewhere too."""
        if self.sharing is None or (self.ctx is not None and self.ctx.name is None):
            return False  # nothing to change, or the body of a def that is refused anyway
        pb = self.sharing.problem(None if self.ctx is None else self.ctx.name, name)
        if pb is None:
            return False
        where, why = pb
        self.err("bad_append", f"{construct} on a shared list"[:60], f"{name!r} may hold a list also held elsewhere "
                 f"({why}, line {where}): in Python {construct} would change it there too; BotScript never shares "
                 "values", f'{fix}; a copy such as b = a or x = d["k"] is a separate value: store it back with set_key',
                 line)
        return True

    def _target(self, s: str, line: int) -> str:
        """A name that receives a value (assignment, for, parameter): never a defined function's name (0.3)."""
        if s in self.funcs:
            self.err("function_as_value", f"{s} = ..."[:60], f"{s!r} is a function defined in the program: it cannot "
                     "receive a value", "choose another variable name", line)
        return self._name(s, line)

    # ------------------------------------------------------------ 0.3: the program and its functions
    def program(self, body) -> list:
        defs = [s for s in body if isinstance(s, ast.FunctionDef)]
        for s in defs:
            name, ln = s.name, s.lineno
            if not _NAME.match(name):
                self._name(name, ln)
            elif name in CALLABLE or name in PYTHON_BUILTIN_NAMES:
                kind = "an API or BotScript builtin" if name in CALLABLE else "a Python builtin"
                self.err("bad_def", f"def {name}"[:60], f"{name!r} is {kind} function name: a defined function needs "
                         "its own name", f"call it e.g. my_{name}", ln)
            elif name in self.funcs:
                self.err("bad_def", f"def {name}"[:60], f"function {name!r} is defined twice (first on line "
                         f"{self.funcs[name].line})", "give each function its own name", ln)
            else:
                self.funcs[name] = _Func(s)
        if len(defs) > self.lim.max_functions:
            self.err("too_many_functions", "def", f"more than {self.lim.max_functions} functions "
                     f"({len(defs)})", "merge small functions, or write the steps inline",
                     defs[self.lim.max_functions].lineno)
        if any(isinstance(n, ast.AugAssign) or (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                                                and n.func.attr == "append") for s in body for n in ast.walk(s)):
            self.sharing = _Sharing(body, {n: f.node for n, f in self.funcs.items()},
                                    refine=len(defs) <= self.lim.max_functions)
        out = []
        for s in body:
            if isinstance(s, ast.FunctionDef):
                self._def(s, 0)
                out.append(("pass", s.lineno))  # hoisted: the def line itself does nothing when it is reached
            else:
                out.append(self.stmt(s, 0))
        self._recursion()
        self._unreachable(body, top=True)  # code after a statement that always ends the program
        for f in self.funcs.values():
            self._unreachable(f.node.body)
        return out

    # ------------------------------------------------------------ unreachable code
    def _ends_call(self, e) -> tuple[str, bool] | None:
        """What a call certainly ends the program with, and whether that end is a help() call: ('done()', False) for a
        terminal (('help()', True) for help), ('f() (...)', all_help) for a defined
        function whose body always ends it; None otherwise."""
        if isinstance(e, ast.Call) and isinstance(e.func, ast.Name):
            n = e.func.id
            if n in _TERMINALS:
                return f"{n}()", n == "help"
            if n in self.funcs:
                end = self._func_end(n)
                if end is not None:
                    return f"{n}() (that function always ends the program)", end
        return None

    def _stmt_end(self, s) -> tuple[str, bool] | None:
        """Why statement s ends the program on every path, and whether every such path ends on a help() call; None when
        it may not end it (conservative)."""
        if isinstance(s, (ast.Expr, ast.Assign, ast.Return)) and s.value is not None:
            return self._ends_call(s.value)
        if isinstance(s, ast.If) and s.orelse:
            body, other = self._block_end(s.body), self._block_end(s.orelse)
            if body and other:
                if body[1] and other[1]:
                    return "the if/else (every branch ends with help)", True
                return "the if/else (every branch ends with done, help, give_up or not_found)", False
        return None

    def _block_end(self, stmts) -> tuple[str, bool] | None:
        for s in stmts:
            end = self._stmt_end(s)
            if end:
                return end
            if any(isinstance(n, (ast.Return, ast.Break, ast.Continue)) for n in _walk_scope(s)):
                return None  # it may leave the block before
        return None

    def _func_end(self, name: str) -> bool | None:
        """None when the defined function may return; else whether every path of it ends on a help() call."""
        memo = self._ends_memo
        if name not in memo:
            memo[name] = None  # a cycle (refused as recursion anyway) never counts as ending
            end = self._block_end(self.funcs[name].node.body)
            memo[name] = None if end is None else end[1]
        return memo[name]

    def _unreachable(self, stmts, top: bool = False):
        """One unreachable_code entry per block: the first statement that runs code after a statement ending the program
        on every path. Sub-blocks up to that statement are checked too. An ERROR (the program is refused), except
        when every path of that statement ends on a help() call: then a WARNING, the program runs and the reached
        help keeps its effect (final, or reviewed under the loop's conditions)."""
        for i, s in enumerate(stmts):
            subs = [s.body, s.orelse] if isinstance(s, ast.If) else [s.body] if isinstance(s, (ast.While, ast.For)) \
                else []
            for sub in subs:
                self._unreachable(sub)
            end = self._stmt_end(s)
            if not end:
                continue
            rest = [x for x in stmts[i + 1:] if not _inert(x, top)]
            if rest:
                why, all_help = end
                ln = getattr(rest[0], "lineno", 0)
                if all_help:
                    self.err("unreachable_code", "code after a help", f"line {ln} can never run: {why} on line "
                             f"{s.lineno} ends the program with help() on every path (reported only: the help keeps its "
                             "effect)", _UNREACHABLE_HINT, ln, warning=True)
                else:
                    self.err("unreachable_code", "code after a terminal", f"line {ln} can never run: {why} on line "
                             f"{s.lineno} ends the program on every path before it", _UNREACHABLE_HINT, ln)
            return

    def _def(self, s, depth, refused: bool = False):
        """A def statement. refused: a def that is already refused (nested, in a block, async): its body is checked
        for the other errors only."""
        if not refused:  # a refused def was already counted by stmt()
            self._count(s, depth)
        ln = s.lineno
        info = self.funcs.get(s.name)
        own = not refused and info is not None and info.node is s
        a = s.args
        if s.decorator_list:
            self.err("forbidden_syntax", "decorator @...", "decorators are not allowed", "write a plain def", ln)
        if getattr(s, "type_params", None):
            self.err("forbidden_syntax", f"def {s.name}[...]"[:60], "type parameters are not allowed",
                     f"write def {s.name}(...):", ln)
        if a.posonlyargs:
            self.err("forbidden_syntax", "/ in def", "the positional-only marker / is not allowed",
                     "list the parameters: def f(a, b):", ln)
        if a.vararg is not None:
            self.err("forbidden_syntax", f"def {s.name}(*{a.vararg.arg})"[:60], "*args parameters are not allowed",
                     "name each parameter, or pass one list", ln)
        if a.kwonlyargs and a.vararg is None:
            self.err("forbidden_syntax", "* in def", "the keyword-only marker * is not allowed",
                     "list the parameters: def f(a, b=1):", ln)
        if a.kwarg is not None:
            self.err("forbidden_syntax", f"def {s.name}(**{a.kwarg.arg})"[:60], "**kwargs parameters are not allowed",
                     "name each parameter, or pass one dict", ln)
        every = a.posonlyargs + a.args + a.kwonlyargs + [x for x in (a.vararg, a.kwarg) if x is not None]
        if s.returns is not None or any(x.annotation is not None for x in every):
            self.err("forbidden_syntax", f"def {s.name}(x: T)"[:60], "type annotations are not allowed",
                     f"remove them: def {s.name}(a, b=0):", ln)
        params = [x.arg for x in a.args]
        for i, p in enumerate(params):
            self._target(p, ln)
            if p in params[:i]:
                self.err("bad_def", f"def {s.name}({p}, {p})"[:60], f"parameter {p!r} appears twice",
                         "give each parameter its own name", ln)
        defaults = {}
        for p, node in zip(params[len(params) - len(a.defaults):], a.defaults):
            v = self._default(node, depth + 1, ln, s.name, p)
            if v is not _NOCONST:
                defaults[p] = v
        saved = (self.ctx, self.loop_depth)
        self.ctx, self.loop_depth = _Ctx(s.name if own else None, params, _assigned_names(s.body)), 0
        try:
            body = self.block(s.body, depth + 1)
        finally:
            self.ctx, self.loop_depth = saved
        if own:
            self.fir[s.name] = {"line": ln, "params": params, "defaults": defaults,
                                "locals": sorted(set(params) | _assigned_names(s.body)), "body": body}

    def _default(self, node, depth, ln, fname, p):
        """A parameter default: a constant number, string, True, False or None (also -1.5), else bad_def."""
        simple = isinstance(node, ast.Constant) or (
            isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd))
            and isinstance(node.operand, ast.Constant) and _is_number(node.operand.value))
        ir = self.expr(node, depth)
        if simple and ir[0] == "const" and (ir[1] is None or isinstance(ir[1], (bool, int, float, str))):
            return ir[1]
        if simple and ir[0] == "unary" and ir[2][0] == "const" and _is_number(ir[2][1]):
            return -ir[2][1] if ir[1] == "neg" else ir[2][1]
        if not simple:
            self.err("bad_def", f"def {fname}({p}=...)"[:60], "a default value must be a constant (number, string, "
                     "True, False or None)", f"use None, then in the body: if {p} is None: {p} = ...", ln)
        return _NOCONST

    def _recursion(self):
        """0.3: no recursion. One error per cycle of the static call graph (self-calls included), at the def line
        of the first function of the cycle in definition order."""
        done = set()
        for f in self.funcs:
            if f in done or f not in self.graph:
                continue
            parent, queue, found = {}, deque(), False
            for g in sorted(self.graph[f]):
                parent[g] = f
                queue.append(g)
            while queue:
                g = queue.popleft()
                if g == f:
                    found = True
                    break
                for h in sorted(self.graph.get(g, ())):
                    if h not in parent:
                        parent[h] = g
                        queue.append(h)
            if not found:
                continue
            back, x = [], parent[f]
            while x != f:
                back.append(x)
                x = parent[x]
            path = [f] + back[::-1] + [f]
            done.update(path)
            self.err("recursion", f"recursion {f}"[:60], "recursion is not allowed: " + " -> ".join(path)[:150],
                     "repeat with a for or while loop instead of a function that calls itself", self.funcs[f].line)

    def _walk_children(self, node, depth):
        """Check what a refused construct contains, so that every error is reported at once."""
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, ast.stmt):
                self.stmt(ch, depth + 1)
            elif isinstance(ch, ast.expr):
                self.expr(ch, depth + 1)
            elif isinstance(ch, (ast.comprehension, ast.excepthandler, ast.withitem, ast.match_case, ast.arguments,
                                 ast.keyword)):
                self._walk_children(ch, depth + 1)

    # ---------------------------------------------------------------- statements
    def block(self, stmts, depth) -> list:
        return [self.stmt(s, depth) for s in stmts]

    def _loop_body(self, stmts, depth) -> list:
        self.loop_depth += 1
        try:
            return self.block(stmts, depth)
        finally:
            self.loop_depth -= 1

    def _append_stmt(self, call, depth, ln):
        """0.3: 'lst.append(x)' on its own line, lst a plain name -> ('assign', lst, append(lst, x)) (the 0.2 form)."""
        d = depth + 1
        self._count(call, d)
        recv = call.func.value
        if not isinstance(recv, ast.Name):
            self.err("forbidden_attribute", ".append()", "x.append(v) needs a plain variable name before the dot",
                     "assign it to a name first: inner = x[0] then inner.append(v), then store inner back", ln)
            self.expr(recv, d + 1)
            for a in call.args:
                self.expr(a.value if isinstance(a, ast.Starred) else a, d)
            for k in call.keywords:
                self.expr(k.value, d)
            return ("pass", ln)
        name = recv.id
        self._count(call.func, d)
        self._count(recv, d + 1)
        ok = True
        if call.keywords or len(call.args) != 1 or isinstance(call.args[0], ast.Starred):
            self.err("bad_call", f"{name}.append()"[:60], ".append() takes exactly one value (no keyword, no *)",
                     "write lst.append(x) to add one item; lst = lst + other adds several", ln)
            ok = False
        if self.ctx is not None and (name in self.ctx.params or name not in self.ctx.assigned):
            what = "a parameter" if name in self.ctx.params else "not a variable of this function"
            self.err("bad_append", f"{name}.append() in a def"[:60], f"inside a function, .append() may only change a "
                     f"list created in the function; {name!r} is {what}, and a function never changes its caller's "
                     "values",
                     f"write {name} = append({name}, x) and return it; the caller assigns the result"
                     if name in self.ctx.params else
                     f"return the new item and append it at the top level: {name}.append(f(...))", ln)
            ok = False
        elif self._shared(name, ln, f"{name}.append()", f"write {name} = append({name}, x)"):
            ok = False
        target = self._target(name, ln)
        args = [self.expr(a.value if isinstance(a, ast.Starred) else a, d) for a in call.args]
        for k in call.keywords:
            self.expr(k.value, d)
        if not ok:
            return ("pass", ln)
        return ("assign", ln, target, ("call", ln, "append", [("name", name), args[0]], []))

    def stmt(self, s, depth):
        self._count(s, depth)
        ln = getattr(s, "lineno", 0)
        if isinstance(s, ast.Assign):
            if len(s.targets) != 1 or not isinstance(s.targets[0], ast.Name):
                t = s.targets[0] if len(s.targets) == 1 else None
                if isinstance(t, ast.Subscript):
                    self.err("forbidden_syntax", "item assignment x[k] = v", "only 'name = expression' assignments "
                             "are allowed", "write d = set_key(d, \"key\", value) (for a list, build a new list)", ln)
                elif isinstance(t, (ast.Tuple, ast.List)):
                    self.err("forbidden_syntax", "unpacking a, b = ...", "only 'name = expression' assignments are "
                             "allowed", "assign one name per line: a = v[0] then b = v[1]", ln)
                elif isinstance(t, ast.Attribute):
                    self.err("forbidden_attribute", "attribute assignment a.b = v", "attribute access (a.b) is not "
                             "allowed", "use a dict: d = set_key(d, \"key\", value)", ln)
                else:
                    self.err("forbidden_syntax", "chained assignment", "only 'name = expression' assignments are "
                             "allowed", "one assignment per line", ln)
                for t in s.targets:
                    if not isinstance(t, ast.Name):
                        self._walk_children(t, depth)
                self.expr(s.value, depth + 1)
                return ("pass", ln)
            return ("assign", ln, self._target(s.targets[0].id, ln), self.expr(s.value, depth + 1))
        if isinstance(s, ast.AugAssign):
            if not isinstance(s.target, ast.Name) or type(s.op) not in (ast.Add, ast.Sub, ast.Mult, ast.Div):
                self.err("forbidden_syntax", "augmented assignment", "only 'name += -= *= /= expression' is allowed",
                         "write x = x <op> e", ln)
                self.expr(s.value, depth + 1)
                return ("pass", ln)
            name = s.target.id
            if isinstance(s.op, ast.Add) and self.sharing is not None and (
                    _Sharing._is_listy(s.value) or self.sharing.listy(None if self.ctx is None else self.ctx.name, name)):
                # Python's lst += [..] extends the list object in place, like .append
                self._shared(name, ln, f"{name} += [...]", f"write {name} = {name} + [...]")
            return ("augassign", ln, self._target(name, ln), _BIN[type(s.op)], self.expr(s.value, depth + 1))
        if isinstance(s, ast.AnnAssign):
            self.err("forbidden_syntax", "annotation x: T", "type annotations are not allowed", "write x = value", ln)
            if s.value is not None:
                self.expr(s.value, depth + 1)
            return ("pass", ln)
        if isinstance(s, ast.Expr):
            if isinstance(s.value, ast.Call) and isinstance(s.value.func, ast.Attribute) and s.value.func.attr == "append":
                return self._append_stmt(s.value, depth, ln)
            if not isinstance(s.value, ast.Call):
                if isinstance(s.value, ast.Constant) and isinstance(s.value.value, str):
                    return ("pass", ln)  # a docstring-like comment line
                self.err("forbidden_syntax", type(s.value).__name__, "an expression statement must be a call",
                         "a line is an assignment, a call, or a control statement", ln)
                self.expr(s.value, depth + 1)
                return ("pass", ln)
            return ("expr", ln, self.expr(s.value, depth + 1))
        if isinstance(s, ast.If):
            return ("if", ln, self.expr(s.test, depth + 1), self.block(s.body, depth + 1), self.block(s.orelse, depth + 1))
        if isinstance(s, ast.While):
            if s.orelse:
                self.err("forbidden_syntax", "while ... else", "while ... else is not allowed",
                         "remove the else: use a flag variable", ln)
                self.block(s.orelse, depth + 1)
            return ("while", ln, self.expr(s.test, depth + 1), self._loop_body(s.body, depth + 1))
        if isinstance(s, ast.For):
            ok = True
            if s.orelse:
                self.err("forbidden_syntax", "for ... else", "only 'for name in expression:' without else is allowed",
                         "remove the else: use a flag variable", ln)
                self.block(s.orelse, depth + 1)
                ok = False
            if not isinstance(s.target, ast.Name):
                self.err("forbidden_syntax", "for a, b in ...", "only 'for name in expression:' without else is "
                         "allowed", "write for x in xs: then a = x[0]", ln)
                ok = False
            it, body = self.expr(s.iter, depth + 1), self._loop_body(s.body, depth + 1)
            if not ok:
                return ("pass", ln)
            return ("for", ln, self._target(s.target.id, ln), it, body)
        if isinstance(s, (ast.Break, ast.Continue)):
            word = "break" if isinstance(s, ast.Break) else "continue"
            if self.loop_depth == 0:  # 0.3: refused before running (0.2: bad_break at run time); Python refuses it too
                where = "in this function" if self.ctx is not None else "here"
                self.err("forbidden_syntax", f"{word} outside a loop", f"{word} is allowed only inside a for or while "
                         f"loop ({where} it is outside any loop)",
                         "to leave a function use return; to end the program use a terminal" if self.ctx is not None
                         else "to end the program use a terminal (done, help, give_up, not_found)", ln)
                return ("pass", ln)
            return (word, ln)
        if isinstance(s, ast.Pass):
            return ("pass", ln)
        if isinstance(s, ast.Return):
            if self.ctx is None:
                construct, hint = _STMT_HINTS["Return"]
                self.err("forbidden_syntax", "return outside a def", "return is allowed only inside a function (def)",
                         hint, ln)
                if s.value is not None:
                    self.expr(s.value, depth + 1)
                return ("pass", ln)
            return ("return", ln, None if s.value is None else self.expr(s.value, depth + 1))
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):  # top-level defs are handled by program()
            if isinstance(s, ast.AsyncFunctionDef):
                construct, hint = _STMT_HINTS["AsyncFunctionDef"]
                self.err("forbidden_syntax", construct, "async def is not allowed", hint, ln)
            else:
                where = "inside a function" if self.ctx is not None else "inside a block (if, while, for, ...)"
                self.err("forbidden_syntax", f"def {s.name} {where.split(' (')[0]}"[:60], f"a def {where} is not "
                         "allowed: functions are defined at the top level only", _DEF_TOP_HINT, ln)
            self._def(s, depth, refused=True)
            return ("pass", ln)
        kind = type(s).__name__
        construct, hint = _STMT_HINTS.get(kind, (kind, "not available in BotScript"))
        self.err("forbidden_syntax", construct, f"{kind} statements are not allowed", hint, ln)
        self._walk_children(s, depth)
        return ("pass", ln)

    # ---------------------------------------------------------------- expressions
    def expr(self, e, depth):
        self._count(e, depth)
        ln = getattr(e, "lineno", 0)
        d = depth + 1
        bad = ("const", None)
        if isinstance(e, ast.Constant):
            v = e.value
            if v is None or isinstance(v, bool):
                return ("const", v)
            if isinstance(v, int):
                if abs(v) > self.lim.max_int_abs:
                    self.err("value_too_large", "integer literal", f"integer literal larger than {self.lim.max_int_abs}",
                             "use a smaller number", ln)
                    return bad
                return ("const", v)
            if isinstance(v, float):
                if v != v or v in (float("inf"), float("-inf")):
                    self.err("bad_literal", "number", "non-finite number", "use a finite number", ln)
                    return bad
                return ("const", v)
            if isinstance(v, str):
                if len(v) > self.lim.max_str_len:
                    self.err("value_too_large", "string literal", f"string literal longer than {self.lim.max_str_len}",
                             "use a shorter text", ln)
                    return bad
                if _SURROGATE.search(v):  # such a text cannot be encoded as UTF-8 (prompt, records)
                    self.err("bad_literal", "string literal", "lone surrogate characters (\\ud800 to \\udfff) are not "
                             "allowed in strings", "write plain text", ln)
                    return bad
                return ("const", v)
            self.err("bad_literal", f"{type(v).__name__} literal", f"{type(v).__name__} literals are not allowed",
                     "use numbers, strings, True, False or None", ln)
            return bad
        if isinstance(e, ast.Name):
            if e.id in self.funcs:  # 0.3: a defined function is never a value (no h = f, g(f), [f], return f)
                self.err("function_as_value", f"{e.id} as a value"[:60], f"{e.id!r} is a function: it can only be "
                         f"called, {e.id}(...)", f"call it: {self.funcs[e.id].signature()}"[:160], ln)
                return bad
            return ("name", self._name(e.id, ln))
        if isinstance(e, (ast.List, ast.Tuple)):
            if len(e.elts) > self.lim.max_list_len:
                self.err("value_too_large", "list literal", "list literal too long", "build long lists with a loop", ln)
                return bad
            if any(isinstance(x, ast.Starred) for x in e.elts):
                self.err("forbidden_syntax", "*expression", "star expressions are not allowed",
                         "write the items one by one", ln)
                self._walk_children(e, depth)
                return bad
            return ("list", [self.expr(x, d) for x in e.elts])
        if isinstance(e, ast.Dict):
            if any(k is None for k in e.keys):
                self.err("forbidden_syntax", "** in a dict", "** in dict literals is not allowed",
                         "write each key: d = set_key(d, k, v)", ln)
                for k, v in zip(e.keys, e.values):
                    if k is not None:  # the other keys are checked too (all errors at once)
                        self.expr(k, d)
                    self.expr(v, d)
                return bad
            return ("dict", [(self.expr(k, d), self.expr(v, d)) for k, v in zip(e.keys, e.values)])
        if isinstance(e, ast.BinOp):
            if type(e.op) not in _BIN:
                opn = type(e.op).__name__
                self.err("forbidden_syntax", f"operator {opn}", f"operator {opn} is not allowed",
                         "use + - * / // % ** only", ln)
                self.expr(e.left, d)
                self.expr(e.right, d)
                return bad
            left, right = self.expr(e.left, d), self.expr(e.right, d)
            if isinstance(e.op, ast.Pow):  # 0.3: two constants are computed now, under the run-time rule
                a, b = _const_value(e.left, self.lim.max_int_abs), _const_value(e.right, self.lim.max_int_abs)
                if _is_number(a) and _is_number(b):
                    try:
                        power(a, b, self.lim.max_int_abs)
                    except PowerError as x:
                        self.err(x.code, "** (power)", x.message, _POWER_HINT, ln)
                        return bad
            return ("bin", _BIN[type(e.op)], left, right)
        if isinstance(e, ast.UnaryOp):
            if type(e.op) not in _UN:
                opn = type(e.op).__name__
                self.err("forbidden_syntax", f"operator {opn}", f"operator {opn} is not allowed", "use - + not only", ln)
                self.expr(e.operand, d)
                return bad
            return ("unary", _UN[type(e.op)], self.expr(e.operand, d))
        if isinstance(e, ast.BoolOp):
            return ("bool", "and" if isinstance(e.op, ast.And) else "or", [self.expr(x, d) for x in e.values])
        if isinstance(e, ast.Compare):
            ops, ok = [], True
            for op, c in zip(e.ops, e.comparators):
                if isinstance(op, (ast.Is, ast.IsNot)):
                    if isinstance(c, ast.Constant) and c.value is None:
                        # 0.2: on plain values (no user-defined __eq__) "x is None" is exactly "x == None"
                        ops.append(("==" if isinstance(op, ast.Is) else "!=", ("const", None)))
                        continue
                    self.err("forbidden_syntax", "is" if isinstance(op, ast.Is) else "is not",
                             "'is' / 'is not' are allowed only with None", "use == or != (x is None is fine)", ln)
                    self.expr(c, d)
                    ok = False
                    continue
                if type(op) not in _CMP:  # pragma: no cover (every Python comparison is handled above)
                    self.err("forbidden_syntax", type(op).__name__, f"comparison {type(op).__name__} is not allowed",
                             "use == != < <= > >= in, not in", ln)
                    ok = False
                    continue
                ops.append((_CMP[type(op)], self.expr(c, d)))
            left = self.expr(e.left, d)
            return ("cmp", left, ops) if ok else bad
        if isinstance(e, ast.IfExp):
            return ("ifexp", self.expr(e.test, d), self.expr(e.body, d), self.expr(e.orelse, d))
        if isinstance(e, ast.Subscript):
            if isinstance(e.slice, ast.Slice):
                if e.slice.step is not None:
                    self.err("forbidden_syntax", "slice step x[a:b:c]", "slice steps are not allowed",
                             "loop with range() instead", ln)
                    self._walk_children(e, depth)
                    return bad
                lo = None if e.slice.lower is None else self.expr(e.slice.lower, d)
                hi = None if e.slice.upper is None else self.expr(e.slice.upper, d)
                return ("slice", self.expr(e.value, d), lo, hi)
            return ("sub", self.expr(e.value, d), self.expr(e.slice, d))
        if isinstance(e, ast.Call):
            return self._call(e, depth, ln)
        if isinstance(e, ast.FormattedValue):  # only reached inside a refused f-string: check what it holds
            self.expr(e.value, d)
            return bad
        if isinstance(e, ast.Attribute):
            if e.attr in METHOD_FORMS:  # 0.3: a listed method is only ever called, never a value
                self.err("forbidden_attribute", f".{e.attr}"[:40], f"a method name without a call (.{e.attr}) is not "
                         "a value", f"call the method: {METHOD_FORMS[e.attr]}", ln)
            else:
                hint = _METHOD_HINTS.get(e.attr, "no attribute access: use d[\"key\"], d.get(\"key\") or a builtin")
                self.err("forbidden_attribute", f".{e.attr}"[:40], "attribute access (a.b) is not allowed", hint, ln)
            self.expr(e.value, d)
            return bad
        kind = type(e).__name__
        construct, hint = _EXPR_HINTS.get(kind, (kind, "not available in BotScript"))
        self.err("forbidden_syntax", construct, f"{kind} is not allowed", hint, ln)
        self._walk_children(e, depth)
        return bad

    def _call(self, e, depth, ln):
        d = depth + 1
        bad = ("const", None)
        if isinstance(e.func, ast.Attribute):
            if e.func.attr in METHODS:
                return self._method(e, depth, ln)
            if e.func.attr == "append":
                self.err("forbidden_attribute", ".append()", "lst.append(x) is allowed only as a statement on its own "
                         "line (it gives no value)", _METHOD_HINTS["append"], ln)
            else:
                hint = _METHOD_HINTS.get(e.func.attr, _METHODS_LIST)
                self.err("forbidden_attribute", f".{e.func.attr}()"[:40], f"method .{e.func.attr}() is not allowed "
                         "(only the listed methods exist)", hint, ln)
            self.expr(e.func.value, d)
            for a in e.args:
                self.expr(a.value if isinstance(a, ast.Starred) else a, d)
            for k in e.keywords:
                self.expr(k.value, d)
            return bad
        if not isinstance(e.func, ast.Name):
            self.err("forbidden_call", "call of a value", "only calls of a whitelisted name are allowed (no attribute, "
                     "no call of a value)", "call an API function or a builtin by its name", ln)
            self._walk_children(e, depth)
            return bad
        fname = e.func.id
        if fname in self.funcs:
            return self._ucall(e, depth, ln)
        ok = True
        if fname not in CALLABLE:
            self.err("unknown_function", f"{fname}()"[:40], f"{fname!r} is not in the skill API or the builtins",
                     _FUNCTION_HINTS.get(fname, "only the skill API functions and the listed builtins exist"), ln)
            ok = False
        if any(isinstance(a, ast.Starred) for a in e.args) or any(k.arg is None for k in e.keywords):
            self.err("forbidden_syntax", "*args / **kwargs", "*args and **kwargs are not allowed",
                     "pass each argument by itself", ln)
            ok = False
        elif len({k.arg for k in e.keywords}) != len(e.keywords):
            self.err("forbidden_syntax", f"{fname}() repeated keyword", "a keyword argument is repeated",
                     "give each keyword once", ln)
            ok = False
        kws = []
        for k in e.keywords:
            v = self.expr(k.value, d)
            if k.arg is not None:
                kws.append((self._name(k.arg, ln), v))
        args = [self.expr(a, d) for a in e.args if not isinstance(a, ast.Starred)]
        for a in e.args:
            if isinstance(a, ast.Starred):
                self.expr(a.value, d)
        if fname == "help" and not any(isinstance(a, ast.Starred) for a in e.args) \
                and not any(k.arg is None for k in e.keywords):
            ok = self._help_kind(e, ln) and ok  # an error (refused), not a warning
            ok = self._help_args(e, ln) and ok  # every certain argument error too
        if fname == "help":
            pass  # checked above (_help_kind, _help_args: errors), never by _api_args (warnings)
        elif ok and fname in API:
            self._api_args(fname, e, ln)
        elif ok and e.keywords:
            allowed = BUILTIN_KEYWORDS.get(fname, ())
            wrong = [k.arg for k in e.keywords if k.arg not in allowed]
            if wrong:
                self.warn("bad_arg", f"{fname}({wrong[0]}=...)"[:60], f"{fname}() takes no keyword argument {wrong[0]!r}",
                          f"pass the arguments by position: {fname}(...)" if not allowed else
                          f"{fname}() keywords: {', '.join(allowed)}", ln)
        return ("call", ln, fname, args, kws) if ok else bad

    def _method(self, e, depth, ln):
        """0.3: recv.m(args) for m in METHODS -> ('method', line, m, recv, args); positional arguments only, in the
        counts of METHODS (certain errors: refused). The receiver's type is checked when it runs."""
        d = depth + 1
        m = e.func.attr
        _, lo, hi = METHODS[m]
        self._count(e.func, d)
        ok = True
        if e.keywords or any(isinstance(a, ast.Starred) for a in e.args):
            self.err("bad_call", f".{m}()", f".{m}() takes positional arguments only (no keyword, no *, no **)",
                     f"write {METHOD_FORMS[m]}", ln)
            ok = False
        elif not lo <= len(e.args) <= hi:
            want = str(lo) if lo == hi else f"{lo} or {hi}"
            self.err("bad_call", f".{m}()", f".{m}() takes {want} argument{'s' if hi != 1 else ''}, got "
                     f"{len(e.args)}", f"write {METHOD_FORMS[m]}", ln)
            ok = False
        recv = self.expr(e.func.value, d)
        args = [self.expr(a.value if isinstance(a, ast.Starred) else a, d) for a in e.args]
        for k in e.keywords:
            self.expr(k.value, d)
        return ("method", ln, m, recv, args) if ok else ("const", None)

    def _ucall(self, e, depth, ln):
        """0.3: a call of a function defined in the program. Argument errors are certain: refused (bad_call)."""
        d = depth + 1
        fname = e.func.id
        info = self.funcs[fname]
        params, sig = info.params, info.signature()
        ok = True
        if any(isinstance(a, ast.Starred) for a in e.args) or any(k.arg is None for k in e.keywords):
            self.err("forbidden_syntax", "*args / **kwargs", "*args and **kwargs are not allowed",
                     "pass each argument by itself", ln)
            ok = False
        else:
            if len(e.args) > len(params):
                self.err("bad_call", f"{fname}() arguments"[:60], f"{fname}() takes at most {len(params)} arguments, "
                         f"got {len(e.args)}", f"signature: {sig}"[:160], ln)
                ok = False
            given = set(params[:len(e.args)])
            for k in e.keywords:
                if k.arg not in params:
                    self.err("bad_call", f"{fname}({k.arg}=...)"[:60], f"{fname}() has no parameter {k.arg!r}",
                             f"signature: {sig}"[:160], ln)
                    ok = False
                elif k.arg in given:
                    self.err("bad_call", f"{fname}({k.arg}=...)"[:60], f"{fname}() got {k.arg!r} twice",
                             f"signature: {sig}"[:160], ln)
                    ok = False
                else:
                    given.add(k.arg)
            missing = [p for p in params[:len(params) - info.n_defaults] if p not in given]
            if missing:
                self.err("bad_call", f"{fname}() missing {missing[0]}"[:60], f"{fname}() is missing argument "
                         f"{missing[0]!r}", f"signature: {sig}"[:160], ln)
                ok = False
        if self.ctx is not None and self.ctx.name is not None:
            self.graph.setdefault(self.ctx.name, {}).setdefault(fname, ln)
        kws = [(self._name(k.arg, ln), self.expr(k.value, d)) for k in e.keywords if k.arg is not None]
        args = [self.expr(a.value if isinstance(a, ast.Starred) else a, d) for a in e.args]
        for k in e.keywords:
            if k.arg is None:
                self.expr(k.value, d)
        return ("ucall", ln, fname, args, kws) if ok else ("const", None)

    def _help_kind(self, e, ln: int) -> bool:
        """Skill API 0.6: the kind of a help() call, checked as an ERROR. Without kind (by keyword or
        second position): help_without_kind; kind given twice, or a constant outside api.HELP_KINDS: bad_help_kind. A
        kind that is not WRITTEN as a word of api.HELP_KINDS in quotes (a name, a call, an operation, even one that
        would give a valid word) is bad_help_kind too, so no reached help can fail on its kind (for example: k = "dan" +
        "ger"; help("stop now", kind=k) ended runtime_error, then more gestures). True when no error."""
        kinds = [k.value for k in e.keywords if k.arg == "kind"] + list(e.args[1:2])
        if not kinds:
            self.err("help_without_kind", "help() without kind", "help() needs an explicit kind (skill API 0.6): a "
                     "program with a help() without kind is refused", _HELP_KIND_HINT, ln)
            return False
        if len(kinds) > 1:
            self.err("bad_help_kind", "help() kind twice", "help() got kind twice (by position and by keyword)",
                     _HELP_KIND_HINT, ln)
            return False
        v = _const_value(kinds[0], self.lim.max_int_abs)
        if v is not _NOCONST and not (type(v) is str and v in HELP_KINDS):
            shown = repr(v[:24]) if type(v) is str else type(v).__name__
            self.err("bad_help_kind", f"help(kind={shown})"[:60], f"help() kind {shown} is not one of "
                     f"{', '.join(HELP_KINDS)}", _HELP_KIND_HINT, ln)
            return False
        if not (isinstance(kinds[0], ast.Constant) and type(kinds[0].value) is str):  # a word in quotes
            self.err("bad_help_kind", "help(kind=<expression>)", "help() kind must be written as one of the words "
                     f"{', '.join(HELP_KINDS)} in quotes, not computed", _HELP_KIND_HINT, ln)
            return False
        return True

    def _help_args(self, e, ln: int) -> bool:
        """help()'s other certain argument errors, as ERRORS (bad_help_args): more than two
        arguments, a keyword other than why and kind, why given twice or missing, a constant why that is not a text,
        has nothing readable after api.clean_text, or is longer than api.MAX_RAW_TEXT_CHARS (also its constant pieces
        joined by +). A why computed at run time is accepted: the interpreter keeps the help whatever it gives
        (api.check_help). Before, these were warnings and the reached help() failed at run time (for example:
        help(":::", kind="safety"), help("x", "safety", "extra"), help("x", kind="safety", urgent=True)). True when no
        error."""
        ok = True
        if len(e.args) > 2:
            self.err("bad_help_args", "help() arguments", f"help() takes at most 2 arguments (why, kind), got "
                     f"{len(e.args)}", _HELP_KIND_HINT, ln)
            ok = False
        whys = list(e.args[:1])
        for k in e.keywords:
            if k.arg == "why":
                whys.append(k.value)
            elif k.arg != "kind":
                self.err("bad_help_args", f"help({k.arg}=...)"[:60], f"help() has no argument {k.arg!r} (only why "
                         "and kind)", _HELP_KIND_HINT, ln)
                ok = False
        if len(whys) != 1:
            self.err("bad_help_args", "help() why", "help() got why twice" if whys else "help() is missing why",
                     _HELP_KIND_HINT, ln)
            return False
        v = _const_value(whys[0], self.lim.max_int_abs)
        if v is not _NOCONST:
            bad = (type(v) is not str or len(v) > MAX_RAW_TEXT_CHARS or not clean_text(v))
        else:
            bad = sum(len(x) for x in (_str_pieces(whys[0]) or [])) > MAX_RAW_TEXT_CHARS
        if bad:
            self.err("bad_help_args", "help(why=...)", "help() why must be a text with at least one letter or digit "
                     f"(at most {MAX_RAW_TEXT_CHARS} characters before cleaning)", _HELP_KIND_HINT, ln)
            ok = False
        return ok

    def _api_args(self, fname: str, e, ln: int) -> bool:
        """Static check of an API call where the failure is certain if the call is reached (0.2, point 4):
        warnings only. Returns False when a warning was given. The kind of help() is checked by _help_kind (errors),
        not here; help() is not checked here at all (_help_kind and _help_args, errors)."""
        params = API[fname]["params"]
        skip = ("kind",) if fname == "help" else ()
        names = [p.name for p in params]
        sig = f"{fname}({', '.join(names)})"
        ok = True
        if len(e.args) > len(params):
            self.warn("bad_arg", f"{fname}() arguments", f"{fname}() takes at most {len(params)} arguments, got "
                     f"{len(e.args)}", f"signature: {sig}", ln)
            ok = False
        given = {names[i]: a for i, a in enumerate(e.args[:len(params)])}
        for k in e.keywords:
            if k.arg not in names:
                self.warn("bad_arg", f"{fname}({k.arg}=...)"[:60], f"{fname}() has no argument {k.arg!r}",
                         f"signature: {sig}", ln)
                ok = False
            elif k.arg in given:
                if k.arg not in skip:
                    self.warn("bad_arg", f"{fname}({k.arg}=...)"[:60], f"{fname}() got {k.arg!r} twice",
                              f"signature: {sig}", ln)
                    ok = False
            else:
                given[k.arg] = k.value
        missing = [p.name for p in params if p.required and p.name not in given and p.name not in skip]
        if missing:
            self.warn("bad_arg", f"{fname}() missing {missing[0]}"[:60], f"{fname}() is missing argument {missing[0]!r}",
                     f"signature: {sig}", ln)
            ok = False
        for p in params:
            node = given.get(p.name)
            if node is None or p.name in skip:
                continue
            v = _const_value(node, self.lim.max_int_abs)
            if v is not _NOCONST:
                try:
                    _check(p, v)
                except ArgError as x:
                    self.warn("bad_arg", f"{fname}({p.name}=...)"[:60], f"{fname}(): {x}", self._arg_hint(p, v, sig), ln)
                    ok = False
            elif p.type == "text":
                # skill API 0.3: other characters are cleaned, not refused; a text built with + certainly fails only
                # when its constant pieces alone are longer than MAX_RAW_TEXT_CHARS
                raw = sum(len(s) for s in (_str_pieces(node) or []))
                if raw > MAX_RAW_TEXT_CHARS:
                    self.warn("bad_arg", f"{fname}({p.name}=...)"[:60], f"{fname}(): the text is longer than "
                             f"{MAX_RAW_TEXT_CHARS} characters", self._arg_hint(p, None, sig), ln)
                    ok = False
        return ok

    @staticmethod
    def _arg_hint(p: P, v, sig: str) -> str:
        if p.type == "text":  # skill API 0.3 (api.clean_text)
            return (f"text: at least one letter or digit, at most {MAX_RAW_TEXT_CHARS} characters (other characters "
                    "become spaces, 80 are kept)")
        if p.type == "enum":
            return f"{p.name} must be one of: {', '.join(p.choices)}"
        return f"signature: {sig}"


def _bracket_depth(src: str) -> int:
    """Deepest bracket nesting, counted on the tokens (brackets inside strings and comments do
    not count). tokenize builds no tree (no recursion); if it fails, every bracket character counts."""
    def depth_of(chars) -> int:
        depth = best = 0
        for ch in chars:
            if ch in "([{":
                depth += 1
                best = max(best, depth)
            elif ch in ")]}":
                depth = max(0, depth - 1)
        return best
    try:
        ops = [t.string for t in tokenize.generate_tokens(io.StringIO(src).readline)
               if t.type == tokenize.OP and t.string in ("(", ")", "[", "]", "{", "}")]
    except (tokenize.TokenError, SyntaxError, ValueError, RecursionError, MemoryError):
        return depth_of(src)
    return depth_of(ops)


def _refused(errors: list[dict], warnings: list[dict] | None = None) -> Refused:
    first = errors[0]
    n = len(errors)
    shown = errors[:MAX_REPORTED_ERRORS]
    msg = first["message"] if n == 1 else (f"{n} language errors (all listed in errors; nothing ran); first: line "
                                           f"{first['line']}: {first['message']}")
    return Refused(first["code"], msg, first["line"], shown, n, (warnings or [])[:MAX_REPORTED_ERRORS])


def compile_program(src, lim: ProgramLimits | None = None) -> dict:
    """Text -> checked intermediate form {"language", "body", "functions", "n_nodes", "warnings"}. Raises Refused
    (every error found, with the warnings found so far), and NOTHING else: any unexpected Python
    exception of the checker (a defect of this checker, e.g. the int-to-text ValueError of a 4000-digit hex exponent)
    becomes Refused code "checker_internal" with the exception type only, so a model's text can never raise out of a
    run or an episode."""
    try:
        return _compile(src, lim)
    except Refused:
        raise
    except Exception as x:  # noqa: BLE001 (bounded refusal, see above)
        msg = f"the program could not be checked (internal {type(x).__name__}); nothing ran"
        raise Refused("checker_internal", msg, 0, [{"line": 0, "code": "checker_internal", "construct": "program",
                                                     "message": msg, "hint": "write a simpler program: plain numbers, "
                                                     "short lines"}]) from None


def _compile(src, lim: ProgramLimits | None) -> dict:
    lim = lim or ProgramLimits()
    if not isinstance(src, str):
        raise Refused("not_text", "the program must be text")
    if len(src) > lim.max_source_chars:
        raise Refused("too_long", f"the program is longer than {lim.max_source_chars} characters")
    if "\x00" in src:
        raise Refused("syntax_error", "NUL character in the program")
    if _bracket_depth(src) > 2 * lim.max_nesting_depth:
        raise Refused("too_deep", "brackets nested too deeply")
    try:
        tree = ast.parse(src, mode="exec")
    except SyntaxError as e:
        raise Refused("syntax_error", f"syntax error: {e.msg}", e.lineno or 0, [{
            "line": e.lineno or 0, "code": "syntax_error", "construct": "syntax", "message": f"syntax error: {e.msg}",
            "hint": "BotScript uses Python syntax: check brackets, colons and indentation"}]) from None
    except (ValueError, RecursionError, MemoryError, OverflowError):
        raise Refused("syntax_error", "the program could not be parsed") from None
    if not tree.body:
        raise Refused("empty", "the program is empty")
    c = _Checker(lim)
    try:
        body = c.program(tree.body)
    except _Fatal as f:
        raise _refused([f.err] + c.errors, sorted(c.warnings, key=lambda x: x["line"])) from None
    except RecursionError:
        raise Refused("too_deep", "the program is nested too deeply") from None
    warnings = sorted(c.warnings, key=lambda x: x["line"])[:MAX_REPORTED_ERRORS]
    if c.errors:
        raise _refused(sorted(c.errors, key=lambda x: x["line"]), warnings)  # stable: source order within a line
    # 0.3: "functions" holds the compiled top-level defs (name -> line, params, defaults, locals, body)
    return {"language": LANGUAGE, "body": body, "functions": c.fir, "n_nodes": c.n_nodes, "warnings": warnings}
