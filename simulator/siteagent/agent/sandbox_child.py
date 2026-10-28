"""Child process of agent.sandbox: runs one BotScript program and talks JSON lines on stdin and stdout.

Started by agent.sandbox with an empty environment, python -E -s -S -B, and on macOS under sandbox-exec with a
deny-by-default profile (agent.sandbox.build_profile). This process never imports the simulator, the robot, the bench
or the evaluator; the only way out is the pipe to the parent, which re-checks every request.

Protocol (one JSON object per line):
  parent -> child  {"t": "start", "src": <program text>, "limits": {...ProgramLimits}}
  child  -> parent {"t": "call", "name": <api name>, "args": {...}}
  parent -> child  {"t": "result", "value": ...} | {"t": "limit", "code", "message"} | {"t": "arg_error", "message"}
  child  -> parent {"t": "end", "outcome": {...}}
"--selftest" (tests only) reports what the OS layer allows, without running any program.
"""
import json
import os
import sys

_LAB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _LAB)

from siteagent.agent.api import ArgError  # noqa: E402
from siteagent.agent.interp import Interpreter, LimitReached  # noqa: E402
from siteagent.agent.limits import ProgramLimits  # noqa: E402

_IN, _OUT = sys.stdin, sys.stdout
MAX_LINE = 4_000_000


def _send(obj):
    _OUT.write(json.dumps(obj, allow_nan=False) + "\n")
    _OUT.flush()


def _recv():
    line = _IN.readline(MAX_LINE)
    if not line:
        raise SystemExit(3)
    return json.loads(line)


class PipeHost:
    def call(self, name, args):
        _send({"t": "call", "name": name, "args": args})
        m = _recv()
        if m.get("t") == "result":
            return m.get("value")
        if m.get("t") == "limit":
            raise LimitReached(str(m.get("code")), str(m.get("message", "")))
        raise ArgError(str(m.get("message", "refused by the host")))


def _lock_down(lim: ProgramLimits):
    try:
        import resource
        cpu = int(lim.max_interpreter_cpu_s) + 3
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 2))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    except (ImportError, ValueError, OSError):
        pass


def _try(fn):
    try:
        return fn()
    except BaseException as e:  # noqa: BLE001 (report what the OS layer refused)
        return type(e).__name__


def _readlen(path):
    with open(path, "rb") as f:
        return "read:%d" % len(f.read())


def _mach(name):
    import ctypes
    libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    bp = ctypes.c_uint.in_dll(libc, "bootstrap_port")
    port = ctypes.c_uint(0)
    kr = libc.bootstrap_look_up(bp, name.encode(), ctypes.byref(port))
    return "allowed" if kr == 0 and port.value else "denied:%d" % kr


def _spawn():
    pid = os.posix_spawn("/usr/bin/true", ["true"], {})
    os.waitpid(pid, 0)
    return "spawned"


def _fork():
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    os.waitpid(pid, 0)
    return "forked"


def _write(path):
    with open(path, "w") as f:
        f.write("x")
    return "written"


def _selftest(arg):
    """arg: a decoy .env path (0.1 form) or a JSON object {"decoy": path, "reads": {label: path},
    "write": path}. Reports what the OS layer allows; never prints a file's content (lengths only)."""
    spec = json.loads(arg) if arg.startswith("{") else {"decoy": arg}
    decoy = spec["decoy"]
    rep = {"env_keys": sorted(os.environ.keys()),
           "heavy_modules_loaded": sorted(m for m in sys.modules if m.split(".")[0] in ("mujoco", "numpy", "openai")
                                          or m.startswith(("siteagent.robot", "siteagent.sim", "siteagent.taskcheck",
                                                           "siteagent.perception", "siteagent.world",
                                                           "siteagent.bench_task", "siteagent.programs"))),
           "siteagent_modules_loaded": sorted(m for m in sys.modules if m.startswith("siteagent"))}
    import socket

    def loop():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1.0)
        s.connect(("127.0.0.1", 9))
        return "connected"
    rep["loopback_connect"] = _try(loop)
    rep["file_write"] = _try(lambda: _write(spec.get("write") or os.path.join(os.path.dirname(decoy),
                                                                             "written_by_child.txt")))
    rep["decoy_env_read"] = _try(lambda: _readlen(decoy))
    rep["reads"] = {k: _try(lambda p=p: _readlen(p)) for k, p in sorted(spec.get("reads", {}).items())}
    rep["fork"] = _try(_fork)
    rep["posix_spawn"] = _try(_spawn)
    rep["signal_parent"] = _try(lambda: (os.kill(os.getppid(), 0), "allowed")[1])
    rep["mach_launchservices"] = _try(lambda: _mach("com.apple.coreservices.launchservicesd"))
    rep["mach_pasteboard"] = _try(lambda: _mach("com.apple.pasteboard.1"))
    _send({"t": "selftest", "report": rep})
    if spec.get("try_exec"):  # last: exec replaces this process if allowed
        try:
            os.execv("/bin/echo", ["echo", "EXEC_OK"])
        except BaseException as e:  # noqa: BLE001
            _send({"t": "exec", "result": type(e).__name__})


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--selftest":
        _selftest(sys.argv[2])
        return
    start = _recv()
    if start.get("t") != "start":
        raise SystemExit(2)
    lim = ProgramLimits(**start["limits"])
    _lock_down(lim)
    out = Interpreter(PipeHost(), lim).run(start["src"])
    _send({"t": "end", "outcome": out})


if __name__ == "__main__":
    main()
