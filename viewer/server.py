"""Loopback-only visual launcher. No model calls, credentials or external services."""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

BASE = Path(__file__).resolve().parent
REPOSITORY = BASE.parent
SCENARIOS = {
    "in_place": {"title": "Sponge in its usual drawer", "description": "Inspect, find the sponge and scrub the bowl."},
    "moved": {"title": "Sponge moved to another drawer", "description": "The site file is out of date. Let the robot search again."},
    "masked": {"title": "Sponge hidden under a towel", "description": "Watch what happens when the robot cannot finish checking a drawer."},
}
JOB_ID = re.compile(r"^[0-9a-f]{32}$")
TERMINAL = {"completed", "different", "failed", "cancelled", "timed_out"}


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def child_environment():
    # Do not inherit model credentials, tokens or user shell configuration.
    return {"HOME": str(Path.home()), "PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "LANG": "en_US.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}


def analyze_run(directory):
    reproduction = read_json(directory / "episode/REPRODUCTION.json")
    oracle = read_json(directory / "episode/oracle_verdict.json")
    if reproduction is None or oracle is None:
        raise ValueError("This run has no complete evaluator record. Start a new simulation.")
    verdict = oracle["verdict"]
    choice = reproduction["choice"]
    return {
        "label": "Readiness report, uncalibrated simulation",
        "situation": reproduction["situation"],
        "outcome": verdict["outcome_class"], "detail": verdict.get("outcome_detail"),
        "claim": reproduction["result"]["final"].get("claim"),
        "criteria": verdict["criteria"], "violations": verdict["violations"],
        "who_chose": choice["who_chose"], "option": choice["option"],
        "model_called": choice["model_called_during_this_run"],
        "recorded_answer_used": choice["recorded_answer_used"],
        "same_request_text": choice["same_request_text"],
        "comparison": reproduction["comparison_with_recorded_episode"],
        "wall_s": reproduction["wall_s"],
        "files_verified": reproduction["distribution"]["files_verified"],
        "code_and_data_sha256": reproduction["distribution"]["code_and_data_sha256"],
        "scope": "One development kitchen, fixed robot programs and simulated perception. These criteria are not a real-world reliability estimate.",
    }


class Runner:
    def __init__(self, output=BASE / "runs", timeout=600):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.lock = threading.Lock()
        self.jobs = {}
        # Keep completed local sessions available after restarting the viewer.
        for directory in self.output.iterdir():
            if not directory.is_dir() or not JOB_ID.fullmatch(directory.name):
                continue
            session = read_json(directory / "SESSION.json")
            status = read_json(directory / "STATUS.json")
            if not isinstance(session, dict) or session.get("situation") not in SCENARIOS:
                continue
            valid_status = status and status.get("status") in TERMINAL
            self.jobs[directory.name] = {
                "id": directory.name, "situation": session["situation"],
                "started": session["started"], "directory": directory,
                "status": status["status"] if valid_status else "failed",
                "finished": session["started"] + status["elapsed_s"] if valid_status else time.time(),
                "process": None, "cancel_requested": False,
                "error": status.get("error") if valid_status else "The service stopped before this run finished. Start a new run.",
                "exit_code": status.get("exit_code") if valid_status else None,
            }

    def start(self, situation):
        if situation not in SCENARIOS:
            raise ValueError("Choose one of the three listed situations.")
        with self.lock:
            if any(job["status"] not in TERMINAL for job in self.jobs.values()):
                raise RuntimeError("A simulation is already running. Wait for it or stop it first.")
            identifier = uuid.uuid4().hex
            directory = self.output / identifier
            directory.mkdir(exist_ok=False)
            job = {"id": identifier, "situation": situation, "status": "starting",
                   "started": time.time(), "directory": directory, "process": None,
                   "cancel_requested": False, "error": None, "exit_code": None}
            self.jobs[identifier] = job
            (directory / "SESSION.json").write_text(json.dumps({"id": identifier, "situation": situation,
                                                                "started": job["started"]}))
            threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return self.snapshot(identifier)

    def _run(self, job):
        try:
            with (job["directory"] / "process.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen([sys.executable, "-B", str(BASE / "visual_run.py"),
                                            "--situation", job["situation"], "--session", str(job["directory"])],
                                           cwd=BASE, stdout=log, stderr=subprocess.STDOUT,
                                           env=child_environment(), start_new_session=True)
                with self.lock:
                    job["process"] = process
                    job["status"] = "running"
                    cancel = job["cancel_requested"]
                if cancel:
                    self._terminate(process)
                try:
                    code = process.wait(timeout=self.timeout)
                except subprocess.TimeoutExpired:
                    self._terminate(process)
                    job["status"] = "timed_out"
                    job["error"] = "The local run exceeded ten minutes and was stopped. Its frames have been kept."
                    return
                job["exit_code"] = code
                if job["cancel_requested"]:
                    job["status"] = "cancelled"
                elif code == 0:
                    job["status"] = "completed"
                elif code == 5:
                    job["status"] = "different"
                    job["error"] = "This run differs from the recorded reference. Open the analysis for the differences."
                else:
                    job["status"] = "failed"
                    job["error"] = "The simulator could not complete this run. See the local run log below."
        except Exception as exc:
            job["status"] = "failed"
            job["error"] = f"The local simulator could not start ({type(exc).__name__})."
        finally:
            job["finished"] = time.time()
            status = self.snapshot(job["id"])
            (job["directory"] / "STATUS.json").write_text(json.dumps(status, indent=2))

    @staticmethod
    def _terminate(process):
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=3)
        except ProcessLookupError:
            pass

    def get(self, identifier):
        if not JOB_ID.fullmatch(identifier) or identifier not in self.jobs:
            raise KeyError("Run not found.")
        return self.jobs[identifier]

    def snapshot(self, identifier):
        job = self.get(identifier)
        progress = read_json(job["directory"] / "progress.json", {})
        elapsed = (job.get("finished") or time.time()) - job["started"]
        return {"id": identifier, "situation": job["situation"], "status": job["status"],
                "elapsed_s": round(elapsed, 1), "error": job["error"], "exit_code": job["exit_code"],
                "progress": progress, "analysis_available": job["status"] in {"completed", "different"}}

    def cancel(self, identifier):
        with self.lock:
            job = self.get(identifier)
            if job["status"] in TERMINAL:
                return self.snapshot(identifier)
            job["cancel_requested"] = True
            process = job["process"]
        if process:
            self._terminate(process)
        return self.snapshot(identifier)

    def shutdown(self):
        for job in list(self.jobs.values()):
            if job["status"] not in TERMINAL:
                self.cancel(job["id"])


class Handler(BaseHTTPRequestHandler):
    server_version = "SiteShiftLocal/1.0"

    def log_message(self, *args):
        pass

    def valid_host(self):
        return self.headers.get("Host") in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}

    def reply(self, code, content, content_type="application/json", attachment=None):
        if isinstance(content, (dict, list)):
            content = json.dumps(content).encode()
        elif isinstance(content, str):
            content = content.encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
        if attachment:
            self.send_header("Content-Disposition", f'attachment; filename="{attachment}"')
        self.end_headers()
        try:
            self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def file(self, path):
        if not path.is_file():
            return self.reply(404, {"error": "File not found."})
        return self.reply(200, path.read_bytes(), mimetypes.guess_type(path.name)[0] or "application/octet-stream")

    def do_GET(self):
        if not self.valid_host():
            return self.reply(403, {"error": "Use the local SiteShift address."})
        path = urlsplit(self.path).path
        if path == "/api/config":
            return self.reply(200, {"scenarios": SCENARIOS, "local": True,
                                    "estimated_seconds": [30, 180], "timeout_seconds": self.server.runner.timeout})
        match = re.fullmatch(r"/api/runs/([0-9a-f]{32})(?:/(.*))?", path)
        if match:
            identifier, action = match.groups()
            try:
                job = self.server.runner.get(identifier)
                if action is None:
                    return self.reply(200, self.server.runner.snapshot(identifier))
                if action == "log":
                    text = (job["directory"] / "process.log").read_text(encoding="utf-8") if (job["directory"] / "process.log").exists() else "Starting…"
                    return self.reply(200, text[-24000:], "text/plain; charset=utf-8")
                if action == "report":
                    return self.file(job["directory"] / "episode/REPRODUCTION.json")
                frame = re.fullmatch(r"frames/([0-9]{5})\.jpg", action or "")
                if frame:
                    return self.file(job["directory"] / "frames" / (frame[1] + ".jpg"))
            except KeyError:
                return self.reply(404, {"error": "Run not found."})
        if path == "/":
            return self.file(BASE / "web/index.html")
        if path in {"/app.js", "/style.css", "/preview.jpg"}:
            return self.file(BASE / "web" / path[1:])
        if path.startswith("/evidence/"):
            # Only the already-public evidence website, never arbitrary files.
            root = (REPOSITORY / "site").resolve()
            target = (root / path[len("/evidence/"):]).resolve()
            if not target.is_relative_to(root) or any(p.startswith(".") for p in target.relative_to(root).parts):
                return self.reply(404, {"error": "Page not found."})
            if target.is_dir():
                target /= "index.html"
            return self.file(target)
        return self.reply(404, {"error": "Page not found."})

    def do_POST(self):
        if not self.valid_host():
            return self.reply(403, {"error": "Use the local SiteShift address."})
        origin = self.headers.get("Origin")
        allowed_origins = {f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"}
        if self.headers.get("X-SiteShift") != "local-run" or origin not in allowed_origins:
            return self.reply(403, {"error": "Start runs from the local SiteShift page."})
        if self.headers.get("Content-Type") != "application/json":
            return self.reply(415, {"error": "Expected JSON."})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 2 or length > 1024:
                raise ValueError("Request too large or empty.")
            body = json.loads(self.rfile.read(length))
            path = urlsplit(self.path).path
            if path == "/api/runs":
                if not isinstance(body, dict) or set(body) != {"situation"} or not isinstance(body["situation"], str):
                    raise ValueError("Choose a situation.")
                return self.reply(202, self.server.runner.start(body["situation"]))
            match = re.fullmatch(r"/api/runs/([0-9a-f]{32})/(cancel|analyze)", path)
            if match:
                identifier, action = match.groups()
                job = self.server.runner.get(identifier)
                if action == "cancel":
                    return self.reply(200, self.server.runner.cancel(identifier))
                if job["status"] not in {"completed", "different"}:
                    return self.reply(409, {"error": "Let the simulation finish before running its analysis."})
                return self.reply(200, analyze_run(job["directory"]))
            return self.reply(404, {"error": "Action not found."})
        except (ValueError, json.JSONDecodeError) as exc:
            return self.reply(400, {"error": str(exc)})
        except RuntimeError as exc:
            return self.reply(409, {"error": str(exc)})
        except KeyError:
            return self.reply(404, {"error": "Run not found."})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    runner = Runner()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.runner = runner
    print(f"SiteShift is ready at http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        runner.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
