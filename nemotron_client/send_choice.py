"""Botnes SiteShift: show (dry run) or send (--execute) ONE recorded choice request to NVIDIA Nemotron 3 Super on
Nebius Token Factory.

Standard library only (tested with Python 3.9 and 3.12). It reads the request given, choice_prompt.py and, if present,
the recorded answer next to the request (choice_response.json). It writes nothing, except with --execute --save-dir,
and then only inside the new folder given.

usage:
  python3 nemotron_client/send_choice.py recorded/demoA_A1_31000/choice_request.json
      DRY RUN (the default): checks the request, then prints the endpoint, the headers and the exact bytes of the
      body that --execute would send, with their sha256 (the request's fingerprint). Nothing is sent, nothing is
      written, and the key is not read: no key is needed.
  python3 nemotron_client/send_choice.py recorded/demoA_A1_31000/choice_request.json --execute --save-dir NEW_FOLDER
      ONE real, paid call: reads the key from the environment variable NEBIUS_API_KEY (or the variable named by
      --key-env) at run time, creates NEW_FOLDER and records the attempt in it, then sends one POST to
      <base-url>chat/completions with no retry and without following any redirect, and prints the answer, its schema
      check, whether the option is one the robot's state allowed, the usage the provider reported and the cost at the
      given prices. The key is never printed, logged or written, and it is sent only to the endpoint given: a redirect
      (3xx) is refused and reported, never followed. --save-dir can be left out, but then nothing about the attempt
      is kept on disk and nothing prevents a second send.
      Status: this public client is tested offline only (fake transport, see tests/). The two recorded calls were
      sent by the private lab's client; this one has not been used for a real call.

Checks made before anything is sent (each refusal prints "refused: ..." on stderr, exits with code 2 and sends
nothing):
  - the request file holds the choice request of choice_prompt (same prompt version and system text sha256);
  - the request has exactly the keys model, messages, temperature, max_completion_tokens; two text-only messages,
    [system, user], with role and content only; the system text is choice_prompt.SYSTEM and the user text is what
    choice_prompt.build_choice_messages rebuilds from its own task and scene parts;
  - the recomputed fingerprint (sha256 of the canonical body below) equals the file's request_sha256 and, if given,
    --expect-sha256;
  - --execute only: the key is present in the environment and is one line with no space;
  - --execute --save-dir only: the folder did not exist, and its parent is a folder (created if missing, refused if it
    is a file); the folder is then created and the request and the attempt marker below are written in it.

Attempt record (--execute --save-dir NEW_FOLDER). The folder is reserved BEFORE anything is sent:
  request_body.json   the exact bytes sent (their sha256 is the fingerprint); written before the send
  attempt.json        the attempt marker, written (and flushed to disk) before the send: "status": "sending", the UTC
                      start time, the endpoint, the model, the fingerprint, the sha256 of this file and of
                      choice_prompt.py; never rewritten
  outcome.json        written after the send: the final status (below), the UTC end time, the HTTP status if any
  response_body.json  HTTP 200 only: the provider's response body (no header), any copy of the key replaced by
                      [redacted]
  result.json         a chat completion only: the same JSON as printed (check, option allowed, usage, cost)
  error_body.txt      any other HTTP status: the response body, redacted the same way
The "status" in outcome.json:
  answered               a chat completion came back and was checked; read check.ok and
                         option_allowed_by_the_robot_state in result.json before using the option
  http_error             the endpoint answered with an HTTP error status
  redirect_refused       the endpoint answered with a redirect (3xx), which was not followed
  not_a_chat_completion  HTTP 200, but the body is not a chat completion
  uncertain              no answer came back (timeout, connection lost...): the request may or may not have reached
                         the provider, and may have been billed
attempt.json without outcome.json is also uncertain: the run stopped during or just after the send.
os.mkdir creates the folder or fails if it exists, in one step: of two runs given the same folder, at the same time or
one after the other, only the first creates it and only that one sends; the other is refused before sending. This
client never retries and never deletes or rewrites a record. After an error or an interruption, keep the folder as it
is and look at it before deciding anything; a new attempt needs a new folder. This protects one folder against a second
send; it is not a de-duplication on the provider's side.

Exit codes:
  0  the dry run finished, or (--execute) a chat completion came back and was processed. 0 does NOT mean the option
     is accepted: read check.ok and option_allowed_by_the_robot_state in the printed result (and result.json). When
     either is false, the demo's code refuses the answer and the robot asks for help.
  2  either refused before anything was sent ("refused: ..." on stderr; a malformed command line also exits with 2),
     or (--execute) the request was sent, or may have been, and no usable answer came back: transport error, HTTP
     error, redirect refused, body that is not a chat completion, or a record that could not be finished. A 2 is not
     proof that nothing was sent or billed: the lines printed before it, and outcome.json, say which case it is.
  1  an unexpected Python error (traceback). If "sending ONE request" was printed before it, treat it as uncertain.

Documented parameters (defaults in brackets):
  --base-url      the OpenAI-compatible Token Factory endpoint [https://api.tokenfactory.nebius.com/v1/]; https only,
                  plain http only when the host is exactly 127.0.0.1, ::1 or localhost (local tests); no user or
                  password in the address, no query, no fragment
  --key-env       the name of the environment variable that holds the key [NEBIUS_API_KEY]
  --timeout-s     the request timeout in seconds [120]
  --price-in      USD per million input tokens [0.30]   (the reviewed prices of nvidia/nemotron-3-super-120b-a12b
  --price-out     USD per million output tokens [0.90]   used for the two recorded calls; check the current list)
  --save-dir      with --execute: a NEW folder, created before the send, that keeps the attempt record above

The body sent is the canonical JSON of the request (keys sorted, no spaces, UTF-8), so its sha256 is the request's
fingerprint. The two recorded calls were sent by the private lab's guarded client through the OpenAI Python SDK with the
same JSON object (its byte layout may differ); that client is not published.
Nemotron answers again when you send again: at temperature 0 the answer may still differ from the recorded one.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import http.client
import json
import os
import platform
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import choice_prompt as CP  # noqa: E402

DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
DEFAULT_KEY_ENV = "NEBIUS_API_KEY"
DEFAULT_PRICE_IN = "0.30"
DEFAULT_PRICE_OUT = "0.90"
REQUEST_KEYS = {"model", "messages", "temperature", "max_completion_tokens"}
LOCAL_HTTP_HOSTS = {"127.0.0.1", "::1", "localhost"}  # plain http is allowed only for these exact host names
USER_AGENT = "botnes-siteshift-choice-client/1.0"
KEY_SHOWN_AS = "<the value of ${env}, read only with --execute, never printed>"
ATTEMPT_SCHEMA = "siteshift-choice-attempt-1"


class Refused(Exception):
    """Nothing is sent; the message says why."""


def canonical_body(request: dict) -> bytes:
    """The bytes sent: JSON with sorted keys and no spaces, UTF-8. Their sha256 is the request's fingerprint."""
    return json.dumps(request, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def fingerprint(request: dict) -> str:
    return hashlib.sha256(canonical_body(request)).hexdigest()


def split_user_text(user: str) -> tuple:
    """(task, scene) of a choice request's user text."""
    head = "TASK (public statement): "
    tail = "\n\n" + CP.menu_text() + "\n\n" + CP.ANSWER_LINE
    if not user.startswith(head) or not user.endswith(tail):
        raise Refused("the user text was not built by choice_prompt (its head or tail differs)")
    task, sep, scene = user[len(head):-len(tail)].partition("\n\n")
    if not sep:
        raise Refused("the user text has no scene part")
    return task, scene


def check_request(doc, expect_sha256=None) -> tuple:
    """(request, fingerprint, valid options) when every check passes; Refused otherwise."""
    if not isinstance(doc, dict):
        raise Refused("the request file is not a JSON object")
    if doc.get("choice_prompt_version") != CP.CHOICE_PROMPT_VERSION or doc.get("system_sha256") != CP.SYSTEM_SHA256:
        raise Refused("not a choice request of this choice_prompt version (version or system text sha256)")
    req = doc.get("request")
    if not isinstance(req, dict) or set(req) != REQUEST_KEYS:
        raise Refused(f"request keys must be exactly {sorted(REQUEST_KEYS)}")
    msgs = req["messages"]
    if not isinstance(msgs, list) or len(msgs) != 2 or not all(isinstance(m, dict) for m in msgs) \
            or [sorted(m) for m in msgs] != [["content", "role"]] * 2 \
            or [m["role"] for m in msgs] != ["system", "user"] \
            or not all(isinstance(m["content"], str) for m in msgs):
        raise Refused("messages must be [system, user], text only, with the keys role and content only")
    if msgs[0]["content"] != CP.SYSTEM:
        raise Refused("the system text is not choice_prompt.SYSTEM")
    task, scene = split_user_text(msgs[1]["content"])
    try:
        rebuilt = CP.build_choice_messages(task, scene)
    except CP.ChoiceTextError as e:
        raise Refused(f"choice_prompt refuses the user text ({e})") from None
    if rebuilt != msgs:
        raise Refused("choice_prompt does not rebuild these messages")
    if not isinstance(req["model"], str) or not req["model"]:
        raise Refused("model must be a non-empty string")
    if req["temperature"] != 0 or isinstance(req["temperature"], bool):
        raise Refused("temperature must be 0 (the demo's value)")
    if not isinstance(req["max_completion_tokens"], int) or isinstance(req["max_completion_tokens"], bool) \
            or not 1 <= req["max_completion_tokens"] <= 16384:
        raise Refused("max_completion_tokens must be an integer from 1 to 16384")
    fp = fingerprint(req)
    seen = {"recomputed": fp, "request_sha256": doc.get("request_sha256")}
    if expect_sha256 is not None:
        seen["--expect-sha256"] = expect_sha256
    if len(set(seen.values())) != 1:
        raise Refused(f"fingerprint mismatch: {seen}")
    valid = (doc.get("menu") or {}).get("valid_options")
    if not isinstance(valid, list) or not valid or not set(valid) <= set(CP.OPTIONS):
        raise Refused(f"menu.valid_options must be a non-empty subset of {list(CP.OPTIONS)}")
    return req, fp, valid


def endpoint(base_url: str) -> str:
    """<base-url>chat/completions, after checking the parsed address (not a prefix of the text)."""
    try:
        parts = urllib.parse.urlsplit(base_url)
        host, _port = parts.hostname, parts.port  # .port raises ValueError on a malformed port
    except ValueError:
        raise Refused("--base-url is not a valid address") from None
    if not host:
        raise Refused("--base-url has no host")
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise Refused("--base-url must not hold a user name or password")
    if parts.query or parts.fragment:
        raise Refused("--base-url must not have a query or a fragment")
    if parts.scheme == "https":
        pass
    elif parts.scheme == "http" and host in LOCAL_HTTP_HOSTS:
        pass
    else:
        raise Refused("--base-url must use https:// (plain http only when the host is exactly 127.0.0.1, ::1 or "
                      "localhost)")
    return base_url.rstrip("/") + "/chat/completions"


def read_key(env_name: str) -> str:
    """The key from the environment, at run time only. Never printed: only "present" or a refusal."""
    key = os.environ.get(env_name)
    if not key:
        raise Refused(f"no key in the environment variable {env_name}; nothing sent")
    if key != key.strip() or any(c.isspace() for c in key):
        raise Refused(f"the value of {env_name} contains spaces or line breaks; nothing sent")
    return key


def redact(text: str, secret) -> str:
    """text with every copy of secret, as is or as written inside a JSON string, replaced by [redacted]."""
    if not secret:
        return text
    for form in {secret, json.dumps(secret)[1:-1], json.dumps(secret, ensure_ascii=False)[1:-1]}:
        if form:
            text = text.replace(form, "[redacted]")
    return text


def cost_usd(usage: dict, price_in: Decimal, price_out: Decimal):
    try:
        p, c = int(usage["prompt_tokens"]), int(usage["completion_tokens"])
    except (KeyError, TypeError, ValueError):
        return None
    return str(((Decimal(p) * price_in + Decimal(c) * price_out) / Decimal(1000000)).quantize(Decimal("0.0000001")))


def recorded_answer(request_path: Path):
    """The recorded answer next to the request (choice_response.json), when there is one."""
    p = request_path.parent / "choice_response.json"
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def dry_run(path: Path, req: dict, fp: str, valid: list, url: str, key_env: str, out=sys.stdout) -> int:
    body = canonical_body(req)
    w = out.write
    w("DRY RUN: nothing is sent, nothing is written, the key is not read.\n")
    w(f"request file: {path.name} (in {path.parent.name}/)\n")
    w(f"model: {req['model']}; temperature {req['temperature']}; max_completion_tokens {req['max_completion_tokens']};"
      f" text only, 2 messages\n")
    w(f"fingerprint, sha256 of the body below: {fp} (equal to the file's request_sha256)\n")
    w(f"endpoint: POST {url}\n")
    w("headers: Content-Type: application/json | Accept: application/json | User-Agent: " + USER_AGENT
      + " | Authorization: Bearer " + KEY_SHOWN_AS.replace("${env}", "$" + key_env) + "\n")
    w(f"body, exact bytes ({len(body)} bytes, UTF-8):\n")
    w(body.decode("utf-8") + "\n")
    w("\nthe same messages, as text:\n")
    for m in req["messages"]:
        w(f"----- {m['role']} -----\n{m['content']}\n")
    w("-----\n")
    w(f"options the robot's state allowed (checked by the demo's code before the call): {', '.join(valid)}\n")
    rec = recorded_answer(path)
    if rec and rec.get("request_sha256") == fp:
        chk = rec.get("check") or {}
        w(f"recorded answer of this request ({rec.get('call_date')}, {rec.get('model_id')}): option "
          f"{chk.get('option')}; reason: {chk.get('reason')}; settled cost {(rec.get('cost') or {}).get('settled_usd')}"
          " USD\n")
    w("to send it for real (one paid call, your own key in the environment): add --execute --save-dir NEW_FOLDER\n")
    return 0


class RefuseRedirect(urllib.request.HTTPRedirectHandler):
    """Never follow a redirect. The key goes to the endpoint given and nowhere else: a 3xx answer comes back as an
    HTTP error, is reported, and nothing more is sent."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def no_redirect_open(*handlers):
    """An urlopen-like function whose opener refuses every redirect. `handlers` are extra urllib handlers (the
    offline tests pass a fake transport here)."""
    return urllib.request.build_opener(RefuseRedirect, *handlers).open


def post_json(url: str, body: bytes, headers: dict, timeout: float, opener=None, auth=None) -> tuple:
    """(HTTP status, response body bytes). One attempt, no retry, no redirect followed.

    `auth` (the Authorization value) is attached as an unredirected header: even an opener that follows redirects
    would not copy it to another address."""
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    if auth is not None:
        req.add_unredirected_header("Authorization", auth)
    open_ = opener or no_redirect_open()
    try:
        with open_(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def utc_now() -> str:
    """The time now in UTC, ISO 8601 with milliseconds, for example 2026-10-28T09:15:02.123Z."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def json_text(obj) -> str:
    return json.dumps(obj, indent=1, ensure_ascii=False) + "\n"


def write_new(path: Path, data, flush_to_disk: bool = False) -> None:
    """Write a file that must not exist yet (mode "x"): a record is never overwritten."""
    with open(path, "xb") as f:
        f.write(data if isinstance(data, bytes) else data.encode("utf-8"))
        if flush_to_disk:
            f.flush()
            os.fsync(f.fileno())


def reserve_attempt(save_dir: Path, path: Path, req: dict, fp: str, url: str) -> None:
    """Create save_dir, then write the exact body and the attempt marker in it. Called before anything is sent, so every
    failure here is a refusal.

    os.mkdir either creates the folder or fails because it exists, in one step: when two runs are given the same
    folder, only one of them gets past this point and sends."""
    try:
        save_dir.parent.mkdir(parents=True, exist_ok=True)  # e.g. my_calls/ the first time; fails if it is a file
    except OSError as e:
        raise Refused(f"the parent of --save-dir {save_dir.name} is not a folder and cannot be created "
                      f"({type(e).__name__}); nothing sent") from None
    try:
        os.mkdir(save_dir)
    except FileExistsError:
        raise Refused(f"--save-dir {save_dir.name} exists: it may hold an earlier attempt, keep it as it is; each "
                      "attempt needs a new folder; nothing sent") from None
    except OSError as e:
        raise Refused(f"cannot create --save-dir {save_dir.name} ({type(e).__name__}); nothing sent") from None
    marker = {"schema": ATTEMPT_SCHEMA, "status": "sending", "started_at_utc": utc_now(), "endpoint": "POST " + url,
              "model": req["model"], "request_sha256": fp, "request_file": f"{path.parent.name}/{path.name}",
              "client_sha256": {"send_choice.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                "choice_prompt.py": hashlib.sha256(Path(CP.__file__).read_bytes()).hexdigest()},
              "python": platform.python_version(),
              "sending": "one POST, no retry, no redirect followed",
              "if_outcome_json_is_missing": "the run stopped during or just after the send: whether the provider "
                                            "received or billed the request is unknown; do not send it again before "
                                            "looking into it"}
    try:
        write_new(save_dir / "request_body.json", canonical_body(req))
        write_new(save_dir / "attempt.json", json_text(marker), flush_to_disk=True)
    except OSError as e:
        raise Refused(f"cannot write the attempt record in {save_dir.name}/ ({type(e).__name__}); the folder stays as "
                      "it is; nothing sent") from None


def finish(save_dir, outcome: dict, code: int, files: dict, out) -> int:
    """With --save-dir: write the response files, then outcome.json, next to the attempt marker. Returns the exit
    code."""
    if save_dir is None:
        return code
    record = {"schema": ATTEMPT_SCHEMA, **outcome, "finished_at_utc": utc_now(),
              "files": sorted(["request_body.json", "attempt.json", "outcome.json", *files])}
    try:
        for name, text in files.items():
            write_new(save_dir / name, text)
        write_new(save_dir / "outcome.json", json_text(record))
    except OSError as e:
        out.write(f"could not finish the record in {save_dir.name}/ ({type(e).__name__}): treat this attempt as "
                  "uncertain\n")
        return 2
    out.write(f"written in {save_dir.name}/: {', '.join(record['files'])} (status {outcome['status']}; no header; any "
              "copy of the key redacted)\n")
    return code


def execute(path: Path, req: dict, fp: str, valid: list, url: str, a, opener=None, out=sys.stdout) -> int:
    save_dir = Path(a.save_dir) if a.save_dir else None
    price_in, price_out = Decimal(a.price_in), Decimal(a.price_out)
    key = read_key(a.key_env)
    out.write(f"key present in ${a.key_env}: yes (not shown)\n")
    if save_dir is not None:
        reserve_attempt(save_dir, path, req, fp, url)
        out.write(f"attempt recorded before sending: {save_dir.name}/request_body.json and attempt.json\n")
    else:
        out.write("no --save-dir: nothing about this attempt is kept on disk\n")
    headers = {"Content-Type": "application/json", "Accept": "application/json", "User-Agent": USER_AGENT}
    body = canonical_body(req)
    out.write(f"sending ONE request {fp[:16]}... to POST {url} (no retry, no redirect followed, timeout "
              f"{a.timeout_s} s)\n")
    t0 = time.monotonic()
    try:
        status, raw = post_json(url, body, headers, a.timeout_s, opener, auth="Bearer " + key)
    except (urllib.error.URLError, OSError, http.client.HTTPException, ValueError) as e:
        reason = redact(type(e).__name__ + ": " + str(getattr(e, "reason", e)), key)[:300]
        del key
        out.write(f"transport error, no answer: {reason}\n")
        out.write("the request may or may not have reached the provider, and may have been billed: do not send it "
                  "again before looking into it\n")
        return finish(save_dir, {"status": "uncertain", "error": reason}, 2, {}, out)
    latency = round(time.monotonic() - t0, 3)
    # Everything printed or written from here on comes from this redacted text; the key is then dropped.
    text = redact(raw.decode("utf-8", errors="replace"), key)
    del key
    if 300 <= status < 400:
        out.write(f"HTTP {status}: redirect refused, not followed (the key is sent only to {url}); no answer: "
                  f"{text[:300]}\n")
        return finish(save_dir, {"status": "redirect_refused", "http_status": status, "latency_s": latency}, 2,
                      {"error_body.txt": text}, out)
    if status != 200:
        out.write(f"HTTP {status}, no answer: {text[:500]}\n")
        return finish(save_dir, {"status": "http_error", "http_status": status, "latency_s": latency}, 2,
                      {"error_body.txt": text}, out)
    try:
        data = json.loads(text)
        choice = data["choices"][0]
        message = choice.get("message") or {}
        content, finish_reason = message.get("content"), choice.get("finish_reason")
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        out.write(f"HTTP 200 but not a chat completion: {text[:300]}\n")
        return finish(save_dir, {"status": "not_a_chat_completion", "http_status": 200, "latency_s": latency}, 2,
                      {"response_body.json": text}, out)
    usage = data.get("usage") or {}
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    check = CP.parse_answer(content, finish_reason)
    allowed = bool(check["ok"]) and check["option"] in valid
    result = {"status": "answered", "model": data.get("model"), "finish_reason": finish_reason, "latency_s": latency,
              "answer_text": content, "check": check, "option_allowed_by_the_robot_state": allowed,
              "if_not_allowed": None if allowed else "the demo's code refuses the answer and the robot asks for help",
              "usage": usage, "reasoning_chars": len(reasoning) if isinstance(reasoning, str) else None,
              "cost_usd_at_given_prices": cost_usd(usage, price_in, price_out),
              "prices_usd_per_million_tokens": {"input": a.price_in, "output": a.price_out},
              "request_sha256": fp}
    rec = recorded_answer(path)
    if rec and rec.get("request_sha256") == fp:
        result["recorded_option"] = (rec.get("check") or {}).get("option")
        result["same_option_as_recorded"] = result["recorded_option"] == check["option"]
    out.write(json_text(result))
    outcome = {"status": "answered", "http_status": 200, "latency_s": latency, "check_ok": bool(check["ok"]),
               "option_allowed_by_the_robot_state": allowed}
    return finish(save_dir, outcome, 0, {"response_body.json": text, "result.json": json_text(result)}, out)


def main(argv=None, opener=None, out=None) -> int:
    out = out or sys.stdout
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("request", help="a choice_request.json, e.g. recorded/demoA_A1_31000/choice_request.json")
    ap.add_argument("--execute", action="store_true", help="send ONE real, paid call (key from the environment)")
    ap.add_argument("--expect-sha256", help="refuse unless the request's fingerprint is this one")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--key-env", default=DEFAULT_KEY_ENV)
    ap.add_argument("--timeout-s", type=float, default=120.0)
    ap.add_argument("--price-in", default=DEFAULT_PRICE_IN, help="USD per million input tokens")
    ap.add_argument("--price-out", default=DEFAULT_PRICE_OUT, help="USD per million output tokens")
    ap.add_argument("--save-dir", help="with --execute: a NEW folder, created before the send, that keeps the attempt "
                                       "record (request, marker, outcome, response; no header, key redacted)")
    a = ap.parse_args(argv)
    path = Path(a.request)
    try:
        if a.save_dir and not a.execute:
            raise Refused("--save-dir is for --execute only (a dry run writes nothing)")
        if not (0 < a.timeout_s <= 600):
            raise Refused("--timeout-s must be between 0 and 600")
        try:
            Decimal(a.price_in), Decimal(a.price_out)
        except ArithmeticError:
            raise Refused("--price-in and --price-out must be numbers") from None
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise Refused(f"cannot read the request file ({type(e).__name__})") from None
        req, fp, valid = check_request(doc, a.expect_sha256)
        url = endpoint(a.base_url)
        if not a.execute:
            return dry_run(path, req, fp, valid, url, a.key_env, out)
        return execute(path, req, fp, valid, url, a, opener, out)
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
