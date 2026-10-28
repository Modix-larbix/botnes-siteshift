"""Botnes SiteShift: write recorded/ from the page's own data, or check that recorded/ is exactly that data.

The two real Nemotron calls of the three situations (scenes A4 and A1) are already in the page data
(site/data/episodes.js: each demo episode's robot.task_choice and cost). This tool takes them out of it, unchanged, as
two small JSON files per call, so that the exact request and the exact answer can be read without the page:

  recorded/<episode id>/choice_request.json   what Nemotron received (the request body) and the menu the code allowed
  recorded/<episode id>/choice_response.json  what Nemotron returned (exact answer text), its schema check, the cost

Only "model", "temperature" and "max_completion_tokens" are not written as such in the page data: the model id is the
episode's cost.model_id, and temperature 0 and 16384 output tokens are the demo's fixed values. They are proven by the
fingerprint: sha256 of the request as canonical JSON must equal the request_sha256 that the page data carries.

Checks (each one an error, exit 1): the fingerprint; the system text equal to choice_prompt.SYSTEM; the user text equal
to what choice_prompt.build_choice_messages rebuilds from its own task and scene parts; the answer's schema check
(choice_prompt.parse_answer, finish reason "stop") equal to the check the page data carries; the call date and the
model id equal in the episode's choice and cost records.

usage (from the repository root or anywhere; standard library only, no network):
  python3 -B tools/extract_recorded.py            write recorded/<id>/ (refuses to overwrite a different file)
  python3 -B tools/extract_recorded.py --check    rebuild in memory and compare byte for byte with recorded/
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "nemotron_client"))
import choice_prompt as CP  # noqa: E402

EPISODES_JS = REPO / "site" / "data" / "episodes.js"
RECORDED = REPO / "recorded"
HEAD = re.compile(r"^(?://[^\n]*\n)*window\.SITESHIFT = ")
DEMO_TEMPERATURE = 0.0
DEMO_MAX_COMPLETION_TOKENS = 16384
FINGERPRINT_RULE = ("sha256 of json.dumps(request, sort_keys=True, separators=(',', ':'), ensure_ascii=False) "
                    "encoded in UTF-8")
REQUEST_SCHEMA = "siteshift-recorded-choice-request-1"
RESPONSE_SCHEMA = "siteshift-recorded-choice-response-1"


def fingerprint(request: dict) -> str:
    payload = json.dumps(request, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_page_data(path: Path = EPISODES_JS) -> dict:
    text = path.read_text(encoding="utf-8")
    m = HEAD.match(text)
    if not m:
        raise SystemExit(f"{path}: unexpected head")
    return json.loads(text[m.end():].rstrip().rstrip(";"))


def split_user_text(user: str) -> tuple[str, str]:
    """(task, scene) of a choice request's user text, so that build_choice_messages can rebuild it."""
    head = "TASK (public statement): "
    tail = "\n\n" + CP.menu_text() + "\n\n" + CP.ANSWER_LINE
    if not user.startswith(head) or not user.endswith(tail):
        raise ValueError("user text: not built by choice_prompt (head or tail differs)")
    task, sep, scene = user[len(head):-len(tail)].partition("\n\n")
    if not sep:
        raise ValueError("user text: no scene part")
    return task, scene


def records(data: dict) -> dict:
    """{relative path: bytes} of recorded/, and the problems found."""
    out, problems = {}, []
    for e in data["episodes"]:
        if (e.get("source") or {}).get("key") != "demoA":
            continue
        tc = (e.get("robot") or {}).get("task_choice") or {}
        if tc.get("who_chose") != "nemotron":
            continue  # scene A5: the rule chose, no model call, nothing to record
        eid, cost, req = e["id"], e.get("cost") or {}, tc.get("request") or {}
        messages = [{"role": m["role"], "content": m["exact_text"]} for m in req.get("messages") or []]
        request = {"model": cost.get("model_id"), "messages": messages, "temperature": DEMO_TEMPERATURE,
                   "max_completion_tokens": DEMO_MAX_COMPLETION_TOKENS}
        fp = fingerprint(request)
        if fp != req.get("request_sha256"):
            problems.append(f"{eid}: recomputed fingerprint {fp} is not the page's {req.get('request_sha256')}")
        if [m["role"] for m in messages] != ["system", "user"] or messages[0]["content"] != CP.SYSTEM:
            problems.append(f"{eid}: messages are not [choice_prompt.SYSTEM, user]")
        else:
            try:
                task, scene = split_user_text(messages[1]["content"])
                if CP.build_choice_messages(task, scene) != messages:
                    problems.append(f"{eid}: choice_prompt does not rebuild the user text")
            except ValueError as err:
                problems.append(f"{eid}: {err}")
        ans = tc.get("answer") or {}
        answer_text = (ans.get("answer_text") or {}).get("exact_text")
        page_check = {"ok": ans.get("ok"), "option": ans.get("option"),
                      "reason": (ans.get("reason") or {}).get("exact_text"), "refusal": ans.get("refusal")}
        if CP.parse_answer(answer_text, "stop") != page_check:
            problems.append(f"{eid}: parse_answer differs from the page's check")
        if tc.get("call_date") != cost.get("call_date"):
            problems.append(f"{eid}: call dates differ between the choice and the cost records")
        options = tc.get("options") or []
        valid = [o["option"] for o in options if o.get("valid")]
        req_doc = {
            "schema": REQUEST_SCHEMA,
            "episode_id": eid,
            "scene": e.get("scene"),
            "kitchen": e.get("kitchen"),
            "label": e.get("label"),
            "extracted_from": f"site/data/episodes.js, episode {eid}: robot.task_choice.request (messages, "
                              "request_sha256), robot.task_choice.options and conditions, cost.model_id",
            "choice_prompt_version": CP.CHOICE_PROMPT_VERSION,
            "system_sha256": CP.SYSTEM_SHA256,
            "request_sha256": req.get("request_sha256"),
            "fingerprint_rule": FINGERPRINT_RULE,
            "text_only": req.get("text_only"),
            "request": request,
            "menu": {"rule_text": tc.get("rule_text"), "options": options, "valid_options": valid,
                     "conditions": tc.get("conditions"), "who_chose": tc.get("who_chose"),
                     "who_chose_text": tc.get("who_chose_text")},
            "note": tc.get("note"),
        }
        resp_doc = {
            "schema": RESPONSE_SCHEMA,
            "episode_id": eid,
            "label": e.get("label"),
            "extracted_from": f"site/data/episodes.js, episode {eid}: robot.task_choice.answer, call_note, "
                              "call_date, and cost",
            "request_sha256": req.get("request_sha256"),
            "model_id": cost.get("model_id"),
            "call_date": tc.get("call_date"),
            "call_note": tc.get("call_note"),
            "answer_text": answer_text,
            "answer_sha256": hashlib.sha256(answer_text.encode("utf-8")).hexdigest() if isinstance(answer_text, str)
            else None,
            "check": page_check,
            "option_allowed_by_the_robot_state": page_check["option"] in valid,
            "cost": {"model_calls": cost.get("model_calls"), "settled_usd": cost.get("settled_usd"),
                     "cost_status": cost.get("cost_status"), "note": cost.get("note")},
            "not_in_this_record": "token usage, latency and the model's separate reasoning text: they are in "
                                  "simulator/recorded_choices/ (choice_call.json, choice_response.json); the "
                                  "provider's raw response is not published; the page data keeps the answer text, "
                                  "its check and the settled cost",
        }
        for name, doc in (("choice_request.json", req_doc), ("choice_response.json", resp_doc)):
            out[f"{eid}/{name}"] = (json.dumps(doc, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    if len({k.split("/")[0] for k in out}) != 2:
        problems.append(f"{len(out) // 2} recorded calls found in the page data, not 2")
    return {"files": out, "problems": problems}


def main(argv: list) -> int:
    check = "--check" in argv
    rep = records(load_page_data())
    files, problems = rep["files"], list(rep["problems"])
    if check:
        on_disk = {p.relative_to(RECORDED).as_posix() for p in RECORDED.rglob("*.json")}
        for rel, data in files.items():
            p = RECORDED / rel
            if not p.is_file() or p.read_bytes() != data:
                problems.append(f"recorded/{rel}: differs from the page data")
        for rel in sorted(on_disk - set(files)):
            problems.append(f"recorded/{rel}: not from the page data")
    elif not problems:
        for rel, data in files.items():
            p = RECORDED / rel
            if p.exists() and p.read_bytes() != data:
                problems.append(f"recorded/{rel}: exists and differs; not overwritten")
                continue
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
    print(json.dumps({"mode": "check" if check else "write", "files": sorted(files),
                      "sha256": {k: hashlib.sha256(v).hexdigest() for k, v in sorted(files.items())},
                      "problems": problems}, indent=1))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
