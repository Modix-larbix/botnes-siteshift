"""The text of the choice request: the only text a language model received in the demo.

Standard library only: this module imports nothing of the robot code, sends nothing and reads no file.

The model receives text only: a short system message (the rules, a menu of two options, the JSON answer format) and a
user message (the public task statement, the SCENE STATE text the robot wrote at that moment, verbatim, and the menu).
No image. It chooses between two fixed team programs:
  scrub_bowl   "look again, pick the sponge, scrub the bowl on the landing tray"
  ask_help     stop and ask a person, listing the zones still to check
report_not_found is never in this menu: it is the inspection program's own rule, checked by the robot's claim guard
with no model call.

The answer must be one JSON object {"option": ..., "reason": ...} after any completed reasoning block. parse_answer()
checks the schema only; the demo client checks the option's condition again against the robot's own SCENE STATE text
(demo_client.menu_conditions) and refuses anything else (the robot then asks for help).
"""
from __future__ import annotations

import hashlib
import json
import re

CHOICE_PROMPT_VERSION = "demoA-choice-1"
SCHEMA = "demoA-choice-request-1"
OPTIONS = ("scrub_bowl", "ask_help")
ANSWER_KEYS = frozenset({"option", "reason"})
MAX_REASON_CHARS = 300
MAX_TASK_CHARS = 400
SCENE_PREFIX = "SCENE STATE NOW"

MENU = {
    "scrub_bowl": ("a fixed program written by the team: look at the sponge's region again and observe, pick the "
                   "sponge the robot tracks, find the bowl on the landing tray and scrub it for 3 s; the robot's own "
                   "guards check the pick and the end of the task"),
    # help_v1.bs lists the zones with the robot's own fields; how to check each zone is the interface's text, not the
    # program's (open point 2), so the menu does not promise it
    "ask_help": "a fixed program written by the team: stop and ask a person for help, listing the zones still to check",
}

SYSTEM = (
    "You choose the next step of a simulated kitchen robot (simulation, simulated perception) from a closed menu of "
    "two options. Each option runs a fixed program written by the team; you do not write a program.\n"
    "Rules:\n"
    "- Read the robot's own SCENE STATE text in the user message. It is the robot's reading, as text; there are no "
    "images.\n"
    "- scrub_bowl is valid only if the robot tracks a sponge it is not holding, the line \"hand:\" reads "
    "\"holding no\", and it reports 0 contact incidents and 0 grip events. The code checks these conditions again "
    "and refuses scrub_bowl when one of them does not hold.\n"
    "- ask_help is always valid.\n"
    "- Answer with one JSON object and nothing else: {\"option\": \"scrub_bowl\" or \"ask_help\", \"reason\": "
    "\"one short sentence\"}. The reason has at most 300 characters."
)
SYSTEM_SHA256 = hashlib.sha256(SYSTEM.encode("utf-8")).hexdigest()

MENU_HEADER = "MENU (choose exactly one option):"
ANSWER_LINE = 'Answer with the JSON object only: {"option": "...", "reason": "..."}.'


class ChoiceTextError(ValueError):
    """The inputs of a choice request are not what the demo expects (nothing is built)."""


def menu_text() -> str:
    return MENU_HEADER + "\n" + "\n".join(f"- {k}: {MENU[k]}" for k in OPTIONS)


def build_choice_messages(task_text: str, scene_text: str) -> list[dict]:
    """[system, user] messages of the choice request. task_text: the public task statement (the loop request's own
    TASK part, its "text"); scene_text: the SCENE STATE block of the loop request, verbatim (never changed here)."""
    if not isinstance(task_text, str) or not task_text.strip() or len(task_text) > MAX_TASK_CHARS:
        raise ChoiceTextError(f"task text must be a non-empty str of at most {MAX_TASK_CHARS} characters")
    if not isinstance(scene_text, str) or not scene_text.startswith(SCENE_PREFIX):
        raise ChoiceTextError(f"scene text must start with {SCENE_PREFIX!r}")
    user = "\n\n".join([f"TASK (public statement): {task_text}", scene_text, menu_text(), ANSWER_LINE])
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def fixed_texts() -> list[str]:
    """The texts this module writes (system, menu, fixed lines): what a forbidden-word test reads. The SCENE STATE
    text is the robot's, shown verbatim in an "exact text" block, and is not part of it."""
    return [SYSTEM, menu_text(), ANSWER_LINE, "TASK (public statement): "]


_FENCE = re.compile(r"```|~~~")
_JSON_FENCE = re.compile(r"\A```(?:json)?[ \t]*\n(.*)\n[ \t]*```\Z", re.DOTALL)


def parse_answer(text, finish_reason) -> dict:
    """Schema check of Nemotron's answer: {"ok", "option", "reason", "refusal"}. refusal is a short code when the answer
    is cut (finish_reason other than "stop"), has an unfinished reasoning block, is not exactly one JSON object with
    exactly the keys option and reason, names an option outside the menu, or has no readable reason. The option's
    condition is checked by the caller (demo_client), never here."""
    def no(code: str) -> dict:
        return {"ok": False, "option": None, "reason": None, "refusal": code}
    if finish_reason != "stop":
        return no("cut_or_stopped:" + str(finish_reason)[:20])
    if not isinstance(text, str):
        return no("no_text")
    answer = text.split("</think>", 1)[-1].strip()
    if not answer or re.search(r"</?think\b", answer, flags=re.IGNORECASE):
        return no("unfinished_reasoning")
    fences = _FENCE.findall(answer)
    if fences:
        m = _JSON_FENCE.match(answer)
        if len(fences) != 2 or m is None:
            return no("not_one_json_object")
        answer = m.group(1).strip()
    try:
        obj = json.loads(answer)
    except ValueError:
        return no("not_one_json_object")
    if not isinstance(obj, dict):
        return no("not_one_json_object")
    if set(obj) != ANSWER_KEYS:
        return no("keys_not_option_and_reason")
    option, reason = obj["option"], obj["reason"]
    if not isinstance(option, str) or option not in OPTIONS:
        return no("option_outside_menu")
    if not isinstance(reason, str) or not reason.strip() or len(reason) > MAX_REASON_CHARS \
            or not reason.isprintable():
        return no("reason_unreadable")
    return {"ok": True, "option": option, "reason": reason.strip(), "refusal": None}
