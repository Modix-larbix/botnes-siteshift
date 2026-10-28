"""Offline tests of the Nemotron choice client and of recorded/. No network, no key, no file written in the repository.

The --execute path is tested with a FAKE transport (nothing leaves the machine) and a FAKE key held in a test-only
environment variable, never NEBIUS_API_KEY. The redirect tests run the real urllib handler chain with a fake HTTPS
handler in place of the network. The attempt-record tests (--save-dir) write only in temporary folders.

usage: python3 -B tests/test_nemotron_client.py      (standard library only)
"""
from __future__ import annotations

import copy
import email.message
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
import urllib.response
from contextlib import ExitStack, redirect_stderr
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "nemotron_client"))
import choice_prompt as CP  # noqa: E402
import send_choice as SC  # noqa: E402

RECORDED = {"demoA_A4_31000": ("37b01a38b0bb034786b1997d5b988ef3bfa60dce3e41833e891858e38950ab93", "0.0003930"),
            "demoA_A1_31000": ("47da009a38512d10e505f3e1e18bb765f1d1360223f3d31ad66491d76683efb7", "0.0006012")}
FAKE_ENV = "SITESHIFT_TEST_FAKE_KEY"
FAKE_KEY = "fake-test-value-not-a-key-0123456789"


def request_path(eid: str) -> Path:
    return REPO / "recorded" / eid / "choice_request.json"


def no_network(*args, **kwargs):
    raise AssertionError("the network was used")


def no_net():
    """Any attempt to open a connection through urllib's default paths fails the test."""
    stack = ExitStack()
    stack.enter_context(mock.patch.object(urllib.request, "urlopen", no_network))
    stack.enter_context(mock.patch.object(SC, "no_redirect_open", lambda *handlers: no_network))
    return stack


class FakeResponse:
    def __init__(self, status: int, body: bytes):
        self.status, self._body = status, body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeHTTPS(urllib.request.HTTPSHandler):
    """A fake transport inside urllib's real handler chain: the endpoint answers 302 to another host, which would
    answer 200. Records every request that reaches it; no socket is opened."""

    ENDPOINT = "https://api.tokenfactory.nebius.com/v1/chat/completions"
    ELSEWHERE = "https://elsewhere.invalid/collect"

    def __init__(self):
        super().__init__()
        self.seen = []

    def https_open(self, req):
        self.seen.append({"url": req.full_url, "method": req.get_method(), "headers": dict(req.header_items())})
        hdrs = email.message.Message()
        if req.full_url == self.ENDPOINT:
            hdrs["Location"] = self.ELSEWHERE
            resp = urllib.response.addinfourl(io.BytesIO(b"moved"), hdrs, req.full_url, 302)
            resp.msg = "Found"
        else:
            resp = urllib.response.addinfourl(io.BytesIO(b"{}"), hdrs, req.full_url, 200)
            resp.msg = "OK"
        return resp


class FakeOpener:
    """Records what would have been sent; answers with a fixed chat completion."""

    def __init__(self, content: str, usage: dict, status: int = 200):
        self.content, self.usage, self.status, self.sent = content, usage, status, []

    def __call__(self, req: urllib.request.Request, timeout: float):
        self.sent.append({"url": req.full_url, "method": req.get_method(), "body": req.data,
                          "headers": dict(req.header_items()), "timeout": timeout})
        body = {"model": "nvidia/nemotron-3-super-120b-a12b",
                "choices": [{"message": {"content": self.content, "reasoning_content": "x" * 10},
                             "finish_reason": "stop"}], "usage": self.usage}
        return FakeResponse(self.status, json.dumps(body).encode("utf-8"))


def run(argv, opener=None):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stderr(err):
        code = SC.main(argv, opener=opener, out=out)
    return code, out.getvalue(), err.getvalue()


class Recorded(unittest.TestCase):
    def test_two_recorded_calls_with_their_fingerprints(self):
        for eid, (sha, usd) in RECORDED.items():
            doc = json.loads(request_path(eid).read_text(encoding="utf-8"))
            self.assertEqual(SC.fingerprint(doc["request"]), sha)
            self.assertEqual(doc["request_sha256"], sha)
            resp = json.loads((request_path(eid).parent / "choice_response.json").read_text(encoding="utf-8"))
            self.assertEqual(resp["request_sha256"], sha)
            self.assertEqual(resp["cost"]["settled_usd"], usd)
            self.assertEqual(hashlib.sha256(resp["answer_text"].encode("utf-8")).hexdigest(), resp["answer_sha256"])

    def test_total_cost_of_the_two_calls(self):
        from decimal import Decimal
        total = sum(Decimal(v[1]) for v in RECORDED.values())
        self.assertEqual(str(total), "0.0009942")

    def test_messages_are_rebuilt_by_choice_prompt(self):
        for eid in RECORDED:
            msgs = json.loads(request_path(eid).read_text(encoding="utf-8"))["request"]["messages"]
            task, scene = SC.split_user_text(msgs[1]["content"])
            self.assertEqual(CP.build_choice_messages(task, scene), msgs)

    def test_recorded_answers_pass_the_schema_check(self):
        for eid in RECORDED:
            resp = json.loads((request_path(eid).parent / "choice_response.json").read_text(encoding="utf-8"))
            self.assertEqual(CP.parse_answer(resp["answer_text"], "stop"), resp["check"])
            self.assertEqual(resp["check"]["option"], "scrub_bowl")
            self.assertTrue(resp["option_allowed_by_the_robot_state"])

    def test_recorded_folder_is_the_page_data(self):
        p = subprocess.run([sys.executable, "-B", str(REPO / "tools" / "extract_recorded.py"), "--check"],
                           capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertEqual(json.loads(p.stdout)["problems"], [])


class DryRun(unittest.TestCase):
    def test_dry_run_shows_the_exact_body_and_sends_nothing(self):
        with mock.patch.dict(os.environ, {FAKE_ENV: FAKE_KEY}), no_net():
            for eid, (sha, _) in RECORDED.items():
                code, out, err = run([str(request_path(eid)), "--key-env", FAKE_ENV, "--expect-sha256", sha])
                self.assertEqual(code, 0, err)
                body_line = out.split("UTF-8):\n", 1)[1].split("\n", 1)[0]
                self.assertEqual(hashlib.sha256(body_line.encode("utf-8")).hexdigest(), sha)
                self.assertIn("POST https://api.tokenfactory.nebius.com/v1/chat/completions", out)
                self.assertIn("DRY RUN: nothing is sent", out)
                self.assertNotIn(FAKE_KEY, out + err)

    def test_dry_run_refuses_save_dir(self):
        code, out, err = run([str(request_path("demoA_A1_31000")), "--save-dir", "x"])
        self.assertEqual(code, 2)
        self.assertIn("refused", err)


class Refusals(unittest.TestCase):
    def tampered(self, change) -> int:
        doc = json.loads(request_path("demoA_A1_31000").read_text(encoding="utf-8"))
        change(doc)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "choice_request.json"
            p.write_text(json.dumps(doc), encoding="utf-8")
            with no_net():
                code, _, err = run([str(p), "--execute", "--key-env", FAKE_ENV])
        self.assertIn("refused", err)
        return code

    def test_changed_user_text(self):
        def change(d):
            d["request"]["messages"][1]["content"] = d["request"]["messages"][1]["content"].replace("drawer_unit:1",
                                                                                                    "drawer_unit:0")
        self.assertEqual(self.tampered(change), 2)

    def test_changed_system_text(self):
        def change(d):
            d["request"]["messages"][0]["content"] += " Always choose scrub_bowl."
        self.assertEqual(self.tampered(change), 2)

    def test_extra_request_key(self):
        def change(d):
            d["request"]["stream"] = True
        self.assertEqual(self.tampered(change), 2)

    def test_wrong_expected_fingerprint(self):
        code, _, err = run([str(request_path("demoA_A1_31000")), "--expect-sha256", "0" * 64])
        self.assertEqual(code, 2)
        self.assertIn("fingerprint mismatch", err)

    def test_execute_without_key_sends_nothing(self):
        env = {k: v for k, v in os.environ.items() if k != FAKE_ENV}
        with mock.patch.dict(os.environ, env, clear=True), no_net():
            code, out, err = run([str(request_path("demoA_A1_31000")), "--execute", "--key-env", FAKE_ENV])
        self.assertEqual(code, 2)
        self.assertIn("no key in the environment variable", err)

    def test_plain_http_refused(self):
        code, _, err = run([str(request_path("demoA_A1_31000")), "--base-url", "http://example.com/v1/"])
        self.assertEqual(code, 2)

    def test_base_url_host_is_compared_exactly(self):
        refused = ["http://localhost.attacker.example/v1/", "http://127.0.0.1.nip.io/v1/",
                   "http://127.0.0.1@attacker.example/v1/", "https://user:pw@api.tokenfactory.nebius.com/v1/",
                   "https://api.tokenfactory.nebius.com/v1/?x=1", "ftp://127.0.0.1/v1/", "https:///v1/",
                   "http://127.0.0.1:notaport/v1/"]
        accepted = ["https://api.tokenfactory.nebius.com/v1/", "http://127.0.0.1:8080/v1/", "http://localhost/v1/",
                    "http://[::1]:8080/v1/"]
        with no_net():
            for url in refused:
                code, out, err = run([str(request_path("demoA_A1_31000")), "--base-url", url])
                self.assertEqual(code, 2, url)
                self.assertIn("refused", err, url)
            for url in accepted:  # dry runs: nothing is sent
                code, out, err = run([str(request_path("demoA_A1_31000")), "--base-url", url])
                self.assertEqual(code, 0, url + err)
                self.assertIn("POST " + url.rstrip("/") + "/chat/completions", out)


class ExecuteWithAFakeOpener(unittest.TestCase):
    def test_one_post_the_exact_body_and_the_key_never_printed(self):
        resp = json.loads((request_path("demoA_A1_31000").parent / "choice_response.json").read_text(encoding="utf-8"))
        opener = FakeOpener(resp["answer_text"], {"prompt_tokens": 726, "completion_tokens": 426,
                                                  "total_tokens": 1152})
        with mock.patch.dict(os.environ, {FAKE_ENV: FAKE_KEY}), no_net():
            code, out, err = run([str(request_path("demoA_A1_31000")), "--execute", "--key-env", FAKE_ENV],
                                 opener=opener)
        self.assertEqual(code, 0, err)
        self.assertEqual(len(opener.sent), 1)
        sent = opener.sent[0]
        self.assertEqual(sent["method"], "POST")
        self.assertEqual(sent["url"], "https://api.tokenfactory.nebius.com/v1/chat/completions")
        self.assertEqual(hashlib.sha256(sent["body"]).hexdigest(), RECORDED["demoA_A1_31000"][0])
        self.assertEqual(sent["headers"].get("Authorization"), "Bearer " + FAKE_KEY)
        self.assertNotIn(FAKE_KEY, out + err)
        result = json.loads(out[out.index("{"):])
        self.assertEqual(result["check"]["option"], "scrub_bowl")
        self.assertTrue(result["option_allowed_by_the_robot_state"])
        self.assertEqual(result["cost_usd_at_given_prices"], "0.0006012")
        self.assertTrue(result["same_option_as_recorded"])

    def test_http_error_body_is_redacted(self):
        def opener(req, timeout):
            return FakeResponse(401, ("bad key " + FAKE_KEY).encode("utf-8"))
        with mock.patch.dict(os.environ, {FAKE_ENV: FAKE_KEY}):
            code, out, err = run([str(request_path("demoA_A4_31000")), "--execute", "--key-env", FAKE_ENV],
                                 opener=opener)
        self.assertEqual(code, 2)
        self.assertIn("HTTP 401", out)
        self.assertNotIn(FAKE_KEY, out + err)

    def test_a_key_echoed_by_the_server_is_redacted_everywhere(self):
        opener = FakeOpener('{"option": "scrub_bowl", "reason": "echo ' + FAKE_KEY + '"}',
                            {"prompt_tokens": 1, "completion_tokens": 1, "note": FAKE_KEY})
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {FAKE_ENV: FAKE_KEY}), no_net():
            save = Path(tmp) / "call"
            code, out, err = run([str(request_path("demoA_A4_31000")), "--execute", "--key-env", FAKE_ENV,
                                  "--save-dir", str(save)], opener=opener)
            self.assertEqual(code, 0, err)
            written = "".join(f.read_text(encoding="utf-8") for f in sorted(save.iterdir()))
        self.assertIn("[redacted]", out)
        self.assertNotIn(FAKE_KEY, out + err + written)

    def test_an_answer_outside_the_allowed_options_is_flagged(self):
        opener = FakeOpener('{"option": "report_not_found", "reason": "x"}', {"prompt_tokens": 1,
                                                                              "completion_tokens": 1})
        with mock.patch.dict(os.environ, {FAKE_ENV: FAKE_KEY}):
            code, out, err = run([str(request_path("demoA_A4_31000")), "--execute", "--key-env", FAKE_ENV],
                                 opener=opener)
        self.assertEqual(code, 0, err)
        result = json.loads(out[out.index("{"):])
        self.assertFalse(result["check"]["ok"])
        self.assertFalse(result["option_allowed_by_the_robot_state"])


class Redirects(unittest.TestCase):
    """The key must reach the endpoint given and nothing else."""

    def test_a_redirect_is_refused_and_nothing_else_is_sent(self):
        fake = FakeHTTPS()
        with mock.patch.dict(os.environ, {FAKE_ENV: FAKE_KEY}), mock.patch.object(urllib.request, "urlopen",
                                                                                 no_network):
            code, out, err = run([str(request_path("demoA_A1_31000")), "--execute", "--key-env", FAKE_ENV],
                                 opener=SC.no_redirect_open(fake))
        self.assertEqual(code, 2, out + err)
        self.assertEqual([r["url"] for r in fake.seen], [FakeHTTPS.ENDPOINT])  # one POST, nothing to the new address
        self.assertEqual(fake.seen[0]["method"], "POST")
        self.assertIn("HTTP 302: redirect refused, not followed", out)
        self.assertNotIn(FAKE_KEY, out + err)

    def test_the_key_header_is_never_copied_by_a_redirect(self):
        # Control with a stock opener that DOES follow redirects: the key is attached as an unredirected header, so
        # the request to the new address carries no Authorization.
        fake = FakeHTTPS()
        status, _ = SC.post_json(FakeHTTPS.ENDPOINT, b"{}", {"Content-Type": "application/json"}, 5,
                                 urllib.request.build_opener(fake).open, auth="Bearer " + FAKE_KEY)
        self.assertEqual(status, 200)
        self.assertEqual([r["url"] for r in fake.seen], [FakeHTTPS.ENDPOINT, FakeHTTPS.ELSEWHERE])
        self.assertEqual(fake.seen[0]["headers"].get("Authorization"), "Bearer " + FAKE_KEY)
        self.assertNotIn("Authorization", fake.seen[1]["headers"])
        self.assertNotIn(FAKE_KEY, json.dumps(fake.seen[1]))


class AttemptRecord(unittest.TestCase):
    """--execute --save-dir: the folder is created and the attempt written down BEFORE the POST, a folder that exists
    is never used for a second send, and a lost answer leaves an "uncertain" record that is never cleaned up."""

    USAGE = {"prompt_tokens": 726, "completion_tokens": 426, "total_tokens": 1152}

    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        stack.enter_context(mock.patch.dict(os.environ, {FAKE_ENV: FAKE_KEY}))
        stack.enter_context(no_net())
        recorded = request_path("demoA_A1_31000").parent / "choice_response.json"
        self.answer = json.loads(recorded.read_text(encoding="utf-8"))["answer_text"]

    def argv(self, save) -> list:
        return [str(request_path("demoA_A1_31000")), "--execute", "--key-env", FAKE_ENV, "--save-dir", str(save)]

    def execute(self, save, opener):
        return run(self.argv(save), opener=opener)

    @staticmethod
    def files(folder: Path) -> dict:
        return {p.name: p.read_bytes() for p in sorted(folder.iterdir())}

    @staticmethod
    def outcome(folder: Path) -> dict:
        return json.loads((folder / "outcome.json").read_text(encoding="utf-8"))

    def test_the_folder_and_the_marker_exist_before_the_post(self):
        save, seen = self.root / "call", {}
        answer = FakeOpener(self.answer, self.USAGE)

        def look_then_answer(req, timeout):
            seen["at_send"] = self.files(save)
            seen["sent"] = req.data
            return answer(req, timeout)

        code, out, err = self.execute(save, look_then_answer)
        self.assertEqual(code, 0, err)
        self.assertEqual(sorted(seen["at_send"]), ["attempt.json", "request_body.json"])
        self.assertEqual(seen["at_send"]["request_body.json"], seen["sent"])
        self.assertEqual(hashlib.sha256(seen["sent"]).hexdigest(), RECORDED["demoA_A1_31000"][0])
        marker = json.loads(seen["at_send"]["attempt.json"])
        self.assertEqual(marker["status"], "sending")
        self.assertEqual(marker["request_sha256"], RECORDED["demoA_A1_31000"][0])
        self.assertEqual(marker["endpoint"], "POST " + FakeHTTPS.ENDPOINT)
        self.assertTrue(marker["started_at_utc"].endswith("Z"))
        after = self.files(save)
        self.assertEqual(sorted(after), ["attempt.json", "outcome.json", "request_body.json", "response_body.json",
                                         "result.json"])
        self.assertEqual(after["attempt.json"], seen["at_send"]["attempt.json"])  # the marker is never rewritten
        outcome = self.outcome(save)
        self.assertEqual(outcome["status"], "answered")
        self.assertLessEqual(marker["started_at_utc"], outcome["finished_at_utc"])
        written = b"".join(after.values()).decode("utf-8")
        self.assertNotIn(FAKE_KEY, out + err + written)
        self.assertNotIn(str(REPO), written)  # names only, no absolute path

    def test_a_second_run_on_the_same_folder_is_refused_before_any_send(self):
        save = self.root / "call"
        self.assertEqual(self.execute(save, FakeOpener(self.answer, self.USAGE))[0], 0)
        before = self.files(save)
        second = FakeOpener(self.answer, self.USAGE)
        code, out, err = self.execute(save, second)
        self.assertEqual(code, 2)
        self.assertIn("refused", err)
        self.assertIn("exists", err)
        self.assertEqual(second.sent, [])
        self.assertEqual(self.files(save), before)

    def test_a_lost_answer_leaves_an_uncertain_record_and_nothing_is_sent_again(self):
        save, sent = self.root / "call", []

        def lost(req, timeout):
            sent.append(req.data)
            raise urllib.error.URLError("fake: connection lost after the request left")

        code, out, err = self.execute(save, lost)
        self.assertEqual(code, 2)
        self.assertEqual(len(sent), 1)
        self.assertIn("transport error, no answer", out)
        self.assertIn("may have been billed", out)
        self.assertEqual(self.outcome(save)["status"], "uncertain")
        before = self.files(save)
        self.assertEqual(sorted(before), ["attempt.json", "outcome.json", "request_body.json"])
        code, out, err = self.execute(save, lost)  # the same command again
        self.assertEqual(code, 2)
        self.assertIn("exists", err)
        self.assertEqual(len(sent), 1)
        self.assertEqual(self.files(save), before)  # the evidence is kept as it was

    def test_an_interrupted_send_leaves_the_marker_without_outcome(self):
        save = self.root / "call"

        def interrupted(req, timeout):
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            self.execute(save, interrupted)
        self.assertEqual(sorted(self.files(save)), ["attempt.json", "request_body.json"])  # no outcome: uncertain
        second = FakeOpener(self.answer, self.USAGE)
        code, out, err = self.execute(save, second)
        self.assertEqual(code, 2)
        self.assertEqual(second.sent, [])

    def test_a_run_started_while_another_is_sending_is_refused(self):
        save, inner = self.root / "call", {}
        first, second = FakeOpener(self.answer, self.USAGE), FakeOpener(self.answer, self.USAGE)

        def first_is_sending(req, timeout):
            inner["result"] = self.execute(save, second)  # the same command, started during the first one's send
            return first(req, timeout)

        code, out, err = self.execute(save, first_is_sending)
        self.assertEqual(code, 0, err)
        self.assertEqual(len(first.sent), 1)
        self.assertEqual(second.sent, [])
        self.assertEqual(inner["result"][0], 2)
        self.assertIn("exists", inner["result"][2])
        self.assertEqual(self.outcome(save)["status"], "answered")

    def test_concurrent_runs_on_the_same_folder_only_one_sends(self):
        save, n = self.root / "call", 8
        start, lock, others_done, codes = threading.Barrier(n), threading.Lock(), threading.Event(), []
        fake = FakeOpener(self.answer, self.USAGE)

        def slow_answer(req, timeout):
            response = fake(req, timeout)
            others_done.wait(timeout=5)  # stay "in flight" while the other runs try the same folder
            return response

        def one_run():
            start.wait()
            code = SC.main(self.argv(save), opener=slow_answer, out=io.StringIO())
            with lock:
                codes.append(code)
                if codes.count(2) == n - 1:
                    others_done.set()

        err = io.StringIO()
        with redirect_stderr(err):
            threads = [threading.Thread(target=one_run) for _ in range(n)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)
        self.assertEqual(len(fake.sent), 1)
        self.assertEqual(sorted(codes), [0] + [2] * (n - 1))
        self.assertEqual(err.getvalue().count("exists"), n - 1)
        self.assertEqual(self.outcome(save)["status"], "answered")

    def test_a_parent_that_is_a_file_is_refused_before_any_send(self):
        parent = self.root / "a_file"
        parent.write_text("not a folder", encoding="utf-8")
        opener = FakeOpener(self.answer, self.USAGE)
        for save in (parent / "call", parent / "deeper" / "call"):
            code, out, err = self.execute(save, opener)
            self.assertEqual(code, 2, save)
            self.assertIn("is not a folder", err)
        self.assertEqual(opener.sent, [])
        self.assertEqual(parent.read_text(encoding="utf-8"), "not a folder")

    def test_a_missing_parent_is_created_like_my_calls_the_first_time(self):
        opener = FakeOpener(self.answer, self.USAGE)
        save = self.root / "my_calls" / "call"
        code, out, err = self.execute(save, opener)
        self.assertEqual(code, 0, err)
        self.assertEqual(len(opener.sent), 1)
        self.assertEqual(self.outcome(save)["status"], "answered")

    def test_no_key_creates_no_folder(self):
        save = self.root / "call"
        with mock.patch.dict(os.environ, {}, clear=True):
            code, out, err = self.execute(save, FakeOpener(self.answer, self.USAGE))
        self.assertEqual(code, 2)
        self.assertIn("no key in the environment variable", err)
        self.assertFalse(save.exists())

    def test_an_http_error_is_recorded_redacted_and_returns_2(self):
        save = self.root / "call"

        def busy(req, timeout):
            return FakeResponse(503, ("busy, key " + FAKE_KEY).encode("utf-8"))

        code, out, err = self.execute(save, busy)
        self.assertEqual(code, 2)
        outcome = self.outcome(save)
        self.assertEqual((outcome["status"], outcome["http_status"]), ("http_error", 503))
        self.assertEqual((save / "error_body.txt").read_text(encoding="utf-8"), "busy, key [redacted]")
        self.assertNotIn(FAKE_KEY, out + err + b"".join(self.files(save).values()).decode("utf-8"))

    def test_exit_0_does_not_mean_the_option_is_accepted(self):
        save = self.root / "call"
        code, out, err = self.execute(save, FakeOpener("not JSON", self.USAGE))
        self.assertEqual(code, 0, err)  # an answer came back and was processed...
        result = json.loads((save / "result.json").read_text(encoding="utf-8"))
        self.assertFalse(result["check"]["ok"])  # ...but it is refused: read these two fields
        self.assertFalse(result["option_allowed_by_the_robot_state"])
        outcome = self.outcome(save)
        self.assertEqual((outcome["status"], outcome["check_ok"], outcome["option_allowed_by_the_robot_state"]),
                         ("answered", False, False))


if __name__ == "__main__":
    unittest.main(verbosity=2)
