"""Offline tests of the public launcher's checks (demo/public_adapter.py, demo/run_episode.py).

  env -u NEBIUS_API_KEY -u OPENAI_API_KEY PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m unittest discover -s tests -v

(or any Python 3.12). Standard library only: no network, no simulation, no MuJoCo. The robot's request fingerprint
function is imported from siteagent/agent/llm.py, which uses the standard library only. Each test works on a temporary
copy of the package.
"""
from __future__ import annotations

import contextlib
import io
import json
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE / "demo"))
sys.dont_write_bytecode = True
import public_adapter as PA  # noqa: E402
import demo_client as DC  # noqa: E402
import run_episode as RE  # noqa: E402

DC.use_root(PACKAGE)
LLM = DC.root_module("siteagent.agent.llm", PACKAGE)
COPIED = ("siteagent", "demo", "recorded_choices", "assets", "tf_guardrails.py", "requirements.txt", PA.MANIFEST_NAME)
MOVED = "recorded_choices/moved_A1"


def copy_package(dest: Path) -> Path:
    """The package's code and data at another absolute path (no Panda model, no virtual environment, no runs)."""
    dest.mkdir()
    for name in COPIED:
        src = PACKAGE / name
        if src.is_dir():
            shutil.copytree(src, dest / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copyfile(src, dest / name)
    return dest




class CopyCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = copy_package(Path(self.tmp.name) / "relocated_package")
        self.manifest = PA.load_manifest(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def load(self, record=MOVED):
        return PA.load_public_record(self.root / record, self.root, self.manifest, LLM)

    def accept_edit(self, name: str, record=MOVED):
        """Make PROVENANCE.json and the manifest agree with an edited record file, as a consistent edit would, so
        that the record's own checks are the ones tested."""
        d = self.root / record
        prov = json.loads((d / "PROVENANCE.json").read_text())
        prov["files"][name]["public_sha256"] = PA.sha256_file(d / name)
        (d / "PROVENANCE.json").write_text(json.dumps(prov, indent=1, sort_keys=True) + "\n")
        for rel in (f"{record}/{name}", f"{record}/PROVENANCE.json"):
            entry = next(e for e in self.manifest["code_and_data"] if e["path"] == rel)
            entry["sha256"] = PA.sha256_file(self.root / rel)


class Relocation(CopyCase):
    def test_relocated_copy_with_same_content_is_accepted(self):
        self.assertNotEqual(self.root.resolve(), PACKAGE.resolve())
        self.assertEqual(PA.verify_distribution(self.root, self.manifest), [])
        for record, sha in (("recorded_choices/moved_A1",
                             "47da009a38512d10e505f3e1e18bb765f1d1360223f3d31ad66491d76683efb7"),
                            ("recorded_choices/in_place_A4",
                             "37b01a38b0bb034786b1997d5b988ef3bfa60dce3e41833e891858e38950ab93")):
            rec = self.load(record)
            self.assertEqual(rec["request_sha256"], sha)
            self.assertFalse(rec["fake"])

    def test_one_changed_code_byte_is_refused(self):
        p = self.root / "siteagent" / "agent" / "limits.py"
        text = p.read_text()
        self.assertIn("max_steps: int = 20000", text)
        p.write_text(text.replace("max_steps: int = 20000", "max_steps: int = 20001", 1))
        problems = PA.verify_distribution(self.root, self.manifest)
        self.assertIn("siteagent/agent/limits.py differs from the manifest", problems)
        with self.assertRaisesRegex(ValueError, "does not match its manifest"):
            self.load()

    def test_unlisted_file_is_refused(self):
        (self.root / "siteagent" / "extra.py").write_text("x = 1\n")
        self.assertIn("file not in the manifest: siteagent/extra.py", PA.verify_distribution(self.root, self.manifest))

    def test_comment_edit_needs_a_declaration_and_unchanged_code(self):
        rel = "siteagent/agent/limits.py"
        p = self.root / rel
        original = p.read_text()
        p.write_text(original.replace('"""', '"""Edited docstring line.\n', 1) + "# a new comment\n")
        self.assertIn(f"{rel} differs from the manifest", PA.verify_distribution(self.root, self.manifest))
        entry = next(e for e in self.manifest["code_and_data"] if e["path"] == rel)
        entry["sha256"], entry["change"] = PA.sha256_file(p), "comments_or_docstrings_only"
        self.assertEqual(PA.verify_distribution(self.root, self.manifest), [])
        p.write_text(p.read_text().replace("max_steps: int = 20000", "max_steps: int = 20001", 1))
        entry["sha256"] = PA.sha256_file(p)
        self.assertIn(f"{rel}: its code differs from the snapshot (not only comments or docstrings)",
                      PA.verify_distribution(self.root, self.manifest))

    def test_recorded_choice_survives_a_declared_comment_edit_of_the_spending_guard(self):
        p = self.root / "tf_guardrails.py"
        p.write_text(p.read_text() + "# a new comment\n")
        entry = next(e for e in self.manifest["code_and_data"] if e["path"] == "tf_guardrails.py")
        entry["sha256"], entry["change"] = PA.sha256_file(p), "comments_or_docstrings_only"
        self.assertEqual(self.load()["request_sha256"],
                         "47da009a38512d10e505f3e1e18bb765f1d1360223f3d31ad66491d76683efb7")
        p.write_text(p.read_text().replace('NEVER_REACH_USD = Decimal("200")', 'NEVER_REACH_USD = Decimal("201")', 1))
        entry["sha256"] = PA.sha256_file(p)
        with self.assertRaisesRegex(ValueError, "tf_guardrails.py: its code differs from the snapshot"):
            self.load()


class Records(CopyCase):
    def test_missing_receipt_is_refused(self):
        (self.root / MOVED / "choice_call.json").unlink()
        with self.assertRaisesRegex(ValueError, "missing \\['choice_call.json'\\]"):
            self.load()

    def test_missing_client_log_is_refused(self):
        (self.root / MOVED / "client_ops.jsonl").unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            self.load()

    def test_altered_response_is_refused(self):
        p = self.root / MOVED / "choice_response.json"
        doc = json.loads(p.read_text())
        doc["answer_text"] = doc["answer_text"].replace("scrub_bowl", "ask_help", 1)
        p.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        with self.assertRaisesRegex(ValueError, "differs from its PROVENANCE.json entry"):
            self.load()
        self.accept_edit("choice_response.json")
        with self.assertRaisesRegex(ValueError, "answer text does not match its sha256"):
            self.load()

    def test_receipt_that_disagrees_with_the_response_is_refused(self):
        p = self.root / MOVED / "choice_call.json"
        doc = json.loads(p.read_text())
        doc["model_id"] = "another/model"
        p.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        self.accept_edit("choice_call.json")
        with self.assertRaisesRegex(ValueError, "differs from the call receipt"):
            self.load()

    def test_different_request_text_in_the_record_is_refused(self):
        p = self.root / MOVED / "choice_request.json"
        doc = json.loads(p.read_text())
        doc["request"]["messages"][1]["content"] = doc["request"]["messages"][1]["content"].replace(
            "coverage 0.71", "coverage 0.72", 1)
        p.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        self.accept_edit("choice_request.json")
        with self.assertRaisesRegex(ValueError, "request fingerprints differ"):
            self.load()

    def test_record_of_another_code_tree_is_refused(self):
        p = self.root / MOVED / "choice_request.json"
        doc = json.loads(p.read_text())
        doc["code_root_siteagent_tree_sha256"] = "0" * 64
        p.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        self.accept_edit("choice_request.json")
        with self.assertRaisesRegex(ValueError, "another code tree"):
            self.load()

    def test_published_records_hold_no_private_path(self):
        for rec in ("moved_A1", "in_place_A4"):
            prov = json.loads((PACKAGE / "recorded_choices" / rec / "PROVENANCE.json").read_text())
            for name in PA.RECORD_FILES:
                text = (PACKAGE / "recorded_choices" / rec / name).read_text()
                self.assertIsNone(re.search(r'"/[A-Za-z]', text), f"{rec}/{name} holds an absolute path")
                self.assertEqual(prov["files"][name]["public_sha256"], PA.sha256_file(PACKAGE / "recorded_choices"
                                                                                      / rec / name))
                self.assertEqual(len(prov["files"][name]["original_sha256"]), 64)


class LiveRequest(unittest.TestCase):
    """The demo client (its logic unchanged from the recorded episodes) serves a recorded answer only on the same
    texts."""

    def setUp(self):
        manifest = PA.load_manifest(PACKAGE)
        self.rec = PA.load_public_record(PACKAGE / MOVED, PACKAGE, manifest, LLM)
        req = json.loads((PACKAGE / MOVED / "choice_request.json").read_text())
        user = req["request"]["messages"][1]["content"]
        head = "TASK (public statement): "
        self.task = user[len(head):user.index("\n\nSCENE STATE NOW (")]
        self.scene = user[user.index("SCENE STATE NOW ("):user.index("\n\nMENU (choose")]
        self.programs = DC.load_programs(PACKAGE / "demo" / "programs")

    def loop_request(self, scene: str) -> dict:
        report = {"status": DC.HELP_REVIEW_CODE,
                  "error": {"kind": "harness_rule", "code": DC.HELP_REVIEW_CODE,
                            "message": DC.NOTE_HEAD + "test: " + DC.NOTE_PERCEIVED + ")"}}
        user = (DC.TASK_HEAD + json.dumps({"text": self.task}) + DC.REPORT_HEAD + json.dumps(report) + "\n\n" + scene
                + DC.REMINDER_HEAD + self.task)
        return {"model": LLM.MODEL_ID, "messages": [{"role": "system", "content": "test"},
                                                    {"role": "user", "content": user}],
                "temperature": 0.0, "max_completion_tokens": 16384}

    def client(self, rec):
        c = DC.DemoClient("replay", self.programs, root=PACKAGE, recorded=rec)
        c.complete(self.loop_request(self.scene))  # request 1: the inspection program
        return c

    def test_same_texts_serve_the_recorded_answer(self):
        rec = dict(self.rec, loop_request_sha256=LLM.request_fingerprint(self.loop_request(self.scene)))
        c = self.client(rec)
        out = c.complete(self.loop_request(self.scene))
        self.assertIsNone(c.refusal)
        self.assertEqual(c.choice["choice_request_sha256"], self.rec["request_sha256"])
        self.assertEqual((c.choice["who_chose"], c.choice["option"]), ("nemotron", "scrub_bowl"))
        self.assertIn("```botscript", out.text)

    def test_different_request_text_is_refused(self):
        changed = self.scene.replace("coverage 0.71", "coverage 0.72", 1)
        self.assertNotEqual(changed, self.scene)
        rec = dict(self.rec, loop_request_sha256=LLM.request_fingerprint(self.loop_request(changed)))
        c = self.client(rec)
        with self.assertRaises(LLM.ClientNotReady):
            c.complete(self.loop_request(changed))
        self.assertEqual(c.refusal["code"], "choice_fingerprint_mismatch")

    def test_different_loop_request_is_refused(self):
        c = self.client(dict(self.rec))  # the recorded loop request fingerprint, not this test's request
        with self.assertRaises(LLM.ClientNotReady):
            c.complete(self.loop_request(self.scene))
        self.assertEqual(c.refusal["code"], "loop_fingerprint_mismatch")


class Launcher(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_main(self, argv, **kw):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = RE.main(argv, **kw)
        return code, err.getvalue()

    def test_existing_output_folder_is_refused(self):
        out = self.base / "moved_run"
        out.mkdir()
        (out / "keep.txt").write_text("earlier run")
        code, err = self.run_main(["--situation", "moved", "--out", str(out)])
        self.assertEqual(code, 2)
        self.assertIn("already exists", err)
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["keep.txt"])

    def test_reserved_output_names_are_refused(self):
        for name in ("real_moved", "pilot1", "validation_moved", "probe_format_1"):
            self.assertIsNotNone(PA.output_problem(self.base / name), name)
        self.assertIsNone(PA.output_problem(self.base / "moved_local_1"))

    def test_other_platforms_are_refused_before_anything_runs(self):
        out = self.base / "moved_on_linux"
        code, err = self.run_main(["--situation", "moved", "--out", str(out)], platform="linux")
        self.assertEqual(code, 2)
        self.assertIn("needs macOS", err)
        self.assertFalse(out.exists())
        self.assertIn("needs macOS", PA.platform_problem("win32"))
        self.assertIn("missing", PA.platform_problem("darwin", sandbox_exec=str(self.base / "no_sandbox_exec")))

    def test_other_python_versions_are_refused_before_anything_runs(self):
        # the code check compares Python 3.12 syntax trees: another version is named as such, not as changed code
        out = self.base / "moved_on_3_9"
        code, err = self.run_main(["--situation", "moved", "--out", str(out)], python_version=(3, 9))
        self.assertEqual(code, 2)
        self.assertIn("needs Python 3.12 (found 3.9)", err)
        self.assertNotIn("differs", err)
        self.assertFalse(out.exists())
        self.assertIn("found 3.13", PA.python_problem((3, 13, 0)))
        self.assertIsNone(PA.python_problem((3, 12)))

    def test_the_launcher_writes_no_compiled_files(self):
        # importing the launcher turns bytecode writing off, so a run started without -B leaves no stale .pyc behind
        src = (PACKAGE / "demo" / "run_episode.py").read_text(encoding="utf-8")
        self.assertLess(src.index("sys.dont_write_bytecode = True"), src.index("import public_adapter"))


if __name__ == "__main__":
    unittest.main()
