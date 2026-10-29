"""Checks of the local launch boundary. No simulator or model is called."""
import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path

import server


class LocalBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.runner = server.Runner(Path(cls.temp.name))
        cls.http = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        cls.http.runner = cls.runner
        cls.port = cls.http.server_port
        cls.thread = threading.Thread(target=cls.http.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.http.shutdown()
        cls.http.server_close()
        cls.thread.join()
        cls.temp.cleanup()

    def request(self, method, path, data=None, **override):
        headers = {"Origin": f"http://127.0.0.1:{self.port}", "X-SiteShift": "local-run",
                   "Content-Type": "application/json"}
        headers.update(override)
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        connection.request(method, path, None if data is None else json.dumps(data), headers)
        response = connection.getresponse()
        code, content = response.status, response.read()
        connection.close()
        return code, content

    def test_external_origin_cannot_start_a_run(self):
        code, _ = self.request("POST", "/api/runs", {"situation": "moved"}, Origin="https://other.example")
        self.assertEqual(code, 403)
        self.assertFalse(self.runner.jobs)

    def test_host_header_cannot_rebind_the_service(self):
        code, _ = self.request("GET", "/api/config", Host="other.example")
        self.assertEqual(code, 403)

    def test_invalid_situation_cannot_reach_a_shell(self):
        code, _ = self.request("POST", "/api/runs", {"situation": "moved; echo invalid"})
        self.assertEqual(code, 400)
        self.assertFalse(self.runner.jobs)

    def test_user_cannot_set_an_output_path(self):
        code, _ = self.request("POST", "/api/runs", {"situation": "moved", "out": "/tmp/override"})
        self.assertEqual(code, 400)

    def test_private_files_are_not_served(self):
        for path in ["/server.py", "/.env", "/simulator/demo/run_episode.py", "/evidence/../server.py"]:
            self.assertEqual(self.request("GET", path)[0], 404, path)

    def test_missing_or_incomplete_record_is_not_an_analysis(self):
        with self.assertRaises(ValueError):
            server.analyze_run(Path(self.temp.name) / "missing")

    def test_child_environment_does_not_inherit_credentials(self):
        environment = server.child_environment()
        self.assertEqual(set(environment), {"HOME", "PATH", "LANG", "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED"})

    def test_completed_session_survives_a_service_restart(self):
        identifier = "b" * 32
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / identifier
            path.mkdir()
            (path / "SESSION.json").write_text(json.dumps({"situation": "masked", "started": 1000}))
            (path / "STATUS.json").write_text(json.dumps({"status": "completed", "elapsed_s": 23.5,
                                                        "error": None, "exit_code": 0}))
            restored = server.Runner(Path(folder)).snapshot(identifier)
            self.assertEqual(restored["status"], "completed")
            self.assertEqual(restored["elapsed_s"], 23.5)
            self.assertTrue(restored["analysis_available"])

    def test_interrupted_session_is_never_restored_as_running(self):
        identifier = "a" * 32
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / identifier
            path.mkdir()
            (path / "SESSION.json").write_text(json.dumps({"situation": "moved", "started": 1000}))
            restored = server.Runner(Path(folder)).snapshot(identifier)
            self.assertEqual(restored["status"], "failed")
            self.assertFalse(restored["analysis_available"])


if __name__ == "__main__":
    unittest.main()
