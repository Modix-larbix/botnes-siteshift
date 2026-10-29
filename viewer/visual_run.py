"""Add spectator rendering to the frozen public launcher, without editing its files.

The observer receives the simulator only inside the evaluator process. Frames and
progress never become inputs to the robot. The original launcher still verifies
all source files and compares the new episode with the published recorded facts.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parent
SIMULATOR = BASE.parent / "simulator"
sys.path.insert(0, str(SIMULATOR / "demo"))


def atomic_json(path: Path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--situation", required=True, choices=("in_place", "moved", "masked"))
    parser.add_argument("--session", required=True, type=Path)
    args = parser.parse_args()
    session = args.session.resolve()
    frames = session / "frames"
    frames.mkdir(exist_ok=False)

    import demo_client as DC
    import public_adapter as PA
    import run_episode

    # Verify disk contents before importing the frozen robot. main() verifies
    # them again. The only replaced component is a spectator recorder.
    manifest = PA.load_manifest(SIMULATOR)
    problems = PA.verify_distribution(SIMULATOR, manifest) + PA.verify_panda_assets(SIMULATOR)
    if problems:
        raise RuntimeError("; ".join(problems))
    DC.use_root(SIMULATOR)
    BT = DC.root_module("siteagent.bench_task", SIMULATOR)
    BA = DC.root_module("siteagent.agent.bench_agent", SIMULATOR)

    import mujoco
    from PIL import Image

    class LiveRecorder:
        def __init__(self, sim, title, truth_text, every=20):
            self.sim = sim
            self.renderer = mujoco.Renderer(sim.model, height=450, width=800)
            self.small = mujoco.Renderer(sim.model, height=225, width=320)
            self.camera = mujoco.MjvCamera()
            self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.camera.lookat[:] = [0.05, 0.10, sim.site.worktop_height]
            self.camera.distance = 2.1
            self.camera.elevation = -62.0
            self.camera.azimuth = 200.0
            self.ticks, self.count = 0, 0
            self.skill, self.calls = "Simulating robot movement", []
            self.finished = False
            self.started = time.monotonic()
            self.capture()

        def capture(self):
            self.renderer.update_scene(self.sim.data, camera=self.camera)
            view = Image.new("RGB", (1120, 450), "#edf1f5")
            view.paste(Image.fromarray(self.renderer.render()), (0, 0))
            for index, camera in enumerate(("robot_eye", "panda/wrist_cam")):
                self.small.update_scene(self.sim.data, camera=camera)
                view.paste(Image.fromarray(self.small.render()), (800, index * 225))
            self.count += 1
            destination = frames / f"{self.count:05d}.jpg"
            temporary = destination.with_suffix(".tmp")
            view.save(temporary, format="JPEG", quality=84)
            temporary.replace(destination)
            atomic_json(session / "progress.json", {
                "frame": self.count, "sim_s": round(float(self.sim.data.time), 2),
                "skill": self.skill, "calls": self.calls[-60:],
                "render_wall_s": round(time.monotonic() - self.started, 2),
                "observer_finished": self.finished,
            })

        def __call__(self, sim):
            self.ticks += 1
            if self.ticks % 20 == 0:
                self.capture()

        def on_call(self, record):
            self.skill = record["skill"]
            self.calls.append({"i": record["i"], "sim_s": record["t_s"],
                               "skill": record["skill"], "ok": record["ok"],
                               "reason": record.get("reason")})

        def finish(self, verdict_text, claim):
            self.skill = "Episode finished"
            self.finished = True
            self.capture()

        def save(self, path, **kwargs):
            # Frames are already on disk. No GIF is made here; playback uses
            # those actual rendered frames, with no change to physics time.
            self.renderer.close()
            self.small.close()
            return frames

    BT.TaskRecorder = LiveRecorder
    original_episode = BA.run_agent_episode

    def episode_with_observer(*positional, **keywords):
        keywords["gif_path"] = str(session / "spectator_frames")
        return original_episode(*positional, **keywords)

    BA.run_agent_episode = episode_with_observer
    return run_episode.main(["--situation", args.situation, "--out", str(session / "episode")])


if __name__ == "__main__":
    raise SystemExit(main())
