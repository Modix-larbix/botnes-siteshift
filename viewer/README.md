# Visual simulation launcher

From the repository root, run `sh setup.sh`, then `./start.command`. Open http://127.0.0.1:8766/ in a browser on the same Mac. The server listens on this machine only. It cannot run from a static website or a remote phone browser.

## Using the demo

Choose one of three situations and click **Play simulation**. That click starts a new process running the published MuJoCo episode. The overview and the two robot cameras update from this run while it computes. The overview is a spectator view, never an input to the robot.

The first pass can look uneven because calculation and rendering run together. After completion, **Replay movements** plays its saved images in order, at 10 frames per second. Images are sampled about every 0.4 simulated seconds, so playback is about four times faster than simulated time. The simulated clock and elapsed calculation time are different quantities.

**Run analysis** reads this run's `REPRODUCTION.json` and `oracle_verdict.json`. It shows the task criteria, final claim, source checks and comparison with the recorded episode. The hidden-sponge example ends by asking for help, so some bowl-scrubbing criteria remain unmet because scrubbing was not attempted. The analysis is a readiness report from uncalibrated simulation, not a real-world qualification.

The analysis does not rebuild the earlier website's claim-by-claim report. **Recorded evidence** opens that separate, unchanged website under `/evidence/`.

**Stop run** terminates the local process group, including the robot program. Images already written remain available for playback. A run is stopped after ten minutes if it has not finished. Only one run can execute at a time.

## Files and source checks

`server.py` serves the interface and manages local runs. `visual_run.py` verifies the simulator manifest, replaces only its optional spectator recorder in memory, and supplies the recorder argument to the original episode function. It does not edit the frozen simulator files. The observer does not feed information back to the robot. The original launcher still checks every compared episode fact and refuses a recorded model choice when its request differs.

The child process receives a fixed environment without model credentials. No model or network service is called. Installation uses the network to obtain pinned dependencies and the Panda assets.

Each session is kept under `viewer/runs/<id>/`, including rendered frames, the episode files and a local process log. These outputs are ignored by git. Completed runs remain available after a server restart in the same browser tab. They are development reproductions, not new blind-validation examples.

## Troubleshooting

If setup cannot find `uv`, install it from its [official instructions](https://docs.astral.sh/uv/getting-started/installation/). The simulator needs macOS and Python 3.12; setup creates `simulator/.venv` with the pinned packages. Other operating systems can view the recorded website described in the repository README.

If port 8766 is already in use, check whether another SiteShift launcher is still running. You can also run `./start.command --port 8767` and use the address it prints. Keep that terminal window open. Ctrl+C and normal terminal shutdown stop the active simulation before the server exits. If a run is refused or differs from the reference, open its local run log and analysis. Do not interpret a difference as a matched reproduction.

For the viewer's offline boundary tests, run `simulator/.venv/bin/python -B viewer/test_viewer.py` from the repository root. These checks do not call a model or run the robot. See [Testing](../docs/TESTING.md) for the other checks and their scope.

Optional frontend regression checks use Node.js 18 or later: `node --test viewer/test_ui.js`. They do not run a browser, simulator or model.
