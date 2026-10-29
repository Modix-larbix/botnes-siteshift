# How SiteShift works

One episode, from what the robot senses to how its report is marked, with the file and the function that does each
step. Paths under `simulator/` are the robot code that recorded the three episodes (frozen on October 20, 2026; see
[PROVENANCE.md](PROVENANCE.md)). Paths under `site/` are the replay page. Paths without a folder are relative to
`simulator/siteagent/`.

```
simulated camera: depth and object labels, with noise              perception/sensor.py  SimulatedCamera
  -> the robot's own perception and zone search                    perception/estimate.py, search.py, sitemodel.py
  -> its state text: each zone seen, cleared or not cleared        agent/host.py  render_scene_state
  -> code checks eight conditions and lists the valid options      simulator/demo/demo_client.py  menu_conditions
       only ask_help valid       -> a rule takes it, no model call
       scrub_bowl and ask_help   -> Nemotron chooses one, from text   simulator/demo/choice_prompt.py
                                    code checks the answer again      choice_prompt.parse_answer, DemoClient._request_2
  -> a team-written task program: pick and scrub, or ask for help  simulator/demo/programs/*.bs
       gestures, and the robot's guards that accept or refuse them robot/skills.py  Robot; agent/host.py  GuardedHost
  -> the evaluator grades the episode from the simulator's truth   bench_task.py  run_task_episode; taskcheck.py
  -> the page writes the robot's report from its records           site/app.js  reportClaims
  -> after Reveal, each claim is marked against the evaluator      site/app.js  markClaim
```

## Who decides what

| Step | Decided by | Where |
| --- | --- | --- |
| Inspection route, task gestures | Fixed programs written by the team | `simulator/demo/programs/inspect_v2.bs`, `scrub_v2.bs`, `help_v1.bs` |
| The three zone labels | The robot's own readings | `perception/`, `search.py`, `sitemodel.py`, `agent/host.py` |
| Which next steps are valid | Code, eight conditions on the robot's text | `simulator/demo/demo_client.py` `menu_conditions` |
| `scrub_bowl` or `ask_help`, when both are valid | Nemotron 3 Super, from a text summary; code checks its answer again | `simulator/demo/choice_prompt.py`, `DemoClient._request_2` |
| Where the arm picks | The task program, following the sponge the robot perceived | `scrub_v2.bs`, `robot/skills.py` `Robot.pick` |
| Accepting or refusing a gesture or a claim | The robot's guards | `agent/host.py` `GuardedHost`, `agent/loop.py` |
| Success or failure, where the sponge started | The evaluator (simulator truth, never given to the robot) | `bench_task.py`, `taskcheck.py` |
| The report's claims | The page's code, from the robot's records; no model | `site/app.js` `reportClaims` |
| Each claim's mark | A fixed rule, against the evaluator's records | `site/app.js` `markClaim` |

Nemotron does not see images, does not plan, does not move the arm and does not check anything. It picks one of two
programs the team wrote, once code has found both valid.

## Step by step

**1. World and sensors.** `world/build.py` `build` writes the MuJoCo kitchen (development kitchen 31000) around the
Franka Emika Panda model from MuJoCo Menagerie, which `simulator/assets/fetch_panda.sh` downloads at a pinned commit.
`world/variants.py` `make_variant` moves the sponge to another drawer for the "moved" situation (in this kitchen it
swaps places with the spray bottle), and
`world_dev/occlusion.py` `applied` lays the catalog towel over the sponge for the "masked" situation. The robot
receives only the frames of `perception/sensor.py` `SimulatedCamera`: depth with noise and per-pixel object labels.
This is simulated perception, not a vision model reading color images.

**2. Inspection.** The fixed inspection program `simulator/demo/programs/inspect_v2.bs` runs inside the agent loop
`agent/loop.py` `AgentLoop`. Every program is first checked by `agent/lang.py` `compile_program` (a small restricted
language), then run in a separate process under the macOS sandbox by `agent/sandbox.py` `run_isolated`, with the
interpreter of `agent/interp.py`. Each gesture call goes through `agent/host.py` `GuardedHost`, which can refuse it,
and then through `robot/skills.py` `Robot` (`calibrate`, `open_drawer`, `look_pose`, ...).

**3. The map with three labels.** `perception/estimate.py` `Perception` keeps the tracks of what the robot sees.
`search.py` `SiteSearch` and `sitemodel.py` `SiteModel` decide, zone by zone, whether the robot's views cover a zone
well enough to report the sponge as not found in the part inspected; `sitemodel.py` `could_hide` answers whether the
unseen part of a zone could still hide it. `agent/host.py` `render_scene_state` writes the robot's state text: a zone where it observed a
sponge (*sponge seen*), a "cleared" zone, glossed in the text as "not found in the part inspected" (*sponge not found
in the checked zone*), and each "not cleared" zone with its reason (*still to check*).

**4. Code checks the options.** After the inspection, the loop asks its client for the next program.
`simulator/demo/demo_client.py` `DemoClient._request_2` reads the robot's state text (`parse_scene_state`), and
`menu_conditions` checks eight conditions on it: the text is readable, the hand holds nothing, 0 contact incidents and
0 grip events in the robot's own count, search outcome "found", a sponge track not held, no "held" line, and a help
review triggered by a sponge the robot perceived but has not picked. All eight true: both `scrub_bowl` and `ask_help`
are valid. Otherwise only `ask_help` is valid, and a rule takes it with no model call (situation 3).

**5. Nemotron's choice, checked again by code.** When both options are valid, `simulator/demo/choice_prompt.py`
`build_choice_messages` builds two text-only messages: a fixed system text (the rules, the menu of two, the JSON answer
format) and the task with the robot's state text, verbatim. The request (temperature 0) is fingerprinted by the sha256
of its canonical JSON. In the recorded episodes it was sent once to `nvidia/nemotron-3-super-120b-a12b` on Nebius Token
Factory. A local run never calls a model: `DemoClient._request_2` serves the recorded answer only when the request has
the recorded fingerprint and the loop's own request is the same too; otherwise the run stops with a refusal
(`choice_fingerprint_mismatch`, `loop_fingerprint_mismatch`). The answer is checked by `choice_prompt.parse_answer`
(one JSON object with the expected keys, an option from the menu, finish reason "stop"), then the code checks again
that the option is valid in this state (`condition_not_met` otherwise). A refused answer makes the robot ask for help.
`nemotron_client/send_choice.py` is the standalone public client that rebuilds the same request (dry run by default).

**6. Task program and gestures.** The chosen option names a team-written program (`demo_client.OPTION_PROGRAM`):
`scrub_v2.bs` picks the sponge the robot itself tracked and scrubs the bowl on the landing tray; `help_v1.bs` stops and
asks a person, listing the zones still to check. The gestures are `robot/skills.py` `Robot.pick` and `Robot.scrub`.
The robot's guards accept or refuse: `agent/host.py` `GuardedHost.pick_precondition` before a pick, and in
`agent/loop.py`, `refuse_done_without_evidence` and `refuse_done_after_incident` before the robot may report the task
done, and `not_found_guard` before it may conclude "sponge not found".

**7. The evaluator.** `bench_task.py` `run_task_episode` runs the episode and keeps the simulator's truth on the bench
side, never given to the robot. `taskcheck.py` `TaskMonitor` follows the sponge, the gripper and the bowl contact
during the episode, and `taskcheck.py` `verdict` grades the end with the thresholds of
`config/task_criteria_v1_4.json`: 8 task criteria (bowl contact, scrub motion, the bowl kept still under the sponge,
the sponge kept in the gripper during the scrub, no drop, no other dish scrubbed, force on the bowl within limit, bowl
upright and in place). The episode folder gets
`oracle_verdict.json` (bench side), and `observable_feedback.json` and `trace.json` (robot side, from `feedback.py`
`observable_feedback`). `agent/bench_agent.py` `run_agent_episode` adds the agent's records: `agent_record.json` and
`requests.json` (every request text the robot built).

**8. The report, written by code.** No model writes the report. `site/app.js` `reportClaims(r)` takes only
`episode.robot`, the robot's exported records, and writes one claim per assertion in four groups: what it did, what it
checked and found, what it did not check, what it refused or held back.

**9. The marks, after Reveal.** `site/app.js` `markClaim(c, v)` takes one claim and `episode.evaluator` only. A claim is
**true** when the evaluator's records hold a positive record of that exact assertion, **false** when they hold a
record that contradicts it, and **not checkable** otherwise; never true by default. `reportBlock(e, revealed)` draws the
marks only once the reader presses Reveal. Reveal is a display rule, not access control: the page's data files hold
the evaluator's records from the start.

## From the simulator to the page

The page reads `site/data/episodes.js`, which was exported from the three recorded episode folders by an exporter that
is not part of this repository. A local run of the simulator (`simulator/demo/run_episode.py`) writes the same kinds of
records into a new folder and compares them with the recorded facts (`simulator/recorded_choices/RECORDED_EPISODE_FACTS.json`);
it does not rebuild the page data, and the page keeps showing the recorded episodes.

## Where to start reading

1. `simulator/demo/run_episode.py` `main`: the public entry point of one situation.
2. `agent/bench_agent.py` `run_agent_episode`, then `bench_task.py` `run_task_episode`: one episode, robot side and
   bench side.
3. `simulator/demo/demo_client.py` `DemoClient`: the programs served to the loop, the menu and the recorded choice.
4. `agent/host.py` `render_scene_state`: the state text the menu and Nemotron read.
5. `taskcheck.py` `verdict` with `config/task_criteria_v1_4.json`: the evaluator.
6. `site/app.js`: the reading map at the top of the file, then `reportClaims` and `markClaim`.


## Visual launcher added October 29, 2026

The new entry point is `start.command` at the repository root, after `sh setup.sh`. It starts `viewer/server.py` on the loopback interface. Clicking Play simulation starts `viewer/visual_run.py` in a fresh process, which uses the original `simulator/demo/run_episode.py` launcher. The frozen files and their manifest are unchanged.

The visual wrapper supplies `gif_path` to the original `run_agent_episode` and replaces its optional `TaskRecorder` in memory with an observer that writes JPEG frames and progress. It renders an overview and the robot's two cameras. No spectator information becomes an input to the robot. Rendering adds work, but does not change the physics steps or task program. Each run still compares its recorded facts with the reference; matching facts, rather than a claim of identical execution under any observer, are the check used here.

The server manages one process group at a time, supports cancellation and a ten-minute timeout, and keeps outputs in a new `viewer/runs/<id>/` folder. Run analysis reads the new episode's reproduction and evaluator records. It does not produce the earlier claim-by-claim report. The old `site/` website is served separately at `/evidence/` and continues to show only recorded episodes.

The visual overview exposes simulator truth to the spectator from the start. The old evidence website's Reveal convention does not apply to this overview. The robot's information boundary remains the original program isolation. Browser JSON actions require the local origin and a custom header, and the server binds only to 127.0.0.1. No model credentials are inherited by the simulation child. See [the viewer guide](../viewer/README.md).
