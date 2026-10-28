# Botnes SiteShift

**Change the kitchen, keep the mission.**

A robot that works on a site acts on a site file, and places change between visits. A robot that acts on an
out-of-date file, or reports a result it cannot back up, makes a person check behind it. SiteShift shows, **in
simulation only** (MuJoCo physics, simulated perception, no physical robot), a robot arm that inspects a kitchen that
may no longer match its site file, keeps what it could not settle as "still to check", then does the task or asks a
person. Its report is then checked claim by claim against the simulator's records, which the robot never sees.

Submission to the Nebius x NVIDIA Global AI Hackathon, Physical AI track. Times in this README are in UTC.

[Results](#results) · [Run it](#run-it) · [How it works](#how-it-works) · [Limits](#limits) · [Contents](#repository-contents) · [License](#license-and-third-parties)

## Results

One development kitchen (31000), one simulated Franka Panda arm, one mission: find the sponge, pick it up, scrub the
bowl. The site file says drawer 0; we change the kitchen without telling the robot. Robot code frozen October 20, 2026.

| # | Situation | What the robot did | Who chose the next step | Evaluator, after Reveal | Report: claims, true, false, not checkable |
| --- | --- | --- | --- | --- | --- |
| 1 | Sponge in its place (the control; `in_place` in the simulator) | Saw it in drawer 0, picked it, scrubbed the bowl | Nemotron: `scrub_bowl`, from a menu of two | Success, 8 of 8 task criteria | 13, 8, 0, 5 |
| 2 | Sponge moved: it swapped drawers with the spray bottle (`moved`) | Saw it in drawer 1, picked it there, scrubbed the bowl | Nemotron: `scrub_bowl`, from a menu of two | Success, 8 of 8 task criteria | 13, 8, 0, 5 |
| 3 | Sponge masked by a towel in drawer 0 (`masked`) | Picked nothing, left drawer 0 *still to check*, asked a person | A rule: only `ask_help` was valid, no model call | Help with the sponge present (expected); no false "not found" | 13, 6, 0, 7 |
| | **The three reports together** | | | | **39, 22, 0, 17** |

**No false claim detected among the checkable claims of these three simulated episodes; 17 of 39 claims not
checkable.** The counts are not a success rate. Each claim and its mark: [docs/CLAIMS.md](docs/CLAIMS.md).

## Run it

**(a) The replay page (the recorded episodes): no install, no key, no network.** Python 3.8 or later, from the repository root:

```sh
python3 -m http.server 8765 --bind 127.0.0.1 --directory site
```

Open <http://127.0.0.1:8765/> and click **Start with situation 1**. Each situation shows the robot's side first (its
readings, its map, its choice, its action, its report); **Reveal and check its claims** adds the evaluator's records
and each claim's mark. Nothing is simulated and no model is called while you watch.

**(b) The simulator (runs them again): macOS only (tested on Apple silicon), Python 3.12, [uv](https://docs.astral.sh/uv/), git.** The
install fetches MuJoCo and NumPy from PyPI and the Panda model from MuJoCo Menagerie at a pinned commit (uv also
downloads Python 3.12 if it is missing); then all is offline:

```sh
cd simulator
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
sh assets/fetch_panda.sh
env -u NEBIUS_API_KEY -u OPENAI_API_KEY PYTHONDONTWRITEBYTECODE=1 \
    .venv/bin/python -B demo/run_episode.py --situation moved --out runs/moved_1
cd ..                                             # back to the repository root
```

Use `in_place`, `moved` or `masked`, each time with a new `--out` folder. A run takes about 20 to 30 seconds, writes
that folder (the robot's records, the evaluator's verdict, `REPRODUCTION.json`) and ends with a summary such as
`"exit_code": 0, "final_claim": "task_done", "who_chose": "nemotron", "matches_recorded_episode": true`. For `masked`
the summary says `"who_chose": "rule"`, `"final_claim": "help_requested:regions_still_to_check_1"` and `"success":
false`: that is the help request the table above shows as expected (`help_with_sponge_present` is true in
`REPRODUCTION.json`). No model is called: the recorded Nemotron answer is reused only if the robot's request text is
byte-identical to the recorded one; otherwise the run stops with a refusal (exit 3). On another machine rendering or
physics may differ: the run then lists each difference from the recorded episode (exit 5) instead of hiding it.
Another system or Python version is refused before anything runs (exit 2).

A run does not rebuild the page or its report. It checks that the robot's records and the evaluator's verdict equal
those of the recorded episode the page shows (`matches_recorded_episode`). The page's code writes the report and its
marks from those recorded records; the exporter from an episode folder to `site/data/` is not included. More in
[simulator/README.md](simulator/README.md).

**Checks without installing anything** (Python 3.8 or later, standard library, offline). From the repository root:

```sh
python3 -B tools/extract_recorded.py --check      # recorded/ equals the page data; fingerprints and answers check out
python3 -B tests/test_nemotron_client.py          # 31 tests of the client, offline (fake transport, fake key)
python3 -B nemotron_client/send_choice.py recorded/demoA_A1_31000/choice_request.json   # dry run: sends nothing
cd assembly_pilot && python3 -B recount.py results/   # recounts the second pilot, exit 0 if all match
cd results && shasum -a 256 -c SHA256SUMS         # or: sha256sum -c SHA256SUMS
cd ../..                                          # back to the repository root
```

What each check covers and the exit codes: [docs/TESTING.md](docs/TESTING.md). One real call with your own key in
`NEBIUS_API_KEY` (billed to you, about 0.0006 USD expected; the answer may differ from the recorded one) adds
`--execute --save-dir my_calls/first_call` to the dry run. Run it once; after an error, look at that folder first.

## How it works

1. Simulated sensors (depth, object labels, noise) feed the robot's perception during a team-written inspection.
2. The robot labels each zone *sponge seen*, *sponge not found in the checked zone* or *still to check*.
3. Code checks eight conditions on the robot's text; if both options are valid, Nemotron picks one, re-checked by code.
4. A team-written program picks and scrubs, or asks a person; the evaluator grades the run from the simulator's truth.
5. The page's code writes the report from the robot's records and marks each claim after Reveal ([details](docs/ARCHITECTURE.md)).

## What Nemotron does and does not do

NVIDIA Nemotron 3 Super (`nvidia/nemotron-3-super-120b-a12b`) on Nebius Token Factory makes one decision in situations
1 and 2: `scrub_bowl` or `ask_help`, once code has found both valid, from a text-only request (the rules, the menu of
two, the robot's state text) at temperature 0. It does not see images, plan, move the arm, write the report or check
anything. Two real calls stand behind the demo, on October 20, 2026 at 22:15:00 (situation 1) and 22:14:56 (situation
2): 2.4 s and 3.3 s, 0.0009942 USD in total, computed from the usage the provider reported (details in
[docs/PROVENANCE.md](docs/PROVENANCE.md)). Situation 3 made no call. The public client in
[`nemotron_client/`](nemotron_client/) rebuilds the exact request; it is tested offline only and has not made a real
call itself. Earlier development episodes on the page replay 9 more calls in which Nemotron wrote whole programs.

## Second pilot: AssembleNIST on Nebius AI Cloud

The pretrained policy of AssembleNIST by Tao Sun (MIT license, not NVIDIA code), used as is on its peg insertion task
R16, ran 960 paired episodes in one CPU-only Nebius AI Cloud Serverless Job on October 26, 2026. Environment success:
317, 315 and 159 of 320; inserted at report time: 192, 197 and 140 of 320 (two measures, kept apart). Our report's
"part held" disagrees with the audit 32 times out of 960. Details and recount: [assembly_pilot/](assembly_pilot/README.md).

## Limits

- Simulation only: one development kitchen, one task, simulated perception (the robot reads depth and object labels
  from the simulated camera, not color images). No physical robot, and no claim about one.
- Three episodes are illustrations, not a reliability estimate or a measure of what Nemotron adds. The programs were
  written by the team; Nemotron only chooses between two of them.
- 17 of the 39 claims are not checkable; the zone checks cover the site file's zones only. Reveal is a display rule,
  not access control: the page's data files already hold the evaluator's results.
- No cloth is lifted. The masked situation uses the thick catalog towel; in a diagnostic on modified recorded sensor
  images, the check behind "sponge not found" was wrong in 13 of 16 cases when a thin cloth was reported as the
  catalog towel. That case remains open.
- Each episode starts with a calibration contact over 20 N (peaks 48.6, 25.9 and 93.9 N), exempt under the bench rule
  and shown after Reveal; the "0 contact incidents" checked before the choice is the robot's own count.
- The simulator runs on macOS only and was tested on one Mac. The page's exporter, the development seed registry and
  the kitchens kept for a later blind validation are not published. Strings in the frozen code keep development notes.
- Second pilot: one task, one batch, an uncalibrated simulator, the simulator's part pose instead of camera perception;
  our report's "part held" was wrong in 30 of the 71 times it said "held".

## Repository contents

| Path | What it is |
| --- | --- |
| `site/` | The replay page: `index.html`, `app.js`, `style.css`, `data/*.js` (the recorded episodes), `frames/` (camera images) |
| `simulator/` | The robot code that recorded the episodes, its evaluator, the demo launcher and the public entry point `demo/run_episode.py` |
| `nemotron_client/` | `choice_prompt.py` (the request text and the answer check) and `send_choice.py` (dry run, or one real call) |
| `recorded/` | The two real Nemotron calls: request and answer, as the page data hold them |
| `tools/`, `tests/` | `extract_recorded.py` (rebuilds or checks `recorded/`) and the client's tests |
| `assembly_pilot/` | The second pilot: its README, six exported result files, the recount, the trial programs that ran in the Job |
| `docs/` | [ARCHITECTURE.md](docs/ARCHITECTURE.md), [TESTING.md](docs/TESTING.md), [PROVENANCE.md](docs/PROVENANCE.md) (fingerprints, what changed for publication), [CLAIMS.md](docs/CLAIMS.md) |

## License and third parties

Apache License 2.0 ([`LICENSE`](LICENSE)); third parties in [`NOTICE`](NOTICE). MuJoCo (Apache-2.0) and NumPy come
from PyPI. The Franka Emika Panda model of MuJoCo Menagerie (Apache-2.0) is fetched at a pinned commit, not
redistributed. AssembleNIST (MIT license, Tao Sun) ran in the second pilot's Job only and is not redistributed; it asks
to be cited with AlphaNIST and SGS. Nemotron 3 Super is used through Nebius Token Factory, no weights included
([model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8)). Names of companies and products
are used descriptively; this project is not endorsed by them.

Links, each opened on October 27, 2026: [hackathon rules](https://nebiusglobalaihackathon.devpost.com/rules),
[NVIDIA Nemotron](https://developer.nvidia.com/topics/ai/nemotron), [Token Factory quickstart](https://docs.tokenfactory.nebius.com/quickstart),
[Serverless Jobs quickstart](https://docs.nebius.com/serverless/quickstart/jobs), [MuJoCo](https://github.com/google-deepmind/mujoco)
(Todorov, Erez and Tassa, IROS 2012), [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (Zakka,
Tassa et al., 2022), [AssembleNIST](https://github.com/suniique/assembly-nist), [AlphaNIST](https://alpha-nist.github.io/),
[SGS](https://sgs-rl.github.io/), [NIST assembly task boards](https://www.nist.gov/el/intelligent-systems-division-73500/robotic-grasping-and-manipulation-assembly/assembly).

Built with AI coding assistants (Claude and Codex). The results shown are checked by the scripts and tests listed in
[TESTING.md](docs/TESTING.md).
