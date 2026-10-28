# SiteShift simulator (macOS)

This folder holds the simulation code behind the three SiteShift situations, and a launcher that re-runs each one
offline, from this folder, and compares the result with the recorded episode shown on the replay page. For the whole
project, see the [repository README](../README.md); the path from observation to verdict through these files is in
[docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md), and what each test proves in [docs/TESTING.md](../docs/TESTING.md).

**Simulation only:** MuJoCo physics, a simulated camera and simulated perception, a development kitchen (kitchen
31000). No physical robot.

## What runs

1. The robot inspects the kitchen: it opens the drawers, looks at each zone it can reach and labels it from its own
   readings: *sponge seen*, *sponge not found in the checked zone*, or *still to check*.
2. Code reads the robot's own state text and lists the next steps that are valid: scrub the bowl with the sponge, or
   ask a person for help.
3. When both are valid (situations `in_place` and `moved`), the choice is the one NVIDIA Nemotron 3 Super made from a
   text summary when the episode was recorded. The launcher serves that recorded answer only if the robot's request
   text in this run has exactly the recorded sha256; otherwise the run stops with a refusal. No model is called. When
   only "ask for help" is valid (`masked`), a rule takes it.
4. The robot runs the chosen program (programs written by the team, in `demo/programs/`): pick the sponge and scrub
   the bowl, or stop and ask for help.
5. The evaluator grades the episode from the simulator's records, which the robot never sees.

| Situation | Scene | What changed in the kitchen | Noise / camera seed | Who chooses |
| --- | --- | --- | --- | --- |
| `in_place` | A4 | nothing: the sponge is where the site file puts it | 960040 / 960041 | recorded Nemotron answer |
| `moved` | A1 | the sponge was moved to another drawer | 960000 / 960001 | recorded Nemotron answer |
| `masked` | A5 | a towel lies over the sponge in its drawer | 960050 / 960051 | rule (one valid option) |

## Install (once)

Needs macOS, Python 3.12, [uv](https://docs.astral.sh/uv/) (or pip), git, and network access to PyPI and GitHub for
this step only (uv also downloads Python 3.12 if it is missing).

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
sh assets/fetch_panda.sh
```

Without uv, with a Python 3.12 already installed: `python3.12 -m venv .venv && .venv/bin/pip install -r
requirements.txt`, then `sh assets/fetch_panda.sh`.

`fetch_panda.sh` downloads the Franka Emika Panda model from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) at a pinned commit (Apache-2.0) into
`menagerie/franka_emika_panda/`, and checks all 70 files against `assets/PANDA_ASSETS.sha256`. The model is not part
of this repository.

## Run

```sh
env -u NEBIUS_API_KEY -u OPENAI_API_KEY PYTHONDONTWRITEBYTECODE=1 \
    .venv/bin/python -B demo/run_episode.py --situation moved --out runs/moved_1
```

Replace `moved` with `in_place` or `masked`, with a new `--out` each time. Each run takes about 20 to 30 seconds and
needs no network. The launcher refuses to start with a model API key in its environment (it never calls a model), or
under another Python than 3.12 (its code check compares Python 3.12 syntax trees), and runs without writing compiled
Python files next to the code. `--check-only` checks everything without simulating and prints `code_and_data_sha256`,
the fingerprint of the 78 checked files: compare it with the value in
[docs/PROVENANCE.md](../docs/PROVENANCE.md).

MuJoCo may print a warning that `ARB_clip_control` is unavailable and depth accuracy will be limited, and writes it to
`MUJOCO_LOG.TXT` in the folder the run starts from (the repository's `.gitignore` leaves that file out). The runs
recorded in [RUN_PROOF.md](RUN_PROOF.md) printed it too and still matched the recorded episodes.

For `masked`, the summary says `"who_chose": "rule"`, `"final_claim": "help_requested:regions_still_to_check_1"`,
`"outcome_class": "search_unsuccessful"` and `"success": false`: the robot asked for help while the sponge was there,
which is the expected end (`help_with_sponge_present` true and `false_not_found` false in `REPRODUCTION.json`).

Each run writes a new folder (`--out` must not exist): the robot's records (`agent_record.json`, `requests.json`), the
evaluator's verdict (`oracle_verdict.json`, `episode_row.json`) and `REPRODUCTION.json`, which compares the run with
the recorded episode (`recorded_choices/RECORDED_EPISODE_FACTS.json`). A run is a local reproduction: it never
replaces a recorded episode.

| Exit code | Meaning |
| --- | --- |
| 0 | finished, and equal to the recorded episode on every compared fact |
| 2 | refused before simulating (nothing written): platform, Python version, files, Panda model, output folder, environment |
| 3 | the demo client refused during the episode, for example a request text different from the recorded one |
| 4 | compiled Python files appeared in the code folders |
| 5 | finished, but different from the recorded episode (listed in `REPRODUCTION.json`) |
| 1 | an unexpected error |

## Tests

```sh
env -u NEBIUS_API_KEY -u OPENAI_API_KEY PYTHONDONTWRITEBYTECODE=1 \
    .venv/bin/python -B -m unittest discover -s tests -v
```

20 offline checks of the launcher (no package needed, any Python 3.12): a copy at another path is accepted when its
files match `MANIFEST_SOURCE_MAP.json`; one changed code byte, a missing call receipt, an altered answer or a different
request text is refused; a run never writes into an existing folder; other platforms and other Python versions are
refused.

## Layout

| Path | What it is |
| --- | --- |
| `siteagent/`, `tf_guardrails.py` | the robot, its world and its evaluator: 53 files of the frozen code that produced the recorded episodes, with the same code (only comments and docstrings were rewritten, see below) |
| `demo/run_episode.py`, `demo/public_adapter.py` | the public launcher and its checks |
| `demo/run_demo.py`, `demo/demo_client.py`, `demo/choice_prompt.py`, `demo/programs/` | the original demo launcher, client and choice prompt (same logic; comments rewritten, and a few values replaced for publication, listed in `MANIFEST_SOURCE_MAP.json`) and the team programs (unchanged) |
| `demo/demo_seed_registry.json` | the demo's kitchen and seed pairs, read by the world code's refusal rules |
| `recorded_choices/` | the two recorded Nemotron choices (request, answer, call receipt, client log), private fields removed, with `PROVENANCE.json` |
| `assets/` | the Panda download script and the sha256 of its files |
| `MANIFEST_SOURCE_MAP.json` | every file with its sha256 and source; checked before each run |

Where to start reading: `demo/run_episode.py` calls `run_agent_episode` in `siteagent/agent/bench_agent.py`, which
builds the world and the evaluator (`siteagent/bench_task.py`, `siteagent/world/`), runs the agent loop
(`siteagent/agent/loop.py`, programs isolated by `siteagent/agent/sandbox.py`), perception and zone labels
(`siteagent/perception/`, `siteagent/sitemodel.py`, `siteagent/search.py`), gestures (`siteagent/robot/skills.py`)
and grading (`siteagent/taskcheck.py` with `siteagent/config/task_criteria_v1_4.json`). `demo/demo_client.py`
answers the loop: inspection program, menu of next steps, recorded choice.

## Limits

- macOS only: robot programs run inside the macOS sandbox (`sandbox-exec`). On another system the launcher refuses;
  running there would need a different isolation and a new review. Started from inside another macOS sandbox, the
  robot's programs cannot start: the run then ends with differences from the recorded episode (exit 5), not with a
  result.
- The comparison with the recorded episode is exact. On other hardware or other library builds, rendering or physics
  may differ slightly; the launcher then reports the differences, or the demo client refuses a changed request text.
  It never serves the recorded answer on a different text.
- The robot code is the frozen code. Its comments and docstrings were rewritten to explain the code instead of the
  project's history; the code itself is unchanged, and `MANIFEST_SOURCE_MAP.json` proves it per file (the same syntax
  tree without docstrings as the original, `code_ast_sha256`, checked again before every run). Comments that mention a
  test refer to the project's own test suite, which is not part of this package. Some string values inside the frozen
  code and its JSON files still carry development notes (names, dates, review labels, a time zone); they were left
  as they are because changing them would change the frozen code.
- In the original launcher (`demo/run_demo.py`, `demo/demo_client.py`), the folder paths of the development lab, the
  local time zone of their time stamps (now UTC), the references to private review notes, one message, the reviewer's
  name in the seed table records and the file names of the development seed registry and spending journal were
  replaced for publication. Their logic is unchanged; each replaced value is listed in `MANIFEST_SOURCE_MAP.json`
  (`publication_edits`). Their default folders point to `lab_not_included/`, which does not exist here:
  `demo/run_episode.py` passes every path itself.
