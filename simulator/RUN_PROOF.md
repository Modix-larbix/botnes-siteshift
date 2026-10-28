# Run proof

What was run to check this folder, on October 27, 2026 (times in UTC), and what it shows.

Machine: Apple silicon Mac, macOS 27.0.1 (arm64), Python 3.12.13. No model was called and no API key was present in
any run.

## 1. What differs from the code that recorded the episodes

**Robot code (`siteagent/`, `tf_guardrails.py`, 50 Python files and 3 JSON files): same code.** The comments and
docstrings of the 50 Python files were rewritten for readers: project history, review labels, dates and references
to files that are not in this package were removed or replaced by the reason they carried, and every module got a
short docstring saying what it does and what it does not do. Nothing else changed: no name, no string literal, no
constant, no layout of code. The 3 JSON files and the 3 team programs (`demo/programs/*.bs`) are byte-identical.

Proof, for each of these Python files: Python 3.12 `ast.dump` of the file with every module, class and function
docstring removed (`include_attributes=False`, so positions and comments do not count) is the same before and after.
`MANIFEST_SOURCE_MAP.json` records it per file (`change`: `comments_or_docstrings_only`, with
`source_code_ast_sha256` equal to `code_ast_sha256`), and before every run the launcher recomputes the syntax-tree
hash of each robot file and refuses to start when it differs from the original's. No robot file reads its own
docstrings or source at run time. Because the text of `ast.dump` changes between Python versions, the launcher
refuses any Python other than 3.12 instead of reporting every file as changed.

**Original launcher (`demo/run_demo.py`, `demo/demo_client.py`): same logic, a few values replaced for
publication.** Besides their comments and docstrings, these values changed (each listed in `MANIFEST_SOURCE_MAP.json`
under `publication_edits`): the development lab's absolute path, now a placeholder inside this package that does not
exist and that `demo/run_episode.py` does not use; the time zone of their time stamps, now UTC; references to private
review notes and day-first dates in the seed-table records they write into `DEMO_IDENTITY.json`; the reviewer's name
in those records, now "review"; the file name of the development seed registry (the default of the original command
line, which `demo/run_episode.py` does not use) and the name of the development spending journal; the module
docstring of `demo/run_demo.py` (the `--help` text of its own command line, which `demo/run_episode.py` does not
read); and one printed message that named a person. `demo/choice_prompt.py` has comment and docstring changes only.
The only new code is `demo/run_episode.py` and `demo/public_adapter.py`.

`MANIFEST_SOURCE_MAP.json` change counts: `comments_or_docstrings_only` 51, `publication_edits` 2, `identical` 12 (the
3 robot JSON files, the 3 team programs and 6 record files), `redacted` 4 (record files), `new` 9.
`code_and_data_sha256` (the 78 checked files): `b582d11d8eb9fe97502fbc43638ed1b77ca9c4361b328970c8a1b8c4bd6fd062`.

## 2. Fresh install from a copy of the repository, following its README

The whole repository was copied to a new location. From that copy, the commands of the repository README's block (b)
ran as written, in a clean environment (no API key, no virtual environment, no Python variables, a new uv cache,
so every package came from PyPI):

| Command (from `simulator/`) | Exit code | Time |
| --- | --- | --- |
| `uv venv --python 3.12 .venv` | 0 | 0.1 s |
| `uv pip install --python .venv/bin/python -r requirements.txt` | 0 | 0.7 s (10 packages) |
| `sh assets/fetch_panda.sh` | 0 | 1.8 s |
| `env -u NEBIUS_API_KEY -u OPENAI_API_KEY PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B demo/run_episode.py --situation moved --out runs/moved_1` | 0 | 26.8 s |
| the same with `--situation in_place --out runs/in_place_1` | 0 | 25.5 s |
| the same with `--situation masked --out runs/masked_1` | 0 | 20.0 s |
| `.venv/bin/python -B -m unittest discover -s tests -v` (20 tests) | 0 | 5.4 s |
| `.venv/bin/python -B demo/run_episode.py --situation <each> --check-only` | 0, 0, 0 | under 1 s each |
| the moved command again, same `--out` (folder exists) | 2, refused, nothing written | under 1 s |
| the moved command with a placeholder `NEBIUS_API_KEY` in the environment | 2, refused, nothing written | under 1 s |
| the moved command with Python 3.9 | 2, refused ("needs Python 3.12"), nothing written | under 1 s |
| the moved command without `-B` and without `PYTHONDONTWRITEBYTECODE` | 2, refused, nothing written, no compiled file left | under 1 s |

Times are those of the whole command; the episodes themselves took 22.9 s (moved), 24.5 s (in_place) and 19.3 s
(masked), between 20:37:17 and 20:38:25. The Panda model came from mujoco_menagerie at commit
`71f066ad0be9cd271f7ed58c030243ef157af9f4`; all 70 files matched `assets/PANDA_ASSETS.sha256`. They also have the same
bytes as the copy of the model the recorded episodes used. Each run printed MuJoCo's warning that `ARB_clip_control`
is unavailable and depth accuracy will be limited, and MuJoCo wrote it to `MUJOCO_LOG.TXT` in `simulator/`; the
results below are unaffected. No compiled Python file was left in the copy.

Earlier the same day, the same three situations ran with exit code 0 and the same results on other clean copies: with
a fresh install, with the Python environment that produced the recorded episodes (the same pinned versions, plus
other packages such as Pillow 12.3.0 that were never loaded), and before the last comment, docstring and launcher
value edits.

## 3. Results per situation

| | moved (A1) | in_place (A4) | masked (A5) |
| --- | --- | --- | --- |
| Zone labels at the choice (robot's readings) | sponge seen: drawer_unit:1; still to check: drawer_unit:0; not found in the checked zone: sink_basin, landing_tray, drying_mat | sponge seen: drawer_unit:0; not found in the checked zone: drawer_unit:1, sink_basin, landing_tray, drying_mat | still to check: drawer_unit:0; not found in the checked zone: drawer_unit:1, sink_basin, landing_tray, drying_mat |
| Choice | recorded Nemotron answer: scrub_bowl | recorded Nemotron answer: scrub_bowl | rule: ask_help (only valid option, no model) |
| Choice request sha256 | `47da009a...` equal to the recorded one | `37b01a38...` equal to the recorded one | none (no request) |
| Gestures | pick of the sponge it tracked in drawer_unit:1, scrub felt 3.0 s | pick in drawer_unit:0, scrub felt 3.0 s | none; asks a person for help |
| Final claim | task_done | task_done | help_requested:regions_still_to_check_1 |
| Evaluator | success, 8 of 8 task criteria met | success, 8 of 8 task criteria met | search_unsuccessful, with help_with_sponge_present true and false_not_found false: the sponge was under the towel, so the help request came with the sponge present |
| Contact incidents, grip events | 0, 0 | 0, 0 | 0, 0 |
| Same as the recorded episode | yes, every compared fact | yes, every compared fact | yes, every compared fact |

Compared facts (`recorded_choices/RECORDED_EPISODE_FACTS.json`): the sha256 of every request the robot built, the zone
labels, the choice, the pick positions, the scrub results, the final claim, the evaluator's row and the sha256 of its
verdict. These files are byte-identical to the recorded episode folders in every run: `requests.json` (the robot's full
request texts), `oracle_verdict.json`, `observable_feedback.json`, `trace.json`, and for masked `world_dev_state.json`
(towel and sponge positions after settling). Compared with the earlier runs of the same day, the `result`, `choice`
and comparison parts of `REPRODUCTION.json` are equal; what differs is only wall-clock times, folder paths and what
identifies the code (`code_and_data_sha256` and the sha256 of the edited files).

## 4. What ran

The main process imported 46 of the 50 robot files in moved and in_place, and 47 in masked (which also builds the
towel world with `siteagent/world_dev/occlusion2.py`); `siteagent/agent/sandbox_child.py` ran as the separate,
sandboxed program process. Never imported in these runs: `siteagent/agent/mock_policy.py` and
`siteagent/agent/real_client.py`; they stay because functions of `bench_agent.py` import them on other paths. No file
outside the 50 was imported from the package besides the launcher's own five files in `demo/`. Third-party packages
loaded: mujoco and numpy only. No networking module (socket, ssl, http.client, urllib.request, openai) was loaded in
the main process, and the program process runs under a macOS sandbox profile that denies network access. The whole
run could not also be wrapped in a no-network sandbox: macOS refuses a sandbox inside a sandbox, and the robot starts
its own.

## 5. What this shows, and what it does not

Shown: on this Mac, a fresh copy of the repository, installed from PyPI with the pinned versions and the pinned Panda
model by the README's own commands, re-runs the three situations offline and reproduces the recorded episodes: the
same request texts, so the recorded Nemotron answer was served in moved and in_place; the same zone labels, gestures,
final claims and evaluator verdicts. Rewriting comments and docstrings, and the launcher's publication edits, changed
no result. The launcher refuses a changed code byte, a missing or altered record file, a different request text, an
existing output folder, a key in the environment, a non-macOS platform and another Python version.

Not shown: other Macs (Intel, other macOS versions, other graphics hardware) or other systems; there, rendering or
physics may differ and the launcher would report it or refuse. Nemotron was not called again, so this does not show
it would answer the same way today. The replay page and its claim scores are not recomputed by this package. The
checks guard against mistakes, not against someone who edits both the code and its manifest. This is a simulation:
it says nothing measured about a physical robot.
