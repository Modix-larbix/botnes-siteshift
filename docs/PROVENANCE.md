# Provenance

Where each part of this repository comes from, what was changed for publication, and the fingerprints that tie it to
what ran. Times are in UTC. A tree sha256 below is the sha256 of the compact JSON list of `[relative path, file
sha256]` pairs, sorted by path. A per-file manifest of the whole repository (path, size, sha256) is kept with the
submission outside the repository, so that it can cover every file, and is not published; inside the repository,
`simulator/MANIFEST_SOURCE_MAP.json` lists every file of `simulator/`.

## The frozen robot code

The robot code was frozen on October 20, 2026 (the frozen copy was made at 08:13 UTC), before the demo episodes were
recorded. Its internal name is DESIG-4: 126 files under `siteagent/`, tree sha256
`5fbd3df1f3df536f8717b25ba3dd427fe2ab1c41c65621ef6e39a353e271e599`, with `tf_guardrails.py` of sha256
`17dad834ef62e99a20d4c8c7c646824a12032164cf707630b52b2de251ad895a`.

`simulator/` holds the 53 files of it that the three situations use: 52 of the 126 files under `siteagent/`, plus
`tf_guardrails.py` (50 Python files and 3 JSON files: limits 0.8, task criteria 1.4, budget configuration). The other
74 files of `siteagent/` are not included: campaign and analysis scripts, older limits and criteria versions, batch
manifests, the development seed registry, development worlds not used here, other launchers and stubs. The selection
is a static closure of the imports of the demo's entry points; the runs of October 27, 2026 imported 47 of the 50
Python files in the main process, ran `siteagent/agent/sandbox_child.py` as the program process, and imported nothing
outside them.

## What changed for publication

Every file of `simulator/` is listed in `simulator/MANIFEST_SOURCE_MAP.json` with its sha256, its origin and its
change from the source. The launcher checks the listed files before every run and refuses to start on any difference.

| Change | Files | What it means |
| --- | --- | --- |
| `identical` | 12 | Same bytes as the source: the 3 robot JSON files, the 3 team programs, 6 record files |
| `comments_or_docstrings_only` | 51 | The 50 robot Python files and `demo/choice_prompt.py`. Comments and docstrings were rewritten to explain the code instead of the project's history. Proof: the syntax tree without docstrings (Python 3.12 `ast.dump`, positions ignored; comments never reach it) is the same as the source's, `code_ast_sha256` equal to `source_code_ast_sha256`. The launcher recomputes it for every robot file before each run. |
| `publication_edits` | 2 | `demo/run_demo.py` and `demo/demo_client.py`, the launcher that recorded the episodes: same logic, with the replaced values listed in the manifest (the development lab's absolute path, the local time zone of their time stamps, now UTC, references to private review notes, one message that named a person, the reviewer's name in the seed table records, now "review", the file name of the development seed registry, and the name of the development spending journal). |
| `redacted` | 4 | The two recorded choice requests and call receipts: absolute paths (the code root and the record folder), an internal spending authorization id, a marker and the development lab's spending journal fields removed. Each folder's `PROVENANCE.json` links every file to the sha256 of its original. |
| `new` | 9 | Written for publication: `demo/run_episode.py` and `demo/public_adapter.py` (the public entry point and its checks), the demo seed registry (only kitchen 31000 and the three seed pairs), the recorded facts, the two `PROVENANCE.json`, the Panda fetch script and its file list |

The launcher's original check of a recorded choice required the whole 126-file tree at the absolute path of the
development machine. `demo/public_adapter.py` replaces it with a relocation rule: every listed file has its listed
sha256, no unlisted file sits in the code folders, every robot file is the source's bytes or a declared comment edit
with the same syntax tree, and the record names the source tree and the source `tf_guardrails.py`. The demo client
still refuses a request text that differs from the recorded one.

Some string values inside the frozen robot code and its JSON files still carry development notes (names, dates,
review labels, a time zone, a few labels in a second language). They were left as they are: changing them would
change the frozen code.

`simulator/` fingerprint of the part the launcher checks (`code_and_data_sha256`, over the 78 listed code and data
files, printed by `demo/run_episode.py --check-only`):
`b582d11d8eb9fe97502fbc43638ed1b77ca9c4361b328970c8a1b8c4bd6fd062`. The launcher's code check needs Python 3.12 (the
text of a syntax tree changes between Python versions); another version is refused.

## The Panda model

Not distributed. `simulator/assets/fetch_panda.sh` downloads `franka_emika_panda` from MuJoCo Menagerie
(<https://github.com/google-deepmind/mujoco_menagerie>, Apache-2.0) at commit
`71f066ad0be9cd271f7ed58c030243ef157af9f4` and checks 70 files (`panda.xml`, 67 meshes, `LICENSE`, `README.md`)
against `simulator/assets/PANDA_ASSETS.sha256` (sha256 `458570d9fadda0833c65ac363c60c75fafa4c323447b0c681fe9b1eb4c167455`).
These 70 files have the same bytes as the copy the recorded episodes used. The robot code loads the model and edits it
in memory (see `NOTICE`).

## The recorded episodes and Nemotron calls

The two Nemotron choices were made on October 20, 2026, at 22:14:56 (situation 2, sponge moved, scene A1) and
22:15:00 (situation 1, sponge in its place, scene A4), each with one request sent once. The three episodes shown on the
page were recorded with the frozen code between 22:19 and 22:23 the same day: A1 and A4 replay the recorded answers,
A5 (situation 3, sponge masked) used the rule and made no call. The two calls were sent by a development script that
is not included, through the robot code's guarded client (`simulator/siteagent/agent/real_client.py`, which the public
launcher never imports).

| Situation | Request fingerprint (sha256 of the canonical JSON body) | Answer | Input / output tokens (reasoning) | Latency | Token cost |
| --- | --- | --- | --- | --- | --- |
| 1, sponge in its place (A4) | `37b01a38b0bb034786b1997d5b988ef3bfa60dce3e41833e891858e38950ab93` | `scrub_bowl` | 698 / 204 (164) | 2.4 s | 0.0003930 USD |
| 2, sponge moved (A1) | `47da009a38512d10e505f3e1e18bb765f1d1360223f3d31ad66491d76683efb7` | `scrub_bowl` | 726 / 426 (380) | 3.3 s | 0.0006012 USD |

Costs are computed from the usage the provider reported, at 0.30 USD per million input tokens and 0.90 USD per million
output tokens (0.0009942 USD for both); they are not a reading of an invoice. The token counts and latencies are in
the call receipts, `simulator/recorded_choices/*/choice_call.json`. The earlier development episodes on the page (robot code of October 7, 2026) replay 9 more
recorded calls in which Nemotron wrote the robot's programs, 0.0546 USD of token cost. None of these figures is the
project's total Token Factory spend.

The same calls appear in two forms. `recorded/` holds the request and the answer of each call as the page data carry
them, rebuilt and checked by `tools/extract_recorded.py`. `simulator/recorded_choices/` holds the five files the
launcher needs to replay a choice (request, request sha256, answer, call receipt, client log), redacted as above;
its `choice_response.json` also holds the model's separate reasoning text and the usage the provider reported. The raw
provider responses stay private.

## The replay page

`site/` is a copy of the page export of October 27, 2026 (version j): 518 files, 6,882,299 bytes, tree sha256
`794f5f897869668ceb20b65256cbb3ede40d0d20911aa3ae41019e14c2d571f5`. It differs from an earlier export of the same day
(version h, tree sha256 `a2358b1d909eb486112f8483f07aca38ac61926053fb6ca2b1203889ff824d39`) in two files only.
`site/app.js`: comments rewritten, a reading map added at the top, and the message shown when the data file does not
load; a token comparison of the two files (comments removed) finds one differing token, that message. `site/style.css`:
comments only; without its comments it is byte-identical. The data, the images and the scoring code are unchanged, so
the scores shown are the same.

The page data (`site/data/*.js`) were exported from the recorded episode folders by an exporter that is not included.
They keep, as recorded, some internal identifiers: names of files of the development lab, version labels, the
recorded local time stamps with their UTC offset, dates written day first, and labels in a second language (the
`zone_labels_fr` fields). The page shows dates as "Oct 20, 2026" and call times in UTC. The call records of
`recorded/` and `simulator/recorded_choices/` keep their recorded time stamps too: a call made on October 20 at 22:14
UTC is written with the next day's date and a +02:00 offset, and `recorded/*/choice_response.json` keeps the page
data's day-first `call_note`.

## The second pilot

`assembly_pilot/results/` holds six of the files the Nebius AI Cloud Job exported on October 26, 2026, byte-identical;
`results/SHA256SUMS` lists them, and `recount.py` checks them against the `evidence_manifest.json` written inside the
Job. `assembly_pilot/trial/` holds the four program files that ran in the Job, byte-identical to what ran, comments
included. For publication, only the module docstring and two comments of `recount.py` changed, to say that it compares
with the constants `STATED` and does not read the README (same syntax tree without docstrings; its output is
byte-identical to `recount_output.json`).
