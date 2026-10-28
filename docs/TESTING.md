# Testing: what each check covers, and what it does not

Every command below runs offline, needs no key and calls no model. The simulator's install is the only step that uses
the network (PyPI and GitHub, once).

## 1. Without installing anything

Python 3.8 or later, standard library only (tested with Python 3.9.6 and 3.12.13, with network access denied). From
the repository root:

```sh
python3 -B tools/extract_recorded.py --check
python3 -B tests/test_nemotron_client.py
python3 -B nemotron_client/send_choice.py recorded/demoA_A1_31000/choice_request.json \
    --expect-sha256 47da009a38512d10e505f3e1e18bb765f1d1360223f3d31ad66491d76683efb7
cd assembly_pilot && python3 -B recount.py results/
cd results && shasum -a 256 -c SHA256SUMS
cd ../..
```

| Command | What it checks | What it does not show |
| --- | --- | --- |
| `tools/extract_recorded.py --check` | `recorded/` equals the two calls as the page data hold them; each request's fingerprint is recomputed and each answer passes the answer check | The provider's raw responses (private) |
| `tests/test_nemotron_client.py` | 31 tests of the public client with a fake transport and a fake key (below) | A real call: this client has not made one |
| `send_choice.py` (dry run) | The exact body the client would send; its sha256 equals the recorded fingerprint. Nothing is sent | That Nemotron would give the same answer today |
| `assembly_pilot/recount.py results/` | Recounts the second pilot from the six exported JSON files, checks each file's sha256 against the manifest written inside the Job, and compares every count and keypoint distance with the constants `STATED` in `recount.py` (the numbers its README states for this batch); exit 0 when all match. Its output equals `recount_output.json` | A new evaluation: the raw 25 Hz signals were not exported, so the audit is recounted, not recomputed |
| `shasum -a 256 -c SHA256SUMS` | The six result files are the listed ones | Anything about their content |

The client's 31 tests, by class:

- `Recorded` (5): the two recorded calls and their fingerprints, their total cost, the messages rebuilt by
  `choice_prompt.py`, the recorded answers pass the schema check, `recorded/` equals the page data.
- `DryRun` (2): the dry run prints the exact body and sends nothing; it refuses `--save-dir`.
- `Refusals` (7): a changed user text, a changed system text, an extra request key, a wrong expected fingerprint,
  `--execute` without a key (nothing sent), plain http, a base URL whose host only looks like the expected one.
- `ExecuteWithAFakeOpener` (4): one POST with the exact body and the key never printed, an HTTP error body redacted, a
  key echoed by the server redacted everywhere, an answer outside the allowed options flagged.
- `Redirects` (2): a redirect is refused and nothing else is sent; the key header is never carried by a redirect.
- `AttemptRecord` (11): the folder and the attempt marker exist before the POST; a second run on the same folder is
  refused before any send; a lost answer leaves an "uncertain" record and nothing is sent again; an interrupted send
  leaves the marker; a run started while another is sending is refused; of concurrent runs on one folder only one
  sends; a parent that is a file is refused before any send; a missing parent is created; no key creates no folder; an
  HTTP error is recorded, redacted and returns 2; exit 0 does not mean the option is accepted.

### Exit codes of the client (`nemotron_client/send_choice.py`)

| Code | Meaning |
| --- | --- |
| 0 | The dry run finished, or (`--execute`) a chat completion came back and was processed. **0 does not mean the option is accepted**: read `check.ok` and `option_allowed_by_the_robot_state` in the printed result. When either is false, the demo's code refuses the answer and the robot asks for help. |
| 2 | Either refused before anything was sent, or (`--execute`) the request was sent, or may have been, and no usable answer came back: transport error, HTTP error, refused redirect, a body that is not a chat completion, or a record that could not be finished. **A 2 is not proof that nothing was sent or billed**: the printed lines and `outcome.json` say which case it is. |
| 1 | An unexpected Python error. If "sending ONE request" was printed before it, treat the call as uncertain. |

With `--execute --save-dir NEW_FOLDER`, the folder is created before the send and keeps the request, an attempt marker
and the outcome (`answered`, `http_error`, `redirect_refused`, `not_a_chat_completion` or `uncertain`). After an error
or an interruption, keep the folder and look at it before deciding anything: a new attempt needs a new folder. This
protects one folder against a second send; it is not a de-duplication on the provider's side.

## 2. The simulator (macOS, Python 3.12)

The launcher's tests need no package, only Python 3.12: the `.venv` of the simulator's install, or any Python 3.12.
From the repository root:

```sh
cd simulator
env -u NEBIUS_API_KEY -u OPENAI_API_KEY PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m unittest discover -s tests -v
cd ..
```

20 tests of `demo/public_adapter.py`, `demo/run_episode.py` and the demo client, offline: a copy at another path is
accepted when its files match `MANIFEST_SOURCE_MAP.json`; one changed code byte is refused; an unlisted file is
refused; a comment-only edit is accepted only when declared and its syntax tree is unchanged; a missing call receipt or
client log, an altered answer, or a receipt that disagrees with the answer is refused; a different request text is
refused, in the record and during a run through the demo client; a different loop request is refused; a record of
another code tree is refused; an existing output folder and reserved output names are refused; other platforms and
other Python versions are refused; the launcher turns off the writing of compiled Python files before it imports
anything of the package.

`--check-only` prints `code_and_data_sha256`, the fingerprint of the 78 checked code and data files; compare it with the
value in [PROVENANCE.md](PROVENANCE.md). The checks catch mistakes, not someone who rewrites both a file and its listed
fingerprints in `MANIFEST_SOURCE_MAP.json`: that changes `code_and_data_sha256`.

The runs themselves need the install of [simulator/README.md](../simulator/README.md). Each one compares the run with
the recorded episode (`simulator/recorded_choices/RECORDED_EPISODE_FACTS.json`: the sha256 of every request the robot
built, the zone labels, the choice, the pick positions, the scrub results, the final claim, the evaluator's row and the
sha256 of its verdict) and exits with:

| Code | Meaning |
| --- | --- |
| 0 | Finished, and equal to the recorded episode on every compared fact |
| 2 | Refused before simulating, nothing written: platform, Python version, files, Panda model, output folder, environment |
| 3 | The demo client refused during the episode, for example a request text different from the recorded one |
| 4 | Compiled Python files appeared in the code folders |
| 5 | Finished, but different from the recorded episode; the differences are listed in `REPRODUCTION.json` |
| 1 | An unexpected error |

On October 27, 2026, a fresh copy of the repository on an Apple silicon Mac (macOS 27.0.1, Python 3.12.13, packages
from PyPI, Panda model fetched at the pinned commit), installed and run with the README's own commands, passed the 20
tests and ran the three situations with exit code 0, in about 20 to 30 seconds each; `requests.json`,
`oracle_verdict.json`, `observable_feedback.json` and `trace.json` were byte-identical to the recorded episodes. Details: [simulator/RUN_PROOF.md](../simulator/RUN_PROOF.md). This shows a
reproduction on that machine only: on other hardware or library builds, rendering or physics may differ, and the
launcher then reports the difference (exit 5) or refuses a changed request text (exit 3).

## 3. The replay page

No automated test of the page is in this repository. Five test suites of the development repository check the page
and its data, and are not published because they read files of the development lab: 414 tests in all on the page
export of October 27, 2026 that `site/` holds (182, 72, 49, 44 and 67), all passing on that export. Among other things they check the
data of the three situations, the three zone labels, that the code that marks the claims is unchanged, a scan for
forbidden words in the page's own texts, that on the home screen above the earlier episodes' cards and on the screens
of the three situations nothing of the evaluator, the scene or a camera image is drawn before Reveal (by sentinel
values, in a headless render), the phone width and both color themes, the team's programs and the demo client. These
figures are reported here; they cannot be reproduced from this repository.

A headless-browser check of the page served from 127.0.0.1 (desktop and phone widths, light and dark themes) found 0
requests outside 127.0.0.1, 0 console errors, no evaluator content and no camera image before Reveal in the three
situations, and after Reveal the scores 13/8/0/5, 13/8/0/5 and 13/6/0/7, 39/22/0/17 in total. That check script is
not included either. The earlier episodes' cards, lower on the home screen, describe their scene and outcome up
front, and the trap and Counters screens have no Reveal step: they show the evaluator's results directly. Reveal is a display rule, not
access control: every data file, the evaluator's records included, loads with the page.

## What none of these checks shows

- How a physical robot would do: everything here is simulation.
- A reliability rate: three episodes are illustrations, not a sample.
- What Nemotron would answer today: no check calls a model.
- Protection against a deliberate edit: the checks catch mistakes, not someone who changes a file and its listed
  sha256 together.
