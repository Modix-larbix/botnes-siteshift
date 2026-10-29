#!/bin/sh
# Install the visual demo on macOS. Downloads Python packages and the pinned Panda model.
set -eu
cd "$(dirname "$0")"
if [ "$(uname -s)" != Darwin ]; then
  echo 'The simulator currently requires macOS. See README.md for the recorded evidence page on other systems.' >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo 'Install uv first using https://docs.astral.sh/uv/getting-started/installation/ and run sh setup.sh again.' >&2
  exit 1
fi
if [ ! -x simulator/.venv/bin/python ]; then
  uv venv --python 3.12 simulator/.venv
fi
simulator/.venv/bin/python -B -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else "The existing simulator/.venv needs Python 3.12. Use a fresh repository folder.")'
uv pip install --python simulator/.venv/bin/python -r viewer/requirements.txt
(cd simulator && sh assets/fetch_panda.sh)
echo 'Ready. Run ./start.command and open http://127.0.0.1:8766 in your browser.'
