#!/bin/zsh
# Double-click on macOS, or run ./start.command from the repository folder.
set -eu
cd "$(dirname "$0")"
if [[ ! -x simulator/.venv/bin/python ]]; then
  print 'Install the local environment first: sh setup.sh'
  exit 1
fi
print 'Keep this window open while using the simulation. Open the address printed below in your browser.'
exec simulator/.venv/bin/python -B viewer/server.py "$@"
