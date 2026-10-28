#!/usr/bin/env bash
# Copyright 2026 Botnes (Moad Larabi)
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
# SIMULATION ONLY. Entry point of ONE disposable Nebius CPU Job (Linux x86-64, ubuntu:24.04 image) for the
# AssembleNIST bounded trial. Never run on a workstation. It creates no cloud resource and needs no secret.
# Injected files (each < 64 KiB): /opt/job_entry.sh, /opt/run_trial.py, /opt/evidence_export.py, /opt/requirements.txt.
# Evidence (review T1-T3, 26/10/2026): always sent through stdout as BOTNES_EVIDENCE packages (evidence_export.py):
# installed versions after the setup, each phase's plan, each condition as soon as it ends, then a common finalisation
# per phase (success, error, refused projection, ABORTED, TERM) and, in the EXIT trap, a job finalisation with ONE
# BOTNES_EVIDENCE_FINAL line. Without that line the package is INCOMPLETE. Rebuilt and verified on the Mac with
# evidence_export.py decode. A mounted bucket at $BOTNES_OUT_DIR (default /out), if any, only gets an extra copy.
# Bucket (review A1, 26/10/2026): only a NEW bucket holding nothing else, mounted with a NAMED credentials profile whose
# key is scoped to that one bucket (never the 'default' profile), the key revoked after retrieval; or no bucket at all
# (evidence through the logs only, videos only under the optional cap). Before any third-party code runs, this script stops if a secret-like variable, a
# credentials file or credential-like mount options are visible in the container (values never printed).
# Exit codes: 0 complete, 3 partial (time guard), 4 error, 5 smoke incomplete, 6 bank projection over its cap
# (main run stopped before the bank build), 7 main run ended by TERM after exporting its evidence, 10 smoke failed
# (main run not started), 137 a phase killed after its 30 s grace, 143 this shell got TERM, 2 refused at start.
set -euo pipefail
umask 022

JOB_T0=$(date +%s)
JOB_BUDGET_MIN=${BOTNES_JOB_BUDGET_MIN:-110}   # the Job --timeout is 2h; the last 10 min are kept for copy/teardown
UPSTREAM_URL="https://github.com/suniique/assembly-nist"
UPSTREAM_COMMIT="eb5d15e004591d380c0d521da67e025ebee37703"
CKPT_SHA256="eb2cd4e9aacda2d26d23c059570522559e9fcd812a64f0aef60daea83408581e"
TORCH_PIN="torch==2.14.1+cpu"
TORCH_INDEX="https://download.pytorch.org/whl/cpu"

say() { printf '[job %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die() { say "ERROR: $*"; exit 2; }
left() { echo $(( JOB_T0 + JOB_BUDGET_MIN * 60 - $(date +%s) )); }

# the result line is printed on every exit, a refusal at start included (exit 2)
RUN_ID="assemblenist_R16_$(date -u +%Y%m%dT%H%M%SZ)"
MIRROR=""; LOCAL_OUT=""; SETUP_LOG=""
EVIDENCE=/opt/evidence_export.py
finish() {
  local rc=$?
  set +eu
  trap '' TERM INT                            # the final export is not cut by a second TERM
  if [[ $rc -ne 0 && -n "$SETUP_LOG" && -f "$SETUP_LOG" ]]; then say "setup log, last 40 lines:"; tail -n 40 "$SETUP_LOG"; fi
  # job finalisation (review T1): send whatever is still unsent, job_manifest.json, then the single final marker
  if [[ -n "$LOCAL_OUT" && -d "$LOCAL_OUT" && -r "$EVIDENCE" ]] && command -v python3.12 >/dev/null 2>&1; then
    timeout 120 python3.12 -I -B "$EVIDENCE" final --root "$LOCAL_OUT" --run "$RUN_ID" --job-exit "$rc" \
      --setup-log "$SETUP_LOG" || say "final evidence export failed: the package is INCOMPLETE by rule"
  else
    say "no evidence export possible (no Python yet or no output folder): the package is INCOMPLETE by rule"
  fi
  if [[ -n "$MIRROR" && -n "$LOCAL_OUT" ]]; then cp -rn "$LOCAL_OUT/." "$MIRROR/" 2>/dev/null; fi
  printf 'BOTNES_JOB_RESULT {"label":"SIMULATION","run":"%s","exit":%d,"bucket":%s,"minutes":%d}\n' \
    "$RUN_ID" "$rc" "$([[ -n "$MIRROR" ]] && echo true || echo false)" $(( ($(date +%s) - JOB_T0) / 60 ))
}
trap finish EXIT
# >>> phase runner (review T1). This shell is PID 1 of the container: without a TERM trap it would ignore TERM, and
# the platform's later KILL would leave no evidence. Each phase runs in the background so that a TERM reaching this
# shell is forwarded at once to the phase (timeout passes it on to Python, which exports its evidence and exits 7).
PHASE_PID=""; PHASE_RC=0; TERMINATED=0
on_term() {
  TERMINATED=1
  if [[ -n "$PHASE_PID" ]]; then
    say "TERM received: forwarded to the running phase (it exports its evidence), then the final export"
    kill -TERM "$PHASE_PID" 2>/dev/null || true
  else
    say "TERM received outside a phase: final export of whatever exists"
    exit 143
  fi
}
run_phase() {   # run_phase SECONDS COMMAND...: the phase's own exit code goes to PHASE_RC
  local secs=$1 r
  shift
  set +e
  timeout --preserve-status --kill-after=30s "$secs" "$@" &
  PHASE_PID=$!
  while :; do
    wait "$PHASE_PID"; PHASE_RC=$?
    kill -0 "$PHASE_PID" 2>/dev/null || break   # still there: the wait was cut by a trapped signal, wait again
  done
  if [[ $PHASE_RC -gt 128 ]]; then wait "$PHASE_PID" 2>/dev/null; r=$?; [[ $r -eq 127 ]] || PHASE_RC=$r; fi
  PHASE_PID=""
  set -e
}
# <<< phase runner
trap on_term TERM INT

[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die "Linux x86_64 required"
[[ "${BOTNES_DISPOSABLE_JOB:-}" == 1 ]] || die "BOTNES_DISPOSABLE_JOB=1 required (disposable Job only)"
[[ "$(id -u)" == 0 ]] || die "the disposable container must run as root"
for f in /opt/run_trial.py "$EVIDENCE" /opt/requirements.txt; do [[ -r "$f" ]] || die "missing injected file $f"; done
for v in NEBIUS_API_KEY NEBIUS_IAM_TOKEN OPENAI_API_KEY NVIDIA_API_KEY NGC_API_KEY HF_TOKEN AWS_ACCESS_KEY_ID \
         AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN; do
  [[ -z "${!v:-}" ]] || die "variable $v is set: this Job must carry no secret (third-party code runs here)"
done
# review A1: any secret-like variable NAME (values never printed), credentials files, credential-like mount options
SECRET_NAMES=$(compgen -e | grep -Ei '(KEY|TOKEN|SECRET|PASSW|CREDENTIAL)' | tr '\n' ' ' || true)
[[ -z "$SECRET_NAMES" ]] || die "secret-like variable(s) set: ${SECRET_NAMES}(values not printed); this Job must carry no secret"
for f in /root/.aws /root/.nebius /root/.config/nebius /root/.config/gcloud /root/.netrc /root/.docker/config.json \
         "${HOME:-/root}/.aws" "${AWS_SHARED_CREDENTIALS_FILE:-/nonexistent}" "${AWS_CONFIG_FILE:-/nonexistent}"; do
  [[ ! -e "$f" ]] || die "credentials path visible in the container: $f (content not read); run without it"
done
if grep -qiE '(secret|access_?key|passw|token|credential)' /proc/mounts 2>/dev/null; then
  die "credential-like mount options visible in /proc/mounts (not printed); run without a bucket or fix the mount"
fi

WORK=/opt/botnes-trial
mkdir "$WORK"                                   # must not exist: nothing is reused from an earlier run
LOCAL_OUT="$WORK/out/$RUN_ID"
mkdir -p "$LOCAL_OUT"
SETUP_LOG="$LOCAL_OUT/setup_log.txt"
OUT_ROOT="${BOTNES_OUT_DIR:-/out}"
if [[ -d "$OUT_ROOT" ]] && mkdir "$OUT_ROOT/$RUN_ID" 2>/dev/null; then
  if printf 'SIMULATION run %s started %s\n' "$RUN_ID" "$(date -u +%FT%TZ)" > "$OUT_ROOT/$RUN_ID/job_started.txt"; then
    MIRROR="$OUT_ROOT/$RUN_ID"
  fi
fi
MIRROR_SMOKE=(); MIRROR_MAIN=()
if [[ -n "$MIRROR" ]]; then MIRROR_SMOKE=(--mirror "$MIRROR/smoke"); MIRROR_MAIN=(--mirror "$MIRROR/main"); fi

say "SIMULATION run $RUN_ID, $(nproc) vCPU, evidence through stdout, bucket copy: ${MIRROR:-none}"
sha256sum /opt/job_entry.sh /opt/run_trial.py "$EVIDENCE" /opt/requirements.txt 2>/dev/null | sed 's/^/[job] injected /' || true

export DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC
say "apt: Python 3.12, git, Mesa with OSMesa (same native set as the 18/10 probe)"
timeout 600 apt-get -q update >>"$SETUP_LOG" 2>&1
timeout 900 apt-get -q install -y --no-install-recommends ca-certificates git python3.12 python3.12-venv \
  libosmesa6 libgl1 libglx-mesa0 libgl1-mesa-dri libegl-mesa0 mesa-libgallium libgbm1 libglfw3 >>"$SETUP_LOG" 2>&1
dpkg-query -W -f='${Package}\t${Version}\n' > "$LOCAL_OUT/native_packages.tsv"

say "fetch AssembleNIST at commit $UPSTREAM_COMMIT"
UP="$WORK/assembly-nist"
git init -q "$UP"
git -C "$UP" remote add origin "$UPSTREAM_URL"
GIT_LFS_SKIP_SMUDGE=1 GIT_TERMINAL_PROMPT=0 timeout 900 \
  git -C "$UP" -c protocol.file.allow=never fetch -q --depth 1 origin "$UPSTREAM_COMMIT" >>"$SETUP_LOG" 2>&1
git -C "$UP" -c advice.detachedHead=false checkout -q FETCH_HEAD
[[ "$(git -C "$UP" rev-parse HEAD)" == "$UPSTREAM_COMMIT" ]] || die "upstream commit mismatch"
printf '%s  %s\n' "$CKPT_SHA256" "$UP/checkpoints/e8192_it12500.pt" | sha256sum --check --status \
  || die "checkpoint SHA-256 mismatch"

say "Python venv: $TORCH_PIN from the PyTorch CPU index, the rest pinned from PyPI, binary wheels only"
python3.12 -m venv "$WORK/venv" >>"$SETUP_LOG" 2>&1
PY="$WORK/venv/bin/python"
PIPI=("$PY" -m pip --disable-pip-version-check --no-input install --no-cache-dir --only-binary :all: -q)
timeout 1200 "${PIPI[@]}" --index-url "$TORCH_INDEX" "$TORCH_PIN" >>"$SETUP_LOG" 2>&1
timeout 900 "${PIPI[@]}" --index-url https://pypi.org/simple -r /opt/requirements.txt >>"$SETUP_LOG" 2>&1
"$PY" -m pip check >>"$SETUP_LOG" 2>&1 || die "pip check failed"
"$PY" -m pip freeze > "$LOCAL_OUT/pip_freeze.txt"
say "setup done after $(( ($(date +%s) - JOB_T0) / 60 )) min"
# installed versions leave at once (progressive export); the job finalisation would send them anyway
timeout 60 python3.12 -I -B "$EVIDENCE" emit --root "$LOCAL_OUT" --run "$RUN_ID" --pkg setup \
  pip_freeze.txt native_packages.tsv || say "setup evidence not sent now; the job finalisation retries"

export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBGL_ALWAYS_SOFTWARE=true GALLIUM_DRIVER=llvmpipe
NT=$(nproc)
COMMON=(--upstream "$UP" --upstream-commit "$UPSTREAM_COMMIT" --nthread "$NT" --job-start-epoch "$JOB_T0"
        --job-budget-min "$JOB_BUDGET_MIN" --stdout-export --run-id "$RUN_ID" --evidence-root "$LOCAL_OUT"
        --evidence-ledger "$LOCAL_OUT/evidence_ledger.jsonl")
cd "$WORK"

# The phases print straight to stdout (the Job log is the evidence channel; no local log copy that would contain the
# export lines a second time).
say "smoke run: 5 start states (timed, for the bank projection), 3 x 2 episodes of 1.2 s, one short video"
run_phase 1200 "$PY" -I -B /opt/run_trial.py --mode smoke "${COMMON[@]}" \
  --out "$LOCAL_OUT/smoke" --data "$WORK/data_smoke" "${MIRROR_SMOKE[@]}"
[[ $TERMINATED -eq 0 ]] || exit 143
[[ $PHASE_RC -eq 0 ]] || { say "smoke failed (exit $PHASE_RC): main run not started"; exit 10; }

REMAIN=$(left)
[[ $REMAIN -gt 900 ]] || die "less than 15 min left for the main run"
say "main run: reset bank (README recipe, R16 only) then 3 x 320 episodes, $(( REMAIN / 60 )) min left"
run_phase "$REMAIN" "$PY" -I -B /opt/run_trial.py --mode main "${COMMON[@]}" --bank-workers 16 \
  --smoke-plan "$LOCAL_OUT/smoke/plan.json" --out "$LOCAL_OUT/main" --data "$WORK/data" "${MIRROR_MAIN[@]}"
[[ $TERMINATED -eq 0 ]] || exit 143
say "main run exit $PHASE_RC (0 complete, 3 partial by time guard, 4 error, 6 bank projection over its cap, 7 TERM)"
exit "$PHASE_RC"
