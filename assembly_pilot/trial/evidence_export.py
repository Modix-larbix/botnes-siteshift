#!/usr/bin/env python3
# Copyright 2026 Botnes (Moad Larabi)
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""SIMULATION ONLY. Evidence export and recovery of the AssembleNIST bounded trial (format botnes-evidence-2, 26/10/2026).

Standard library only, on both sides; imports nothing third-party.

In the disposable Job (imported by run_trial.py, called by job_entry.sh), evidence files leave through stdout, the only
channel without a bucket, as packages of numbered and checksummed chunks:
  - progressively: installed versions after the setup, the plan before any rollout, each condition as soon as it ends;
  - by a common finalisation of each phase that runs on success, on any error (also after a finished condition), on a
    refused bank projection, with ABORTED conditions and on TERM;
  - by a job finalisation (EXIT trap of job_entry.sh) that sends whatever is still unsent and prints ONE final marker
    listing every package.
Evidence files are never dropped to fit a size. Only videos are optional (sent while under a cap, otherwise listed as
not exported). Raw signals (signals/*.npz) are never sent through the logs (RAW_SIGNALS_NOTE).

Line protocol (a log prefix such as a timestamp is tolerated before each marker):
  BOTNES_EVIDENCE_BEGIN {meta: run, seq, pkg, sha256 of the tar.gz, bytes, chunks, files with bytes and sha256}
  BOTNES_EVIDENCE <run> <seq> <index> <count> <first 16 hex of sha256(chunk)> <base64 chunk>
  BOTNES_EVIDENCE_END {run, seq, chunks, sha256}
  BOTNES_EVIDENCE_PHASE_END {...}   (for humans reading the log)
  BOTNES_EVIDENCE_FINAL {run, job_exit, packages, interrupted_packages, phases}

On the receiving side (the Mac), `decode` rebuilds the files from saved Job logs into a NEW empty folder, checks every
chunk, package and file against its checksum and the manifests against the files, recounts the episodes and writes
recovery_report.json. A missing chunk, package, manifest, mandatory file or final marker makes the package INCOMPLETE:
never a favourable conclusion and never a silent relaunch. So do episode counts that disagree (planned, results,
robot reports, audits, per condition and combined; count_problems). Log lines saved wrapped in JSON by a log collector
({"log": "BOTNES_EVIDENCE ...", ...}) are unwrapped first. Checksums prove the transport, not the scientific truth of
the content.

  python3 -I -B evidence_export.py decode --log job_logs.txt --out <new folder> [--run RUN_ID]
  exit 0 complete package of a complete run, 1 complete package of a run that did not complete (report it as such),
       3 INCOMPLETE package, 2 refused (no evidence line, several runs without --run, output folder not empty)
"""
from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import io
import json
import os
import re
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath

LABEL = "SIMULATION"
FORMAT = "botnes-evidence-2"
TAG = "BOTNES_EVIDENCE"
CHUNK_CHARS = 2000
PHASES = ("smoke", "main")
# Mandatory files, relative to the job root ("job") or to each phase folder. A run without one of them is reported
# INCOMPLETE; no file is ever dropped to make a package smaller.
MANDATORY = {
    "job": ("pip_freeze.txt", "native_packages.tsv", "job_manifest.json"),
    "smoke": ("plan.json", "results.json", "audit.json", "robot_reports.jsonl", "summary.json", "evidence_manifest.json"),
    "main": ("plan.json", "bank/projection.json", "results.json", "audit.json", "robot_reports.jsonl", "summary.json",
             "evidence_manifest.json"),
}
EVIDENCE_SUFFIXES = (".json", ".jsonl", ".txt", ".tsv")
OPTIONAL_SUFFIXES = (".mp4",)
NEVER_SENT_SUFFIXES = (".npz",)
LOCAL_ONLY = ("evidence_ledger.jsonl", "setup_log.txt")   # bookkeeping; the full setup log (its tail is sent)
RAW_SIGNALS_NOTE = ("Raw 25 Hz signals (signals/*.npz) are never sent through the log export; they stay in the "
                    "disposable container (and in a bucket copy only if one was mounted). From the exported JSON "
                    "anyone can recount every verdict and summary, but the audit cannot be fully recomputed from the "
                    "signals.")
RUN_RE = re.compile(r"[A-Za-z0-9_.:-]{1,80}")
MAX_UNPACKED = 1 << 30


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_json(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _count_lines(p):
    try:
        with open(p, "rb") as f:
            return sum(1 for ln in f if ln.strip())
    except OSError:
        return None


def _write_atomic(p, text):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".part")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, p)


def kind(path):
    name = Path(path).name
    if name in LOCAL_ONLY or name.endswith(".part"):
        return "local"
    if name.endswith(NEVER_SENT_SUFFIXES):
        return "raw_signals"
    if name.endswith(OPTIONAL_SUFFIXES):
        return "optional"
    if name.endswith(EVIDENCE_SUFFIXES):
        return "evidence"
    return "other"


NOT_SENT = {"raw_signals": "never sent through the logs (RAW_SIGNALS_NOTE)", "local": "local bookkeeping, not sent",
            "other": "not an evidence file type, not sent"}


def walk(base):
    base = Path(base)
    if not base.is_dir():
        return []
    return sorted(p for p in base.rglob("*") if p.is_file() and not p.is_symlink())


class Exporter:
    """Sends files of one job root through stdout and records each package in a ledger shared by the job's processes,
    so that nothing is sent twice and the final marker can list every package."""
    raw_signals_note = RAW_SIGNALS_NOTE

    def __init__(self, root, run_id="local", ledger=None, enabled=True, optional_cap_kb=600, chunk=CHUNK_CHARS,
                 stream=None):
        if not RUN_RE.fullmatch(str(run_id)):
            raise ValueError("run id: 1-80 letters, digits or _ . : - only")
        self.root = Path(root).resolve()
        self.run = str(run_id)
        self.ledger = Path(ledger) if ledger else self.root / "evidence_ledger.jsonl"
        self.enabled = bool(enabled)
        self.cap = int(optional_cap_kb) * 1024
        self.chunk = int(chunk)
        self.stream = stream

    def say(self, line):
        s = self.stream or sys.stdout
        s.write(line + "\n")
        s.flush()

    def rel(self, p):
        p = Path(p)
        p = p.resolve() if p.is_absolute() else (self.root / p).resolve()
        try:
            return p.relative_to(self.root).as_posix()
        except ValueError:
            return None

    def entries(self):
        try:
            text = self.ledger.read_text(encoding="utf-8")
        except OSError:
            return []
        out = []
        for ln in text.splitlines():
            try:
                out.append(json.loads(ln))
            except ValueError:
                pass                                     # a line cut by a kill is ignored
        return out

    def sent(self):
        done = {}
        for e in self.entries():
            if e.get("state") == "sent":
                done.update(e.get("files", {}))
        return done

    def _record(self, entry):
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ledger, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, separators=(",", ":")) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def emit(self, pkg, paths):
        """Send every listed file that exists, is not raw signals and was not already sent with the same content, in
        full whatever its size. Returns the package meta, or None when there was nothing to send."""
        if not self.enabled:
            return None
        done, items, seen = self.sent(), [], set()
        for p in paths:
            r = self.rel(p)
            if r is None or r in seen:
                continue
            seen.add(r)
            fp = self.root / r
            if not fp.is_file() or kind(fp) in ("raw_signals", "local"):
                continue
            data = fp.read_bytes()
            h = _sha(data)
            if done.get(r) != h:
                items.append((r, data, h))
        if not items:
            return None
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            for r, data, _ in items:
                ti = tarfile.TarInfo(r)
                ti.size, ti.mode, ti.mtime = len(data), 0o644, 0
                tf.addfile(ti, io.BytesIO(data))
        blob = buf.getvalue()
        b64 = base64.b64encode(blob).decode("ascii")
        n = -(-len(b64) // self.chunk)
        seq = sum(1 for e in self.entries() if e.get("state") == "begun")
        meta = {"v": FORMAT, "label": LABEL, "run": self.run, "seq": seq, "pkg": pkg, "sha256": _sha(blob),
                "bytes": len(blob), "chunks": n, "chunk_chars": self.chunk, "over_envelope": len(b64) > self.cap,
                "files": [{"path": r, "bytes": len(d), "sha256": h} for r, d, h in items]}
        self._record({"state": "begun", "seq": seq, "pkg": pkg, "at": _now()})
        self.say(f"{TAG}_BEGIN " + json.dumps(meta, separators=(",", ":")))
        for j in range(n):
            c = b64[j * self.chunk:(j + 1) * self.chunk]
            self.say(f"{TAG} {self.run} {seq} {j} {n} {_sha(c.encode('ascii'))[:16]} {c}")
        self.say(f"{TAG}_END " + json.dumps({"run": self.run, "seq": seq, "pkg": pkg, "chunks": n,
                                             "sha256": meta["sha256"]}, separators=(",", ":")))
        self._record({"state": "sent", "seq": seq, "pkg": pkg, "sha256": meta["sha256"], "chunks": n,
                      "bytes": len(blob), "files": {r: h for r, _, h in items}})
        return meta


def _jsonl_rows(p):
    try:
        with open(p, encoding="utf-8") as f:
            return [json.loads(ln) for ln in f if ln.strip()]
    except (OSError, ValueError):
        return None


def episode_counts(phase_dir):
    """Planned, completed, reported and audited episodes per condition, read from the files on disk only; and the
    combined files (results.json, robot_reports.jsonl, audit.json) counted per condition, with their episode keys."""
    d = Path(phase_dir)
    plan = (_read_json(d / "plan.json") or {}).get("plan") or {}
    planned = {cid: len(v.get("bank_index", [])) for cid, v in plan.items() if isinstance(v, dict)}
    cdir = d / "conditions"
    cids = set(planned)
    if cdir.is_dir():
        for p in cdir.iterdir():
            for suf in ("_results.json", "_robot_reports.jsonl", "_audit.json", "_ABORTED.json"):
                if p.name.endswith(suf):
                    cids.add(p.name[:-len(suf)])
    per = {}
    for cid in sorted(cids):
        r, au = _read_json(cdir / f"{cid}_results.json"), _read_json(cdir / f"{cid}_audit.json")
        per[cid] = {"planned": planned.get(cid), "completed": len(r.get("episodes", [])) if r else 0,
                    "reports": _count_lines(cdir / f"{cid}_robot_reports.jsonl") or 0,
                    "audits": len(au.get("per_episode", [])) if au else 0,
                    "aborted": (cdir / f"{cid}_ABORTED.json").is_file()}
    res, au = _read_json(d / "results.json"), _read_json(d / "audit.json")
    total = {"planned": sum(v for v in planned.values()), "completed": sum(v["completed"] for v in per.values()),
             "in_results_json": len(res.get("episodes", [])) if res else None,
             "in_robot_reports_jsonl": _count_lines(d / "robot_reports.jsonl"),
             "in_audit_json": len(au.get("per_episode", [])) if au else None}
    files = {"results": d / "results.json", "reports": d / "robot_reports.jsonl", "audits": d / "audit.json"}
    rows = {"results": res.get("episodes") if isinstance(res, dict) else None, "reports": _jsonl_rows(files["reports"]),
            "audits": au.get("per_episode") if isinstance(au, dict) else None}
    combined, keys = {}, {}
    for name, rs in rows.items():
        if not isinstance(rs, list):
            continue
        keys[name] = [(str(r.get("condition")), json.dumps(r.get("episode"))) if isinstance(r, dict) else ("?", "?")
                      for r in rs]
    for name, ks in keys.items():                           # only the combined files that exist and can be read
        for cid, _ in ks:
            combined.setdefault(cid, dict.fromkeys(keys, 0))[name] += 1
    return {"per_condition": per, "total": total, "combined_per_condition": combined,
            "combined_unreadable": sorted(n for n in rows if n not in keys and files[n].is_file()),
            "duplicate_episode_keys": {n: len(k) - len(set(k)) for n, k in keys.items() if len(k) != len(set(k))},
            "episode_keys_differ": len({frozenset(k) for k in keys.values()}) > 1}


def count_problems(c, require_all=False):
    """Episode counts that disagree (review of 26/10, B13): for each condition not ABORTED, its planned episodes, its
    own files (results, robot reports, audits) and the combined files must give the same number; no condition outside
    the plan; no episode twice; the same episodes in the three combined files. require_all: the phase says complete,
    so every planned condition must be there in full. Returns plain sentences, empty when the counts agree."""
    out, per, comb = [], c.get("per_condition") or {}, c.get("combined_per_condition") or {}
    planned = {cid: v.get("planned") for cid, v in per.items() if v.get("planned") is not None}
    for cid in sorted(set(per) | set(comb)):
        v = per.get(cid) or {"planned": None, "completed": 0, "reports": 0, "audits": 0, "aborted": False}
        cb = comb.get(cid)
        if planned and cid not in planned:
            out.append(f"{cid}: episodes of a condition that is not in the plan")
        if v.get("aborted"):
            if v["completed"] or v["reports"] or v["audits"] or (cb and any(cb.values())):
                out.append(f"{cid}: marked ABORTED but episodes are recorded for it")
            continue
        seen = {}
        if v["completed"] or v["reports"] or v["audits"]:
            seen.update({"its results": v["completed"], "its robot reports": v["reports"], "its audits": v["audits"]})
        if cb:
            seen.update({"combined " + {"reports": "robot reports"}.get(n, n): x for n, x in cb.items()})
        if seen and planned.get(cid) is not None and (require_all or v["completed"] or v["reports"] or v["audits"]):
            seen["planned"] = planned[cid]
        elif not seen and require_all and planned.get(cid):
            seen = {"planned": planned[cid], "recorded": 0}
        if len(set(seen.values())) > 1:
            out.append(f"{cid}: " + ", ".join(f"{k} {n}" for k, n in seen.items()))
    t = c.get("total") or {}
    tot = {k: t.get(k) for k in ("in_results_json", "in_robot_reports_jsonl", "in_audit_json") if t.get(k) is not None}
    if len(set(tot.values())) > 1:
        out.append("combined files: " + ", ".join(f"{k} {n}" for k, n in tot.items()))
    for n in c.get("combined_unreadable") or []:
        out.append(f"combined {n}: file present but not readable line by line")
    for n, k in sorted((c.get("duplicate_episode_keys") or {}).items()):
        out.append(f"combined {n}: {k} episode(s) recorded twice")
    if c.get("episode_keys_differ"):
        out.append("combined files: not the same episodes in results, robot reports and audits")
    return out


def _inventory(ex, base, skip, own_pkg):
    done, files, evid, opt = ex.sent(), [], [], []
    for p in walk(base):
        if p.name in skip:
            continue
        r, k = ex.rel(p), kind(p)
        data = p.read_bytes()
        e = {"path": r, "bytes": len(data), "sha256": _sha(data), "kind": k}
        if k in NOT_SENT:
            e["export"] = NOT_SENT[k]
        elif done.get(r) == e["sha256"]:
            e["export"] = "sent earlier"
        elif not ex.enabled:
            e["export"] = "not sent: log export off"
        elif k == "evidence":
            e["export"] = f"sent in package {own_pkg}"
            evid.append(p)
        else:
            opt.append((p, e))
        files.append(e)
    total, chosen = 0, []
    for p, e in opt:                                        # videos: while under the optional cap
        if (total + e["bytes"]) * 4 / 3 <= ex.cap:
            total += e["bytes"]
            chosen.append(p)
            e["export"] = f"sent in package {own_pkg.replace('-final', '-optional')}"
        else:
            e["export"] = f"not exported: over the optional cap of {ex.cap // 1024} KiB of base64"
    return files, evid, chosen


def finalize_phase(ex, phase_dir, phase, exit_code, status, note=None):
    """Common end of one phase, whatever happened (success, error, refused projection, ABORTED, TERM): writes
    evidence_manifest.json (every file with its checksum and export status, mandatory files missing, episode counts)
    and sends everything not sent yet, the manifest included."""
    d = Path(phase_dir).resolve()
    d.mkdir(parents=True, exist_ok=True)
    files, evid, chosen = _inventory(ex, d, ("evidence_manifest.json",), f"{phase}-final")
    missing = [m for m in MANDATORY[phase] if m != "evidence_manifest.json" and not (d / m).is_file()]
    counts = episode_counts(d)
    problems = count_problems(counts, status == "complete")
    manifest = {"label": LABEL, "format": FORMAT, "run": ex.run, "phase": phase, "exit": exit_code, "status": status,
                "note": note, "complete": status == "complete" and not missing and not problems,
                "mandatory": list(MANDATORY[phase]), "mandatory_missing": missing, "episode_counts": counts,
                "episode_count_problems": problems, "raw_signals_sent": False,
                "raw_signals_note": RAW_SIGNALS_NOTE, "files": files, "written_utc": _now()}
    mp = d / "evidence_manifest.json"
    _write_atomic(mp, json.dumps(manifest, indent=1) + "\n")
    ex.emit(f"{phase}-final", evid + [mp])
    if chosen:
        ex.emit(f"{phase}-optional", chosen)
    if ex.enabled:
        ex.say(f"{TAG}_PHASE_END " + json.dumps({"run": ex.run, "phase": phase, "exit": exit_code, "status": status,
                                                 "mandatory_missing": missing, "episode_count_problems": len(problems),
                                                 "episodes": manifest["episode_counts"]["total"]},
                                                separators=(",", ":")))
    return manifest


def finalize_job(ex, job_exit, setup_log=None, tail_lines=400):
    """End of the job (EXIT trap of job_entry.sh), whatever happened: sends what is still unsent (for instance the
    files of a phase killed before its own finalisation), job_manifest.json, then the single final marker."""
    root = ex.root
    if setup_log and Path(setup_log).is_file():
        lines = Path(setup_log).read_bytes().splitlines()[-tail_lines:]
        _write_atomic(root / "setup_log_tail.txt", b"\n".join(lines).decode("utf-8", "replace") + "\n")
    phases = {}
    for ph in PHASES:
        m = _read_json(root / ph / "evidence_manifest.json")
        if m:
            phases[ph] = {k: m.get(k) for k in ("exit", "status", "complete", "mandatory_missing")}
            phases[ph]["episodes"] = (m.get("episode_counts") or {}).get("total")
        elif (root / ph).is_dir():
            phases[ph] = {"status": "manifest missing: the phase ended before its own finalisation", "complete": False}
        else:
            phases[ph] = {"status": "not run", "complete": False}
    files, evid, chosen = _inventory(ex, root, ("job_manifest.json",), "job-final")
    missing = [m for m in MANDATORY["job"] if m != "job_manifest.json" and not (root / m).is_file()]
    manifest = {"label": LABEL, "format": FORMAT, "run": ex.run, "job_exit": job_exit, "phases": phases,
                "job_mandatory_missing": missing, "raw_signals_sent": False, "raw_signals_note": RAW_SIGNALS_NOTE,
                "files": files, "written_utc": _now()}
    _write_atomic(root / "job_manifest.json", json.dumps(manifest, indent=1) + "\n")
    ex.emit("job-final", evid + [root / "job_manifest.json"])
    if chosen:
        ex.emit("job-optional", chosen)
    entries = ex.entries()
    sent = [{k: e.get(k) for k in ("seq", "pkg", "sha256", "chunks", "bytes")} for e in entries
            if e.get("state") == "sent"]
    begun = {e.get("seq") for e in entries if e.get("state") == "begun"}
    final = {"v": FORMAT, "label": LABEL, "run": ex.run, "job_exit": job_exit, "packages": sent,
             "interrupted_packages": sorted(begun - {s["seq"] for s in sent}), "phases": phases,
             "job_mandatory_missing": missing, "raw_signals_sent": False, "written_utc": _now()}
    ex.say(f"{TAG}_FINAL " + json.dumps(final, separators=(",", ":")))
    return final


# --------------------------------------------------------------------------- receiving side
def _unsafe(name):
    parts = name.split("/")
    return (not name or name.startswith("/") or "\\" in name or ":" in name
            or any(x in ("", ".", "..") for x in parts) or ".." in PurePosixPath(name).parts)


def _unwrap(ln, k):
    """A line that a log collector saved as JSON, possibly after a prefix (Docker or Kubernetes style:
    {"log": "BOTNES_EVIDENCE ...\\n", "stream": "stdout"}): the first string inside it that holds the tag. Otherwise
    the line itself (a raw line keeps its prefix: a BEGIN line's own JSON comes after the tag)."""
    j = ln.find("{")
    if j < 0 or j > k:
        return ln
    try:
        todo = [json.loads(ln[j:])]
    except ValueError:
        return ln
    while todo:
        x = todo.pop(0)
        if isinstance(x, str) and TAG in x:
            return x
        todo.extend(x.values() if isinstance(x, dict) else x if isinstance(x, list) else [])
    return ln


def _tagged_lines(f, P):
    for raw in f:
        k = raw.find(TAG)
        if k < 0:
            continue
        P["tagged"] += 1
        for ln in _unwrap(raw, k).splitlines():
            k = ln.find(TAG)
            if k >= 0:
                yield ln[k:].rstrip()


def parse_log(path):
    P = {"begins": {}, "ends": {}, "finals": {}, "chunks": {}, "phase_ends": [], "bad": [], "conflicts": [],
         "tagged": 0}
    with open(path, encoding="utf-8", errors="replace") as f:
        for s in _tagged_lines(f, P):
            try:
                if s.startswith(TAG + "_BEGIN "):
                    m = json.loads(s[len(TAG) + 7:])
                    key = (str(m["run"]), int(m["seq"]))
                    if key in P["begins"] and P["begins"][key] != m:
                        P["conflicts"].append(f"package {key[1]}: two different BEGIN lines")
                    P["begins"][key] = m
                elif s.startswith(TAG + "_END "):
                    m = json.loads(s[len(TAG) + 5:])
                    P["ends"][(str(m["run"]), int(m["seq"]))] = m
                elif s.startswith(TAG + "_FINAL "):
                    m = json.loads(s[len(TAG) + 7:])
                    if str(m["run"]) in P["finals"] and P["finals"][str(m["run"])] != m:
                        P["conflicts"].append("two different final markers")
                    P["finals"][str(m["run"])] = m
                elif s.startswith(TAG + "_PHASE_END "):
                    P["phase_ends"].append(json.loads(s[len(TAG) + 11:]))
                elif s.startswith(TAG + " "):
                    parts = s.split(" ")
                    if len(parts) != 7:
                        P["bad"].append("malformed chunk line (cut or altered)")
                        continue
                    _, run, seq, j, n, h, c = parts
                    key, j = (run, int(seq)), int(j)
                    if _sha(c.encode("ascii", "replace"))[:16] != h:
                        P["bad"].append(f"package {seq} chunk {j}: checksum mismatch (line cut or altered)")
                        continue
                    have = P["chunks"].setdefault(key, {})
                    if j in have and have[j] != c:
                        P["conflicts"].append(f"package {seq} chunk {j}: two different contents")
                    have[j] = c
            except (ValueError, KeyError, TypeError) as e:
                P["bad"].append(f"unreadable {TAG} line ({e.__class__.__name__})")
    return P


def _unpack(blob, meta, out, got, reasons, budget, label):
    want = {f["path"]: f for f in meta.get("files", [])}
    try:
        tf = tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz")
        members = tf.getmembers()
    except (tarfile.TarError, OSError, EOFError) as e:
        reasons.append(f"{label}: unreadable archive ({e.__class__.__name__})")
        return False, budget
    names = [x.name for x in members]
    if len(set(names)) != len(names) or set(names) != set(want):
        reasons.append(f"{label}: archive content differs from its file list")
        return False, budget
    for x in members:
        if not x.isfile() or _unsafe(x.name):
            reasons.append(f"{label}: refused member {x.name!r} (not a regular file or unsafe path)")
            return False, budget
        budget -= x.size
        if budget < 0 or x.size != want[x.name]["bytes"]:
            reasons.append(f"{label}: member {x.name!r} over the unpack limit or of an unexpected size")
            return False, budget
    root = out.resolve()
    for x in members:
        data = tf.extractfile(x).read()
        h = want[x.name]["sha256"]
        if _sha(data) != h:
            reasons.append(f"{label}: {x.name} checksum mismatch")
            return False, budget
        if x.name in got:
            if got[x.name] != h:
                reasons.append(f"{x.name}: two packages carry different contents")
            continue
        dest = root.joinpath(*x.name.split("/"))
        if root not in dest.resolve().parents:
            reasons.append(f"{label}: refused member {x.name!r} (outside the output folder)")
            return False, budget
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        got[x.name] = h
    return True, budget


def decode(log_path, out_dir, run=None, max_unpacked=MAX_UNPACKED, stream=None):
    def say(m):
        print(m, file=stream or sys.stdout)

    out = Path(out_dir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        say(f"refused: {out} exists and is not an empty folder; recover into a NEW folder")
        return 2
    P = parse_log(log_path)
    runs = sorted({k[0] for k in P["begins"]} | set(P["finals"]) | {k[0] for k in P["chunks"]})
    if not runs:
        say(f"no {TAG} line found in {log_path}: nothing recovered; the package is INCOMPLETE" if not P["tagged"] else
            f"{P['tagged']} line(s) of {log_path} mention {TAG} but none could be read (cut, altered or saved in an "
            "unknown wrapper): nothing recovered; the package is INCOMPLETE; save the Job's raw stdout lines")
        return 2
    if run is None and len(runs) > 1:
        say(f"refused: several runs in this log {runs}; pass --run (each Job is reported, none replaces another)")
        return 2
    run = run or runs[0]
    if run not in runs:
        say(f"refused: run {run!r} not in this log {runs}")
        return 2
    reasons = P["bad"] + P["conflicts"]
    final = P["finals"].get(run)
    if final is None:
        reasons.append(f"final marker {TAG}_FINAL missing: the job did not finish its export, or the log is cut")
    listed = {int(p["seq"]): p for p in (final or {}).get("packages", [])}
    interrupted = set((final or {}).get("interrupted_packages", []))
    seqs = sorted(set(listed) | {s for r, s in P["begins"] if r == run} | {s for r, s in P["chunks"] if r == run})
    out.mkdir(parents=True, exist_ok=True)
    got, pkgs, budget = {}, [], max_unpacked
    for seq in seqs:
        m = P["begins"].get((run, seq))
        name = (m or listed.get(seq) or {}).get("pkg")
        rec = {"seq": seq, "pkg": name}
        pkgs.append(rec)
        label = f"package {seq} ({name})"
        if seq in interrupted:
            rec["state"] = "interrupted during the job, its files sent again later: ignored"
            continue
        if m is None:
            rec["state"] = "BEGIN line missing"
            reasons.append(f"{label}: BEGIN line missing")
            continue
        if final is not None and seq not in listed:
            reasons.append(f"{label}: not listed in the final marker")
        elif final is not None and listed[seq].get("sha256") != m.get("sha256"):
            rec["state"] = "checksum differs from the final marker"
            reasons.append(f"{label}: checksum differs from the final marker")
            continue
        cs, n = P["chunks"].get((run, seq), {}), int(m["chunks"])
        miss = [j for j in range(n) if j not in cs]
        if miss:
            rec["state"] = f"{len(miss)} of {n} chunks missing"
            reasons.append(f"{label}: {len(miss)} of {n} chunks missing, first {miss[:5]}: fetch more log lines")
            continue
        if (run, seq) not in P["ends"]:
            reasons.append(f"{label}: END line missing")
        try:
            blob = base64.b64decode("".join(cs[j] for j in range(n)), validate=True)
        except (binascii.Error, ValueError):
            rec["state"] = "base64 invalid"
            reasons.append(f"{label}: invalid base64")
            continue
        if _sha(blob) != m.get("sha256"):
            rec["state"] = "archive checksum mismatch"
            reasons.append(f"{label}: archive checksum mismatch (incomplete or altered)")
            continue
        ok, budget = _unpack(blob, m, out, got, reasons, budget, label)
        rec.update(state="ok" if ok else "refused", files=len(m.get("files", [])), chunks=n)
    missing = [f"{ph}/{f}" for ph in PHASES for f in MANDATORY[ph] if f"{ph}/{f}" not in got]
    missing += [f for f in MANDATORY["job"] if f not in got]
    if missing:
        reasons.append(f"mandatory files missing: {missing}")
    manifests = {"job": _read_json(out / "job_manifest.json")}
    for ph in PHASES:
        manifests[ph] = _read_json(out / ph / "evidence_manifest.json")
    for mf in manifests.values():
        for e in (mf or {}).get("files", []):
            if str(e.get("export", "")).startswith("sent") and got.get(e.get("path")) != e.get("sha256"):
                reasons.append(f"{e.get('path')}: listed as sent by its manifest, not recovered intact")
    recount = {ph: episode_counts(out / ph) for ph in PHASES if (out / ph).is_dir()}
    for ph, mf in manifests.items():
        if ph == "job" or ph not in recount:
            continue
        if mf and mf.get("episode_counts") != recount[ph]:
            reasons.append(f"{ph}: episode counts differ between its manifest and the recovered files")
        for p in count_problems(recount[ph], (mf or {}).get("status") == "complete"):
            reasons.append(f"{ph}: episode counts disagree: {p}")
    not_exported = [e for mf in manifests.values() for e in (mf or {}).get("files", [])
                    if not str(e.get("export", "")).startswith("sent")]
    complete = not reasons
    job_exit = (final or {}).get("job_exit")
    report = {"label": LABEL, "format": FORMAT, "run": run, "log": str(log_path),
              "package": "complete" if complete else "INCOMPLETE", "reasons": reasons, "job_exit": job_exit,
              "phases": (final or {}).get("phases") or {ph: (manifests[ph] or {}).get("status") for ph in PHASES},
              "packages": pkgs, "files_recovered": len(got), "episodes_recounted": recount,
              "not_exported": [{k: e.get(k) for k in ("path", "bytes", "export")} for e in not_exported],
              "raw_signals_sent": False, "raw_signals_note": RAW_SIGNALS_NOTE,
              "rule": ("INCOMPLETE package: no favourable conclusion may be drawn from it and no relaunch is silent "
                       "(one-Job rule: any relaunch is reported together with this one)") if not complete else
                      ("complete package; the run itself ended with job exit %s, reported as such" % job_exit),
              "note": "checksums prove the transport, not the scientific truth of the content; SIMULATION only"}
    (out / "recovery_report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    say(f"{'complete' if complete else 'INCOMPLETE'} package, run {run}: {len(got)} files in {out}, job exit {job_exit}")
    for r in reasons[:20]:
        say(f"  - {r}")
    if not complete:
        say("  " + report["rule"])
    say(f"  report: {out / 'recovery_report.json'}")
    return 3 if not complete else (0 if job_exit == 0 else 1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("emit", help="in the Job: send files (relative to --root) as one package")
    f = sub.add_parser("final", help="in the Job (EXIT trap): send what is unsent, job_manifest.json, final marker")
    for p in (e, f):
        p.add_argument("--root", required=True)
        p.add_argument("--run", required=True)
        p.add_argument("--ledger")
        p.add_argument("--optional-cap-kb", type=int, default=600)
    e.add_argument("--pkg", required=True)
    e.add_argument("files", nargs="+")
    f.add_argument("--job-exit", type=int, required=True)
    f.add_argument("--setup-log")
    d = sub.add_parser("decode", help="on the Mac: rebuild and verify a package from saved Job logs")
    d.add_argument("--log", required=True)
    d.add_argument("--out", required=True, help="a NEW folder")
    d.add_argument("--run")
    a = ap.parse_args(argv)
    if a.cmd == "decode":
        return decode(a.log, a.out, a.run)
    ex = Exporter(a.root, a.run, a.ledger, True, a.optional_cap_kb)
    if a.cmd == "emit":
        ex.emit(a.pkg, [ex.root / x for x in a.files])
        return 0
    finalize_job(ex, a.job_exit, a.setup_log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
