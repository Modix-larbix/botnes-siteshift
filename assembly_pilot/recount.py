#!/usr/bin/env python3
# Copyright 2026 the Botnes SiteShift authors. Licensed under the Apache License, Version 2.0.
"""SIMULATION ONLY. Independent recount of the AssembleNIST pilot results (task R16, MuJoCo).

Standard library only. It reads the JSON files exported by the Job and recounts, per condition, the
environment success, the safety faults, the part inserted at report time, the paired transitions against
the reference, the robot report's main claims and their audit, the "part held" statements and their
disagreements, and the keypoint distances at report time of the successes not inserted at report time.

Usage:
    python3 recount.py results/                  # prints the recount as JSON
    python3 recount.py results/ --out recount_output.json

Exit code: 0 when every cross-check agrees and every recounted number equals its expected value in
STATED below; 1 when anything disagrees (the output says what); 2 when a file is missing or unreadable.
STATED holds, written into this file, the numbers that README.md states for the batch published on
October 27, 2026 (the Job of October 26, UTC). The script never reads README.md: if another batch is
added, STATED and README.md must be updated together.

What it cannot do: the raw 25 Hz signals were not exported, so the audit verdicts can be recounted and
cross-checked here, not recomputed from the signals.
"""
import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

CONDITIONS = ("REF", "V1_OFFSET", "V2_OILY")
VARIANTS = ("V1_OFFSET", "V2_OILY")
CLAIMS = ("still to check", "not inserted", "inserted", "safety fault detected")
SCORED = ("inserted", "not inserted")
FILES = ("results.json", "audit.json", "robot_reports.jsonl", "summary.json", "plan.json",
         "evidence_manifest.json")
PAIR_KEYS = ("seed", "trial", "bank_index", "start_family", "start_level")

# Expected values: the numbers README.md states for this batch, copied here (this script never reads README.md).
# Episodes out of 320 per condition.
STATED = {
    "per_condition": {
        "REF": {"episodes": 320, "env_success": 317, "env_safety_failures": 3, "inserted_at_report": 192,
                "env_success_not_inserted_at_report": 125},
        "V1_OFFSET": {"episodes": 320, "env_success": 315, "env_safety_failures": 5, "inserted_at_report": 197,
                      "env_success_not_inserted_at_report": 118},
        "V2_OILY": {"episodes": 320, "env_success": 159, "env_safety_failures": 10, "inserted_at_report": 140,
                    "env_success_not_inserted_at_report": 19},
    },
    "paired_vs_REF": {
        "V1_OFFSET": {"REF_success_variation_failure": 3, "REF_failure_variation_success": 1},
        "V2_OILY": {"REF_success_variation_failure": 158, "REF_failure_variation_success": 0},
    },
    "all_conditions": {
        "episodes": 960,
        "main_claims": {"still to check": 928, "not inserted": 14, "inserted": 0, "safety fault detected": 18},
        "scored_main_claims_confirmed_by_audit": 14,
        "scored_main_claims_contradicted_by_audit": 0,
        "part_held_disagreements": 32,
        "part_held_false_positives": 30,
        "part_held_false_negatives": 2,
        "part_held_said_held": 71,
        "part_held_said_not_held": 889,
        "env_success_not_inserted_at_report": 262,
        "env_success_not_inserted_at_report_at_most_2_0_mm": 259,
        "safety_fault_claims_env_flag_agrees": 18,
        "reports_at_15_s": 942,
    },
    # exact McNemar p of the paired transitions, as README.md writes them (3 significant digits, or a bound)
    "mcnemar_p": {"V1_OFFSET": {"equals": "0.625"}, "V2_OILY": {"below": 1e-40}},
    # "part held" per condition: how often the report said "held", how often that was wrong, all disagreements
    "part_held_per_condition": {
        "REF": {"said_held": 8, "false_positives": 3, "disagreements": 4},
        "V1_OFFSET": {"said_held": 10, "false_positives": 5, "disagreements": 5},
        "V2_OILY": {"said_held": 53, "false_positives": 22, "disagreements": 23},
    },
    # keypoint distance (truth.d_asm_mm) at report time of the successes not inserted at report time; a value
    # given as text is compared at the number of decimals README.md writes it with
    "d_asm_mm_success_not_inserted": {
        "REF": {"n": 125, "at_most_2_0_mm": 125, "min": "1.0", "max": "1.9", "median": "1.2"},
        "V1_OFFSET": {"n": 118, "at_most_2_0_mm": 117, "min": "1.0", "max": "2.01", "median": "1.3"},
        "V2_OILY": {"n": 19, "at_most_2_0_mm": 17, "median": "1.2", "above_2_0_mm": ["2.03", "38.2"]},
    },
    "inserted_at_report_max_d_asm_mm": "1.0",
}


def as_stated(value, stated):
    """The value written with as many decimals as the stated text, e.g. 1.875 against "1.9" gives "1.9"."""
    decimals = len(stated.split(".", 1)[1]) if "." in stated else 0
    return f"{value:.{decimals}f}"


class Unreadable(Exception):
    pass


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise Unreadable(f"{path.name}: {e.__class__.__name__}")


def load_jsonl(path):
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        return [json.loads(x) for x in lines if x.strip()]
    except (OSError, ValueError) as e:
        raise Unreadable(f"{path.name}: {e.__class__.__name__}")


def index(records, what, problems):
    out = {}
    for r in records:
        key = (r.get("condition"), r.get("episode"))
        if key in out:
            problems.append(f"{what}: duplicate episode {key}")
        out[key] = r
    return out


def mcnemar_exact_p(b, c):
    """Two-sided exact McNemar test on the discordant pairs (binomial, p = 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def recount(folder):
    problems = []
    paths = {name: folder / name for name in FILES}
    missing = [n for n, p in paths.items() if not p.is_file()]
    if missing:
        raise Unreadable("missing file(s): " + ", ".join(missing))
    hashes = {n: sha256(p) for n, p in paths.items()}

    results = load_json(paths["results.json"])
    audit = load_json(paths["audit.json"])
    reports = load_jsonl(paths["robot_reports.jsonl"])
    summary = load_json(paths["summary.json"])
    plan = load_json(paths["plan.json"])
    manifest = load_json(paths["evidence_manifest.json"])

    # 1. The files are the ones the Job exported: compare with the sha256 its manifest recorded.
    recorded = {f["path"].split("/", 1)[-1]: f["sha256"] for f in manifest.get("files", [])
                if isinstance(f, dict) and "path" in f and "sha256" in f}
    integrity = {}
    for name in FILES:
        if name == "evidence_manifest.json":
            integrity[name] = "the manifest cannot list its own sha256"
            continue
        exp = recorded.get(name)
        ok = exp == hashes[name]
        integrity[name] = "matches the Job's evidence_manifest.json" if ok else "DIFFERS from the manifest"
        if not ok:
            problems.append(f"{name}: sha256 differs from evidence_manifest.json")

    eps = index(results.get("episodes", []), "results.json", problems)
    aud = index(audit.get("per_episode", []), "audit.json", problems)
    rep = index(reports, "robot_reports.jsonl", problems)
    if not (set(eps) == set(aud) == set(rep)):
        problems.append("results.json, audit.json and robot_reports.jsonl do not hold the same episodes")

    per = {}
    for c in CONDITIONS:
        keys = sorted(k for k in eps if k[0] == c)
        claims = {k: 0 for k in CLAIMS}
        other_claims = {}
        d = {"episodes": len(keys), "env_success": 0, "env_safety_failures": 0, "inserted_at_report": 0,
             "env_success_not_inserted_at_report": 0, "inserted_at_report_without_env_success": 0}
        scored = {"n": 0, "confirmed_by_audit": 0, "contradicted_by_audit": 0,
                  "recomputed_from_audit_truth_agrees": 0}
        held = {"n": 0, "agree": 0, "false_positives": 0, "false_negatives": 0, "said_held": 0, "said_not_held": 0}
        d_not_ins, d_ins = [], []
        at_15 = 0
        safety = {"n": 0, "env_flag_agrees": 0, "env_safety_failure_true": 0}
        for k in keys:
            e, a, r = eps[k], aud.get(k, {}), rep.get(k, {})
            truth = a.get("truth", {})
            succ = e.get("env_success") is True
            ins = e.get("part_inserted_at_report") is True
            d["env_success"] += succ
            d["env_safety_failures"] += e.get("env_safety_failure") is True
            d["inserted_at_report"] += ins
            d["env_success_not_inserted_at_report"] += succ and not ins
            d["inserted_at_report_without_env_success"] += ins and not succ
            if truth.get("part_inserted") is not ins:
                problems.append(f"{k}: part_inserted_at_report differs between results.json and audit.json")
            at_15 += r.get("report_s") == 15.0
            dist = truth.get("d_asm_mm")
            if isinstance(dist, (int, float)):
                if succ and not ins:
                    d_not_ins.append(float(dist))
                if ins:
                    d_ins.append(float(dist))
            elif succ or ins:
                problems.append(f"{k}: truth.d_asm_mm missing in audit.json")

            claim = r.get("claim")
            if claim in claims:
                claims[claim] += 1
            else:
                other_claims[str(claim)] = other_claims.get(str(claim), 0) + 1
            if claim != e.get("robot_claim") or claim != a.get("audited", {}).get("main_claim", {}).get("claim"):
                problems.append(f"{k}: main claim differs between the three files")

            if claim in SCORED:
                scored["n"] += 1
                verdict = a.get("audited", {}).get("main_claim", {}).get("true")
                scored["confirmed_by_audit"] += verdict is True
                scored["contradicted_by_audit"] += verdict is False
                recomputed = (claim == "inserted") == (truth.get("part_inserted") is True)
                scored["recomputed_from_audit_truth_agrees"] += recomputed == (verdict is True)
                if e.get("robot_claim_true") is not verdict:
                    problems.append(f"{k}: robot_claim_true differs from the audit verdict")
            if claim == "safety fault detected":
                safety["n"] += 1
                safety["env_flag_agrees"] += e.get("safety_fault_env_flag_agrees") is True
                safety["env_safety_failure_true"] += e.get("env_safety_failure") is True

            said = r.get("part_held_claim", {}).get("value")
            real = truth.get("part_held_by_both_fingers")
            if isinstance(said, bool) and isinstance(real, bool):
                held["n"] += 1
                held["agree"] += said == real
                held["said_held"] += said
                held["said_not_held"] += not said
                held["false_positives"] += said and not real
                held["false_negatives"] += real and not said
                if a.get("audited", {}).get("part_held", {}).get("true") is not (said == real):
                    problems.append(f"{k}: part held verdict in audit.json differs from claim vs truth")
                if e.get("holding_claim_true") is not (said == real):
                    problems.append(f"{k}: holding_claim_true in results.json differs from claim vs truth")
            else:
                problems.append(f"{k}: part held claim or truth missing")
        held["disagreements"] = held["false_positives"] + held["false_negatives"]
        d_not_ins.sort()
        d["env_success_not_inserted_at_report_d_asm_mm"] = {
            "n": len(d_not_ins), "min": d_not_ins[0] if d_not_ins else None,
            "median": statistics.median(d_not_ins) if d_not_ins else None,
            "max": d_not_ins[-1] if d_not_ins else None,
            "at_most_2_0_mm": sum(x <= 2.0 for x in d_not_ins),
            "above_2_0_mm": [x for x in d_not_ins if x > 2.0]}
        d["inserted_at_report_max_d_asm_mm"] = max(d_ins) if d_ins else None
        d["reports_at_15_s"] = at_15
        d["main_claims"] = claims
        if other_claims:
            d["unexpected_claims"] = other_claims
            problems.append(f"{c}: unexpected main claims {other_claims}")
        d["scored_main_claims"] = scored
        d["safety_fault_flags_consistency_check_not_scored"] = safety
        d["part_held"] = held
        per[c] = d

    # Paired transitions: same episode index, same start and same policy noise as REF.
    paired = {}
    for v in VARIANTS:
        b = c_ = mismatched_keys = 0
        for (cond, ep), e in eps.items():
            if cond != "REF":
                continue
            other = eps.get((v, ep))
            if other is None:
                problems.append(f"{v}: episode {ep} missing for pairing")
                continue
            if any(e.get(x) != other.get(x) for x in PAIR_KEYS):
                mismatched_keys += 1
            s0, s1 = e.get("env_success") is True, other.get("env_success") is True
            b += s0 and not s1
            c_ += s1 and not s0
        if mismatched_keys:
            problems.append(f"{v}: {mismatched_keys} pairs with a different start or seed")
        paired[v] = {"REF_success_variation_failure": b, "REF_failure_variation_success": c_,
                     "pairs_with_different_start_or_seed": mismatched_keys,
                     "mcnemar_exact_p": float(f"{mcnemar_exact_p(b, c_):.3g}")}

    total = {"episodes": sum(per[c]["episodes"] for c in CONDITIONS),
             "main_claims": {k: sum(per[c]["main_claims"][k] for c in CONDITIONS) for k in CLAIMS},
             "scored_main_claims_confirmed_by_audit":
                 sum(per[c]["scored_main_claims"]["confirmed_by_audit"] for c in CONDITIONS),
             "scored_main_claims_contradicted_by_audit":
                 sum(per[c]["scored_main_claims"]["contradicted_by_audit"] for c in CONDITIONS),
             "part_held_disagreements": sum(per[c]["part_held"]["disagreements"] for c in CONDITIONS),
             "part_held_false_positives": sum(per[c]["part_held"]["false_positives"] for c in CONDITIONS),
             "part_held_false_negatives": sum(per[c]["part_held"]["false_negatives"] for c in CONDITIONS),
             "part_held_said_held": sum(per[c]["part_held"]["said_held"] for c in CONDITIONS),
             "part_held_said_not_held": sum(per[c]["part_held"]["said_not_held"] for c in CONDITIONS),
             "env_success_not_inserted_at_report":
                 sum(per[c]["env_success_not_inserted_at_report"] for c in CONDITIONS),
             "env_success_not_inserted_at_report_at_most_2_0_mm":
                 sum(per[c]["env_success_not_inserted_at_report_d_asm_mm"]["at_most_2_0_mm"] for c in CONDITIONS),
             "safety_fault_claims_env_flag_agrees":
                 sum(per[c]["safety_fault_flags_consistency_check_not_scored"]["env_flag_agrees"] for c in CONDITIONS),
             "reports_at_15_s": sum(per[c]["reports_at_15_s"] for c in CONDITIONS)}

    # 2. Cross-check with the Job's own summaries (summary.json, results.json "summary", audit.json).
    job_summaries = {"summary.json": summary.get("per_condition", {}),
                     "results.json summary": results.get("summary", {}),
                     "audit.json per_condition": audit.get("per_condition", {})}
    vs_job = {}
    for src, s in job_summaries.items():
        diffs = []
        for c in CONDITIONS:
            sc = s.get(c, {})
            pairs = [("episodes", sc.get("n"), per[c]["episodes"]),
                     ("env_success", sc.get("env_success"), per[c]["env_success"]),
                     ("env_safety_failures", sc.get("env_safety_failures"), per[c]["env_safety_failures"]),
                     ("part_held agreements", sc.get("audited", {}).get("part_held", {}).get("true"),
                      per[c]["part_held"]["agree"]),
                     ("scored main claims confirmed", sc.get("audited", {}).get("main_claim", {}).get("true"),
                      per[c]["scored_main_claims"]["confirmed_by_audit"])]
            job_claims = sc.get("robot_claims", {})
            for k in CLAIMS:
                pairs.append((f"claim {k}", job_claims.get(k, 0), per[c]["main_claims"][k]))
            for name, job_value, ours in pairs:
                if job_value != ours:
                    diffs.append(f"{c} {name}: job {job_value}, recount {ours}")
        vs_job[src] = diffs or "agrees"
        problems.extend(f"{src}: {x}" for x in diffs)
    for v in VARIANTS:
        js = summary.get("paired_vs_REF", {}).get(v, {})
        for k in ("REF_success_variation_failure", "REF_failure_variation_success"):
            if js.get(k) != paired[v][k]:
                problems.append(f"summary.json paired_vs_REF {v} {k}: job {js.get(k)}, recount {paired[v][k]}")
    planned = plan.get("episodes_per_condition")
    if planned is not None and any(per[c]["episodes"] != planned for c in CONDITIONS):
        problems.append(f"plan.json planned {planned} episodes per condition, recount differs")

    # 3. Compare with STATED, the numbers README.md states (README.md itself is not read).
    vs_readme = []
    for c in CONDITIONS:
        for k, v in STATED["per_condition"][c].items():
            if per[c][k] != v:
                vs_readme.append(f"{c} {k}: stated {v}, recount {per[c][k]}")
    for v in VARIANTS:
        for k, x in STATED["paired_vs_REF"][v].items():
            if paired[v][k] != x:
                vs_readme.append(f"paired {v} {k}: stated {x}, recount {paired[v][k]}")
    for k, x in STATED["all_conditions"].items():
        if total[k] != x:
            vs_readme.append(f"all conditions {k}: stated {x}, recount {total[k]}")
    for c, st in STATED["part_held_per_condition"].items():
        for k, x in st.items():
            if per[c]["part_held"][k] != x:
                vs_readme.append(f"{c} part held {k}: stated {x}, recount {per[c]['part_held'][k]}")
    for c, st in STATED["d_asm_mm_success_not_inserted"].items():
        got = per[c]["env_success_not_inserted_at_report_d_asm_mm"]
        for k, x in st.items():
            if k == "above_2_0_mm":
                ours = [as_stated(v, s_) for v, s_ in zip(got[k], x)] if len(got[k]) == len(x) else got[k]
            elif isinstance(x, str):
                ours = as_stated(got[k], x) if got[k] is not None else None
            else:
                ours = got[k]
            if ours != x:
                vs_readme.append(f"{c} d_asm_mm {k}: stated {x}, recount {ours}")
    for v, st in STATED["mcnemar_p"].items():
        p_ = paired[v]["mcnemar_exact_p"]
        if "equals" in st and f"{p_:.3g}" != st["equals"]:  # the recount keeps p at 3 significant digits
            vs_readme.append(f"paired {v} McNemar p: stated {st['equals']}, recount {p_}")
        if "below" in st and not p_ < st["below"]:
            vs_readme.append(f"paired {v} McNemar p: stated below {st['below']}, recount {p_}")
    ins_max = max(per[c]["inserted_at_report_max_d_asm_mm"] for c in CONDITIONS)
    if ins_max > float(STATED["inserted_at_report_max_d_asm_mm"]):
        vs_readme.append(f"inserted at report time: a keypoint distance of {ins_max} mm, above the stated "
                         f"{STATED['inserted_at_report_max_d_asm_mm']} mm")

    timings = results.get("timings", {})
    sim_s = round(sum(float(timings.get(c, {}).get("rollout_wall_s", 0.0)) for c in CONDITIONS), 1)

    out = {
        "label": "SIMULATION (MuJoCo), recount of exported JSON, standard library only",
        "task": results.get("task"), "part": results.get("part"),
        "upstream": results.get("upstream"),
        "measures": {
            "env_success": "the environment's own success predicate, latched at its first occurrence "
                           "(keypoints within 1 mm, no simultaneous safety failure, 15 s limit)",
            "inserted_at_report": "part keypoints within 1 mm in the simulator at the robot report instant",
            "note": "two different measures, never merged",
        },
        "files_sha256": hashes,
        "files_vs_job_manifest": integrity,
        "per_condition": per,
        "paired_vs_REF": paired,
        "all_conditions": total,
        "episode_simulation_wall_s": {"per_condition": {c: timings.get(c, {}).get("rollout_wall_s")
                                                        for c in CONDITIONS}, "sum": sim_s},
        "cross_check_vs_job_summaries": vs_job,
        "vs_numbers_stated_in_README": vs_readme or "all match",
        "problems": problems,
        "limits": "Raw 25 Hz signals were not exported: the audit verdicts are recounted and cross-checked, "
                  "not recomputed from the signals.",
    }
    return out, (0 if not problems and not vs_readme else 1)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Recount the AssembleNIST pilot results (simulation only).")
    ap.add_argument("results", help="folder holding results.json, audit.json, robot_reports.jsonl, ...")
    ap.add_argument("--out", help="also write the recount to this JSON file")
    a = ap.parse_args(argv)
    try:
        out, code = recount(Path(a.results))
    except Unreadable as e:
        print(f"recount: {e}", file=sys.stderr)
        return 2
    text = json.dumps(out, indent=1, ensure_ascii=True) + "\n"
    sys.stdout.write(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    return code


if __name__ == "__main__":
    sys.exit(main())
