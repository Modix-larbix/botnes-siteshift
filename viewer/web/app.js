"use strict";
const $ = (id) => document.getElementById(id);
const terminal = new Set(["completed", "different", "failed", "cancelled", "timed_out"]);
let current = null;
let latestFrame = 0;
let pollTimer = null;
let playbackTimer = null;
let polling = false;
let pollFailures = 0;
let nextPollAt = 0;

async function api(path, body) {
  const options = body === undefined ? {} : {
    method: "POST", headers: {"Content-Type": "application/json", "X-SiteShift": "local-run"},
    body: JSON.stringify(body),
  };
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "The local service did not respond. Reload this page and try again.");
  return data;
}

function say(text, error = false) {
  $("status").textContent = text;
  $("status").classList.toggle("error", error);
}

function setBusy(busy) {
  $("start").disabled = busy;
  $("stop").hidden = !busy;
  $("scenarios").disabled = busy;
}

function frameUrl(number) {
  return `/api/runs/${current}/frames/${String(number).padStart(5, "0")}.jpg`;
}

function showFrame(number) {
  if (!current || number < 1) return;
  $("robot-view").src = frameUrl(number);
  $("scrubber").value = number;
  $("frame-label").textContent = `${number} / ${latestFrame} frames`;
}

function stopPlayback() {
  clearInterval(playbackTimer);
  playbackTimer = null;
  $("playback").textContent = "Replay movements";
}

function renderProgress(data) {
  const progress = data.progress || {};
  const seconds = Math.floor(data.elapsed_s);
  $("clock").textContent = `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")} elapsed`;
  $("activity").hidden = false;
  $("sim-time").textContent = `${Number(progress.sim_s || 0).toFixed(1)} s`;
  $("frame-count").textContent = progress.frame || 0;
  $("current-action").textContent = progress.skill ? friendlySkill(progress.skill) : "Checking the published code and building your scene…";
  const calls = progress.calls || [];
  $("events").replaceChildren(...calls.slice(-8).map((call) => {
    const element = document.createElement("li");
    element.textContent = `${friendlySkill(call.skill)}${call.ok ? " ✓" : " · " + (call.reason || "not completed")}`;
    return element;
  }));
  if (progress.frame && progress.frame !== latestFrame) {
    latestFrame = progress.frame;
    $("scrubber").max = latestFrame;
    $("preparing").hidden = true;
    $("empty-view").hidden = true;
    $("robot-view").alt = "Newly rendered images from this local simulation: overview, robot camera and wrist camera.";
    if (!playbackTimer) showFrame(latestFrame);
  }
}

function friendlySkill(skill) {
  const names = {"open_drawer": "Opening a drawer", "close_drawer": "Closing a drawer", "look_pose": "Inspecting the scene", "calibrate": "Calibrating the robot", "move_to": "Moving the arm", "look": "Inspecting the scene", "observe": "Observing the scene", "pick": "Picking the sponge", "pick_sponge": "Picking the sponge", "scrub": "Scrubbing the bowl", "scrub_bowl": "Scrubbing the bowl", "help": "Asking for help", "home": "Moving the arm home", "move": "Moving the arm"};
  return names[skill] || skill.replaceAll("_", " ");
}

async function update() {
  if (!current || polling || Date.now() < nextPollAt) return;
  const run = current;
  polling = true;
  try {
    const data = await api(`/api/runs/${run}`);
    if (current !== run) return;
    pollFailures = 0;
    nextPollAt = 0;
    renderProgress(data);
    if (terminal.has(data.status)) {
      clearInterval(pollTimer);
      setBusy(false);
      $("preparing").hidden = true;
      $("live-dot").className = "status-dot idle";
      $("replay").hidden = !latestFrame;
      $("analyze").disabled = !data.analysis_available;
      $("start").innerHTML = '<span aria-hidden="true">▶</span> Run again';
      if (data.status === "completed") {
        $("viewer-label").textContent = "Local run completed";
        say("Simulation complete. Click Replay movements for smoother playback, or Run analysis to check the result.");
      } else if (data.status === "cancelled") {
        $("viewer-label").textContent = "Run stopped";
        say("Run stopped. You can replay its saved images or start again.");
      } else {
        $("viewer-label").textContent = data.status === "different" ? "Run completed with differences" : "Run interrupted";
        say(data.error || "This run did not finish. You can try again.", true);
        $("run-log").hidden = false;
        const log = await (await fetch(`/api/runs/${run}/log`)).text();
        if (current === run) $("log-content").textContent = log;
      }
    } else {
      $("viewer-label").textContent = latestFrame ? "Running locally" : "Preparing your run";
      $("live-dot").className = "status-dot running";
      say("A new simulation is computing on your Mac. You can stop it at any time.");
    }
  } catch (error) {
    if (current === run) {
      pollFailures += 1;
      nextPollAt = Date.now() + Math.min(5000, 500 * 2 ** Math.min(pollFailures, 4));
      setBusy(true);
      say("Connection interrupted. Reconnecting; the local run may still be active. Keep the launcher window open.", true);
    }
  } finally { polling = false; }
}

$("start").addEventListener("click", async () => {
  stopPlayback();
  clearInterval(pollTimer);
  pollFailures = 0;
  nextPollAt = 0;
  setBusy(true);
  $("analyze").disabled = true;
  $("analysis").hidden = true;
  $("run-log").hidden = true;
  $("replay").hidden = true;
  $("preparing").hidden = false;
  $("empty-view").hidden = true;
  $("viewer-label").textContent = "Preparing your run";
  say("Starting a new local simulation…");
  try {
    const situation = document.querySelector('input[name="situation"]:checked').value;
    const data = await api("/api/runs", {situation});
    current = data.id;
    latestFrame = 0;
    sessionStorage.setItem("siteshift-run", current);
    await update();
    pollTimer = setInterval(update, 350);
  } catch (error) {
    setBusy(false);
    $("preparing").hidden = true;
    say(error.message, true);
  }
});

$("stop").addEventListener("click", async () => {
  $("stop").disabled = true;
  try { await api(`/api/runs/${current}/cancel`, {}); await update(); }
  catch (error) { say(error.message, true); }
  finally { $("stop").disabled = false; }
});

$("playback").addEventListener("click", () => {
  if (playbackTimer) { stopPlayback(); return; }
  let number = 1;
  showFrame(number);
  $("playback").textContent = "Pause replay";
  playbackTimer = setInterval(() => {
    number += 1;
    if (number > latestFrame) { stopPlayback(); return; }
    showFrame(number);
  }, 100);
});
$("scrubber").addEventListener("input", () => { stopPlayback(); showFrame(Number($("scrubber").value)); });

const criterionNames = {
  "target_contact": "Contact with the target", "sponge_held": "Sponge held", "coverage": "Scrubbing coverage",
  "scrub_motion": "Scrubbing movement", "target_held_during_scrub": "Bowl stays still during scrubbing",
  "grip_during_scrub": "Grip during scrubbing", "no_drop": "No dropped sponge",
  "no_other_dish_scrubbed": "No other dish scrubbed", "target_force_within_limit": "Force within the task threshold",
  "target_upright_and_in_place": "Bowl upright and in place",
};
$("analyze").addEventListener("click", async () => {
  const analyzedRun = current;
  $("analyze").disabled = true;
  say("Reading the evaluator records from your run…");
  try {
    const report = await api(`/api/runs/${analyzedRun}/analyze`, {});
    // A new run may have started while this response was being read.
    if (current !== analyzedRun) return;
    $("analysis").hidden = false;
    const asking = report.claim?.startsWith("help_requested");
    $("outcome").textContent = asking ? "The robot asked for help" : report.outcome === "success" ? "The bowl-scrubbing task met the evaluator criteria" : "The evaluator did not mark the task as successful";
    $("result-description").textContent = asking
      ? report.situation === "masked" && report.comparison.equal
        ? "The sponge was present, but the towel left part of the search unresolved. The robot stopped and asked for help. It did not claim that the sponge was absent."
        : "The robot stopped and requested help. Check the evaluator criteria and any differences from the reference below for this run."
      : "The evaluator checked the physical episode after it finished. This is a simulator result, not evidence of real-world performance.";
    const pairs = [
      ["Run", analyzedRun.slice(0, 10)],
      ["Evaluator outcome", report.outcome.replaceAll("_", " ")],
      ["Compared facts", report.comparison.equal ? "Match the recorded reference" : "Differ from the recorded reference"],
      ["Next-step choice", report.recorded_answer_used ? "Recorded Nemotron answer, matching request text" : "Rule: one valid option"],
      ["Live model call", report.model_called ? "Yes" : "None"],
      ["Source files checked", String(report.files_verified)],
      ["Computation time", `${report.wall_s} seconds`],
    ];
    $("run-facts").replaceChildren(...pairs.flatMap(([term, value]) => {
      const dt = document.createElement("dt"), dd = document.createElement("dd");
      dt.textContent = term; dd.textContent = value; return [dt, dd];
    }));
    $("evaluator-detail").textContent = report.detail || "";
    $("evaluator-detail").hidden = !report.detail;
    $("violations-box").hidden = !report.violations.length;
    $("violations").replaceChildren(...report.violations.map((violation) => {
      const item = document.createElement("li");
      item.textContent = typeof violation === "string" ? violation : JSON.stringify(violation);
      return item;
    }));
    $("criteria").replaceChildren(...Object.entries(report.criteria).map(([name, met]) => {
      const item = document.createElement("li"), mark = document.createElement("span"), text = document.createElement("span");
      mark.className = met ? "pass" : "fail"; mark.textContent = met ? "✓" : "○";
      text.textContent = `${criterionNames[name] || name.replaceAll("_", " ")}: ${met ? "met" : "not met"}`;
      item.append(mark, text); return item;
    }));
    $("download-report").href = `/api/runs/${analyzedRun}/report`;
    $("differences-box").hidden = report.comparison.equal;
    $("differences").textContent = JSON.stringify(report.comparison.differences, null, 2);
    say("Analysis complete. The report below comes from your new run.");
    $("analysis").scrollIntoView({behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth", block: "start"});
  } catch (error) { if (current === analyzedRun) say(error.message, true); }
  finally { if (current === analyzedRun) $("analyze").disabled = false; }
});

// Refreshing the page does not silently start another run.
(async () => {
  const remembered = sessionStorage.getItem("siteshift-run");
  if (!remembered) return;
  try {
    const data = await api(`/api/runs/${remembered}`);
    current = remembered;
    const selected = document.querySelector(`input[value="${data.situation}"]`);
    if (selected) selected.checked = true;
    setBusy(!terminal.has(data.status));
    await update();
    if (!terminal.has(data.status)) pollTimer = setInterval(update, 350);
  } catch { sessionStorage.removeItem("siteshift-run"); }
})();
