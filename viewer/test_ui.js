// Offline regression checks of the viewer's async state. No browser or robot is started.
// Run with: node --test viewer/test_ui.js
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const source = fs.readFileSync(path.join(__dirname, "web", "app.js"), "utf8");
const oldRun = "a".repeat(32);
const newRun = "b".repeat(32);

function response(data) {
  return Promise.resolve({ok: true, json: async () => data});
}

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return {promise, resolve, reject};
}

function report(overrides = {}) {
  return {
    situation: "moved", claim: "task_done", outcome: "success", detail: "", violations: [],
    comparison: {equal: true, differences: []}, recorded_answer_used: true,
    same_request_text: true, who_chose: "nemotron", option: "scrub_bowl",
    model_called: false, files_verified: 78, wall_s: 30,
    criteria: {target_contact: true}, ...overrides,
  };
}

function harness(fetch, {remembered = null} = {}) {
  const elements = new Map();
  function element(name) {
    return {
      name, events: {}, textContent: "", disabled: false, hidden: false,
      classList: {toggle() {}},
      addEventListener(type, handler) { this.events[type] = handler; },
      replaceChildren(...children) { this.children = children; },
      append(...children) { this.children = children; },
      scrollIntoView() {},
    };
  }
  const document = {
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, element(id));
      return elements.get(id);
    },
    querySelector() { return {value: "moved"}; },
    createElement(tag) { return element(tag); },
  };
  const context = vm.createContext({
    document, fetch, console,
    sessionStorage: {getItem() { return remembered; }, setItem() {}, removeItem() {}},
    setInterval() { return 1; }, clearInterval() {},
    matchMedia() { return {matches: true}; },
  });
  const initialization = vm.runInContext(source, context);
  return {
    initialization,
    element: (id) => document.getElementById(id),
    run: (code) => vm.runInContext(code, context),
    click: (id) => document.getElementById(id).events.click(),
  };
}

test("an old analysis cannot appear under a newly started run's ID", async () => {
  const pending = deferred();
  const app = harness((url) => {
    if (url.endsWith("/analyze")) return pending.promise;
    if (url === "/api/runs") return response({id: newRun});
    return response({status: "running", elapsed_s: 0, progress: {}});
  });
  app.run(`current = '${oldRun}'`);
  const analysis = app.click("analyze");
  await app.click("start");
  pending.resolve({ok: true, json: async () => report()});
  await analysis;
  assert.equal(app.element("analysis").hidden, true);
  assert.equal(app.element("analyze").disabled, true);
  assert.notEqual(app.element("download-report").href, `/api/runs/${newRun}/report`);
});

test("an old analysis error cannot overwrite a new run's status", async () => {
  const pending = deferred();
  const app = harness((url) => {
    if (url.endsWith("/analyze")) return pending.promise;
    if (url === "/api/runs") return response({id: newRun});
    return response({status: "running", elapsed_s: 0, progress: {}});
  });
  app.run(`current = '${oldRun}'`);
  const analysis = app.click("analyze");
  await app.click("start");
  const status = app.element("status").textContent;
  pending.reject(new Error("stale analysis failure"));
  await analysis;
  assert.equal(app.element("status").textContent, status);
  assert.equal(app.element("analyze").disabled, true);
});

test("a late status response cannot complete a newer run", async () => {
  const pending = deferred();
  const app = harness(() => pending.promise);
  app.run(`current = '${oldRun}'`);
  const update = app.run("update()");
  app.run(`current = '${newRun}'; setBusy(true)`);
  app.element("viewer-label").textContent = "Preparing the new run";
  pending.resolve({ok: true, json: async () => ({status: "completed", elapsed_s: 30, progress: {}, analysis_available: true})});
  await update;
  assert.equal(app.element("viewer-label").textContent, "Preparing the new run");
  assert.equal(app.element("start").disabled, true);
  assert.equal(app.element("stop").hidden, false);
});

test("a help request in the moved case does not invent a towel", async () => {
  const app = harness(() => response(report({
    claim: "help_requested:regions_still_to_check_1", outcome: "search_unsuccessful",
    situation: "moved", comparison: {equal: false, differences: ["claim"]},
  })));
  app.run(`current = '${oldRun}'`);
  await app.click("analyze");
  assert.doesNotMatch(app.element("result-description").textContent, /towel/i);
  assert.equal(app.element("download-report").href, `/api/runs/${oldRun}/report`);
});

test("the matching masked reference can explain the towel", async () => {
  const app = harness(() => response(report({
    claim: "help_requested:regions_still_to_check_1", outcome: "search_unsuccessful",
    situation: "masked", comparison: {equal: true, differences: []},
  })));
  app.run(`current = '${oldRun}'`);
  await app.click("analyze");
  assert.match(app.element("result-description").textContent, /towel/i);
  assert.equal(app.element("download-report").href, `/api/runs/${oldRun}/report`);
});

test("a transient status error keeps Stop available and polling can recover", async () => {
  let requests = 0;
  const app = harness(() => {
    requests += 1;
    if (requests === 1) return Promise.reject(new Error("temporary connection failure"));
    return response({status: "completed", elapsed_s: 30, progress: {}, analysis_available: true});
  });
  app.run(`current = '${oldRun}'; setBusy(true)`);
  await app.run("update()");
  assert.equal(app.element("start").disabled, true);
  assert.equal(app.element("stop").hidden, false);
  assert.match(app.element("status").textContent, /reconnect/i);
  assert.equal(app.run("current"), oldRun);
  // Advance beyond the retry delay without a real sleep.
  app.run("nextPollAt = 0");
  await app.run("update()");
  assert.equal(requests, 2);
  assert.equal(app.element("start").disabled, false);
  assert.equal(app.element("stop").hidden, true);
  assert.equal(app.element("analyze").disabled, false);
});

test("a physical failure displays its violation even when all task criteria passed", async () => {
  const app = harness(() => response(report({
    outcome: "physical_failure", detail: "arm_collision",
    violations: [{type: "arm_collision", link_force_peak_n: 25.5}],
    criteria: {target_contact: true, scrub_motion: true},
    comparison: {equal: false, differences: ["violations"]},
  })));
  app.run(`current = '${oldRun}'`);
  await app.click("analyze");
  assert.equal(app.element("violations-box").hidden, false);
  assert.equal(app.element("evaluator-detail").textContent, "arm_collision");
  assert.match(JSON.stringify(app.element("violations").children), /arm.collision/i);
  assert.doesNotMatch(app.element("outcome").textContent, /did not meet every criterion/i);
});

for (const status of ["failed", "different", "timed_out"]) {
  test(`a ${status} run remains terminal when loading its log fails`, async () => {
    const failure = `Recorded terminal status: ${status}`;
    const app = harness((url) => {
      if (url.endsWith("/log")) return Promise.reject(new Error("log connection failure"));
      return response({status, elapsed_s: 30, progress: {},
        error: failure, analysis_available: status === "different"});
    });
    app.run(`current = '${oldRun}'; setBusy(true)`);
    await app.run("update()");
    assert.equal(app.element("start").disabled, false);
    assert.equal(app.element("stop").hidden, true);
    assert.equal(app.element("status").textContent, failure);
    assert.doesNotMatch(app.element("status").textContent, /reconnect/i);
    assert.equal(app.element("viewer-label").textContent,
      status === "different" ? "Run completed with differences" : "Run interrupted");
    assert.equal(app.run("pollFailures"), 0);
    assert.match(app.element("log-content").textContent, /log could not be loaded/i);
    assert.equal(app.element("analyze").disabled, status !== "different");
  });
}

test("an old analysis is ignored even before the new Start POST returns", async () => {
  const analysisResponse = deferred(), startResponse = deferred();
  const app = harness((url) => {
    if (url.endsWith("/analyze")) return analysisResponse.promise;
    if (url === "/api/runs") return startResponse.promise;
    return response({status: "running", elapsed_s: 0, progress: {}});
  });
  app.run(`current = '${oldRun}'`);
  const analysis = app.click("analyze");
  const start = app.click("start");
  assert.equal(app.run("current"), null);
  const status = app.element("status").textContent;
  analysisResponse.resolve({ok: true, json: async () => report()});
  await analysis;
  assert.equal(app.element("analysis").hidden, true);
  assert.equal(app.element("analyze").disabled, true);
  assert.equal(app.element("status").textContent, status);
  assert.equal(app.element("download-report").href, undefined);
  startResponse.resolve({ok: true, json: async () => ({id: newRun})});
  await start;
  assert.equal(app.run("current"), newRun);
});

test("a delayed saved-session restore cannot replace a new Start", async () => {
  const restoreResponse = deferred(), startResponse = deferred();
  const app = harness((url) => {
    if (url === `/api/runs/${oldRun}`) return restoreResponse.promise;
    if (url === "/api/runs") return startResponse.promise;
    return response({status: "running", elapsed_s: 0, progress: {}});
  }, {remembered: oldRun});
  const start = app.click("start");
  restoreResponse.resolve({ok: true, json: async () => ({
    id: oldRun, situation: "masked", status: "completed", elapsed_s: 30, progress: {},
    analysis_available: true,
  })});
  await app.initialization;
  assert.equal(app.run("current"), null);
  assert.equal(app.element("viewer-label").textContent, "Preparing your run");
  assert.equal(app.element("start").disabled, true);
  assert.equal(app.element("analyze").disabled, true);
  startResponse.resolve({ok: true, json: async () => ({id: newRun})});
  await start;
  assert.equal(app.run("current"), newRun);
  assert.equal(app.element("start").disabled, true);
});

test("an episode that never reached a choice does not invent a rule decision", async () => {
  const app = harness(() => response(report({
    claim: "exception", outcome: "execution_error", recorded_answer_used: false,
    same_request_text: null, who_chose: null, option: null,
    comparison: {equal: false, differences: ["execution_error"]},
  })));
  app.run(`current = '${oldRun}'`);
  await app.click("analyze");
  const facts = app.element("run-facts").children;
  const index = facts.findIndex((item) => item.textContent === "Next-step choice");
  assert.ok(index >= 0);
  assert.equal(facts[index + 1].textContent, "Choice not reached");
  assert.doesNotMatch(facts[index + 1].textContent, /rule/i);
});
