/* Botnes SiteShift: replays recorded Site Agent episodes. No network, no library.
   Two separated worlds: the robot column reads episode.robot only; the truth column reads episode.evaluator only.

   Reading map, in file order (search for the names):
   - Data: window.SITESHIFT (data/episodes.js, the recorded episodes) and window.SITESHIFT_FRAMES (data/frames.js,
     the re-rendered camera frames), plus two side objects: hand-written SYNTHETIC fixtures and REHEARSAL material.
     All are generated outside this repository and shipped as they are.
   - Labels and display: esc, regionName, placeName, us and usDates (US wording of the data's strings, at display
     time only), codeWords, codeLabel, titleOf, outcomeWords, chip, exactText (the only reader of the robot's
     verbatim text), badges, isDemo; planSVG (the schematic plan), robotFrame and outsideView (camera images).
   - The three situations and the home: SITUATIONS, situationOf, materialFor, choiceOf, situationsStrip, stripTruth
     (the evaluator's row of the strip), decidersBlock, home, demoList.
   - Navigation: home, episode (step-by-step screen of an earlier episode), situationTabs and nextSituation (between
     the three situations), trap, counters, about.
   - Inspection: inspectScreen, the screen of one situation: whoDecides, reportBlock, then inspectStep1 (the
     inspection player; inspectFrame adds its cameras after Reveal), inspectStep2 (the map), inspectStep3 (the
     choice) and inspectStep4 (the task, with inspectTruth, the evaluator box, and inspectCameras).
   - reportClaims: writes the robot's report from its own records only; its one argument is episode.robot.
   - markClaim: marks one claim against the evaluator's records only; its arguments are the claim and
     episode.evaluator. The marking rules are written above REPORT_GROUPS.
   - Reveal: the home, an earlier episode's screen and each situation's screen keep a revealed flag, false when the
     screen opens, and draw the robot side first. While it is false, the functions that draw the evaluator's records
     (stripTruth, truthColumn, verdicts, inspectTruth, the marks of reportBlock) return before drawing any of them, no
     camera image is drawn, and the functions marked "robot side" never read episode.evaluator. Why: the reader first
     assesses the robot from what it knew; the images are color renders of the simulation and can show what the robot
     had not perceived. For the three situations, the scene as set up also waits for Reveal. The earlier episodes'
     cards and screens describe their scene and outcome up front, and the trap and Counters screens have no Reveal
     step: they show the evaluator's results directly. Reveal is a display rule, not access control: every data file,
     the evaluator's records included, loads with the page.
   - Sources: about (route #/about) shows each earlier episode's source and the files it was read from, with their
     sha256; exact identifiers stay in collapsed blocks.
   - route: the hash router at the end of the file: #/ (home), #/inspect/<id>, #/episode/<id>, #/trap, #/counters,
     #/about. */
(function () {
  "use strict";

  var D = window.SITESHIFT;
  var app = document.getElementById("app");
  if (!D || !D.episodes) {
    app.innerHTML = '<p class="panel">No episode data: <code>site/data/episodes.js</code> did not load. Check that it was copied with this page; if it is missing, copy the <code>site</code> folder from the repository again.</p>';
    return;
  }
  var EP = {};
  D.episodes.forEach(function (e) { EP[e.id] = e; });
  // Re-rendered camera frames (data/frames.js, generated outside this repository): only for episodes whose replay
  // matched the recording. The page works without them (schematic only).
  var FR = (window.SITESHIFT_FRAMES && window.SITESHIFT_FRAMES.episodes) || {};
  // SYNTHETIC fixtures (data/demoA_fixtures.js): hand-written stand-ins for the inspection screen, not robot runs.
  // Their own object, never added to D.episodes, never counted, linked from no screen (only their own #/inspect/
  // route opens one); an id of the recorded episodes always wins, so a fixture can never stand in for one.
  var FXD = window.SITESHIFT_DEMO_FIXTURES || null;
  var FX = {};
  ((FXD && FXD.episodes) || []).forEach(function (e) { if (!EP[e.id] && e.synthetic) FX[e.id] = e; });
  // REHEARSAL material of the three situations (data/demoA_rehearsal.js): episodes prepared before a situation had
  // its recorded episode, never shown as results. Their own object, like the fixtures: never added to D.episodes,
  // never counted, marked on every screen and block; a recorded episode of the same situation always wins (the data
  // shipped with this page hold none: all three situations have their recorded episode).
  var RHD = window.SITESHIFT_DEMO_REHEARSAL || null;
  var RH = {};
  ((RHD && RHD.episodes) || []).forEach(function (e) { if (!EP[e.id] && !FX[e.id] && e.rehearsal) RH[e.id] = e; });

  // ------------------------------------------------------------------ helpers
  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function el(html) { var d = document.createElement("div"); d.innerHTML = html; return d; }
  function regionName(r) {
    var m = /^drawer_unit:(\d+)$/.exec(r || "");
    if (m) return "drawer " + m[1];
    return ({ sink_basin: "sink basin", landing_tray: "landing tray", drying_mat: "drying mat",
      bins_interior: "bins", dishwasher_basket: "dishwasher" })[r] || String(r || "").replace(/_/g, " ");
  }
  function placeName(p) {  // "drawers/1" (evaluator change log) -> "drawer 1"
    var m = /^drawers\/(\d+)$/.exec(p || "");
    if (m) return "drawer " + m[1];
    return String(p || "").replace(/_/g, " ").replace("/", " ");
  }
  // ------------------------------------------------------------------ wording for a US reader
  // The data files keep their recorded strings and stay byte-identical: us rewrites a string when the page shows it.
  // Never applied to the robot's verbatim blocks (exactText) nor inside the scoring (reportClaims, markClaim). Dates
  // as "Oct 20, 2026", times in UTC, US spelling, the robot code named by its date instead of its internal version
  // name (an exact identifier stays only in a collapsed details block, for traceability), "not found" instead of
  // "absence".
  var MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  function usDate(d, m, y) { return +m >= 1 && +m <= 12 ? MONTHS[+m - 1] + " " + (+d) + ", " + y : d + "/" + m + "/" + y; }
  var CODE_WORDS = { "v4.6": "robot code in use on Oct 6, 2026", "v4.7": "robot code of Oct 7, 2026",
    "v4.7.2": "robot code of Oct 10, 2026", "v4.7.3": "frozen robot code of Oct 16, 2026",
    "v4.7.3c": "revised robot code of Oct 16, 2026", "v4.8": "robot code of Oct 17, 2026",
    "DESIG-4": "frozen robot code of Oct 20, 2026" };
  var CODE_TOKEN = /\b(v\d+(?:\.\d+)+[a-z]?|DESIG-\d+)(?![\w.-])/g;
  // whole phrases of the data first (the generic rules below would leave them awkward), then the generic rules
  var PHRASES = [
    ["on 06/10/2026 (code v4.6), replayed", "on Oct 6, 2026 (with the robot code of that day), replayed"],
    ["(memory program of a v4.6 episode)", "(a program kept in memory from an episode that day)"],
    ["(second attempt of a v4.6 episode)", "(the second attempt in an episode that day)"],
    ["before the v4.7.3 changes", "before the changes of Oct 16, 2026"],
    ["Later version (frozen), gate of", "Later version (frozen), check of"],
    ["Our earlier version, gate replay of", "Our earlier version, replayed for the check of"],
    ["Frozen robot code of 20/10/2026 (DESIG-4), in simulation", "Frozen robot code of Oct 20, 2026, in simulation"],
    ["Dry run of v4.8 on", "Dry run of the robot code of Oct 17, 2026 on"],
    ["Accepted under the v4.7 rules, not checked again with the stricter v4.7.3c.", "Accepted under the rules of the " +
      "robot code of Oct 7, 2026, not checked again with the stricter revised code of Oct 16, 2026."],
    ["ended as a lost absence", "ended with no \"not found\" reached"],
    ["the v4.8 state table", "the state table of the robot code of Oct 17, 2026"],
    ["the code of v4.7.2 was not kept", "the robot code of Oct 10, 2026 was not kept"],
    ["Demo A, fixed programs", "Fixed programs, written by the team"],
    ["from its record (choice_call.json)", "from its call record"]
  ];
  var SPELLING = [[/colour/g, "color"], [/Colour/g, "Color"], [/neighbour/g, "neighbor"], [/Neighbour/g, "Neighbor"],
    [/behaviour/g, "behavior"], [/artefact/g, "artifact"], [/\bgrey\b/g, "gray"], [/catalogue/g, "catalog"],
    [/recognis(e|ed|es|ing)\b/g, "recogniz$1"], [/\bcentre/g, "center"], [/\bmetres?\b/g, function (w) { return w.replace("re", "er"); }],
    [/\b([Ff])alse absence\b/g, "$1alse \"not found\""], [/\blost absence\b/g, "no \"not found\" reached"]];
  function usDates(s) {  // the dates only: for the exact identifiers kept for traceability (their version names stay)
    return String(s == null ? "" : s).replace(/\b(\d{1,2})\/(\d{1,2})\/(\d{4})\b/g, function (_, d, m, y) { return usDate(d, m, y); });
  }
  function us(s) {
    var t = String(s == null ? "" : s);
    PHRASES.forEach(function (p) { t = t.split(p[0]).join(p[1]); });
    SPELLING.forEach(function (p) { t = t.replace(p[0], p[1]); });
    t = usDates(t);
    t = t.replace(/\bcode (v\d+(?:\.\d+)+[a-z]?|DESIG-\d+)(?![\w.-])/g, function (w, v) { return CODE_WORDS[v] || w; });
    return t.replace(CODE_TOKEN, function (v) { return CODE_WORDS[v] || v; });
  }
  function codeWords(e) {  // "robot code of Oct 7, 2026", from the episode's code version and date
    var c = (e && e.code) || {}, d = /^(\d{1,2})\/(\d{1,2})\/(\d{4})$/.exec(String(c.date || ""));
    return CODE_WORDS[c.version] || ("robot code" + (d ? " of " + usDate(d[1], d[2], d[3]) : ""));
  }
  function codeLabel(lbl) {  // a frame manifest's code label, "DESIG-4 (frozen copy of 20/10/2026)" -> its plain name
    var m = /^(v\d+(?:\.\d+)+[a-z]?|DESIG-\d+)(?![\w.-])/.exec(String(lbl || ""));
    return m && CODE_WORDS[m[1]] ? "the " + CODE_WORDS[m[1]] : us(lbl);
  }
  function cap1(s) { s = String(s); return s.charAt(0).toUpperCase() + s.slice(1); }
  // the titles and verdict words of the earlier episodes, for a reader (the data keep their recorded words)
  var TITLES = { "Absence established": "Sponge not found in the checked zones",
    "Absence established, with Nemotron": "Sponge not found in the checked zones, with Nemotron",
    "The trap, v4.7.3c (neighbour seed 410008), R10": "The trap, revised code (neighbor seed 410008), R10",
    "The trap, v4.7.3c (neighbour seed 410008), R14": "The trap, revised code (neighbor seed 410008), R14",
    "The v4.8 state table (dry run)": "The robot's state table (dry run)" };
  var OUTCOMES = { "Absence established": "Correct \"not found\" (no sponge in the kitchen)",
    "False absence accepted": "False \"not found\" accepted" };
  function titleOf(e) {
    var t = String((e && e.title) || ""), m = /^Demo A, scene (\w+), kitchen (\w+)$/.exec(t);
    if (m) {  // an episode of the three situations: its situation's name (after Reveal only, like the title itself)
      var sit = SITUATIONS.filter(function (s) { return s.scene === m[1]; })[0];
      return sit ? "Situation " + (SITUATIONS.indexOf(sit) + 1) + ", " + sit.name.toLowerCase() + ", kitchen " + m[2] :
        "Recorded episode, kitchen " + m[2];
    }
    return TITLES[t] || us(t);
  }
  function outcomeWords(v) { var t = String((v && v.outcome_text) || ""); return OUTCOMES[t] || us(t); }
  // Zone labels: exactly three (seen, not found, still to check), worded by the data (D.zone_labels). Word and color
  // always go together, so a label never relies on color alone.
  var LBL = D.zone_labels || {};
  var STATES = ["seen", "not_found", "to_check"];
  function chip(state) {
    return '<span class="chip st-' + esc(state) + '">' + esc(LBL[state] || state) + "</span>";
  }
  // The robot's verbatim texts (its messages, programs, call arguments, the SCENE STATE it sent) are shown as recorded,
  // in a block marked "exact text": they use its own words, never the page's three labels. The only reader of exact_text.
  function exactText(x, what, inline) {
    if (!x || x.exact_text == null) return "";
    var tag = '<span class="exact-tag">exact text' + (what ? ", " + esc(what) : "") + "</span>";
    if (inline) return '<span class="exact inline">' + tag + "<code>" + esc(x.exact_text) + "</code></span>";
    return '<div class="exact">' + tag + "<pre>" + esc(x.exact_text) + "</pre>" +
      (x.truncated ? '<span class="exact-cut">shortened by the page at the end mark</span>' : "") + "</div>";
  }
  function wrapWords(t, n) {  // label lines for the plan, at most n characters each
    var out = [], line = "";
    String(t).split(" ").forEach(function (w) {
      if (line && (line + " " + w).length > n) { out.push(line); line = w; } else line = line ? line + " " + w : w;
    });
    if (line) out.push(line);
    return out;
  }
  function badges(e) {
    var c = e.cost;
    var b = '<span class="badge src-' + esc(e.source.key) + '"><span class="dot" aria-hidden="true"></span>' +
      esc(us(e.source.badge)) + "</span>";
    b += '<span class="badge">' + esc(cap1(codeWords(e))) + "</span>";
    b += '<span class="badge">' + esc(c.model_calls) + " model call" + (c.model_calls === 1 ? "" : "s") + "</span>";
    // a recorded episode of the three situations whose one choice call has a settled cost says so in the header too,
    // as step 3 does; a rule's choice (0 USD, no call), a rehearsal and a fixture never say "settled"
    b += '<span class="badge">' + esc(c.settled_usd) + " USD" + (e.source.key === "nemotron_recorded" ?
      " (settled token cost)" : isDemo(e) && !e.synthetic && !e.rehearsal && c.cost_status === "settled" &&
      c.model_calls > 0 ? " settled" : "") + "</span>";
    if (c.uncertain_usd) b += '<span class="badge warnc">' + esc(c.uncertain_usd) + " USD uncertain, apart</span>";
    b += '<span class="badge">Kitchen ' + esc(e.kitchen) + ", seed " + esc(e.seed) + "</span>";
    b += '<span class="badge sim">Simulation</span>';
    return '<div class="badges">' + b + "</div>";
  }
  // An episode of the three situations (source key "demoA" in the data) illustrates: it is never counted, never
  // listed on Counters or Sources, and opens on its inspection screen only, where the scene as set up comes after
  // Reveal.
  function isDemo(e) { return !!(e && e.source && e.source.key === "demoA"); }
  function setNav(name) {
    document.querySelectorAll(".top nav a").forEach(function (a) {  // the header's nav only (not a situation switcher)
      if (a.getAttribute("data-nav") === name) a.setAttribute("aria-current", "page");
      else a.removeAttribute("aria-current");
    });
  }

  // ------------------------------------------------------------------ the plan (robot side only)
  var SLOT_X = { left_wing: 0, front_left: 1, front_center: 2, front_right: 3, right_wing: 4 };
  var MOD_REGION = { sink: "sink_basin", landing: "landing_tray", drying_mat: "drying_mat", bins: "bins_interior",
    dishwasher: "dishwasher_basket" };
  // focus (optional, robot side): {regions: [...], kind: "look" | "pick" | "help", text}: what the robot does there,
  // drawn as an outline and a dashed line from the robot, so that its behavior shows on the plan, not only in words
  var GLYPH = { seen: "\u25C9", not_found: "\u2713", to_check: "?" };  // a symbol with each label, never alone
  function planSVG(dossier, regions, told, focus) {
    var mods = (dossier && dossier.modules) || [];
    var W = 640, H = 360, colW = 121, gap = 6, x0 = 8;
    var FILL = { seen: "seen", not_found: "notfound", to_check: "tocheck" };
    var FOCUS = { look: "var(--accent)", pick: "var(--ok)", help: "var(--warn)" };
    var want = (focus && focus.regions) || [], marks = "";
    var s = '<svg viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="Schematic plan of the kitchen, robot readings' +
      (focus && focus.text ? ". " + esc(focus.text) : "") + '">';
    s += '<text x="' + W / 2 + '" y="' + (H - 6) + '" text-anchor="middle" font-size="14" fill="var(--muted)">robot</text>';
    s += '<circle cx="' + W / 2 + '" cy="' + (H - 30) + '" r="10" fill="none" stroke="var(--muted)" stroke-width="2"/>';
    mods.forEach(function (m) {
      var col = SLOT_X[m.slot];
      if (col === undefined) return;
      var x = x0 + col * (colW + gap), wing = (m.slot === "left_wing" || m.slot === "right_wing");
      var y = wing ? 40 : 8, h = wing ? 262 : 290;
      var regs = m.kind === "drawers" ? Array.from({ length: m.n_drawers || 0 }, function (_, k) { return "drawer_unit:" + k; })
        : (MOD_REGION[m.kind] ? [MOD_REGION[m.kind]] : []);
      s += '<rect x="' + x + '" y="' + y + '" width="' + colW + '" height="' + h + '" rx="9" fill="var(--panel)" stroke="var(--line)" stroke-width="1.5"/>';
      s += '<text x="' + (x + 9) + '" y="' + (y + 21) + '" font-size="15" fill="var(--muted)">' + esc(m.kind.replace("_", " ")) + "</text>";
      var n = Math.max(regs.length, 1), rh = (h - 34) / n;
      regs.forEach(function (r, k) {
        var st = regions[r] ? regions[r].state : "to_check";  // a zone with no robot reading is still to check
        var ry = y + 30 + k * rh, lines = wrapWords(LBL[st] || st, 14), room = Math.max(1, Math.floor((rh - 28) / 14));
        s += '<rect x="' + (x + 5) + '" y="' + ry + '" width="' + (colW - 10) + '" height="' + (rh - 6) +
          '" rx="6" style="fill:var(--st-' + (FILL[st] || "tocheck") + ')" stroke="rgba(0,0,0,.18)"/>';
        s += '<text x="' + (x + 12) + '" y="' + (ry + 19) + '" font-size="16" font-weight="650" fill="var(--ink)">' + esc(regionName(r)) + "</text>";
        lines.slice(0, room).forEach(function (ln, j) {
          s += '<text x="' + (x + 12) + '" y="' + (ry + 35 + j * 14) + '" font-size="12.5" fill="var(--ink)">' +
            (j === 0 ? esc(GLYPH[st] || "") + " " : "") + esc(ln) + "</text>";
        });
        var fy = ry + 35 + Math.min(lines.length, room) * 14 + 2;
        if (told[r] && fy < ry + rh - 8) {
          s += '<text x="' + (x + 12) + '" y="' + fy + '" font-size="12" fill="var(--muted)">file: ' + esc(told[r]) + "</text>";
        }
        if (want.indexOf(r) >= 0) {  // drawn last, over every module
          var c = FOCUS[focus.kind] || FOCUS.look;
          marks += '<line x1="' + W / 2 + '" y1="' + (H - 41) + '" x2="' + (x + colW / 2) + '" y2="' + (ry + rh - 6) +
            '" stroke="' + c + '" stroke-width="2.5" stroke-dasharray="7 6"/>' +
            '<rect x="' + (x + 2) + '" y="' + (ry - 3) + '" width="' + (colW - 4) + '" height="' + rh + '" rx="8" fill="none" stroke="' +
            c + '" stroke-width="4"' + (focus.kind === "help" ? ' stroke-dasharray="10 6"' : "") + "/>";
        }
      });
    });
    return s + marks + "</svg>";
  }
  // ------------------------------------------------------------------ camera frames
  function frameStepAt(F, k) {  // the last step at or before k that has a frame (a skill call ended there), else -1
    var best = -1;
    Object.keys(F.steps || {}).forEach(function (s) { var i = +s; if (i <= k && i > best) best = i; });
    return best;
  }
  // words (optional): step j in plain words, from the robot's own record. Given by the inspection player of the three
  // situations (through inspectFrame): the caption then names the frame's step and its action, and says when the
  // frame comes from an earlier step than the current one. Without it (an earlier episode's screen), a shorter
  // caption names the frame's step by its skill.
  function robotFrame(e, k, revealed, words) {  // the mast and wrist cameras only, never the outside view; after Reveal only
    var F = FR[e.id];
    if (F && F.status === "identical" && !revealed) {
      return '<div class="noimg">Camera images appear after <strong>Reveal</strong>: they are color renders of the scene ' +
        "from the robot's camera poses, so they can show what the robot has not perceived yet. Before that, this column " +
        "shows the robot's own readings only.</div>";
    }
    if (!F || F.status !== "identical") {
      return '<div class="noimg">No image: ' + esc(us(F && F.reason ? F.reason : "not re-simulated. Schematic only.")) + "</div>";
    }
    var j = frameStepAt(F, k);
    if (j < 0) return '<div class="noimg">No camera frame yet: no skill call has ended at this step.</div>';
    var s = F.steps[String(j)], a = s.after || {}, when = "after step " + (j + 1) + " (" +
      esc(STEP_WORDS[a.name] || String(a.name || "").replace(/_/g, " ")) + ")";  // the step in words, not its skill id
    if (words) {
      var at = "at step " + (j + 1) + ": " + esc(words(j));
      return '<figure class="cam step-cam"><div class="camrow">' +
        '<div><img src="' + esc(s.robot_eye) + '" width="640" height="360" alt="Mast camera ' + at + '"><span>mast camera</span></div>' +
        '<div><img src="' + esc(s.wrist_cam) + '" width="640" height="360" alt="Wrist camera ' + at + '"><span>wrist camera</span></div>' +
        "</div><figcaption><strong>Robot cameras " + at + ".</strong> " + (j === k ? "" : "This frame comes from an earlier " +
          "step than the current step " + (k + 1) + ", " + esc(words(k)) + ", which has no frame of its own: a frame is " +
          "rendered only when a skill call ends. ") + "Re-rendered from a replay with " + esc(codeLabel(F.provenance.code_label)) +
        ". Color render of what the cameras see in the simulation, shown after Reveal; the robot itself reads simulated " +
        "depth and labels, with noise, not this image, and may not have perceived what it shows.</figcaption></figure>";
    }
    return '<figure class="cam"><div class="camrow">' +
      '<div><img src="' + esc(s.robot_eye) + '" width="640" height="360" alt="Mast camera, ' + when + '"><span>mast camera</span></div>' +
      '<div><img src="' + esc(s.wrist_cam) + '" width="640" height="360" alt="Wrist camera, ' + when + '"><span>wrist camera</span></div>' +
      '</div><figcaption><strong>Robot camera, re-rendered from a replay with ' + esc(codeLabel(F.provenance.code_label)) + "</strong>. " +
      (j === k ? "Frame " + when + "." : "Last frame, " + when + "; no skill call since.") +
      " Color render of what the camera sees in the simulation, shown after Reveal; the robot itself reads simulated depth and labels, with noise, not this image, and may not have perceived what it shows." +
      "</figcaption></figure>";
  }
  function outsideView(e, k) {  // evaluator side only: called from truthColumn after Reveal
    var F = FR[e.id];
    if (!F || F.status !== "identical") return "";
    var j = frameStepAt(F, k);
    if (j < 0) return "";
    var s = F.steps[String(j)];
    return '<figure class="cam outside"><img src="' + esc(s.overview) + '" width="640" height="360" alt="Outside view of the kitchen, after step ' +
      (j + 1) + '"><figcaption><strong>Outside view, evaluator side, never given to the robot.</strong> After step ' + (j + 1) +
      ", re-rendered from a replay with " + esc(codeLabel(F.provenance.code_label)) + ".</figcaption></figure>";
  }

  function legend() {
    return '<div class="legend">' + STATES.map(function (k) { return chip(k); }).join("") + "</div>" +
      (D.labels_rule ? '<p class="small muted">' + esc(D.labels_rule) + "</p>" : "");
  }
  // seenIn (optional): zone -> objects the robot saw there; shown on a "still to check" line (the masked situation:
  // the towel it saw in the drawer it could not check), from its own records, never a reason it did not give
  function zoneList(regs, seenIn) {  // one line per zone: label (word and color), then its precision and how to check it
    return '<ul class="regions">' + (regs.length ? regs.map(function (x) {
      // the coverage apart, unless its precision already gives it ("view obs-002, coverage 0.93")
      var cov = x.coverage != null && String(x.precision || "").indexOf("coverage") < 0;
      var here = x.state === "to_check" && seenIn && seenIn[x.region] ? seenIn[x.region] : [];
      return '<li><span class="rname">' + esc(regionName(x.region)) + '</span><span class="rv">' + chip(x.state) +
        (cov ? '<span class="muted small">coverage ' + esc(x.coverage) + "</span>" : "") + "</span>" +
        '<span class="gloss">' + esc(x.precision) + (x.how ? "; how to check: " + esc(x.how) : "") +
        (here.length ? '; <strong class="seen-here">seen in this zone: ' + esc(here.join(", ")) + "</strong>" : "") +
        "</span></li>";
    }).join("") : '<li class="muted">no zone in its search</li>') + "</ul>";
  }
  function seenByZone(objs) {  // the robot's objects by zone, as words, its handles and the sponge left out
    var m = {};
    (objs || []).forEach(function (o) {
      if (o.category === "drawer_handle" || o.category === "sponge") return;
      var c = String(o.category).replace(/_/g, " ");
      m[o.region] = m[o.region] || [];
      if (m[o.region].indexOf(c) < 0) m[o.region].push(c);
    });
    return m;
  }

  // ------------------------------------------------------------------ the three situations
  // "SiteShift: change the kitchen, keep the mission." One kitchen (31000), one mission, one frozen robot code. We
  // start from a kitchen that matches the robot's site file, then move the sponge (another object takes its place) or
  // lay a towel over it. Each situation shows its recorded episode when the data hold one (window.SITESHIFT), else its
  // rehearsal, marked, else nothing (a card that does not open). What we changed is said in general words, never the
  // zone: where the sponge started is the simulator's truth, shown after Reveal only. A known exception: the
  // situation's name and what we changed, with the robot's own readings, can still let a viewer guess it before
  // Reveal. The robot is never told what we changed.
  var MISSION = "Find the sponge, pick it up and scrub the bowl on the landing tray.";
  // The declared limit of this robot code, shown before Reveal on the masked card, in the masked screen's header and
  // in NOT_DONE. Its numbers (13 wrong of 16 cases) come from a diagnostic kept outside this repository: it ran on
  // modified recorded sensor images, not full physics episodes, and assumed that the thin cloth is recognized as the
  // catalog object "towel".
  var LIMIT_THIN_CLOTH = "Known limit: this scene uses the thick catalog towel. In a diagnostic using modified " +
    "recorded sensor images, the check that leads to \"sponge not found\" was wrong in 13 of 16 cases when a thin " +
    "cloth was reported as the " +
    "catalog object \"towel\". That recognition was assumed. These were not full physics episodes; the case remains " +
    "open.";
  var SITUATIONS = [
    { key: "in_place", scene: "A4", kitchen: "31000", name: "Sponge in its place",
      change: "Nothing changed: the kitchen matches the robot's site file." },
    { key: "moved", scene: "A1", kitchen: "31000", name: "Sponge moved",
      change: "We moved the sponge away from the place its site file gives; another object took its place. The " +
        "robot is not told." },
    { key: "masked", scene: "A5", kitchen: "31000", name: "Sponge masked by a cloth",
      change: "We laid a towel over the sponge. The robot is not told.", limit: LIMIT_THIN_CLOTH }
  ];
  // The version id of the frozen robot code the three situations were recorded with. A recorded episode made with
  // another code version is never shown as one of them, whatever its scene and kitchen.
  var DEMO_CODE = "DESIG-4";
  function limitLine(sit) { return sit && sit.limit ? '<p class="small limitline">' + esc(sit.limit) + "</p>" : ""; }
  function situationOf(e) {  // a recorded demo episode or a rehearsal of one of the three situations; a fixture never is
    if (!e || e.synthetic || !e.code || e.code.version !== DEMO_CODE) return null;
    var key = e.rehearsal ? e.rehearsal.situation : null;
    return SITUATIONS.filter(function (s) {
      return key ? s.key === key : isDemo(e) && e.scene === s.scene && String(e.kitchen) === s.kitchen;
    })[0] || null;
  }
  function materialFor(sit) {  // a recorded episode first, then its rehearsal, else none
    var rec = D.episodes.filter(function (e) { return situationOf(e) === sit && e.robot && e.robot.inspection; })[0];
    if (rec) return rec;
    return Object.keys(RH).map(function (k) { return RH[k]; }).filter(function (e) { return situationOf(e) === sit; })[0] || null;
  }
  function choiceOf(e) {  // robot side: who chose the task (or will), in a few words; a fixture's stand-in is never Nemotron
    if (!e) return { key: "none", text: "No episode yet" };
    var tc = (e.robot && e.robot.task_choice) || {};
    if ((tc.problems || []).length) return { key: "doubt", text: "Choice not shown: the page's reader found a problem" };
    if (tc.who_chose === "awaiting_call") return { key: "awaiting", text: "Nemotron to choose: awaiting the recorded Nemotron call" };
    if (tc.who_chose === "nemotron") {
      return e.synthetic ? { key: "stand_in", text: "A SYNTHETIC stand-in answer, written by hand in place of Nemotron" } :
        { key: "nemotron", text: "Decided by Nemotron, from a menu of two (one recorded call)" };
    }
    if (tc.who_chose === "rule") return { key: "rule", text: "Decided by the rule: only one option was valid, no call" };
    if (tc.who_chose === "rule_after_refused_answer") return { key: "rule", text: "Decided by the rule, after the code refused Nemotron's answer" };
    if (tc.who_chose === "inspection_rule") return { key: "rule", text: "Decided by the inspection program's rule, no call" };
    return { key: "none", text: "No choice to make" };
  }
  // where the material comes from: simulation, then recorded, rehearsal or SYNTHETIC, then who chose. inScreen: the
  // inspection screen's header, whose first badges already say simulation and the source (a rehearsal, a fixture)
  function provBadges(e, inScreen) {
    var b = inScreen ? "" : '<span class="badge sim">Simulation</span>';
    if (!e) b += '<span class="badge pend">Awaiting its recorded episode</span>';
    else if (e.synthetic) b += inScreen ? "" : '<span class="badge"><span class="synth-tag">SYNTHETIC</span> fixture, not a robot run</span>';
    else if (e.rehearsal) b += inScreen ? "" : '<span class="badge rh"><span class="rh-tag">REHEARSAL</span> not shown as a result</span>';
    else b += '<span class="badge rec">Recorded episode, replayed: no call while you watch</span>';
    if (e) b += '<span class="badge who who-' + choiceOf(e).key + '">' + esc(choiceOf(e).text) + "</span>";
    return '<div class="badges">' + b + "</div>";
  }
  function provTag(e) {  // the strip's short form of provBadges
    return '<p class="small prov">' + (!e ? '<span class="muted">No episode yet</span>' : e.rehearsal ?
      '<span class="rh-tag">REHEARSAL</span> not shown as a result' : '<span class="rec-tag">RECORDED</span> replayed, simulation') + "</p>";
  }
  function frozenAt(e) { return Math.min(e.robot.inspection.frozen_at_step, e.robot.timeline.length); }
  var LOOK_REFUSED = { nothing_unseen: "nothing left unseen" };
  // a refused step on the friezes (the numbered step lists): the benign "nothing left unseen" in plain words and a
  // neutral style, any other refusal in red with its code
  function refusedClass(x) { return x.ok !== false ? "" : LOOK_REFUSED[x.reason] ? " benign" : " fail"; }
  function refusedWords(x, prefix) {
    if (x.ok !== false) return "";
    return " (" + (LOOK_REFUSED[x.reason] ? esc(LOOK_REFUSED[x.reason]) : esc(prefix) + esc(x.reason)) + ")";
  }
  function looksText(e) {  // its views, then where it looked again (the unseen part of a zone), from its own records
    var tl = e.robot.timeline.slice(0, frozenAt(e)), c = e.robot.inspection.counters || {};
    var all = tl.filter(function (s) { return s.kind === "observe"; }).length;
    var v = c.views_in_inspection != null ? c.views_in_inspection : all;  // the program's views, not the opening one
    var again = tl.filter(function (s) { return s.name === "look_at_region" && s.part === "unseen"; }).map(function (s) {
      return esc(regionName(s.region)) + (s.ok !== false ? "" : LOOK_REFUSED[s.reason] ? " (" + LOOK_REFUSED[s.reason] + ")" :
        ' (refused: <span class="mono">' + esc(s.reason) + "</span>)");
    });
    return v + " view" + (v === 1 ? "" : "s") + (all > v ? " in the inspection (" + all + " with the opening one)" : "") +
      "; it looked again at the unseen part of " + (again.length ? again.join(", ") : "no zone");
  }
  function spongeText(e) {  // where it saw the sponge, against its site file (both robot side)
    var seen = e.robot.inspection.regions.filter(function (z) { return z.state === "seen"; }).map(function (z) { return regionName(z.region); });
    var file = ((e.robot.dossier && e.robot.dossier.stored_items) || []).filter(function (it) { return it.category === "sponge"; })
      .map(function (it) { return regionName(it.region); });
    return (seen.length ? "Seen in " + seen.join(", ") : "Not seen") + (file.length ? "; its site file says " + file.join(", ") : "");
  }
  function toCheckText(e) {  // the zones of its search still to check, with what it saw there
    var ins = e.robot.inspection, seen = seenByZone((e.robot.timeline[frozenAt(e) - 1] || {}).objects);
    var z = ins.regions.filter(function (x) { return x.state === "to_check"; }).map(function (x) {
      return esc(regionName(x.region)) + (seen[x.region] ? " (seen there: " + esc(seen[x.region].join(", ")) + ")" : "");
    });
    return (z.length ? z.join(", ") : "none in its search") + "; outside its search: ask a person";
  }
  function pickZone(e) {  // the zone of an accepted pick: the zone its task program looked at just before it
    var last = null, out = null;
    e.robot.timeline.slice(frozenAt(e)).forEach(function (s) {
      if (s.region) last = s.region;
      if (s.name === "pick" && s.ok === true && out === null) out = last;
    });
    return out;
  }
  function asksHelp(e) { return /^help_requested/.test(String(e.robot.final_claim || "")); }
  // the task's own steps after the inspection, from its records: a scrub is said only when one was accepted, a
  // refused pick is said as such, and "no arm move" only when no gesture followed the inspection
  function taskSteps(e) { return e.robot.timeline.slice(frozenAt(e)); }
  function scrubbed(e) { return taskSteps(e).some(function (s) { return s.name === "scrub" && s.ok === true; }); }
  function refusedPick(e) { return taskSteps(e).filter(function (s) { return s.name === "pick" && s.ok === false; })[0] || null; }
  function armMoved(e) { return taskSteps(e).some(function (s) { return s.kind === "skill"; }); }
  function reasonWords(s) { return String((s && s.reason) || "no reason recorded").replace(/_/g, " "); }
  function toCheckZones(e) {
    return e.robot.inspection.regions.filter(function (x) { return x.state === "to_check"; }).map(function (x) { return regionName(x.region); });
  }
  function actionText(e) {  // what the arm did after the choice, from its own records
    var r = e.robot, p = pickZone(e), rp = refusedPick(e), z = toCheckZones(e);
    var zones = z.length ? "; still to check: " + z.join(", ") + "." : ".";
    if ((r.task_choice || {}).who_chose === "awaiting_call") return "No task yet: awaiting the recorded Nemotron call.";
    if (p) return "Picks the sponge in " + regionName(p) + (scrubbed(e) ? " and scrubs the bowl; " : "; no scrub accepted; ") +
      (r.final_claim === "task_done" ? "its done() is accepted by its guard." : "then " + r.final_claim_text + ".");
    if (rp) return "Its pick was refused by its guard (" + reasonWords(rp) + "); " + (asksHelp(e) ? "asks a person" + zones :
      "then " + r.final_claim_text + ".");
    if (asksHelp(e)) return "Asks a person, with no pick" + zones;
    return "Ends: " + r.final_claim_text + ".";
  }
  function miniMap(e) {  // its map at the end of the inspection, three labels, and where the arm went
    var p = pickZone(e), help = asksHelp(e);
    return '<ul class="mini-map">' + e.robot.inspection.regions.map(function (z) {
      var act = z.region === p ? '<span class="act pick">picked here</span>' :
        help && z.state === "to_check" ? '<span class="act help">ask a person</span>' : "";
      return '<li><span class="rname">' + esc(regionName(z.region)) + '</span><span class="rv">' + chip(z.state) + act + "</span></li>";
    }).join("") + "</ul>";
  }
  function outcomeClass(v) {
    return { success: "good", absence_established: "good", false_absence_accepted: "badc", false_success_claimed: "badc" }[v.outcome_class] || "warnc";
  }
  // a help with the sponge present says so, and whether it was the expected end (the masked situation: the robot
  // never lifts the cloth), instead of the bare class "Help". One wording for the strip's row and the screen's
  // evaluator line. Evaluator side: called after Reveal only.
  function helpWords(v, sit) {
    return v.outcome_class === "help" && v.help_with_sponge_present && !v.false_absence ?
      "Help with the sponge present" + (sit && sit.key === "masked" ? " (expected)" : "") : null;
  }
  // tally (optional): adds this report's marks to the strip's total, after the Reveal guard below only
  function stripTruth(e, revealed, sit, tally) {  // evaluator side: nothing of it before Reveal
    if (!revealed) return '<p class="small muted">After Reveal.</p>';
    if (!e) return '<p class="small muted">No episode yet.</p>';
    var v = e.evaluator, zc = v.zone_check || {}, cr = v.criteria || {}, keys = Object.keys(cr);
    var noFalse = zc.valid && !(zc.zones || []).some(function (z) { return z.error; });
    var hw = helpWords(v, sit), said = hw ? hw + (noFalse ? '; no false "not found"' : "") : outcomeWords(v);
    var h = (e.rehearsal ? '<p class="small"><span class="rh-tag">REHEARSAL</span> check, not a result</p>' : "") +
      '<p class="v ' + outcomeClass(v) + '"><strong>' + esc(said) + "</strong>" + (v.outcome_class === "success" ?
        ", task criteria " + keys.filter(function (k) { return cr[k]; }).length + " of " + keys.length : "") + "</p>";
    h += '<p class="small">' + (v.sponge_present === false ? "No sponge in the kitchen." : "The sponge started in " +
      esc(regionName(v.sponge_truth_region)) + ".") + "</p>";
    // the robot's report, checked against the evaluator's records: its score, one line, after Reveal only (the guard
    // above); a count, never a success rate
    var cl = reportOf(e), mk = cl ? cl.map(function (c) { return markClaim(c, v); }) : null;
    if (mk && tally) addMarks(tally, mk);
    if (mk) h += '<p class="small report-score"><strong>Its report, checked against the evaluator\'s records:</strong> ' +
      esc(reportScore(mk)) + " The count is not a success rate.</p>";
    if (zc.valid) {
      var bad = (zc.zones || []).filter(function (z) { return z.error; }).length;
      h += '<p class="small">Zone check of its map: ' + (bad ? '<strong class="badc">' + bad + ' false "not found"</strong>' :
        'no false "not found"') + "</p>";
    } else h += '<p class="small warnc">' + esc(zc.why_not || "No zone-by-zone check.") + "</p>";
    if (v.outcome_class !== "awaiting_call") h += '<p class="small">False "not found": ' + (v.false_absence ? '<strong class="badc">yes</strong>' : "no") + "</p>";
    return h;
  }
  function situationCard(sit, e) {
    // numbered in the order a reader should follow; the first one says to start there
    var n = SITUATIONS.indexOf(sit) + 1;
    var h = '<p class="eyebrow sit-n">Situation ' + n + " of " + SITUATIONS.length + "</p><h3>" + esc(sit.name) +
      '</h3><p class="change"><span class="k">What we changed</span> ' + esc(sit.change) + "</p>" + limitLine(sit) + provBadges(e);
    if (e && e.rehearsal) h += '<p class="small pendline">Why a rehearsal: ' + esc(e.rehearsal.reasons.join("; ")) + ".</p>";
    if (!e) h += '<p class="small pendline">No episode of this situation yet.</p>';
    return e ? '<a class="sit-card" href="#/inspect/' + esc(e.id) + '">' + h + '<span class="go">' + (n === 1 ?
      "Start here: open situation 1" : "Open situation " + n) + "</span></a>" :
      '<div class="sit-card off" aria-disabled="true">' + h + "</div>";
  }
  function situationsStrip(mats, revealed) {  // side by side: what changed, what the robot did differently, then the truth
    var rows = [["Its map after the inspection", miniMap], ["Where it looked", looksText], ["Sponge", spongeText],
      ["Still to check", toCheckText], ["Who decides", function (e) { return esc(choiceOf(e).text); }], ["What it does", actionText]];
    var h = '<section class="panel strip" aria-labelledby="strip-h"><h2 id="strip-h">What changed, and what the robot did differently</h2>' +
      '<p class="small muted">Its own readings and actions first. Its inspection route is nearly the same in the three ' +
      "situations; the difference shows in what it reads, who decides and where the arm goes.</p><div class=\"strip-grid\">";
    // tally: the three reports' marks, added up by stripTruth after its Reveal guard, for the line under the three
    // columns; robot side here, nothing of the evaluator is read in this function
    var tally = { episodes: 0, n: 0, "true": 0, "false": 0, nc: 0 };
    SITUATIONS.forEach(function (sit, j) {
      var e = mats[j];
      h += '<div class="strip-col"><h3><span class="sit-num" aria-hidden="true">' + (j + 1) + "</span>" + esc(sit.name) +
        "</h3>" + provTag(e) + '<dl class="strip-rows">';
      h += "<dt>What we changed</dt><dd>" + esc(sit.change) + "</dd>";
      rows.forEach(function (rw) {
        h += "<dt>" + esc(rw[0]) + "</dt><dd>" + (e ? (rw[1] === spongeText || rw[1] === actionText ? esc(rw[1](e)) : rw[1](e)) :
          '<span class="muted">No episode yet.</span>') + "</dd>";
      });
      h += '<dt id="' + (j === 0 ? "strip-truth-h" : "strip-truth-" + j) + '" tabindex="-1">The evaluator\'s result</dt><dd>' +
        stripTruth(e, revealed, sit, tally) + "</dd></dl></div>";
    });
    h += "</div>" + (revealed ? (tally.episodes === SITUATIONS.length ? reportTotal(tally) : "") :
      '<button class="reveal" data-act="reveal">Reveal the evaluator\'s results</button>') + "</section>";
    return h;
  }
  function decidersBlock(mats) {  // who decides what, for the three situations: team, code, Nemotron, the evaluator
    // while a situation still waits for its recorded call, the page says the call is not made yet
    var waiting = (mats || []).some(function (e) { return e && e.rehearsal && choiceOf(e).key === "awaiting"; });
    return '<section class="panel deciders" aria-labelledby="deciders-h"><h2 id="deciders-h">Who decides what</h2><div class="wd-grid">' +
      '<div class="wd team"><h3>Programmed by the team</h3><p class="small">The inspection: open each drawer, look at each zone, ' +
      "look again at the unseen part. The task gestures: pick the sponge and scrub the bowl, or stop and ask a person. Where " +
      "the arm picks follows the sponge the robot itself perceived.</p></div>" +
      '<div class="wd code"><h3>Checked by code</h3><p class="small">Which options are valid: eight conditions read in the ' +
      "robot's own text. Its guards accept or refuse a pick, a done() and a \"not found\".</p></div>" +
      '<div class="wd model"><h3>Nemotron</h3><p class="small">One thing only: scrub the bowl or ask a person, from the ' +
      "robot's text summary, when the code finds both valid. Text only, no images. At most one call per situation, " +
      "recorded beforehand and replayed here" + (waiting ? " (not made yet for these rehearsals)" : "") + "; none when " +
      "only one option is valid. The model: NVIDIA Nemotron 3 Super, " +
      "through Nebius Token Factory.</p></div>" +
      '<div class="wd eval"><h3>Evaluator</h3><p class="small">The simulator\'s truth, never given to the robot: where the ' +
      "sponge started, the task criteria, a zone-by-zone check of the map. After Reveal.</p></div></div></section>";
  }
  function robotsNote() {  // secondary: one robot, in simulation, nothing to click; no other robot is listed, since
    // nothing in this package supports one
    return '<section class="panel robots" aria-labelledby="robots-h"><h2 id="robots-h">Robots</h2><ul class="robot-list">' +
      '<li class="on"><strong>Panda arm</strong> <span class="small">Supported today, in simulation: the whole route on this page. ' +
      "No other robot is supported.</span></li></ul></section>";
  }

  // ------------------------------------------------------------------ home
  var SCENES = [
    { n: "1", id: "s1_moved_31000", t: "Sponge moved, found by searching" },
    { n: "2", id: "s2_poorly_seen_41008", t: "Poorly seen zone", focus: "found",
      line: "The sponge stands on its edge at the back of drawer 1, under the counter top. Straight views miss it; a look at the part not yet seen finds it." },
    { n: "3", id: "s2_poorly_seen_41008", t: "Help, sponge out of reach", focus: "end",
      line: "Same episode, at the end: the robot refuses its own pick under the counter top and asks for help instead of forcing it." },
    { n: "4", id: "s1b_moved_31001", t: "Physical success", focus: "end" },
    { n: "5", id: "s5_absent_41000", t: "Sponge not found in the checked zones", focus: "end" },
    { n: "6", route: "#/trap", t: "The trap, before and after", id: "t_before_g2_r10" }
  ];
  var homeState = { revealed: false };
  // the top block a visitor reads first, in plain words: what SiteShift adds, how to read the page (the robot's side
  // first, then Reveal), and where to start. Robot side and method only: no result, no mark, no evaluator record
  // (the Reveal rule, see the reading map at the top). first: the material of situation 1, or null (then no start
  // link)
  var NEMOTRON_ADDS = "When both are valid, NVIDIA Nemotron (Nemotron 3 Super, through Nebius Token Factory) picks one " +
    "from the robot's text summary, with no images: at most one call per situation, recorded beforehand and replayed here.";
  function addsBlock(first) {
    return '<div class="adds" aria-labelledby="adds-h"><h2 id="adds-h">What SiteShift adds</h2><ol class="adds-list">' +
      "<li><strong>Inspection before acting.</strong> The robot looks at each zone it can reach and marks it with one of " +
      "three labels: " + chip("seen") + " " + chip("not_found") + " " + chip("to_check") + ". Where its views do not " +
      "settle it, the zone stays still to check.</li>" +
      "<li><strong>Code checks the options, Nemotron chooses.</strong> Code decides which tasks are valid: scrub the bowl, " +
      "or ask a person. " + esc(NEMOTRON_ADDS) + " The inspection and the arm's gestures are fixed programs written by the " +
      "team.</li>" +
      "<li><strong>Its own report, checked after the run.</strong> The page's code writes the robot's report from the " +
      "robot's own records (its steps, its zone labels, its difference report, its final claim), one claim per " +
      "assertion; no model writes it. After the run, each claim is checked against the evaluator's records (the " +
      "simulator's records, never shown to the " +
      "robot) and marked true, false or not checkable.</li></ol>" +
      '<p class="howto"><strong>How to read this page.</strong> Open the three situations in order. First see what the ' +
      "robot does, from its own records only: its readings, its map, its choice, its action and its report. Then press " +
      "Reveal to see what the evaluator's records confirm, contradict or cannot check. What stays unknown stays on the " +
      "page.</p>" + (first ? '<a class="start" href="#/inspect/' + esc(first.id) + '">Start with situation 1: ' +
      esc(SITUATIONS[0].name.toLowerCase()) + "</a>" : "") + "</div>";
  }
  function home() {
    setNav("home");
    var mats = SITUATIONS.map(materialFor);
    var h = '<section class="hero" aria-labelledby="hero-h"><p class="eyebrow">Botnes SiteShift</p>' +
      '<h1 id="hero-h" class="hero-line">Change the kitchen, keep the mission.</h1>' +
      '<p class="mission"><span class="k">Mission</span> ' + esc(MISSION) + "</p>" +
      '<p class="lead">Same kitchen, same robot, same mission. We start from a kitchen that matches the robot\'s site file, ' +
      "then move the sponge or lay a towel over it. The robot is not told. It inspects the zones it can reach, marks each " +
      "one with one of three labels, then does the task or asks a person. An evaluator the robot never sees checks the " +
      "result.</p>" + addsBlock(mats[0]) +
      '<div class="badges"><span class="badge sim">Simulation (MuJoCo), simulated perception</span><span class="badge">Panda ' +
      'arm, development kitchen 31000</span><span class="badge rec">Replayed from records: no model call while you ' +
      "watch</span></div></section>";
    h += '<section class="situations" aria-labelledby="sit-h"><h2 id="sit-h">Three situations, one kitchen, one mission</h2>' +
      '<div class="sit-cards">' + SITUATIONS.map(function (s, j) { return situationCard(s, mats[j]); }).join("") + "</div></section>";
    h += situationsStrip(mats, homeState.revealed) + decidersBlock(mats) + robotsNote();
    h += demoList();
    h += '<section class="earlier" aria-labelledby="earlier-h"><h2 id="earlier-h">Earlier recorded episodes, robot code of Oct 7 to Oct 16, 2026, for reference</h2>' +
      '<p class="lead">A robot is sent into a kitchen it has a site file for. The site file is out of date. ' +
      "Each scene replays a recorded episode: what the robot knew, what it did, and, once you reveal it, what an " +
      "evaluator the robot never sees (the simulator's truth) says. Selected scenes illustrate; they are not a results table.</p>";
    h += '<div class="cards">';
    SCENES.forEach(function (sc) {
      var e = EP[sc.id], href = sc.route || ("#/episode/" + sc.id + (sc.focus ? "?at=" + sc.focus : ""));
      var src = sc.route ? "Fixed replays, robot code of Oct 10 and Oct 16, 2026" : us(e.source.badge) + ", " + codeWords(e);
      var line = sc.line || (sc.route ? "Same case, same seed: our earlier version accepts a false \"not found\"; the later one does not." : us(e.story));
      h += '<a class="card" href="' + href + '"><span class="num">Scene ' + sc.n + '</span><span class="t">' + esc(sc.t) +
        '</span><span class="s">' + esc(line) + '</span><span class="badges"><span class="badge src-' +
        esc(sc.route ? "fixed_replay" : e.source.key) + '"><span class="dot" aria-hidden="true"></span>' + esc(src) + "</span></span></a>";
    });
    h += "</div>";
    h += '<section class="more panel"><h2>More recorded episodes</h2><ul>';
    D.episodes.filter(function (e) { return !isDemo(e) && ["complement", "trap_refusal", "trap_corrected", "format"].indexOf(e.role) >= 0; })
      .forEach(function (e) {
        h += '<li><a href="#/episode/' + esc(e.id) + '">' + esc(titleOf(e)) + "</a> " +
          '<span class="muted small">' + esc(us(e.source.badge)) + ", " + esc(codeWords(e)) + "</span></li>";
      });
    h += "</ul></section></section>";
    app.innerHTML = h;
    app.onclick = function (ev) {  // Reveal, for the strip: the evaluator's row of the three situations
      var t = ev.target.closest("[data-act]");
      if (!t || t.getAttribute("data-act") !== "reveal") return;
      homeState.revealed = true;
      if (app.setAttribute) app.setAttribute("aria-live", "off");  // the page is redrawn; the focus says where it went
      home();
      var f = document.getElementById("strip-truth-h");
      if (f && f.focus) f.focus();
    };
  }
  // The recorded episodes of the three situations. The links name the situation (the name its card already shows
  // before Reveal) and the kitchen only: what was set up in a scene is the simulator's truth, shown after Reveal. The
  // SYNTHETIC fixtures are not listed on this public page: their #/inspect/synthetic_* routes stay, for the team's
  // tests. No internal name (the data's scene ids, such as A4) is shown.
  function demoList() {
    var demo = D.episodes.filter(function (e) { return e.source.key === "demoA" && e.robot.inspection; });
    var h = '<section class="more panel"><h2>The recorded episodes: inspect, map, choose, act</h2>' +
      '<p class="small muted">One session: a fixed inspection program, the map of what the robot checked, a choice ' +
      "from a menu of two, the task, then the verdict. Simulation only.</p>";
    h += demo.length ? "<ul>" + demo.map(function (e) {
      var sit = situationOf(e);
      return '<li><a href="#/inspect/' + esc(e.id) + '">' + (sit ? "Situation " + (SITUATIONS.indexOf(sit) + 1) + ", " +
        esc(sit.name.toLowerCase()) : "Recorded episode") + ", kitchen " + esc(e.kitchen) + "</a> " +
        '<span class="muted small">' + esc(us(e.source.badge)) + ", " + esc(codeWords(e)) + "</span></li>";
    }).join("") + "</ul>" : '<p class="small">No recorded episode of the three situations yet: awaiting the recorded ' +
      "Nemotron calls and the seed review.</p>";
    return h + "</section>";
  }

  // ------------------------------------------------------------------ episode
  var timer = null;
  function stopTimer() { if (timer) { clearInterval(timer); timer = null; } }

  function marks(e) {
    var tl = e.robot.timeline, n = Math.max(tl.length - 1, 1), out = [];
    (e.robot.attempts || []).forEach(function (a) {
      if (!a.host_span) return;
      var at = Math.min(a.host_span[1], tl.length) - 1;
      var kind = a.status === "not_found_refused" ? "refused" : a.status === "help_review" ? "help" : "attempt";
      out.push({ at: at, kind: kind, text: "attempt " + a.n + ": " + a.result });
    });
    var seen = -1;
    tl.some(function (s, k) {
      if (Object.keys(s.regions || {}).some(function (r) { return s.regions[r].state === "seen"; })) { seen = k; return true; }
      return false;
    });
    if (seen >= 0) out.push({ at: seen, kind: "found", text: "the robot sees the sponge" });
    return { list: out, n: n, seen: seen };
  }

  function robotColumn(e, k) {
    var r = e.robot, s = r.timeline[k] || {}, last = k === r.timeline.length - 1;
    var told = {};
    ((r.dossier && r.dossier.stored_items) || []).forEach(function (it) { told[it.region] = (told[it.region] ? told[it.region] + ", " : "") + it.category; });
    var h = '<h2>What the robot knows</h2><div class="who">Robot readings only. ' + (r.reconstructed ? esc(us(r.reconstructed_note)) : "") + "</div>";
    h += '<dl class="kv"><dt>Site file</dt><dd>' + (((r.dossier && r.dossier.stored_items) || []).map(function (it) {
      return esc(it.category.replace("_", " ")) + " in " + esc(regionName(it.region));
    }).join("; ") || "none") + "</dd>";
    if (s.search_so_far) h += '<dt>Search</dt><dd><span class="mono">' + esc(s.search_so_far) + "</span> (its code)</dd>";
    else if (last && r.final_scope && r.final_scope.outcome) h += '<dt>Search, at the end</dt><dd><span class="mono">' + esc(r.final_scope.outcome) + "</span> (its code)</dd>";
    h += "</dl>";
    var fin = !!(last && r.final_scope && r.final_scope.regions && r.final_scope.regions.length);
    var regs = fin ? r.final_scope.regions.concat(r.final_scope.outside || []) :
      Object.keys(s.regions || {}).map(function (name) { return s.regions[name]; });
    h += "<h3>" + (fin ? "Zones, from its own search reading at the end" : "Zones at this step, from its own readings") + "</h3>";
    var banner = fin ? r.final_scope.banner : s.banner;
    if (banner) h += '<div class="banner" role="note">' + esc(banner) + "</div>";
    h += zoneList(regs);
    if (fin && r.final_scope.footnote) h += '<p class="small muted">' + esc(r.final_scope.footnote) + "</p>";
    var objs = (s.objects || []).filter(function (o) { return o.category !== "drawer_handle"; });
    if (objs.length) {
      h += '<h3>Objects seen</h3><p class="objs">' + objs.slice().reverse().map(function (o) {
        return esc(o.category.replace("_", " ")) + " in " + esc(regionName(o.region)) + " (" + esc(o.obs) + ")" + (o.note ? ", " + esc(o.note) : "");
      }).join(" · ") + "</p>";
    }
    (r.guard || []).forEach(function (g) {
      var a = (r.attempts || [])[g.attempt];
      if (a && a.host_span && k >= Math.min(a.host_span[1], r.timeline.length) - 1) {
        h += '<p class="small"><strong>Its guard</strong> on "not found": ' + (g.decision === "refused" ?
          '<span class="badc">REFUSED</span>, still to check per its guard: ' + esc((g.not_cleared || []).map(regionName).join(", ")) :
          '<span class="good">accepted</span>') + "</p>";
      }
    });
    if (last && r.hand_text) h += '<p class="small">At the end, ' + esc(r.hand_text) + "</p>";
    if (r.scene_state_now && r.scene_state_now.length) {
      h += "<h3>SCENE STATE NOW, as sent to the model (fake model, dry run)</h3>";
      r.scene_state_now.forEach(function (q) {
        h += '<p class="small muted">request ' + esc(q.request) + "</p>" + exactText(q, "the robot's text as sent");
      });
    }
    return h;
  }

  function describeChange(c, demo) {  // one evaluator change-log entry in plain words (demo: a situation, not a trap)
    if (c.change === "world_dev") {
      var where = c.drawer ? " in " + placeName(c.drawer) : "";
      if (c.towel) return (demo ? "" : "trap: ") + "a towel over the " + (c.towel.over === "sponge" ? "sponge" : "drawer floor") + where;
      if (c.sponge) return "trap: the sponge " + String(c.sponge.pose || "placed") + where;
      return "trap " + String(c.mode || "") + where;
    }
    if (c.change === "removed") return String(c.item).replace("_", " ") + " removed" + (c.was ? " (was in " + placeName(c.was) + ")" : "");
    return String(c.item).replace("_", " ") + " " + String(c.change) + (c.from ? " from " + placeName(c.from) : "") +
      (c.to ? " to " + placeName(c.to) : "");
  }
  function truthColumn(e, revealed, k) {
    var v = e.evaluator;
    var h = '<h2>Ground truth</h2><div class="who">Evaluator view, never given to the robot.</div>';
    if (!revealed) return h + '<button class="reveal" data-act="reveal">Reveal the truth</button>';
    h += outsideView(e, k);
    h += '<dl class="kv">';
    h += "<dt>Sponge</dt><dd>" + (v.sponge_present === false ? "<strong>not in the kitchen</strong>" :
      "in <strong>" + esc(regionName(v.sponge_truth_region)) + "</strong>") + "</dd>";
    h += "<dt>Changes since the site file</dt><dd>" + ((v.site_file_changes || []).map(function (c) { return esc(describeChange(c)); }).join("; ") || "none") +
      "</dd>";
    h += "<dt>Task target</dt><dd>" + esc(v.target || "none") + "</dd>";
    h += "</dl>";
    return h;
  }

  function verdicts(e, revealed) {
    var r = e.robot, v = e.evaluator;
    var h = '<h2>Verdicts</h2><div class="verdict robot"><h3>Guard (robot side)</h3>';
    if ((r.guard || []).length) {
      h += r.guard.map(function (g) {
        return '<p class="small">"' + esc(String(g.claim).replace(/_/g, " ")) + '": ' + (g.decision === "refused" ?
          '<strong class="badc">REFUSED</strong>, still to check per its guard: ' + esc((g.not_cleared || []).map(regionName).join(", ")) :
          '<strong class="good">ACCEPTED</strong>') + "</p>";
      }).join("");
    } else h += '<p class="small muted">No "not found" claim reached the guard in this episode.</p>';
    if (r.help_reviews) h += '<p class="small">Help reviews: ' + esc(r.help_reviews) + ' (a help is not accepted while regions remain to search)</p>';
    h += '<p class="small">Final: <strong>' + esc(r.final_claim_text) + "</strong></p>";
    if (r.end_of_replay) h += '<div class="endreplay">End of replay: no recorded answer for the next step. This is an artifact of the replay, not a robot answer.</div>';
    h += '</div><div class="verdict truth"><h3>Evaluator (simulator truth)</h3>';
    if (!revealed) return h + '<button class="reveal" data-act="reveal">Reveal the truth</button></div>';
    var cr = v.criteria || {}, keys = Object.keys(cr), okc = keys.filter(function (k) { return cr[k]; }).length;
    var cls = { success: "good", absence_established: "good", false_absence_accepted: "badc", false_success_claimed: "badc" }[v.outcome_class] || "warnc";
    h += '<p class="big ' + cls + '">' + esc(outcomeWords(v)) + "</p>";
    if (v.outcome_class === "success") h += '<p class="small">' + okc + " of " + keys.length + " task criteria" +
      (v.scrub ? "; scrub " + esc(v.scrub.duration_s) + " s at " + esc(v.scrub.force_mean_n) + " N mean" : "") + "</p>";
    h += '<p class="small">False "not found": ' + (v.false_absence ? '<strong class="badc">yes</strong>' : "no") +
      " · False success: " + (v.false_success ? '<strong class="badc">yes</strong>' : "no") +
      (v.help_with_sponge_present ? " · Help with the sponge present" : "") + "</p>";
    h += '<p class="small">Violations: ' + (v.violations.length ? esc(v.violations.join(", ")) : "0") + "</p>";
    var c = v.contacts_over_20n;
    h += '<p class="small">Hand contacts over 20 N, counted: ' + (c.counted ? '<strong class="warnc">' + c.counted +
      " (peak " + esc(c.counted_peak_n) + " N), an incident</strong>" : "0") + "</p>";
    if (c.calibration_exempt) h += '<p class="small muted">Calibration contacts, exempt by the bench rule: ' + c.calibration_exempt +
      " (peak " + esc(c.calibration_peak_n) + " N), shown, not hidden</p>";
    return h + "</div>";
  }

  function attemptsBox(e, curAttempt) {
    var a = e.robot.attempts || [];
    var h = "<h2>Model plan and corrections</h2>";
    if (!a.length) return h + '<p class="small muted">' + (e.source.key === "scripted_chain" ?
      "Scripted bench chain: no model program in this episode." : "No program recorded.") + "</p>";
    h += '<ol class="attempts">';
    a.forEach(function (x) {
      h += '<li class="' + (x.n === curAttempt ? "cur" : "") + '"><div><strong>' + x.n + "</strong> " + esc(us(x.result)) + "</div>" +
        '<div class="o">' + esc(us(x.origin)) + "</div>";
      if (x.help_review_message) h += exactText(x.help_review_message, "its help review's message");
      if (x.guard_message) h += exactText(x.guard_message, "its guard's message");
      if (x.message) h += exactText(x.message, "its message");
      if (x.program_text) h += "<details><summary>Show the program text (as recorded, shown as text, not as evidence)</summary>" +
        exactText(x.program_text, "the program as recorded") + "</details>";
      h += "</li>";
    });
    return h + "</ol>";
  }

  function episode(id, query) {
    var e = EP[id];
    if (!e) { app.innerHTML = '<p class="panel">Unknown episode.</p>'; return; }
    if (isDemo(e)) {  // its own screen only: the step-by-step screen would show its scene before Reveal
      var hist = window.history;
      if (hist && hist.replaceState) hist.replaceState(null, "", "#/inspect/" + id);
      inspectScreen(id);
      return;
    }
    setNav(e.role.indexOf("trap") === 0 ? "trap" : "home");
    var tl = e.robot.timeline, M = marks(e);
    var st = { k: 0, revealed: false };
    if (query.at === "end") st.k = tl.length - 1;
    if (query.at === "found" && M.seen >= 0) st.k = M.seen;
    function curAttempt() { var s = tl[st.k] || {}; return s.attempt; }
    function draw() {
      var s = tl[st.k] || {};
      var h = '<div class="ep-head"><div><a href="#/" class="small">All scenes</a><h1>Scene ' + esc(e.scene) + ". " + esc(titleOf(e)) + "</h1>" +
        badges(e) + '<p class="lead">' + esc(us(e.story)) + "</p>" + (e.caveat ? '<p class="small warnc">' + esc(us(e.caveat)) + "</p>" : "") +
        (e.program ? '<p class="small muted">Program: ' + esc(us(e.program)) + ". " + esc(us(e.cost.note)) + ".</p>" : "") +
        '<p class="small muted">' + esc(us(e.source.text)) + " " + esc(us(e.code.note)) + ".</p></div></div>";
      var regionsNow = (st.k === tl.length - 1 && e.robot.final_scope && e.robot.final_scope.regions.length) ?
        e.robot.final_scope.regions.concat(e.robot.final_scope.outside || []).reduce(function (m, x) { m[x.region] = x; return m; }, {}) : (s.regions || {});
      var told = {};
      ((e.robot.dossier && e.robot.dossier.stored_items) || []).forEach(function (it) { told[it.region] = (told[it.region] ? told[it.region] + ", " : "") + it.category.replace("_", " "); });
      h += '<div class="cols"><section class="col robot" aria-label="What the robot knows">' + robotColumn(e, st.k) + "</section>" +
        '<section class="col plan" aria-label="Robot view, schematic plan"><h2>Robot view / plan</h2>' + robotFrame(e, st.k, st.revealed) +
        '<div class="who">Its readings drawn on the site file\'s layout.</div>' +
        planSVG(e.robot.dossier, regionsNow, told) + '<div class="cap">Schematic, not to scale. Color and word give the robot\'s own reading of each zone, one of three labels.</div>' + legend() + "</section>" +
        '<section class="col truth" aria-label="Ground truth">' + truthColumn(e, st.revealed, st.k) + "</section></div>";
      var pct = function (at) { return (100 * at / M.n).toFixed(2) + "%"; };
      h += '<section class="timeline panel" aria-label="Timeline"><div class="tl-row">' +
        '<button class="btn" data-act="prev" aria-label="Previous step">&#9664;</button>' +
        '<button class="btn primary" data-act="play" aria-label="Play or pause">' + (timer ? "Pause" : "Play") + "</button>" +
        '<button class="btn" data-act="next" aria-label="Next step">&#9654;</button>' +
        '<input type="range" min="0" max="' + (tl.length - 1) + '" value="' + st.k + '" aria-label="Step" data-act="seek">' +
        '<span class="speed">step ' + (st.k + 1) + " of " + tl.length + " · replay, accelerated, one step every 0.7 s</span></div>" +
        '<div class="marks" aria-hidden="true">' + M.list.map(function (m) {
          return '<span class="mark ' + m.kind + '" style="left:' + pct(m.at) + '" title="' + esc(m.text) + '"></span>';
        }).join("") + "</div>" +
        '<div class="step-now"><strong>' + esc(s.name) + "</strong> " + exactText(s.args, "", true) + " " +
        (s.ok === true ? '<span class="good">ok</span>' : s.ok === false ? '<span class="badc">' + esc(s.reason) + "</span>" : "") +
        (s.reason && s.ok === true && s.reason !== "ok" ? ' <span class="muted">(' + esc(s.reason) + ")</span>" : "") +
        (s.attempt != null ? ' <span class="muted small">· program ' + s.attempt + "</span>" : "") +
        (function () { var F = FR[e.id], f = F && F.status === "identical" && F.steps ? F.steps[String(st.k)] : null,
          tt = f && f.after ? f.after.sim_t : null;
          return tt != null ? ' <span class="muted small">· simulated time at the end of this step ' + esc(tt) + " s (replay)</span>" : ""; })() + "</div>" +
        '<p class="small muted">Marks: red, "not found" refused by the guard; amber, help not accepted yet; green, sponge first seen; gray, end of a program. Keys: arrows, space.</p></section>';
      h += '<div class="bottom"><section class="panel">' + attemptsBox(e, curAttempt()) + "</section>" +
        '<section class="panel">' + verdicts(e, st.revealed) + "</section></div>";
      app.innerHTML = h;
    }
    function go(k) { st.k = Math.max(0, Math.min(tl.length - 1, k)); draw(); }
    function toggle() {
      if (timer) { stopTimer(); draw(); return; }
      if (st.k >= tl.length - 1) st.k = 0;
      timer = setInterval(function () { if (st.k >= tl.length - 1) { stopTimer(); draw(); return; } go(st.k + 1); }, 700);
      draw();
    }
    app.onclick = function (ev) {
      var t = ev.target.closest("[data-act]");
      if (!t) return;
      var a = t.getAttribute("data-act");
      if (a === "reveal") { st.revealed = true; draw(); }
      else if (a === "prev") { stopTimer(); go(st.k - 1); }
      else if (a === "next") { stopTimer(); go(st.k + 1); }
      else if (a === "play") toggle();
    };
    app.oninput = function (ev) { if (ev.target.getAttribute("data-act") === "seek") { stopTimer(); go(+ev.target.value); } };
    keyHandler = function (ev) {
      if (/input|textarea|select/i.test(ev.target.tagName) && ev.target.type !== "range") return;
      if (ev.key === "ArrowRight") { ev.preventDefault(); stopTimer(); go(st.k + 1); }
      else if (ev.key === "ArrowLeft") { ev.preventDefault(); stopTimer(); go(st.k - 1); }
      else if (ev.key === " " && ev.target.tagName !== "BUTTON" && ev.target.tagName !== "SUMMARY") { ev.preventDefault(); toggle(); }
    };
    draw();
  }
  var keyHandler = null;
  document.addEventListener("keydown", function (ev) { if (keyHandler) keyHandler(ev); });

  // ------------------------------------------------------------------ the inspection screen of the three situations
  // Four blocks, one per step: inspect, map, choice, execution and verification. The robot's blocks read e.robot, its
  // cost and its identity line only. The simulator's truth, the scene as set up, the zone-by-zone check and any camera
  // image come after Reveal only: in inspectTruth (step 4), and in step 1's player, which then also shows the robot's
  // cameras at its current step (inspectFrame). The reader first sees what the robot knew; the camera images are color
  // renders of the simulation and can show what it had not perceived. A SYNTHETIC fixture (e.synthetic) gets a banner
  // on the screen and a tag on every block.
  var NOT_DONE = [
    "No vision by a model: Nemotron receives text only, no image; no Cosmos-Reason judge.",
    "No map between sessions: the map lives in this episode only; no program library is read or stored.",
    "No cloth lifted: in the masked situation the robot leaves the drawer still to check and asks a person; under an " +
      "object resting on the floor (a bowl, for example), the robot's claim does not cover the floor beneath it.",
    LIMIT_THIN_CLOTH.replace("this scene uses", "the masked situation uses"),
    "No universal mapping: only the zones of the site file (drawers, sink basin, landing tray, drying mat).",
    "No new task: the one checked task, find the sponge, pick it and scrub the bowl on the landing tray.",
    "No validation number from the demo: the demo episodes illustrate; they are not a reliability estimate.",
    "MuJoCo simulation, simulated perception, development kitchens. No physical robot."
  ];
  var NO_CALL = {
    rule: "Only one option was valid (ask for help), so the rule chose, with no call.",
    inspection_rule: "No choice was needed: the inspection program's rule said not found, and the robot's claim " +
      "guard checked it.",
    no_request_2: "The end of the inspection was final: there was no choice to make.",
    no_review: "Request 2 was not the robot's help review: the demo client served its fixed help."
  };
  var STEP_WORDS = { calibrate: "calibrate", look_pose: "look pose", open_drawer: "open", look_at_region: "look at",
    look_down_at: "look down", search_scope: "reads its search", difference_report: "reads its differences with " +
    "the site file", handle_for_region: "finds the handle of", tracks: "reads its tracks", holding: "reads its hand",
    base_xy: "reads its base position", pick: "pick", scrub: "scrub" };
  var CAMERAS = { wrist_cam: "wrist", robot_eye: "mast" };
  var SAID_ID = "i-step-said";  // step 1's line for screen readers, said when the step changes (see inspectScreen)
  // the menu's conditions in reading order (the recorded journal keeps them sorted by name)
  var CONDITION_ORDER = ["scene_state_readable", "hand_holding_no", "contact_incidents_0", "grip_events_0",
    "search_found", "sponge_track_not_held", "no_held_line", "review_note_perceived_sponge"];
  function inOrder(conds) {
    var at = function (c) { var i = CONDITION_ORDER.indexOf(c.condition); return i < 0 ? CONDITION_ORDER.length : i; };
    return conds.slice().sort(function (a, b) { return at(a) - at(b); });
  }
  function stepWords(s) {  // one step in plain words, from the robot's own record (never from its exact args)
    if (s.kind === "observe") return "view " + (s.obs_id || "") + (s.camera ? " (" + (CAMERAS[s.camera] || s.camera) + " camera)" : "");
    var w = STEP_WORDS[s.name] || String(s.name || "").replace(/_/g, " ");
    if (s.region) w += (s.part === "unseen" ? " the unseen part of " : " ") + regionName(s.region);
    return w;
  }
  function synthTag(e) { return e.synthetic ? '<span class="synth-tag">SYNTHETIC</span>' : e.rehearsal ? '<span class="rh-tag">REHEARSAL</span>' : ""; }
  // who acts in each step, on its heading: the team's fixed programs, the robot's own readings, the choice (Nemotron,
  // a rule, a SYNTHETIC stand-in, or awaiting the recorded call), the evaluator after Reveal
  function whoTag(n, e) {
    var tc = e.robot.task_choice || {}, c = choiceOf(e);
    var t = n === 1 ? "fixed program, by the team" : n === 2 ? "the robot's own readings" : n === 4 ?
      "fixed program, by the team; evaluator after Reveal" : c.key === "nemotron" ? "Nemotron, menu of two" :
      c.key === "awaiting" ? "Nemotron: awaiting the recorded call" : c.key === "stand_in" ? "SYNTHETIC stand-in, not Nemotron" :
      c.key === "rule" ? (tc.who_chose === "rule_after_refused_answer" ? "rule, after a refused answer" : "rule, no call") :
      c.key === "doubt" ? "not shown" : "no choice";
    return '<span class="who-tag who-' + (n === 3 ? c.key : n === 2 ? "robot" : "team") + '">' + esc(t) + "</span>";
  }
  function stepHead(n, title, e) {
    return '<div class="step-head"><span class="step-n" aria-hidden="true">' + n + '</span><h2 id="step-' + n +
      '">Step ' + n + ". " + esc(title) + "</h2>" + synthTag(e) + '<span class="sim-tag">simulation</span>' + whoTag(n, e) + "</div>";
  }
  function costLine(e) {  // the episode's cost: the record of its one choice call, as exported
    var c = e.cost || {};
    if (e.synthetic) return esc(us(c.note)) + ".";
    return esc(c.model_calls) + " model call" + (c.model_calls === 1 ? "" : "s") + ", " + esc(c.settled_usd) +
      " USD settled" + (c.uncertain_usd ? ", " + esc(c.uncertain_usd) + " USD uncertain, apart" : "") + ": " +
      esc(us(c.note)) + ".";
  }
  // the call's time: the instant of the call record (the exported cost.call_date), shown only when the choice's own
  // record carries the same instant (the exporter, outside this repository, refuses an episode where they differ);
  // never for a fixture, which made no call. The record's local time and UTC offset are converted to UTC, as in
  // "Oct 20, 2026, 22:14 UTC", seconds left out.
  function callStamp(e) {
    var d = String(((e.cost || {}).call_date) || "");
    var m = /^(\d{4})-(\d\d)-(\d\d)T(\d\d):(\d\d)(?::\d\d(?:\.\d+)?)?(Z|([+-])(\d\d):(\d\d))$/.exec(d);
    if (e.synthetic || !m || d !== String((((e.robot || {}).task_choice) || {}).call_date || "")) return null;
    var off = m[6] === "Z" ? 0 : (m[7] === "-" ? -1 : 1) * (+m[8] * 60 + +m[9]);
    var u = new Date(Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4], +m[5]) - off * 60000);
    function two(x) { return (x < 10 ? "0" : "") + x; }
    return { local: m[3] + "/" + m[2] + "/" + m[1], utc: MONTHS[u.getUTCMonth()] + " " + u.getUTCDate() + ", " +
      u.getUTCFullYear() + ", " + two(u.getUTCHours()) + ":" + two(u.getUTCMinutes()) + " UTC" };
  }
  function callTime(e) {
    var c = callStamp(e);
    return c ? ' <span class="calltime">Call time: ' + c.utc + ", from the call record.</span>" : "";
  }
  function callNote(e) {  // the call note without its local date when the call time (in UTC) follows it
    var n = String(((e.robot || {}).task_choice || {}).call_note || ""), c = callStamp(e);
    if (c) n = n.split("made on " + c.local + " on ").join("made on ");
    return us(n);
  }
  function byZone(objs) {  // "landing tray: bowl, cup; drawer 0: spray bottle"
    var m = {}, order = [];
    objs.forEach(function (o) {
      if (!m[o.region]) { m[o.region] = []; order.push(o.region); }
      var c = String(o.category).replace(/_/g, " ");
      if (m[o.region].indexOf(c) < 0) m[o.region].push(c);
    });
    return order.map(function (z) { return esc(regionName(z)) + ": " + esc(m[z].join(", ")); }).join("; ");
  }

  var LOOK_WORDS = { open_drawer: "opening", look_at_region: "looking at", handle_for_region: "finding the handle of" };
  function lookingAt(tl, k) {  // robot side: the zone its program acts on at step k (a view: the zone it last turned to)
    var now = tl[k] || {};
    for (var j = k; j >= 0; j -= 1) {
      var s = tl[j] || {};
      if (j < k && now.kind !== "observe") break;
      if (s.region && LOOK_WORDS[s.name]) {
        var w = s.name === "look_at_region" && s.part === "unseen" ? "looking at the unseen part of" : LOOK_WORDS[s.name];
        return { regions: [s.region], kind: "look", text: (j < k ? "view " + (now.obs_id || "") + ", " : "") + w + " " +
          regionName(s.region) + (s.ok === false ? " (refused: " + (LOOK_REFUSED[s.reason] || String(s.reason)) + ")" : "") };
      }
    }
    return now.name ? { regions: [], kind: "look", text: stepWords(now) } : null;
  }
  // revealed: after Reveal only, the player also shows the robot's cameras at its current step (inspectFrame); before
  // Reveal it is not called, so nothing of it reaches the page. When a frame is shown, the block's "who" line says so:
  // those images are not the robot's readings
  function inspectStep1(e, k, revealed) {  // robot side: the fixed inspection program, step by step
    var cam = (revealed ? inspectFrame(e, k) : "");
    var r = e.robot, ins = r.inspection, tl = r.timeline, c = ins.counters || {};
    var last = Math.min(ins.frozen_at_step, tl.length) - 1, s = tl[k] || {}, atEnd = k >= last;
    var start = ((r.attempts || [])[0] || {}).host_span ? r.attempts[0].host_span[0] : 0;
    var views = 0, tView = null;
    tl.slice(0, k + 1).forEach(function (x) {
      if (x.kind === "observe") { views += 1; if (x.view_t_s != null) tView = x.view_t_s; }
    });
    var map = {};
    ins.regions.forEach(function (z) {
      var here = atEnd ? z : (s.regions || {})[z.region];
      if (here) map[z.region] = here;
    });
    var regs = ins.regions.map(function (z) { return map[z.region]; }).filter(Boolean);
    var told = {};
    ((r.dossier && r.dossier.stored_items) || []).forEach(function (it) {
      told[it.region] = (told[it.region] ? told[it.region] + ", " : "") + it.category.replace("_", " ");
    });
    var h = stepHead(1, "Inspect the accessible zones", e) +
      '<div class="banner" role="note">' + esc(ins.program_note) + "</div>" +
      '<p class="who">Robot side: its own readings and records, never the truth.' + (cam ? " Under the plan, since Reveal: " +
        "color renders of the simulation from its camera poses, which it does not read." : "") + "</p>" +
      '<div class="counters"><span class="badge">step ' + (k + 1) + " of " + (last + 1) + "</span>" +
      '<span class="badge">views ' + views + " of " + esc(c.max_views) + "</span>" +
      '<span class="badge">simulated time at its last view: ' + (tView == null ? "no view yet" : esc(tView) + " s") +
      " of " + esc(c.max_sim_s) + " s</span></div>" +
      // what a screen reader hears when the step changes, and only that: "step k of N: <words>", written by
      // inspectScreen after the redraw; the screen itself is not announced
      '<p class="sr-only" id="' + SAID_ID + '" aria-live="polite" aria-atomic="true"></p>';
    h += '<div class="tl-row"><button class="btn" data-act="i-prev" aria-label="Previous inspection step">&#9664;</button>' +
      '<button class="btn primary" data-act="i-play" aria-label="Play or pause the inspection">' + (timer ? "Pause" : "Play") +
      "</button>" + '<button class="btn" data-act="i-next" aria-label="Next inspection step">&#9654;</button>' +
      '<input type="range" min="0" max="' + last + '" value="' + k + '" aria-label="Inspection step" data-act="i-seek"></div>';
    h += '<ol class="frieze" aria-label="Inspection steps">' + tl.slice(0, last + 1).map(function (x, j) {
      return '<li><button data-act="i-step" data-k="' + j + '" class="' + (j < start ? "pre" : "") + refusedClass(x) +
        '"' + (j === k ? ' aria-current="step"' : "") + ">" + (j + 1) + " " + esc(stepWords(x)) +
        refusedWords(x, "") + "</button></li>";
    }).join("") + "</ol>";
    if (start > 0) h += '<p class="small muted">Steps 1 to ' + start + ": the loop's opening, before the inspection program.</p>";
    var f = lookingAt(tl, k);
    h += '<div class="step-grid"><div>' + planSVG(r.dossier, map, told, f) +
      '<p class="now small">' + (f ? "<strong>Now:</strong> " + esc(f.text) : "Not looking at a zone yet.") + "</p>" +
      '<div class="cap">Schematic, not to scale; the robot\'s readings drawn on its site file\'s layout. The outline and the ' +
      "dashed line show where it is looking at this step.</div>" + cam + "</div><div>" +
      "<h3>" + (atEnd ? "End of the inspection: its own search reading" : "Zones at this step, from its own readings") +
      "</h3>" + zoneList(regs) + "</div></div>";
    return h;
  }
  // The inspection player's cameras: after Reveal only, the key frame at or before the current step k, the robot's
  // mast and wrist cameras through robotFrame (never the outside view), its caption naming the frame's step and
  // action. Called once, by inspectStep1, behind its "revealed" test: no call, no image, no path before Reveal. Nothing for an entry that is not "identical" or an episode with no frame
  // entry (a rehearsal, a fixture): step 4 gives the reason.
  function inspectFrame(e, k) {
    var F = FR[e.id], tl = e.robot.timeline;
    if (!F || F.status !== "identical") return "";
    return robotFrame(e, k, true, function (j) {
      var x = tl[j] || {};
      return stepWords(x) + (x.ok === false ? " (" + (LOOK_REFUSED[x.reason] || "refused: " + x.reason) + ")" : "");
    });
  }

  function inspectStep2(e) {  // robot side: the map frozen at the end of the inspection
    var r = e.robot, ins = r.inspection, n = ins.frozen_at_step, d = ins.differences || {};
    var banner = ins.pick_step != null ? ins.map_frozen_banner + " The pick is step " + ins.pick_step + "." :
      "Map frozen: " + ins.checked_before + ".";
    var h = stepHead(2, "Show the observations and what remains to check", e) +
      '<div class="banner frozen" role="note"><strong>' + esc(banner) + "</strong></div>" +
      '<p class="who">Robot side: its own readings, never the truth.</p>' +
      zoneList(ins.regions.concat(ins.outside || []), seenByZone((r.timeline[n - 1] || {}).objects)) + legend();
    var objs = ((r.timeline[n - 1] || {}).objects || []).filter(function (o) {
      return o.category !== "drawer_handle" && o.category !== "sponge";
    });
    if (objs.length) h += "<h3>Objects seen, by zone</h3><p class=\"objs\">" + byZone(objs) + "</p>";
    var shown = d.shown || [];
    var diff = shown.filter(function (x) { return x.type !== "confirmed" && x.type !== "module_confirmed"; });
    var same = shown.filter(function (x) { return x.type === "confirmed"; });
    var mods = shown.filter(function (x) { return x.type === "module_confirmed"; });
    h += "<h3>Differences with the site file</h3><ul class=\"lines small\">" + (diff.length ? diff.map(function (x) {
      return "<li><strong>" + esc(String(x.category).replace(/_/g, " ")) + "</strong>: " + esc(x.text) +
        (x.declared_region ? "; site file: " + esc(regionName(x.declared_region)) : "") +
        ((x.observed_in || []).length ? "; seen in " + esc(x.observed_in.map(regionName).join(", ")) : "") +
        (x.observed_region ? "; seen in " + esc(regionName(x.observed_region)) : "") +
        (x.entry ? exactText(x.entry, "its report entry") : "") + "</li>";
    }).join("") : "<li>none in its report</li>") + "</ul>" +
      '<p class="small muted">Its own report counts ' + esc(d.n_differences_or_unknowns) + " difference(s) or unknown(s)" +
      (same.length ? "; seen at the place the site file gives: " + esc(same.map(function (x) { return String(x.category).replace(/_/g, " "); }).join(", ")) : "") +
      (mods.length ? "; modules seen at their slots: " + mods.length : "") + ".</p>";
    h += '<p class="small"><strong>End of the inspection program:</strong> ' + esc(ins.end_text) + ".</p>";
    if (!ins.complete) h += '<p class="small warnc">Its log could not be read in full (' + esc(ins.rejected_log_lines) +
      " line(s) rejected): a zone the page cannot read stays still to check.</p>";
    if ((ins.readings_differ_in || []).length) h += '<p class="small warnc">Its two readings differ for ' +
      esc(ins.readings_differ_in.map(regionName).join(", ")) + ": still to check.</p>";
    return h + '<p class="small muted">Source: ' + esc(ins.source) + ".</p>" + '<p class="small muted">' + esc(ins.footnote) + "</p>";
  }

  function inspectStep3(e) {  // robot side: the choice of a task already available
    var r = e.robot, tc = r.task_choice || {}, syn = !!e.synthetic, doubt = (tc.problems || []).length > 0;
    var h = stepHead(3, "Choose a task already available", e) + "<p><strong>" + esc(tc.who_chose_text) + "</strong></p>" +
      // the menu's rule, only where there was a menu and the model had two valid options to choose from (where the
      // rule chose, the model proposed nothing)
      ((tc.options || []).length && tc.options.every(function (o) { return o.valid; }) ?
        '<p class="small">' + esc(tc.rule_text) + "</p>" : "");
    if ((tc.options || []).length) {
      h += '<ul class="menu" aria-label="Menu of two options">' + tc.options.map(function (o) {
        var chosen = tc.option === o.option;
        return '<li class="' + (chosen ? "chosen" : "") + '"><div><span class="mono">' + esc(o.option) + "</span> " +
          (o.valid ? '<span class="opt valid">valid</span>' : '<span class="opt">not valid</span>') +
          (chosen ? ' <strong class="opt-chosen">chosen</strong>' : "") + '</div><div class="small">' + esc(o.program_text) + "</div></li>";
      }).join("") + "</ul>";
    }
    if ((tc.conditions || []).length) {
      h += "<h3>Conditions, as the code read them in the robot's SCENE STATE text</h3><ul class=\"conds\">" +
        inOrder(tc.conditions).map(function (c) {
          return "<li>" + (c.holds ? '<span class="good">holds</span>' : '<span class="badc">does not hold</span>') +
            ": " + esc(c.condition_text) + "</li>";
        }).join("") + "</ul>";
    }
    if (tc.who_chose === "awaiting_call" && tc.request && !doubt) {  // a rehearsal: the request is written, not sent
      h += '<p class="nocall awaiting"><strong>No call yet: awaiting the recorded Nemotron call.</strong> The request ' +
        "below is written and frozen; it has not been sent. Once the call is made, this step shows Nemotron's answer " +
        "and the code's check, and the task runs.</p>" +
        '<details data-key="s3-request"><summary id="s3-request-sum">The exact text Nemotron will receive (not sent yet): ' +
        "choice_request.json, text only, no images</summary>" +
        '<p class="small sha">sha256 ' + esc(tc.request.request_sha256) + "</p>" +
        (tc.request.messages || []).map(function (m) {
          return exactText(m, m.role === "system" ? "system message" : "user message, with the robot's SCENE STATE as it wrote it");
        }).join("") + "</details>";
    } else if (!tc.request) {
      h += '<p class="nocall"><strong>No model call.</strong> ' + esc(NO_CALL[tc.who_chose] || tc.who_chose_text) + "</p>";
    } else if (doubt) {  // a doubted choice shows no answer, no call note, no call cost; the reader's line below
      h += '<p class="small warnc"><strong>Not shown: the answer, the call note and the cost of this choice call.</strong></p>';
    } else {
      var a = tc.answer || {};
      h += "<h3>" + (syn ? "The fixture's answer (SYNTHETIC, written by hand)" : "Nemotron's answer") + "</h3>" +
        '<p class="small">Option <span class="mono">' + esc(a.option || "none") + "</span>; the code's check: " +
        (a.ok ? '<strong class="good">accepted</strong>, its condition holds' : '<strong class="badc">refused</strong> (' +
          esc(a.refusal) + "): the robot asks for help") + ".</p>" +
        exactText(a.reason, syn ? "its reason, written by hand (SYNTHETIC)" : "Nemotron's reason, as recorded") +
        '<p class="small callnote">' + esc(callNote(e)) + callTime(e) + "</p>" + '<p class="small">Cost: ' + costLine(e) + "</p>" +
        '<details data-key="s3-request"><summary id="s3-request-sum">The exact text ' +
        (syn ? "of the request (SYNTHETIC fixture: never sent)" : "sent to Nemotron") +
        ": choice_request.json, text only, no images</summary>" +
        '<p class="small sha">sha256 ' + esc((tc.request || {}).request_sha256) + "</p>" +
        ((tc.request || {}).messages || []).map(function (m) {
          return exactText(m, m.role === "system" ? "system message" : "user message, with the robot's SCENE STATE as it wrote it");
        }).join("") + exactText(a.answer_text, syn ? "the answer, written by hand (SYNTHETIC)" : "Nemotron's full answer, as recorded") +
        "</details>";
    }
    if (tc.note && !doubt) h += '<p class="small muted">' + esc(tc.note) + "</p>";
    if (doubt) h += '<p class="small warnc">The page\'s reader found: ' + esc(tc.problems.join("; ")) + ".</p>";
    return h;
  }

  function wentPlan(e) {  // robot side: where the arm went after the choice (its pick), or the zones it asks a person to check
    var r = e.robot, p = pickZone(e), told = {};
    var map = {};
    r.inspection.regions.forEach(function (z) { map[z.region] = z; });
    ((r.dossier && r.dossier.stored_items) || []).forEach(function (it) {
      told[it.region] = (told[it.region] ? told[it.region] + ", " : "") + it.category.replace("_", " ");
    });
    var help = asksHelp(e) ? r.inspection.regions.filter(function (z) { return z.state === "to_check"; }).map(function (z) { return z.region; }) : [];
    // a help says what the robot did, and "no arm move" only when no gesture followed the inspection
    var rp = refusedPick(e), ask = "it asks a person" + (help.length ? " to check " + help.map(regionName).join(", ") : "");
    var f = p ? { regions: [p], kind: "pick", head: "Where the arm went:",
      text: "the arm picked the sponge in " + regionName(p) + ", the zone it looked at just before the pick" } :
      asksHelp(e) ? { regions: help, kind: "help", head: "What it did:", text: (rp ? "its pick was refused by its guard (" +
        reasonWords(rp) + "); " : armMoved(e) ? "" : "no arm move; ") + ask } : null;
    if (!f) return "";
    return '<div class="went">' + planSVG(r.dossier, map, told, f) + '<p class="now small"><strong>' + esc(f.head) + "</strong> " +
      esc(f.text) + '.</p><div class="cap">Its map from the end of the inspection, frozen; the outline shows its action.</div></div>';
  }
  function inspectStep4(e, revealed) {  // robot side; the truth side comes from inspectTruth, after Reveal only
    var r = e.robot, ins = r.inspection, n = ins.frozen_at_step, task = r.timeline.slice(n);
    var later = (r.attempts || []).filter(function (a) { return a.n >= 1; });
    var awaiting = (r.task_choice || {}).who_chose === "awaiting_call";
    var h = stepHead(4, "Execute the task", e) + '<div class="step-grid"><div><h3>Robot side</h3>' +
      '<p class="who">Its own records, never the truth.</p>';
    if (awaiting) {
      h += '<p class="nocall awaiting"><strong>No task yet: awaiting the recorded Nemotron call.</strong> The task program ' +
        "(pick the sponge and scrub the bowl, or ask a person) runs after the choice; nothing is shown here until then.</p>";
    } else {
      h += later.length ? '<ol class="attempts">' + later.map(function (a) {
        return "<li><div><strong>Program " + a.n + "</strong> " + esc(a.result) + '</div><div class="o">' + esc(a.origin) + "</div></li>";
      }).join("") + "</ol>" : '<p class="small">No task program: ' + esc(ins.end_text) + ".</p>";
      h += wentPlan(e);
    }
    if (task.length) {
      h += '<ol class="frieze" aria-label="Task steps">' + task.map(function (x, j) {
        var at = n + j + 1;
        return '<li><span class="fz' + refusedClass(x) + (at === ins.pick_step ? " pickstep" : "") + '">' + at + " " +
          esc(stepWords(x)) + (x.ok === false ? refusedWords(x, "refused: ") : x.ok === true && (x.name === "pick" || x.name === "scrub") ?
          " (accepted)" : "") + "</span></li>";
      }).join("") + "</ol>";
    }
    if (ins.pick_step != null) h += '<div class="banner frozen" role="note">' + esc(ins.map_frozen_banner) + "</div>";
    (r.guard || []).forEach(function (g) {
      h += '<p class="small"><strong>Its guard</strong> on "not found": ' + (g.decision === "refused" ?
        '<span class="badc">REFUSED</span>, still to check per its guard: ' + esc((g.not_cleared || []).map(regionName).join(", ")) :
        '<span class="good">accepted</span>') + "</p>";
    });
    if (r.final_claim === "task_done") h += '<p class="small">Its <span class="mono">done()</span>: <span class="good">accepted</span> by its guard (the task evidence).</p>';
    h += '<p class="small">Final: <strong>' + esc(r.final_claim_text) + "</strong></p>";
    if (r.hand_text) h += '<p class="small">At the end' + (awaiting ? " of the inspection" : "") + ", " + esc(r.hand_text) + "</p>";
    var b = r.budget_at_end;
    if (b) h += '<p class="small muted">Its budget at the end: simulated time ' + esc(b.sim_s_used) + " s of " + esc(b.max_sim_s) +
      " s; views " + esc(b.observations) + " of " + esc(b.max_observations) + "; skill calls " + esc(b.skill_calls) + " of " +
      esc(b.max_skill_calls) + ".</p>";
    return h + '</div><div class="truthbox">' + inspectTruth(e, revealed) + "</div></div>";
  }

  // The evaluator box's frame, chosen robot side, from the exported timeline and the robot's own labels at the end of
  // its inspection, never from the simulator's truth: an episode that ends by asking for help shows the last key frame
  // whose step was an accepted look at a zone it left "still to check" (helpViewStep), else, like any other end (a
  // done task), the last key frame.
  var LOOK_SKILLS = ["look_at_region", "look_down_at", "look_under_top"];
  function helpViewStep(e, keys) {  // robot side: keys, the steps that have a key frame; -1 if none fits or no help
    if (!asksHelp(e)) return -1;
    var tl = e.robot.timeline, best = -1;
    var left = e.robot.inspection.regions.filter(function (z) { return z.state === "to_check"; }).map(function (z) { return z.region; });
    keys.forEach(function (i) {
      var x = tl[i] || {};
      if (i > best && x.kind === "skill" && x.ok === true && LOOK_SKILLS.indexOf(x.name) >= 0 && left.indexOf(x.region) >= 0) best = i;
    });
    return best;
  }
  function inspectCameras(e) {  // the robot's own cameras only; called by inspectTruth after Reveal only
    var F = FR[e.id];
    if (!F || F.status !== "identical") {
      return '<div class="noimg">No camera image: ' + esc(e.synthetic ? "SYNTHETIC fixture, nothing was rendered." :
        (F && F.reason) || "not re-simulated.") + "</div>";
    }
    var last = frameStepAt(F, e.robot.timeline.length - 1);
    if (last < 0) return '<div class="noimg">No camera frame recorded for this episode.</div>';
    var v = helpViewStep(e, Object.keys(F.steps || {}).map(Number)), j = v >= 0 ? v : last, s = F.steps[String(j)] || {};
    var fp = F.provenance || {}, a = e.robot.timeline[j] || {};  // the step, the replay, its proof
    // why this frame: said in the caption, with the zone and the robot's own end
    var why = v >= 0 ? "last camera view of a zone left still to check" : "last frame";
    var because = v >= 0 ? "; the robot ended by asking for help, and its own labels left " + esc(regionName(a.region)) +
      " still to check at the end of its inspection" : asksHelp(e) ? "; the robot ended by asking for help, and no key " +
      "frame is a camera view of a zone it left still to check" : "";
    return '<figure class="cam"><div class="camrow"><div><img src="' + esc(s.robot_eye) + '" width="640" height="360" alt="Mast camera, after step ' +
      (j + 1) + '"><span>mast camera</span></div><div><img src="' + esc(s.wrist_cam) + '" width="640" height="360" alt="Wrist camera, after step ' +
      (j + 1) + '"><span>wrist camera</span></div></div><figcaption>Robot cameras, re-rendered from the replay, ' + why + " (after step " +
      (j + 1) + (a.name ? ", " + esc(stepWords(a)) : "") + because + ")" + (fp.code_label ? ", " + esc(codeLabel(fp.code_label).replace(/^the /, "")) : "") +
      "; shown because the replay matched the recorded episode (proof: " + esc(fp.proof_status) + "). Color render of " +
      "the simulation, shown after Reveal; the robot reads simulated depth and labels, not this image.</figcaption></figure>";
  }

  function inspectTruth(e, revealed) {  // evaluator side: nothing of it before Reveal
    var h = '<h3 id="truth-h" tabindex="-1">Evaluator (simulator truth)</h3><p class="who">Evaluator view, never given to the robot.</p>';
    if (!revealed) {
      return h + '<p class="small muted">The simulator\'s truth, the scene as set up, the zone-by-zone check and any camera ' +
        'image appear after Reveal.</p><button class="reveal" data-act="reveal">Reveal the truth</button>';
    }
    var v = e.evaluator, zc = v.zone_check || {}, c = v.contacts_over_20n || {};
    if (e.rehearsal) h += '<p class="small"><span class="rh-tag">REHEARSAL</span> The evaluator of a rehearsal: a check, not a result.</p>';
    h += '<p class="big ' + outcomeClass(v) + '">' + esc(helpWords(v, situationOf(e)) || outcomeWords(v)) + "</p>";
    h += '<dl class="kv"><dt>Sponge</dt><dd>' + (v.sponge_present === false ? "<strong>not in the kitchen</strong>" :
      "started in <strong>" + esc(regionName(v.sponge_truth_region)) + "</strong>") + "</dd>" +
      "<dt>Scene</dt><dd>" + esc(titleOf(e)) + ". " + esc(us(e.story)) + "</dd>" +
      "<dt>Changes since the site file</dt><dd>" + ((v.site_file_changes || []).map(function (x) { return esc(describeChange(x, true)); }).join("; ") || "none") + "</dd></dl>";
    var cr = v.criteria || {}, keys = Object.keys(cr);
    if (v.outcome_class === "success") h += '<p class="small">Task criteria met: ' + keys.filter(function (k) { return cr[k]; }).length +
      " of " + keys.length + (v.scrub ? "; scrub " + esc(v.scrub.duration_s) + " s at " + esc(v.scrub.force_mean_n) + " N mean" : "") + "</p>";
    if (v.outcome_class !== "awaiting_call") {  // a rehearsal waiting for its call has no verdict: the task has not run
      h += '<p class="small">False "not found": ' + (v.false_absence ? '<strong class="badc">yes</strong>' : "no") + " · False success: " +
        (v.false_success ? '<strong class="badc">yes</strong>' : "no") + (v.help_with_sponge_present ? " · Help with the sponge present" : "") + "</p>";
    }
    h += '<p class="small">Hand contacts over 20 N, counted: ' + esc(c.counted) + (c.calibration_exempt ? "; calibration contacts, exempt by the bench rule, shown: " +
      esc(c.calibration_exempt) + " (peak " + esc(c.calibration_peak_n) + " N)" : "") + "</p>";
    h += "<h3>Zone by zone, the robot's map against where the sponge started</h3>";
    if (zc.rule_text) h += '<p class="small muted">' + esc(zc.rule_text) + "</p>";
    if (zc.valid) {
      h += '<ul class="regions zcheck">' + (zc.zones || []).map(function (z) {
        return '<li class="' + (z.error ? "zc-error" : "") + '"><span class="rname">' + esc(regionName(z.region)) + '</span><span class="rv">' +
          chip(z.state) + "</span><span class=\"gloss\">" + (z.error ? "<strong>FALSE NOT FOUND.</strong> " : "") + esc(z.check_text) +
          (z.state === "to_check" ? "; how to check: step 2" : "") + "</span></li>";  // each still to check says how
      }).join("") + "</ul>";
    } else {
      h += '<p class="small warnc">' + esc(zc.why_not || "No zone-by-zone check for this episode.") + "</p>";
    }
    h += inspectCameras(e);
    var idt = e.demo_identity;
    if (idt) {
      h += '<details data-key="s4-identity"><summary class="small" id="s4-identity-sum">Identity and files read (for traceability)</summary><p class="small sha">programs: ' +
        esc(Object.keys(idt.programs || {}).map(function (k) { return k + " " + String(idt.programs[k]).slice(0, 12); }).join(", ")) +
        "; client " + esc(idt.client) + "; choice text " + esc(idt.choice_prompt) + "; robot code tree " + esc(String(idt.robot_tree_sha256).slice(0, 16)) +
        (idt.world_dev ? "; development world " + esc(idt.world_dev) : "") + '</p><ul class="lines mono">' +
        (e.provenance.files || []).map(function (f) { return "<li>[" + esc(f.side) + "] " + esc(f.file) + " " + esc(String(f.sha256).slice(0, 16)) + "</li>"; }).join("") +
        "</ul></details>";
    }
    return h;
  }

  // ------------------------------------------------------- the robot's report, checked against the evaluator's records
  // The idea: a person should not need to watch the robot; what it says it did, what it says it checked and what it
  // says it did not look at should be checkable against records. On the screen of a recorded episode of the three
  // situations, before Reveal: the robot's report, its own claims only, in plain sentences this page writes from its
  // exported records (reportClaims: its only argument is e.robot, so it cannot reach the evaluator). After Reveal
  // only: each claim marked against the simulator's truth (markClaim: its only arguments are one claim and
  // e.evaluator, never the robot's records), "true", "false", or "not checkable" when the evaluator's records do not
  // cover it, and a one-line score; on the home, the same score in the strip's evaluator row (stripTruth, after its
  // guard). Same separation as helpViewStep and inspectTruth. A claim type with no robot record is left out: no
  // sentence is written by hand. A zone left still to check makes no claim about the sponge: "not checkable"; after
  // Reveal the page says whether the sponge was there and why it was left unchecked, in the robot's own reading. A
  // rehearsal and a SYNTHETIC fixture get no report: they are not recorded robot runs.
  // The marking rules. One claim, one assertion: what its records add under a claim (its guard's acceptances, view
  // numbers, coverage, where an item rests) is shown as its details, never covered by a mark. A mark is "true" only on
  // a positive record of the evaluator for that exact assertion, "false" only on a record that contradicts it, and
  // "not checkable" otherwise: never true by default, never true because no contrary record exists. Why: a missing
  // record proves nothing, so a claim with no record that shows it must not count as true. Four rules follow:
  // (1) an object's place is never proven by the change log having no entry for it (no inventory of objects is
  // exported); (2) no false "not found" in the evaluator's flags is not "it did not conclude not found" nor "it asked a
  // person": each is its own claim, and the flags are computed from the bench's copy of its final claim; (3) criteria
  // not met measure "no scrub that meets the criteria", never "no scrub"; (4) the sponge where it saw one does not show
  // the sequence "a help reopened, then the task went on": its own log's account, not checkable.
  var REPORT_GROUPS = [["did", "What it did"], ["found", "What it checked and found"],
    ["unchecked", "What it did not check"], ["held", "What it refused or held back"]];
  var MARK_WORDS = { "true": "true", "false": "false", nc: "not checkable" };
  var REVIEW_PERCEIVED = "the robot perceives a sponge it has not picked";  // its help review's own words (export)
  var OBJECT_DIFF = { confirmed: 1, not_seen_at_declared_location: 1, observed_not_declared: 1 };  // modules: left out
  function listWords(xs) { return xs.length < 2 ? xs.join("") : xs.slice(0, -1).join(", ") + " and " + xs[xs.length - 1]; }
  function catWords(c) { var w = String(c || "").replace(/_/g, " "); return w.charAt(0).toUpperCase() + w.slice(1); }
  function reportOf(e) {  // robot side: the claims of a recorded episode of the three situations, else null
    return e && !e.rehearsal && !e.synthetic && situationOf(e) && e.robot && e.robot.inspection ? reportClaims(e.robot) : null;
  }
  function reportClaims(r) {  // robot side: r is e.robot, its exported records; nothing else is in reach
    var e = { robot: r }, ins = r.inspection, n = frozenAt(e), insp = r.timeline.slice(0, n), out = [];
    // what it did: the inspection, then the task's accepted steps and its end, from its own records
    var c = ins.counters || {}, views = c.views_in_inspection != null ? c.views_in_inspection :
      insp.filter(function (s) { return s.kind === "observe"; }).length;
    var opened = insp.filter(function (s) { return s.name === "open_drawer" && s.ok === true; }).map(function (s) { return regionName(s.region); });
    var again = insp.filter(function (s) { return s.name === "look_at_region" && s.part === "unseen" && s.ok === true; })
      .map(function (s) { return regionName(s.region); });
    // its refused looks again too, in the words of the home strip, so that the claim does not leave them out
    var refusedLooks = insp.filter(function (s) { return s.name === "look_at_region" && s.part === "unseen" && s.ok === false; })
      .map(function (s) { return regionName(s.region) + " (" + (LOOK_REFUSED[s.reason] || "refused: " + s.reason) + ")"; });
    // one claim (it ran the team's program); its steps are its details, not separate claims (one claim, one assertion)
    out.push({ group: "did", kind: "inspected", text: "It ran the team's inspection program.", rec: (opened.length ?
      "it opened " + listWords(opened) + ", took " : "it took ") + views + " view" + (views === 1 ? "" : "s") + (again.length ?
      " and looked again at the unseen part of " + listWords(again) : "") + (refusedLooks.length === 1 ?
      "; its look again at the unseen part of " + refusedLooks[0] + " was refused" : refusedLooks.length ?
      "; its looks again at the unseen part of " + listWords(refusedLooks) + " were refused" : "") });
    var p = pickZone(e);
    if (p) out.push({ group: "did", kind: "pick", region: p, text: "It picked up the sponge in " + regionName(p) + ".",
      rec: "its guard accepted the pick" });
    // rule 3: the claim says what the bench's criteria measure, a scrub that meets them
    if (scrubbed(e)) out.push({ group: "did", kind: "scrub", text: "It scrubbed the bowl, a scrub that meets the task's contact and stroke criteria.",
      rec: "on the landing tray; its scrub step was accepted" });
    if (r.final_claim === "task_done") out.push({ group: "did", kind: "done", text: "It reported the task done.",
      rec: "its guard accepted its done()" });
    if (asksHelp(e)) {
      var m = /regions_still_to_check_(\d+)/.exec(String(r.final_claim)), k = m ? +m[1] : null;
      var left = ((r.final_scope || {}).regions || []).filter(function (z) { return z.state === "to_check"; }).map(function (z) { return regionName(z.region); });
      out.push({ group: "did", kind: "help", count: k, text: "It stopped and asked a person" + (k == null ? "." : ", listing " + k +
        " zone" + (k === 1 ? "" : "s") + " still to check" + (k && left.length === k ? ": " + listWords(left) : "") + ".") });
    }
    // rule 3: with no accepted scrub step, the claim is what the criteria can measure: no scrub that meets them (a
    // short contact can fail them). "It did not report the task done" is not a second claim: its end is
    // one record, said once (its help, or its done)
    if (!scrubbed(e) && r.final_claim !== "task_done") {
      out.push({ group: "did", kind: "no_scrub", text: "It made no scrub of the bowl that meets the task's contact and stroke criteria.",
        rec: "no accepted scrub step in its records" });
    }
    // what it checked and found: its map at the end of the inspection (two of the three labels), then the objects its
    // difference report placed against its site file (the sponge apart: its zones say it; modules left out). A zone's
    // claim is its label (the chip); its view, coverage and what rests there are its details
    var fileAt = {};
    ((r.dossier && r.dossier.stored_items) || []).forEach(function (it) { if (!fileAt[it.category]) fileAt[it.category] = it.region; });
    ins.regions.forEach(function (z) {
      if (z.state !== "seen" && z.state !== "not_found") return;
      out.push({ group: "found", kind: "zone", region: z.region, state: z.state, text: "", rec: z.precision +
        (z.state === "seen" && fileAt.sponge && fileAt.sponge !== z.region ? "; its site file gives " + regionName(fileAt.sponge) : "") });
    });
    var objs = {}, order = [];
    (((ins.differences || {}).shown) || []).forEach(function (x) {
      if (x.category === "sponge" || !OBJECT_DIFF[x.type]) return;
      var o = objs[x.category] || (objs[x.category] = { declared: null, notAt: false, confirmed: false, seenIn: [] });
      if (order.indexOf(x.category) < 0) order.push(x.category);
      if (x.type === "confirmed") { o.confirmed = true; o.declared = x.declared_region; }
      if (x.type === "not_seen_at_declared_location") { o.notAt = true; o.declared = x.declared_region; }
      (x.observed_in || []).concat(x.observed_region ? [x.observed_region] : []).forEach(function (g) { if (o.seenIn.indexOf(g) < 0) o.seenIn.push(g); });
    });
    order.forEach(function (cat) {
      var o = objs[cat], w = catWords(cat), seen = listWords(o.seenIn.map(regionName)), d = regionName(o.declared);
      var kind = o.notAt ? (o.seenIn.length ? "object_moved" : "object_not_seen") : o.confirmed ? "object_in_place" : "object_new";
      out.push({ group: "found", kind: kind, category: cat, seenIn: o.seenIn.slice(), text: kind === "object_in_place" ?
        w + ": seen where its site file gives it, " + d + "." : kind === "object_moved" ? w + ": not seen where its site file " +
        "gives it (" + d + "); seen in " + seen + "." : kind === "object_not_seen" ? w + ": not seen where its site file gives " +
        "it (" + d + "), nor in its other views." : w + ": seen in " + seen + ", a place its site file does not give." });
    });
    // what it did not check: the zones of its search still to check, then those outside its search
    var there = seenByZone((r.timeline[n - 1] || {}).objects);
    ins.regions.concat(ins.outside || []).forEach(function (z) {
      if (z.state !== "to_check") return;
      var cov = z.coverage != null && String(z.precision || "").indexOf("coverage") < 0, saw = there[z.region] || [];
      out.push({ group: "unchecked", kind: "zone", region: z.region, state: "to_check", outside: ins.regions.indexOf(z) < 0,
        precision: z.precision, coverage: z.coverage, text: "", rec: z.precision + (cov ? "; coverage " + z.coverage : "") +
        (saw.length ? "; seen in this zone: " + listWords(saw) : "") + (z.how ? ". How to check: " + z.how : "") });
    });
    // what it refused or held back, from its own records: its help review reopened a help while it perceived a sponge;
    // no "sponge not found" concluded while zones of its search stayed still to check
    var seenZ = ins.regions.filter(function (z) { return z.state === "seen"; }).map(function (z) { return z.region; });
    if ((r.attempts || []).some(function (a) { return a.status === "help_review" && String(a.result).indexOf(REVIEW_PERCEIVED) >= 0; })) {
      out.push({ group: "held", kind: "kept_going", seenIn: seenZ, text: "It did not stop with a help while it saw a sponge it " +
        "had not picked" + (seenZ.length ? " (in " + listWords(seenZ.map(regionName)) + ")" : "") + ": its help review reopened " +
        "the inspection program's help." });
    }
    var open = ins.regions.filter(function (z) { return z.state === "to_check"; }).map(function (z) { return regionName(z.region); });
    var concluded = (r.guard || []).some(function (g) { return g.decision !== "refused"; }) || /not_found/.test(String(r.final_claim));
    // rule 2: one assertion; its help is its own claim (group "did"), its zones still to check theirs
    if (!seenZ.length && open.length && !concluded) {
      out.push({ group: "held", kind: "no_not_found", text: "It did not conclude \"sponge not found\".", rec: open.length +
        " zone" + (open.length === 1 ? " of its search was" : "s of its search were") + " still to check: " + listWords(open) });
    }
    return out;
  }
  // the evaluator's zone check, verdict by verdict (the verdict names of the exported zone check): the mark and why
  var ZONE_MARK = { confirmed: ["true", "The sponge started here."], correct: ["true", "The sponge did not start here."],
    correct_no_sponge: ["true", "There was no sponge in the kitchen."],
    false_not_found: ["false", "The sponge started here: a false \"not found\"."],
    not_confirmed: ["false", "The sponge did not start here."], not_confirmed_no_sponge: ["false", "There was no sponge in the kitchen."],
    claimed_nothing_here: ["nc", "the sponge started here"], claimed_nothing_elsewhere: ["nc", "the sponge did not start here"],
    claimed_nothing_no_sponge: ["nc", "there was no sponge in the kitchen"] };
  function markClaim(c, v) {  // evaluator side: one claim (from reportClaims) against v, the evaluator side; nothing else
    // true only on a positive record of the evaluator for this exact assertion, false only on a record that
    // contradicts it, "not checkable" otherwise: a missing field or a log with no entry is never a proof
    var zc = v.zone_check || {}, cr = v.criteria || {}, truth = v.sponge_truth_region, none = v.sponge_present === false;
    var present = v.sponge_present === true, checked = zc.valid === true;
    function M(mark, why, more) { return { mark: mark, why: why, more: more || "" }; }
    function crit() {  // the two scrub criteria as the evaluator records them, or "missing"
      function w(x) { return x === true || x === false ? String(x) : "missing"; }
      return "its contact criterion: " + w(cr.target_contact) + "; its stroke criterion: " + w(cr.scrub_motion);
    }
    if (c.kind === "inspected") return M("nc", "The evaluator's records on this page do not cover the robot's views or the drawers it opened.");
    if (c.kind === "pick") {
      // true: the sponge started in that zone (a valid zone check: nothing moved it before its view of the pick) and the
      // evaluator measured it in the gripper
      var held = cr.grip_during_scrub === true || cr.no_drop === true;
      if (none) return M("false", "There was no sponge in the kitchen.");
      if (!held) return M("nc", "The evaluator's records do not show the sponge in the gripper where the robot says it picked it.");
      if (!checked || !present || !truth) return M("nc", zc.why_not || "No zone-by-zone check for this episode: where the " +
        "sponge started does not show where it was picked.");
      if (truth === c.region) return M("true", "The sponge started in " + regionName(truth) + ", and the evaluator measured it " +
        "in the gripper " + (cr.grip_during_scrub === true ? "during the scrub." : "until the end."));
      return M("false", "The sponge started in " + regionName(truth) + ", not in " + regionName(c.region) + ".");
    }
    if (c.kind === "scrub") {  // rule 3: the claim is a scrub that meets the criteria
      if (cr.scrub_motion === true && cr.target_contact === true) return M("true", "The evaluator measured a scrub of the " +
        "bowl that meets its contact and stroke criteria" + (v.scrub ? ": " + v.scrub.duration_s + " s at " + v.scrub.force_mean_n +
        " N mean." : "."));
      if (cr.scrub_motion === false || cr.target_contact === false) return M("false", "The evaluator measured no scrub of the " +
        "bowl that meets its criteria (" + crit() + ").");
      return M("nc", "The evaluator's scrub criteria are missing (" + crit() + ").");
    }
    if (c.kind === "no_scrub") {  // rule 3: criteria not met measure "no scrub that meets them", never "no scrub"
      if (cr.scrub_motion === true && cr.target_contact === true) return M("false", "The evaluator measured a scrub of the " +
        "bowl that meets its contact and stroke criteria.");
      if (cr.scrub_motion === false || cr.target_contact === false) return M("true", "The evaluator measured no scrub of the " +
        "bowl that meets its criteria (" + crit() + ")." + (v.scrub ? " It did measure the sponge on the bowl for " +
        v.scrub.duration_s + " s at " + v.scrub.force_mean_n + " N mean, short of them." : ""));
      return M("nc", "The evaluator's scrub criteria are missing (" + crit() + ").");
    }
    if (c.kind === "done") {  // true: the evaluator's verdict is a success with every task criterion recorded true
      var ks = Object.keys(cr), met = ks.filter(function (x) { return cr[x] === true; }).length;
      var off = ks.filter(function (x) { return cr[x] === false; }).length;
      if (v.false_success === true) return M("false", "The evaluator's verdict: a false success claim.");
      if (off) return M("false", "The evaluator records " + off + " task criteri" + (off === 1 ? "on" : "a") + " false: " + met +
        " of " + ks.length + " met, not a success.");
      if (typeof v.outcome_class === "string" && v.outcome_class !== "success") return M("false", "The evaluator's verdict: " +
        String(v.outcome_text || v.outcome_class) + ", not a success.");
      if (v.outcome_class === "success" && ks.length && met === ks.length) return M("true", "The evaluator's verdict: success, " +
        "task criteria " + met + " of " + ks.length + (v.false_success === false ? "; no false success." : "."));
      return M("nc", "No verdict of the evaluator on the task.");
    }
    if (c.kind === "help") {
      // bench_detail is the bench's copy of the robot's own final claim (the bench copies it; it does not observe the
      // end on its own), not an independent record of the end. The same end is "not checkable"; a copy that disagrees
      // with the robot's records is shown as a problem, "false", never hidden.
      var hm = /^help_requested(?::regions_still_to_check_(\d+))?/.exec(String(v.bench_detail || ""));
      var copyNote = ": its copy of the robot's final claim disagrees with the robot's records.";
      if (!hm) return v.bench_detail ? M("false", "The bench recorded another end, not a help request" + copyNote) :
        M("nc", "The evaluator recorded no detail of the end.");
      if (c.count != null && hm[1] != null && +hm[1] !== c.count) return M("false", "The bench recorded a help request with " +
        hm[1] + " zone(s) still to check" + copyNote);
      return M("nc", "The bench recorded the same end, a help request" + (hm[1] != null ? " with " + hm[1] + " zone" +
        (+hm[1] === 1 ? "" : "s") + " still to check" : "") + "; it copies the robot's own final claim, so this is not an " +
        "independent check.");
    }
    if (c.kind === "zone") {
      if (!checked) return M("nc", zc.why_not || "No zone-by-zone check for this episode.");
      var z = (zc.zones || []).filter(function (x) { return x.region === c.region; })[0], zm = z && ZONE_MARK[z.verdict];
      if (!z || z.state !== c.state) {
        // the robot's records and the evaluator's copy of its map disagree. Shown as a problem, "false", never "not
        // checkable"; the sponge's starting place is said when it decides the claim on its own.
        var copy = "The robot's records and the evaluator's copy of its map disagree on this zone (" + (z ? "the copy holds \"" +
          (LBL[z.state] || z.state) + "\"" : "the copy has no entry for it") + ").";
        if (c.state === "not_found" && !none && truth === c.region) return M("false", "The sponge started here: a false " +
          "\"not found\". " + copy);
        if (c.state === "seen" && (none || (truth && truth !== c.region))) return M("false", (none ? "There was no sponge " +
          "in the kitchen. " : "The sponge did not start here. ") + copy);
        return M("false", copy);
      }
      if (!zm) return M("nc", "The evaluator's zone check gives a verdict this page does not know: " + z.verdict + ".");
      if (zm[0] === "true" && z.error !== false) return z.error === true ? M("false", "The evaluator's zone check flags an " +
        "error on this zone (verdict: " + z.verdict + ").") : M("nc", "The evaluator's zone check has no error flag for this zone.");
      if (zm[0] !== "nc") return M(zm[0], zm[1]);
      var here = z.verdict === "claimed_nothing_here", read = "its own reading said \"" + c.precision + "\"" +
        (c.coverage != null && String(c.precision).indexOf("coverage") < 0 ? " (coverage " + c.coverage + ")" : "");
      return M("nc", "It makes no claim about the sponge here.", "In truth: " + zm[1] + ". Why it stayed still to check: " +
        (c.outside ? "this zone is outside its search, so it made no claim about it and leaves it to a person" + (here ?
          "; the sponge was there, and only a person can check it." : ".") : read + ", so it made no claim about this zone" + (here ?
          "; the sponge was there, so a \"not found\" here would have been false." : z.verdict === "claimed_nothing_no_sponge" ?
          "." : "; the sponge was elsewhere, but its views alone could not rule one out here.")));
    }
    if (/^object_/.test(c.kind)) {
      // rule 1: the evaluator exports no inventory of objects, only its log of changes since the site file. An entry
      // that places the object is a positive record; no entry proves neither where it is nor that it is there (a
      // changed site file entry, or an object that does not exist, has no entry either)
      var log = v.site_file_changes;
      if (!Array.isArray(log)) return M("nc", "No change log from the evaluator.");
      var about = log.filter(function (x) { return x.item === c.category || (x.change === "world_dev" && x.towel && c.category === "towel"); });
      var at = about.map(function (x) { return x.change === "world_dev" ? placeName(x.drawer) : x.change === "removed" ? null : placeName(x.to); });
      var said = "The evaluator's log of changes since the site file: " + about.map(function (x) { return describeChange(x, true); }).join("; ") + ".";
      var w = String(c.category).replace(/_/g, " "), match = c.seenIn.length > 0 && c.seenIn.every(function (g) { return at.indexOf(regionName(g)) >= 0; });
      // an entry that removes it or places it in a zone where it was not seen contradicts the claim, even beside one
      // that matches
      var seenW = c.seenIn.map(regionName), off = at.some(function (x) { return x === null || (x && seenW.indexOf(x) < 0); });
      var noEntry = "The evaluator's log of changes since the site file has no entry for the " + w + ". That shows neither " +
        "that it is there nor where: the evaluator's records hold no inventory of objects.";
      if (!about.length && c.kind !== "object_not_seen") return M("nc", noEntry);
      if (c.kind === "object_in_place") return M("false", said);
      if (c.kind === "object_moved" || c.kind === "object_new") return M(match && !off ? "true" : "false", said);
      return M("nc", "The evaluator does not record what the robot saw.", "In truth: " + (about.length ? said :
        "the evaluator's log of changes since the site file has no entry for the " + w + "."));
    }
    if (c.kind === "kept_going") {
      // rule 4: the sequence (its help review reopened a help, then the task went on) is its own log's account; where
      // the sponge started is its zone claims' fact, not this sequence. False only when the evaluator's records
      // contradict what it says it saw
      if (none) return M("false", "There was no sponge in the kitchen.");
      if (checked && truth && c.seenIn.indexOf(truth) < 0) return M("false", "The sponge started in " + regionName(truth) +
        ", not where it saw one.");
      return M("nc", "Its own log's account: the evaluator's records do not cover its help reviews. Where the sponge started " +
        "is checked by its zone claims; that does not show this sequence (a help reopened, then the task went on).");
    }
    if (c.kind === "no_not_found") {
      // rule 2: the evaluator's only record of what it concluded is the bench's copy of its own final claim, and its
      // false and unsupported "not found" flags are computed from that copy: never a proof that it did not conclude
      // "not found". A copy or a flag that says it did is a contradiction
      if (/not_found/.test(String(v.bench_detail || ""))) return M("false", "The bench recorded a \"not found\" end: its copy " +
        "of the robot's final claim disagrees with the robot's records.");
      if (v.outcome_class === "absence_established" || v.outcome_class === "false_absence_accepted") return M("false",
        "The evaluator's verdict, " + String(v.outcome_text || v.outcome_class) + ", is a \"not found\" end: it disagrees " +
        "with the robot's records.");
      if (v.false_absence === true || v.unsupported_absence === true) return M("false", "The evaluator's records hold " +
        (v.false_absence === true ? "a false" : "an unsupported") + " \"not found\".");
      return M("nc", "The evaluator has no record of what it concluded apart from the bench's copy of its own final claim; " +
        "its false and unsupported \"not found\" flags" + (v.false_absence === false && v.unsupported_absence === false ? "" :
        " (missing here)") + " are computed from that copy, so they do not show what it did not conclude.");
    }
    return M("nc", "No evaluator record for this claim.");
  }
  function reportScore(mk) {  // "13 claims, 10 true, 0 false, 3 not checkable."
    var n = { "true": 0, "false": 0, nc: 0 };
    mk.forEach(function (m) { n[m.mark] += 1; });
    return mk.length + " claim" + (mk.length === 1 ? "" : "s") + ", " + n["true"] + " true, " + n["false"] + " false" +
      (n.nc ? ", " + n.nc + " not checkable" : "") + ".";
  }
  // the counts behind the score lines, and their wording: a detection among the checkable claims of these simulated
  // episodes, with the claims not checkable said beside it; computed from the marks (a false mark changes the
  // sentence), shown after Reveal only by its callers
  function newTally() { return { episodes: 0, n: 0, "true": 0, "false": 0, nc: 0 }; }
  function addMarks(t, mk) { t.episodes += 1; mk.forEach(function (m) { t.n += 1; t[m.mark] += 1; }); return t; }
  function falseLine(t, scope) {
    var f = t["false"];
    return (f ? f + " false claim" + (f === 1 ? "" : "s") + " detected" : "No false claim detected") +
      " among the checkable claims of " + scope + "; " + t.nc + " of " + t.n + " claims not checkable.";
  }
  function reportTotal(t) {  // the home, under the three columns, after Reveal (situationsStrip)
    return '<p class="report-total"><strong>The three reports together, checked against the evaluator\'s records:</strong> ' +
      t.n + " claims, " + t["true"] + " true, " + t["false"] + " false, " + t.nc + " not checkable. " +
      esc(falseLine(t, "these three simulated episodes")) + " The counts are not a success rate.</p>";
  }
  function claimList(cl, mk) {  // the claims by group; mk (after Reveal only): their marks, in the same order
    return REPORT_GROUPS.map(function (g) {
      var idx = [];
      cl.forEach(function (c, i) { if (c.group === g[0]) idx.push(i); });
      if (!idx.length) return "";
      return '<div class="report-group"><h3>' + esc(g[1]) + '</h3><ul class="claims">' + idx.map(function (i) {
        var c = cl[i], m = mk ? mk[i] : null;
        // the claim (a zone's: its name and label), then its details from its records, which no mark covers
        return '<li class="claim">' + (c.kind === "zone" ? '<span class="rname">' + esc(regionName(c.region)) + "</span> " +
          chip(c.state) : esc(c.text)) + (c.rec ? ' <span class="crec">Its details, not checked: ' + esc(c.rec) + ".</span>" : "") +
          (m ? '<div class="cmark"><span class="vmark m-' + m.mark + '">' +
          MARK_WORDS[m.mark] + "</span> " + esc(m.why) + (m.more ? '<span class="ctruth">' + esc(m.more) + "</span>" : "") +
          "</div>" : "") + "</li>";
      }).join("") + "</ul></div>";
    }).join("");
  }
  function reportBlock(e, revealed) {  // robot side, then the evaluator's marks after Reveal only
    var cl = reportOf(e);
    if (!cl) return "";
    var h = '<section class="panel report" aria-labelledby="report-h"><div class="step-head"><h2 id="report-h" tabindex="-1">' +
      (revealed ? "The robot's report, checked against the evaluator's records" : "The robot's report") +
      '</h2><span class="sim-tag">simulation</span></div>' +
      '<p class="who">Its own claims only: sentences this page writes from its records (its steps, its zone labels, its ' +
      "difference report, its final claim), nothing from the evaluator. One claim, one assertion; the details from its " +
      "records under a claim are not checked.";
    if (!revealed) {
      return h + " After Reveal, each claim is checked against the simulator's truth.</p>" + claimList(cl, null) +
        '<button class="reveal" data-act="reveal" data-focus="report-h">Reveal and check its claims</button></section>';
    }
    var v = e.evaluator, mk = cl.map(function (c) { return markClaim(c, v); });
    return h + " Since Reveal, each claim is marked against the simulator's truth: true only on a record of the evaluator " +
      "that shows it, false on one that contradicts it, not checkable when none does. The count is not a success rate.</p>" +
      '<p class="report-score">' + esc(reportScore(mk)) + "</p>" +
      '<p class="report-claim">' + esc(falseLine(addMarks(newTally(), mk), "this simulated episode")) + "</p>" +
      claimList(cl, mk) + "</section>";
  }
  // after the report, the next situation in reading order (1, 2, 3), then back to the home
  function nextSituation(sit) {
    var j = SITUATIONS.indexOf(sit), nx = SITUATIONS[j + 1], m = nx ? materialFor(nx) : null;
    return '<p class="sit-next">' + (m ? '<a href="#/inspect/' + esc(m.id) + '">Next: situation ' + (j + 2) + " of " +
      SITUATIONS.length + ", " + esc(nx.name.toLowerCase()) + "</a>" : '<a href="#/">Back to the three situations</a>') + "</p>";
  }

  function situationTabs(cur) {  // the three situations, side by side; one that has no episode yet does not open
    return '<nav class="sit-tabs" aria-label="The three situations">' + SITUATIONS.map(function (s, j) {
      var m = materialFor(s), w = (j + 1) + ". " + esc(s.name);  // numbered in reading order
      return m ? '<a href="#/inspect/' + esc(m.id) + '"' + (s === cur ? ' aria-current="page"' : "") + ">" + w + "</a>" :
        '<span class="off" aria-disabled="true">' + w + "</span>";
    }).join("") + "</nav>";
  }
  // Who decided what in this episode, robot side: the team's fixed programs, the code's checks, the choice
  // (Nemotron's recorded answer, a rule, a SYNTHETIC stand-in, or awaiting the recorded call), and the evaluator,
  // after Reveal only. Nemotron's reason is its own text, in a marked exact-text block.
  // Where the arm went, by who chose, robot side: the sponge track is named only when the arm went for the sponge (a
  // pick, accepted or refused, in its own records) or will once Nemotron has chosen (a rehearsal awaiting its call),
  // and "not Nemotron" only when Nemotron (or a fixture's stand-in) chose or will; with no pick (the masked situation:
  // the rule chose to ask a person) it says so, with no word about Nemotron.
  // revealed: after Reveal, the evaluator card also says where any camera image sits (step 4, and step 1's player)
  // and that it is a color render, not the robot's reading; before Reveal it says neither
  var NO_PICK = {
    rule: "No pick in this episode: the rule chose to ask a person, so the arm did not go for the sponge.",
    rule_after_refused_answer: "No pick in this episode: the code refused the answer, so the rule chose to ask a person.",
    inspection_rule: "No pick in this episode: there was no task to choose after the inspection.",
    nemotron: "No pick in this episode: the option chosen was to ask a person."
  };
  function armLine(e, c) {
    var tc = e.robot.task_choice || {};
    if (pickZone(e) || refusedPick(e) || c.key === "awaiting") {
      return "Where the arm picks follows the sponge track the robot itself perceived" +
        (c.key === "nemotron" || c.key === "stand_in" || c.key === "awaiting" ? ", not Nemotron." : ".");
    }
    return NO_PICK[tc.who_chose] || "No pick in this episode.";
  }
  function whoDecides(e, revealed) {
    var r = e.robot, tc = r.task_choice || {}, c = choiceOf(e), a = tc.answer || {};
    var conds = tc.conditions || [], held = conds.filter(function (x) { return x.holds; }).length;
    var ran = (r.attempts || []).filter(function (x) { return x.n >= 1; }).map(function (x) { return x.origin; });
    var sha = String((tc.request || {}).request_sha256 || "").slice(0, 12);
    var model;
    if (c.key === "nemotron") {
      model = '<p class="small"><strong>Asked once, text only.</strong> Its choice from the menu of two: <span class="mono">' +
        esc(a.option || "none") + "</span>, " + (a.ok ? "accepted by the code's check." : "refused by the code's check: the robot asks for help.") +
        "</p>" + exactText(a.reason, "Nemotron's reason, as recorded", true) +
        '<p class="small muted">The exact text it received, the call\'s date and its cost: step 3 (sha256 ' + esc(sha) + "...).</p>";
    } else if (tc.who_chose === "rule_after_refused_answer") {
      model = '<p class="small"><strong>Asked once, text only.</strong> The code refused its answer, so the rule chose: ask a ' +
        "person. Its answer and the exact text it received: step 3.</p>";
    } else if (c.key === "awaiting") {
      model = '<p class="small"><strong>Not asked yet.</strong> It will choose between <span class="mono">scrub_bowl</span> and ' +
        '<span class="mono">ask_help</span> from the robot\'s text summary, with no images: awaiting the recorded Nemotron call. ' +
        "The exact text it will receive: step 3 (sha256 " + esc(sha) + "...).</p>";
    } else if (c.key === "stand_in") {
      model = '<p class="small"><strong>Not asked.</strong> <span class="synth-tag">SYNTHETIC</span> A stand-in answer written by ' +
        'hand, in place of Nemotron, named <span class="mono">' + esc(tc.option || "none") + "</span>; no call was made.</p>";
    } else if (c.key === "rule") {
      model = '<p class="small"><strong>Not asked.</strong> ' + esc(NO_CALL[tc.who_chose] || tc.who_chose_text) + "</p>";
    } else if (c.key === "doubt") {
      model = '<p class="small warnc"><strong>Not shown.</strong> The page\'s reader found a problem with this choice (step 3).</p>';
    } else model = '<p class="small"><strong>Not asked.</strong> There was no choice to make.</p>';
    // the block carries the episode's marking (REHEARSAL, SYNTHETIC) and the simulation tag, like the steps
    return '<section class="panel deciders" aria-labelledby="wd-h"><div class="step-head"><h2 id="wd-h">Who decided what in ' +
      "this episode</h2>" + synthTag(e) + '<span class="sim-tag">simulation</span></div><div class="wd-grid">' +
      '<div class="wd team"><h3>Programmed by the team</h3><p class="small">The inspection program (step 1) and the task ' +
      "programs (step 4): " + (c.key === "awaiting" ? "the task program waits for the choice." : ran.length ? esc(ran.join("; ")) + "." :
        "no task program ran.") + " " + esc(armLine(e, c)) + "</p></div>" +
      '<div class="wd code"><h3>Checked by code</h3><p class="small">' + (conds.length ? held + " of " + conds.length +
        " conditions hold, read in the robot's own text (step 3). " : "") + "Its guards accept or refuse a pick, a done() and " +
      "a \"not found\".</p></div>" +
      '<div class="wd model"><h3>Nemotron</h3>' + model + "</div>" +
      '<div class="wd eval"><h3>Evaluator</h3><p class="small">The simulator\'s truth, never given to the robot: after Reveal, ' +
      "in step 4." + (revealed ? " Any camera image is a color render of the simulation, not what the robot read: in step " +
        "4, and step by step in step 1." : "") + "</p></div></div></section>";
  }
  // how to read a situation screen, in reading order (robot side first, then Reveal); a fixed text, no record read
  var SCREEN_HOWTO = "First, what the robot did, from its own records only: who decided what, its report, then steps 1 " +
    "to 4 (its readings, its map, its choice, its action). Then press Reveal: each claim of its report is marked true, " +
    "false or not checkable against the evaluator's records. Where the sponge actually started, the zone-by-zone check " +
    "and any camera image appear only then: in step 4, and the robot's cameras at each step in step 1.";
  function inspectScreen(id) {
    var e = EP[id] || FX[id] || RH[id];
    if (!e || !e.robot || !e.robot.inspection) {
      app.innerHTML = '<p class="panel">No inspection screen for this episode. <a href="#/">All scenes</a></p>';
      return;
    }
    setNav("home");
    // screen readers: every step redraws the screen, so #app is not announced here (the router turns it back on);
    // only the step line of step 1 is, when the step changes
    if (app.setAttribute) app.setAttribute("aria-live", "off");
    var tl = e.robot.timeline, last = Math.min(e.robot.inspection.frozen_at_step, tl.length) - 1;
    var st = { k: 0, revealed: false }, said = 0;
    function sayStep() {  // "step k of N: <words>", into the step line once the redraw is in place
      var n = document.getElementById(SAID_ID);
      if (n) n.textContent = "step " + (st.k + 1) + " of " + (last + 1) + ": " + stepWords(tl[st.k] || {});
    }
    function draw() {
      var ae = document.activeElement, act = ae && ae.getAttribute ? ae.getAttribute("data-act") : null;
      // an element without data-act (the truth's heading after Reveal, a summary) keeps the focus by its id
      var aid = !act && ae && ae.id && (!app.contains || app.contains(ae)) ? ae.id : null;
      var opened = {};  // the details blocks open before the redraw stay open after it
      if (app.querySelectorAll) {
        Array.prototype.forEach.call(app.querySelectorAll("details[data-key]"), function (d) {
          if (d.open) opened[d.getAttribute("data-key")] = true;
        });
      }
      // no internal name in the heading: the data's scene ids (such as A4) stay out of sight
      var sit = situationOf(e), plain = sit ? "Inspect, map, choose, act" : (e.synthetic ? "Synthetic fixture " + esc(e.scene) :
        "Recorded episode, kitchen " + esc(e.kitchen)) + ": inspect, map, choose, act";
      var h = '<div class="ep-head"><div><a href="#/" class="small">' + (sit ? "The three situations" : "All scenes") + "</a>" +
        (sit ? situationTabs(sit) + '<p class="eyebrow">Situation ' + (SITUATIONS.indexOf(sit) + 1) + " of 3, kitchen " + esc(sit.kitchen) +
          "</p><h1>" + esc(sit.name) + '</h1><p class="sub-h">' + plain + '</p><p class="change"><span class="k">What we changed</span> ' +
          esc(sit.change) + "</p>" + limitLine(sit) : "<h1>" + plain + "</h1>");
      if (e.synthetic) h += '<div class="synth-banner" role="note"><span class="synth-tag">SYNTHETIC</span> ' + esc(e.synthetic.banner) + "</div>";
      if (e.rehearsal) h += '<div class="rh-banner" role="note"><span class="rh-tag">REHEARSAL</span> ' + esc(e.rehearsal.banner) + "</div>";
      h += badges(e) + provBadges(e, true) + '<p class="small muted">' + esc(us(e.label)) + " " + esc(us(e.source.text)) + " " + esc(us(e.code.note)) + ".</p>" +
        '<p class="howto small"><strong>How to read this screen.</strong> ' + esc(SCREEN_HOWTO) + "</p></div></div>" +
        whoDecides(e, st.revealed) + reportBlock(e, st.revealed) +
        (sit ? nextSituation(sit) : "");
      h += '<ol class="steps">' + [inspectStep1(e, st.k, st.revealed), inspectStep2(e), inspectStep3(e), inspectStep4(e, st.revealed)].map(function (b, j) {
        return '<li><section class="step panel" aria-labelledby="step-' + (j + 1) + '">' + b + "</section></li>";
      }).join("") + "</ol>";
      h += '<section class="panel notdo"><h2>What this demo does not do</h2><ul class="lines small">' +
        NOT_DONE.map(function (x) { return "<li>" + esc(x) + "</li>"; }).join("") + "</ul></section>";
      app.innerHTML = h;
      if (app.querySelectorAll) {
        Array.prototype.forEach.call(app.querySelectorAll("details[data-key]"), function (d) {
          if (opened[d.getAttribute("data-key")]) d.open = true;
        });
      }
      if (act && app.querySelector) {  // keep the keyboard focus on the control that was used
        var b = app.querySelector(act === "i-step" ? '[data-act="i-step"][data-k="' + st.k + '"]' : '[data-act="' + act + '"]');
        if (b && b.focus) b.focus();
      } else if (aid) {
        var f = document.getElementById(aid);
        if (f && f.focus) f.focus();
      }
      if (st.k !== said) {  // the step changed: say it, once the new step line is in the page
        said = st.k;
        if (typeof setTimeout === "function") setTimeout(sayStep, 60); else sayStep();
      }
    }
    function go(k) { st.k = Math.max(0, Math.min(last, k)); draw(); }
    function toggle() {
      if (timer) { stopTimer(); draw(); return; }
      if (st.k >= last) st.k = 0;
      timer = setInterval(function () { if (st.k >= last) { stopTimer(); draw(); return; } go(st.k + 1); }, 700);
      draw();
    }
    app.onclick = function (ev) {
      var t = ev.target.closest("[data-act]");
      if (!t) return;
      var a = t.getAttribute("data-act");
      if (a === "reveal") {
        st.revealed = true;
        draw();
        // the report's own Reveal button keeps the reader on the report; any other goes to the truth
        var f = t.getAttribute("data-focus") === "report-h" ? document.getElementById("report-h") : document.getElementById("truth-h");
        if (f && f.focus) f.focus();
      } else if (a === "i-prev") { stopTimer(); go(st.k - 1); }
      else if (a === "i-next") { stopTimer(); go(st.k + 1); }
      else if (a === "i-play") toggle();
      else if (a === "i-step") { stopTimer(); go(+t.getAttribute("data-k")); }
    };
    app.oninput = function (ev) { if (ev.target.getAttribute("data-act") === "i-seek") { stopTimer(); go(+ev.target.value); } };
    keyHandler = function (ev) {
      if (/input|textarea|select|summary/i.test(ev.target.tagName)) return;  // the slider moves itself
      if (ev.key === "ArrowRight") { ev.preventDefault(); stopTimer(); go(st.k + 1); }
      else if (ev.key === "ArrowLeft") { ev.preventDefault(); stopTimer(); go(st.k - 1); }
    };
    draw();
  }

  // ------------------------------------------------------------------ trap
  function finalRegions(e) {
    var r = e.robot, last = r.timeline[r.timeline.length - 1] || {};
    var m = {};
    Object.keys(last.regions || {}).forEach(function (k) { m[k] = last.regions[k]; });
    (r.final_scope.regions || []).concat(r.final_scope.outside || []).forEach(function (x) { m[x.region] = x; });
    return m;
  }
  function trapSide(e, head) {
    var r = e.robot, v = e.evaluator, g = (r.guard || [])[0];
    var a = r.attempts || [];
    var told = {};
    ((r.dossier && r.dossier.stored_items) || []).forEach(function (it) { told[it.region] = it.category.replace("_", " "); });
    var h = '<section class="panel side"><h2>' + esc(head) + "</h2><h3>" + esc(cap1(codeWords(e))) + " · " + esc(us(e.code.note)) + "</h3>" +
      '<p class="small muted">Seed ' + esc(e.seed) + ", program " + esc((e.program || "").split(",")[0]) + ". " + esc(us(e.cost.note)) + ".</p>";
    h += planSVG(r.dossier, finalRegions(e), told);
    h += '<ul class="lines">';
    var d2 = finalRegions(e)["drawer_unit:2"];
    if (d2) h += "<li>Drawer 2, robot's own reading at the end: " + chip(d2.state) + " " + esc(d2.precision) + "</li>";
    a.forEach(function (x) { h += "<li>Program " + x.n + ": " + esc(us(x.result)) + "</li>"; });
    if (g) h += "<li>Guard on \"not found\": " + (g.decision === "refused" ? '<strong class="badc">refused</strong>, still to check per its guard: ' +
      esc((g.not_cleared || []).map(regionName).join(", ")) : '<strong>accepted</strong>') + "</li>";
    if (r.end_of_replay) h += '<li><div class="endreplay">End of replay: no recorded answer for the next step. Not a robot answer.</div></li>';
    h += '</ul><div class="verdict truth" style="margin-top:10px"><h3>Evaluator (simulator truth)</h3>' +
      '<p class="small">Sponge: ' + (v.sponge_present === false ? "not in the kitchen" : "in " + esc(regionName(v.sponge_truth_region)) +
        ", under the towel of this scene's trap") + '</p><p class="big ' + (v.false_absence ? "badc" : "good") + '">' +
      (v.false_absence ? 'FALSE "NOT FOUND" accepted' : 'No false "not found"') + "</p>" +
      (v.contacts_over_20n.counted ? '<p class="small warnc">Incident: hand contact ' + esc(v.contacts_over_20n.counted_peak_n) +
        " N, over the 20 N threshold, no violation declared by the bench.</p>" : "") +
      '<p class="small"><a href="#/episode/' + esc(e.id) + '?at=end">Open this episode step by step</a></p></div></section>';
    return h;
  }
  function trap() {
    setNav("trap");
    var A = EP.t_before_g2_r10, B = EP.t_after_o2_r10, R = EP.t_refusal_o2_r14, C10 = EP.t_corrected_dc_r10,
      C14 = EP.t_corrected_dc_r14, K = EP.s5_absent_41000;
    var h = "<h1>Scene 6. The trap: a towel over the sponge</h1>" +
      '<p class="lead">Kitchen 41000, seed 410007, program R10 written by Nemotron on Oct 6, 2026. Fixed-program replays, ' +
      "no model call, 0 USD. Neither side is the robot code of the three situations (frozen on Oct 20, 2026); both were " +
      "replaced later.</p>";
    h += '<div class="pair">' + trapSide(A, "Our earlier version") + trapSide(B, "Later version, same case") + "</div>";
    h += '<section class="panel" style="margin-top:14px"><h2>Around the pair</h2><ul class="lines">';
    var rg = (R.robot.guard || [])[0];
    h += '<li><a href="#/episode/' + esc(R.id) + '?at=end">Explicit refusal</a> (' + esc(codeWords(R)) + ', program R14, same seed): "not found" ' +
      (rg ? esc(rg.decision) + ", still to check per its guard: " + esc((rg.not_cleared || []).map(regionName).join(", ")) : "") + ".</li>";
    function fa(x) { return x.evaluator.false_absence ? 'false "not found"' : 'no false "not found"' + (x.robot.end_of_replay ? " (the replay then ends, no recorded answer)" : ""); }
    h += '<li><a href="#/episode/' + esc(C10.id) + '?at=end">' + esc(cap1(codeWords(C10))) + "</a>, neighbor seed 410008: R10, " + esc(fa(C10)) +
      '; <a href="#/episode/' + esc(C14.id) + '?at=end">R14</a>, ' + esc(fa(C14)) + ".</li>";
    h += '<li>Control, <a href="#/episode/' + esc(K.id) + '?at=end">same towel, no sponge</a> (' + esc(codeWords(K)) + ", seed " + esc(K.seed) + "): " +
      esc(outcomeWords(K.evaluator).replace(/^Correct/, "correct")) + ". The robot does not refuse everything: without a sponge, its \"sponge not found in the checked zones\" is accepted.</li>";
    h += "</ul></section>";
    app.innerHTML = h;
  }

  // ------------------------------------------------------------------ counters
  var ORDER = [["success", "Successes"], ["absence_established", "Correct \"not found\" (no sponge in the kitchen)"], ["help", "Helps (no help is called justified until the rule written ahead is validated)"],
    ["lost_absence", "No \"not found\" reached (no sponge in the kitchen)"], ["other_stop", "Other stops"], ["end_of_replay", "End of replay (no recorded answer), not a robot answer"],
    ["false_absence_accepted", "False \"not found\" accepted"], ["false_success_claimed", "False success claimed"]];
  function counters() {
    setNav("counters");
    var eps = D.episodes.filter(function (e) { return e.source.key !== "dry_run" && !isDemo(e); }), N = eps.length;
    var cnt = {};
    eps.forEach(function (e) { cnt[e.evaluator.outcome_class] = (cnt[e.evaluator.outcome_class] || 0) + 1; });
    var sum = ORDER.reduce(function (s, o) { return s + (cnt[o[0]] || 0); }, 0);
    var inc = eps.filter(function (e) { return e.evaluator.incident; }).length;
    var cal = eps.filter(function (e) { return e.evaluator.contacts_over_20n.calibration_exempt; });
    var calPeak = Math.max.apply(null, cal.map(function (e) { return e.evaluator.contacts_over_20n.calibration_peak_n || 0; }));
    var nem = eps.filter(function (e) { return e.source.key === "nemotron_recorded"; });
    var usd = nem.reduce(function (s, e) { return s + parseFloat(e.cost.settled_usd); }, 0);
    var calls = nem.reduce(function (s, e) { return s + e.cost.model_calls; }, 0);
    var h = "<h1>Counters</h1><p class=\"lead\">Selected demo episodes: " + N + ". Counts, not rates. Selected scenes illustrate; " +
      "they are not a results table.</p>";
    h += '<div class="pair"><section class="panel"><table class="ct"><caption>Outcome classes (they add up to ' + N + ")</caption><tbody>";
    ORDER.forEach(function (o) { h += "<tr><td>" + esc(o[1]) + '</td><td class="n">' + (cnt[o[0]] || 0) + " of " + N + "</td></tr>"; });
    h += '<tr class="sum"><td>Total</td><td class="n">' + sum + " of " + N + "</td></tr></tbody></table>";
    h += '<table class="ct" style="margin-top:12px"><caption>Counted apart, not an outcome class</caption><tbody>' +
      "<tr><td>Incidents (hand contacts over 20 N outside calibration, violations), even on success</td><td class=\"n\">" + inc + " of " + N + "</td></tr>" +
      "<tr><td>Calibration contacts, exempt by the bench rule, shown not hidden (peak " + calPeak + " N)</td><td class=\"n\">" + cal.length + " of " + N + "</td></tr>" +
      "</tbody></table></section>";
    h += '<section class="panel"><h2>Cost</h2><p>Recorded Nemotron episodes: ' + nem.length + ", " + calls + " model calls, " +
      usd.toFixed(4) + " USD settled token cost in total. Uncertain reservations, apart: none recorded for these episodes. " +
      "Replays: 0 model calls, 0 USD.</p>" +
      '<h2 style="margin-top:14px">Episodes counted</h2><ul class="lines small">' + eps.map(function (e) {
        return '<li><a href="#/episode/' + esc(e.id) + '?at=end">' + esc(titleOf(e)) + "</a>: " + esc(outcomeWords(e.evaluator)) +
          ' <span class="muted">(' + esc(us(e.source.badge)) + ", " + esc(codeWords(e)) + ")</span></li>";
      }).join("") + "</ul></section></div>";
    app.innerHTML = h;
  }

  // ------------------------------------------------------------------ sources
  // the plain provenance in view (source, robot code by its date, whether the replay matched); the exact identifiers
  // (internal version names, paths, hashes, the proof method) in a collapsed block, for traceability
  function about() {
    setNav("about");
    var h = "<h1>Sources and provenance</h1><p class=\"lead\">" + esc(us(D.label)) + " Every episode keeps the files it was read from, " +
      "with their sha256. Paths are relative to the bench's output folder. The exact identifiers (the lab's internal version " +
      "names, paths and hashes) are in the collapsed block of each episode.</p>";
    h += '<section class="panel"><h2>Sources</h2><ul class="lines">';
    var seen = {}, listed = D.episodes.filter(function (e) { return !isDemo(e); });  // a demo episode: on its own screen
    listed.forEach(function (e) { if (!seen[e.source.key]) { seen[e.source.key] = 1; h += "<li><strong>" + esc(us(e.source.badge)) + ".</strong> " + esc(us(e.source.text)) + "</li>"; } });
    h += "</ul></section>";
    listed.forEach(function (e) {
      var F = FR[e.id], fp = F && F.provenance, same = F && F.status === "identical";
      var frames = !same ? '<p class="small muted">Camera frames: none. ' + esc(us(F && F.reason ? F.reason : "Not re-simulated.")) + "</p>" :
        '<p class="small">Camera frames: ' + esc(F.n_steps_with_frames) + " steps, re-rendered from a replay with " + esc(codeLabel(fp.code_label)) +
        "; the replay matched the recorded episode (proof: <strong>" + esc(fp.proof_status) + "</strong>; its method is in the " +
        "exact identifiers below).</p>";
      var ids = same ? '<p class="mono">code ' + esc(usDates(fp.code_label)) + "; proof method: " + esc(usDates(fp.proof_method)) + "</p>" +
        '<p class="mono">code root ' + esc(usDates(fp.code_root)) + ", tree " + esc(String(fp.code_root_tree_sha256).slice(0, 16)) + "; harness " +
        esc(fp.harness) + " " + esc(fp.harness_version) + " " + esc(String(fp.harness_sha256).slice(0, 16)) + "; " + esc(fp.proof_file) + "</p>" : "";
      h += '<section class="panel" style="margin-top:10px"><h3>' + esc(titleOf(e)) + '</h3><p class="small muted">' + esc(us(e.source.badge)) +
        ", " + esc(codeWords(e)) + "</p>" + frames +
        '<details><summary class="small">Exact identifiers and the ' + e.provenance.files.length + " files read (for traceability)</summary>" +
        '<p class="mono">episode ' + esc(e.provenance.episode) + "; code " + esc(e.code.version) + "</p>" + ids +
        '<ul class="lines mono">' +
        e.provenance.files.map(function (f) { return "<li>[" + esc(f.side) + "] " + esc(f.file) + " " + esc(f.sha256.slice(0, 16)) + "</li>"; }).join("") +
        "</ul></details></section>";
    });
    h += '<details><summary class="small muted">Export record (for traceability)</summary><p class="mono">exported ' + esc(D.exported_at) +
      " (local time of the export machine; the record names no time zone), exporter " + esc(D.exporter_sha256.slice(0, 16)) + "</p></details>";
    app.innerHTML = h;
  }

  // ------------------------------------------------------------------ router
  function route() {
    stopTimer(); keyHandler = null; app.onclick = null; app.oninput = null;
    if (app.setAttribute) app.setAttribute("aria-live", "polite");  // the inspection screen turns it off (it announces its step line only)
    var hash = location.hash.replace(/^#/, "") || "/";
    var q = {}, parts = hash.split("?");
    (parts[1] || "").split("&").forEach(function (kv) { var p = kv.split("="); if (p[0]) q[p[0]] = decodeURIComponent(p[1] || ""); });
    var path = parts[0];
    if (path.indexOf("/episode/") === 0) episode(path.slice(9), q);
    else if (path.indexOf("/inspect/") === 0) inspectScreen(path.slice(9));
    else if (path === "/trap") trap();
    else if (path === "/counters") counters();
    else if (path === "/about") about();
    else { homeState.revealed = false; home(); }
    window.scrollTo(0, 0);
  }
  window.addEventListener("hashchange", route);
  route();
})();
