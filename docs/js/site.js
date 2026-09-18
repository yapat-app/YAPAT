(function () {
  "use strict";

  var config = window.YAPAT_DOCS;
  if (!config) return;

  function flatPages(pages, out) {
    out = out || [];
    (pages || []).forEach(function (p) {
      if (p.pages && p.pages.length) {
        flatPages(p.pages, out);
      } else if (p.href) {
        out.push(p);
      }
    });
    return out;
  }

  var sections = {};
  (config.sections || []).forEach(function (s) { sections[s.id] = s; });

  // First published page of a section (used as its landing link).
  function firstPage(section) {
    return flatPages(section.pages)[0] || null;
  }

  var pad = function (n) { return n < 10 ? "0" + n : String(n); };

  // ---- Documentation entry cards ----------------------------------------

  // Inline-SVG illustrations (64px grid); colour and stroke come from CSS.
  var icons = {
    "getting-started":
      '<path d="M12 50 30 34 52 14"/>' +
      '<circle class="knock" cx="12" cy="50" r="6"/><circle class="knock" cx="30" cy="34" r="6"/>' +
      '<circle class="fill" cx="52" cy="14" r="7"/>',
    concepts:
      '<path d="M32 12 18 32 10 52M32 12 46 32 54 52"/>' +
      '<circle class="fill" cx="32" cy="12" r="6"/>' +
      '<circle class="knock" cx="18" cy="32" r="5"/><circle class="knock" cx="46" cy="32" r="5"/>' +
      '<circle class="knock" cx="10" cy="52" r="5"/><circle class="knock" cx="54" cy="52" r="5"/>',
    guides:
      '<rect x="12" y="6" width="40" height="52" rx="6"/>' +
      '<path d="M20 20l3 3 6-6M35 21h9M20 32l3 3 6-6M35 33h9M20 44l3 3 6-6M35 45h9"/>',
    reference:
      '<path d="M32 16C24 10 14 10 8 12v36c6-2 16-2 24 4 8-6 18-6 24-4V12c-6-2-16-2-24 4zM32 16v36"/>' +
      '<path d="M25 24l-8 7 8 7M39 24l8 7-8 7"/>'
  };

  var tones = {
    "getting-started": "violet",
    concepts: "blue",
    guides: "green",
    reference: "amber"
  };

  var mount = document.querySelector("[data-doc-cards]");
  if (mount) {
    var ids = config.homeCards || [];
    var cards = ids.map(function (id) { return sections[id]; }).filter(Boolean);

    mount.innerHTML = cards.map(function (s, i) {
      var page = firstPage(s);

      var icon = icons[s.id]
        ? '<span class="card-icon tone-' + (tones[s.id] || "blue") + '" aria-hidden="true"><svg class="card-icon-svg" viewBox="0 0 64 64">' + icons[s.id] + "</svg></span>"
        : '<span class="card-icon" aria-hidden="true"></span>';
      var inner =
        '<div class="card-head">' +
        '<span class="card-index">' + pad(i + 1) + "</span>" +
        icon +
        "</div>" +
        '<h3 class="card-title">' + s.label + "</h3>" +
        '<p class="card-desc">' + s.summary + "</p>";

      if (page) {
        return (
          '<a class="card" href="' + page.href + '">' + inner +
          '<span class="card-cta">Explore section <span>→</span></span>' +
          "</a>"
        );
      }

      return (
        '<div class="card is-coming" aria-disabled="true">' + inner +
        '<span class="coming-badge">Coming soon</span>' +
        "</div>"
      );
    }).join("");
  }

  // ---- Hero call-to-action wiring ----------------------------------------

  var ctas = document.querySelectorAll("[data-cta]");
  Array.prototype.forEach.call(ctas, function (el) {
    var target = sections[el.getAttribute("data-cta")];
    var page = target && firstPage(target);
    if (page) el.setAttribute("href", page.href);
  });
})();

// ---- Light / dark theme toggle --------------------------------------------
// Same storage key and logic as the documentation pages (docs-framework.js);
// the early <head> script in index.html applies the theme before first paint.
(function themeToggle() {
  "use strict";

  var THEME_KEY = "yapat-docs-theme";

  function savedTheme() {
    try {
      var s = window.localStorage.getItem(THEME_KEY);
      if (s === "dark" || s === "light") return s;
    } catch (e) { /* storage unavailable */ }
    return null;
  }

  function currentTheme() {
    var saved = savedTheme();
    if (saved) return saved;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";
  }

  // Warm the cache so the dark DFKI logo is ready when the theme is toggled.
  new Image().src = "assets/img/dfki_Logo_dark_transparent.png";

  function applyTheme(theme) {
    if (theme === "dark") document.documentElement.setAttribute("data-theme", "dark");
    else document.documentElement.removeAttribute("data-theme");
    var btn = document.querySelector(".docs-theme-toggle");
    if (btn) {
      btn.setAttribute("aria-pressed", theme === "dark" ? "true" : "false");
      btn.setAttribute("aria-label", theme === "dark" ? "Switch to light theme" : "Switch to dark theme");
    }
  }

  applyTheme(currentTheme());

  var btn = document.querySelector(".docs-theme-toggle");
  if (!btn) return;
  btn.addEventListener("click", function () {
    var next = currentTheme() === "dark" ? "light" : "dark";
    applyTheme(next);
    try { window.localStorage.setItem(THEME_KEY, next); } catch (e) { /* storage unavailable */ }
  });
})();

// ---- Hero workspace: coordinated illustrative motion ----------------------
// One controller, one 1 s heartbeat, started as soon as the script runs.
// Each beat changes something small in one or more panels; CSS transitions
// (home.css) make every change smooth. The sequence is fixed (no randomness).
// Illustrative only, not real YAPAT data. Idle when the visitor prefers
// reduced motion or the tab is hidden.
(function heroWorkspace() {
  "use strict";

  var svg = document.querySelector(".hero-svg");
  if (!svg || !window.matchMedia) return;

  var BEAT = 1000;                 // ms, the visual heartbeat
  var STATE_BEATS = 5;             // beats per annotation state
  var STATES = 3;
  var ROW = 26;                    // feed row pitch (SVG units)
  var HOT = [7, 11, 3];            // emphasised histogram bar per state
  var BOX = [0, 36, 16];           // annotation box offset per state (SVG units)
  var CREEP = [0, 6, 0, -6];       // slow drift of the box between states
  var FOCUS = [[316, 118], [434, 112], [382, 162]]; // cluster centres
  var PZ_CENTER = [384, 134];
  var BAR_SEQ = [6, 4, 8, 5, 9, 3, 7, 11, 2, 10, 6, 5];
  var BAR_DELTA = [0.08, -0.06, 0.10, -0.05, 0.07, -0.08, 0.06, -0.09, 0.10, -0.06, 0.05, -0.07];
  var AUDIO_BEATS = 6;             // one playhead pass, ~6 s on screen
  var AUDIO_SECS = 3.0;            // ...representing a 3 s clip (BirdNET window)
  var AUDIO_SPAN = 30;             // box travel (SVG units) across one pass
  var LIT_BEATS = 2;               // how long a spectrogram cell stays lit

  var feed = svg.querySelector(".v-feed");
  var boxg = svg.querySelector(".v-boxg");
  var pz = svg.querySelector(".v-pz");
  var playhead = svg.querySelector(".v-playhead");
  var timeNow = svg.querySelector(".v-time-now");
  var bars = Array.prototype.slice.call(svg.querySelectorAll(".hb"));
  var points = Array.prototype.slice.call(svg.querySelectorAll(".v-pz .pt"));
  var cells = Array.prototype.slice.call(svg.querySelectorAll(".s1, .s2, .s3"));
  var reduce = window.matchMedia("(prefers-reduced-motion: reduce)");

  // Spectrogram rows, grouped by y so a "frequency band" can sweep through.
  var rows = {};
  cells.forEach(function (c) {
    var y = c.getAttribute("y");
    (rows[y] = rows[y] || []).push(c);
  });
  var rowKeys = Object.keys(rows).sort(function (a, b) { return a - b; });

  var scales = bars.map(function () { return 1; });
  var lit = [];                    // [{beat, els}]
  var dimmed = [];
  var beat = 0;
  var feedIndex = 0;
  var timer = 0;

  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  function setState(state) {
    svg.setAttribute("data-state", String(state));
    bars.forEach(function (bar, i) { bar.classList.toggle("hot", i === HOT[state]); });
  }

  function renderFeed() {
    if (feed) feed.style.setProperty("--feed-y", -ROW * feedIndex + "px");
  }

  function light(els, n) {
    els.forEach(function (el) { el.classList.add("lit"); });
    lit.push({ beat: n, els: els });
    while (lit.length && lit[0].beat <= n - LIT_BEATS) {
      lit.shift().els.forEach(function (el) { el.classList.remove("lit"); });
    }
  }

  function spectrogram(n) {
    // Band sweeps down and back up through the frequency rows.
    var span = rowKeys.length * 2 - 2;
    var k = n % span;
    var row = rows[rowKeys[k < rowKeys.length ? k : span - k]];
    // Plus three individual cells picked from a fixed sequence.
    var extra = [0, 1, 2].map(function (j) { return cells[(n * 17 + j * 29 + 5) % cells.length]; });
    light(row.concat(extra), n);
  }

  function distribution(n) {
    var i = BAR_SEQ[n % BAR_SEQ.length];
    scales[i] = clamp(scales[i] + BAR_DELTA[n % BAR_DELTA.length], 0.78, 1.22);
    bars[i].style.setProperty("--s", scales[i].toFixed(3));
  }

  function clock(secs) {
    var t = secs.toFixed(1);
    return "00:" + (t.length < 4 ? "0" : "") + t;
  }

  // Audio playhead: same beat clock. Each beat starts a 1 s linear move to the
  // next slice of the pass; at the end it fades out, jumps back unseen, fades in.
  function audio(n) {
    if (!playhead) return 0;
    var k = n % AUDIO_BEATS;
    if (k === 0) {
      svg.classList.add("ph-reset");
      svg.style.setProperty("--ph", "0");
      void playhead.getBoundingClientRect();
      svg.classList.remove("ph-reset", "ph-end");
    }
    var next = (k + 1) / AUDIO_BEATS;
    svg.style.setProperty("--ph", next.toFixed(4));
    if (k === AUDIO_BEATS - 1) svg.classList.add("ph-end");
    if (timeNow) timeNow.textContent = clock(next * AUDIO_SECS);
    return next;
  }

  // The annotation box keeps its state offset and creep, and travels with the
  // playhead so the selection tracks the approximate audio position.
  function box(n, state, phase) {
    if (!boxg) return;
    var x = BOX[state] * 0.4 + CREEP[(n >> 1) % CREEP.length] * 0.5 + AUDIO_SPAN * phase;
    boxg.style.setProperty("--box-x", x.toFixed(1) + "px");
  }

  function projection(n, state) {
    if (!pz || !points.length) return;
    // Points: two gently dim/brighten each beat (fixed sequence).
    var a = points[(n * 7 + 3) % points.length];
    a.classList.add("dim");
    dimmed.push(a);
    if (dimmed.length > 4) dimmed.shift().classList.remove("dim");

    // Viewport: 10-beat cycle, scale stays within 0.97 to 1.03.
    var phase = n % 10, s = null, pan = false;
    if (phase === 0) { s = 1; }
    else if (phase === 2) { s = 0.97; }
    else if (phase === 4) { s = 1; pan = true; }
    else if (phase === 6) { s = 1.03; pan = true; }
    else if (phase === 8) { s = 1; }
    if (s === null) return;
    var f = FOCUS[state];
    var x = pan ? (PZ_CENTER[0] - f[0]) * 0.08 : 0;
    var y = pan ? (PZ_CENTER[1] - f[1]) * 0.08 : 0;
    pz.style.setProperty("--pz-s", String(s));
    pz.style.setProperty("--pz-x", x.toFixed(1) + "px");
    pz.style.setProperty("--pz-y", y.toFixed(1) + "px");
  }

  function tick() {
    var state = Math.floor(beat / STATE_BEATS) % STATES;
    if (beat % STATE_BEATS === 0) {
      setState(state);
      if (beat > 0) { feedIndex += 1; renderFeed(); }
    }
    spectrogram(beat);
    distribution(beat);
    box(beat, state, audio(beat));
    projection(beat, state);
    beat += 1;
  }

  function start() {
    if (timer || reduce.matches || document.hidden) return;
    tick();                                   // first change happens now
    timer = window.setInterval(tick, BEAT);
  }

  function stop() {
    window.clearInterval(timer);
    timer = 0;
  }

  function resetStatic() {
    stop();
    beat = 0;
    feedIndex = 0;
    setState(0);
    renderFeed();
    scales = bars.map(function () { return 1; });
    bars.forEach(function (b) { b.style.removeProperty("--s"); });
    lit.forEach(function (l) { l.els.forEach(function (el) { el.classList.remove("lit"); }); });
    lit = [];
    dimmed.forEach(function (p) { p.classList.remove("dim"); });
    dimmed = [];
    svg.classList.remove("ph-end", "ph-reset");
    svg.style.removeProperty("--ph");
    if (timeNow) timeNow.textContent = "00:01.4";
    [boxg, pz].forEach(function (el) {
      if (el) ["--box-x", "--pz-s", "--pz-x", "--pz-y"].forEach(function (v) { el.style.removeProperty(v); });
    });
  }

  // After the last duplicated feed row settles, jump back to the first.
  if (feed) {
    feed.addEventListener("transitionend", function (e) {
      if (e.target !== feed || feedIndex < STATES) return;
      feed.style.transition = "none";
      feedIndex = 0;
      renderFeed();
      void feed.getBoundingClientRect();
      feed.style.transition = "";
    });
  }

  function sync() {
    if (reduce.matches) resetStatic();
    else if (document.hidden) stop();
    else start();
  }

  if (reduce.addEventListener) reduce.addEventListener("change", sync);
  document.addEventListener("visibilitychange", sync);

  setState(0);
  sync();
})();

// Let keyboard scrolling (Space / PageDown / arrows) reach the scroll area
// without needing a click first.
(function () {
  "use strict";
  var scroller = document.querySelector(".page-scroll");
  if (scroller) scroller.focus({ preventScroll: true });
})();
