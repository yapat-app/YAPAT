/**
 * Documentation framework.
 *
 * Renders the shared page shell (header, sidebar, breadcrumbs,
 * on-this-page toc, previous/next, footer) from docs-config.js.
 * A documentation page only contains its content - the framework
 * detects the current page from the URL, looks it up in the config
 * and builds the entire skeleton around the content.
 *
 * Usage (every page):
 *
 *   <body>
 *     <main data-docs-content>
 *       ... page content ...
 *     </main>
 *     <script src="../../js/docs-config.js"></script>
 *     <script src="../../js/docs-framework.js"></script>
 *   </body>
 */

(function () {
  "use strict";

  var CONFIG = window.YAPAT_DOCS;
  var contentRoot = document.querySelector("[data-docs-content]");
  if (!CONFIG || !contentRoot) {
    // Nothing to build: never leave the raw content hidden.
    document.documentElement.classList.remove("docs-js");
    return;
  }

  var esc = function (s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  };

  var norm = function (href) {
    return String(href).replace(/^\/+|\/+$/g, "");
  };

  // ---- config model -------------------------------------------------

  var flatPages = []; // every published page, depth-first, in nav order

  function collect(pages, ancestors) {
    (pages || []).forEach(function (p) {
      var chain = ancestors.concat([p]);
      if (p.pages && p.pages.length) {
        collect(p.pages, chain);
      } else if (p.href) {
        flatPages.push({ id: p.id, title: p.title, href: norm(p.href), chain: chain });
      }
    });
  }

  CONFIG.sections.forEach(function (s) { collect(s.pages, [s]); });

  // ---- current page detection (from URL) ----------------------------

  var current = null;

  (function detect() {
    var path = (window.location.pathname || "").split("?")[0].split("#")[0];
    var normPath = path.replace(/\/$/, "");
    var best = null;
    var bestSegs = -1;

    flatPages.forEach(function (p) {
      var suffix = "/" + p.href;
      if (normPath === suffix || normPath.endsWith(suffix)) {
        var segs = p.href.split("/").length;
        if (segs > bestSegs) { best = p; bestSegs = segs; }
      }
    });
    current = best;
  })();

  // Relative path from the page file back to the docs root.
  // A page N path segments deep needs ROOT = "../../" (N ups).
  var ROOT = current
    ? "../".repeat(current.href.split("/").length)
    : (window.DOCS_ROOT || "..") + "/";

  function url(href) {
    return href ? ROOT + norm(href) + "/" : null;
  }

  // ---- light / dark theme -------------------------------------------

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
  new Image().src = ROOT + "assets/img/dfki_Logo_dark_transparent.png";

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

  // ---- helpers used by multiple renderers ---------------------------

  function containsActive(item) {
    if (!current) return false;
    if (item.id === current.id) return true;
    return (item.pages || []).some(containsActive);
  }

  function hasPlannedChild(item) {
    return (item.pages || []).some(function (p) {
      return p.pages ? hasPlannedChild(p) : !p.href;
    });
  }

  // ---- header -------------------------------------------------------

  var searchIcon =
    '<svg class="docs-search-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
      "<circle cx=\"11\" cy=\"11\" r=\"8\"></circle>" +
      '<line x1="21" y1="21" x2="16.65" y2="16.65"></line>' +
    "</svg>";

  var iconSun =
    '<svg class="docs-theme-icon icon-sun" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
      "<circle cx=\"12\" cy=\"12\" r=\"4\"></circle>" +
      '<line x1="12" y1="1" x2="12" y2="3"></line>' +
      '<line x1="12" y1="21" x2="12" y2="23"></line>' +
      '<line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>' +
      '<line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>' +
      '<line x1="1" y1="12" x2="3" y2="12"></line>' +
      '<line x1="21" y1="12" x2="23" y2="12"></line>' +
      '<line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>' +
      '<line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>' +
    "</svg>";

  var iconMoon =
    '<svg class="docs-theme-icon icon-moon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
      '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>' +
    "</svg>";

  function renderHeader() {
    var kbd = /Mac/i.test(navigator.platform || "") ? "&#8984;K" : "Ctrl K";
    return (
      '<header class="docs-header">' +
        '<div class="docs-header-inner">' +
          '<button class="docs-nav-toggle" type="button" aria-label="Toggle navigation" aria-expanded="false" aria-controls="docs-sidebar">' +
            '<span class="docs-nav-toggle-bar"></span>' +
            '<span class="docs-nav-toggle-bar"></span>' +
          "</button>" +
          '<a class="logo" href="' + ROOT + 'index.html">YAPAT<span class="logo-sub">Documentation</span></a>' +
          '<div class="docs-search" data-docs-search>' +
            searchIcon +
            '<input class="docs-search-input" type="search" placeholder="Search documentation&hellip;" ' +
              'aria-label="Search documentation" autocomplete="off" spellcheck="false" ' +
              'role="combobox" aria-expanded="false" aria-controls="docs-search-results">' +
            '<span class="docs-search-kbd" aria-hidden="true">' + kbd + "</span>" +
            '<div class="docs-search-drop" id="docs-search-results" role="listbox" aria-label="Search results"></div>' +
          "</div>" +
          '<div class="header-actions">' +
            '<button class="docs-theme-toggle" type="button" aria-label="Switch to dark theme" aria-pressed="false">' +
              iconSun + iconMoon +
            "</button>" +
            '<a class="github" href="' + esc(CONFIG.repoUrl) + '" target="_blank" rel="noopener noreferrer">' +
              "GitHub <span>↗</span>" +
            "</a>" +
            '<img class="dfki-logo" src="' + ROOT + 'assets/img/dfki_Logo.jpg" alt="DFKI logo">' +
          "</div>" +
        "</div>" +
      "</header>"
    );
  }

  // ---- global search -------------------------------------------------

  var searchIndex = flatPages.map(function (p) {
    return {
      title: p.title,
      href: p.href,
      path: p.chain.map(function (n) { return n.title || n.label; })
    };
  });

  // ---- sidebar ------------------------------------------------------

  function renderNavItem(item, isRoot) {
    var label = esc(item.title);

    if (item.pages && item.pages.length) {
      var open = (isRoot && containsActive(item)) || hasPlannedChild(item);
      return (
        "<details class=\"docs-nav-group\" " + (open ? "open" : "") + ">" +
          "<summary class=\"docs-nav-summary\">" + label + "</summary>" +
          '<div class="docs-nav-sub">' +
            item.pages.map(function (p) { return renderNavItem(p, false); }).join("") +
          "</div>" +
        "</details>"
      );
    }

    if (item.href) {
      var active = current && item.id === current.id;
      return (
        '<a class="docs-nav-link' + (active ? " is-active" : "") + '" ' +
          'href="' + url(item.href) + '"' +
          (active ? ' aria-current="page"' : "") +
          ">" + label + "</a>"
      );
    }

    return '<span class="docs-nav-link is-planned" title="Planned">' + label + "</span>";
  }

  function renderSidebar() {
    var sections = CONFIG.sections.map(function (s) {
      return (
        '<div class="docs-nav-section">' +
          '<div class="docs-nav-section-title">' + esc(s.label) + "</div>" +
          '<div class="docs-nav-section-links">' +
            (s.pages || []).map(function (p) { return renderNavItem(p, true); }).join("") +
          "</div>" +
        "</div>"
      );
    });
    return (
      '<nav class="docs-nav" id="docs-sidebar" aria-label="Documentation">' +
        sections.join("") +
      "</nav>"
    );
  }

  // ---- breadcrumbs --------------------------------------------------

  function renderBreadcrumbs() {
    if (!current) return "";
    var crumb = function (label, href, isCurrent) {
      var inner = href
        ? '<a href="' + href + '">' + esc(label) + "</a>"
        : "<span>" + esc(label) + "</span>";
      return '<li class="docs-crumb' + (isCurrent ? " is-current" : "") + '">' + inner + "</li>";
    };

    var items = '<li class="docs-crumb"><a href="' + ROOT + 'index.html">Home</a></li>';
    current.chain.forEach(function (node, i) {
      var isLast = i === current.chain.length - 1;
      var label = node.title || node.label;
      if (isLast) {
        items += crumb(label, null, true);
      } else if (node.pages && node.href) {
        items += crumb(label, url(node.href), false);
      } else {
        items += crumb(label, null, false);
      }
    });
    return '<nav class="docs-breadcrumbs" aria-label="Breadcrumb"><ol>' + items + "</ol></nav>";
  }

  // ---- on-this-page toc ---------------------------------------------

  function slugify(text) {
    return String(text).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "") || "section";
  }

  function buildToc() {
    var headings = contentRoot.querySelectorAll("h2, h3");
    if (!headings.length) return { list: "", items: [] };

    var used = {};
    var items = [];
    headings.forEach(function (h) {
      var base = slugify(h.textContent);
      var id = base;
      var n = (used[base] || 0) + 1;
      used[base] = n;
      if (n > 1) id = base + "-" + n;
      h.id = id;
      items.push({ level: h.tagName === "H3" ? 3 : 2, id: id, text: h.textContent });
    });

    var list = items.map(function (it) {
      return (
        '<li class="docs-toc-item docs-toc-level-' + it.level + '">' +
          '<a href="#' + it.id + '">' + esc(it.text) + "</a>" +
        "</li>"
      );
    }).join("");

    return { list: "<ol>" + list + "</ol>", items: items };
  }

  // ---- previous / next ----------------------------------------------

  function renderPrevNext() {
    if (flatPages.length < 2 || !current) return "";
    var idx = flatPages.indexOf(current);
    var prev = idx > 0 ? flatPages[idx - 1] : null;
    var next = idx < flatPages.length - 1 ? flatPages[idx + 1] : null;

    var link = function (p, cls) {
      var arrow =
        '<span class="docs-pn-arrow" aria-hidden="true">' + (cls === "is-prev" ? "←" : "→") + "</span>";
      var text =
        '<span class="docs-pn-text">' +
          '<span class="docs-pn-label">' + (cls === "is-prev" ? "Previous" : "Next") + "</span>" +
          '<span class="docs-pn-title">' + esc(p.title) + "</span>" +
        "</span>";
      return (
        '<a class="docs-pn ' + cls + '" href="' + url(p.href) + '">' +
          (cls === "is-prev" ? arrow + text : text + arrow) +
        "</a>"
      );
    };

    return (
      '<nav class="docs-prevnext" aria-label="Page navigation">' +
        (prev ? '<div class="docs-pn-side">' + link(prev, "is-prev") + "</div>" : "") +
        (next ? '<div class="docs-pn-side is-next">' + link(next, "is-next") + "</div>" : "") +
      "</nav>"
    );
  }

  // ---- footer -------------------------------------------------------

  // ---- footer -------------------------------------------------------

  function renderFooter() {
    return (
      '<footer class="docs-footer">' +
        '<div class="docs-footer-inner">' +
          '<div class="docs-footer-row">' +
            '<div class="docs-footer-brand">' +
              '<img class="docs-footer-logo" src="' + ROOT + 'assets/img/dfki_Logo.jpg" alt="DFKI">' +
  '<span class="docs-footer-divider" aria-hidden="true"></span>' +
              '<span class="docs-footer-name">YAPAT</span>' +
              '<span class="docs-footer-sub">Yet Another PAM Annotation Tool</span>' +
            "</div>" +
            '<span class="docs-footer-copy">&copy; 2026 DFKI &mdash; German Research Center for Artificial Intelligence</span>' +
            '<a class="docs-footer-github" href="' + esc(CONFIG.repoUrl) + '" target="_blank" rel="noopener noreferrer">' +
              "GitHub <span>&#8599;</span>" +
            "</a>" +
          "</div>" +
        "</div>" +
      "</footer>"
    );
  }

  // ---- assemble the page --------------------------------------------

  var toc = buildToc();

  // Wide tables / code must scroll inside their own box, never widen the page.
  Array.prototype.forEach.call(contentRoot.querySelectorAll("table"), function (table) {
    var wrap = document.createElement("div");
    wrap.className = "docs-table-scroll";
    wrap.tabIndex = 0;
    wrap.setAttribute("role", "region");
    wrap.setAttribute("aria-label", "Scrollable table");
    table.parentNode.insertBefore(wrap, table);
    wrap.appendChild(table);
  });
  Array.prototype.forEach.call(contentRoot.querySelectorAll("pre"), function (pre) {
    pre.tabIndex = 0;
  });

  var contentHTML = contentRoot.innerHTML;

  document.body.className = "docs-page";
  document.body.innerHTML =
    renderHeader() +
    '<div class="docs-scroll" tabindex="-1"><div class="docs-body">' +
      '<aside class="docs-sidebar-outer">' + renderSidebar() + "</aside>" +
      '<main class="docs-main">' +
        renderBreadcrumbs() +
        (toc.list
          ? '<details class="docs-toc-mobile"><summary>On this page</summary>' + toc.list + "</details>"
          : "") +
        '<article class="docs-content" data-main-content>' + contentHTML + "</article>" +
        renderPrevNext() +
      "</main>" +
      (toc.list ? '<aside class="docs-toc-outer"><nav class="docs-toc" aria-label="On this page"><div class="docs-toc-title">On this page</div>' + toc.list + "</nav></aside>" : "") +
      renderFooter() +
    "</div></div>" +
    '<div class="docs-backdrop" aria-hidden="true"></div>';

  applyTheme(currentTheme());

  // Let keyboard scrolling (Space / PageDown / arrows) work without a click first.
  (function () {
    var scroller = document.querySelector(".docs-scroll");
    if (scroller) scroller.focus({ preventScroll: true });
  })();

  // ---- global search -------------------------------------------------

  (function search() {
    var root = document.querySelector("[data-docs-search]");
    if (!root) return;
    var input = root.querySelector(".docs-search-input");
    var drop = root.querySelector(".docs-search-drop");

    function matches(entry, query) {
      var terms = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
      if (!terms.length) return true;
      var hay = (entry.title + " " + entry.path.join(" ")).toLowerCase();
      return terms.every(function (t) { return hay.indexOf(t) !== -1; });
    }

    function results(query) {
      var q = (query || "").trim().toLowerCase();
      var found = searchIndex.filter(function (e) { return matches(e, q); });
      found.sort(function (a, b) {
        var at = a.title.toLowerCase();
        var bt = b.title.toLowerCase();
        var ap = a.path.join(" ").toLowerCase();
        var bp = b.path.join(" ").toLowerCase();
        if (q) {
          var ai = at.indexOf(q);
          var bi = bt.indexOf(q);
          if (ai === 0 && bi !== 0) return -1;
          if (bi === 0 && ai !== 0) return 1;
          if (ai !== -1 || bi !== -1) return (ai === -1 ? 99 : ai) - (bi === -1 ? 99 : bi);
        }
        return ap.indexOf(q) - bp.indexOf(q);
      });
      return found.slice(0, q ? 12 : 8);
    }

    function render(list) {
      if (!list.length) {
        drop.innerHTML = '<div class="docs-search-empty">No matching pages</div>';
      } else {
        drop.innerHTML = list.map(function (entry) {
          var trail = entry.path.slice(0, -1);
          return (
            '<a class="docs-search-result" href="' + url(entry.href) + '" role="option">' +
              '<span class="docs-search-result-title">' + esc(entry.title) + "</span>" +
              (trail.length
                ? '<span class="docs-search-result-path">' + esc(trail.join(" &middot; ")) + "</span>"
                : "") +
            "</a>"
          );
        }).join("");
      }
      root.setAttribute("data-open", "true");
      input.setAttribute("aria-expanded", "true");
    }

    function close() {
      drop.innerHTML = "";
      root.setAttribute("data-open", "false");
      input.setAttribute("aria-expanded", "false");
    }

    function syncPlaceholder() {
      input.setAttribute("placeholder",
        window.innerWidth <= 700 ? "Search\u2026" : "Search documentation\u2026");
    }

    input.addEventListener("focus", function () { render(results(input.value)); });
    input.addEventListener("input", function () { render(results(input.value)); });

    document.addEventListener("click", function (e) {
      if (!root.contains(e.target)) close();
    });

    input.addEventListener("keydown", function (e) {
      var items = drop.querySelectorAll(".docs-search-result");
      var active = drop.querySelector(".is-active");
      var idx = Array.prototype.indexOf.call(items, active);
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        if (!items.length) return;
        e.preventDefault();
        var next = e.key === "ArrowDown"
          ? (idx + 1) % items.length
          : (idx - 1 + items.length) % items.length;
        Array.prototype.forEach.call(items, function (el, i) {
          el.classList.toggle("is-active", i === next);
        });
        items[next].scrollIntoView({ block: "nearest" });
      } else if (e.key === "Enter") {
        var target = items[idx >= 0 ? idx : 0];
        if (target) { e.preventDefault(); window.location.href = target.getAttribute("href"); }
      } else if (e.key === "Escape") {
        close();
        input.blur();
      }
    });

    document.addEventListener("keydown", function (e) {
      var k = e.key.toLowerCase();
      if ((e.metaKey || e.ctrlKey) && k === "k") {
        e.preventDefault();
        input.focus();
        input.select();
        render(results(input.value));
      } else if (e.key === "/") {
        var el = document.activeElement;
        if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.isContentEditable)) return;
        e.preventDefault();
        input.focus();
        render(results(input.value));
      }
    });

    syncPlaceholder();
    window.addEventListener("resize", syncPlaceholder);
  })();

  // ---- scrollspy for the on-this-page toc ---------------------------

  (function scrollspy() {
    var links = document.querySelectorAll(".docs-toc a, .docs-toc-mobile a");
    var headings = Array.prototype.filter.call(
      document.querySelectorAll(".docs-main h2, .docs-main h3"),
      function (h) { return h.id; }
    );
    if (!links.length || !headings.length) return;

    var group = {};
    links.forEach(function (a) { group[a.getAttribute("href")] = a; });

    var activeLink = null;

    function computeOffset() {
      var header = document.querySelector(".docs-header");
      return (header ? header.getBoundingClientRect().height : 64) + 40;
    }

    function update() {
      var marker = computeOffset();
      var current = null;
      for (var i = 0; i < headings.length; i++) {
        var top = headings[i].getBoundingClientRect().top;
        if (top <= marker) current = headings[i];
        else break;
      }
      var target = current ? group["#" + current.id] : null;
      if (target === activeLink) return;
      if (activeLink) activeLink.classList.remove("is-active");
      if (target) target.classList.add("is-active");
      activeLink = target;
    }

    var scroller = document.querySelector(".docs-scroll");
    (scroller || window).addEventListener("scroll", update, { passive: true });
    window.addEventListener("resize", update);
    update();
  })();

  // ---- smooth scroll for on-this-page links -------------------------

  var reduceMotion = window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (!reduceMotion) {
    document.addEventListener("click", function (e) {
      var a = e.target.closest ? e.target.closest('a[href^="#"]') : null;
      if (!a || !a.closest(".docs-toc, .docs-toc-mobile")) return;
      var id = a.getAttribute("href").slice(1);
      var target = document.getElementById(id);
      if (!target) return;
      e.preventDefault();
      target.scrollIntoView({ behavior: "smooth", block: "start" });
      history.replaceState(null, "", "#" + id);
    });
  }

  // ---- mobile navigation drawer -------------------------------------

  // Must match the mobile breakpoint (max-width: 767px) in docs.css.
  var MOBILE_QUERY = "(max-width: 767px)";

  (function drawer() {
    var toggle = document.querySelector(".docs-nav-toggle");
    var backdrop = document.querySelector(".docs-backdrop");
    var sidebar = document.querySelector(".docs-sidebar-outer");
    var header = document.querySelector(".docs-header");
    var root = document.documentElement;
    if (!toggle || !backdrop || !sidebar) return;

    function isOpen() {
      return document.body.classList.contains("docs-nav-open");
    }

    function setOpen(open) {
      document.body.classList.toggle("docs-nav-open", open);
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
    }

    toggle.addEventListener("click", function () {
      var open = !isOpen();
      setOpen(open);
      // Move keyboard focus into the drawer (it follows the header in the DOM).
      if (open) {
        var first = sidebar.querySelector("a.docs-nav-link.is-active, a.docs-nav-link");
        if (first) first.focus();
      }
    });
    backdrop.addEventListener("click", function () { setOpen(false); toggle.focus(); });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && isOpen()) {
        setOpen(false);
        toggle.focus();
      }
    });
    sidebar.addEventListener("click", function (e) {
      if (e.target.closest("a")) setOpen(false);
    });

    // The drawer and its backdrop start directly below the sticky header.
    function syncHeaderHeight() {
      if (header) root.style.setProperty("--docs-header-h", header.offsetHeight + "px");
    }

    syncHeaderHeight();
    window.addEventListener("resize", syncHeaderHeight);
    if (window.ResizeObserver && header) new ResizeObserver(syncHeaderHeight).observe(header);

    // Leaving the mobile layout closes the drawer so state never leaks.
    var mq = window.matchMedia(MOBILE_QUERY);
    var onChange = function () { syncHeaderHeight(); if (!mq.matches) setOpen(false); };
    if (mq.addEventListener) mq.addEventListener("change", onChange);
    else if (mq.addListener) mq.addListener(onChange);
  })();

  // ---- sidebar scroll persistence -------------------------------------
  // Every docs page is a separate document and the shell is rebuilt on load, so
  // the sidebar (its own scroll container) would start at the top each time.
  // Remember where it was and put it back. The position is stored as "this
  // link sat N px below the top of the sidebar" rather than a raw scrollTop, so
  // it still lines up when expanded groups above it differ between pages.
  // Desktop/tablet only: the mobile drawer keeps its existing behaviour.

  (function sidebarScroll() {
    var KEY = "yapat-docs-sidebar";
    var side = document.querySelector(".docs-sidebar-outer");
    if (!side) return;

    var mq = window.matchMedia(MOBILE_QUERY);
    var clicked = null;   // {href, off, at} recorded when a sidebar link is activated

    function links() {
      return Array.prototype.slice.call(side.querySelectorAll("a.docs-nav-link"))
        .filter(function (a) { return a.offsetParent !== null; });
    }

    function snapshot(link) {
      var s = side.getBoundingClientRect();
      var r = link.getBoundingClientRect();
      return { href: link.getAttribute("href"), off: r.top - s.top, top: side.scrollTop };
    }

    function save(state) {
      try { window.sessionStorage.setItem(KEY, JSON.stringify(state)); } catch (e) { /* storage unavailable */ }
    }

    function load() {
      try {
        var raw = window.sessionStorage.getItem(KEY);
        window.sessionStorage.removeItem(KEY);
        return raw ? JSON.parse(raw) : null;
      } catch (e) { return null; }
    }

    side.addEventListener("click", function (e) {
      var a = e.target.closest && e.target.closest("a.docs-nav-link");
      if (!a || mq.matches) return;
      clicked = snapshot(a);
      clicked.at = Date.now();
    });

    // Covers clicks, search results, and back/forward alike.
    window.addEventListener("pagehide", function () {
      if (mq.matches) return;
      if (clicked && Date.now() - clicked.at < 3000) return save(clicked);
      var s = side.getBoundingClientRect();
      var first = links().filter(function (a) { return a.getBoundingClientRect().top >= s.top; })[0];
      if (first) save(snapshot(first));
    });

    if (mq.matches) return;

    var saved = load();
    if (saved) {
      var target = null;
      links().some(function (a) {
        if (a.getAttribute("href") === saved.href) { target = a; return true; }
        return false;
      });
      if (target) {
        var delta = (target.getBoundingClientRect().top - side.getBoundingClientRect().top) - saved.off;
        side.scrollTop += delta;
      } else if (typeof saved.top === "number") {
        side.scrollTop = saved.top;
      }
    }

    // Reveal the active page with the least movement, only if it is off-screen.
    var active = side.querySelector("a.docs-nav-link.is-active");
    if (active) {
      var sr = side.getBoundingClientRect();
      var ar = active.getBoundingClientRect();
      if (ar.top < sr.top) side.scrollTop -= sr.top - ar.top + 8;
      else if (ar.bottom > sr.bottom) side.scrollTop += ar.bottom - sr.bottom + 8;
    }
  })();

  // ---- theme toggle --------------------------------------------------

  (function themeToggle() {
    var btn = document.querySelector(".docs-theme-toggle");
    if (!btn) return;
    btn.addEventListener("click", function () {
      var next = currentTheme() === "dark" ? "light" : "dark";
      applyTheme(next);
      try { window.localStorage.setItem(THEME_KEY, next); } catch (e) { /* storage unavailable */ }
    });
  })();
})();