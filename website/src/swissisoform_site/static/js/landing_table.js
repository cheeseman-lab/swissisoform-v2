/* Landing gene table — one row per gene, expandable to its isoforms.
 *
 * Data: /api/landing.json (swissisoform_site/landing.py), compact by design —
 * shared vocabularies (tags, keywords) referenced by integer index. Built once
 * here into typed arrays so filtering 40k isoforms is a pass over bitsets.
 *
 * Rendering is virtualised: the flattened list of visible rows (genes, plus the
 * isoform rows of every expanded gene) is laid out by measured heights, and only
 * the rows on screen (+ overscan) exist in the DOM. Rows can change height (chips
 * or keywords expanded inline, wrapping on narrow screens); the height cache is
 * keyed per row, so a re-measure moves only what is below it.
 *
 * Filters keep the controls the card grid had (same element ids): name/UniProt
 * search and the keyword facet match per gene; the tag facet (fired pipeline tags
 * + AI synthesis tags), the category toggles and the ORF type toggles match per
 * isoform — a gene shows when any of its isoforms matches, and expanding it
 * lists the matching ones.
 */
(function () {
  "use strict";

  var table = document.getElementById("gene-table");
  if (!table) return;
  var body = table.querySelector(".gt-body");
  var loading = table.querySelector(".gt-loading");
  var noResults = document.getElementById("no-results");
  var resultCount = document.getElementById("result-count");
  var GENE_BASE = table.dataset.geneBase; // ".../genes/__GENE__"
  var SCAN_QS = table.dataset.scanQs || "";

  var CHIP_N = 6; // fired-tag chips shown before "+N"
  var KW_N = 3; // keywords shown before "+N"
  var OVERSCAN = 600; // px rendered above/below the viewport
  var EST = { g: 48, h: 28, i: 40, m: 32, e: 14 }; // estimated heights by row type
  var NARROW = window.matchMedia("(max-width: 760px)"); // matches site.css

  // ---------------------------------------------------------------- helpers
  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function geneHref(name) {
    return GENE_BASE.replace("__GENE__", encodeURIComponent(name)) + SCAN_QS;
  }
  function isoHref(name, slug) { // slug from the payload (data.tis_slug), never re-derived
    return GENE_BASE.replace("__GENE__", encodeURIComponent(name)) + "/isoforms/" +
      encodeURIComponent(slug) + SCAN_QS;
  }
  function store(fn) { try { return fn(); } catch (e) { return null; } }

  // ------------------------------------------------------------------ model
  var D, G, ISO; // payload, genes, flat isoform list
  var W; // 32-bit words per isoform tag bitset
  var isoBits, isoCats; // Uint32Array(nIso*W), Uint8Array(nIso) fired-category mask
  var isoOrf, orfTypes = [], orfCount = []; // Uint8Array(nIso) index into orfTypes
  var tagKeys = [], tagLabel = [], tagCat = [], tagIsAi = []; // tag universe
  var tagCount, kwCount;
  var STATE_FIRES = { m: 1, i: 1, x: 1 }; // green (met/interesting) or red = fires

  function build(data) {
    D = data;
    G = data.genes;
    ISO = [];
    var nP = data.tags.length, nA = data.llm_tags.length;
    data.tags.forEach(function (t) {
      tagKeys.push("p:" + t.id); tagLabel.push(t.label); tagCat.push(t.category); tagIsAi.push(false);
    });
    data.llm_tags.forEach(function (t) {
      tagKeys.push("a:" + t); tagLabel.push(t); tagCat.push(null); tagIsAi.push(true);
    });
    var T = nP + nA;
    W = Math.max(1, Math.ceil(T / 32));
    var n = 0;
    G.forEach(function (g) { n += g.i.length; });
    isoBits = new Uint32Array(n * W);
    isoCats = new Uint8Array(n);
    isoOrf = new Uint8Array(n);
    var orfIdx = {};
    tagCount = new Uint32Array(T);
    kwCount = new Uint32Array(data.keywords.length);
    var catIdx = {};
    data.groups.forEach(function (gr, k) { catIdx[gr.letter] = k; });

    G.forEach(function (g, gi) {
      g._lname = (g.n + " " + (g.u || "")).toLowerCase();
      g._kw = new Set(g.k);
      g.k.forEach(function (k) { kwCount[k]++; });
      g._first = ISO.length;
      var catTags = data.groups.map(function () { return new Set(); });
      var uncat = new Set();
      g.i.forEach(function (iso, ii) {
        var x = ISO.length;
        ISO.push(iso);
        iso._gi = gi; iso._ii = ii; iso._x = x;
        var mask = 0;
        for (var k = 0; k < data.groups.length; k++) {
          if (STATE_FIRES[iso.f.charAt(3 * k)]) mask |= 1 << k;
        }
        isoCats[x] = mask;
        if (!(iso.o in orfIdx)) { orfIdx[iso.o] = orfTypes.length; orfTypes.push(iso.o); orfCount.push(0); }
        isoOrf[x] = orfIdx[iso.o];
        orfCount[orfIdx[iso.o]]++;
        iso.g.forEach(function (ti) {
          isoBits[x * W + (ti >> 5)] |= 1 << (ti & 31);
          tagCount[ti]++;
          var c = data.tags[ti].category;
          if (c in catIdx) catTags[catIdx[c]].add(ti); else uncat.add(ti);
        });
        iso.y.forEach(function (ai) {
          var ti = nP + ai;
          isoBits[x * W + (ti >> 5)] |= 1 << (ti & 31);
          tagCount[ti]++;
        });
      });
      g._catCounts = catTags.map(function (s) { return s.size; });
      g._uncat = uncat.size;
    });
  }

  // ---------------------------------------------------------------- filters
  var F = { q: "", kw: [], kwAnd: true, tags: [], tagAnd: true, cats: [], catAnd: true, orfs: [],
            sort: "name" };
  var open = new Set(); // gene names expanded
  var showAll = new Set(); // gene names showing every isoform under a filter
  var openChips = new Set(); // isoform keys with all chips shown
  var openKw = new Set(); // gene names with all keywords shown
  var visible = []; // [{gi, matched: [ii...]}]

  function isoFilterActive() { return F.tags.length > 0 || F.cats.length > 0 || F.orfs.length > 0; }

  function isoMatches(x) {
    if (F.orfs.length && F.orfs.indexOf(isoOrf[x]) < 0) return false;
    if (F.tags.length) {
      var hit = F.tagAnd;
      for (var j = 0; j < F.tags.length; j++) {
        var ti = F.tags[j];
        var on = (isoBits[x * W + (ti >> 5)] >>> (ti & 31)) & 1;
        if (F.tagAnd && !on) { hit = false; break; }
        if (!F.tagAnd && on) { hit = true; break; }
      }
      if (!hit) return false;
    }
    if (F.cats.length) {
      var m = isoCats[x], want = 0;
      F.cats.forEach(function (k) { want |= 1 << k; });
      if (F.catAnd ? (m & want) !== want : (m & want) === 0) return false;
    }
    return true;
  }

  function applyFilters() {
    var q = F.q.trim().toLowerCase();
    var active = isoFilterActive();
    var out = [];
    var nIso = 0;
    for (var gi = 0; gi < G.length; gi++) {
      var g = G[gi];
      if (q && g._lname.indexOf(q) < 0) continue;
      if (F.kw.length) {
        var ok = F.kwAnd;
        for (var j = 0; j < F.kw.length; j++) {
          var has = g._kw.has(F.kw[j]);
          if (F.kwAnd && !has) { ok = false; break; }
          if (!F.kwAnd && has) { ok = true; break; }
        }
        if (!ok) continue;
      }
      var matched = null;
      if (active) {
        matched = [];
        for (var ii = 0; ii < g.i.length; ii++) {
          if (isoMatches(g._first + ii)) matched.push(ii);
        }
        if (!matched.length) continue;
      }
      nIso += matched ? matched.length : g.i.length;
      out.push({ gi: gi, matched: matched });
    }
    if (F.sort === "firing") {
      out.sort(function (a, b) {
        var A = G[a.gi], B = G[b.gi];
        return B.ns - A.ns || B.ni - A.ni || (A.n < B.n ? -1 : A.n > B.n ? 1 : 0);
      });
    }
    visible = out;
    var filtering = q || F.kw.length || active;
    resultCount.hidden = false;
    resultCount.textContent = out.length + " gene" + (out.length === 1 ? "" : "s") + " · " +
      nIso + (active ? " matching" : "") + " isoform" + (nIso === 1 ? "" : "s") +
      (filtering ? " (of " + D.n_genes + " genes)" : "");
    noResults.hidden = out.length > 0;
    rebuildRows();
    saveUrl();
  }

  // -------------------------------------------------------------- row list
  var rows = []; // [{t:'g'|'h'|'i'|'m', gi, ii, key}]
  var heights = new Map(); // key -> measured px
  var offsets = new Float64Array(1); // prefix sums, length rows+1

  function rebuildRows() {
    rows = [];
    var active = isoFilterActive();
    visible.forEach(function (v) {
      var g = G[v.gi];
      rows.push({ t: "g", gi: v.gi, key: "g:" + g.n });
      if (!open.has(g.n)) return;
      // The column sub-header is hidden on narrow screens (rows stack), so it
      // takes no row there rather than a reserved-but-empty one.
      if (!NARROW.matches) rows.push({ t: "h", gi: v.gi, key: "h:" + g.n });
      var all = !active || showAll.has(g.n);
      var list = all ? g.i.map(function (_, ii) { return ii; }) : v.matched;
      list.forEach(function (ii) {
        rows.push({ t: "i", gi: v.gi, ii: ii, key: "i:" + g.n + ":" + ii,
                    hl: active && all && v.matched.indexOf(ii) >= 0 });
      });
      if (active && v.matched.length < g.i.length) {
        rows.push({ t: "m", gi: v.gi, key: "m:" + g.n, all: all, n: g.i.length, k: v.matched.length });
      }
      // A gap closes the expanded block, so the next gene reads as separate.
      rows.push({ t: "e", gi: v.gi, key: "e:" + g.n });
    });
    relayout();
  }

  function relayout() {
    offsets = new Float64Array(rows.length + 1);
    for (var r = 0; r < rows.length; r++) {
      var h = heights.get(rows[r].key);
      offsets[r + 1] = offsets[r] + (h || EST[rows[r].t]);
    }
    body.style.height = offsets[rows.length] + "px";
    table.setAttribute("aria-rowcount", String(rows.length + 1));
    dirty = true;
    schedule();
  }

  function firstRowAt(y) { // last r with offsets[r] <= y
    var lo = 0, hi = rows.length;
    while (lo < hi) {
      var mid = (lo + hi + 1) >> 1;
      if (offsets[mid] <= y) lo = mid; else hi = mid - 1;
    }
    return Math.min(lo, Math.max(0, rows.length - 1));
  }

  // -------------------------------------------------------------- rendering
  function flagsHtml(f) {
    var out = '<span class="cdlmps">';
    D.groups.forEach(function (gr, k) {
      var code = f.charAt(3 * k), nMet = f.charAt(3 * k + 1), nEv = f.charAt(3 * k + 2);
      var state = D.flag_codes[code] || "pending";
      var label = state === "pending" ? "Not assessed"
        : D.legacy_labels[state] || (nMet + "/" + nEv + " criteria met");
      out += '<span class="cdlmps-letter cdlmps-' + state + '" title="' + esc(gr.name + ": " + label) +
        '">' + gr.letter + "</span>";
    });
    return out + "</span>";
  }

  function orfChip(orf) {
    var c = orf === "extended" ? ["EXT", "ext"] : orf === "truncated" ? ["TRUNC", "trunc"]
      : [String(orf || "?").toUpperCase(), "other"];
    return '<span class="iso-chip ' + c[1] + '">' + esc(c[0]) + "</span>";
  }

  function isoLabel(iso, ti) {
    var t = D.tags[ti];
    return (iso.l && iso.l[ti]) || (t.labels && t.labels[iso.o]) || t.label;
  }

  function chipsHtml(iso, key) {
    var n = iso.g.length;
    if (!n) return '<span class="gt-none">none fired</span>';
    var all = openChips.has(key);
    var show = all ? iso.g : iso.g.slice(0, CHIP_N);
    var catName = {};
    D.groups.forEach(function (gr) { catName[gr.letter] = gr.name; });
    var out = show.map(function (ti) {
      var t = D.tags[ti];
      var cls = "gt-chip" + (t.category ? " cat-" + t.category : " gt-chip-uncat");
      var tip = isoLabel(iso, ti) + " — " + (catName[t.category] || "uncategorised") +
        (t.kind === "derived" ? " criterion " + (t.criterion || "") : " " + (t.kind || "") + " tag");
      return '<span class="' + cls + '" title="' + esc(tip) + '">' + esc(isoLabel(iso, ti)) + "</span>";
    }).join("");
    if (n > CHIP_N) {
      out += '<button type="button" class="gt-more-chips" data-chips="' + esc(key) + '">' +
        (all ? "less" : "+" + (n - CHIP_N)) + "</button>";
    }
    return out;
  }

  // Gene-row badge tooltip: these are distinct tags across ALL isoforms, not the
  // Best isoform's and not a sum — say so, since the Best column sits beside it.
  function countTip(g, catName, c) {
    var n = g.i.length;
    return c + " different " + catName + " tag" + (c === 1 ? "" : "s") + " fired across " +
      (n === 1 ? "this gene's isoform" : "all " + n + " of this gene's isoforms") +
      (n === 1 ? "" : " — each tag counted once, however many isoforms fire it") +
      ". Expand the gene to see them per isoform.";
  }

  function catCountsHtml(g) {
    var out = "";
    D.groups.forEach(function (gr, k) {
      var c = g._catCounts[k];
      if (!c) return;
      out += '<span class="gt-catcount cat-' + gr.letter + '" title="' + esc(countTip(g, gr.name, c)) +
        '">' + gr.letter + "<b>" + c + "</b></span>";
    });
    if (g._uncat) {
      out += '<span class="gt-catcount gt-chip-uncat" title="' + esc(countTip(g, "uncategorised", g._uncat)) +
        '">?<b>' + g._uncat + "</b></span>";
    }
    return out || '<span class="gt-none">none</span>';
  }

  // Which isoform the gene's "Best" lineup belongs to, and why it was picked.
  function bestTip(g, best) {
    var kind = best.o === "extended" ? "extension" : best.o === "truncated" ? "truncation" : best.o;
    var tip = "Best isoform: " + kind + " " + best.t + " (" + best.a + " aa), net CDLMPS score " + best.ns;
    if (g.i.length > 1) tip += " — the highest of this gene's " + g.i.length + " isoforms";
    return tip + ". Hover a letter for that category.";
  }

  function geneRowHtml(row) {
    var g = G[row.gi];
    var isOpen = open.has(g.n);
    var nExt = 0, nTr = 0;
    g.i.forEach(function (iso) { if (iso.o === "extended") nExt++; else if (iso.o === "truncated") nTr++; });
    var nOther = g.i.length - nExt - nTr;
    var split = (nExt ? '<span class="chip-orf chip-extended">' + nExt + " ext</span>" : "") +
      (nTr ? '<span class="chip-orf chip-truncated">' + nTr + " trunc</span>" : "") +
      (nOther ? '<span class="chip-orf chip-other">' + nOther + " other</span>" : "");
    var kwAll = openKw.has(g.n);
    var kws = kwAll ? g.k : g.k.slice(0, KW_N);
    var kwHtml = kws.map(function (k) {
      var kw = D.keywords[k];
      return '<button type="button" class="kw-chip" data-kw="' + k + '" title="Add “' + esc(kw) +
        '” to the function filter">' + esc(kw) + "</button>";
    }).join("");
    if (g.k.length > KW_N) {
      kwHtml += '<button type="button" class="kw-toggle" data-kwmore="' + esc(g.n) + '">' +
        (kwAll ? "less" : "+" + (g.k.length - KW_N) + " more") + "</button>";
    }
    var best = g.i[g.b];
    return '<div class="gt-row gt-gene' + (isOpen ? " is-open" : "") + '" role="row" data-key="' + esc(row.key) + '">' +
      '<span class="gt-c gt-c-toggle" role="cell"><button type="button" class="gt-toggle" data-gene="' + esc(g.n) +
        '" aria-expanded="' + isOpen + '" aria-label="' + (isOpen ? "Collapse " : "Expand ") + esc(g.n) +
        ' isoforms">' + (isOpen ? "▾" : "▸") + "</button></span>" +
      '<span class="gt-c gt-c-gene" role="cell"><a href="' + esc(geneHref(g.n)) + '">' + esc(g.n) + "</a></span>" +
      '<span class="gt-c gt-c-uniprot" role="cell">' + (g.u ? '<a href="https://www.uniprot.org/uniprotkb/' +
        encodeURIComponent(g.u) + '/entry" target="_blank" rel="noopener">' + esc(g.u) + "</a>" : "") + "</span>" +
      '<span class="gt-c gt-c-iso" role="cell"><span class="gt-n">' + g.i.length + "</span>" + split + "</span>" +
      '<span class="gt-c gt-c-flags" role="cell"' + (best ? ' title="' + esc(bestTip(g, best)) + '"' : "") +
        ">" + (best ? flagsHtml(best.f) : "") + "</span>" +
      '<span class="gt-c gt-c-tags" role="cell">' + catCountsHtml(g) + "</span>" +
      '<span class="gt-c gt-c-kw" role="cell">' + kwHtml + "</span>" +
      "</div>";
  }

  function subHeadHtml(row) {
    return '<div class="gt-row gt-subhead" role="row" data-key="' + esc(row.key) + '">' +
      '<span class="gt-c gt-c-toggle" role="cell"></span>' +
      '<span class="gt-c gt-c-tis" role="columnheader">Isoform</span>' +
      '<span class="gt-c gt-c-len" role="columnheader">Length · AI tags</span>' +
      '<span class="gt-c gt-c-flags" role="columnheader">CDLMPS</span>' +
      '<span class="gt-c gt-c-fired" role="columnheader">Fired tags, by category</span>' +
      "</div>";
  }

  function isoRowHtml(row) {
    var g = G[row.gi], iso = g.i[row.ii];
    var ai = iso.y.map(function (k) {
      return '<span class="iso-tag-chip sm" title="AI synthesis tag">' + esc(D.llm_tags[k]) + "</span>";
    }).join("");
    return '<div class="gt-row gt-iso' + (row.hl ? " is-match" : "") + '" role="row" data-key="' + esc(row.key) + '">' +
      '<span class="gt-c gt-c-toggle" role="cell"></span>' +
      '<span class="gt-c gt-c-tis" role="cell">' + orfChip(iso.o) +
        '<a class="gt-iso-link" href="' + esc(isoHref(g.n, iso.s)) + '" title="' + esc(iso.t) + '">' +
        esc(iso.t) + "</a></span>" +
      '<span class="gt-c gt-c-len" role="cell"><span class="gt-aa">' + iso.a + " aa</span>" + ai + "</span>" +
      '<span class="gt-c gt-c-flags" role="cell">' + flagsHtml(iso.f) + "</span>" +
      '<span class="gt-c gt-c-fired" role="cell">' + chipsHtml(iso, row.key) + "</span>" +
      "</div>";
  }

  function moreRowHtml(row) {
    var g = G[row.gi];
    var txt = row.all ? "Show only the " + row.k + " matching isoform" + (row.k === 1 ? "" : "s")
      : "Show all " + row.n + " isoforms (" + (row.n - row.k) + " not matching)";
    return '<div class="gt-row gt-more" role="row" data-key="' + esc(row.key) + '">' +
      '<span class="gt-c gt-c-toggle" role="cell"></span>' +
      '<span class="gt-c gt-c-wide" role="cell"><button type="button" class="gt-showall" data-gene="' +
      esc(g.n) + '">' + txt + "</button></span></div>";
  }

  function gapRowHtml(row) {
    return '<div class="gt-row gt-gap" role="row" aria-hidden="true" data-key="' + esc(row.key) + '"></div>';
  }

  var RENDER = { g: geneRowHtml, h: subHeadHtml, i: isoRowHtml, m: moreRowHtml, e: gapRowHtml };
  var pending = false;
  var dirty = true; // rows or layout changed since the last draw
  var drawn0 = -1, drawn1 = -1; // the row range currently in the DOM
  function schedule() {
    if (pending) return;
    pending = true;
    requestAnimationFrame(function () { pending = false; render(0); });
  }

  // Which control has focus, as (row key, index among the row's controls), so a
  // redraw that replaces the row's elements can put focus back on its twin.
  function focusSpot() {
    var a = document.activeElement;
    if (!a || !body.contains(a)) return null;
    var rowEl = a.closest("[data-key]");
    if (!rowEl) return null;
    var ctl = Array.prototype.indexOf.call(rowEl.querySelectorAll("a,button"), a);
    return ctl < 0 ? null : { key: rowEl.getAttribute("data-key"), ctl: ctl };
  }
  function restoreFocus(spot) {
    if (!spot) return;
    for (var e = 0; e < body.children.length; e++) {
      if (body.children[e].getAttribute("data-key") !== spot.key) continue;
      var el = body.children[e].querySelectorAll("a,button")[spot.ctl];
      if (el) el.focus({ preventScroll: true });
      return;
    }
  }

  function render(pass) {
    if (!rows.length) { body.innerHTML = ""; drawn0 = drawn1 = -1; dirty = false; return; }
    var top = body.getBoundingClientRect().top + window.scrollY;
    var y0 = Math.max(0, window.scrollY - top - OVERSCAN);
    var y1 = window.scrollY - top + window.innerHeight + OVERSCAN;
    var r0 = firstRowAt(y0), r1 = r0;
    while (r1 < rows.length && offsets[r1] < y1) r1++;
    // Scrolling within the drawn window changes nothing: keep the DOM (and focus).
    if (!dirty && r0 === drawn0 && r1 === drawn1) return;
    var spot = focusSpot();
    var html = "";
    for (var r = r0; r < r1; r++) {
      html += RENDER[rows[r].t](rows[r]).replace(
        'role="row"', 'role="row" aria-rowindex="' + (r + 2) + '" style="top:' + offsets[r] + 'px"'
      );
    }
    body.innerHTML = html;
    drawn0 = r0; drawn1 = r1; dirty = false;
    restoreFocus(spot);
    // Measure what was drawn; if any row's real height differs from the cached
    // one, re-layout and draw again (at most twice per frame).
    var changed = false;
    var els = body.children;
    for (var e = 0; e < els.length; e++) {
      var key = els[e].getAttribute("data-key");
      var h = els[e].offsetHeight;
      if (heights.get(key) !== h) { heights.set(key, h); changed = true; }
    }
    if (changed && pass < 2) {
      offsets = new Float64Array(rows.length + 1);
      for (var k = 0; k < rows.length; k++) {
        offsets[k + 1] = offsets[k] + (heights.get(rows[k].key) || EST[rows[k].t]);
      }
      body.style.height = offsets[rows.length] + "px";
      dirty = true;
      render(pass + 1);
    }
  }

  window.addEventListener("scroll", schedule, { passive: true });
  // Phones resize on scroll (the URL bar shows and hides): only a width change
  // re-wraps rows, so only then are the measured heights thrown away.
  var lastWidth = window.innerWidth, resizeTimer = 0;
  window.addEventListener("resize", function () {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(function () {
      if (!G) return;
      if (window.innerWidth !== lastWidth) {
        lastWidth = window.innerWidth;
        heights.clear();
        rebuildRows();
      } else {
        schedule();
      }
    }, 150);
  });

  // ---------------------------------------------------------- row actions
  body.addEventListener("click", function (ev) {
    var b = ev.target.closest("button");
    if (!b) return;
    if (b.classList.contains("gt-toggle")) {
      var name = b.getAttribute("data-gene");
      if (open.has(name)) open.delete(name); else open.add(name);
      saveSession(); rebuildRows();
    } else if (b.classList.contains("gt-showall")) {
      var n = b.getAttribute("data-gene");
      if (showAll.has(n)) showAll.delete(n); else showAll.add(n);
      saveSession(); rebuildRows();
    } else if (b.hasAttribute("data-chips")) {
      var key = b.getAttribute("data-chips");
      if (openChips.has(key)) openChips.delete(key); else openChips.add(key);
      heights.delete(key); relayout();
    } else if (b.hasAttribute("data-kwmore")) {
      var gname = b.getAttribute("data-kwmore");
      if (openKw.has(gname)) openKw.delete(gname); else openKw.add(gname);
      heights.delete("g:" + gname); relayout();
    } else if (b.classList.contains("kw-chip")) {
      kwFacet.add(Number(b.getAttribute("data-kw")));
    }
  });

  // ------------------------------------------------------------ facets
  // One typeahead + chip-list implementation for both facets (the card grid had
  // two copies of this). ``options()`` returns [{v, label, count, badge}].
  function Facet(cfg) {
    var input = document.getElementById(cfg.input);
    var dropdown = document.getElementById(cfg.dropdown);
    var controls = document.getElementById(cfg.controls);
    var selectedEl = document.getElementById(cfg.selected);
    var clearBtn = document.getElementById(cfg.clear);
    var self = this;
    this.sel = cfg.sel; // shared array in F

    function renderSelected() {
      selectedEl.innerHTML = self.sel.map(function (v) {
        return '<span class="kw-tag">' + cfg.badge(v) + esc(cfg.label(v)) +
          '<button type="button" class="kw-tag-x" data-v="' + v + '" aria-label="Remove">×</button></span>';
      }).join("");
      controls.hidden = self.sel.length === 0;
    }
    // Hiding also empties it, so Enter can never pick an option nobody can see.
    function hideDropdown() { dropdown.hidden = true; dropdown.innerHTML = ""; }
    function renderDropdown() {
      var q = input.value.trim().toLowerCase();
      var opts = cfg.options().filter(function (o) {
        return self.sel.indexOf(o.v) < 0 && (!q || o.search.indexOf(q) >= 0);
      }).slice(0, 500);
      if (!opts.length || document.activeElement !== input) { hideDropdown(); return; }
      dropdown.innerHTML = opts.map(function (o) {
        return '<button type="button" class="kw-option" data-v="' + o.v + '">' + o.badge +
          '<span class="kw-option-name">' + esc(o.label) + '</span><span class="kw-option-count">' +
          o.count + "</span></button>";
      }).join("");
      dropdown.hidden = false;
    }
    this.add = function (v) {
      if (self.sel.indexOf(v) < 0) self.sel.push(v);
      input.value = ""; hideDropdown();
      renderSelected(); applyFilters();
    };
    this.sync = renderSelected;
    input.addEventListener("input", renderDropdown);
    input.addEventListener("focus", renderDropdown);
    input.addEventListener("blur", function () { setTimeout(hideDropdown, 120); });
    input.addEventListener("keydown", function (e) {
      if (e.key === "Enter") {
        var first = dropdown.hidden ? null : dropdown.querySelector(".kw-option");
        if (first) { e.preventDefault(); self.add(Number(first.getAttribute("data-v"))); }
      } else if (e.key === "Escape") { hideDropdown(); }
    });
    dropdown.addEventListener("mousedown", function (e) {
      var o = e.target.closest(".kw-option");
      if (o) { e.preventDefault(); self.add(Number(o.getAttribute("data-v"))); }
    });
    selectedEl.addEventListener("click", function (e) {
      var x = e.target.closest(".kw-tag-x");
      if (!x) return;
      var v = Number(x.getAttribute("data-v"));
      self.sel.splice(self.sel.indexOf(v), 1);
      renderSelected(); applyFilters();
    });
    clearBtn.addEventListener("click", function () {
      self.sel.length = 0; renderSelected(); applyFilters();
    });
    document.querySelectorAll('input[name="' + cfg.mode + '"]').forEach(function (r) {
      r.addEventListener("change", function () { F[cfg.modeKey] = r.value === "and"; applyFilters(); });
    });
  }

  function wireOrf(orfBox) {
    // ORF type toggles, built from the types present, in the order the gene
    // rows list isoforms (extension, truncation, then by frequency).
    var ORF_LABEL = { extended: "Extension", truncated: "Truncation", uorf: "uORF", uoorf: "uoORF",
                      internal_oof: "Internal OOF", "3utr_orf": "3′UTR ORF", alt_orf: "Alt ORF" };
    var ORF_RANK = { extended: 0, truncated: 1 };
    var orfOrder = orfTypes.map(function (_, k) { return k; }).sort(function (a, b) {
      var ra = ORF_RANK[orfTypes[a]] == null ? 2 : ORF_RANK[orfTypes[a]];
      var rb = ORF_RANK[orfTypes[b]] == null ? 2 : ORF_RANK[orfTypes[b]];
      return ra - rb || orfCount[b] - orfCount[a];
    });
    orfBox.innerHTML = orfOrder.map(function (k) {
      var t = orfTypes[k];
      return '<button type="button" class="orf-toggle' + (F.orfs.indexOf(k) >= 0 ? " active" : "") +
        '" data-orf="' + k + '" aria-pressed="' + (F.orfs.indexOf(k) >= 0) + '" title="' +
        esc(orfCount[k] + " " + t + " isoform" + (orfCount[k] === 1 ? "" : "s")) + '">' +
        esc(ORF_LABEL[t] || t || "unknown") + '<span class="orf-n">' + orfCount[k] + "</span></button>";
    }).join("");
    orfBox.addEventListener("click", function (e) {
      var b = e.target.closest(".orf-toggle");
      if (!b) return;
      var k = Number(b.getAttribute("data-orf"));
      var at = F.orfs.indexOf(k);
      if (at >= 0) F.orfs.splice(at, 1); else F.orfs.push(k);
      b.classList.toggle("active", at < 0);
      b.setAttribute("aria-pressed", String(at < 0));
      applyFilters();
    });
  }

  var kwFacet, tagFacet;
  function wireControls() {
    var kwOptions = null, tagOptions = null;
    kwFacet = new Facet({
      input: "kw-input", dropdown: "kw-dropdown", controls: "kw-controls", selected: "kw-selected",
      clear: "kw-clear", mode: "kwmode", modeKey: "kwAnd", sel: F.kw,
      label: function (v) { return D.keywords[v]; },
      badge: function () { return ""; },
      options: function () {
        if (!kwOptions) {
          kwOptions = D.keywords.map(function (kw, k) {
            return { v: k, label: kw, count: kwCount[k], badge: "", search: kw.toLowerCase() };
          }).sort(function (a, b) { return b.count - a.count; });
        }
        return kwOptions;
      },
    });
    function tagBadge(v) {
      return tagIsAi[v] ? '<span class="gt-badge gt-badge-ai">AI</span>'
        : '<span class="gt-badge' + (tagCat[v] ? " cat-" + tagCat[v] : " gt-chip-uncat") + '">' +
          esc(tagCat[v] || "?") + "</span>";
    }
    tagFacet = new Facet({
      input: "tag-input", dropdown: "tag-dropdown", controls: "tag-controls", selected: "tag-selected",
      clear: "tag-clear", mode: "tagmode", modeKey: "tagAnd", sel: F.tags,
      label: function (v) { return tagLabel[v]; },
      badge: tagBadge,
      options: function () {
        if (!tagOptions) {
          tagOptions = [];
          for (var v = 0; v < tagKeys.length; v++) {
            if (!tagCount[v]) continue;
            tagOptions.push({ v: v, label: tagLabel[v], count: tagCount[v], badge: tagBadge(v),
                              search: (tagLabel[v] + " " + tagKeys[v]).toLowerCase() });
          }
          tagOptions.sort(function (a, b) { return b.count - a.count; });
        }
        return tagOptions;
      },
    });

    var search = document.getElementById("gene-search");
    search.value = F.q;
    var t = null;
    search.addEventListener("input", function () {
      clearTimeout(t);
      t = setTimeout(function () { F.q = search.value; applyFilters(); }, 80);
    });

    var catMode = document.getElementById("cat-mode");
    document.querySelectorAll(".cat-toggle").forEach(function (b) {
      var k = D.groups.findIndex(function (gr) { return gr.letter === b.getAttribute("data-cat"); });
      b.classList.toggle("active", F.cats.indexOf(k) >= 0);
      b.addEventListener("click", function () {
        var at = F.cats.indexOf(k);
        if (at >= 0) F.cats.splice(at, 1); else F.cats.push(k);
        b.classList.toggle("active", at < 0);
        catMode.hidden = F.cats.length < 2;
        applyFilters();
      });
    });
    catMode.hidden = F.cats.length < 2;
    document.querySelectorAll('input[name="catmode"]').forEach(function (r) {
      r.addEventListener("change", function () { F.catAnd = r.value === "and"; applyFilters(); });
    });

    var orfBox = document.getElementById("orf-toggles");
    if (orfBox) wireOrf(orfBox);

    var sort = document.getElementById("sort-mode");
    sort.value = F.sort;
    sort.addEventListener("change", function () { F.sort = sort.value; applyFilters(); });

    function setMode(name, and) {
      var r = document.querySelector('input[name="' + name + '"][value="' + (and ? "and" : "or") + '"]');
      if (r) r.checked = true;
    }
    setMode("kwmode", F.kwAnd); setMode("tagmode", F.tagAnd); setMode("catmode", F.catAnd);
    kwFacet.sync(); tagFacet.sync();
  }

  // ------------------------------------------------------------ state
  // Filters live in the URL (shareable, survive reload); which genes are open
  // and the scroll position live in sessionStorage so Back from an isoform page
  // lands where the reader left.
  var SESSION_KEY = "gt:" + location.pathname;

  function readUrl() {
    var p = new URLSearchParams(location.search);
    F.q = p.get("q") || "";
    var kwIdx = {};
    D.keywords.forEach(function (k, i) { kwIdx[k] = i; });
    (p.get("kw") || "").split("|").forEach(function (k) { if (k in kwIdx) F.kw.push(kwIdx[k]); });
    var tIdx = {};
    tagKeys.forEach(function (k, i) { tIdx[k] = i; });
    (p.get("tag") || "").split("|").forEach(function (k) { if (k in tIdx) F.tags.push(tIdx[k]); });
    (p.get("cat") || "").split("").forEach(function (c) {
      var k = D.groups.findIndex(function (gr) { return gr.letter === c; });
      if (k >= 0 && F.cats.indexOf(k) < 0) F.cats.push(k);
    });
    (p.get("orf") || "").split(",").forEach(function (o) {
      var k = orfTypes.indexOf(o);
      if (o && k >= 0 && F.orfs.indexOf(k) < 0) F.orfs.push(k);
    });
    F.kwAnd = p.get("km") !== "or"; F.tagAnd = p.get("tm") !== "or"; F.catAnd = p.get("cm") !== "or";
    F.sort = p.get("sort") === "firing" ? "firing" : "name";
  }

  function saveUrl() {
    var p = new URLSearchParams(location.search);
    function set(k, v) { if (v) p.set(k, v); else p.delete(k); }
    set("q", F.q.trim());
    set("kw", F.kw.map(function (k) { return D.keywords[k]; }).join("|"));
    set("tag", F.tags.map(function (k) { return tagKeys[k]; }).join("|"));
    set("cat", F.cats.map(function (k) { return D.groups[k].letter; }).join(""));
    set("orf", F.orfs.map(function (k) { return orfTypes[k]; }).join(","));
    set("km", F.kwAnd ? "" : "or"); set("tm", F.tagAnd ? "" : "or"); set("cm", F.catAnd ? "" : "or");
    set("sort", F.sort === "firing" ? "firing" : "");
    var qs = p.toString();
    var url = location.pathname + (qs ? "?" + qs : "") + location.hash;
    if (url !== location.pathname + location.search + location.hash) {
      store(function () { history.replaceState(history.state, "", url); });
    }
  }

  // Scroll is saved as the row at the top of the viewport plus how far into it,
  // not as raw pixels: after a reload unmeasured rows use estimated heights, so
  // the same pixel offset would land on a different row (far off on a phone).
  function bodyTop() { return body.getBoundingClientRect().top + window.scrollY; }
  function topAnchor() {
    var y = window.scrollY - bodyTop();
    if (!rows.length || y <= 0) return null;
    var r = firstRowAt(y);
    return { key: rows[r].key, dy: y - offsets[r] };
  }
  function anchorY(a) {
    for (var r = 0; r < rows.length; r++) {
      if (rows[r].key === a.key) return bodyTop() + offsets[r] + a.dy;
    }
    return null;
  }

  function saveSession() {
    store(function () {
      sessionStorage.setItem(SESSION_KEY, JSON.stringify({
        open: Array.from(open), all: Array.from(showAll), y: window.scrollY, anchor: topAnchor(),
      }));
    });
  }

  // Only Back/Forward returns to where the reader was; a reload or a fresh visit
  // starts collapsed at the top.
  function readSession() {
    var s = store(function () { return JSON.parse(sessionStorage.getItem(SESSION_KEY) || "null"); });
    if (!s) return null;
    (s.open || []).forEach(function (n) { open.add(n); });
    (s.all || []).forEach(function (n) { showAll.add(n); });
    return s;
  }

  window.addEventListener("pagehide", saveSession);

  // ------------------------------------------------------------ boot
  fetch(table.dataset.src, { credentials: "same-origin" })
    .then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    })
    .then(function (data) {
      build(data);
      readUrl();
      var nav = store(function () { return performance.getEntriesByType("navigation")[0].type; });
      var saved = nav === "back_forward" ? readSession() : null;
      wireControls();
      applyFilters();
      loading.remove(); // only once the table is built, so a failure above still shows
      if (saved && (saved.anchor || saved.y)) {
        if ("scrollRestoration" in history) history.scrollRestoration = "manual";
        var restore = function () {
          var y = saved.anchor ? anchorY(saved.anchor) : null;
          window.scrollTo(0, y == null ? saved.y : y);
        };
        // Lay out, jump to the anchor, then correct once the rows drawn around it
        // have been measured (their real heights move the anchor's offset).
        requestAnimationFrame(function () {
          restore();
          requestAnimationFrame(function () { requestAnimationFrame(restore); });
        });
      }
    })
    .catch(function (e) {
      if (!loading.isConnected) body.after(loading);
      loading.textContent = "Could not load the gene table (" + e.message + "). Reload to try again.";
    });
})();
