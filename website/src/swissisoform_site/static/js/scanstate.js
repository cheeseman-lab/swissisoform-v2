/* The tab's VCF scan, held in sessionStorage — the only code that touches it.
 *
 * The server keeps nothing about a scan (issue #33, step 1): POST
 * /api/variants/scan returns the whole result and vcf_drop.js hands it to save().
 * Every page that shows scan context reads it back through load(). Two fixed keys,
 * one scan per tab; a new upload replaces the old one, and closing the tab ends it.
 *
 *   swissiso.scan.summary  {schema, filename, scanned_at, counts, genes,
 *                           provenance, derived, hits_stored}
 *   swissiso.scan.hits     [hit, …] — each with x and protein_change already set
 *
 * Two checks on every read, against values the server writes on <body>:
 *   data-scan-schema   a stored shape this build does not know is discarded;
 *   data-index-version a scan resolved against another ORF index is kept but
 *                      marked stale, so nothing draws it at coordinates that may
 *                      have moved.
 *
 * Also fills the scan breadcrumb and hit-count chips that templates leave empty:
 *   .js-scan-crumb                  shown when a scan exists (link to /variants)
 *   .js-scan-hits[data-gene]        "N variants from the uploaded VCF"
 *   .js-scan-hits[data-tis]         "N variants from the uploaded VCF hit this isoform"
 */
(function () {
  "use strict";

  var SUMMARY_KEY = "swissiso.scan.summary";
  var HITS_KEY = "swissiso.scan.hits";
  //: Hits kept when the full list does not fit — the same cap the results table uses.
  var PARTIAL_HITS = 1000;

  var cached; // undefined = not read yet; null = no usable scan

  function bodyData(name) {
    return (document.body && document.body.getAttribute("data-" + name)) || "";
  }

  function remove() {
    try {
      sessionStorage.removeItem(SUMMARY_KEY);
      sessionStorage.removeItem(HITS_KEY);
    } catch (e) {
      /* Storage disabled: there is nothing to remove. */
    }
  }

  function clear() {
    remove();
    cached = null;
  }

  /* Store one scan. Hits first and the summary last, so a hits write that fails
     part-way can never leave a summary claiming a full list. Returns
     "full" or "partial"; throws if even the summary cannot be stored. */
  function save(summary, hits) {
    remove();
    var stored = "full";
    try {
      sessionStorage.setItem(HITS_KEY, JSON.stringify(hits));
    } catch (e) {
      // QuotaExceededError (~5 MB per origin): keep what the results table shows.
      stored = "partial";
      try {
        sessionStorage.setItem(HITS_KEY, JSON.stringify(hits.slice(0, PARTIAL_HITS)));
      } catch (e2) {
        stored = "none";
      }
    }
    var record = Object.assign({}, summary, { hits_stored: stored });
    sessionStorage.setItem(SUMMARY_KEY, JSON.stringify(record));
    cached = undefined;
    return stored;
  }

  /* The stored scan as {summary, hits, stale}, or null. Parsed once per page. */
  function load() {
    if (cached !== undefined) return cached;
    cached = null;
    var rawSummary, rawHits;
    try {
      rawSummary = sessionStorage.getItem(SUMMARY_KEY);
      rawHits = sessionStorage.getItem(HITS_KEY);
    } catch (e) {
      return null;
    }
    if (!rawSummary) return null;
    var summary, hits;
    try {
      summary = JSON.parse(rawSummary);
      hits = rawHits ? JSON.parse(rawHits) : [];
    } catch (e) {
      clear();
      return null;
    }
    if (!summary || summary.schema !== bodyData("scan-schema") || !Array.isArray(hits)) {
      clear();
      return null;
    }
    var stored = (summary.provenance && summary.provenance.index_version) || "";
    cached = { summary: summary, hits: hits, stale: stored !== bodyData("index-version") };
    return cached;
  }

  function hitsWhere(field, value) {
    var scan = load();
    if (!scan) return [];
    return scan.hits.filter(function (h) {
      return h[field] === value;
    });
  }

  function plural(n, word) {
    return n + " " + word + (n === 1 ? "" : "s");
  }

  /* Distinct variants among hits: one variant inside three isoforms is 1. */
  function countVariants(hits) {
    var seen = {};
    hits.forEach(function (h) {
      seen[h.line_no + ":" + h.alt] = true;
    });
    return Object.keys(seen).length;
  }

  function fillChrome() {
    var scan = load();
    if (!scan) return;
    document.querySelectorAll(".js-scan-crumb").forEach(function (el) {
      el.hidden = false;
    });
    document.querySelectorAll(".js-scan-hits").forEach(function (el) {
      if (scan.stale) {
        el.textContent = "· scan made against an older catalogue — re-upload to see its variants";
      } else {
        var gene = el.getAttribute("data-gene");
        var tis = el.getAttribute("data-tis");
        var hits = gene ? hitsWhere("gene", gene) : hitsWhere("tis_id", tis);
        var text = "· " + plural(countVariants(hits), "variant") + " from the uploaded VCF";
        if (tis) text += " hit this isoform";
        if (scan.summary.hits_stored !== "full") text += " (of the hits this tab could keep)";
        el.textContent = text;
      }
      el.hidden = false;
    });
  }

  window.SwissScan = {
    save: save,
    load: load,
    clear: clear,
    hitsForGene: function (name) {
      return hitsWhere("gene", name);
    },
    hitsForTis: function (tisId) {
      return hitsWhere("tis_id", tisId);
    },
    countVariants: countVariants,
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", fillChrome);
  } else {
    fillChrome();
  }
})();
