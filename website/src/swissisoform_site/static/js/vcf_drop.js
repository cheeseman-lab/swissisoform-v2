/* Landing-page VCF drop zone.
 *
 * Scans a VCF in the browser (scan_worker.js — Pyodide running the server's own
 * scan code) so the file never leaves the machine, stores the result in this tab
 * (SwissScan.save, sessionStorage) and opens /variants, which reads it back. The
 * filename never leaves the browser either: it is taken from the File.
 *
 * Only when the browser cannot run the scanner at all (no Web Workers or
 * WebAssembly, or Pyodide failed to load) is the server offered — as a button the
 * user presses, never silently, because it means uploading the file. A file the
 * scanner refuses (too large, a runaway line, unreadable) is reported, not retried
 * on the server: the server would refuse it the same way. Nor is a page left open
 * across a deploy (its downloads are gone): reloading scans locally, so that is
 * what it is told to do.
 *
 * The result is logged to the browser console, success or failure — the fastest
 * way to inspect what the scan produced, and it leaks nothing.
 *
 * No spinner: the codebase has no animation anywhere, so a status line and a
 * border change suffice; the worker reports each stage as it starts.
 */
(function () {
  "use strict";

  var zone = document.getElementById("vq-drop");
  var input = document.getElementById("vq-file");
  var status = document.getElementById("vq-status");
  var serverButton = document.getElementById("vq-server");
  if (!zone || !input || !status) return;

  var STALE_MESSAGE =
    "The site was updated while this page was open. Reload and try again.";
  var busy = false;
  var worker = null;
  var pendingFile = null; // offered to the server after a scanner failure

  function setState(cls, message) {
    if (serverButton) serverButton.hidden = true;
    zone.classList.remove("is-over", "is-busy", "is-error");
    if (cls) zone.classList.add(cls);
    status.textContent = message || "";
  }

  function humanBytes(n) {
    if (n < 1024) return n + " B";
    if (n < 1024 * 1024) return (n / 1024).toFixed(0) + " KB";
    return (n / (1024 * 1024)).toFixed(1) + " MB";
  }

  /* Server error shapes are {error, message}; surface the message so a failure is
     diagnosable from the page rather than only from the logs. */
  function describe(payload, response) {
    if (payload && payload.message) return payload.message;
    if (payload && payload.error) return payload.error;
    return "Scan failed (HTTP " + response.status + ").";
  }

  /* Store the scan for this tab, then open the results page. */
  function keep(file, payload) {
    var scan = window.SwissScan;
    var expected = document.body.getAttribute("data-scan-schema");
    if (!scan || !payload || payload.schema !== expected) {
      // The page and the server disagree on the shape — a deploy landed between
      // loading this page and uploading.
      busy = false;
      setState("is-error", STALE_MESSAGE);
      return;
    }
    var summary = Object.assign({}, payload, {
      filename: file.name,
      scanned_at: new Date().toISOString(),
    });
    delete summary.hits;
    try {
      scan.save(summary, payload.hits || []);
    } catch (err) {
      busy = false;
      console.error("[swissisoform] could not store the scan", err);
      setState("is-error", "This browser would not store the scan (is site storage blocked?).");
      return;
    }
    var counts = payload.counts || {};
    setState(
      "is-busy",
      "Found " + (counts.hits || 0) + " hits in " + (counts.genes_hit || 0) +
        " genes — opening results…"
    );
    window.location.href = "/variants";
  }

  function dropWorker() {
    if (worker) worker.terminate();
    worker = null;
  }

  /* The scanner could not run in this browser: say why and offer the server. */
  function offerServer(file, why) {
    busy = false;
    console.error("[swissisoform] browser scan unavailable:", why);
    setState("is-error", "This browser could not run the scan (" + why + ").");
    if (serverButton && file) {
      pendingFile = file;
      serverButton.hidden = false;
    }
  }

  function scanInBrowser(file) {
    if (busy || !file) return;
    var cfg = zone.dataset;
    if (!cfg.index) {
      // No ORF index in this deployment: neither path can scan, so say so here
      // rather than uploading the file only for the server to refuse it.
      setState("is-error", "Variant scanning is unavailable: this deployment has no ORF index.");
      return;
    }
    if (typeof Worker === "undefined" || typeof WebAssembly === "undefined") {
      offerServer(file, "it has no Web Worker or WebAssembly support");
      return;
    }
    busy = true;
    setState("is-busy", "Scanning " + file.name + " (" + humanBytes(file.size) + ") in your browser…");
    try {
      // A module worker: Pyodide 314 will not boot in a classic one.
      worker = worker || new Worker(cfg.worker, { type: "module" });
    } catch (err) {
      offerServer(file, "the scan worker would not start");
      return;
    }
    worker.onmessage = function (event) {
      var msg = event.data;
      if (msg.type === "status") {
        setState("is-busy", msg.message);
      } else if (msg.type === "done") {
        var payload = JSON.parse(msg.json);
        console.log("[swissisoform] browser scan result", payload);
        if (payload.error) {
          busy = false;
          setState("is-error", describe(payload, { status: "n/a" }));
          return;
        }
        keep(file, payload);
      } else if (msg.type === "failed") {
        // The interpreter may be dead (out of memory, a fatal error) and would fail
        // every later scan; the next drop starts a fresh worker.
        dropWorker();
        if (msg.stale) {
          busy = false;
          setState("is-error", STALE_MESSAGE);
        } else if (msg.infra) {
          offerServer(file, msg.message);
        } else {
          busy = false;
          console.error("[swissisoform] browser scan failed", msg.message);
          setState("is-error", "The scan failed: " + msg.message);
        }
      }
    };
    worker.onerror = function (event) {
      dropWorker();
      offerServer(file, (event && event.message) || "the scan worker crashed");
    };
    worker.postMessage({
      file: file,
      pyodide: cfg.pyodide,
      scanner: cfg.scanner,
      index: cfg.index,
    });
  }

  if (serverButton) {
    serverButton.addEventListener("click", function () {
      var file = pendingFile;
      pendingFile = null;
      uploadToServer(file);
    });
  }

  /* The fallback: the server route scans the upload and keeps nothing. */
  function uploadToServer(file) {
    if (busy || !file) return;
    busy = true;
    setState("is-busy", "Uploading " + file.name + " (" + humanBytes(file.size) + ") to scan on the server…");

    var body = new FormData();
    body.append("vcf", file);

    fetch("/api/variants/scan", { method: "POST", body: body })
      .then(function (response) {
        // A 413 from Werkzeug and every handled error both return JSON, but a
        // proxy-level failure may not — fall back to text so we never throw here.
        return response
          .json()
          .catch(function () {
            return null;
          })
          .then(function (payload) {
            console.log(
              "[swissisoform] scan response " + response.status,
              payload
            );
            if (!response.ok) {
              busy = false;
              setState("is-error", describe(payload, response));
              return;
            }
            keep(file, payload);
          });
      })
      .catch(function (err) {
        busy = false;
        console.error("[swissisoform] scan request failed", err);
        setState("is-error", "Could not reach the server.");
      });
  }

  input.addEventListener("change", function () {
    if (input.files && input.files.length) scanInBrowser(input.files[0]);
  });

  // dragover must be cancelled on every event or the browser navigates to the file.
  ["dragenter", "dragover"].forEach(function (name) {
    zone.addEventListener(name, function (e) {
      e.preventDefault();
      e.stopPropagation();
      if (!busy) setState("is-over", "Release to scan.");
    });
  });

  ["dragleave", "dragend"].forEach(function (name) {
    zone.addEventListener(name, function (e) {
      e.preventDefault();
      e.stopPropagation();
      if (!busy) setState(null, "");
    });
  });

  zone.addEventListener("drop", function (e) {
    e.preventDefault();
    e.stopPropagation();
    var files = e.dataTransfer && e.dataTransfer.files;
    if (files && files.length) scanInBrowser(files[0]);
    else setState("is-error", "That drop carried no file.");
  });
})();
