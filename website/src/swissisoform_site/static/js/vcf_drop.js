/* Landing-page VCF drop zone.
 *
 * Posts a VCF to /api/variants/scan and navigates to the results page it names.
 * Unlike the gene filters beside it, this talks to the server: resolving a
 * variant needs the ORF index, which is far too large to ship to the browser.
 *
 * The response JSON is logged to the browser console on every reply, success or
 * failure — the fastest way to inspect what the backend actually returned, and it
 * leaks nothing (the payload is already here).
 *
 * No spinner: the codebase has no animation anywhere, and a scan of 237k records
 * measures ~0.5 s server-side, so a status line and a border change suffice.
 */
(function () {
  "use strict";

  var zone = document.getElementById("vq-drop");
  var input = document.getElementById("vq-file");
  var status = document.getElementById("vq-status");
  if (!zone || !input || !status) return;

  var busy = false;

  function setState(cls, message) {
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

  function upload(file) {
    if (busy || !file) return;
    busy = true;
    setState("is-busy", "Scanning " + file.name + " (" + humanBytes(file.size) + ")…");

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
            var counts = (payload && payload.counts) || {};
            setState(
              "is-busy",
              "Found " +
                (counts.hits || 0) +
                " hits in " +
                (counts.genes_hit || 0) +
                " genes — opening results…"
            );
            window.location.href =
              (payload && payload.redirect) || "/variants/" + payload.vcf_id;
          });
      })
      .catch(function (err) {
        busy = false;
        console.error("[swissisoform] scan request failed", err);
        setState("is-error", "Could not reach the server.");
      });
  }

  input.addEventListener("change", function () {
    if (input.files && input.files.length) upload(input.files[0]);
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
    if (files && files.length) upload(files[0]);
    else setState("is-error", "That drop carried no file.");
  });
})();
