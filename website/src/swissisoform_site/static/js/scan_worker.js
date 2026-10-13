/* The VCF scan, run in the browser (issue #33, step 2).
 *
 * A module Web Worker, so the page stays responsive while it runs — module, not
 * classic, because Pyodide 314 refuses to boot in a classic worker. It boots
 * Pyodide (pinned, from jsDelivr — browserscan.PYODIDE_URL), unpacks
 * the scan package the server built from its own modules
 * (browserscan.scanner_archive), loads the version-keyed ORF index, writes the dropped File into Pyodide's in-memory filesystem and calls
 * swissisoform_site.scanview.scan_file_json — the code the server route runs, so
 * the result is the same response body the server would have returned.
 *
 * Messages in:   {file, pyodide, scanner, index}  (URLs from the drop zone)
 * Messages out:  {type: "status", message}
 *                {type: "done", json}             the response body, or a refusal
 *                                                 ({error, message}), as JSON text
 *                {type: "failed", infra, stale, message}
 *                    stale = a download of ours is gone (404): the site was
 *                            redeployed under this page, so reload it;
 *                    infra = the scanner could not run here (offer the server);
 *                    neither = the scan itself failed. vcf_drop.js discards this
 *                            worker on any failure, since the interpreter may be
 *                            dead (e.g. out of memory).
 *
 * The VCF bytes exist only in this worker's memory and are unlinked after the
 * scan; nothing here sends anything anywhere — it only fetches the downloads.
 */
"use strict";

var booting = null; // Promise<pyodide>, reused across scans
var loadedIndex = ""; // URL of the index currently in Python as INDEX

function status(message) {
  self.postMessage({ type: "status", message: message });
}

/* ``stale``: our own URLs are content-keyed, so a 404 means this page predates a
   deploy. Pyodide's CDN is not ours; its failures stay infrastructure. */
function fetchBytes(url, ours) {
  return fetch(url).then(function (response) {
    if (!response.ok) {
      var err = new Error("HTTP " + response.status + " for " + url);
      err.stale = Boolean(ours) && response.status === 404;
      throw err;
    }
    return response.arrayBuffer();
  });
}

/* A Python exception's message is its whole traceback; the page shows a line. */
function lastLine(err) {
  var text = String((err && err.message) || err).trim();
  return text.split("\n").pop();
}

function boot(cfg) {
  if (!booting) {
    booting = (async function () {
      status("Loading the scanner — first time only, a few seconds…");
      var pyodideModule = await import(cfg.pyodide + "pyodide.mjs");
      var py = await pyodideModule.loadPyodide({ indexURL: cfg.pyodide });
      py.unpackArchive(await fetchBytes(cfg.scanner, true), "zip", { extractDir: "/scanner" });
      // The classifier warns per REF mismatch; on a wrong-assembly file that is one
      // console line per variant, so warnings stay off here.
      py.runPython(
        "import sys, logging\n" +
          "sys.path.insert(0, '/scanner')\n" +
          "logging.disable(logging.WARNING)\n" +
          "from swissisoform_site.scanview import load_portable_index, scan_file_json, size_refusal\n"
      );
      return py;
    })();
    // A failed boot must not poison the next attempt.
    booting.catch(function () {
      booting = null;
    });
  }
  return booting;
}

async function ensureIndex(py, url) {
  if (loadedIndex === url) return;
  status("Loading the catalogue…");
  py.FS.writeFile("/index.bin.gz", new Uint8Array(await fetchBytes(url, true)));
  py.runPython("INDEX = load_portable_index('/index.bin.gz')");
  py.FS.unlink("/index.bin.gz");
  loadedIndex = url;
}

/* Stream the File into the in-memory filesystem: no second full copy in JS. */
async function stage(py, file, path) {
  var stream = py.FS.open(path, "w");
  try {
    var reader = file.stream().getReader();
    var position = 0;
    for (;;) {
      var chunk = await reader.read();
      if (chunk.done) break;
      py.FS.write(stream, chunk.value, 0, chunk.value.length, position);
      position += chunk.value.length;
    }
  } finally {
    py.FS.close(stream);
  }
}

self.onmessage = async function (event) {
  var cfg = event.data;
  var py;
  try {
    py = await boot(cfg);
    await ensureIndex(py, cfg.index);
  } catch (err) {
    self.postMessage({
      type: "failed",
      infra: !err.stale,
      stale: Boolean(err.stale),
      message: lastLine(err),
    });
    return;
  }

  // Before the copy: a file past the size cap would otherwise fill the tab's
  // memory before the scan could refuse it.
  var refusal = py.runPython("size_refusal(" + Number(cfg.file.size) + ")");
  if (refusal) {
    self.postMessage({ type: "done", json: refusal });
    return;
  }

  var path = "/upload.vcf";
  try {
    status("Scanning " + cfg.file.name + " in your browser…");
    await stage(py, cfg.file, path);
    var json = py.runPython("scan_file_json('" + path + "', INDEX)");
    self.postMessage({ type: "done", json: json });
  } catch (err) {
    self.postMessage({ type: "failed", infra: false, stale: false, message: lastLine(err) });
  } finally {
    try {
      py.FS.unlink(path);
    } catch (e) {
      /* Never written: nothing to remove. */
    }
  }
};
