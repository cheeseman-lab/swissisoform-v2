"""``vcf_drop.js``: how the drop zone reacts to what the scan worker reports.

Run under Node against stub elements and a stub ``Worker`` that records what it
was sent and lets the test answer as the worker would. Needs ``node`` (see
``jsrun``).
"""

from __future__ import annotations

from jsrun import JS_DIR, needs_node, run_script

pytestmark = needs_node

_HARNESS = r"""
const fs = require("fs"), vm = require("vm"), assert = require("assert");
const src = fs.readFileSync(process.argv[1], "utf8");

function element(extra) {
  const listeners = {};
  return Object.assign({
    hidden: false,
    textContent: "",
    dataset: {},
    classList: {
      set: new Set(),
      add(c) { this.set.add(c); },
      remove(...cs) { cs.forEach((c) => this.set.delete(c)); },
    },
    listeners,
    addEventListener(name, fn) { (listeners[name] = listeners[name] || []).push(fn); },
    fire(name, event) { (listeners[name] || []).forEach((fn) => fn(event || {})); },
  }, extra);
}

/* A page with the drop zone, and a Worker stub that records every instance. */
function page() {
  const workers = [];
  class Worker {
    constructor(url, opts) {
      this.url = url; this.opts = opts; this.sent = []; this.terminated = false;
      workers.push(this);
    }
    postMessage(msg) { this.sent.push(msg); }
    terminate() { this.terminated = true; }
    reply(data) { this.onmessage({ data }); }
  }
  const zone = element({
    dataset: { index: "/i.json.gz", scanner: "/s.zip", pyodide: "/p/", worker: "/w.js" },
  });
  const input = element({ files: [] });
  const status = element();
  const server = element({ hidden: true });
  const ids = { "vq-drop": zone, "vq-file": input, "vq-status": status, "vq-server": server };
  const ctx = {
    window: { SwissScan: null, location: {} },
    document: {
      getElementById: (id) => ids[id] || null,
      body: { getAttribute: () => "c2" },
    },
    Worker, WebAssembly: {}, console: { log() {}, error() {} },
  };
  vm.createContext(ctx);
  vm.runInContext(src, ctx);
  const drop = (name) => {
    input.files = [{ name, size: 10 }];
    input.fire("change");
  };
  return { workers, zone, status, server, drop };
}

const checks = {};

checks.a_failed_scan_discards_its_worker = () => {
  const p = page();
  p.drop("a.vcf");
  assert.strictEqual(p.workers.length, 1);
  p.workers[0].reply({ type: "failed", infra: false, stale: false, message: "MemoryError" });
  assert.ok(p.workers[0].terminated);
  assert.ok(p.status.textContent.includes("MemoryError"));
  // The next drop must not reuse the dead interpreter.
  p.drop("b.vcf");
  assert.strictEqual(p.workers.length, 2);
  assert.strictEqual(p.workers[1].sent[0].file.name, "b.vcf");
};

checks.a_refusal_keeps_the_warm_worker = () => {
  const p = page();
  p.drop("a.vcf");
  p.workers[0].reply({ type: "done", json: '{"error":"expands_too_large","message":"too big"}' });
  assert.ok(!p.workers[0].terminated);
  assert.strictEqual(p.status.textContent, "too big");
  p.drop("b.vcf");
  assert.strictEqual(p.workers.length, 1);
};

checks.a_stale_page_is_told_to_reload_not_offered_the_server = () => {
  const p = page();
  p.drop("a.vcf");
  p.workers[0].reply({ type: "failed", infra: false, stale: true, message: "HTTP 404" });
  assert.ok(p.status.textContent.includes("Reload"));
  assert.strictEqual(p.server.hidden, true);
  assert.ok(p.workers[0].terminated);
};

checks.a_scanner_that_cannot_run_offers_the_server = () => {
  const p = page();
  p.drop("a.vcf");
  p.workers[0].reply({ type: "failed", infra: true, stale: false, message: "no wasm" });
  assert.strictEqual(p.server.hidden, false);
};

checks[process.argv[2]]();
"""


def _check(name: str) -> None:
    run_script(_HARNESS, str(JS_DIR / "vcf_drop.js"), name)


def test_a_failed_scan_discards_its_worker_so_the_next_drop_boots_fresh() -> None:
    _check("a_failed_scan_discards_its_worker")


def test_a_refused_file_keeps_the_warm_worker() -> None:
    _check("a_refusal_keeps_the_warm_worker")


def test_a_page_from_before_a_deploy_is_told_to_reload() -> None:
    _check("a_stale_page_is_told_to_reload_not_offered_the_server")


def test_a_scanner_that_cannot_run_still_offers_the_server() -> None:
    _check("a_scanner_that_cannot_run_offers_the_server")
