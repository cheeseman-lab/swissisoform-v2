"""``scanstate.js``: the tab's scan in sessionStorage.

Run under Node against an in-memory ``sessionStorage`` with a byte quota, and a
``document`` whose ``<body>`` carries the two values the server writes. Needs
``node`` (see ``jsrun``).
"""

from __future__ import annotations

from jsrun import JS_DIR, needs_node, run_script

pytestmark = needs_node

_HARNESS = r"""
const fs = require("fs"), vm = require("vm"), assert = require("assert");
const src = fs.readFileSync(process.argv[1], "utf8");

function makeStore(quota) {
  const store = new Map();
  store.api = {
    getItem: k => (store.has(k) ? store.get(k) : null),
    removeItem: k => store.delete(k),
    setItem: (k, v) => {
      let used = 0;
      for (const [a, b] of store) if (a !== k) used += b.length;
      if (used + v.length > quota) {
        const e = new Error("quota"); e.name = "QuotaExceededError"; throw e;
      }
      store.set(k, v);
    },
  };
  return store;
}

/* A fresh page load: new window, same sessionStorage. */
function page(store, attrs) {
  const document = {
    readyState: "complete",
    body: { getAttribute: n => attrs[n.replace("data-", "")] || null },
    querySelectorAll: () => [],
    addEventListener() {},
  };
  const ctx = { window: {}, document, sessionStorage: store.api };
  vm.createContext(ctx);
  vm.runInContext(src, ctx);
  return ctx.window.SwissScan;
}

const attrs = { "scan-schema": "c1", "index-version": "v1" };
// Shaped like real hits: shared strings, per-hit strings, numbers and nulls.
const hits = Array.from({ length: 3000 }, (_, i) => ({
  line_no: i, chrom: "chr17", pos: 48071000 + i, ref: "C", alt: "G",
  gene: i % 2 ? "A" : "B", tis_id: "t" + (i % 3), orf_type: "extended",
  frame: i % 5 ? "isoform" : "canonical", residue: i % 7 ? i : null,
  region: i % 2 ? "unique" : "shared", consequence: "missense_variant",
  aa_ref: "P", aa_alt: "L", consequence_note: "",
  x: i % 7 ? i + 0.5 : null, protein_change: "p.Pro" + i + "Leu · isoform frame",
}));
const summary = { schema: "c1", provenance: { index_version: "v1" }, counts: { hits: 3000 } };
const checks = {};

checks.round_trip = () => {
  const store = makeStore(5e6);
  assert.strictEqual(page(store, attrs).save(summary, hits), "full");
  const scan = page(store, attrs).load();
  assert.strictEqual(scan.hits.length, 3000);
  assert.strictEqual(scan.stale, false);
  assert.strictEqual(scan.summary.hits_stored, "full");
};
checks.lookups = () => {
  const store = makeStore(5e6);
  page(store, attrs).save(summary, hits);
  assert.strictEqual(page(store, attrs).hitsForGene("A").length, 1500);
  assert.strictEqual(page(store, attrs).hitsForTis("t0").length, 1000);
  assert.strictEqual(page(store, attrs).countVariants(hits.slice(0, 4)), 4);
};
checks.other_index_is_stale = () => {
  const store = makeStore(5e6);
  page(store, attrs).save(summary, hits);
  assert.strictEqual(page(store, { ...attrs, "index-version": "v2" }).load().stale, true);
};
checks.unknown_schema_is_discarded = () => {
  const store = makeStore(5e6);
  page(store, attrs).save(summary, hits);
  assert.strictEqual(page(store, { ...attrs, "scan-schema": "c2" }).load(), null);
  assert.strictEqual(store.size, 0);
};
checks.pack_round_trips_exactly = () => {
  const api = page(makeStore(5e6), attrs);
  const packed = api.pack(hits);
  // Compared as JSON: objects built inside the vm context have that realm's
  // prototypes, which deepStrictEqual would report as a difference.
  const same = (a, b) => assert.strictEqual(JSON.stringify(a), JSON.stringify(b));
  same(api.unpack(JSON.parse(JSON.stringify(packed))), hits);
  same(api.unpack(api.pack([])), []);
  // All-string columns become tables; numbers and null-able columns stay as is.
  same(packed.tables.gene, ["B", "A"]);
  assert.ok(!("residue" in packed.tables) && !("x" in packed.tables) && !("pos" in packed.tables));
};
checks.packing_shrinks_what_is_stored = () => {
  const api = page(makeStore(5e6), attrs);
  const plain = JSON.stringify(hits).length;
  const packed = JSON.stringify(api.pack(hits)).length;
  assert.ok(packed < plain * 0.6, `packed ${packed} vs plain ${plain}`);
};
checks.unpacked_storage_from_before_is_discarded = () => {
  const store = makeStore(5e6);
  page(store, attrs).save(summary, hits);
  store.set("swissiso.scan.hits", JSON.stringify(hits));  // the old array layout
  assert.strictEqual(page(store, attrs).load(), null);
  assert.strictEqual(store.size, 0);
};
checks.over_quota_keeps_the_first_thousand = () => {
  // Room for 1,000 packed hits (plus the summary) but not 3,000.
  const api = page(makeStore(5e6), attrs);
  const room = JSON.stringify(api.pack(hits.slice(0, 1000))).length + 2000;
  assert.ok(JSON.stringify(api.pack(hits)).length > room);
  const store = makeStore(room);
  assert.strictEqual(page(store, attrs).save(summary, hits), "partial");
  const scan = page(store, attrs).load();
  assert.strictEqual(scan.hits.length, 1000);
  assert.strictEqual(scan.summary.hits_stored, "partial");
};
checks.new_upload_replaces_the_old = () => {
  const store = makeStore(5e6);
  page(store, attrs).save(summary, hits);
  page(store, attrs).save(summary, hits.slice(0, 5));
  assert.strictEqual(page(store, attrs).load().hits.length, 5);
};
checks.corrupt_json_is_cleared = () => {
  const store = makeStore(5e6);
  page(store, attrs).save(summary, hits);
  store.set("swissiso.scan.hits", "{not json");
  assert.strictEqual(page(store, attrs).load(), null);
  assert.strictEqual(store.size, 0);
};
checks.clear_removes_both_keys = () => {
  const store = makeStore(5e6);
  page(store, attrs).save(summary, hits);
  const api = page(store, attrs);
  api.clear();
  assert.strictEqual(store.size, 0);
  assert.strictEqual(api.load(), null);
};
checks.no_scan_is_null = () => {
  assert.strictEqual(page(makeStore(5e6), attrs).load(), null);
};

checks[process.argv[2]]();
"""


def _check(name: str) -> None:
    run_script(_HARNESS, str(JS_DIR / "scanstate.js"), name)


def test_round_trip() -> None:
    _check("round_trip")


def test_lookups_filter_the_stored_hits() -> None:
    _check("lookups")


def test_a_scan_from_another_index_is_kept_but_stale() -> None:
    _check("other_index_is_stale")


def test_an_unknown_schema_is_discarded() -> None:
    _check("unknown_schema_is_discarded")


def test_packing_round_trips_exactly() -> None:
    _check("pack_round_trips_exactly")


def test_packing_shrinks_what_is_stored() -> None:
    _check("packing_shrinks_what_is_stored")


def test_storage_in_the_old_unpacked_layout_is_discarded() -> None:
    _check("unpacked_storage_from_before_is_discarded")


def test_over_quota_keeps_the_first_thousand_hits() -> None:
    _check("over_quota_keeps_the_first_thousand")


def test_a_new_upload_replaces_the_old_scan() -> None:
    _check("new_upload_replaces_the_old")


def test_corrupt_storage_is_cleared_not_rendered() -> None:
    _check("corrupt_json_is_cleared")


def test_clear_removes_both_keys() -> None:
    _check("clear_removes_both_keys")


def test_no_scan_reads_as_none() -> None:
    _check("no_scan_is_null")
