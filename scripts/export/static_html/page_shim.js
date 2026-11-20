// Runs first in every page of the single-file export (export_static_html.py).
// E     : {url path: body} for everything the page fetches at runtime.
// PAGES : every page path in the export, for in-site navigation.
(function () {
  var E = __EMBEDS__;
  var PAGES = __PAGES__;
  var nativeFetch = window.fetch ? window.fetch.bind(window) : null;

  window.fetch = function (u, o) {
    var s = typeof u === "string" ? u : (u && u.url) || "";
    var p;
    try { p = new URL(s, "http://x/").pathname; } catch (e) { p = s; }
    if (Object.prototype.hasOwnProperty.call(E, p)) {
      var json = p.endsWith(".json") || p.indexOf("/structure-") === 0;
      var headers = { "Content-Type": json ? "application/json" : "text/plain" };
      return Promise.resolve(new Response(E[p], { status: 200, headers: headers }));
    }
    // A same-site URL the export did not capture (VCF scan, downloads, JSON dump).
    if (s.charAt(0) === "/") {
      return Promise.resolve(new Response("not in this export", { status: 404 }));
    }
    return nativeFetch ? nativeFetch(u, o) : Promise.reject(new Error("no fetch"));
  };

  function go(u) {
    var p = decodeURIComponent(String(u).split("#")[0].split("?")[0]);
    if (PAGES.indexOf(p) >= 0) {
      window.parent.postMessage({ go: p, hash: String(u).split("#")[1] || "" }, "*");
    }
  }
  // The exporter rewrites `window.location = X` to `window.__swissisoGo = X`.
  Object.defineProperty(window, "__swissisoGo", { set: go });

  document.addEventListener("click", function (ev) {
    var a = ev.target.closest && ev.target.closest("a[href]");
    if (!a) return;
    var h = a.getAttribute("href");
    if (!h || h.charAt(0) !== "/") return;
    ev.preventDefault();
    go(h);
  }, true);
})();
