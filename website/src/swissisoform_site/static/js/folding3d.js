/* SwissIsoform v2 — interactive folding viewer (3Dmol.js).
 *
 *   window.swissFolding(divId, cifUrl, colorsUrl)
 *     Mounts a rotatable 3Dmol viewer in #divId, loads the CIF, and colours each
 *     residue from the precomputed map at colorsUrl ({resi: "#rrggbb"}). The
 *     colours are decided offline by scripts/export/build_folding_colors.py (pLDDT ramp
 *     on the whole protein, a diverging ramp on the differential region) — the
 *     viewer is a dumb display, it does not recolour. Falls back to a plain
 *     pLDDT b-factor gradient if the colour map can't be fetched.
 */
(function () {
  if (typeof window === "undefined") return;

  // Standard pLDDT gradient (b-factor) used only when the colour map is absent.
  function plddtFallback(atom) {
    var t = Math.max(0, Math.min(1, (atom.b - 50) / 40));
    var r, g, b;
    if (t < 0.5) { r = 230 + (245 - 230) * t * 2; g = 126 + (245 - 126) * t * 2; b = 34 + (245 - 34) * t * 2; }
    else { var u = (t - 0.5) * 2; r = 245 + (30 - 245) * u; g = 245 + (80 - 245) * u; b = 245 + (200 - 245) * u; }
    return "#" + [r, g, b].map(function (x) { return ("0" + Math.round(x).toString(16)).slice(-2); }).join("");
  }

  window.swissFolding = function (divId, cifUrl, colorsUrl) {
    var el = document.getElementById(divId);
    if (!el) return;
    if (typeof $3Dmol === "undefined") {
      el.innerHTML = '<div class="molstar-fallback">3Dmol failed to load from CDN.</div>';
      return;
    }

    function draw(colors) {
      var viewer = $3Dmol.createViewer(el, { backgroundColor: "white" });
      fetch(cifUrl)
        .then(function (r) { return r.text(); })
        .then(function (cif) {
          viewer.addModel(cif, "cif");
          var colorfunc = colors
            ? function (a) { return colors[a.resi] || "#bdbdbd"; }
            : plddtFallback;
          viewer.setStyle({}, { cartoon: { colorfunc: colorfunc } });
          viewer.zoomTo();
          viewer.render();
          viewer.zoom(1.2, 600);
        })
        .catch(function (err) {
          console.error("[swissisoform] folding load failed for " + divId + ":", err);
          el.innerHTML =
            '<div class="molstar-fallback molstar-fallback-error">Failed to load structure.</div>';
        });
    }

    if (!colorsUrl) { draw(null); return; }
    fetch(colorsUrl)
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (colors) { draw(colors); })
      .catch(function () { draw(null); });
  };
})();
