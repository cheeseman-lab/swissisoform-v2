/* SwissIsoform v2 — PAE (predicted aligned error) heatmap, canvas renderer.
 *
 *   window.swissPAE(divId, paeUrl, opts)
 *     Fetches the precomputed PAE JSON ({L, pae:[row-major floats]}) from
 *     paeUrl and paints the L×L map into a <canvas> in #divId. PAE(i,j) is the
 *     expected position error at residue j when the structure is aligned on
 *     residue i — low (dark green) = confident relative positioning, high
 *     (white) = uncertain. Off-diagonal blocks going dark means two regions are
 *     confidently placed relative to each other (a rigid domain); a bright block
 *     means their relative position is unresolved.
 *
 *     We render to canvas (not Plotly) because the site ships plotly-basic,
 *     which has no heatmap trace — and a PAE map is fundamentally an image, so
 *     canvas is both lighter and crisper. Precompute is offline
 *     (swissisoform.export.pae); this is a dumb display.
 *
 *   opts: { diffLo, diffHi }  1-based inclusive differential-region bounds
 *     (optional) — drawn as a faint outline so the reader can see how the
 *     gained/lost region relates structurally to the rest.
 */
(function () {
  if (typeof window === "undefined") return;

  // Fixed cap so colours are comparable across proteins (AlphaFold uses ~31.75 Å).
  var PAE_MAX = 31.75;
  // 3-stop green ramp: 0 Å dark green → mid → ~white at PAE_MAX ("Greens" reversed).
  var STOPS = [
    [0.0, [0, 68, 27]],
    [0.5, [90, 174, 97]],
    [1.0, [247, 252, 245]],
  ];

  function ramp(t) {
    t = Math.max(0, Math.min(1, t));
    for (var i = 1; i < STOPS.length; i++) {
      if (t <= STOPS[i][0]) {
        var a = STOPS[i - 1];
        var b = STOPS[i];
        var u = (t - a[0]) / (b[0] - a[0] || 1);
        return [
          Math.round(a[1][0] + (b[1][0] - a[1][0]) * u),
          Math.round(a[1][1] + (b[1][1] - a[1][1]) * u),
          Math.round(a[1][2] + (b[1][2] - a[1][2]) * u),
        ];
      }
    }
    return STOPS[STOPS.length - 1][1];
  }

  function draw(el, data, opts) {
    var L = data.L;
    var pae = data.pae;
    if (!L || !pae || pae.length !== L * L) {
      el.innerHTML = '<div class="pae-fallback">PAE data malformed.</div>';
      return;
    }

    el.innerHTML = "";
    el.classList.add("pae-panel");

    // Paint the L×L map at native resolution, then let CSS scale it to fit.
    var off = document.createElement("canvas");
    off.width = L;
    off.height = L;
    var octx = off.getContext("2d");
    var img = octx.createImageData(L, L);
    for (var k = 0; k < L * L; k++) {
      var c = ramp(pae[k] / PAE_MAX);
      var p = k * 4;
      img.data[p] = c[0];
      img.data[p + 1] = c[1];
      img.data[p + 2] = c[2];
      img.data[p + 3] = 255;
    }
    octx.putImageData(img, 0, 0);

    var canvas = document.createElement("canvas");
    canvas.className = "pae-canvas";
    // Display size is CSS-driven (square, responsive); back it with a crisp
    // raster at a sensible device pixel budget.
    var px = 320;
    canvas.width = px;
    canvas.height = px;
    var ctx = canvas.getContext("2d");
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(off, 0, 0, L, L, 0, 0, px, px);

    // Differential-region outline (1-based inclusive bounds → pixel span).
    if (opts && opts.diffLo && opts.diffHi && opts.diffHi >= opts.diffLo) {
      var s = ((opts.diffLo - 1) / L) * px;
      var e = (opts.diffHi / L) * px;
      ctx.strokeStyle = "rgba(140, 40, 160, 0.9)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 3]);
      // Band on both axes = the diagonal block for the gained/lost region.
      ctx.strokeRect(s, s, e - s, e - s);
      ctx.setLineDash([]);
    }

    var fig = document.createElement("figure");
    fig.className = "pae-figure";
    fig.appendChild(canvas);

    var cap = document.createElement("figcaption");
    cap.className = "pae-caption";
    cap.innerHTML =
      '<span class="pae-axislabel">Aligned residue (rows) · scored residue (cols)</span>' +
      '<span class="pae-legend"><span class="pae-swatch"></span>0 Å (confident) → ' +
      Math.round(PAE_MAX) +
      " Å (uncertain)</span>";
    fig.appendChild(cap);
    el.appendChild(fig);
  }

  window.swissPAE = function (divId, paeUrl, opts) {
    var el = document.getElementById(divId);
    if (!el) return;
    if (!paeUrl) {
      el.innerHTML = '<div class="pae-empty">PAE not available.</div>';
      return;
    }
    fetch(paeUrl)
      .then(function (r) {
        return r.ok ? r.json() : null;
      })
      .then(function (data) {
        if (!data) {
          el.innerHTML = '<div class="pae-empty">PAE not available.</div>';
          return;
        }
        draw(el, data, opts || {});
      })
      .catch(function (err) {
        console.error("[swissisoform] PAE load failed for " + divId + ":", err);
        el.innerHTML = '<div class="pae-fallback">Failed to load PAE.</div>';
      });
  };
})();
