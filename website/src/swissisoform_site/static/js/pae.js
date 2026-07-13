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
 *     canvas is both lighter and crisper. Residue-axis ticks + numbers are drawn
 *     in a small margin around the plot. Precompute is offline
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

  // Residue-axis tick spacing: a "nice" 1-2-2.5-5 step giving ~7-9 ticks across
  // L (e.g. 25 for ~185-240 aa, 50 for ~250-450 aa, 100 for ~900 aa). The 2.5
  // stop keeps paired canonical/isoform panels of similar length on the same
  // increment instead of jumping between 20 and 50.
  function niceStep(L) {
    var target = L / 7;
    var mag = Math.pow(10, Math.floor(Math.log10(target)));
    var norm = target / mag;
    var nice = norm < 1.5 ? 1 : norm < 2.25 ? 2 : norm < 3.5 ? 2.5 : norm < 7.5 ? 5 : 10;
    return Math.max(1, nice * mag);
  }

  // Plot margins (canvas px): the left/bottom margins hold the tick numbers AND
  // an axis title; top/right just breathe. S = square plot side. The top margin
  // is derived so the whole canvas is square (width === height), not just the plot.
  var ML = 50, MR = 12, MB = 42, S = 300;
  var N = ML + S + MR;   // 362 — square canvas side
  var MT = N - S - MB;   // top margin chosen so the canvas is square
  var TICK = "#64748b";  // slate tick numbers — legible on the white PAE card
  var TITLE = "#475569"; // slightly darker for the axis titles

  function draw(el, data, opts) {
    var L = data.L;
    var pae = data.pae;
    if (!L || !pae || pae.length !== L * L) {
      el.innerHTML = '<div class="pae-fallback">PAE data malformed.</div>';
      return;
    }

    el.innerHTML = "";
    el.classList.add("pae-panel");

    // Paint the L*L map at native resolution, then scale it into the plot area.
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
    // Back the canvas at devicePixelRatio so it stays crisp on HiDPI/retina
    // screens — a 1x backing store is the usual "low-res/blurry" culprit. The
    // CSS sizes the display (max-width 344px); all drawing below stays in CSS
    // units and ctx.scale(dpr) maps them onto the denser backing store.
    var dpr = Math.min(window.devicePixelRatio || 1, 3);
    canvas.width = Math.round(N * dpr);
    canvas.height = Math.round(N * dpr);
    var ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(off, 0, 0, L, L, ML, MT, S, S);

    // Map a residue index (0..L) to a pixel offset along the plot axis.
    function px(r) { return (r / L) * S; }

    // Differential-region outline (1-based inclusive bounds → pixel span).
    if (opts && opts.diffLo && opts.diffHi && opts.diffHi >= opts.diffLo) {
      var s0 = px(opts.diffLo - 1), e0 = px(opts.diffHi);
      ctx.strokeStyle = "rgba(140, 40, 160, 0.9)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 3]);
      ctx.strokeRect(ML + s0, MT + s0, e0 - s0, e0 - s0);
      ctx.setLineDash([]);
    }

    // Axis ticks + residue numbers.
    var step = niceStep(L);
    ctx.strokeStyle = TICK;
    ctx.fillStyle = TICK;
    ctx.lineWidth = 1;
    ctx.font = '10px ui-monospace, "SF Mono", Menlo, monospace';
    // x-axis (bottom): scored residue. Round to whole px + the .5 line offset so
    // ticks and numbers land on the pixel grid (crisp, not fuzzy).
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (var rx = step; rx <= L - step * 0.35; rx += step) {
      var xr = Math.round(ML + px(rx));
      ctx.beginPath();
      ctx.moveTo(xr + 0.5, MT + S);
      ctx.lineTo(xr + 0.5, MT + S + 4);
      ctx.stroke();
      ctx.fillText(String(rx), xr, MT + S + 6);
    }
    // y-axis (left): aligned residue
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (var ry = step; ry <= L - step * 0.35; ry += step) {
      var yr = Math.round(MT + px(ry));
      ctx.beginPath();
      ctx.moveTo(ML - 4, yr + 0.5);
      ctx.lineTo(ML, yr + 0.5);
      ctx.stroke();
      ctx.fillText(String(ry), ML - 6, yr);
    }

    // Axis titles.
    ctx.fillStyle = TITLE;
    ctx.font = '600 10px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';
    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";
    ctx.fillText("scored residue", ML + S / 2, MT + S + MB - 6);
    ctx.save();
    ctx.translate(12, MT + S / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("aligned residue", 0, 0);
    ctx.restore();

    var fig = document.createElement("figure");
    fig.className = "pae-figure";
    fig.appendChild(canvas);

    var cap = document.createElement("figcaption");
    cap.className = "pae-caption";
    cap.innerHTML =
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
