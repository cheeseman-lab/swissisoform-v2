/* Uploaded-VCF markers on the gene figure.
 *
 * The server builds the figure without them — the scan lives in the browser — so
 * applyUploadedHits() adds them to a copy of the figure dict before it is drawn.
 * Everything about where and how they are drawn comes from the server:
 *
 *   hit.x, hit.protein_change      set at scan time (scanview.enrich_hits), so the
 *                                  browser never needs the ORF index;
 *   layout.meta.variant_rows       row order, spacing, labels, colours, the marker
 *                                  style and the y-range / height formula
 *                                  (plots/protein.py, _variant_rows_meta);
 *   trace.meta.row, annotation     which existing traces and labels belong to which
 *     name "variant-row:<term>"    consequence row, so rows can be re-spaced when an
 *                                  upload brings a consequence the gene has none of.
 */
(function () {
  "use strict";

  var ROW_PREFIX = "variant-row:";

  function escapeHtml(text) {
    return String(text == null ? "" : text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  /* Scan hits → one marker per (variant, x, consequence).
     One variant inside N isoforms yields N hits, and shared-region hits all map to
     the same x — without this they stack into one diamond with one reachable hover.
     Distinct variants that happen to share an x stay separate. */
  function mergeHits(hits) {
    var merged = {};
    var order = [];
    hits.forEach(function (hit) {
      if (hit.x === null || hit.x === undefined) return; // no protein-axis position
      var change = hit.ref + ">" + hit.alt;
      var consequence = hit.consequence || "other";
      var key = [hit.chrom, hit.pos, change, hit.x, consequence].join("|");
      var mark = merged[key];
      if (!mark) {
        mark = merged[key] = {
          x: hit.x,
          consequence: consequence,
          inUnique: false,
          proteinChange: hit.protein_change || "",
          detail:
            hit.chrom + ":" + Number(hit.pos).toLocaleString("en-US") + " " + change +
            " · line " + hit.line_no,
          isoforms: {},
        };
        order.push(key);
      }
      // A hit in the unique region of ANY isoform makes the mark prominent.
      mark.inUnique = mark.inUnique || hit.region === "unique";
      mark.isoforms[hit.tis_id] = true;
    });
    return order.map(function (key) {
      var mark = merged[key];
      var n = Object.keys(mark.isoforms).length;
      if (n > 1) mark.detail += " · in " + n + " isoforms";
      return mark;
    });
  }

  function rank(order, term) {
    var i = order.indexOf(term);
    return i === -1 ? order.length : i;
  }

  function sortRows(order, terms) {
    return terms.slice().sort(function (a, b) {
      return rank(order, a) - rank(order, b) || (a < b ? -1 : a > b ? 1 : 0);
    });
  }

  /* A patched deep copy of fig with the uploaded markers, or fig itself when there
     is nothing to add or the figure predates the row contract. */
  function applyUploadedHits(fig, hits) {
    var rows = fig && fig.layout && fig.layout.meta && fig.layout.meta.variant_rows;
    if (!rows || !hits || !hits.length) return fig;
    var marks = mergeHits(hits);
    if (!marks.length) return fig;

    var out = JSON.parse(JSON.stringify(fig));
    var layout = out.layout;
    layout.annotations = layout.annotations || [];

    // Rows: the server's plus any consequence only the upload has, in its order.
    var terms = rows.present.slice();
    marks.forEach(function (m) {
      if (terms.indexOf(m.consequence) === -1) terms.push(m.consequence);
    });
    terms = sortRows(rows.order, terms);
    function yFor(term) {
      return rows.base + terms.indexOf(term) * rows.step;
    }

    // Re-space what the server drew, then label the new rows.
    out.data.forEach(function (trace) {
      var row = trace.meta && trace.meta.row;
      if (row) trace.y = trace.y.map(function () { return yFor(row); });
    });
    layout.annotations.forEach(function (ann) {
      if (ann.name && ann.name.indexOf(ROW_PREFIX) === 0) {
        ann.y = yFor(ann.name.slice(ROW_PREFIX.length));
      }
    });
    terms.forEach(function (term) {
      if (rows.present.indexOf(term) !== -1) return;
      layout.annotations.push({
        name: ROW_PREFIX + term,
        x: rows.label_x,
        y: yFor(term),
        xref: "x",
        yref: "y",
        xanchor: "right",
        xshift: -6,
        text: escapeHtml((rows.labels[term] || term).slice(0, rows.label_max)),
        showarrow: false,
        font: { size: rows.label_size, color: rows.label_color },
      });
    });

    // One diamond trace per row, one legend entry across all of them.
    var legendShown = false;
    terms.forEach(function (term) {
      var own = marks.filter(function (m) { return m.consequence === term; });
      if (!own.length) return;
      var marker = Object.assign({}, rows.uploaded_marker, {
        color: rows.colors[term] || rows.fallback_color,
      });
      out.data.push({
        type: "scatter",
        mode: "markers",
        x: own.map(function (m) { return m.x; }),
        y: own.map(function () { return yFor(term); }),
        marker: marker,
        hovertext: own.map(function (m) {
          return (
            "<b>From your VCF</b><br>" + escapeHtml(m.detail) + "<br>" +
            escapeHtml(m.proteinChange) + "<br>" + escapeHtml(term) + " · " +
            (m.inUnique ? "unique" : "shared") + " region"
          );
        }),
        hoverinfo: "text",
        name: "uploaded VCF",
        showlegend: !legendShown,
        legendgroup: "uploaded",
      });
      legendShown = true;
    });

    // Grow the y-range and height exactly as plots/protein.py would have.
    var top = rows.base + (terms.length - 1) * rows.step + rows.top_pad;
    var guideTop = Math.max(top, rows.canon_y + rows.min_above_canon);
    var yLo = layout.yaxis.range[0];
    var yHi = guideTop + rows.y_hi_pad;
    layout.yaxis.range = [yLo, yHi];
    layout.height = Math.floor(rows.height_base + rows.height_per_y * (yHi - yLo));
    (layout.shapes || []).forEach(function (shape) {
      if (shape.name === "truncation-guide") shape.y1 = guideTop;
    });
    return out;
  }

  window.applyUploadedHits = applyUploadedHits;
})();
