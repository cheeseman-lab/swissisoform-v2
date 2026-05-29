// website/src/swissisoform_site/static/js/isoform_page.js
//
// V2 isoform page interaction:
//   - Evidence tiles: clicking a tile head toggles its body's hidden attribute
//     and flips aria-expanded. Multiple tiles can be open at once.
//   - Overlay toggles: variant/domain/motif/phyloP checkboxes restyle both
//     plots via Plotly.restyle. No-op if the toggle DOM isn't present.

(function () {
  // Tile expand/collapse on click
  document.querySelectorAll(".evidence-tile .tile-head").forEach((head) => {
    head.addEventListener("click", () => {
      const tile = head.closest(".evidence-tile");
      const body = tile.querySelector(".tile-body");
      const expanded = head.getAttribute("aria-expanded") === "true";
      head.setAttribute("aria-expanded", expanded ? "false" : "true");
      if (expanded) body.setAttribute("hidden", "");
      else body.removeAttribute("hidden");
    });
  });

  // Overlay toggles (no-op if .overlay-toggles is not in the DOM)
  function restyleByPrefix(divId, prefix, visible) {
    const gd = document.getElementById(divId);
    if (!gd || !gd.data) return;
    const indices = gd.data
      .map((tr, i) => (tr.name || "").startsWith(prefix) ? i : -1)
      .filter((i) => i >= 0);
    if (indices.length === 0) return;
    Plotly.restyle(gd, { visible: visible ? true : "legendonly" }, indices);
  }

  const PREFIXES = {
    variants: "variant ",
    domains: "domain:",
    motifs: "motif:",
    phylop: "phyloP",
  };

  document.querySelectorAll(".overlay-toggles input[type=checkbox]").forEach((cb) => {
    cb.addEventListener("change", () => {
      const key = cb.getAttribute("data-overlay");
      const prefix = PREFIXES[key];
      if (!prefix) return;
      restyleByPrefix("graph-transcript", prefix, cb.checked);
      restyleByPrefix("graph-protein", prefix, cb.checked);
    });
  });
})();
