// website/src/swissisoform_site/static/js/isoform_page.js
//
// V2 isoform page interaction:
//   - Tab strip: clicking a tab unhides its body and hides the others
//   - Overlay toggles: variant/domain/motif/phyloP checkboxes restyle both
//     plots via Plotly.restyle. Trace visibility is matched by trace name
//     prefix ("variant ", "domain:", "motif:", "phyloP").

(function () {
  const tabs = document.querySelectorAll(".tab-strip .tab");
  const bodies = document.querySelectorAll(".tab-body");
  tabs.forEach((t) => {
    t.addEventListener("click", () => {
      tabs.forEach((x) => x.setAttribute("aria-selected", "false"));
      bodies.forEach((b) => b.setAttribute("hidden", ""));
      t.setAttribute("aria-selected", "true");
      const key = t.getAttribute("data-tab");
      const body = document.querySelector(`[data-tab-body="${key}"]`);
      if (body) body.removeAttribute("hidden");
    });
  });

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
