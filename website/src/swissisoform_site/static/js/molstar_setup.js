/* SwissIsoform v2 — Mol* helper for the side-by-side folding panel.
 *
 *   window.swissisoformViewer(divId, cifUrl)
 *     Creates one Mol* viewer in the element #divId and loads the CIF. Uses the
 *     molstar@4 *viewer* UMD bundle, which exposes only `molstar.Viewer` —
 *     loadStructureFromUrl applies Mol*'s default predicted-model (pLDDT)
 *     colouring for Boltz/AlphaFold CIFs. Residue-level selection/colouring and
 *     superposition are NOT available in this bundle (they need the full Mol*
 *     library), so the differential region is conveyed by the 2D protein track
 *     and the caption rather than painted into the 3D cartoon.
 */
(function () {
  if (typeof window === "undefined") return;

  window.swissisoformViewer = function (divId, cifUrl) {
    var el = document.getElementById(divId);
    if (!el) return;
    if (typeof molstar === "undefined") {
      el.innerHTML = '<div class="molstar-fallback">Mol* failed to load from CDN.</div>';
      return;
    }
    // molstar.Viewer.create wants the element *id string* on this UMD build —
    // passing a DOM element mounts a blank (black) canvas.
    return molstar.Viewer.create(divId, {
      layoutIsExpanded: false,
      layoutShowControls: false,
      layoutShowRemoteState: false,
      layoutShowSequence: false,
      layoutShowLog: false,
      layoutShowLeftPanel: false,
      viewportShowExpand: true,
      viewportShowSelectionMode: false,
      viewportShowAnimation: false,
      pdbProvider: "rcsb",
      emdbProvider: "rcsb",
    })
      .then(function (viewer) {
        return viewer.loadStructureFromUrl(cifUrl, "mmcif");
      })
      .catch(function (err) {
        console.error("[swissisoform] Mol* load failed for " + divId + ":", err);
        el.innerHTML =
          '<div class="molstar-fallback molstar-fallback-error">Failed to load structure: ' +
          (err && err.message ? err.message : err) +
          "</div>";
      });
  };
})();
