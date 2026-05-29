/* SwissIsoform v2 — Mol* helpers for the dual folding panel.
 *
 * Two exported window functions:
 *
 *   window.swissisoformHighlight(viewer, residueStart, residueEnd, isHighlightable)
 *     Best-effort 3D highlight of residues [start, end] on the loaded structure.
 *     Wraps the Mol* 4.x plugin-builder API: builds a label_seq_id range
 *     selection, adds a new representation (cartoon, uniform red) for it, and
 *     focuses the camera on the range. Silently no-ops on older / newer
 *     Mol* versions — the HTML residue strip below the viewer is the
 *     guaranteed-visible fallback.
 *
 *   window.swissisoformResidueStrip(parentEl, start, end, total, isActive)
 *     Appends a thin horizontal residue bar to ``parentEl``: ``total`` little
 *     spans (gray by default, red within [start, end] if isActive). Always
 *     visible, version-independent, also acts as a legend for the 3D viewer.
 *
 * Both functions are idempotent — calling twice with the same parent is a
 * no-op the second time.
 */
(function () {
  if (typeof window === "undefined") return;

  // ────────────────────────────────────────────────────────────── 3D highlight
  window.swissisoformHighlight = function (viewer, residueStart, residueEnd, isHighlightable) {
    if (!isHighlightable) return;
    if (!viewer || !viewer.plugin) return;
    if (typeof molstar === "undefined") return;
    if (!residueEnd || residueEnd < residueStart) return;

    var plugin = viewer.plugin;
    var MS = molstar.MolScriptBuilder;
    if (!MS) {
      console.warn("[swissisoform] Mol* MolScriptBuilder unavailable; skipping 3D highlight");
      return;
    }

    try {
      var structures = plugin.managers.structure.hierarchy.current.structures;
      if (!structures || !structures.length) {
        console.warn("[swissisoform] no loaded structure; skipping 3D highlight");
        return;
      }
      var struct = structures[0];

      // label_seq_id is the canonical Mol* residue numbering for mmCIF —
      // Boltz / AlphaFold CIFs always start at 1, so [start, end] inclusive
      // maps directly to the 1-based positions we get from diff_start/end.
      var expression = MS.struct.generator.atomGroups({
        "residue-test": MS.core.rel.inRange([
          MS.struct.atomProperty.macromolecular.label_seq_id(),
          residueStart,
          residueEnd,
        ]),
      });

      var builders = plugin.builders && plugin.builders.structure;
      if (!builders || !builders.representation) {
        console.warn("[swissisoform] plugin.builders.structure.representation missing");
        return;
      }

      // Attach a new selection component to the loaded structure cell.
      // Then layer a uniform-red cartoon representation over it. The original
      // representation is left intact, so the diff region shows red over the
      // default chain color.
      var componentBuilder = builders.component || builders;
      var componentPromise = null;
      if (componentBuilder.addComponent) {
        componentPromise = componentBuilder.addComponent(struct.cell, {
          type: "selection",
          label: "Differential region",
          selection: expression,
        });
      } else if (builders.tryAddComponentFromExpression) {
        componentPromise = builders.tryAddComponentFromExpression(
          struct.cell,
          expression,
          "diff-region",
          { label: "Differential region" }
        );
      }

      if (!componentPromise) {
        console.warn("[swissisoform] no available component-builder method on Mol* 4.x");
        return;
      }

      componentPromise
        .then(function (component) {
          if (!component) return null;
          return builders.representation.addRepresentation(component, {
            type: "cartoon",
            color: "uniform",
            colorParams: { value: 0xd62728 },
          });
        })
        .then(function () {
          if (plugin.managers && plugin.managers.camera && plugin.managers.camera.focusLoci) {
            try {
              plugin.managers.camera.reset();
            } catch (e) {
              // camera reset is non-critical
            }
          }
        })
        .catch(function (e) {
          console.warn("[swissisoform] 3D highlight build failed:", e);
        });
    } catch (e) {
      console.warn("[swissisoform] swissisoformHighlight threw (non-fatal):", e);
    }
  };

  // ─────────────────────────────────────────────────────── HTML residue strip
  window.swissisoformResidueStrip = function (parentEl, start, end, total, isActive) {
    if (!parentEl || !parentEl.parentNode) return;
    if (!total || total <= 0) return;

    // Inject after the viewer div so the strip sits below it. Re-running is a
    // no-op.
    var host = parentEl.parentNode;
    if (host.querySelector(".residue-strip")) return;

    var strip = document.createElement("div");
    strip.className = "residue-strip" + (isActive ? " residue-strip-active" : "");
    strip.setAttribute(
      "title",
      isActive
        ? "Residues " + start + "–" + end + " (red) are the differential region"
        : "No differential region on this structure"
    );

    // Cap the per-residue bar count for very long proteins — beyond ~600
    // residues the bar slivers become single pixels anyway.
    var step = total > 600 ? Math.ceil(total / 600) : 1;
    for (var i = 1; i <= total; i += step) {
      var bar = document.createElement("span");
      bar.className = "residue-bar";
      if (isActive && i >= start && i <= end) bar.className += " diff";
      strip.appendChild(bar);
    }
    host.appendChild(strip);
  };
})();
