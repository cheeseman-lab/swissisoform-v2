/* SwissIsoform v2 — Mol* helpers for the single superimposed folding panel.
 *
 * Exported window functions:
 *
 *   window.swissisoformSuperposed(opts)
 *     Loads canonical + isoform CIFs into ONE Mol* viewer, superposes the
 *     isoform onto the canonical on their shared core (best-effort), colors
 *     both grey, and overlays the differential residue range in red on
 *     whichever structure carries it.
 *
 *   window.swissisoformHighlight(viewer, residueStart, residueEnd, isHighlightable)
 *     Best-effort 3D highlight of residues [start, end] on the loaded structure.
 *     Wraps the Mol* 4.x plugin-builder API: builds a label_seq_id range
 *     selection, adds a new representation (cartoon, uniform red) for it, and
 *     focuses the camera on the range. Silently no-ops on older / newer
 *     Mol* versions.
 */
(function () {
  if (typeof window === "undefined") return;

  // ──────────────────────────────────────────────── label_seq_id range select
  // Shared MolScript expression for residues [lo, hi] (1-based, inclusive).
  // label_seq_id is the canonical Mol* residue numbering for mmCIF —
  // Boltz / AlphaFold CIFs always start at 1, so [lo, hi] maps directly to the
  // 1-based positions we get from diff_start/end.
  function residueRangeExpression(lo, hi) {
    var MS = molstar.MolScriptBuilder;
    if (!MS) return null;
    return MS.struct.generator.atomGroups({
      "residue-test": MS.core.rel.inRange([
        MS.struct.atomProperty.macromolecular.label_seq_id(),
        lo,
        hi,
      ]),
    });
  }

  // ──────────────────────────────────────────── single superimposed viewer
  window.swissisoformSuperposed = async function (opts) {
    var el = opts.el;
    var canonicalUrl = opts.canonicalUrl;
    var isoformUrl = opts.isoformUrl;
    var diffLo = opts.diffLo;
    var diffHi = opts.diffHi;
    var diffOnCanonical = opts.diffOnCanonical;

    if (!el) return;
    if (typeof molstar === "undefined") {
      el.innerHTML = '<div class="molstar-fallback">Mol* failed to load from CDN.</div>';
      return;
    }

    var GREY = 0x9ca3af;
    var RED = 0xd62728;

    var viewer;
    try {
      viewer = await molstar.Viewer.create(el, {
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
      });
    } catch (e) {
      console.error("[swissisoform] Mol* viewer create failed:", e);
      el.innerHTML =
        '<div class="molstar-fallback molstar-fallback-error">Failed to init viewer: ' +
        (e && e.message ? e.message : e) +
        "</div>";
      return;
    }

    var plugin = viewer.plugin;
    var sb = plugin.builders && plugin.builders.structure;

    // Loads one CIF, draws a grey cartoon, and (when diffRange is given) layers
    // a red cartoon over the differential residues. Returns the structure ref.
    async function loadColored(url, color, diffRange) {
      if (!url) return null;
      try {
        var data = await plugin.builders.data.download(
          { url: url },
          { state: { isGhost: false } }
        );
        var traj = await sb.parseTrajectory(data, "mmcif");
        var model = await sb.createModel(traj);
        var struct = await sb.createStructure(model);

        var comp = await sb.tryCreateComponentStatic(struct, "polymer");
        if (comp) {
          await sb.representation.addRepresentation(comp, {
            type: "cartoon",
            color: "uniform",
            colorParams: { value: color },
          });
        }

        if (diffRange && diffRange[1] >= diffRange[0] && diffRange[1] > 0) {
          var sel = residueRangeExpression(diffRange[0], diffRange[1]);
          if (sel) {
            var diffComp = await sb.tryCreateComponentFromExpression(
              struct,
              sel,
              "diff-region",
              { label: "Differential region" }
            );
            if (diffComp) {
              await sb.representation.addRepresentation(diffComp, {
                type: "cartoon",
                color: "uniform",
                colorParams: { value: RED },
              });
            }
          }
        }
        return struct;
      } catch (e) {
        console.error("[swissisoform] structure load failed for " + url + ":", e);
        return null;
      }
    }

    var diffRange = [diffLo, diffHi];
    await loadColored(canonicalUrl, GREY, diffOnCanonical ? diffRange : null);
    await loadColored(isoformUrl, GREY, diffOnCanonical ? null : diffRange);

    // Superpose isoform onto canonical on their shared core (best-effort). The
    // two structures are the SAME protein differing only at the N-terminus, so
    // superposition is a refinement, not a correctness requirement — if the API
    // shape differs across Mol* builds, the catch leaves both loaded + colored.
    try {
      var structs = plugin.managers.structure.hierarchy.current.structures;
      if (structs && structs.length >= 2 &&
          plugin.managers.structure.superposition &&
          plugin.managers.structure.superposition.superposeStructures) {
        await plugin.managers.structure.superposition.superposeStructures([
          structs[0].cell,
          structs[1].cell,
        ]);
      }
    } catch (e) {
      console.warn("[swissisoform] superposition skipped:", e);
    }

    try {
      plugin.managers.camera.reset();
    } catch (e) {
      // camera reset is non-critical
    }
  };

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

      var expression = residueRangeExpression(residueStart, residueEnd);

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
})();
