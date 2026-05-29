// website/src/swissisoform_site/static/js/isoform_page.js
//
// V2 isoform page interaction:
//   - Evidence tiles: clicking a tile head opens a single <dialog> modal
//     containing the tile's body content (cloned from its <template>).
//     The 12-tile grid stays unchanged underneath — no DOM reflow.
//     Modal closes on Esc (native), the X button, or backdrop click.
//   - Overlay toggles: variant/domain/motif/phyloP checkboxes restyle both
//     plots via Plotly.restyle. No-op if the toggle DOM isn't present.

(function () {
  // Evidence modal lightbox
  const modal = document.getElementById("evidence-modal");
  const modalTitle = document.getElementById("evidence-modal-title");
  const modalBody = document.getElementById("evidence-modal-body");
  const modalClose = document.querySelector(".evidence-modal-close");

  function openModal(tile) {
    if (!modal || !modalTitle || !modalBody) return;
    const label = tile.querySelector(".tile-label")?.textContent.trim() || "Evidence";
    const id = tile.querySelector(".tile-id")?.textContent.trim() || "";
    modalTitle.textContent = id ? `${id} · ${label}` : label;
    modalBody.innerHTML = "";
    const tpl = tile.querySelector("template.tile-body-template");
    if (tpl) {
      modalBody.appendChild(tpl.content.cloneNode(true));
    }
    if (typeof modal.showModal === "function") {
      modal.showModal();
    } else {
      modal.setAttribute("open", "");
    }
  }

  function closeModal() {
    if (!modal) return;
    if (typeof modal.close === "function") modal.close();
    else modal.removeAttribute("open");
    if (modalBody) modalBody.innerHTML = "";
  }

  document.querySelectorAll(".evidence-tile .tile-head").forEach((head) => {
    head.addEventListener("click", () => {
      const tile = head.closest(".evidence-tile");
      if (tile) openModal(tile);
    });
  });

  modalClose?.addEventListener("click", closeModal);
  // Click outside (on backdrop): native <dialog> dispatches a click event
  // whose target is the dialog itself when the user clicks the backdrop.
  modal?.addEventListener("click", (e) => {
    if (e.target === modal) closeModal();
  });
  // Esc is handled natively by <dialog>; clear body when it closes.
  modal?.addEventListener("close", () => {
    if (modalBody) modalBody.innerHTML = "";
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
