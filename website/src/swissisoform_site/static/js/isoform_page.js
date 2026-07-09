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
    // The SAE card carries wide multi-column tables — give it a roomier modal.
    modal.classList.toggle("evidence-modal-wide", tile.classList.contains("sae-tile"));
    modalBody.innerHTML = "";
    const tpl = tile.querySelector("template.tile-body-template");
    if (tpl) {
      modalBody.appendChild(tpl.content.cloneNode(true));
      // Template content is inert until cloned here — wire sorting + expanders on the clone.
      makeVariantTablesSortable(modalBody);
      wireSaeExpanders(modalBody);
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

  // Sortable variant tables. Each column header toggles asc/desc; missing/empty
  // sort keys always sink to the bottom. Keys come from each cell's data-sort
  // (raw number or lowercased text), not the rendered string.
  function makeVariantTablesSortable(root) {
    (root || document).querySelectorAll("table.js-sortable").forEach((table) => {
      const tbody = table.querySelector("tbody");
      const heads = Array.prototype.slice.call(table.querySelectorAll("thead th.sortable"));
      if (!tbody || !heads.length) return;

      function sortBy(th, colIndex) {
        const isNum = th.getAttribute("data-sort-type") === "num";
        // asc → desc → asc; a fresh column starts ascending.
        const dir = th.getAttribute("aria-sort") === "ascending" ? "descending" : "ascending";
        heads.forEach((h) => h.removeAttribute("aria-sort"));
        th.setAttribute("aria-sort", dir);
        const sign = dir === "ascending" ? 1 : -1;

        const cellKey = (tr) => {
          const cell = tr.children[colIndex];
          if (!cell) return "";
          const raw = cell.dataset.sort;
          return raw !== undefined ? raw : cell.textContent.trim();
        };

        const rows = Array.prototype.slice.call(tbody.querySelectorAll("tr"));
        rows.sort((a, b) => {
          const ka = cellKey(a);
          const kb = cellKey(b);
          const ea = ka === "" || ka == null;
          const eb = kb === "" || kb == null;
          if (ea && eb) return 0;
          if (ea) return 1;   // empties always last, regardless of direction
          if (eb) return -1;
          if (isNum) return (parseFloat(ka) - parseFloat(kb)) * sign;
          return ka.localeCompare(kb) * sign;
        });
        rows.forEach((tr) => tbody.appendChild(tr));
      }

      heads.forEach((th, colIndex) => {
        th.addEventListener("click", () => sortBy(th, colIndex));
        th.addEventListener("keydown", (e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            sortBy(th, colIndex);
          }
        });
      });
    });
  }
  makeVariantTablesSortable();

  // SAE table expanders: each button toggles the `collapsed` class on its
  // .sae-table-block (CSS hides rows 11+ when collapsed) and swaps its label.
  function wireSaeExpanders(root) {
    (root || document).querySelectorAll(".sae-expand-btn").forEach((btn) => {
      const block = btn.closest(".sae-table-block");
      if (!block) return;
      const fullLabel = btn.textContent.trim();  // "Show all N"
      btn.addEventListener("click", () => {
        const collapsed = block.classList.toggle("collapsed");
        btn.setAttribute("aria-expanded", String(!collapsed));
        btn.textContent = collapsed ? fullLabel : "Show top 10";
      });
    });
  }
})();
