/* The /variants results page, drawn from this tab's stored scan.
 *
 * The server sends the same empty page to everyone (templates/variants.html) with
 * only site-wide facts in #vq-config. Everything about the scan comes from
 * SwissScan.load(): the summary (counts, genes, provenance, derived) and the hits.
 *
 * Every value from the scan is written with textContent, never innerHTML: VCF
 * fields are user-supplied text, and there is no Jinja autoescape on this path.
 */
(function () {
  "use strict";

  //: Hit rows drawn. The scan records up to 20,000; the full list is in the download.
  var PAGE_MAX_HITS = 1000;

  var plural = window.SwissScan.plural;
  var config = JSON.parse(document.getElementById("vq-config").textContent);
  var known = {};
  config.known_genes.forEach(function (g) {
    known[g] = true;
  });

  function $(id) {
    return document.getElementById(id);
  }

  /* el("p", {class: "x"}, ["text", child, …]) — strings become text nodes. */
  function el(tag, attrs, children) {
    var node = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) {
      if (k === "style") node.style.cssText = attrs[k];
      else node.setAttribute(k, attrs[k]);
    });
    (children || []).forEach(function (c) {
      if (c === null || c === undefined || c === false) return;
      node.appendChild(typeof c === "string" || typeof c === "number"
        ? document.createTextNode(String(c)) : c);
    });
    return node;
  }

  function fmt(n) {
    return Number(n || 0).toLocaleString("en-US");
  }

  function geneUrl(gene) {
    return config.gene_url.replace("__GENE__", encodeURIComponent(gene));
  }

  // The slug comes with the hit (scanview.tis_slug), so the rule lives in one place.
  function isoformUrl(gene, slug) {
    return config.isoform_url
      .replace("__GENE__", encodeURIComponent(gene))
      .replace("__SLUG__", encodeURIComponent(slug));
  }

  function renderHeader(scan) {
    var s = scan.summary;
    var p = s.provenance || {};
    var box = $("scan-header");
    if (s.filename) box.appendChild(el("p", { class: "vq-sub" }, [s.filename]));
    // Naming the catalogue is what makes zero hits interpretable: without it,
    // "nothing found" is indistinguishable from "wrong index staged".
    box.appendChild(el("p", { class: "vq-sub" }, [
      "Resolved against ",
      el("strong", {}, [p.catalog_genes || "?"]), " genes / ",
      el("strong", {}, [p.catalog_isoforms || "?"]), " isoforms · index ",
      el("code", {}, [p.index_version || "unversioned"]),
    ]));
    // Stated next to the results: an hg19 file does not fail, it succeeds and
    // reports plausible garbage.
    box.appendChild(el("p", { class: "vq-sub" }, [
      "Positions were read as ", el("strong", {}, ["GRCh38"]), " coordinates.",
    ]));
    box.appendChild(el("p", { class: "vq-sub" }, [
      "Kept only in this browser tab; nothing about it is stored on the server.",
    ]));
  }

  function renderNotices(scan) {
    var s = scan.summary;
    var c = s.counts || {};
    var assembly = (s.derived || {}).assembly || {};
    var box = $("scan-notices");
    if (scan.stale) {
      box.appendChild(el("p", { class: "vq-partial" }, [
        el("b", {}, ["This scan was made against an older catalogue."]),
        " The site's ORF index has changed since, so residues and figure positions may " +
        "no longer line up. Upload the VCF again to refresh it.",
      ]));
    }
    if (s.hits_stored && s.hits_stored !== "full") {
      box.appendChild(el("p", { class: "vq-partial" }, [
        el("b", {}, ["This scan was too large to keep in full."]),
        s.hits_stored === "partial"
          ? " This tab kept the first " + fmt(scan.hits.length) + " of " + fmt(c.hits) +
            " hits; gene pages show markers for those only. Counts and the genes list " +
            "below are complete."
          : " No hits could be kept; counts and the genes list below are complete.",
      ]));
    }
    // The one failure this scan can detect on the user's behalf; before the funnel
    // because it calls every number after it into question.
    if (assembly.suspect) {
      box.appendChild(el("p", { class: "vq-assembly" }, [
        el("b", {}, ["These coordinates may not be GRCh38."]),
        " " + fmt(assembly.mismatches) + " of the " + fmt(assembly.ref_checked) +
        " hits whose REF base was checked (" + assembly.percent + "%) disagree with the " +
        "GRCh38 reference at that position. That is the signature of a VCF called " +
        "against a different assembly — hg19/GRCh37 most often. Lift the file over to " +
        "GRCh38 and scan it again; until then, treat the residues and consequences " +
        "below as unreliable.",
      ]));
    }
  }

  function step(num, label, isHit) {
    return el("div", { class: "vq-step" + (isHit ? " is-hit" : "") }, [
      el("span", { class: "vq-step-num" }, [num]),
      el("span", { class: "vq-step-lbl" }, [label]),
    ]);
  }

  function renderFunnel(scan) {
    var s = scan.summary;
    var c = s.counts || {};
    var d = s.derived || {};
    var assembly = d.assembly || {};
    var box = $("scan-funnel");
    // A budget stop makes every number below a partial; said before the funnel.
    if (c.stopped) {
      box.appendChild(el("p", { class: "vq-partial" }, [
        el("b", {}, ["This scan did not finish."]),
        " It stopped after " + fmt(c.lines) + " records " +
        (c.stopped === "time" ? "on its time budget" : "on its record budget") +
        ", so every count below describes only the part of the file that was read. " +
        "Split the VCF and scan it in pieces for a complete answer.",
      ]));
    }
    box.appendChild(el("div", { class: "vq-funnel" }, [
      step(fmt(c.lines), "records"),
      step(fmt(c.alleles), "alleles"),
      step(fmt(d.passing), "passed filter"),
      step(fmt(d.in_orf_alleles), "in an ORF", true),
      step(c.genes_hit || 0, "genes hit"),
    ]));
    // Losses first — they account for the narrowing above.
    var drops = "Dropped: " + fmt(c.skipped_non_pass) + " failed the caller's FILTER · " +
      fmt(c.no_orf) + " outside every annotated ORF · " + fmt(c.off_catalog_contig) +
      " on a contig the catalogue doesn't cover";
    Object.keys(c.rejected || {}).forEach(function (reason) {
      drops += " · " + fmt(c.rejected[reason]) + " " + reason.replace(/_/g, " ");
    });
    if (c.truncated) drops += " · hit list truncated at the cap";
    box.appendChild(el("p", { class: "vq-drops" }, [drops]));
    if (assembly.mismatches && !assembly.suspect) {
      box.appendChild(el("p", { class: "vq-drops" }, [
        fmt(assembly.mismatches) + " of the " + fmt(assembly.ref_checked) +
        " hits whose REF base was checked (" + assembly.percent + "%) disagree with " +
        "GRCh38 — normal at this rate, but a high fraction would mean the file was " +
        "called against another assembly. Intronic spans and indels are excluded: the " +
        "classifier never compares their REF.",
      ]));
    }
    // Hit records outnumber variants when a position sits in several isoforms.
    if (d.in_orf_alleles && c.hits && c.hits !== d.in_orf_alleles) {
      box.appendChild(el("p", { class: "vq-drops" }, [
        "Those " + plural(d.in_orf_alleles, "variant") + " produce " + c.hits +
        " hit records — one per (variant, isoform) pair.",
      ]));
    }
    if (!(s.genes || []).length) {
      box.appendChild(el("p", { class: "empty" }, [
        "No variant fell inside an annotated ORF. That is the usual result — most " +
        "somatic variants are intronic, UTR or intergenic, and the catalogue covers " +
        "only the coding regions of TIS-supported ORFs on transcripts that survived " +
        "filtering.",
      ]));
    }
  }

  function renderGenes(scan) {
    var genes = scan.summary.genes || [];
    if (!genes.length) return;
    var box = $("scan-genes");
    genes.forEach(function (g) {
      var name = known[g.gene]
        ? el("a", { class: "vq-gene-name", href: geneUrl(g.gene) }, [g.gene])
        : el("span", { class: "vq-gene-name" }, [g.gene]);
      box.appendChild(el("div", { class: "vq-gene" }, [
        name,
        // Both counts: one variant inside three isoforms is 1 variant, 3 hits.
        el("span", { class: "vq-gene-meta" }, [
          plural(g.n_variants, "variant") + " · " + plural(g.n_hits, "hit") + " across " +
          plural(g.n_isoforms, "isoform") + " · " + g.n_unique + " unique-region · " +
          g.n_shared + " shared",
        ]),
        known[g.gene] ? null
          : el("span", { class: "vq-gene-nopage" }, ["no detail page in this build"]),
      ]));
    });
    $("scan-genes-sec").hidden = false;
  }

  function hitRow(h) {
    // tis_id is chrom:pos:strand:codon:transcript; the start and codon are what
    // tell the sibling rows for one variant apart.
    var parts = String(h.tis_id || "").split(":");
    var tisLabel = (parts[1] || "") + " " + (parts[3] || "");
    var color = config.conseq_color[h.consequence] || "#64748b";
    var label = config.conseq_short[h.consequence] ||
      String(h.consequence || "").replace(/_/g, " ");
    return el("tr", {}, [
      el("td", { class: "var-mono" }, [h.chrom + ":" + fmt(h.pos)]),
      el("td", { class: "var-mono" }, [h.ref + "→" + h.alt]),
      el("td", {}, [h.gene]),
      el("td", { class: "var-mono" }, [
        known[h.gene] ? el("a", { href: isoformUrl(h.gene, h.tis_slug) }, [tisLabel]) : tisLabel,
      ]),
      el("td", {}, [el("span", { class: "orf-chip orf-" + h.orf_type }, [h.orf_type])]),
      // 0-based internally; +1 for display so it reads as p.N.
      el("td", { class: "var-num" }, h.residue === null || h.residue === undefined
        ? ["—"]
        : [String(h.residue + 1) + " ", el("span", { class: "vq-gene-meta" }, ["(" + h.frame + ")"])]),
      el("td", {}, [
        el("span", { class: "conseq-chip", style: "color: " + color }, [label]),
        // Shown, not a tooltip: it is why the amino acids beside it are blank.
        h.consequence_note ? el("span", { class: "conseq-note" }, [h.consequence_note]) : null,
      ]),
      el("td", { class: "var-mono" }, [
        h.aa_ref || h.aa_alt ? (h.aa_ref || "?") + "→" + (h.aa_alt || "?") : "—",
      ]),
      el("td", {}, [h.region]),
      el("td", { class: "var-num" }, [h.line_no]),
    ]);
  }

  function renderHits(scan) {
    var s = scan.summary;
    var c = s.counts || {};
    var d = s.derived || {};
    if (!(s.genes || []).length) return;
    var hits = scan.hits;
    var shown = hits.slice(0, PAGE_MAX_HITS);
    var box = $("scan-hits");
    // "Hits", not "Variants": one row per (variant, isoform) pair.
    box.appendChild(el("h2", {}, ["Hits (" + fmt(c.hits) + ")"]));
    box.appendChild(el("p", { class: "vq-sub" }, [
      plural(d.in_orf_alleles || 0, "variant") + ", one row per isoform each falls " +
      "inside — the same nucleotide is a different residue in each ORF, because each " +
      "starts at a different TIS.",
    ]));
    if (shown.length < (c.hits || 0)) {
      box.appendChild(el("p", { class: "vq-sub" }, [
        "Showing the first " + fmt(shown.length) + " of " + fmt(c.hits) + ". " +
        "Download the scan JSON below for every hit this tab kept.",
      ]));
    }
    var head = ["Position", "Change", "Gene", "Isoform TIS", "ORF type", "Residue",
      "Consequence", "AA change", "Region", "VCF line"].map(function (t) {
      return el("th", {}, [t]);
    });
    box.appendChild(el("div", { class: "variants-scroll" }, [
      el("table", { class: "variants-table" }, [
        el("thead", {}, [el("tr", {}, head)]),
        el("tbody", {}, shown.map(hitRow)),
      ]),
    ]));
    $("scan-hits-sec").hidden = false;
  }

  function wireButtons(scan) {
    $("scan-download").addEventListener("click", function () {
      var full = Object.assign({}, scan.summary, { hits: scan.hits });
      var blob = new Blob([JSON.stringify(full, null, 2)], { type: "application/json" });
      var link = el("a", {
        href: URL.createObjectURL(blob),
        download: (scan.summary.filename || "scan").replace(/\.vcf(\.gz)?$/i, "") +
          ".swissisoform-scan.json",
      });
      document.body.appendChild(link);
      link.click();
      link.remove();
      // Revoked on the next tick: revoking synchronously can cancel the download.
      setTimeout(function () { URL.revokeObjectURL(link.href); }, 0);
    });
    $("scan-clear").addEventListener("click", function () {
      window.SwissScan.clear();
      window.location.reload();
    });
  }

  var scan = window.SwissScan && window.SwissScan.load();
  if (!scan) {
    $("scan-empty").hidden = false;
    return;
  }
  renderHeader(scan);
  renderNotices(scan);
  renderFunnel(scan);
  renderGenes(scan);
  renderHits(scan);
  wireButtons(scan);
  $("scan-body").hidden = false;
})();
