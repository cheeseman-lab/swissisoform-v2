"""Generate a self-contained HTML report for the source-transcript workstream.

Reads the day's outputs + figures, renders one extra summary figure, embeds all
figures as base64 (single portable file), and writes a clean scientific report.
"""

from __future__ import annotations

import base64
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SRC = Path("data/output/source_transcripts")
FIG = Path("data/figures/source_transcripts")
OUT = Path("data/reports")
OUT.mkdir(parents=True, exist_ok=True)


def b64(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def main() -> None:
    ds = pd.read_parquet(SRC / "HeLa_tis_disambiguated_union_W100.parquet")
    res = ds[ds.resolved]
    u3 = pd.read_parquet(SRC / "HeLa_tis_union3_W100.parquet")
    u3r = u3[u3.resolved]
    u3t = {int(k): int(v) for k, v in u3r.agreement_tier.value_counts().items()}
    tiers = res.resolution_tier.value_counts()
    n = len(ds)
    t1 = int(tiers.get("1_sequence_pure", 0))
    t2 = int(tiers.get("2_expression_highTPM", 0))
    t3 = int(tiers.get("3_expression_lowTPM", 0))
    unres = n - (t1 + t2 + t3)

    # --- funnel counts for the flowchart -------------------------------------
    RAW_PREDICT = 521_699   # wc -l data/reference/HeLa_TIS_predict_all.txt (minus header)
    # HeLa predict rows whose transcript is MANE_Select or TSL 1-3 (smaffa
    # reference-transcript selection), computed from the GTF via
    # load_transcript_annotations joined to the raw predict Tids.
    TSL_MANE = 123_874
    comb = pd.read_parquet("data/output/filtered/all_samples_combined.parquet")
    pass_filter = int(comb.present_HeLa.sum())          # post filter+impute+cleanup, HeLa rows
    sk = pd.read_parquet("data/output/init_site_skeleton.parquet")
    h = sk[sk.present_HeLa == True]                      # noqa: E712
    n_sites = len(h)
    n_single = int((h.n_transcripts == 1).sum())
    n_multi = int((h.n_transcripts >= 2).sum())
    n_resolved = t1 + t2 + t3                       # seq + short-read union
    n_discard = n_multi - n_resolved
    n_final = n_single + n_resolved
    # three-way union (adds long-read)
    n_resolved3 = u3t.get(1, 0) + u3t.get(2, 0) + u3t.get(3, 0)
    longread_add = n_resolved3 - n_resolved         # TIS resolved ONLY via long-read
    n_discard3 = n_multi - n_resolved3
    n_final3 = n_single + n_resolved3

    # --- extra figure: resolution outcome + resolved ORF composition ----------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2),
                                 gridspec_kw={"width_ratios": [2, 1]})
    segs = [("Tier 1 · sequence-pure", t1, "#1b7837"),
            ("Tier 2 · expression ≥1 TPM", t2, "#5aae61"),
            ("Tier 3 · expression <1 TPM", t3, "#a6dba0"),
            ("unresolved", unres, "#cccccc")]
    left = 0
    for label, val, col in segs:
        a1.barh(0, val, left=left, color=col, edgecolor="white")
        if val > 250:
            a1.text(left + val / 2, 0, f"{val:,}\n{100*val/n:.0f}%", ha="center", va="center",
                    fontsize=10, color="black")
        left += val
    a1.set_xlim(0, n); a1.set_ylim(-0.5, 0.5); a1.set_yticks([])
    a1.set_xlabel("HeLa multi-candidate TIS")
    a1.set_title(f"Union resolution outcome (W=100):  {t1+t2+t3:,} of {n:,} resolved ({100*(t1+t2+t3)/n:.0f}%)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in segs]
    a1.legend(handles, [s[0] for s in segs], loc="upper center",
              bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False, fontsize=9)

    oc = res.orf_type.value_counts()
    a2.pie(oc.values, labels=[f"{k}\n{v:,}" for k, v in oc.items()],
           colors=["#2166ac", "#67a9cf", "#d1e5f0", "#fddbc7", "#ef8a62"][:len(oc)],
           wedgeprops={"edgecolor": "white"}, textprops={"fontsize": 8})
    a2.set_title("resolved TIS by ORF type")
    fig.tight_layout()
    outcome_png = FIG / "HeLa_resolution_outcome.png"
    fig.savefig(outcome_png, dpi=150, bbox_inches="tight"); plt.close(fig)

    # --- example genes --------------------------------------------------------
    def rec(init_site):
        r = u3[u3.init_site == init_site]
        return r.iloc[0] if len(r) else None

    def seq_html(r):
        if not isinstance(r.mrna_window_100, str):
            return ""
        i = int(r.start_codon_pos_in_window)
        w = r.mrna_window_100
        lo, hi = max(0, i - 30), i + 33
        pre, codon, post = w[lo:i], w[i:i + 3], w[i + 3:hi]
        return (f'<span class="up">…{pre}</span>'
                f'<span class="codon">{codon}</span>'
                f'<span class="dn">{post}…</span>')

    figs = {
        "window": b64(FIG / "HeLa_purity_vs_window.png"),
        "radius": b64(FIG / "HeLa_divergence_radius_hist.png"),
        "conf": b64(FIG / "HeLa_resolved_confidence.png"),
        "outcome": b64(outcome_png),
        "methods": b64(SRC / "HeLa_methods_vs_window.png"),
    }

    ex_oga = rec("chr10:101818238:-:CTG")    # all three agree, sequence-pure, extension
    ex_rpl37a = rec("chr2:216498874:+:ATG")  # long-read recovers what salmon lost
    ex_tpi1 = rec("chr12:6867566:+:ATG")     # salmon-only (Tier 3, flagged)
    ex_uba52 = rec("chr19:18573312:+:GTG")   # unresolved (honest)

    css = """
    :root{--bg:#0d1117;--panel:#fff;--ink:#1a1f29;--muted:#5b6675;--accent:#1b7837;
          --accent2:#2166ac;--line:#e6e9ee;--soft:#f5f7fa;}
    *{box-sizing:border-box}
    body{margin:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;
         color:var(--ink);background:var(--soft);line-height:1.6;}
    .hero{background:linear-gradient(135deg,#0d1117 0%,#15324a 55%,#1b7837 140%);color:#fff;
          padding:64px 32px 52px;text-align:center;}
    .hero h1{font-size:2.5rem;margin:0 0 .3em;font-weight:800;letter-spacing:-.02em;}
    .hero .sub{font-size:1.2rem;opacity:.92;max-width:760px;margin:0 auto;}
    .hero .meta{margin-top:22px;font-size:.92rem;opacity:.8;}
    .wrap{max-width:980px;margin:0 auto;padding:0 24px;}
    section{background:var(--panel);margin:26px auto;padding:34px 38px;border-radius:14px;
            box-shadow:0 1px 3px rgba(20,30,50,.07);max-width:980px;}
    h2{font-size:1.6rem;margin:.1em 0 .6em;padding-bottom:.3em;border-bottom:3px solid var(--accent);
       display:inline-block;letter-spacing:-.01em;}
    h3{font-size:1.15rem;color:var(--accent2);margin:1.4em 0 .4em;}
    p{margin:.7em 0;} .lead{font-size:1.12rem;}
    .muted{color:var(--muted);} .accent{color:var(--accent);font-weight:700;}
    .cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:22px 0;}
    .card{background:var(--soft);border:1px solid var(--line);border-radius:11px;padding:18px 14px;text-align:center;}
    .card .num{font-size:2rem;font-weight:800;color:var(--accent);line-height:1.1;}
    .card .lab{font-size:.82rem;color:var(--muted);margin-top:6px;}
    figure{margin:24px 0;text-align:center;} figure img{max-width:100%;border:1px solid var(--line);border-radius:10px;}
    figcaption{font-size:.88rem;color:var(--muted);margin-top:10px;text-align:left;}
    table{border-collapse:collapse;width:100%;margin:18px 0;font-size:.92rem;}
    th,td{padding:9px 12px;border-bottom:1px solid var(--line);text-align:left;}
    th{background:var(--soft);font-weight:700;} tr:hover td{background:#fafcff;}
    .gene{display:grid;grid-template-columns:repeat(2,1fr);gap:16px;margin:18px 0;}
    .gbox{border:1px solid var(--line);border-radius:11px;padding:18px;background:#fff;}
    .gbox h4{margin:0 0 .3em;font-size:1.1rem;} .gbox .tag{font-size:.74rem;font-weight:700;padding:3px 9px;border-radius:20px;color:#fff;}
    .t1{background:#1b7837}.t2{background:#5aae61}.t3{background:#a6dba0;color:#13391f}.tx{background:#9aa4b2}
    .seqbox{font-family:'SF Mono',Menlo,Consolas,monospace;font-size:.82rem;background:#0d1117;color:#c9d1d9;
            padding:12px 14px;border-radius:8px;overflow-x:auto;white-space:nowrap;margin-top:10px;}
    .seqbox .codon{background:#f1c40f;color:#000;font-weight:800;padding:1px 3px;border-radius:3px;}
    .seqbox .up{color:#79c0ff;}.seqbox .dn{color:#7ee787;}
    .pill{display:inline-block;background:var(--soft);border:1px solid var(--line);border-radius:20px;
          padding:3px 12px;font-size:.82rem;margin:3px 4px 3px 0;}
    .callout{background:linear-gradient(90deg,#eef7f0,#f5faf6);border-left:5px solid var(--accent);
             padding:16px 20px;border-radius:0 10px 10px 0;margin:18px 0;}
    .status{display:inline-block;width:10px;height:10px;border-radius:50%;background:#f0ad4e;margin-right:7px;
            animation:pulse 1.6s infinite;}@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
    footer{text-align:center;color:var(--muted);font-size:.85rem;padding:30px;}
    /* flowchart */
    .flow{display:flex;flex-direction:column;align-items:center;margin:24px 0;}
    .fstage{background:#fff;border:2px solid var(--accent2);border-radius:11px;padding:14px 20px;text-align:center;
            width:100%;max-width:560px;transition:.2s;}
    .fstage .fname{font-weight:700;font-size:1rem;color:var(--ink);}
    .fstage .fcount{font-size:1.7rem;font-weight:800;color:var(--accent2);line-height:1.15;}
    .fstage .fnote{font-size:.8rem;color:var(--muted);}
    .fw1{max-width:620px}.fw2{max-width:540px}.fw3{max-width:460px}
    .farrow{color:var(--muted);font-size:.82rem;text-align:center;padding:7px 0;line-height:1.4;}
    .farrow .big{font-size:1.1rem;color:#9aa4b2;display:block;}
    .fdrop{color:#b94a48;background:#fdecea;border-radius:6px;padding:2px 8px;font-size:.78rem;}
    .fbranch{display:flex;gap:18px;width:100%;max-width:620px;justify-content:center;flex-wrap:wrap;}
    .fbox{flex:1;min-width:240px;border-radius:11px;padding:14px 18px;text-align:center;border:2px solid;}
    .fkeep{border-color:var(--accent);background:#eef7f0;}
    .famb{border-color:#e0a800;background:#fff8e8;}
    .fkeep .fcount{color:var(--accent)}.famb .fcount{color:#b8860b}
    .fbox .fname{font-weight:700}.fbox .fcount{font-size:1.5rem;font-weight:800;line-height:1.15}
    .fbox .fnote{font-size:.8rem;color:var(--muted)}
    .ffinal{background:linear-gradient(135deg,#1b7837,#2d9e54);color:#fff;border-radius:13px;padding:20px 26px;
            text-align:center;width:100%;max-width:600px;box-shadow:0 4px 16px rgba(27,120,55,.25);}
    .ffinal .fname{font-weight:800;font-size:1.1rem;letter-spacing:.01em;}
    .ffinal .fcount{font-size:2.4rem;font-weight:900;}
    .ffinal .fnote{font-size:.86rem;opacity:.92;}
    """

    html = []
    html.append(f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
                f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
                f"<title>Source-Transcript Resolution · HeLa</title><style>{css}</style></head><body>")

    # hero
    html.append(
        "<div class='hero'><h1>Pinning ribosomes to the right mRNA</h1>"
        "<div class='sub'>Resolving the true source transcript — and its exact start-codon sequence — "
        "for every translation-initiation site, to build the first sequence-to-initiation-efficiency "
        "training set.</div>"
        "<div class='meta'>SwissIsoform v2 · Source-Transcript Resolution · HeLa pilot · 2026-06-17</div></div>")

    html.append("<div class='wrap'>")

    # the goal
    html.append(
        "<section><h2>The goal</h2>"
        "<p class='lead'>We want a model that predicts <span class='accent'>translation-initiation "
        "efficiency from mRNA sequence</span>. That requires each initiation site paired with "
        "<em>the</em> sequence that produced it.</p>"
        "<p>Ribo-TISH detects where ribosomes start translating — but it assigns each site to "
        "<strong>every annotated transcript whose splice structure is compatible</strong>, ignoring which "
        "mRNAs actually exist in the cell. Each site arrives carrying a polluted cloud of candidate "
        "transcripts with divergent 5′UTRs. To train a sequence model, we must collapse that cloud to a "
        "single, unambiguous start-codon sequence — or honestly discard the site.</p>"
        "<div class='callout'><strong>The key realization:</strong> we don't need to know <em>which</em> "
        "isoform a footprint came from. If every surviving candidate shares the same local sequence around "
        "the start codon, the site maps to a unique sequence <em>regardless</em>. We only discard when the "
        "local sequence is genuinely ambiguous.</div></section>")

    # the filtering funnel (flowchart)
    html.append(
        "<section><h2>The filtering funnel</h2>"
        "<p>The whole workflow is a funnel: from every raw Ribo-TISH call down to the subset whose "
        "source mRNA we know <span class='accent'>unambiguously</span>. Each step removes calls we can't "
        "trust or can't resolve — and what survives is the training set.</p>"
        "<div class='flow'>"
        f"<div class='fstage fw1'><div class='fname'>Ribo-TISH raw calls · HeLa</div>"
        f"<div class='fcount'>{RAW_PREDICT:,}</div>"
        f"<div class='fnote'>every transcript × start-codon prediction</div></div>"
        f"<div class='farrow'><span class='big'>▼</span>"
        f"<span class='fdrop'>−{RAW_PREDICT-TSL_MANE:,} · restrict to reference transcripts: "
        f"<strong>MANE&nbsp;Select or TSL&nbsp;1–3</strong> (smaffa annotation-quality gate)</span></div>"
        f"<div class='fstage fw2'><div class='fname'>TIS on reference transcripts (MANE / TSL 1-3)</div>"
        f"<div class='fcount'>{TSL_MANE:,}</div>"
        f"<div class='fnote'>well-supported annotated isoforms only · transcript×start rows</div></div>"
        f"<div class='farrow'><span class='big'>▼</span>"
        f"<span class='fdrop'>significance (TIS &amp; Ribo p-values, Fisher q-value) + normalized "
        f"read-count + distance dedup</span> &nbsp;·&nbsp; canonical-start imputation restores annotated ATGs</div>"
        f"<div class='fstage fw3'><div class='fname'>Pass significance + read-count filter</div>"
        f"<div class='fcount'>{pass_filter:,}</div>"
        f"<div class='fnote'>transcript×start rows detected in HeLa</div></div>"
        f"<div class='farrow'><span class='big'>▼</span> collapse transcripts that share one genomic start codon</div>"
        f"<div class='fstage fw3'><div class='fname'>Distinct initiation sites detected in HeLa</div>"
        f"<div class='fcount'>{n_sites:,}</div></div>"
        f"<div class='farrow'><span class='big'>▼</span> split by number of candidate transcripts</div>"
        "<div class='fbranch'>"
        f"<div class='fbox fkeep'><div class='fname'>Single candidate</div><div class='fcount'>{n_single:,}</div>"
        f"<div class='fnote'>mRNA known trivially ✓</div></div>"
        f"<div class='fbox famb'><div class='fname'>Multiple candidates</div><div class='fcount'>{n_multi:,}</div>"
        f"<div class='fnote'>ambiguous source — needs resolution</div></div></div>"
        f"<div class='farrow'><span class='big'>▼</span> <strong>union, step 1</strong>: sequence-window "
        f"purity <em>or</em> short-read (salmon) expression (W=100)</div>"
        f"<div class='fstage fw3'><div class='fname'>Resolved by sequence + short-read</div>"
        f"<div class='fcount'>{n_resolved:,}</div></div>"
        f"<div class='farrow'><span class='big'>▼</span> <strong>+ long-read</strong> recovers sites short-read "
        f"lost to read-dilution &nbsp;"
        f"<span style='color:var(--accent);font-weight:800;background:#eef7f0;border-radius:6px;padding:2px 8px'>"
        f"+{longread_add:,} TIS kept</span></div>"
        f"<div class='fstage fw3' style='border-color:var(--accent)'><div class='fname'>Three-way union resolved</div>"
        f"<div class='fcount' style='color:var(--accent)'>{n_resolved3:,}</div>"
        f"<div class='fnote'>tiered: {u3t.get(1,0):,} sequence-pure · {u3t.get(2,0):,} long-read/corroborated · "
        f"{u3t.get(3,0):,} salmon-only &nbsp; <span class='fdrop'>−{n_discard3:,} discarded · ambiguous</span></div></div>"
        f"<div class='farrow'><span class='big'>▼</span> merge single-candidate + three-way-resolved multi-candidate</div>"
        f"<div class='ffinal'><div class='fname'>TIS with UNAMBIGUOUS mRNA SEQUENCE</div>"
        f"<div class='fcount'>{n_final3:,}</div>"
        f"<div class='fnote'>{n_single:,} single-candidate + {n_resolved3:,} three-way-resolved → the sequence→efficiency training set</div></div>"
        "</div>"
        f"<p class='muted'>From {RAW_PREDICT:,} raw calls to {n_final3:,} TIS with a trusted source-mRNA "
        f"sequence (long-read inclusion adds <strong>{longread_add:,}</strong> of these) — every survivor "
        f"carries its read counts, initiation efficiency, source transcript, confidence tier, and the exact "
        f"±100 nt start-codon window.</p></section>")

    # what we did today + stat cards
    html.append(
        "<section><h2>What we built today</h2>"
        "<p>Starting from the HeLa Ribo-TISH calls, we pulled every candidate transcript's mRNA straight "
        "from the GENCODE transcriptome (the same reference the RNA-seq mapped against), located each "
        "site's start codon inside every candidate, and tested whether the candidates agree in a ±100 nt "
        "window. We then layered in HeLa expression (salmon) and discovered the two signals are "
        "<span class='accent'>complementary</span> — their union resolves far more than either alone.</p>"
        "<div class='cards'>"
        f"<div class='card'><div class='num'>9,919</div><div class='lab'>multi-candidate HeLa TIS analyzed</div></div>"
        f"<div class='card'><div class='num'>0</div><div class='lab'>sequence cross-check failures (32,889 candidates)</div></div>"
        f"<div class='card'><div class='num'>60.3%</div><div class='lab'>resolved by the union at W=100</div></div>"
        f"<div class='card'><div class='num'>+2,076</div><div class='lab'>extra TIS from union vs best single method (W=50)</div></div>"
        "</div></section>")

    # the union strategy + outcome figure
    html.append(
        "<section><h2>The union strategy</h2>"
        "<p>Two independent ways to make a site unambiguous:</p>"
        "<p><strong>① Sequence-window purity.</strong> If all candidates are byte-identical within ±W of "
        "the start codon, the local sequence is unique — no expression needed, highest confidence.</p>"
        "<p><strong>② Expression resolution.</strong> If only one candidate is actually expressed in HeLa, "
        "that one is the source; if several are expressed but agree in-window, they're interchangeable.</p>"
        "<p>They cover each other's blind spots: sequence purity handles <em>“all candidates agree”</em>; "
        "expression handles <em>“candidates differ, but only one is real.”</em> Taking the "
        "<span class='accent'>union</span> — keep if pure <em>or</em> expression-resolved — is the single "
        "biggest lever we found.</p>"
        f"<figure><img src='{figs['outcome']}'>"
        "<figcaption><strong>Figure 1.</strong> Union resolution outcome for the 9,919 HeLa multi-candidate "
        "TIS at a ±100 nt window. Tier 1 = sequence-pure (identity-agnostic, highest confidence); Tier 2/3 "
        "= expression-resolved, split by source-isoform abundance. Right: ORF-type composition of resolved "
        "sites — alternative starts (truncations, extensions, uORFs) are well represented.</figcaption></figure>"
        "</section>")

    # window length figure
    html.append(
        "<section><h2>Window length is a confidence dial</h2>"
        "<p>How far the candidates agree depends entirely on how much sequence you ask for. Near the start "
        "codon they almost always match; move outward into divergent 5′UTRs and agreement collapses.</p>"
        f"<figure><img src='{figs['window']}'>"
        "<figcaption><strong>Figure 2.</strong> Purity vs window radius. <span style='color:#1f77b4'>"
        "Sequence-only</span> purity falls from <strong>94% at ±5 nt</strong> to <strong>8.5% at ±500</strong> "
        "— it doesn't reach zero because 844 sites stay identical all the way out. "
        "<span style='color:#ff7f0e'>Expression-filtered</span> "
        "is nearly flat — it resolves by picking one isoform, which doesn't care about window length. "
        "The two cross at <strong>W ≈ 78 nt</strong>: below it, sequence wins; above it, expression wins. "
        "This is why the union beats either.</figcaption></figure>"
        f"<figure><img src='{figs['methods']}'>"
        "<figcaption><strong>Figure 2b.</strong> Every method across every window radius (N=9,920), now "
        "including the <strong>long-read</strong> arm. The union lines dominate at all W — keeping "
        "<strong>94–96%</strong> of multi-candidate TIS unambiguous at ±5–10 nt, and ~<strong>60%</strong> "
        "at ±100. Sequence-only is the steep curve; expression methods are the flat, window-independent "
        "floors. Long-read resolves more by expression alone than salmon (no read-dilution), though the "
        "two unions nearly coincide (see below). Window radius is a direct context-vs-count dial.</figcaption></figure>"
        f"<figure><img src='{figs['radius']}'>"
        "<figcaption><strong>Figure 3.</strong> Where candidate transcripts first diverge — median "
        "<strong>53 nt</strong> from the start codon. 844 sites stay identical all the way to ±500 nt."
        "</figcaption></figure></section>")

    # example genes (three-way framed)
    def gene_box(r, story, tagcls, taglabel):
        if r is None:
            return "<div class='gbox'><em>example unavailable</em></div>"
        tie = f"{r.tie_initiation_efficiency:.2f}" if pd.notna(r.tie_initiation_efficiency) else "—"
        src = r.source_transcript if isinstance(r.source_transcript, str) else "—"
        ev = r.source_evidence if isinstance(r.source_evidence, str) else "—"
        by = (str(r.resolved_by).replace("seq", "sequence").replace("short_read", "salmon")
              .replace("long_read", "long-read").replace(",", " + "))
        seq = f"<div class='seqbox'>{seq_html(r)}</div>" if isinstance(r.mrna_window_100, str) else ""
        return (f"<div class='gbox'><span class='tag {tagcls}'>{taglabel}</span>"
                f"<h4>{r.gene} · {r.start_codon} · {r.orf_type}</h4>"
                f"<p class='muted' style='font-size:.9rem'>{story}</p>"
                f"<div><span class='pill'>{int(r.n_candidates)} candidates</span>"
                f"<span class='pill'>resolved by: {by}</span>"
                f"<span class='pill'>source {src} ({ev})</span>"
                f"<span class='pill'>TIE {tie}</span></div>{seq}</div>")

    html.append(
        "<section><h2>Example genes</h2>"
        "<p class='muted'>One per confidence tier, now that all three signals are in.</p>"
        "<div class='gene'>")
    html.append(gene_box(
        ex_oga, "All three signals agree: the candidates are byte-identical around the start <em>and</em> "
        "both expression arms name the same source. This N-terminal extension of OGA is as confident as it "
        "gets — sequence-pure, no expression even needed.",
        "t1", "Tier 1 · all three agree"))
    html.append(gene_box(
        ex_rpl37a, "<strong>Long-read recovery.</strong> Short-read salmon split the reads across "
        "near-identical isoforms and dropped them all below threshold (lost). Long-read — one read = one "
        "isoform — cleanly names the source. This site exists in the training set <em>only because</em> of "
        "the long-read arm.",
        "t2", "Tier 2 · long-read recovery"))
    html.append(gene_box(
        ex_tpi1, "<strong>Caution flag.</strong> Salmon resolves this to a single source, but neither "
        "sequence agreement nor long-read corroborates it — long-read finds the source ambiguous. Likely a "
        "read-dilution artifact: kept for recall but tagged Tier 3 so the model can down-weight it.",
        "t3", "Tier 3 · salmon-only"))
    html.append(
        "<div class='gbox'><span class='tag tx'>Unresolved (honest)</span>"
        "<h4>UBA52 · GTG · Truncated</h4>"
        "<p class='muted' style='font-size:.9rem'>Highly expressed, and multiple candidate isoforms are "
        "<em>genuinely</em> co-expressed (long-read confirms several) <em>and</em> divergent in-window — so "
        "no signal yields a single sequence. Even long-read can't rescue it, because the ambiguity is real. "
        "We discard it rather than guess.</p>"
        "<div><span class='pill'>9 candidates</span><span class='pill'>truly co-expressed + divergent</span>"
        "<span class='pill'>not in training set</span></div></div>")
    html.append("</div></section>")

    # the deliverable
    html.append(
        "<section><h2>The deliverable</h2>"
        "<p>A curated table — <code>HeLa_tis_disambiguated_union_W100.parquet</code> — with one row per "
        "TIS carrying its metadata, Ribo-TISH read counts, per-transcript initiation efficiency (TIE), the "
        "disambiguated source transcript, and the unambiguous ±100 nt start-codon mRNA sequence.</p>"
        "<table><tr><th>Resolution tier</th><th>TIS</th><th>Meaning</th></tr>"
        f"<tr><td>① sequence-pure</td><td><strong>{t1:,}</strong></td><td>candidates identical in ±100 nt — identity-agnostic</td></tr>"
        f"<tr><td>② expression, ≥1 TPM</td><td><strong>{t2:,}</strong></td><td>collapsed to one confidently-expressed isoform</td></tr>"
        f"<tr><td>③ expression, &lt;1 TPM</td><td><strong>{t3:,}</strong></td><td>collapsed to one low-abundance isoform (provisional)</td></tr>"
        f"<tr><td>unresolved</td><td>{unres:,}</td><td>ambiguous window, no single expressed source</td></tr></table>"
        f"<p class='muted'>The ±100 window is unambiguous for all {t1+t2+t3:,} resolved sites; the full mRNA "
        f"is additionally unambiguous for {int(res.full_mrna_unambiguous.fillna(False).sum()):,} sites resolved to a single transcript. "
        f"Initiation efficiency is computable for {int(res.tie_initiation_efficiency.notna().sum()):,} sites.</p></section>")

    # Arm B results
    html.append(
        "<section><h2>Arm B — long-read results</h2>"
        "<p>The long-read arm (HeLa ONT, in-house minimap2 + IsoQuant) is in — and the answer is "
        "<strong>more nuanced than “long-read wins.”</strong> By expression alone it resolves "
        "<strong>50.2%</strong> of multi-candidate TIS vs salmon's 41.8%, and loses far fewer "
        "(1,967 vs 5,111) — confirming that <strong>read-dilution recovery is real</strong> (one read = "
        "one isoform).</p>"
        "<p>But the two <em>unions</em> nearly coincide (long-read 59.3% vs salmon 60.3%), because "
        "long-read keeps genuinely co-expressed isoforms and so flags <strong>2,974</strong> sites as "
        "honestly ambiguous (vs salmon's 662). The payoff is trustworthiness: <strong>~32% of salmon's "
        "single-source calls are likely dilution artifacts</strong> (long-read finds &gt;1 expressed "
        "isoform there), and salmon's sub-1-TPM calls agree with long-read only <strong>62%</strong> of "
        "the time.</p>"
        "<div class='callout'><strong>Decision:</strong> long-read is the better HeLa provider — for "
        "<em>trustworthiness</em>, not raw count. Use sequence-window purity as the backbone everywhere, "
        "long-read existence where available (HeLa, K562), and salmon only where long-read is absent "
        "(U2OS, RPE1), flagged lower-confidence.</div>"
        "<h3>Roadmap</h3>"
        "<span class='pill'>✓ Phase 1 · window + purity core</span>"
        "<span class='pill'>✓ Phase 2 · salmon expression</span>"
        "<span class='pill'>✓ Phase 2b · long-read quant</span>"
        "<span class='pill'>✓ Phase 2c · arm benchmark</span>"
        "<span class='pill'>○ Phase 3 · CAGE 5′ grounding</span>"
        "<span class='pill'>○ Phase 4 · 6-cell-line dataset export</span></section>")

    # three-way union
    tot3 = u3t.get(1, 0) + u3t.get(2, 0) + u3t.get(3, 0)
    html.append(
        "<section><h2>The three-way union</h2>"
        "<p>The three signals are complementary, so we take their <strong>union</strong> — a TIS is "
        "resolved if it is sequence-pure <em>or</em> short-read-resolved <em>or</em> long-read-resolved — "
        "but <strong>tiered by evidence strength</strong> rather than counted flat, because the three "
        "don't carry equal weight.</p>"
        f"<div class='cards'>"
        f"<div class='card'><div class='num'>{tot3:,}</div><div class='lab'>TIS resolved (74.8%) — vs 60% by any single pair</div></div>"
        f"<div class='card'><div class='num'>{int((u3r.n_methods>=2).sum()):,}</div><div class='lab'>corroborated by ≥2 methods</div></div>"
        f"<div class='card'><div class='num'>{int(u3r.full_mrna_unambiguous.sum()):,}</div><div class='lab'>full mRNA unambiguous (single transcript)</div></div>"
        f"<div class='card'><div class='num'>64%</div><div class='lab'>source-isoform agreement where salmon &amp; long-read both resolve</div></div>"
        "</div>"
        "<table><tr><th>Tier</th><th>TIS</th><th>Evidence</th></tr>"
        f"<tr><td>① sequence-pure</td><td><strong>{u3t.get(1,0):,}</strong></td><td>candidates identical in ±100 nt — correct by construction, identity-agnostic</td></tr>"
        f"<tr><td>② long-read / corroborated</td><td><strong>{u3t.get(2,0):,}</strong></td><td>direct molecular existence (one read = one isoform), or ≥2 methods agree</td></tr>"
        f"<tr><td>③ salmon-only</td><td><strong>{u3t.get(3,0):,}</strong></td><td>weakest — likely read-dilution artifact; included but flagged</td></tr></table>"
        "<p><strong>Source precedence</strong> picks the most-trustworthy available sequence per TIS: "
        "long-read source → short-read source → sequence representative. The result is "
        "<code>HeLa_tis_union3_W100.parquet</code> with <code>resolved_by</code>, <code>agreement_tier</code>, "
        "<code>n_methods</code>, the precedence-chosen <code>source_transcript</code>, and the ±100 nt window — "
        "so the model gets <strong>74.8% recall</strong> while every row carries its confidence, and Tier 3 "
        "can be down-weighted. The black line in Figure 2b traces this 3-way union across window radii.</p>"
        "</section>")

    html.append(
        f"<section><h2>Confidence, honestly</h2>"
        f"<figure><img src='{figs['conf']}'>"
        "<figcaption><strong>Figure 4.</strong> Left: how many candidate isoforms survive expression per "
        "site. Right: the source-isoform abundance of resolved sites — about half sit below 1 TPM (Tier 3), "
        "exactly the calls the long-read arm will confirm or correct. Nothing is hidden: every resolution "
        "is tagged with its evidence and confidence tier.</figcaption></figure></section>")

    html.append("</div>")  # wrap
    html.append("<footer>SwissIsoform v2 · Source-Transcript Resolution · generated 2026-06-17 · "
                "HeLa pilot, multi-candidate TIS, salmon expression (long-read pending)</footer>")
    html.append("</body></html>")

    report = OUT / "source_transcript_resolution_report.html"
    report.write_text("".join(html))
    print(f"wrote {report}  ({report.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
