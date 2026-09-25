# Judging the prompt-variant arms

Weighs 9 arms × 50 isoforms × 7 output units = **3,150 outputs** with
`prometheus-eval/prometheus-8x7b-v2.0`, served locally.

Prometheus is used because it sits outside the Claude family (no self-preference
toward the system under test), its weights are fixed so every judgment is
re-runnable at zero marginal cost, and it is the only option that scales to
~25,200 calls. It has **no isoform biology**: it grades whether a verdict is
supported by the text in front of it. This ranks framings; it does not validate
science.

## Order of operations

```bash
# 1. Deterministic pre-checks. No GPU, no API. Run first — they can change the
#    rubrics, and they decide things a judge should never be asked.
python scripts/judge/run_checks.py

# 2. Build the requests. Ordered by cell so each reference prefix is prefilled
#    once for the ~108 calls that share it.
python scripts/judge/build_requests.py

# 3. Gates, before the expensive run. Each is minutes; the run is hours.
sbatch ... scripts/slurm/run_judge.sbatch --check-context
sbatch ... scripts/slurm/run_judge.sbatch --sanity-anchor
sbatch ... scripts/slurm/run_judge.sbatch --self-consistency

# 4. Score. bf16 on 2x A100-80, resumable: results.jsonl is append-only and
#    completed ids are skipped.
sbatch ... scripts/slurm/run_judge.sbatch --batch-size 64

# 5. Weigh.
python scripts/judge/analyze.py
```

## The unit of comparison

**A cell is one (isoform, output unit), and judging never leaves it.** 50 isoforms
× 7 units (C, D, L, M, P, S, synthesis) = 350 cells.

Never across isoforms — a truncation in a conserved gene and a uORF in a
variant-poor one have incomparable evidence, so a judge ranking them grades
biology. Never across categories — `tags` carries 24 tags in S against 3 in D, so
pooling makes vocabulary thinness look like a framing effect.

Per cell: C(9,2)=36 pairs × 2 presentation orders = 72 calls.

## How results are weighed

| rule | why |
|---|---|
| Bradley-Terry per category, status quo pinned at 0 | every number reads as log-odds vs what we ship |
| Order-inconsistent pairs dropped, not split | Prometheus 2 has position bias; splitting dilutes real signal |
| Cluster bootstrap over isoforms | the 7 units of one isoform share evidence and are not independent |
| Everything in floor units | see below |
| Nothing pooled across categories in a headline | S has 24 tags, D has 3 |

## The noise floor, and why it reorders the question

`criteria_hint_rep` is the status quo run twice. Its disagreement with
`criteria_hint` is the smallest difference this pipeline can resolve — measured
**11.7% overall, but 4.0% in Conservation and 20.0% in Structural
Characteristics**. Against each category's own floor:

| cat | floor | hint | raw | tags | dist |
|---|---|---|---|---|---|
| C | 4.0% | 2.5× | 7.0× | 3.0× | 8.5× |
| D | 6.0% | 2.7× | 5.3× | 2.7× | 7.0× |
| L | 6.0% | 2.7× | 4.3× | 5.7× | 8.3× |
| M | 16.0% | 1.0× | **0.8×** | 1.9× | **0.9×** |
| P | 18.0% | 1.7× | **0.9×** | **0.8×** | 1.3× |
| S | 20.0% | 1.9× | 2.8× | 1.5× | 2.6× |

Five of the eight framing effects in M and P are at or below their floor —
indistinguishable from running one arm twice. Those are the tool-loop categories,
so multi-turn tool use is where the model is least reproducible. Pooled, every
grounding clears the floor by 1.8–3.1×, which would read as "all effects are
real"; a third of them are not.

## Known limits

- **The fabrication rate is a ceiling, not a measurement.** Five systematic
  false-positive classes were found and fixed by reading successive runs (range
  dashes read as minus signs, decimal truncation, scientific notation, percentile
  landmarks, length arithmetic). What remains still contains some derived
  arithmetic. Read the per-arm *spread*, not the level.
- **The reference payload is itself a framing choice.** It is criteria+tags+dist,
  not `raw` — all four is ~25k tokens against a 32k context. The `raw` arms
  therefore carry ~75% more fabrication findings (49/47 vs criteria's 28/28),
  which is the confound made visible rather than a defect of those arms.
- **M and P are excluded from the fabrication check.** Their arms queried the full
  variant/structure tables through tool readers while the reference holds a 30-row
  sample (30 of 13,690 on the worst M cell).
- **No reference answers**, so verdicts are noisier than Prometheus's published
  benchmarks.
- **One rubric, gated on the anchor pairs.** Each pair states the same conclusion
  from the same numbers, one relating the measurements and one listing them, so
  support is equal by construction and only economy can decide. The rubric has to
  prefer the terse read in both presentation orders.
- **Prometheus has a total position bias on this corpus.** Given identical text in
  both slots it picked A in 35 of 35 decided comparisons. Only order-consistent
  pairs count as verdicts; without that filter the ranking would reflect request
  order.
- **48 references were trimmed to fit the 32,768 context**, 25 of them hard
  truncated (all synthesis, 3 isoforms), retaining a median 28,397 of the 29,000
  budget. `requests_meta.json` lists every one; a trimmed cell is weaker evidence.
