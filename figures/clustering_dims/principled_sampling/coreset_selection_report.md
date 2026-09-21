# cheeseman50 coreset

22 static anchors (12 genes) + 28 picks = 50 isoforms.
Space: all-ORF, 6,462 × 391.

**Verification: PASS**

## ORF-type composition

```
              final_50  anchors  rare_fill  sampler  pool  pool_pct
orf_type                                                           
3utr_orf             3        0          3        0    53       0.8
extended            18       12          0        6  1508      23.3
internal_oof         3        0          3        0    56       0.9
truncated           18        9          0        9  4510      69.8
uoorf                4        1          3        0    21       0.3
uorf                 4        0          3        1   314       4.9
```

## Fallback depth

How far past its own first choice the sampler had to reach because a gene was already claimed. Mostly 0 means the constraints barely perturbed the method.

```
rank_used
0    16

max fallback depth: 0
```

## The picks

```
    stage set  component  rank  rank_used gene_name     orf_type  value  inf_norm
rare_fill   Y        1.0   NaN          0      CMC2         uorf  4.147     4.147
rare_fill   Z        1.0   NaN          0      SPEN         uorf -1.601     2.374
rare_fill   W        1.0   NaN          0     PBRM1         uorf  0.009     2.826
rare_fill   Y        1.0   NaN          0     SPIN4 internal_oof  2.946     2.946
rare_fill   Z        1.0   NaN          0   PHACTR4 internal_oof -2.013     2.200
rare_fill   W        1.0   NaN          0      CCT3 internal_oof  0.009     0.983
rare_fill   Y        1.0   NaN          0  SMIM10L1     3utr_orf  2.729     2.729
rare_fill   Z        1.0   NaN          0       APC     3utr_orf -3.910     3.910
rare_fill   W        1.0   NaN          0    SPECC1     3utr_orf -0.033     2.619
rare_fill   Y        1.0   NaN          0    CGGBP1        uoorf  2.084     2.365
rare_fill   Z        1.0   NaN          0   RPS6KA5        uoorf -0.484     2.283
rare_fill   W        1.0   NaN          0      ADAR        uoorf  0.012     2.362
  sampler   Y        1.0   NaN          0    LGALS1     extended  5.011     5.011
  sampler   Y        2.0   NaN          0    NUCKS1    truncated  4.484     4.484
  sampler   Y        3.0   NaN          0    INCENP     extended  4.086     4.086
  sampler   Y        4.0   NaN          0      CBX5    truncated  3.680     3.680
  sampler   Y        5.0   NaN          0  HSP90AA1     extended  6.420     6.420
  sampler   Y        6.0   NaN          0      LMNA     extended  3.952     3.952
  sampler   Z        1.0   NaN          0     KMT2C    truncated -6.176     6.176
  sampler   Z        2.0   NaN          0      GBE1    truncated -3.790     3.790
  sampler   Z        3.0   NaN          0     ATP9A    truncated -3.212     3.212
  sampler   Z        4.0   NaN          0    PIEZO1     extended -3.869     3.869
  sampler   Z        5.0   NaN          0     ETAA1         uorf -2.720     2.720
  sampler   Z        6.0   NaN          0    GTPBP6    truncated -2.741     2.741
  sampler   W        NaN   1.0          0     ACBD5    truncated    NaN     0.308
  sampler   W        NaN   2.0          0     DELE1    truncated    NaN     0.323
  sampler   W        NaN   3.0          0    LRRC58    truncated    NaN     0.383
  sampler   W        NaN   4.0          0     AJUBA     extended    NaN     0.391
```
