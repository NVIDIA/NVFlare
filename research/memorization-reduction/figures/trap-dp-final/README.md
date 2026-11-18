# Final TRAP–DP figures

All 525 fits and 210 final evaluations are accepted. See the [research results](../../TRAP_DP_RESULTS.md)
for interpretation, thresholds, method differences and limitations. Seed 13 is the tuning seed;
29/47 reuse frozen configurations. Every declared strength and seed remains represented.

| Figure | PNG | Vector PDF | Provenance |
|---|---|---|---|
| Headline accuracy and measured leakage | [PNG](accuracy-leakage-headline.png) | [PDF](accuracy-leakage-headline.pdf) | Start here: every baseline/TRAP/DP final, descriptive medians/ranges and paired accuracy-loss evidence. |
| Paired screens by strength and seed | [PNG](paired-screen-overview.png) | [PDF](paired-screen-overview.pdf) | New overview; all 189 paired comparisons, recomputed with unchanged thresholds. |
| Frozen leakage-target attainment | [PNG](frozen-target-attainment.png) | [PDF](frozen-target-attainment.pdf) | New overview; all 70 method/K/target choices, including 39 unattainable choices. |
| Observed privacy and utility | [PNG](privacy-utility.png) | [PDF](privacy-utility.pdf) | Exact accepted canonical pair. |
| Utility versus federation size | [PNG](utility-by-k.png) | [PDF](utility-by-k.pdf) | Exact accepted canonical pair. |
| Directional attacks and nAUC | [PNG](attacks.png) | [PDF](attacks.pdf) | Exact accepted canonical pair. |
| Selected-fit computation and memory | [PNG](computation.png) | [PDF](computation.pdf) | Exact accepted canonical pair. |
| Greedy recovery and training exposure | [PNG](recovery-exposure.png) | [PDF](recovery-exposure.pdf) | Exact accepted canonical pair. |

## Reading the canonical figures

Gray = baseline; green = TRAP; blue = DP; brown = zero-noise control.
Circle/square/triangle markers identify seed 13/29/47. Curves retain declared strengths;
they do not represent a fitted or final-selected frontier. TRAP weight and DP epsilon are distinct quantities.
In the line figures, TRAP weights 0.5/1/2/4 use dotted/dashed/dash-dot/solid lines;
DP epsilon 1/2/4/8 use dashed/dash-dot/solid/dotted lines. Legends list seed 13 once;
the marker key applies to all seeds, including the recovery figure without a separate legend.
Some strength labels overlap in the privacy–utility figure, especially K=16; use the
[individual results](../../results/trap-dp-final/individual-results.md) for exact values.
Axes can have different ranges across panels. The control has no DP guarantee.

The computation figure shows selected-fit measurements. Complete tuning, overhead,
failed work and CPU evidence costs are in the [research summary](../../TRAP_DP_RESULTS.md)
and [allocation phase table](../../results/trap-dp-final/allocation-cost-by-phase.csv).
The recovery figure's lower panels show actual training record/token exposure, distinct from
candidate-exposure membership scores in the attack figure. Zero greedy recovery is not zero risk.

The five canonical pairs are byte-identical to the accepted November 5 delivery. The three overviews use only its accepted scalar tables; they are also copied unchanged. [Provenance and hashes](../../results/trap-dp-final/provenance.json) bind every committed figure. The original PNGs and rendered PDFs were visually reviewed. This PR publishes the results and protocol; the original cluster training and plotting harness is retained separately.
