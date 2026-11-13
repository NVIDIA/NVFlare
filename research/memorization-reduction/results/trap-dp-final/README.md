# Final TRAP–DP scalar tables

The source tables are exact copies of the accepted November 5 scientific delivery's scalar exports;
the headline statistics are derived from those tables.
They contain no new evaluation or final reselection. Blank CSV values mean unavailable/not applicable;
they are not zero. See the [research results](../../TRAP_DP_RESULTS.md),
[figures](../../figures/trap-dp-final/README.md), and
[benchmark protocol](../../TRAP_DP_COMPARISON.md).

| Table | Coverage |
|---|---|
| [All final outcomes](all-210-final-outcomes.csv) | All 210 outcomes, seed roles, stopping and screen results. |
| [Individual results](individual-results.md) | Human-readable complete per-run tables. |
| [Final scalar measurements](final-results-scalars.csv) | All 210 finals, 70 fields, including directional attacks, accuracy, CE, recovery, exposure and fit measurements. |
| [All primary fit measurements](all-fits-scalars.csv) | All 525 fits, including every tuning candidate and actual stopping, throughput and memory. |
| [Paired comparisons](paired-screens.csv) | All 189 paired accuracy/leakage differences, CE ratios and screen outcomes. |
| [Screen counts by strength and seed](screen-counts-every-strength-and-seed.csv) | All 27 method/strength/seed groups, each with seven K comparisons. |
| [Frozen operating points](frozen-operating-points.csv) | 93 selected seed/target rows plus 39 unattainable method/K/target choices. |
| [Frozen choices with achieved leakage and runtime](operating-points-with-achieved-targets-and-runtime.csv) | All 132 frozen rows, with final outcomes; no rematching. |
| [All cross-method target pairs](all-frozen-cross-method-target-pairs.csv) | All 105 K/target/seed pairs, including absent choices and threshold misses. |
| [Compute by method and phase](compute-by-method-and-phase.csv) | All-fit, tuning, replica, control and overlapping selected-fit subsets; do not sum overlapping scopes. |
| [Allocation costs by phase](allocation-cost-by-phase.csv) | Complete 764-allocation accounting, including observers and completed report verification. |
| [Evaluator durations by method](final-common-evaluation-time-by-method.csv) | Descriptive count/sum/minimum/median/maximum; no extra allocation charge. |
| [Reference cost and amortization](reference-construction-and-amortization.csv) | Three reused references and historical training/validation costs. |
| [Canonical primary privacy bounds](canonical-primary-privacy-bounds.csv) | 224 logical DP fits and 225 output-path charges; historical smokes are additional. |
| [All 228 native privacy entries](native-accepted-privacy-scalars-all228.csv) | Full/formal/observed values, with failed-attempt provenance. |
| [Native composed privacy scalars](native-accepted-privacy-composition-scalars.json) | Full-horizon bounds and separately labeled stopping-time diagnostics. |
| [Headline comparison statistics](headline-comparison-statistics.json) | Derived medians, complete ranges, paired screen counts and accuracy losses; no final reselection. |

The canonical cumulative privacy bound is epsilon **152.13496880155347** at delta **1e-5**;
the native scalar composition differs slightly in final decimal places, as disclosed in the research results.
Observed stopping-time expenditure is diagnostic, not an ex-post guarantee. These tables and the complete
research bundle are not DP releases. Seed 13 is tuning, and repeated checkpoints across targets are not independent.

[Publication provenance](provenance.json) records the source and acceptance hashes and every published table/figure hash. The figures and scalar tables are unchanged copies; no new training, inference or final selection took place. Per-fit audit status counts are included in all-fit measurements. Raw audit records and per-record attack scores are not included.

CSV files retain their original CRLF record separators. The local `.gitattributes`
preserves those bytes across Git checkouts so the provenance hashes remain valid.
