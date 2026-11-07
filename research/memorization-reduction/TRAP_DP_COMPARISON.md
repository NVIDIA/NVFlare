# TRAP–DP benchmark protocol

This records the frozen design behind the [November 5, 2026 results](TRAP_DP_RESULTS.md): 525 fits and 210 final evaluations. The method is based on [TRAP: Understanding and Mitigating Privacy Memorization in Language Models](https://arxiv.org/abs/2609.32293). This benchmark predates the current SDK refactor; its source manifest and acceptance identities are pinned in [provenance.json](results/trap-dp-final/provenance.json). The cluster harness and raw data are not part of this PR.

## Task, reference and evaluation data

The APEX-style task extracts five visible fields from applicant dossiers: application ID, address, full name, email and phone number. The 2,000-record target half A contains 1,980 ordinary extraction records and 20 rare records requesting a random four-digit secret absent from the dossier. A disjoint 2,000-record half B trains the frozen reference for each seed. Validation and final-test splits contain 200 records each; exact field and whole-record accuracy use the 198 ordinary extraction records, while CE uses the full test mixture. Teacher-forced token accuracy is a separate metric. No raw dossiers or secret values are published in this PR.

The training secrets are split into ten development members and ten final members. Final membership evaluation pairs the ten final members with 200 nonmembers. The balanced partition uses six strata (four ordinary-record length quartiles and the two secret cohorts), with one fixed partition per K. Increasing K fragments the same union rather than adding data or deliberately introducing heterogeneity. Small member cohorts, previously inspected holdouts and reused identities limit generalization.

References are seed-matched, trained on B at LR 1e-5, selected by validation CE, then reused unchanged. Reference checkpoint hashes and historical costs are in [reference-construction-and-amortization.csv](results/trap-dp-final/reference-construction-and-amortization.csv). Training exclusion does not establish that the pretrained model never encountered related information. The available dossier pools were used to reconstruct the paper's five-field mixture; the sealed original mixture manifest was unavailable. This is not an exact reproduction of the paper's numerical results.

## Frozen design

Pythia-70m-deduped revision `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`, the fixed A/B/validation/test records and balanced six-stratum partitions described above, K = 2/4/6/8/12/16/32, seeds 13/29/47, full participation, record-weighted FedAvg, persistent AdamW, fp32 with TF32 disabled, maximum 40 epochs, 3% warmup, and patience three. Checkpoints minimize validation CE, including initialization. The full 40-epoch LR schedule is retained in short smokes.

| Configuration | LR | Clipping norm | Target local batch |
|---|---:|---:|---:|
| 1 | 2e-5 | 1 | 16 |
| 2 | 2e-5 | 0.1 | 64 |
| 3 | 1e-4 | 0.1 | 16 |
| 4 | 1e-4 | 1 | 64 |
| 5 | 5e-4 | 1 | 16 |
| 6 | 5e-4 | 0.1 | 64 |

This is a bounded joint search, not a factorial search. DP has epsilon 1/2/4/8 at delta 1e-5; TRAP has weight 0.5/1/2/4 and threshold zero. Baseline uses the same six configurations. At each K/strength, seed 13 selects highest validation field accuracy, then lowest validation CE, then lowest configuration number. `configurations-frozen.json` records all 378 validation-utility hashes before either replication or final testing. Seeds 29/47 use these frozen configurations. No test utility or final attack enters selection. Nonfinite/incomplete evidence blocks selection; failed scientific trials do not trigger replacements or extra tuning.

| Phase | DP | TRAP | Baseline | Controls | Total |
|---|---:|---:|---:|---:|---:|
| Seed-13 search | 168 | 168 | 42 | 0 | 378 |
| Seeds 29/47 replication | 56 | 56 | 14 | 0 | 126 |
| Fixed configuration-1 controls | 0 | 0 | 0 | 21 | 21 |
| All fits | 224 | 224 | 56 | 21 | **525** |
| Selected final evaluations | 84 | 84 | 21 | 21 | **210** |

Development operating points use all four frozen strength winners for each method/K. At strongest-nAUC targets 0.1/0.2/0.3/0.4/0.5, choose highest validation field accuracy among eligible strengths, then CE, configuration number and strength. Unattainable targets remain unavailable. This mapping is frozen before final evaluation and applied to all three seeds. Final achieved leakage is reported; it is never used to rematch points.

## Mechanisms and guarantee boundary

`opacus==1.6.0` is pinned. One privacy engine, accountant, optimizer and scheduler survives every round and device transfer. DP minimizes each record's mean completion loss, clips its gradient, adds Gaussian noise, and samples records independently with Poisson probability `q = 1/ceil(N/target_batch)`. Expected batch size is exactly `N*q`, at most public local capacity N; the target batch is capped implicitly by this rate. Every logical update is counted, including empty batches. Empty batches contribute a public zero-gradient dummy followed by the usual noise/update. Physical microbatch splitting never creates additional noise/accounting/scheduler events.

Baseline/TRAP retain token-mean completion loss and shuffled sampling without replacement. Their clipping norm applies to the aggregate gradient. The 21 zero-noise controls use the DP loss/sampling/per-record-clipping implementation at configuration 1; they quantify these combined changes and have **no DP guarantee**. These differences must not all be attributed to noise.

The accepted DP fits used ghost clipping after comparison with explicit per-record gradients on the actual Pythia checkpoint. Standard clipping with physical splitting is the fallback if ghost validation fails. Native and ghost paths use physical batches of 16; standard DP fallback uses one record to bound memory. These implementation choices and achieved throughput are recorded. Numerical unit tests use seeded nonsecure generators solely to make comparisons reproducible. Guarantee-bearing DP fits reject `secure_mode=False` and use a validated `torchcsprng` backend; no silent fallback is permitted. CSPRNG draws remain independent rather than reproducible from the training seed.

Noise is calibrated by RDP accounting for all `40*ceil(N/target_batch)` logical updates. Reports charge the entire actual stopping/selection process, not merely the prefix ending at the selected checkpoint. Accountants and optimizer state are durably saved before sending updates. Completed fits can be resumed at campaign boundaries. A partially released fit is never silently replayed: its failed attempt is preserved and conservatively charged its full calibrated horizon if complete accounting is unavailable. The client exposes state restoration for a verified checkpoint, but does not automatically reconstruct a crashed federated simulator session.

Formal bounds use the full predeclared 40-epoch ceiling. Realized stopping-time accountant values are reported separately as diagnostics, **not** as ex-post DP guarantees: validation-based stopping is adaptive and this implementation does not claim a privacy odometer. For seeds 29/47, the composed formal RDP bound envelopes every allowed batch policy because the winning configuration was chosen using tuning outputs. Fixed-configuration single-fit bounds, observed expenditure, tuning composition and all-run composition remain separate.

Adjacency is addition/removal of one record contribution within fixed public routing and capacity slots, with zero contribution for an absent slot. The record-level trainer represents an absent slot as `None`, retains its public sampling slot, and multiplies its dummy record loss by zero; capacity and normalization remain fixed. Routing, capacities, hyperparameters, auxiliary B references and validation are conditioned on as public. Protection concerns DP model/update outputs. The existing data-dependent partition generator, raw membership audits, non-DP/control outputs, sensitive diagnostics and timing side channels are outside this guarantee. This research artifact as a whole is **not** a DP release. Within one fixed partition, use parallel composition across sites; across tuning candidates, strengths, K, seeds and DP smokes sharing A, compose RDP. The report uses a conservative sum of per-fit maximum site RDP curves, with explicit conditional single-fit, tuning and all-run epsilon values at delta 1e-5.

## Measurement and evaluation

Primary fits use the same instrumented client and GB300 hardware class. Each GPU allocation reserves four GB300s; client work is scheduled across those devices. Records separate synchronized training, reference inference (a subset of training), validation, lease waiting, transfers, Client API send/receive time (including synchronization), audit time including its lease, full fit wall time through shutdown, actual records/completion tokens, peak client GPU/host memory, and throughput. Full Slurm allocations are deduplicated; batch/extern steps are never charged again. CPU preparation, failed probes and orchestration are retained. Queue time is separate. Per-fit reservation time is distinguished from full-lease startup/idle overhead. Final/common attack evaluation and all six tuning candidates have separate evidence.

The three B references are reused byte-for-byte, including their training histories and selected checkpoints. The reference-cost table records historical measured construction/training-validation time and amortization over 168/28/28 TRAP fits per seed. From-scratch reporting adds this reference cost separately; reused references have zero new construction cost. Historical shared-allocation startup cannot be attributed exactly and must not be presented as freshly measured reference end-to-end time. Reference inference during new TRAP training is freshly timed. Historical target allocations are excluded from this new campaign.

Identical evaluators score field/whole-record/token accuracy and CE; directional AUROC and nAUC for identifier log probability, loss++, TRA and 64-candidate exposure; strongest score; candidate exposure/rank summaries; identifier-only and full-prompt greedy recovery; and intermediate updates. Missing local development members are unsupported/unavailable, never zero leakage. Every other audit failure blocks validation. The report retains all paired screens against the newly tuned baseline: field accuracy loss at most 1 percentage point plus strongest-nAUC improvement at least 0.05, and a separate CE ratio at most 1.01.

## Independent acceptance and interpretation

All fits, final evaluations, successful parent allocations and the complete report passed independent evidence checks. The checks bind source, model, reference and input identities; frozen configurations and operating points; saved scores and aggregate recomputations; privacy identities; and terminal allocation costs. They do not regenerate missing per-token losses, generated text or raw candidate vectors. The [results report](TRAP_DP_RESULTS.md) retains those limitations, the historical timing gap and the native/canonical privacy arithmetic qualification.

Every declared strength and individual seed remains reported, including negative screens and unattainable development targets. Seed 13 is tuning, not an independent confirmation seed. A shared empirical nAUC threshold is not privacy equivalence; TRAP has no formal DP guarantee. The bounded six-configuration search does not establish optimal DP performance or universal TRAP superiority.
