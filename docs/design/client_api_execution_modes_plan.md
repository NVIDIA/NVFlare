# Client API Execution Modes — 2.9 Implementation Plan

Companion to `client_api_execution_modes.md` (design). Decomposes the design's 8-step Migration Plan into **36 PRs** with dependencies, sizes, and a release cut line. Scoped against the codebase post-2.8.0; granularity calibrated against the repo's merged-PR history (2026-07-01, see Calibration below).

Size guide: **S** <300 changed LOC, **M** 300–800, **L** 800–2000. Total: 6 S / 18 M / 12 L, no XL.

## Granularity calibration

Measured against the last ~300 merged PRs on main: median merged PR is ~140 LOC / 4 files; 65% land at ≤300 LOC; only ~13% fall in the 300–800 band. Recent comparable programs shipped as many small PRs over weeks: **recipe API ~28 PRs, tensor-offload/streaming reliability ~20, distributed provisioning/multicloud ~18** — each roughly half this program's scope. So ~36 PRs is in line with how this repo actually ships large features, and the plan's M-heavy granularity is already on the *large* side of repo norms (p75–p85). Consolidating further would create PR shapes the repo rarely merges for core code.

An adversarial consolidation pass was run on nine candidate merges; four survived (applied below), five were rejected. The pattern: **safe merges are same-owner/same-module dedups of work two tracks scoped twice, or two halves of one contract with no independent consumer. Bad merges drag an early foundation behind a later integration point, or weld a revert-sensitive change (interface freeze, wire-protocol version skew, security fix) to unrelated code.** Deliberate non-merges are listed in Guardrails at the end.

## Interface freezes (do these first, review across all track owners)

1. **Protocol + auth module** (`nvflare/client/cell/defs.py` + session-token/HMAC-proof helpers) — one module consumed by trainer engine, external_process, and attach (two tracks scoped it independently; it is one PR). Pure constants + crypto; no file I/O.
2. **ClientAPIExecutor skeleton + `ClientAPIBackendSpec`** (`nvflare/app_common/executors/client_api_executor.py`). Freeze the full V1 constructor from the design's Configuration Surface even though only in_process works initially.
3. **F3 aggregate terminal outcome** (`TransferOutcome`, additive `outcome_cb`). Reuse `TransferProgressState` terminal names; `transaction_done_cb` signature untouchable (benign-TIMEOUT callers: `hci/server/binary_transfer.py`, `app_opt/job_launcher/workspace_cell_transfer.py`).

## Wave plan

Tracks: **F3** payload layer · **EX** executor/ScriptRunner · **TE** trainer engine · **EP** external_process · **AT** attach · **CC** CCWF · **CT** compat/docs/tests.

### Wave 0 — foundations (no cross-deps; start immediately, fully parallel)

| PR | Track | Size | Notes |
|---|---|---|---|
| PR-0 Land design doc + this implementation plan in docs/design/ | all | S | **Lands first, merges fast** — the design is already approved; this just puts the reference material in-repo so every subsequent PR links it. Interface-freeze sign-offs happen on TE-1/EX-2/F3-1 themselves |
| F3-1 Aggregate all-receivers terminal transfer outcome | F3 | M | Interface freeze #3. Purely additive next to FINISHED/TIMEOUT/DELETED |
| TE-1 Protocol defs + stateless session-token/HMAC proof helpers | TE+AT | S | Interface freeze #1. Pure toolkit: defs vocabulary + TokenScope + generate/digest + compute/verify_hello_proof + combine_nonces. The stateful executor-side SessionTokenManager (single-use nonce issuance, attach-window expiry, single-session enforcement) is an attach-only need and moves to AT-2 — external_process uses only the lightweight launch-token proof over localhost |
| EX-2 ClientAPIExecutor skeleton + backend spec + analytics-event ownership | EX | M | Interface freeze #2 |
| TE-2 Bootstrap config schema + NVFLARE_CLIENT_API_CONFIG resolution | TE | M | Additive ConfigKeys; consumes EP-1's 0600 writer. Kept separate from TE-1: touches legacy-shared `client/config.py`, different revert profile |
| EP-1 0600 permissions for Client API config files | EP | S | Fixes live exposure in today's `client_api_config.json`; standalone + backportable by design |
| EP-2 TrainerProcessRunner (process-group lifecycle + PGID records) | EP | M | SIGTERM→grace→SIGKILL; preserves SubprocessLauncher's natural-exit window; Windows path platform-guarded |
| EX-1 Export `flare.get_task_name` | EX | S | Ship-today one-liner; kept out of the churn-prone skeleton PR |

### Wave 1 — parallel tracks behind foundations

| PR | Track | Size | Depends |
|---|---|---|---|
| F3-2 Receiver-confirmed completion + retry-aware accounting | F3 | M | F3-1. The version-skew wire change — lands as early as possible for maximum soak; capability-flag gated, both skews interop-tested |
| F3-3 Per-(transfer, receiver) acquire/idle budgets | F3 | M | F3-1. Unconditional per-receiver activity tracking. Must also settle the quorum surface for fan-out: workflows with min_responses-style policy (k-of-N receivers suffices) need either an optional min_receivers on the transaction/facade or a documented pattern of evaluating TransferOutcome.refs against their own threshold — `completed` stays the strict all-receivers certificate either way |
| F3-4 Awaitable producer transfer facade + PAYLOAD_ACQUIRED (via existing progress_cb) | F3 | M | F3-1. Must CHAIN existing DOWNLOAD_COMPLETE_CB, not replace |
| EX-3 in_process backend (consolidate InProcessClientAPIExecutor) | EX | L | EX-2. Behavior-parity bar: "nothing user-visible" |
| TE-3 TrainerCellSession engine (handshake, heartbeat, owner-death, trainer-side authenticated teardown) | TE | L | TE-1, TE-2. Injectable clock + kill hook; AT owner co-reviews the teardown-auth tests |
| EP-5 CP-side orphan reaping of trainer PGIDs | EP | M | EP-2. PID-reuse guard via start-time record |

### Wave 2 — contracts and wiring

| PR | Track | Size | Depends |
|---|---|---|---|
| F3-5 Bounded post-completion linger before producer release | F3 | S | F3-2, F3-4. Gates only the new `wait_released`; kept separate from F3-2 so the wire change isn't welded to exit-timing policy |
| TE-4 TrainerCellSession task/result contracts (receive queue + TASK_READY idempotency + TASK_FAILED; RESULT_READY flow + terminal-outcome blocking + fan-out drain/shutdown gating) | TE | L | TE-3; F3-4 via a narrow stubbed wait-protocol interface. Merged from two Ms: same class, same owner, no independent consumer of either half. Guard: split fan-out drain back out if the diff passes ~1800 LOC |
| EP-3 Launch-scoped token, CJ-side HELLO acceptance, and executor-side session-scoped message enforcement (both modes) | EP | M | TE-1. Enforcement (accept trainer messages only from the bound session) lives here — it is a P0 security control closing the live IPCAgent any-sender gap and must not ride in the P2 attach track |
| EX-4 ScriptRunner `execution_mode` param + launch_external_process mapping | EX | M | EX-2/3. Convert ~22 internal recipe call sites in the same PR |
| CC-1 CCWF transfer-declaration plumbing (receiver sets, stage windows, aux passthrough) | CC | M | F3-3. Declaration-only; absent headers preserve today's defaults — its behavior-neutrality is what de-risks the CC track |
| CT-8 Session observability (state-transition logs + StatsPoolManager view) | CT | M | EX-2; extends as backends land |

### Wave 3 — integration point

| PR | Track | Size | Depends |
|---|---|---|---|
| TE-5 CellClientAPI backend (flare.* on the new engine, backend selection) | TE | M | TE-4, TE-2. Opt-in only; legacy defaults untouched |
| EP-4 external_process backend for ClientAPIExecutor | EP | L | EP-2/3, EX-2/3, TE-5, F3-4. The step-4 integration point; watch for XL creep — split dispatch if needed |
| EP-6 Rank contract (torchrun multi-rank, non-control-rank fail-fast) | EP | M | EP-4, TE-5 |
| CT-1 Acceptance-test harness + core external_process suite + simulator/POC smoke | CT | L | EP-4. Absorbs the EP track's E2E validation (incl. POC-mode smoke); EP owner is co-reviewer as the track's sign-off point. tests/integration_test/fast (Blossom premerge) + xdist-safe unit subset |

### Wave 4 — validation, workflows, attach

| PR | Track | Size | Depends |
|---|---|---|---|
| CT-2 torchrun 2-rank rank-contract CI tests (CPU/gloo) | CT | M | CT-1, EP-4. EP owner co-reviews |
| CT-3 Owner-death (CJ-kill) + payload-lifecycle E2E tests | CT | L | CT-1, F3 track, EP-4. Covers CJ-SIGKILL self-termination + CP reaping; EP owner co-reviews |
| CC-2 Swarm onto the transfer contract (remove lazy-ref machinery) | CC | L | CC-1, F3 track, EP-4. Highest-risk PR in the program; maximally isolated revert unit; retires test_lazy_ref_local_aggr / test_msg_root_ttl |
| CC-3 Cyclic onto the transfer contract | CC | S | CC-1, EP-4. Kept out of CC-1 (would drag the behavior-neutral foundation behind EP-4) and out of CC-2 (swarm revert must not drag cyclic); shared broadcast_final_result declaration coordinates behind the CC-1 helper |
| CC-4 CSE / broadcast-best fan-out (N receivers, per-receiver budgets) | CC | L | CC-1/2, TE-4 fan-out drain |
| CC-5 Re-enable CCWF tensor disk offload (terminal-outcome-gated cleanup) | CC | M | CC-2/4. Deliberately last in track; flag experimental in 2.9 |
| AT-2 Attach session manager backend (CJ side) + attach-side session enforcement | AT | L | TE-1, EX-2, TE-3, EP-3. Includes the stateful `SessionTokenManager` (single-use nonce issuance, attach-window expiry, single-session, invalidation) built on TE-1's stateless proof helpers — deferred here from TE-1 since challenge-response state is an attach-only need |
| AT-3 Trainer-side attach flow (bootstrap config, ad-hoc cell, HELLO proof) | AT | M | AT-2, TE-5. Connects via CP parent_url (ad-hoc listeners default-disabled) |
| CT-5 Rank-contract example updates (multi-gpu/pt, pt-ddp-docker) + SLURM batch-artifact example | CT | M | EP-4. multi-gpu/pt violates the rank contract today; qwen3-vl is the reference |

### Wave 5 — release gate

| PR | Track | Size | Depends |
|---|---|---|---|
| AT-4 Attach hardening (rate limit, bounded proof attempts, reconnect rotation) + E2E smoke + job-config example + delivery docs | AT | L | AT-2/3. Merged from two Ms: same owner, sequential, both inside the P2 tail so they slip together. Full negative-case matrix lives once in CT-4, not here |
| CT-4 Attach + CCWF system-level acceptance suites | CT | L | CT-1, AT track, CC track. Owns the attach negative-case matrix (replay, duplicate attach_id, spoofed teardown) — written once here, deduped from AT-4. CCWF multi-site runs in slow/ (nightly) |
| CT-6 Client API docs overhaul (client_api.rst rewrite, 3rd-party/agent docs) | CT | L | Arg names frozen (steps 3–5 merged). Written against merged code, not the design doc |
| CT-7 Legacy-stack deprecation warnings | CT | S | Coverage gate met. Warnings only in 2.9; the ScriptRunner **default flip is a separate 2.10 PR** by design (different ship dates) |

## Program PR conventions

Every PR in the program uses this description skeleton so reviewers always have the map:

```markdown
## What
<one-paragraph scope — lift from this plan's PR entry>

## Program context
Client API Execution Modes (2.9) — PR <ID>, Wave <N> of the plan.
Design: docs/design/client_api_execution_modes.md § <section(s) this PR implements>
Plan:   docs/design/client_api_execution_modes_plan.md
Depends on: <merged PR links> · Unblocks: <plan IDs>

## Design contracts implemented
<bullets naming the specific contract, e.g. "SIGTERM → grace → SIGKILL process-group stop (design: Process-tree termination)">

## Out of scope (and where it lands instead)
<e.g. "CP-side orphan reaping — EP-5">
```

Notes: the one-paragraph "why" for the whole program lives in PR-0 and gets linked, not pasted. Interface-freeze PRs (TE-1, EX-2, F3-1) additionally name the tracks that consume the frozen surface and require sign-off from those owners before merge.

## Critical path

F3-1 → F3-4 → TE-4 → TE-5 → EP-4 → CT-1 → CT-3 → release gate. The TE-4 merge removed one review cycle from the path. **EP-4** remains the schedule risk to watch — it's where executor, trainer engine, auth, process runner, and the F3 facade meet; its prerequisites are deliberately extracted to keep it L.

## 2.9 cut line (recommendation)

- **P0 — commit for 2.9** (25 PRs): F3 track (5) + EX track (4) + TE track (5) + EP track (6) + CT-1/2/3/6/8. Headline: *external_process and in_process on the new Cell stack with an enforceable payload lifecycle, owner-death handling, session-scoped message enforcement, and the config-permission fix*.
- **P1 — strongly target for 2.9** (6 PRs): CC track (5) + the CCWF half of CT-4. Fully severable (SWARM keeps current behavior until CC-2 lands); ship CC-5 flagged experimental.
- **P2 — stretch for 2.9, else 2.9.x/2.10** (4 PRs): AT track (3) + the attach half of CT-4. Attach is the most self-contained step; nothing in P0/P1 depends on it (its auth/token/enforcement machinery ships in P0 via TE-1/TE-3/EP-3 regardless).
- **CT-7 warnings** (1 PR) close 2.9. **Explicitly deferred to 2.10**: the ScriptRunner default flip and any legacy-class removal — ship 2.9 opt-in with warnings; flip after a release of field soak.

## Effort and staffing

36 PRs ≈ 6 S + 18 M + 12 L. At review-inclusive rates (S ≈ 2 days, M ≈ 1 week, L ≈ 2 weeks) that is ≈ 44 engineer-weeks total; ≈ 34 for P0+P1. With 4–5 engineers owning tracks (F3, TE, EP+EX, CC, AT+CT), full parallelism after Wave 0; calendar floor is the critical path — realistically **10–14 weeks to P0+P1 complete** plus stabilization. If 2.9 code freeze is nearer, cut P2 first, then CC-4/CC-5.

## Guardrails — deliberate non-merges

These pairs are tempting to consolidate and must stay separate:

- **EP-1 (0600 fix) with anything**: live credential exposure on today's path; standalone, cherry-pickable, surgical revert.
- **EX-3 (in_process backend) + EX-4 (ScriptRunner)**: ScriptRunner is the most-used public entry point (golden exported-config tests, ~22 recipe conversions); a backend parity revert must not drag the public parameter surface.
- **Anything into CC-2 (swarm refactor)**: highest-risk PR in the program; must remain a maximally isolated revert unit.
- **F3-3 (budgets) + F3-4 (facade)**: same module, but F3-4 is on the critical path every track waits on; F3-3 serves only CC-1.
- **F3-2 (receiver-confirm wire change) + F3-5 (linger policy)**: the version-skew change wants early landing and a surgical revert; don't weld it to exit-timing policy.
- **EP-3 + AT-2**: same CJ-side HELLO/session machinery, but they straddle the P0/P2 boundary — AT-2 consumes EP-3's module instead.
- **EP-2 (runner) + EP-5 (CP reaping)**: different processes and blast radii; EP-5's false-positive-reap risk needs an independent revert path.
- **CT-7 warnings + default flip**: different ship dates by design (2.9 vs 2.10).

## Cross-cutting risks and constraints (from code scouting)

- **CI**: GitHub premerge is ubuntu-only, CPU-only, xdist — all acceptance tests must be CPU-feasible and port-dynamic; integration tests run only via Blossom (`/build`, fast/ premerge, slow/ nightly); fork PRs get no integration signal, so protocol contracts need cheap unit-level duplicates. torchrun tests: gloo only; multi-GPU/nccl validation is manual.
- **No Windows CI**: process-group termination Windows path ships platform-guarded with skipped tests — flag as a release-note caveat.
- **Version-skew interop** (F3-2): capability-flag gating with explicit old-producer/new-receiver and new-producer/old-receiver tests; confirm sends fire-and-forget.
- **Behavior parity**: in_process consolidation and ScriptRunner wiring have golden-config tests (exported job JSON byte-compare) to protect the most-used public entry point.
- **0600 writer is single-sourced**: EP-1 owns the hardened writer; TE-2 and attach bootstrap tests consume it rather than re-implementing permission handling.
- **Pass-through registration ownership** moves from ClientAPILauncherExecutor.initialize() into the new backend — coordinate EP-4 with CC-1 to avoid double registration during migration.
- **Trainer exit hang**: FlareAgent's atexit/main-thread-join watcher for non-daemon F3 threads solves a real problem (scripts that never call flare.shutdown()); TE-3/TE-5 must carry an equivalent or trainers hang at exit.
