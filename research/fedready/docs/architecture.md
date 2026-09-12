# FedReady Architecture

FedReady separates agent proposals from NVFlare-owned execution and promotion.

```text
task request
  -> server Codex inquiry proposal
  -> NVFlare broadcasts profile task
  -> client structural profiler + client Codex decision
  -> server Codex data-contract proposal
  -> NVFlare broadcasts extraction task
  -> client Codex adapter -> raw-input local-VLM guardrail
  -> structural/provenance validation -> materialization
  -> local renderers -> local-VLM alignment review -> orientation repair
  -> redacted extraction summary
  -> server Codex trainer -> package validation -> mock SimEnv preflight
  -> NVFlare FedAvg job -> aggregate metrics and global model
```

## NVFlare ownership

`FedReadyTaskQueryController` owns the two-round server lifecycle and uses
standard NVFlare `Controller` task distribution. `FedReadyTaskQueryExecutor`
owns client-local profiling, adapter execution, and redaction. `FedJob` packages
the controller and executor, while `FedReadyRecipe.execute(SimEnv(...))` uses
the current Recipe API to launch a local study.

The training phase packages generated Client API code into a regular FedAvg
job. The generated client must initialize FLARE, receive the global model in a
running loop, load the received parameters, compute and send a model diff, and
set `NUM_STEPS_CURRENT_ROUND`. The built-in server workflow aggregates those
validated updates.

## Trust boundaries

- The server sees client identifiers, aggregate profiles, counts, validation
  states, and digests. It does not see local paths or sample records.
- A client Codex worker is scoped to its request workspace and explicitly added
  local dataset roots so it can construct an adapter. Generated adapter code is
  then executed locally and validated before its output is accepted. These are
  research-process boundaries, not an OS-level data sandbox.
- A server Codex worker can write only inside the generated-code workspace.
- No failure falls back to hard-coded task logic or a second agent backend.
- Raw images, data-derived reference images, candidate sheets, and overlays
  remain on the client filesystem. Policy permits image-content review only
  through the loopback VLM; the Codex worker and locally executed adapter still
  have filesystem access needed for adapter construction. `prepare_ref.sh`
  selects canonical final-form references from existing prepared records; the
  assets are not distributed in the source tree or wheel.
- Acquisition-quality scoring and quality-based sample filtering are outside
  this contribution.

## Promotion gates

1. Guardrails authorize every agent action and FLARE payload shape.
2. Client adapters run in a bounded subprocess and produce typed manifests.
3. Provenance checks require label evidence to predate the generated adapter
   and prevent generated outputs from masquerading as source evidence.
4. Materialized records must satisfy the selected built-in or generated data
   contract and contain a non-empty train split.
5. Spatial contracts must produce contract-owned visual artifacts and a strict
   local-VLM consensus decision. For built-in segmentation, a selected flip or
   180-degree rotation is applied to every prepared mask, the repaired artifacts
   are regenerated, and the transform is recorded in `label_orientation` before
   training admission. Generated spatial contracts remain materializer-owned
   and are never generically rewritten.
6. Generated trainer packages are checked for required files, Client API
   behavior, and forbidden reference fallbacks.
7. A one-client mock-data NVFlare simulation must pass before real-client
   packaging and execution.

## Why the contribution uses Controller/Executor

Recent small examples use the Collab API for direct Python server-to-client
calls. FedReady retains Controller/Executor because it needs explicit task
headers, resumable two-round state, per-client failure handling, and exported
message/audit artifacts. It still follows the newer repository convention of
launching through a Recipe and `Recipe.execute`, and its generated training
phase uses the Client API and built-in FedAvg workflow instead of implementing
another aggregation abstraction.
