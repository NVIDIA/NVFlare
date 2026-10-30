# FedReady: Bounded Agents for Federation Formation

This directory contains the NVIDIA FLARE reference implementation of
**FedReady**, a protocol for turning a task request and heterogeneous private
site data into a verified, trainable federation. The source directory, Python
package, and command names consistently use the `fedready` identifier.

<img src="docs/assets/fedready_overview.png" alt="FedReady federation formation workflow: heterogeneous private site representations are harmonized by bounded agents and boundary verifiers before ordinary federated learning begins" width="1100">

*Federation formation end to end. (a) A task request arrives, but the center
cannot inspect the different private representations held by participating
sites. (b) Bounded agents propose a shared contract, local adapters, and
training code; boundary-local verifiers decide which artifacts and sites are
promoted. (c) Once the cohort and artifacts are frozen, every agent exits and
ordinary federated learning begins with one contract and uniform records.*

## Motivation

A federated study spans five stages: local data curation, shared data-contract
formulation, local adaptation to that contract, training-recipe construction,
and federated training. A curated local corpus and the FL runtime can be reused,
but the three stages between them recur whenever a new task changes the
required supervision interface. One study may require a binary mask, another a
boundary or bounding box, and another an image-level diagnosis, even when all
use the same underlying records.

This work is distributed by construction. Only a site can inspect and interpret
its private records, while only the coordinating center can establish a target
shared across sites. FedReady calls the work required to reconcile those views
**federation formation** and treats it as a first-class protocol rather than
informal setup. A federation is ready only when a cohort is admitted, a shared
contract is established, every admitted site can produce valid records under
it, and verified training code exists to consume those records.

## Bounded Agents

A single agent that can inspect private records and decide who joins the
federation collapses the separation that cross-silo FL depends on. FedReady
instead bounds every agent along three dimensions:

- **View:** the evidence it may observe.
- **Output:** the artifacts it may propose.
- **Authority:** the state it may change.

Generative agents have no promotion authority. They propose inquiries,
contracts, adapters, and training code; NVFlare-owned verifiers evaluate those
artifacts at the boundary that owns the required evidence and decide whether
they are promoted. Only typed, path-redacted summaries, aggregate counts,
status codes, and digests cross the federated transport boundary.

## Workflow

1. **Discover:** an NVFlare controller asks every site for safe aggregate
   metadata. Central routing remains high recall because the server cannot see
   the private evidence needed for final eligibility decisions.
2. **Form the contract:** a server Codex worker drafts a shared data contract
   from the task request and redacted summaries.
3. **Adapt locally:** each selected site Codex worker inspects its own data and
   generates an adapter against the locked contract.
4. **Verify at the boundary:** the client executes the adapter, checks
   provenance and contract validity, and performs bounded local visual review.
   Only verified sites enter the cohort.
5. **Prepare shared training:** a server Codex worker generates
   contract-compatible training code. Package checks and a one-client mock-data
   `SimEnv` preflight must pass before export.
6. **Freeze and train:** NVFlare freezes the qualified cohort and artifact
   graph, removes every agent from the loop, and runs ordinary sample-weighted
   FedAvg.

Agent outputs are proposals, not trusted state. A generated local adapter must
execute against client data and produce a provenance-valid manifest. Generated
training code must satisfy package and Client API contracts and pass a local
NVFlare simulation before it can be exported or run.

## Implementation Scope

This contribution provides the stable research artifact with one live
coding-agent backend (`codex`). A client-local VLM performs bounded raw-input
guardrail
inspection and image/label alignment review; for spatial labels, the workflow
renders orientation candidates. For the built-in segmentation contract, it
automatically applies a consensus flip or 180-degree transform before training;
generated spatial contracts retain ownership of their geometry. The separate
acquisition-quality scoring and sample-filtering workflow is deliberately
excluded.

The following are explicit non-goals of this research artifact:

- acquisition-quality grading, dataset-wide quality scoring, or quality-based
  sample filtering;
- general image enhancement, reconstruction, inpainting, or content repair;
- OpenHands, hosted chat APIs, or deterministic runtime fallbacks;
- production isolation or clinical/model-quality claims.

## Repository Layout

```text
research/fedready/
|-- README.md
|-- ACKNOWLEDGEMENTS.md
|-- requirements.txt
|-- pyproject.toml
|-- meta/site-meta.example.json
|-- meta/reference-sources.example.json
|-- prepare_ref.sh            # Select local visual-review references from data
|-- scripts/prepare_references.py # Standalone reference preparation
|-- task_example/             # Prepared and git-ignored reference output
|-- docs/data-download.md     # Public links for the 39-site retinal cohort
|-- docs/architecture.md
|-- docs/assets/fedready_overview.png # Federation-formation overview
|-- fedready/
|   |-- job_data.py           # Data preflight, FedJob construction, and execution
|   |-- job_train.py          # Generated trainer validation and FedAvg execution
|   |-- server.py             # NVFlare Controller and two-round data workflow
|   |-- client.py             # NVFlare client Executor and local validation
|   |-- agents/               # Codex bridge, agent logic, VLM/adapters, preflight
|   |-- data/                 # Profiling, extraction, QC, and typed contracts
|   |-- flare/                # FLARE messages and client/preflight runtimes
|   |-- prompts/              # Prompt loader and scoped JSON prompt bundles
|   `-- utils/                # Atomic I/O, audit logging, and training metrics
`-- tests/                    # Compact critical-path acceptance tests
```

See [docs/architecture.md](docs/architecture.md) for lifecycle and trust-boundary
details.

## Setup

Python 3.11 or newer, the Codex CLI, a local OpenAI-compatible vision endpoint,
and filesystem space for local simulator workspaces are required. GPU
requirements depend on the local VLM and generated training code. Codex must
already be installed and authenticated where the simulator processes run.

From the NVFlare repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -e 'research/fedready[training]'
codex --version
```

By default the local visual path uses
`Qwen/Qwen3-VL-8B-Instruct` at `http://127.0.0.1:8001/v1`. Override those
settings with `FEDREADY_VISION_AGENT_MODEL`,
`FEDREADY_VISION_AGENT_API_BASE_URL`, and
`FEDREADY_VISION_AGENT_API_KEY_ENV`. The endpoint must resolve to localhost;
the workflow sends image-content review requests only to that loopback endpoint.
The client Codex worker and generated adapter can still access the local dataset
to construct and validate the adapter, as described in the trust boundaries.

Before starting an experiment, launch the local endpoint in a separate host
terminal using an environment that provides vLLM, and leave it running:

```bash
export FEDREADY_LOCAL_VISION_API_KEY=local-vlm
OMP_NUM_THREADS=1 VLLM_HOST_IP=127.0.0.1 \
vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --host 127.0.0.1 \
  --port 8001 \
  --api-key "${FEDREADY_LOCAL_VISION_API_KEY}" \
  --limit-mm-per-prompt.image 3 \
  --limit-mm-per-prompt.video 0 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85
```

Export the same API key in the terminal used to run FedReady, then verify the
endpoint before continuing:

```bash
export FEDREADY_LOCAL_VISION_API_KEY=local-vlm
curl --fail --silent --show-error \
  --header "Authorization: Bearer ${FEDREADY_LOCAL_VISION_API_KEY}" \
  http://127.0.0.1:8001/v1/models
```

If vLLM reports a `TCPStore` timeout for `127.0.0.1`, use the IPv6 loopback
interface for both its internal rendezvous and API instead:

```bash
export VLLM_HOST_IP=::1
export VLLM_LOOPBACK_IP=::1
export FEDREADY_VISION_AGENT_API_BASE_URL='http://[::1]:8001/v1'
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}localhost,127.0.0.1,::1"
export no_proxy="${no_proxy:+${no_proxy},}localhost,127.0.0.1,::1"
# Re-run the vllm command above with: --host ::1
```

The contribution targets NVFlare 2.9 APIs on `main`. Until that version is
published, use the editable repository install above rather than replacing the
`nvflare~=2.9.0rc` requirement with an older release.

## Data Preparation

The workflow assumes each site's data is already available on that client's
local filesystem. It does not download, redistribute, or prescribe acquisition
steps for the datasets. Create a site registry from
`meta/site-meta.example.json`; each `data_path` may be absolute or relative to
the project root passed to `job_data.py`.

See [docs/data-download.md](docs/data-download.md) for the download page and
Kaggle reference associated with every site in the 39-site retinal cohort.

```bash
cp research/fedready/meta/site-meta.example.json \
  research/fedready/meta/site-meta.json
# Adjust the 39 relative data/<SITE_ID> paths if your local layout differs.
```

Client IDs must be unique, contain only letters, digits, `.`, `_`, or `-`, and
start with a letter or digit. This keeps every site-owned artifact inside its
run directory.

The local parser shares aggregate structure only. Paths, filenames, sample
identifiers, raw images, annotations, and generated adapter files are redacted
from server-visible responses. Dataset licenses and access terms remain the
operator's responsibility. The contribution does not distribute training data
or reference images.

Before running FedReady, select one existing prepared record for each supported
reference task. Copy `meta/reference-sources.example.json`, replace its four
sample-manifest paths with local paths, and prepare the digest-bound references:

```bash
cp research/fedready/meta/reference-sources.example.json \
  /tmp/fedready-reference-sources.json
# Edit /tmp/fedready-reference-sources.json to point at prepared local data.
bash research/fedready/prepare_ref.sh \
  /tmp/fedready-reference-sources.json
export FEDREADY_TASK_EXAMPLE_DIR="$(pwd)/research/fedready/task_example"
```

The preparer selects real image/target pairs from the declared manifests and
writes the same canonical final-form contract used by visual review. Source
paths may be absolute or relative to the source-config file. See
[task_example/README.md](task_example/README.md) for the accepted prepared-data
record shapes.
`FEDREADY_TASK_EXAMPLE_DIR` is optional for the documented editable install,
but setting it makes the external reference location explicit and also supports
normal wheel installations, where references are intentionally not packaged.

## Run Instructions

All commands below run from the NVFlare repository root. Prepare the references
from existing local records first if the digest-bound bundle is not present:

```bash
bash research/fedready/prepare_ref.sh \
  /tmp/fedready-reference-sources.json
export FEDREADY_TASK_EXAMPLE_DIR="$(pwd)/research/fedready/task_example"
```

Run the data-readiness entry point with a site registry, task, and project root:

```bash
TASK='binary glaucoma classification from fundus images using explicit local diagnosis or screening labels only; harmonize every admitted client to the shared label space 0=no glaucoma or non-referable, 1=an explicitly named glaucoma-spectrum positive or screening-positive state, including confirmed, possible, suspected, probable, or referable glaucoma. Do not use cup/disc masks, CDR, large-cup findings, or opaque numeric diagnosis codes as classification labels unless a local metadata source, README, codebook, legend, schema, or descriptive class-folder name explicitly defines the glaucoma-spectrum diagnosis or screening state. A descriptive local label that explicitly names the requested target remains semantic evidence when qualified as possible, suspected, probable, or referable; do not require a separate codebook merely to restate that descriptive label, and interpret the complete label so explicit negative or exclusion wording remains negative. If a local metadata source has multiple task-positive glaucoma-spectrum values, collapse those values to label 1 only when their local meanings are explicit; if the local evidence cannot support this binary mapping, mark the site unfeasible.'
python -m fedready.job_data \
  research/fedready/meta/site-meta.json \
  "${TASK}" \
  .
```

The value above is the exact task wording used by the showcased retinal
experiment. In particular, the target is glaucoma rather than arbitrary binary
image classification, and the local-evidence constraints are part of the task.

`job_data.py` directly runs the live-runtime preflight and then the two-round
NVFlare data workflow with fixed research defaults. Its JSON result reports the
generated session and run directory. Prepared outputs are isolated under
`data/dataset_fl_runs/<session-id>/`; existing data is never overwritten. The
fixed four-hour result timeout accommodates full materialization of large cohort
archives. The server decision used for training is written to:

```text
runs/<session-id>/server/decisions/extraction_round_summary.json
```

Pass that decision directly to the training entry point:

```bash
# Stop the local VLM after job_data.py exits to release its GPU memory.
python -m fedready.job_train \
  runs/<session-id>/server/decisions/extraction_round_summary.json \
  .
```

`job_train.py` reads the task and ready clients from the data decision, generates
and validates the trainer, resolves that data phase's run-isolated prepared-data
folder, and executes the showcased 100-round FedAvg experiment with fixed
defaults. Both entry points use the current public Recipe pattern: they connect
custom `FedJob` objects to NVFlare's public `SimEnv` and launch them with
`Recipe.execute`. The generated trainer uses NVFlare's Client API loop and
uploads model differences with `NUM_STEPS_CURRENT_ROUND` metadata for FedAvg
aggregation.

On some hosts, SimEnv may print `could not stop AIO loop` during process cleanup
after `Finished FedAvg`. The run is reported as completed with warnings only when
all client results were aggregated, the global model was persisted, and non-empty
client metric artifacts exist. The preflight applies the same evidence-bound
classification; earlier AIO or trainer errors still fail the run.

## Expected Results

A successful data phase reports extracted, screened-out, and failed clients and
writes a redacted extraction summary. A successful training phase produces:

- a Codex-generated training package under the run directory;
- package-validation and mock-data simulation attestations;
- an exported NVFlare job under `jobs/`;
- simulator logs, aggregate metrics, and the persisted global model under the
  training workspace.

### Showcased retinal experiment

The following is the most recent recorded `data_retinal` experiment result for
the exact task, cohort, and workflow described here. It is included as research
evidence, not as a claim that every run reproduces identical generated code or
metrics.

The `data_retinal` experiment queried all 39 registered retinal clients for the
exact glaucoma task above. Round-one aggregate profiling admitted 18 clients to
client-local extraction. Eight of those were screened out locally because their
available evidence could not satisfy the label contract; no admitted client
failed extraction, leaving 10 final training clients.

| Stage | Clients | Outcome |
|---|---:|---|
| Cohort queried in round 1 | 39 | Privacy-safe aggregate profiling |
| Initially admitted to round 2 | 18 | Local adapter generation and validation |
| Screened out in round 2 | 8 | Insufficient task-aligned local label evidence |
| Failed in round 2 | 0 | No unresolved extraction failures |
| Finally admitted to training | 10 | 26,549 image-label records |

The 26,549 records comprise 21,748 training, 2,400 validation, and 2,401 test
records. A Codex-generated ResNet-18-style trainer with GroupNorm then completed
100 FedAvg rounds across all 10 clients, with 10/10 client updates aggregated in
every round and the global model persisted. At round 99, the unweighted
client-mean training loss was 0.3086, validation accuracy was 0.8456, balanced
accuracy was 0.8006, and AUC was 0.8774. The best client-mean validation
accuracy was 0.8750 at round 87. The simulator reported one post-success AIO
loop cleanup warning, but no client-training or aggregation failure.

The curves below are read from the TensorBoard `_client_mean` series: an
unweighted mean of available client aggregate scalars for visualization. FedAvg
model aggregation remains optimizer-step-weighted; with the fixed batch size
and one local epoch used here, that is approximately sample-proportional.

<img src="docs/assets/tensorboard_glaucoma_100_rounds.png" alt="Client-mean training loss and validation accuracy over 100 FedAvg rounds for the 10-client binary glaucoma classification experiment" width="1000">

The agent generates a task-appropriate training strategy from the observed data
contract, so strategy selection adds run-to-run variability beyond ordinary
model-training randomness. For example, an earlier run selected DenseNet-121
with a small random rotation, while this run selected a ResNet-18-style model,
removed that rotation, added EXIF orientation repair, and changed the color
augmentation. The main FL settings remained fixed: 100 rounds, 10 clients,
batch size 2, one local epoch, 128x128 inputs, AdamW with learning rate 0.001,
class-weighted loss, DIFF updates, and optimizer-step-weighted FedAvg. Exact
predictive metrics therefore depend on the task, datasets, generated trainer,
and run budget. The research artifact establishes workflow completion and
failure containment; it does not claim a fixed benchmark score.

## Validation

The focused checks do not call Codex or a live VLM; renderer and reference
preparation tests use only temporary fixtures:

```bash
PYTHONPATH=research/fedready \
  python -m unittest discover -s research/fedready/tests -v
python -m compileall -q research/fedready/fedready
```

Live end-to-end validation additionally requires authenticated Codex, at least
two prepared client datasets, and any framework/GPU dependencies selected by
the generated trainer.

## License

The code is licensed under Apache License 2.0. No pretrained model weights,
training datasets, or reference images are redistributed; `prepare_ref.sh`
selects the required references from locally available prepared records. See
[ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md) for provenance.

## Requirements

Runtime dependencies are listed in `requirements.txt`; package extras are in
`pyproject.toml`. The code requires NVFlare 2.9 because it follows the current
Recipe execution and Client API contracts on `main`.

## Citation

The FedReady paper has not yet been published. Until a formal citation is
available, record the `fedready` contribution and NVIDIA FLARE revisions
used for the experiment.

## Retinal Benchmark Cohort

The retinal experiments use the 39-site retinal/fundus cohort derived from the
[FedAgentBench dataset catalog](https://arxiv.org/abs/2509.23803). The cohort version used here is identified as
`data_retinal`. This research example assumes the cohort has already been
made available at the client-local paths in the site registry.
