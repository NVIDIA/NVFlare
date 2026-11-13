# NVIDIA FLARE Plugin Card

NVIDIA FLARE helps developers adapt machine learning workflows to federated
learning. The `nvidia-nvflare` plugin brings its existing conversion, diagnosis,
statistics, and Auto-FL skills into coding agents, with shared references and
templates included in one package.

**Release status:** candidate for review. Plugin evaluation, signing, and release
approval are pending. This maintainer-authored card records the current evidence;
NVIDIA Verified status and an evaluator-generated publication verdict are pending.

## Plugin information

| Field | Value |
| --- | --- |
| Display name | NVIDIA FLARE |
| Plugin identifier | `nvidia-nvflare` |
| Candidate version | `1.0.0`; versioned independently of the NVFlare SDK and component skills |
| Maintainer | NVIDIA FLARE Team — federatedlearning@nvidia.com |
| Source repository | [NVIDIA/NVFlare](https://github.com/NVIDIA/NVFlare), `2.9` branch |
| Product website | [NVIDIA FLARE](https://nvidia.github.io/NVFlare/) |
| License | [Apache-2.0](LICENSE); existing component notices are retained |
| Package contents | Eight user workflows and one shared support component |
| Integration formats | Agent Plugins 1.0.0, Codex, Claude Code, and Cursor |
| Execution | Local workspace and Python runtime; no bundled MCP server, hooks, or automatic installer |

## Intended users and workflows

The plugin is intended for researchers, data scientists, and ML engineers working
with NVIDIA FLARE. It supports the following workflows:

| Component | Purpose |
| --- | --- |
| [nvflare-orient](skills/nvflare-orient/SKILL.md) | Explain workflow choices and clarify open-ended requests or unresolved conversion ownership |
| [nvflare-convert-pytorch](skills/nvflare-convert-pytorch/SKILL.md) | Adapt plain PyTorch training loops to Client API model exchange, validate locally, and export a job |
| [nvflare-convert-lightning](skills/nvflare-convert-lightning/SKILL.md) | Adapt Lightning training with the Client API patch, validate locally, and export a job |
| [nvflare-convert-huggingface](skills/nvflare-convert-huggingface/SKILL.md) | Adapt Transformers Trainer or TRL SFTTrainer workflows, validate locally, and export a job |
| [nvflare-diagnose-job](skills/nvflare-diagnose-job/SKILL.md) | Investigate failed or stalled jobs using bounded log and runtime evidence |
| [nvflare-fed-stats](skills/nvflare-fed-stats/SKILL.md) | Compute supported tabular or image statistics across sites using FedStatsRecipe |
| [nvflare-autofl](skills/nvflare-autofl/SKILL.md) | Optimize an existing federated job with recorded candidates, metrics, and lifecycle state |
| [nvflare-autofl-report](skills/nvflare-autofl-report/SKILL.md) | Produce a final report, JSON summary, and progress plot for a stopped or interrupted campaign with complete evidence |
| [nvflare-shared](skills/nvflare-shared/SKILL.md) | Supply references and templates used by the other skills; selected through a consuming skill |

Example requests include converting a training script for two simulated sites,
explaining a reported job failure, computing federated statistics for a declared
dataset, or reporting the results of a stopped Auto-FL campaign. Component
instructions define the supported inputs and execution boundaries for each task.

## Inputs and outputs

Inputs depend on the workflow: training code and model definitions, dataset paths
and site assignments, requested statistics, job logs, or Auto-FL configuration,
candidate ledgers, metrics, and lifecycle records. Use concrete local paths and
state the intended environment and resource budget.

Outputs can include edited Python code, job configuration and exported jobs,
validation artifacts, statistical results, diagnostic explanations, optimization
records, Markdown reports, JSON summaries, and plots. Runtime artifacts belong
in the user's workspace; bundled evaluation datasets are test inputs, not runtime
instructions. Generated code and conclusions require review against actual run
results before operational use.

## Requirements and compatibility

Use NVFlare **2.9.0 or later** and the dependencies required by the selected
workflow, such as PyTorch, Lightning, Transformers, or TRL. Install the complete
plugin so sibling references to `nvflare-shared` remain available. Python and GPU
requirements depend on the selected NVFlare release and workload. Small synthetic
fixtures can run on CPU; larger models may require GPU resources.

The host must support local file access and command execution. The Codex manifest
declares `requires_local_executor: true`. Installing the plugin does not install
the SDK or frameworks, download models or datasets, or provision resources.

Native loading and routing were checked on November 12, 2026:

| Host | Version | Model used for the probe | Observed result |
| --- | --- | --- | --- |
| Codex | `0.167.2` | `gpt-6.1-sol` | Nine components discovered; routing and shared-reference probe passed |
| Claude Code | `2.1.312` | `aws/anthropic/bedrock-claude-opus-5-5` | Nine components discovered; routing and shared-reference probe passed |
| Cursor CLI | `2026.11.11-b912975` | Auto selected Grok 4.7 | Nine components discovered; routing and shared-reference probe passed |

These were read-only probes of the same skill and manifest content. They do not
establish completed conversions or compatibility with every host version.
Desktop installation checks and marketplace listings remain pending. The portable
manifest has passed schema validation; Hermes and ClawHub execution has not been
tested for this package.

## Permissions and data handling

The plugin operates through the selected agent's tools and permissions. Conversion
and statistics workflows can edit files and run local simulation. Reporting writes
artifacts. Auto-FL can run or submit jobs in the user-selected simulation, POC, or
production environment; those actions require the corresponding authorization,
credentials, and resource controls. The package itself grants no additional access.

No credential is bundled or required just to load the plugin. Model access uses
the host's configured authentication. Restricted datasets, model repositories,
remote services, and federated deployments can require separate credentials.
Keep secrets outside prompts, generated files, and retained logs.

Files or command output shared with the coding agent may be processed by its
configured model provider. Federated data placement and privacy controls apply to
the FLARE job; they do not automatically govern what the agent reads or sends to
its model. Use synthetic or approved data for initial agent workflows and review
the host's data-handling settings before exposing sensitive inputs.

## Validation evidence

The local preparation used the `2.9` component tree at source revision
`5f2250f1cd63ca2b3aff1952d950e3c3e9af5496`. All nine component directories,
including their instructions, scripts, references, datasets, reports, and retained
signatures, were preserved.

| Check | Result and scope |
| --- | --- |
| Package structure | Agent Plugins schema, Claude strict manifest validation, and repository skill lint passed for the initial candidate |
| Component integrity | All 165 resource hashes in nine retained signature manifests matched packaged files; cryptographic signing trust was not checked locally |
| Native loading and routing | Passed the host probes listed above |
| Plugin security scan | Earlier candidate blocked: OSS SkillEvaluator 0.8.0 with SkillSpector 2.13.3 reported findings and incomplete analysis; final package scan remains required |
| Plugin-level quality, lift, and agent/arm coverage | Not established; full workflow and formal plugin evaluation remain pending |

Packaging and discovery checks do not constitute agent-driven conversion tests
or a plugin benchmark.
Existing component reports describe their own evaluations; their scores are not
aggregated into a plugin score. The packaging guide documents local build and
validation commands. Full workflow evaluation is separate release work.

## Known limitations

Framework detection, generated code, and agent behavior depend on the project and
model. Inspect generated jobs and validation results before using them with real
data or shared resources. Diagnosis is limited by the available evidence. A final
Auto-FL report requires a stopped or interrupted campaign with complete candidate
records; an active campaign must not be represented as a completed experiment.

The report component's unchanged four-case dataset represents **one positive and
three target-relative negatives**. The tested evaluator loader normalizes all
four as positive for the report skill. That classification issue remains open;
a successful execution or green gate alone would not establish correct negative
case interpretation. The component's benchmark and card need regeneration through
the supported evaluation service once the interpretation is resolved.

The incomplete security scan also requires review and resolution. Retained
per-skill signatures cover component content; they do not attest to the plugin's
new manifests, this card, or catalog-generated metadata.

## Release qualification

Before a Verified release, retain complete official scan and evaluation evidence
for the final package, a publication-ready Plugin Card, and the accepted
plugin-level signature or attestation. Record approval covering plugin distribution
and the corresponding nSpect association. The separately approved marketplace
icon, repository review, catalog synchronization, and marketplace installation
checks also remain outstanding.

The builder records source provenance, file digests, and the archive digest in
`build-metadata.json` and `SHA256SUMS` alongside each archive. Release evidence must
identify the reviewed packaging revision and exact distributed content, including
any manifests regenerated by the catalog. Update this card when the contents,
supported environments, or evaluation evidence change.

## Support and references

- [NVIDIA FLARE documentation](https://nvflare.readthedocs.io/en/main)
- [NVIDIA FLARE discussions](https://github.com/NVIDIA/NVFlare/discussions)
- [NVIDIA FLARE issue tracker](https://github.com/NVIDIA/NVFlare/issues)
- [NVIDIA vulnerability reporting](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail)
- [NVIDIA skills catalog](https://github.com/NVIDIA/skills)

Report general questions and reproducible defects through the project community
channels. Use NVIDIA's vulnerability-reporting channel for security issues rather
than a public issue containing sensitive details.
