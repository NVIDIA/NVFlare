# NVIDIA FLARE plugin

`nvidia-nvflare` packages NVIDIA FLARE's existing agent skills for Agent Plugins,
Codex, Claude Code, and Cursor. The proposed initial plugin version is `1.0.0`;
it is independent of the NVFlare Python package and individual skill versions.
This packaging candidate is not yet a signed or approved plugin release.

See the [Plugin Card](plugin-card.md) for intended use, requirements, permissions,
validation evidence, and known limitations.

## Contents

| Directory under `skills/` | Workflow |
| --- | --- |
| `nvflare-orient` | Route open-ended NVIDIA FLARE questions |
| `nvflare-convert-pytorch` | Convert plain PyTorch training to federated jobs |
| `nvflare-convert-lightning` | Convert Lightning training to federated jobs |
| `nvflare-convert-huggingface` | Convert Hugging Face Trainer and TRL SFTTrainer workflows |
| `nvflare-diagnose-job` | Diagnose failed or stalled jobs |
| `nvflare-fed-stats` | Compute federated statistics |
| `nvflare-autofl` | Optimize an existing federated job |
| `nvflare-autofl-report` | Report a stopped or interrupted Auto-FL campaign |
| `nvflare-shared` | Required references and templates used by the other skills |

Install the complete bundle to preserve sibling references. `nvflare-shared` is
a support component, not a separate user workflow; some hosts may nevertheless
list it because they do not interpret NVIDIA's internal-status metadata.

There is no MCP server, hook, or automatic installer. The bundle contains the
complete skill directories, including evaluation fixtures and retained skill
reports and signatures. Evaluation fixtures are grading data, not runtime
instructions. Skill names, instructions, scripts, references, and evaluation
data are preserved by packaging.

The workflows require NVFlare 2.9.0 or later and the Python/framework dependencies
specified by the selected workflow. Installing the plugin does not install
NVFlare, download models, provision GPU resources, or grant permission to submit
jobs. Conversion and Auto-FL workflows can edit files or run jobs; use a scratch
workspace and local simulation for initial testing.

## Manifests and source of truth

The repository root is the plugin root, and `skills/` remains the source of truth.

| Target | Entry point | Skill discovery |
| --- | --- | --- |
| Agent Plugins 1.0.0 | `plugin.json` | Root `skills/` directory |
| Codex | `plugin.json` plus `.codex-plugin/plugin.json` | Portable discovery; compatibility manifest also declares `./skills/` |
| Claude Code | `.claude-plugin/plugin.json` | Default root `skills/` directory |
| Cursor | `plugin.json` or `.cursor-plugin/plugin.json` | Portable discovery or the native `./skills/` path |

Shared identity fields must agree in all four manifests. The Codex compatibility
manifest supplies the display label **NVIDIA FLARE**. There is no inline
`extensions.com.openai` override in the portable manifest.
It also declares `requires_local_executor: true`, because these workflows use
local Python programs and workspaces. The OpenAI marketplace entry is intended
for desktop execution.

The build copies real files and preserves executable permissions. It packages
only tracked skill files, the four manifests, this guide as `README.md`,
`plugin-card.md`, and `LICENSE`. Repository development settings and unrelated source code are excluded.
No component symlinks are used.

## Build and validate

Run these commands from a checkout containing this packaging change, with Python
3.10 or later. The build itself uses only the standard library and Git. Choose a
new output directory for each build:

```bash
python3 dev_tools/agent/package_plugin.py --output /tmp/nvidia-nvflare-candidate
```

The output contains `nvidia-nvflare/`, `nvidia-nvflare-1.0.0.zip`, `SHA256SUMS`, and
`build-metadata.json`. The metadata records the source commit, skill tree, every
packaged file's SHA-256 digest, and the archive digest. ZIP entries use fixed
timestamps so identical inputs produce identical archives with the same toolchain.
No private keys, credentials, or local logs are required by the build.
During preparation, `packaging_matches_source_commit` is false if the manifests,
guide, card, or builder have not been committed. Commit the reviewed packaging and
rebuild before using the source commit as provenance for a release.

The builder requires all nine skills and checks every bundled skill file against
the resource lists in the retained `skill.oms.sig` files. A mismatch, missing
resource, unrecorded skill file, symlink, or changed skill tree fails the build.
These digest comparisons establish content preservation; they do not verify the
signing certificate's trust, constitute a new evaluation, or sign the plugin.

Run the existing skill checks against the extracted package (requires PyYAML in
the development environment):

```bash
python3 -m dev_tools.agent.skills.checks \
  --skills-root /tmp/nvidia-nvflare-candidate/nvidia-nvflare/skills
claude plugin validate /tmp/nvidia-nvflare-candidate/nvidia-nvflare --strict
```

Validate root `plugin.json` against the published Agent Plugins 1.0.0 schema using
a JSON Schema Draft 2020-12 validator. Check the native manifest fields against
the host documentation below. Keep schema validation, native loading, and actual
workflow evaluation as separate evidence; passing one does not establish the others.

With SkillEvaluator OSS 0.8.0 and its security dependencies installed in an
isolated environment, run the deterministic plugin checks explicitly:

```bash
skillevaluator validate /tmp/nvidia-nvflare-candidate/nvidia-nvflare \
  --type plugin --tiers 1 --no-dedup --no-tier3 --no-autopilot --no-llm \
  -r cli,json,html -o /tmp/nvidia-nvflare-tier1
```

Inspect scanner completeness and the detailed findings as well as the exit code.
An incomplete scan is a release blocker. Keep its report for triage; do not remove
signed resources or suppress findings merely to obtain a passing package scan.
This command does not run live agents or establish plugin signing eligibility.

## Host smoke tests

Test a copy extracted outside the repository so source-relative dependencies
cannot hide missing package files. Use a disposable user workspace and avoid
loading separately installed copies of these same skills during the test.

- **Claude Code:** start `claude --plugin-dir /path/to/nvidia-nvflare`. Confirm
  discovery, then try `/nvidia-nvflare:nvflare-orient`. Use the host's plugin listing
  to inspect all bundled skills before running conversion scenarios.
- **Codex:** use the documented local marketplace/import flow for the extracted
  plugin. Confirm the NVIDIA FLARE display label and skill inventory in a fresh
  session. The build command does not change your global agent configuration.
- **Cursor:** import the package through the supported plugin development or team
  marketplace flow for the installed version. Confirm discovery and relative
  references in a fresh session. A CLI that supports `--plugin-dir` can also test
  the extracted directory directly with `agent --plugin-dir /path/to/nvidia-nvflare`.
  Use `--mode ask --sandbox enabled` for a read-only discovery/routing check.
  Native UI loading remains a separate manual check.
- **Other Agent Plugins clients:** load the portable root manifest and confirm
  all nine directories are discovered, including shared resources.

For each host, record its exact version, plugin archive digest, discovered skill
names, and scenario outcomes. Exercise one small PyTorch conversion, one
Lightning conversion, one Hugging Face conversion, failed-job diagnosis,
federated statistics, and the Auto-FL/report boundary. Use only synthetic data
and bounded local runs. Include negative requests that should not select a
converter or the report skill. Host namespacing and skill routing must be checked
in real sessions even though the skill files are unchanged.

A read-only loading/routing probe confirms skill discovery, selection, and shared
reference resolution. It does not establish successful conversions, training,
statistics execution, report generation, or Tier 3 coverage. Retain those workflow
results separately, with their input fixtures and generated artifacts.

## Release evidence checklist

- [ ] Review the exact package inventory, license notices, dependencies, and model
  or dataset references; retain third-party attribution where applicable.
- [ ] Record explicit approval covering this plugin and its distribution, and
  associate the approved OSRB record with the appropriate nSpect registration.
- [ ] Complete native loading and workflow smoke tests for the supported hosts.
- [ ] Add an approved square marketplace icon and confirm each marketplace's
  presentation requirements and final installation entry. The current candidate
  has no icon or published marketplace installation command.
- [ ] Scan and evaluate the final plugin through the supported NVIDIA pipeline;
  retain tool versions, source revision, report links, complete configured
  agent/arm coverage, execution errors, canonical case IDs, and gate verdicts.
- [ ] Resolve or obtain a documented reviewer disposition for the Auto-FL report
  task-polarity concern tracked in [NVFlare #5294](https://github.com/NVIDIA/NVFlare/pull/5294).
  Its four canonical cases represent one positive and three target-relative
  negatives. A report labeling all four positive does not validate that distinction.
  The current `2.9` benchmark also retains a legacy internal container-image path;
  regenerate this report through the evaluation service instead of manually
  editing signed artifacts. Verify the final benchmark and card together.
- [ ] Generate a Plugin Card and plugin-level signature for the exact final
  package. Existing per-skill signatures do not attest to the new manifests or
  the bundle as a whole. Verify the final published package against its attestation.
- [ ] Review and merge the packaging into the public repository, select the release
  revision/version, then prepare the NVIDIA/skills plugin catalog sync PR using
  the program's current submission format. Confirm resulting marketplace content
  and install behavior before announcing availability.

Preparation and local validation can proceed while approval is pending. Keep
unverified items open and avoid describing the candidate as NVIDIA Verified until
the required review, evaluation, signing, and publication steps are complete.

## Packaging references

Specifications consulted for this candidate on November 12, 2026:

- [Agent Plugins 1.0.0](https://agent-plugins.org/) and its
  [manifest schema](https://agent-plugins.org/schemas/1.0.0/plugin.schema.json)
- [Agent Skills specification](https://agentskills.io/specification)
- [Codex plugin packaging](https://developers.openai.com/plugins/build/plugins)
- [Claude Code plugin manifest reference](https://code.claude.com/docs/en/plugins/manifest-reference)
- [Cursor plugin reference](https://cursor.com/docs/reference/plugins)
- [NVIDIA/skills plugin catalog conventions](https://github.com/NVIDIA/skills/blob/main/plugins.d/README.md)
- [NVIDIA FLARE catalog inventory](https://github.com/NVIDIA/skills/blob/main/components.d/nvflare.yml)

The NVIDIA plugin program's approval and marketplace requirements are separate
from the host schemas. Recheck submission guidance before publication.
