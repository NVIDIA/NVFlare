# Plugin workflow preparation

Prepare synthetic inputs for testing the `nvidia-nvflare` plugin outside its
source checkout. These developer helpers are excluded from the plugin archive.
They do not modify the signed component skills or their evaluation datasets.

## Prepare inputs

Build the plugin with `../package_plugin.py` from the repository root. Use a new
directory for each command; existing output directories are rejected.

```bash
python3 dev_tools/agent/package_plugin.py --output /tmp/flare-plugin-candidate
python3 dev_tools/agent/plugin_validation/prepare.py \
  --plugin /tmp/flare-plugin-candidate/nvidia-nvflare \
  --output /tmp/flare-plugin-fixtures
```

`cases.json` describes eleven workspaces, with prompts and expected component
selection. It is a preparation manifest, **not a SkillEvaluator dataset**.
`source-hashes.json` and `plugin-hashes.json` record the immutable inputs.

| Workspace | Intended workflow evidence |
| --- | --- |
| `convert-pytorch` | Convert a tiny manual training loop; run two CPU sites and export |
| `convert-lightning` | Convert a tiny Lightning trainer; run two CPU sites and export |
| `convert-huggingface` | Convert a locally initialized tiny BERT Trainer; no downloads |
| `statistics` | Run federated statistics and compare with the independent oracle |
| `report-stopped` | Generate report, summary and plot from complete synthetic evidence |
| `report-active` | Report status without stopping or finalizing an active campaign |
| `report-pending` | Reject final reporting while candidate evidence is unfinished |
| `diagnose` | Identify the missing file in a clearly synthetic failure log |
| `orient` | Explain workflow choices and request missing project details |
| `nonflare-negative` | Answer a non-federated profiling request without FLARE conversion |
| `statistics-to-pytorch` | Run statistics and conversion while preserving site boundaries |

The statistics oracle stays outside agent workspaces in `oracles/`. It uses
sample variance and standard deviation. The report scores are explicitly marked
synthetic, not measured improvements. Active status is negative for the report
component but positive for the full plugin. A bounded live Auto-FL optimization
case still needs a validated converted job as its input.

## Qualify the fixtures

The checker requires a POSIX host and an environment with the checked-out
NVFlare version and the frameworks. The dependency set in `requirements.txt`
was exercised on macOS arm64 with Python 3.12.9; it is not a complete dependency
lock or a claim of validation on every platform. Keep Torch and torchvision
versions compatible and satisfy Transformers' Hugging Face Hub constraint.

For a fresh environment, install from the intended source revision and resolve
the fixture requirements together:

```bash
python3.12 -m venv /tmp/flare-plugin-runtime
/tmp/flare-plugin-runtime/bin/python -m pip install \
  . -r dev_tools/agent/plugin_validation/requirements.txt
/tmp/flare-plugin-runtime/bin/python dev_tools/agent/plugin_validation/check.py \
  --prepared /tmp/flare-plugin-fixtures \
  --output /tmp/flare-plugin-checks
```

The checker copies inputs into its output directory and then:

1. Checks the CLI's framework routing and runs each of the three original
   training scripts, requiring finite loss.
2. Runs the extracted plugin's actual report helper against the stopped,
   active and pending fixtures; checks artifacts or the specific refusal.
3. Checks that original fixtures, campaign evidence and plugin files are intact.

Commands have a 120-second limit and owned process groups are cleaned up on
exit or interruption. Hub downloads are disabled. Results and bounded command
logs are retained under the output directory. Inspect `results.json` as well as
the exit status. Running these six checks does not execute the remaining five
workflows, run an agent, perform a conversion, or replace official evaluation.

## Run agent workflows next

Use a separate copy of each workspace per host and attempt. Load the extracted
plugin through the host's native mechanism, with no duplicate installed skills.
Keep inputs, grading oracles and final outputs distinct. For each attempt record
the archive digest, case ID, host/model versions, commands, execution errors,
generated files and actual result. Verify simulator completion and exported job
contents for conversion cases; do not count a generated script alone as success.

Test statistics with the recipe's default privacy controls and compare actual
outputs to the oracle with an explicitly recorded tolerance. For report cases,
verify lifecycle state and evidence as well as text. Once a real converted job
passes, add the bounded Auto-FL candidate/stop scenario.

Agree the supported plugin-level dataset, required agents, comparison arms,
grading and signing contract with the evaluation service before submitting the
formal run. Keep these fixtures separate from component `evals/evals.json`.

## Developer checks

```bash
./runtest.sh -s --skip-install dev_tools/agent/plugin_validation
```

## Packaging CI and catalog preparation

The `Plugin Package` workflow runs a lightweight package check on relevant PRs
and merge-queue entries. It builds twice, checks archive bytes and executable
permissions, validates the portable manifest against a checksum-pinned Agent
Plugins 1.0.0 schema, checks native discovery settings, requires all nine
components and their resource manifests, and runs packaged skill lint and every
bundled Markdown/HTML relative-link check. It retains the candidate and results
as workflow artifacts. It has read-only repository permissions and no publishing,
signing, or provider credentials.

Run the same entrypoint locally with PyYAML and jsonschema installed in a test
environment:

```bash
curl --fail --location --silent --show-error --max-time 30 \
  https://agent-plugins.org/schemas/1.0.0/plugin.schema.json \
  --output /tmp/flare-plugin.schema.json
python3 -m dev_tools.agent.plugin_validation.ci_check \
  --schema /tmp/flare-plugin.schema.json \
  --output /tmp/flare-plugin-ci-checks
```

The workflow uses Python 3.12, PyYAML 6.0.3, and jsonschema 4.25.1. The schema's
expected digest is recorded in `ci_check.py`; a changed download fails the check.
Keep outputs outside the checkout. The local result records whether packaging
files already match the source commit, so an uncommitted candidate cannot be
mistaken for an immutable release revision.

Manual workflow dispatch can additionally enable `security_scan`. That job uses
OSS SkillEvaluator 0.8.0 pinned to its release commit, provisioned scanners, and
all applicable deterministic Tier 1 checks. Findings fail the job; failed reports
are retained. A normal packaging-only pass does not establish security coverage.
Tier 2, agent-based Tier 3, signature trust, and publication remain separate gates.

For the later NVIDIA/skills PR, run the catalog's current `build-plugins.py
--check`, `version-plugins.py --check`, and `verify_content_integrity.py` in the
catalog checkout. Review generated native manifests and marketplace entries as
well as skill hashes: the catalog builder can transform metadata. Preserve the
Plugin Card and ensure the accepted attestation covers the final distributed
content. These source-repository checks do not replace the catalog's review or
official publication workflow.

The preparation and qualification steps require no provider credential. Agent
tests may require host authentication; never include credentials in fixtures,
reports or packaging sources.
