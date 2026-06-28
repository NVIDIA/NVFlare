# Agent Benchmark Harness Architecture

This document defines the intended architecture for the NVFLARE agent benchmark
harness. The harness measures how agent-accessible NVFLARE skills affect applied
conversion and diagnosis tasks. It is benchmark infrastructure, not product API.

The benchmark input is intentionally loose: a job is a folder containing scripts,
data, configuration, and human documentation. The harness must not require a job
schema or perform the agent's conversion work. The harness mounts an explicit
benchmark prompt and execution environment, then measures the agent's behavior
and produced artifacts.

The architecture supports multiple agents through adapters. Codex is the
supported adapter until another agent has a defined installation,
configuration, authentication, invocation, event, usage, and final-message
contract. Unsupported agents fail during preflight.

## System Boundary

The benchmark system has five external inputs:

- a job folder;
- a prompt file or rendered prompt;
- a scenario definition or direct run arguments;
- an agent and model selection;
- a skill-exposure mode.

It produces one result tree containing normalized agent artifacts, records,
workspace-delta manifests, metrics, and reports.

The harness compares skill exposure, not evaluator behavior. The only benchmark
modes are:

```text
without_skills
with_skills
```

There is no evaluator axis, no `eval=on` mode, and no runtime skill evaluator in
the benchmark architecture. Correctness and quality are derived from measured
agent output, generated artifacts, validation evidence, source immutability, and
failure analysis.

## Architecture Overview

```text
CLI / Scenario
    |
    v
Host runner and scenario expander
    |
    | builds/selects images, validates inputs, mounts job/prompt/results
    v
Docker runtime
    |
    v
Container runtime environment
    |
    | provides Python/uv/pip, selected NVFLARE wheel/skill visibility, agent CLI, mounted prompt/job/results, progress/timing capture
    v
Agent adapter
    |
    | starts the specified agent CLI, does not modify the prompt, and normalizes agent-specific events
    v
Artifact, record, and report pipeline
```

The host side owns Docker orchestration and run planning. The container runtime
environment provides the neutral infrastructure for one measured run: Python,
uv/pip, NVFLARE, the selected agent CLI, the explicit prompt, the job mount, the
result mount, selected skill visibility, and bounded observation artifacts. Agent adapters own
agent-specific installation metadata, authentication, configuration, invocation,
and parsing. Measurement semantics live in shared artifact, record, event,
timing, quality-signal, and report modules.

## Repository Layout

The target benchmark harness layout lives under `assist_tools/skills_benchmark/`:

```text
assist_tools/skills_benchmark/
|-- README.md
|-- bin/
|   |-- build.sh
|   `-- run.sh
|-- config/
|   |-- agents/
|   |   |-- codex.yaml
|   |   `-- claude.yaml
|   `-- sdks/
|       `-- nvflare-profile.yaml
|-- docker/
|   |-- Dockerfile
|   `-- build_context.dockerignore
`-- skills/
    `-- harness/
        |-- __init__.py
        |-- common.py
        |-- modes.py
        |-- artifacts.py
        |-- events.py
        |-- quality_signals.py
        |-- scenarios.py
        |-- timing.py
        |-- records.py
        |-- record_identity.py
        |-- host/
        |   |-- build.py
        |   |-- common.py
        |   `-- runner.py
        |-- container/
        |   |-- agent_run.py
        |   |-- progress.py
        |   `-- skills.py
        |-- agents/
        |   |-- base.py
        |   |-- config.py
        |   |-- registry.py
        |   |-- parsers.py
        |   `-- classifiers.py
        |-- sdks/
        |   |-- base.py
        |   |-- config.py
        |   `-- registry.py
        `-- reports/
            |-- scenario_report.py
            `-- structure_tree.py
```

This location keeps the harness outside `nvflare/` product packages while
making it an importable assist-tool package. Unit tests for the harness live
under `tests/unit_test/skills_benchmark/`. Integration tests that validate
Docker execution live under `tests/integration_test/skills_benchmark/`.

`docs/design/agent_benchmark_harness.md` is the architecture document. The
harness-local README explains how to build and run the tool.

The nested `skills/harness/host/`,
`skills/harness/container/`, `skills/harness/agents/`, and
`skills/harness/reports/` packages are the canonical implementation
locations. Flat compatibility modules such as
`skills/harness/host_runner.py` or
`skills/harness/agent_run.py` may exist only as explicit re-export shims
with deprecation comments. They must not own independent logic.
The layout is normative for the target architecture. Modules listed here are
the canonical destinations for the corresponding responsibilities; runnable
scenario YAML support requires `skills/harness/scenarios.py` to compile
the YAML into `run_plan.json` before any Docker execution.
Future supported agents add an agent config file under
`config/agents/`, for example
`config/agents/<agent>.yaml`. They do not add a per-agent adapter
subclass when existing launch, skill-exposure, event-parser, usage-parser,
final-message, and exit-classifier registries cover the agent.

## Component Ownership

### CLI Wrappers

`bin/build.sh` and `bin/run.sh` are thin entry points. They translate shell usage
into Python module invocations and do not own benchmark semantics.

### Host Runner

`harness/host/` owns host-side orchestration:

- direct CLI argument parsing;
- scenario file loading and run-plan expansion;
- preflight validation;
- Docker image selection and build invocation;
- explicit Docker context and mount construction;
- result-root creation;
- sequential execution of run-plan entries;
- host-side report orchestration.

The host runner does not parse agent event streams, infer task success from raw
logs, derive skill identity from workflow names, or mutate job input folders.

### Record Identity

`harness/record_identity.py` owns stable identifiers used in run plans, result
paths, and aggregation keys: slugs, collision handling, prompt hashes, job
hashes, and comparison group IDs. Reports consume these identifiers from
scenario/run-plan metadata rather than reconstructing them from directory names.

### Scenario Engine

`harness/scenarios.py` owns scenario parsing, validation, and expansion. A
scenario expands into a concrete `run_plan.json` before the first Docker run
starts. The run plan is the source of truth for execution order and comparison
grouping.

Every benchmark entry point compiles into this same run-plan model. Direct
commands create a single-case scenario and run plan; they do not bypass
scenario validation, result identity, record synthesis, or report aggregation.
Scenario YAML files are runnable inputs only when they are consumed by
`harness/scenarios.py` and produce `run_plan.json`. Scenario examples or YAMLs
without that compiled run plan are design fixtures, not an alternate execution
path.

Preflight validation covers:

- supported agent names;
- unambiguous agent/model selection;
- valid comparison type and mode names;
- job paths that exist and are directories;
- explicit `job_scale` for every scenario job;
- prompt path existence or renderability;
- Docker image availability or build inputs;
- explicit Docker context allowlisting;
- estimated run count and result-tree size.

Execution is sequential by default. Parallelism is a separate scenario field and
must be explicit because concurrent agent runs affect timing, resource
contention, and result interpretation.

Parallelism applies to independent run-plan entries only. A single run-plan
entry owns one container, one result directory, and one runtime staging root. No
two concurrent entries may share `/tmp/agent_benchmark`, a workspace copy, a
record directory, or an agent auth/config write location.
Isolation is container-boundary isolation. A fixed path such as `/tmp/agent_benchmark`
is safe only because each benchmark run gets its own container; if a container
is reused for multiple concurrent run-plan entries, each entry must receive a
unique runtime artifact root.

### Docker Build And Runtime

The Docker layer provides isolated, repeatable execution. The local SDK
checkout is build input only. Runtime images receive the selected SDK through
built wheels and explicit metadata, not by copying the working tree into the
image.

Image identity is resolved from `(sdk profile, agent profile, variant)`. The
current default SDK profile is `nvflare-profile` and the default agent profile
is `codex`, so the default image names remain:

```text
agent-skills-benchmark:<agent>-baseline
agent-skills-benchmark:<agent>-skills
```

SDK-specific build behavior is pluggable through `harness/sdks/`. An SDK
profile declares:

```text
source.type = repo with source.path and markers, or wheels with exact wheel paths
build.type = uv_wheel for repo builds, or provided_wheels for prebuilt wheels
skills and baseline wheel variants
optional build-env toggle used while building each wheel
skills.setup.type = command, copy, or none
Docker build args for package name, import name, version check, and skills setup metadata
metadata files emitted by the SDK setup step
```

For NVFlare, `config/sdks/nvflare-profile.yaml` declares the existing wheel split:

```text
skills wheel     NVFLARE_PACKAGE_AGENT_SKILLS=1
baseline wheel   NVFLARE_PACKAGE_AGENT_SKILLS=0
```

Those wheels are reused across agent images. Agent profiles own agent CLI
installation, version commands, auth-home defaults, optional native
dependencies, and CLI availability probes. The shared builder resolves the
selected `(sdk, agent, variant)` image inputs and passes SDK build arguments
from the SDK adapter plus agent build arguments from the agent adapter.

The Docker convention is one shared Dockerfile with generic `base`, `skills`,
and `baseline` stages. The Dockerfile must not branch on agent names or encode
Codex-specific assumptions such as `CODEX_HOME`, Codex auth file names, or
`codex exec`. It must also not hardcode NVFlare-specific wheel names or skill
install commands. The shared stages expose the configured SDK, Python, the
benchmark harness, prompt mounts, job mounts, result mounts, and generic
benchmark environment. Agent profiles provide commands such as
`AGENT_INSTALL_COMMAND` and `AGENT_VERSION_COMMAND` to add the selected agent
runtime.

### Container Runtime Environment

`harness/container/agent_run.py` is the container-side runtime setup and
observation wrapper for one measured run:

- validate mounted input, prompt, and result directories;
- provide the configured Python, uv/pip, NVFLARE, and agent CLI environment;
- provide a container-local workspace copy from the mounted job folder;
- apply the selected packaged-skill visibility mode;
- copy the prompt verbatim and write prompt metadata into the record directory;
- establish workspace baselines for delta capture;
- hand the prompt and workspace to the selected agent adapter;
- preserve agent stdout, stderr, events, and final message;
- synthesize fallback records on failure;
- run artifact capture and report commands;
- write final run status.

The container runtime environment does not instruct the agent, construct
agent-specific commands, or parse raw agent events. It also does not own report
meaning, quality-gate policy, skill identity trust decisions, or timing
semantics beyond calling the timing module at defined measurement boundaries.
Workflow names, mode names, record paths, result paths, and metric expectations
remain harness metadata unless they are present in the explicit benchmark
prompt selected by the scenario.

### Timing Boundaries

The container runtime environment records timing marks. Adapters do not decide
when a benchmark timer starts or stops.

The headline comparison metric is `agent_elapsed_seconds`. It starts
immediately before the adapter launches the agent process described by
`AgentLaunchSpec`. It stops when that process exits, or when launch fails and
the runtime environment records the launch failure. It excludes Docker image build,
container startup, mounted-input validation, prompt staging, skill exposure,
workspace baseline capture, artifact capture, record synthesis, and report
generation.

The run summary also records phase timings:

```text
container_elapsed_seconds
setup_elapsed_seconds
skill_exposure_elapsed_seconds
agent_elapsed_seconds
post_process_elapsed_seconds
report_elapsed_seconds
```

Skill exposure, agent CLI installation, auth mounting, and configuration setup
are environment preparation. They are recorded separately and are not eligible
for the cost/performance winner metric unless a scenario explicitly declares a
setup-cost experiment.

### Agent Adapter Registry

The harness has a single adapter registry. The registry is the only place that
maps an agent name to agent-specific behavior. Adding a supported agent requires
registering one agent config and one install surface; it must not require
changes to record schemas, report code, mode definitions, quality gates, metric
extraction, workspace-delta capture, timing, or scenario comparison semantics.

Registration validates the complete adapter contract before an agent can be
used. A supported adapter must provide all required launch, skill-exposure,
event, usage, final-message, metadata, and failure-classification behavior at
registration time; missing methods or unsupported capabilities fail preflight,
not an in-progress benchmark run.
The adapter base must be an abstract base class with explicit abstract methods.
Registration may add additional semantic validation, but structural typing alone
is not the architecture contract.

The registry supports three categories:

| Category | Meaning |
| --- | --- |
| `supported` | The adapter has implementation, Docker image support, contract tests, and can run benchmarks. |
| `known_pending` | The agent is named in design or docs, but intentionally fails preflight with a clear unsupported message. |
| `unknown` | The harness rejects the name as invalid input. |

This keeps future names such as `hermes` or `openclaw` visible without allowing
partially implemented agents to run.

The initial registry state is `codex` and `claude` as `supported`; names such as
`hermes` or `openclaw` may be registered as `known_pending` only when the design
and user-facing error message intentionally identify them as planned but not
runnable.

### Agent Adapters

An agent adapter owns only the agent-specific mechanics required to start and
parse one agent surface. It translates the specified agent name and model into
provider-specific CLI, auth, environment, launch, and parsing details. It must
not add prompt text, task hints, benchmark expectations, metric names, record
paths, workflow instructions, or other context beyond the explicit benchmark
prompt selected by the scenario.

If an adapter cannot provide a stable, benchmark-owned default model, it must
require an explicit model from direct CLI `--model` or scenario configuration.
A supported adapter must not let the underlying agent CLI select an implicit
default model for measured runs.

```python
class AgentAdapter:
    name: str
    display_name: str
    default_model: str
    agent_home_env: str
    container_home: str

    def model_from_env(self, env: Mapping[str, str]) -> str: ...
    def build_args(self) -> dict[str, str]: ...
    def image_targets(self) -> AgentImageTargets: ...
    def auth_mounts(self, host_config) -> list[DockerMount]: ...
    def runtime_env(self, config) -> dict[str, str]: ...
    def launch_spec(self, config) -> AgentLaunchSpec: ...
    def skill_exposure(self, config) -> SkillExposureSpec: ...
    def availability_probe(self) -> list[str]: ...
    def normalize_event(self, raw_line: str) -> dict | None: ...
    def parse_usage(self, events_path: Path) -> dict: ...
    def parse_activity(self, events_path: Path) -> dict: ...
    def final_message_source(self, result_dir: Path) -> FinalMessageSource: ...
    def metadata(self) -> dict: ...
    def exit_summary(self, exit_code: int, stderr_path: Path) -> dict: ...
```

The adapter contract covers:

- CLI installation and version metadata;
- host and container auth/config locations;
- runtime environment variables;
- model selection;
- launch working directory, delivery of the explicit prompt without modification,
  stdout/stderr/event stream routing, shell/login behavior, approval and
  sandbox flags, and launch-failure handling;
- skill-exposure mechanism, setup action, probe action, and metadata locations;
- final assistant message capture;
- raw event normalization;
- token and activity extraction;
- CLI availability checks;
- exit-code interpretation;
- agent-specific parser warnings.

`AgentLaunchSpec` has six required fields with no defaults:

| Field | Meaning |
| --- | --- |
| `argv` | Agent CLI argv excluding shell interpolation. |
| `cwd` | Working directory for the measured agent process. |
| `prompt_file` | Path to the already-rendered benchmark prompt. |
| `prompt_input_mode` | How exact prompt bytes are delivered: `stdin` or `file_arg`. |
| `stdout_events_dest` | File path where stdout or structured event stream is captured. |
| `stderr_dest` | File path where stderr is captured. |

Optional fields have explicit defaults:

| Field | Default |
| --- | --- |
| `final_message_dest` | `none` |
| `environment` | `{}` |
| `login_shell` | `false` |
| `approval_flags` | `[]` |
| `sandbox_flags` | `[]` |
| `bypass_reason` | `none` |
| `launch_timeout` | job-scale policy |
| `extra_artifact_paths` | `[]` |

The container runtime needs the required fields to start and observe the
process.
Optional fields describe agent-specific launch behavior, but missing optional
fields must not prevent a supported adapter from running.
`sandbox_flags` and `approval_flags` are audit metadata recorded in
`launch_spec_metadata.json`; any flag that must affect the measured agent
process must also be present in the adapter's `argv` template. `bypass_reason`
must explain why these flags are acceptable inside the isolated benchmark
container.

`prompt_file` is the path to the already-rendered benchmark prompt. If
`prompt_input_mode` is `stdin`, the runtime streams the exact file bytes to the
agent process. If it is `file_arg`, the adapter argv may reference
`{prompt_file}`. Adapter configs must not use `{prompt_text}` or any other
template that injects prompt content into argv, environment variables, or
adapter-owned files.

If an argv template references `{final_message_dest}`, the renderer must provide
a concrete path and set `final_message_dest` to that same path. If no concrete
final-message path is available, rendering fails preflight or the template must
omit the flag pair. It must not substitute the literal value `none`.
`bypass_reason` is required whenever `approval_flags` or `sandbox_flags` disable
interactive approval or sandbox checks; it is recorded in runtime metadata.

`SkillExposureSpec` is mechanism-typed rather than command-shaped:

```text
mechanism_type = cli_install | launch_flag | directory_mount | config_file | preinstalled_home | none
skill_root
source_paths
setup_action
probe_action
disable_action
launch_args
environment
metadata_files
expected_post_setup_state
```

Only fields that make sense for the mechanism are populated. For example, a
Codex-style adapter can use `cli_install`, while an agent that exposes skills
through an added directory can use `launch_flag` or `directory_mount`. The
container runtime requests the adapter's skill exposure spec and records the
returned `SkillExposureSpec`; it does not branch on agent names or
provider-specific paths.

`harness/container/skills.py` is the only component that executes
`SkillExposureSpec` actions. It validates scoped paths, applies `setup_action`
for `with_skills`, applies `disable_action` for `without_skills`, runs
`probe_action` when present, captures declared `metadata_files`, and writes
`SkillExposureResult`. The adapter returns data; it does not execute the setup,
probe, or disable action itself.

`SkillExposureResult` is written by the container runtime after applying the
spec:

```text
status = prepared | disabled | skipped | failed
mechanism_type
installed_paths
disabled_paths
probe_status
probe_output_ref
metadata_files
parser_warnings
```

In `with_skills` mode, the container runtime applies the spec's setup action
unless the mechanism is already prepared. In `without_skills` mode, the
container runtime applies the spec's disable action. If the no-skills image or launch path
already guarantees no packaged skills are visible, the adapter returns
`mechanism_type: none`; the container runtime records `status: skipped` and
performs no file operation.
Setup, probe, and disable actions are scoped to benchmark-owned container paths
such as the benchmark agent home, skill root, workspace config path, or result
directory. They must not operate on the host user's real agent home, global
skill catalog, or unrelated agent configuration. Broad operations such as
`--all` are allowed only when the target root is explicitly the benchmark-owned
agent home or skill root.

`FinalMessageSource` is also typed:

```text
source_type = file | structured_event | stdout_tail | not_available
path
event_selector
tail_bytes
parser_warnings
```

The container runtime always materializes `agent_last_message.txt` as the
normalized artifact. Adapters that cannot ask the CLI to write a final-message
file must extract the final message from structured events or bounded stdout
and record parser warnings when fidelity is lower.

Benchmark launches are non-interactive. If an agent normally asks for approval,
uses an interactive sandbox, or blocks tool execution by default, its adapter
must declare the explicit CLI flags or configuration that make benchmark runs
fully automated inside the isolated container. The adapter metadata must record
those flags and why they are safe in the benchmark context. The shared
runtime layer must not hard-code a provider-specific bypass flag.

The agent process receives a sanitized environment assembled from the adapter's
runtime requirements. Benchmark control variables such as mode, result paths,
record paths, and skill-exposure bookkeeping remain harness-internal state and
are not exposed as task instructions. If a scenario intentionally
exposes a benchmark variable to the agent, that exposure is part of the
experimental condition and must be recorded in prompt or run metadata.

The run plan selects whether the measured run is `with_skills` or
`without_skills`. The container runtime applies that already-selected
visibility mode before the measured agent process starts. The adapter owns the
mechanics for the selected agent: where skills live, how they are installed or
hidden, how availability is probed, and where skill metadata is recorded.
`container/skills.py` applies the adapter-provided spec and records the
outcome; adapters must not directly remove, overwrite, or publish benchmark
skill files as a side effect of constructing the spec.

Adapters may expose observations such as model name, CLI version, raw usage,
raw activity, tool-call summaries, and agent-reported skill identity. They must
not decide benchmark meaning. Timing boundaries, workspace artifacts, process
records, report filters, source immutability policy, pass/fail normalization,
metric validity, skill-identity trust decisions, and failure-root
classification belong to the harness runtime, record layer, and report layer.

Direct CLI commands select agents and models with `--agent` and `--model`.
Scenario YAML selects them with `agents[].name` and `agents[].models`.
After run-plan expansion, shared host and container modules use generic
benchmark runtime names:

```text
BENCHMARK_AGENT
BENCHMARK_AGENT_MODEL
BENCHMARK_AGENT_HOME
BENCHMARK_AGENT_CONFIG_DIR
```

Agent-specific variables such as `CODEX_HOME` or `CLAUDE_CONFIG_DIR` are
translated by the adapter at the host boundary. They
must not become required inputs for shared records, reports, modes, or quality
checks. Shared host, container, record, and report code must not branch on
provider-specific names except through the adapter registry.

Event normalization is agent-specific but emits an agent-neutral event schema.
The container runtime calls the selected adapter directly:

```python
adapter.normalize_event(raw_line: str) -> dict | None
```

If a module-level helper exists, it is only a thin registry wrapper around the
selected adapter method, not a second dispatch model.

`harness/events.py` owns neutral event helpers and common counters.
`harness/agents/config.py`, `harness/agents/parsers.py`, and
`harness/agents/classifiers.py` own raw event normalization, usage parsing,
activity parsing, final-message discovery, exit classification, and CLI
metadata selected by each agent config.

If an agent exposes lower-fidelity output than Codex, the adapter still writes
the neutral artifacts and records parser warnings. Missing token usage, missing
structured tool-call events, or an approximate final-message source are
recorded as limitations; they do not change the benchmark schema.

### Neutral Event And Usage Contracts

Adapters normalize raw agent output into the shared `agent_events.jsonl`,
`agent_usage.json`, `agent_activity.json`, `agent_last_message.txt`, and
`agent_stderr.txt` artifacts.

Normalized shell and tool events use agent-neutral fields:

```text
schema_version
event_type
agent
timestamp
tool_kind
command_text
command_argv
cwd
status
exit_code
started_at
ended_at
output_ref
truncated
redacted
parser_warnings
```

The adapter may omit fields that the agent CLI does not expose, but omitted or
approximated fields must be declared in `parser_warnings`. Reports consume this
neutral event schema rather than provider-specific envelopes such as Codex item
types.

Usage artifacts declare their semantics:

```text
schema_version
source
usage_fidelity
is_cumulative
input_tokens
output_tokens
reasoning_tokens
cache_tokens
tool_tokens
total_tokens
cost
parser_warnings
```

The shared harness may aggregate normalized usage fields. It must not scrape
arbitrary provider raw JSON as the primary usage contract.

Allowed `usage_fidelity` values are:

| Fidelity | Meaning |
| --- | --- |
| `exact` | Reported directly by the agent CLI or provider metadata with documented semantics. |
| `parsed` | Derived from structured agent events with stable parser logic. |
| `approximate` | Estimated from partial, lossy, or text-derived evidence. |
| `unavailable` | The adapter could not provide the field. |

Reports may compare elapsed time across all completed runs because timing
boundaries are harness-defined. Token and cost comparisons require comparable usage sources:
`exact` may be compared with `exact`, and `parsed` may be compared with
`parsed` from the same adapter parser version. Mixed-fidelity comparisons remain
visible in tables but cannot be used as winner-policy tiebreakers unless the
scenario explicitly opts into approximate cost comparison.

### Failure Classification Contract

Adapters normalize provider-specific failures into shared categories:

```text
agent_cli_missing
agent_auth_failure
agent_model_unsupported
agent_rate_limited
agent_context_limit
agent_sandbox_or_approval_failure
agent_internal_error
agent_unknown_failure
harness_preflight_failure
harness_execution_skipped
harness_execution_error
```

The report layer renders these categories and combines them with shared
benchmark evidence. It must not hard-code provider-specific diagnoses such as
Codex model-selection failures outside the adapter layer.
The harness runtime, not the adapter, writes `harness_preflight_failure`,
`harness_execution_skipped`, and `harness_execution_error` when the failure is
clearly caused by run-plan validation, skipped execution policy, Docker/runtime
or artifact/report pipeline failure rather than the agent CLI.

### Shared Benchmark Core

The shared benchmark core is agent-neutral. It owns:

- prompt staging;
- job-folder mounting;
- skill-exposure modes;
- workspace baseline and delta capture;
- source-input immutability checks;
- lifecycle timing;
- normalized record synthesis;
- metric extraction and quality signals;
- failure analysis;
- report rendering;
- scenario expansion and comparison grouping.

The shared core must not know how an agent authenticates, where its home
directory is located, which CLI flags select a model, how its raw events are
shaped, or how it writes a final assistant response. Those are adapter
responsibilities.

Adding a new supported agent should be localized to:

- `config/agents/<agent>.yaml`;
- the agent install descriptor in that config;
- adapter registry configuration;
- auth/config README entries;
- sample event fixtures and adapter contract tests.

If the new agent requires a parser or classifier that does not exist, add that
implementation to the shared parser/classifier registry and reference its ID
from the YAML config. Do not add a per-agent adapter subclass.

If adding an agent requires editing reports, records, modes, metric extraction,
or failure-analysis logic, the adapter boundary is too weak.

### Codex Agent Config

The Codex agent config defines:

- CLI command: `codex exec --json ...`;
- model input: direct-run `--model` or scenario `agents[].models`;
- auth/config mounts: `CODEX_HOME`, host `.codex/auth.json`, and
  `.codex/config.toml`;
- JSONL event normalization;
- cumulative token usage parsing;
- final message path for `--output-last-message`.

### Non-Codex Agent Configs

A non-Codex agent config is supported only when these contracts are known:

- installation and version pinning;
- auth and config mount locations;
- model selection;
- structured event availability;
- final assistant response source;
- token usage source;
- tool-call and shell-command representation;
- exit-code semantics.

If an agent does not expose Codex-like structured events, its adapter still
emits the neutral event contract with lower-fidelity activity fields and parser
warnings in `agent_usage.json` or `agent_activity.json`.

Non-Codex adapters must not introduce new benchmark modes or report sections.
For example, Claude, Hermes, or OpenClaw compare against Codex through the same
`without_skills` and `with_skills` modes, the same result schema, the same
quality gates, and the same reports. Agent-specific sections are allowed only
for raw metadata that does not change benchmark interpretation.

### Agent Adapter Examples

Most adapter behavior is data: CLI argv templates, auth paths,
skill-exposure mechanism, event parser ID, usage parser ID, final-message
source, and exit-classifier ID. `ConfigurableAgentAdapter` is the concrete
adapter class. It reads an agent config file and implements `AgentAdapter`
methods from that config. Adding an agent is config-only when existing
mechanism and parser registry entries cover that agent.

Module ownership:

- `harness/agents/base.py` owns the `AgentAdapter` abstract base class and typed
  specs such as `AgentLaunchSpec`, `SkillExposureSpec`,
  `SkillExposureResult`, and `FinalMessageSource`.
- `harness/agents/config.py` owns `AgentConfig`, YAML validation, template
  rendering, config-driven image/auth/runtime helpers, and
  `ConfigurableAgentAdapter`.
- `harness/agents/registry.py` owns supported/known-pending/unknown agent
  registration and maps an agent name to a loaded `ConfigurableAgentAdapter`.
- `harness/agents/parsers.py` owns parser registries and helpers such as
  `parse_usage_from_events`, `parse_activity_from_events`, and
  `normalize_event_with_parser`.
- `harness/agents/classifiers.py` owns exit-code and failure-classifier
  registries, including `classify_exit`.

Neutral event schema utilities remain in `harness/events.py`.
If a new agent exposes a raw event format that no existing parser supports, the
new code belongs in the shared parser registry under a named parser ID, not in a
per-agent adapter subclass.

#### Generic Configurable Adapter

```python
class ConfigurableAgentAdapter(AgentAdapter):
    """Implements adapter methods that can be safely driven by YAML config.

    Agents are registered by loading YAML configs. Provider-specific parsers and
    classifiers are selected by ID from registries.
    """

    def __init__(self, config_path: Path) -> None:
        self._cfg = AgentConfig.load(config_path)

    @property
    def name(self) -> str: return self._cfg.name
    @property
    def display_name(self) -> str: return self._cfg.display_name
    @property
    def default_model(self) -> str: return self._cfg.default_model
    @property
    def agent_home_env(self) -> str: return self._cfg.agent_home_env
    @property
    def container_home(self) -> str: return self._cfg.container_home

    def launch_spec(self, config) -> AgentLaunchSpec:
        argv = self._cfg.launch.render_argv(config)
        return AgentLaunchSpec(
            argv=argv,
            cwd=config.workspace_dir,
            prompt_file=config.prompt_file,
            prompt_input_mode=self._cfg.launch.prompt_input_mode,
            stdout_events_dest=config.events_dest,
            stderr_dest=config.stderr_dest,
            launch_timeout=config.timeout_seconds,
            bypass_reason=self._cfg.launch.bypass_reason,
        )

    def skill_exposure(self, config) -> SkillExposureSpec:
        return self._cfg.skill_exposure.render(config)

    def final_message_source(self, result_dir: Path) -> FinalMessageSource:
        return self._cfg.final_message.render(result_dir)

    def parse_usage(self, events_path: Path) -> dict:
        return parse_usage_from_events(events_path, self._cfg.usage)

    def parse_activity(self, events_path: Path) -> dict:
        return parse_activity_from_events(events_path, self._cfg.activity)

    def normalize_event(self, raw_line: str) -> dict | None:
        return normalize_event_with_parser(raw_line, self._cfg.events.parser)

    def exit_summary(self, exit_code: int, stderr_path: Path) -> dict:
        return classify_exit(exit_code, stderr_path, self._cfg.exit.classifier)

    def metadata(self) -> dict:
        return {"agent": self.name, "config": str(self._cfg.source_path)}
```

This sketch omits straightforward accessors such as `model_from_env`,
`auth_mounts`, `runtime_env`, image target resolution, and availability probes;
those are still part of the formal `AgentAdapter` contract.

#### Agent Config Files

Each agent ships one YAML config file. Mechanical differences between agents
live here. Provider-specific event, usage, activity, final-message, and
exit-code behavior is selected through named parser or classifier IDs in the
config.

```yaml
# config/agents/codex.yaml
name: codex
display_name: OpenAI Codex CLI
default_model: o3
agent_home_env: CODEX_HOME
container_home: /workspace/.codex

launch:
  prompt_input_mode: stdin
  argv: ["codex", "exec", "--json",
         "--output-last-message", "{final_message_dest}",
         "--dangerously-bypass-approvals-and-sandbox",
         "--model", "{model}"]
  bypass_reason: "isolated container; no persistent side effects outside workspace"

skill_exposure:
  mechanism_type: cli_install
  setup_action: ["codex", "skills", "install", "--path", "{skills_dir}"]
  probe_action:  ["codex", "skills", "list"]
  # Scoped to the benchmark-owned agent home, never the host user's skill state.
  disable_action: ["codex", "skills", "uninstall", "--root", "{container_home}", "--all"]
  metadata_files: ["~/.codex/skills.json"]

final_message:
  source_type: file
  path: "{result_dir}/agent_last_message.txt"

events:
  parser: codex_jsonl
  format: jsonl

usage:
  parser: codex_cumulative_usage
  fidelity: parsed
  is_cumulative: true

activity:
  parser: codex_jsonl_activity

exit:
  classifier: stderr_patterns
  rules:
    - category: agent_cli_missing
      exit_codes: [127]
    - category: agent_auth_failure
      any: [auth, "api key", login]
```

```yaml
# config/agents/claude.yaml
name: claude
display_name: Anthropic Claude Code CLI
default_model: unspecified_default
requires_explicit_model: true
agent_home_env: CLAUDE_CONFIG_DIR
container_home: /workspace/.claude

build:
  args:
    BENCHMARK_DOCKER_AGENT: claude
    BENCHMARK_AGENT_HOME: /workspace/.claude
    AGENT_CLI_NAME: claude
    AGENT_INSTALL_COMMAND: npm install -g "@anthropic-ai/claude-code@latest"
    AGENT_VERSION_COMMAND: claude --version

runtime_env:
  CLAUDE_CONFIG_DIR: "{container_home}"

launch:
  prompt_input_mode: stdin
  argv: ["claude", "--dangerously-skip-permissions",
         "--output-format", "stream-json",
         "--verbose", "--model", "{model}", "--print"]
  sandbox_flags: ["--dangerously-skip-permissions"]
  bypass_reason: "isolated benchmark container; writable state is limited to mounted result/workspace paths"

skill_exposure:
  mechanism_type: launch_flag
  skill_root: "{container_home}/skills"
  launch_args: ["--add-dir", "{skills_dir}"]

final_message:
  source_type: structured_event
  event_selector: {type: result, subtype: success}
  parser: generic_structured_event_message
  parser_warnings:
    - "final message from last result event; may be truncated at context limit"

events:
  parser: claude_stream_json
  format: stream-json

usage:
  parser: claude_stream_usage
  fidelity: parsed
  is_cumulative: false

activity:
  parser: claude_stream_activity

exit:
  classifier: stderr_patterns
  rules:
    - category: agent_cli_missing
      exit_codes: [127]
    - category: agent_sandbox_or_approval_failure
      any: [permission, approval]
```

```yaml
# config/agents/hermes.yaml  (illustrative)
name: hermes
display_name: Hermes Agent CLI
default_model: hermes-3
agent_home_env: HERMES_HOME
container_home: /workspace/.hermes

launch:
  prompt_input_mode: file_arg
  argv: ["hermes", "run", "--no-confirm",
         "--model", "{model}", "--prompt-file", "{prompt_file}"]
  bypass_reason: "--no-confirm disables interactive approval in headless runs"

skill_exposure:
  mechanism_type: config_file
  setup_action:   ["hermes", "skills", "write-config",
                   "--skills-dir", "{skills_dir}", "--output", "{config_path}"]
  disable_action: ["hermes", "skills", "clear-config", "--config", "{config_path}"]
  probe_action:   ["hermes", "skills", "verify",       "--config", "{config_path}"]
  config_path: "{workspace_dir}/.hermes/hermes.yaml"
  metadata_files: ["{config_path}"]

final_message:
  source_type: stdout_tail
  tail_bytes: 8192
  parser_warnings:
    - "Hermes does not emit a structured final-message event; stdout tail used"

events:
  parser: generic_partial_json
  format: partial-json

usage:
  parser: unavailable
  fidelity: unavailable
  parser_warnings:
    - "Hermes CLI does not report token usage"

activity:
  parser: generic_command_activity

exit:
  classifier: generic_cli
```

```yaml
# config/agents/openclaw.yaml  (illustrative)
name: openclaw
display_name: OpenClaw Agent
default_model: claw-2
agent_home_env: OPENCLAW_HOME
container_home: /workspace/.openclaw

launch:
  prompt_input_mode: file_arg
  argv: ["openclaw", "exec", "--batch",
         "--model", "{model}", "--input", "{prompt_file}"]
  bypass_reason: "--batch enables non-interactive execution"

skill_exposure:
  # skills baked into the image; mode difference is entirely in image selection
  mechanism_type: preinstalled_home
  skill_root: /workspace/.openclaw/skills
  probe_action: ["openclaw", "skills", "list"]
  metadata_files: ["/workspace/.openclaw/skills/manifest.json"]

final_message:
  source_type: file
  path: "{result_dir}/openclaw_response.txt"

events:
  parser: generic_jsonl
  format: jsonl

usage:
  parser: openclaw_exact_usage
  fidelity: exact
  is_cumulative: true

activity:
  parser: generic_jsonl_activity

exit:
  classifier: generic_cli
```

#### Parser And Classifier Registries

Agent configs select parser and classifier behavior by ID. The registry maps
IDs such as `codex_jsonl`, `claude_stream_json`, `generic_jsonl`, or
`generic_cli` to shared parser functions. Adding a new agent should not create a
new adapter subclass. If the agent needs a new raw event parser, usage parser,
activity parser, final-message extractor, or exit classifier, that code is
added to the shared registry under a named ID and then referenced from the YAML
config.
Event, usage, activity, and final-message parser IDs are registered in
`harness/agents/parsers.py`. Exit and failure classifier IDs are registered in
`harness/agents/classifiers.py`. `harness/agents/registry.py` validates at
preflight that every parser or classifier ID referenced by an agent YAML exists.

#### Key Differences Across Agent Configs

| Adapter | `mechanism_type` | bypass flag | events format | `usage_fidelity` | final message |
| --- | --- | --- | --- | --- | --- |
| Codex | `cli_install` | `--dangerously-bypass-approvals-and-sandbox` | `jsonl` | `parsed` | `file` |
| Claude | `launch_flag` | `--dangerously-skip-permissions` | `stream-json` | `parsed` | `structured_event` |
| Hermes | `config_file` | `--no-confirm` | `partial-json` | `unavailable` | `stdout_tail` |
| OpenClaw | `preinstalled_home` | `--batch` | `jsonl` | `exact` | `file` |

For `preinstalled_home` agents such as OpenClaw, the `with_skills` vs
`without_skills` difference is entirely in the Docker image layer. The
container runtime still calls `skill_exposure()` to record the spec;
`setup_action` and `disable_action` are absent and the container runtime skips
execution.

The design vocabulary includes `cli_install`, `launch_flag`,
`directory_mount`, `config_file`, `preinstalled_home`, and `none`. The current
runtime implementation supports `launch_flag`, `preinstalled_home`, and `none`.
Adapter configs that use other mechanism types fail validation until the
corresponding container-side execution logic is implemented.

### Artifact Layer

`harness/artifacts.py` owns bounded artifact capture:

- workspace baseline manifests;
- post-run workspace delta manifests;
- source-input immutability checks;
- generated-file structure summaries;
- safe references to large or sensitive artifacts.

The artifact layer records what changed. It does not decide whether a change is
scientifically correct or whether the agent chose the right workflow.

The protected input surface is the original job folder mounted into the
container. It is mounted read-only. The agent works in a container-local
workspace copy. Before the agent starts, the artifact layer captures hash
manifests for both the read-only input mount and the workspace copy. After the
agent exits, it captures the same manifests again and computes:

```text
input_delta_manifest.json
workspace_delta_manifest.json
```

The manifest comparison is content based: stable relative path, file type, size,
and SHA-256 for regular files. Symlinks are recorded but not followed. Large
files may be represented by bounded metadata plus an explicit truncation flag
when hashing would exceed configured artifact limits.

`source_input_modified` is true only when the original input mount's manifest
changes or the harness detects an attempted write against the protected input
surface. Generated files in the workspace copy are expected and are reported in
`workspace_delta_manifest.json`; they do not by themselves mean the original
source input was modified.

### Record Layer

`harness/records.py` owns normalized records. Records combine:

- run identity;
- agent identity;
- mode and skill exposure;
- Docker image and wheel metadata;
- timing;
- token and activity counters;
- final agent message references;
- process metrics;
- validation metrics extracted from generated output;
- quality-signal observations;
- failure-root classification.

Records are generated from observed artifacts. They do not depend on a runtime
evaluator record, evaluator pass/fail, or evaluator score.

### Report Layer

`harness/reports/` owns report generation. Reports consume normalized records,
run summaries, scenario metadata, and artifact manifests. They do not parse raw
agent logs when a normalized source exists.

Reports show:

- scenario and comparison identity;
- agent, model, mode, image, and wheel variant;
- human-readable run status;
- failure analysis and likely root cause for failed runs;
- scalar validation metrics when extractable;
- source immutability and structure checks;
- token, command, timing, and cost-related measurements;
- comparison summaries across modes, agents, models, workflows, and jobs.

Metric sections should be named by metric family, for example
`Metrics (AUROC)` or `Metrics (valid_loss)`. Plot legends should identify the
compared run leg rather than repeating the metric name in every bar label.
If structure-tree analysis is unavailable for a run, reports and summaries
should include `structure_quality_signal` with an explicit unavailable/null
state rather than omitting the field.

## Mode Model

Modes describe skill exposure only:

```text
without_skills
with_skills
```

Agent selection is orthogonal:

```text
agent = codex | claude | ...
mode = without_skills | with_skills
job = /path/to/job-folder
```

This supports comparisons such as:

```text
codex / without_skills
codex / with_skills
claude / without_skills
claude / with_skills
```

The architecture has no `with_skills_eval_on`, `with_skills_eval_off`,
`PROCESS_EVAL`, `NVFLARE_SKILL_EVAL`, or skill-evaluator mode.

`with_skills` means packaged NVFLARE agent skills are made available through the
selected agent's supported skill or instruction mechanism. `without_skills`
means those packaged NVFLARE skills are absent. The mode meaning is shared
across agents; the adapter only supplies the mechanics needed to expose or hide
the same NVFLARE skill content for that agent.

Skill installation, agent CLI installation, authentication setup, and agent
configuration setup are environment preparation. They happen before the measured
agent process starts and are recorded as runtime metadata, not as agent task
work.

## Scenario Model

A scenario defines a matrix across these axes:

| Axis | Meaning |
| --- | --- |
| `agent` | Agent surface such as `codex` or `claude` |
| `agent_model` | Model name within the selected agent |
| `workflow` | Requested NVFLARE workflow such as FedAvg or SCAFFOLD |
| `comparison` | Explicit comparison object |
| `job` | Unstructured input job folder |
| `job_scale` | Scenario-provided scale annotation: `small`, `medium`, `large` |
Important boundaries:

- `agent` and `agent_model` are separate axes.
- `workflow` is separate from the job folder.
- `workflow` does not imply framework or skill package.
- `workflow` is scenario metadata and comparison identity; it is not a hidden
  instruction. If the agent should perform a specific workflow, that request
  must appear in the explicit benchmark prompt.
- `job` remains an unstructured folder.
- `job_scale` controls timeout and resource policy; it is not inferred by
  default.
- `comparison` is explicit and must not be overloaded by shorthand names.

Scenario validation rejects job entries without an explicit `job_scale`. Direct
CLI runs must also produce a run-plan entry with an explicit `job_scale`, either
from a CLI argument or a named direct-run default recorded in the run plan. The
harness must not infer scale from file count, data size, README text, or runtime
duration.

Default resource policy:

| `job_scale` | `agent_elapsed_seconds` timeout | Container timeout | Result-size budget |
| --- | ---: | ---: | ---: |
| `small` | 30 minutes | 40 minutes | 1 GB |
| `medium` | 90 minutes | 120 minutes | 5 GB |
| `large` | 240 minutes | 300 minutes | 20 GB |

Scenarios may override these values explicitly. CPU, memory, and GPU limits are
scenario fields, not inferred from `job_scale`, because agent CLIs and training
dependencies have different resource profiles.

Comparison examples:

```yaml
comparison:
  type: mode_ablation
  modes: [without_skills, with_skills]
```

```yaml
comparison:
  type: agent_comparison
  mode: with_skills
  agents: [codex, claude]
```

```yaml
comparison:
  type: model_comparison
  agent: codex
  mode: with_skills
  models: ["<codex-model-a>", "<codex-model-b>"]
```

```yaml
comparison:
  type: one
  mode: with_skills
```

Expansion rules:

- `mode_ablation` and `one` create one comparison group per
  `(agent, model, workflow, job)` combination.
- `agent_comparison` varies the agent axis. Each compared agent resolves to one
  model. Ambiguous model selection is a validation error.
- `model_comparison` varies the model axis for one explicit agent.
- Workflows and jobs expand outside the compared axis.

For `agent_comparison`, each compared agent must resolve to exactly one model
through one of these sources, in order: an explicit `models_by_agent` entry in
the comparison object, a single model in that agent's top-level scenario entry,
or the adapter's declared default model. If more than one model remains
possible, preflight fails. If an adapter default is used, the run plan records
`model_source: adapter_default`.

## Prompt Model

The prompt is an explicit benchmark input and is the only task instruction the
harness gives the agent. A direct run passes a prompt file. A scenario may
select a prompt file or a declared prompt template, but every rendered byte is
part of the benchmark prompt and is copied into each record directory as
`prompt.txt` with hash metadata.

The harness must not append hidden text to the prompt. It must not add mode
names, workflow instructions, skill hints, record paths, metric expectations,
output-format requirements, or evaluator/reporting instructions unless those
words are already present in the explicit prompt source selected by the
scenario. This is the benchmark bias boundary: skill availability is the
experimental variable, not extra harness guidance.

Prompt templates use a strict variable renderer: every placeholder must have a
value, unknown placeholders fail preflight, and rendered prompts are plain text.
Template variables are substitutions only. They do not authorize the harness to
auto-inject job metadata, workflow text, mode names, or output expectations.
Direct prompt files are treated as already rendered prompts. Rendered scenario
templates are materialized under the result root during scenario write/execution
so scenario compilation does not mutate the scenario source directory.

For skill-ablation comparisons such as `without_skills` versus `with_skills`,
compared mode legs must receive identical prompt bytes unless the scenario is
explicitly labeled as a prompt-ablation experiment. The prompt hash is the
audit mechanism for this rule.

The harness must not rely on prompt text for mode names, record paths, report
filters, or evaluator behavior. Those are harness-supplied configuration values
kept outside the agent prompt.
The prompt hash is computed over the rendered prompt bytes and recorded as a
top-level record-summary field so prompt variation is visible in every
comparison.

## Skill Identity

The harness does not infer skill identity from workflow names, framework names,
or job folders. Skill identity is an observed output, not scenario input.

Accepted evidence sources include:

- structured agent records;
- structured benchmark records;
- explicit structured metadata written by the agent or validation workflow.

Reports should include:

```text
observed_skill_name
skill_name_source = agent_record | benchmark_record | structured_metadata | unknown
```

If no trustworthy skill identity is discovered, reports use `unknown`. The
report layer decides whether the evidence is trustworthy enough to describe as
observed skill identity; it does not infer identity from workflow or job names.

## Result Directory Layout

The result path encodes the axes needed for aggregation and debugging:

```text
results/
`-- <scenario_name>/
    |-- scenario.json
    |-- run_plan.json
    |-- scenario_summary.json
    |-- reports/
    |   |-- scenario_report.md
    |   `-- scenario_report.json
    `-- records/
        `-- agent=<agent>/
            `-- model=<model_slug>/
                `-- workflow=<workflow_slug>/
                    `-- job=<job_slug>/
                        `-- mode=<mode>/
                            |-- record_summary.json
                            |-- agent_events.jsonl
                            |-- agent_usage.json
                            |-- agent_activity.json
                            |-- agent_last_message.txt
                            |-- agent_stderr.txt
                            |-- agent_record.json
                            |-- benchmark_record.json
                            |-- input_delta_manifest.json
                            `-- workspace_delta_manifest.json
```

Slugs are filesystem-safe and stable: lowercase, replace every non-alphanumeric
sequence with `_`, trim leading/trailing `_`, truncate the visible part to 48
characters, and append an 8-character stable hash when truncation or collision
handling is needed. The hash input is the full untruncated source string after
normalization, not only the visible prefix. Empty slugs become `item_<hash>`.
Collision detection runs while expanding the run plan. If two normalized names
would produce the same slug within the same axis and parent path, both slugs get
hash suffixes derived from their full normalized source strings.

Preflight checks the maximum expanded artifact path length before execution.
The default path budget is 240 characters for host-visible paths. Scenarios may
set a shorter result root or an explicit path budget, but they must fail
preflight rather than produce paths that are likely to break on the host or
Docker volume mount.

`scenario.json` stores the resolved scenario, including job paths, prompt
hashes, wheel metadata, image tags, and agent versions. `run_plan.json` stores
expanded record entries in execution order. Reports aggregate by reading
scenario metadata and normalized records, not by guessing from directory names
alone.

The normalized `agent_*` artifacts are the source of truth. Provider-specific
files such as `codex_*`, `claude_*`, `hermes_*`, or `openclaw_*` may exist only
as debug or compatibility artifacts; reports must not require them.

Each run-plan entry executes once. If a run fails, the failure is benchmark
evidence and the harness preserves the artifacts needed to diagnose it. A user
may rerun the scenario into a new result root when they want another sample, but
the harness does not create additional executions for a failed run.

## Summary Schema

Every record summary includes:

```text
scenario_name
comparison_type
agent
agent_model
prompt_hash
prompt_source
workflow
observed_skill_name
skill_name_source
job_slug
job_path
job_scale
mode
skills_enabled
runtime_image
wheel_variant
elapsed_seconds
agent_elapsed_seconds
phase_seconds
token_count
command_count
agent_exit_code
final_container_exit_code
agent_process_passed
failure_root_cause
validation_metric
validation_metric_status
structure_quality_signal
artifact_paths
```

`elapsed_seconds`, when present for compatibility, is an alias for
`agent_elapsed_seconds`. Full setup, post-processing, and report timings live in
`phase_seconds`.

`phase_seconds` uses the same keys as the timing boundary contract:

```text
container_elapsed_seconds
setup_elapsed_seconds
skill_exposure_elapsed_seconds
agent_elapsed_seconds
post_process_elapsed_seconds
report_elapsed_seconds
```

Every comparison summary includes:

```text
comparison_type
group_axes
compared_records[]
aggregate_results{}
winner_policy
quality_gate
```

`winner_policy` describes how the report selected or refused to select a winner,
for example `median_agent_elapsed_seconds_then_tokens_with_quality_gate` or
`no_single_cost_winner`.

`quality_gate` describes the minimum correctness criteria applied before a cost
winner is considered meaningful. The harness must not invent correctness from
cost metrics.

The default quality gate is:

```text
agent_process_passed == true
final_container_exit_code == 0
source_input_modified == false
required_validation_metric_status in {present, not_required}
critical_quality_checks_failed == false
```

Scenarios may override the implemented gate fields shown above, and every
override is recorded in `quality_gate`. The current implementation does not yet
support the broader design vocabulary for named required checks or required
metric families. Unsupported quality-gate fields fail validation instead of
silently changing comparison semantics. A run that does not meet the gate can
still appear in reports, but it is not eligible to win a cost/performance
comparison.

The default winner policy is
`median_agent_elapsed_seconds_then_tokens_with_quality_gate`:

1. Exclude compared records that do not satisfy the quality gate.
2. If no compared record satisfies the gate, set winner policy to
   `no_quality_qualified_winner`.
3. Compare `agent_elapsed_seconds` across compared records.
4. Break ties with `token_count`.
5. If both values tie or required values are missing, report
   `no_single_cost_winner`.

Reports may also show aggregate statistics when a scenario contains multiple
records for the same comparison label.

Scenario execution continues after individual run failures unless the scenario
sets `fail_fast: true`. Scenario summaries report partial results and mark the
scenario as `degraded` when at least one run failed, was skipped, or lacked a
quality-qualified record. A scenario is `failed` when preflight fails before a
run plan is executable, all run-plan entries fail, or a required report cannot
be generated. A fully executed scenario with all required records satisfying
the quality gate is `passed`.

## CI And Test Boundaries

CI exercises harness health, not the full benchmark matrix. A CI scenario uses:

- one supported agent;
- one model;
- one small synthetic job folder;
- one workflow;
- one explicit comparison object;

CI verifies that Docker build/run works, records are produced, reports render,
skill exposure modes behave correctly, and parser assumptions remain valid.

Unit tests cover pure behavior:

- scenario expansion and slugging;
- event normalization;
- token parsing;
- record normalization;
- metric extraction;
- structure-tree rendering;
- report rendering helpers.

Adapter contract tests validate that sample agent outputs map into the neutral
event, usage, final-message, skill-exposure, and failure-classification
contracts. A new agent config is not `supported` until it has:

- sample raw event fixtures;
- usage fixtures with declared token semantics;
- final-message fixtures;
- launch dry-run or preflight coverage;
- skill-exposure probe coverage;
- Docker build metadata coverage;
- a static check that provider-specific names appear only in agent config,
  parser/classifier registry, Docker install, auth README, or declared
  debug/compatibility artifacts.

Integration smoke tests run a tiny synthetic job and verify normalized records
and reports. Long agent runs are opt-in.

## Replay Mode

The harness supports analysis development without live agent credentials through
a replay mode. Replay mode consumes captured neutral artifacts, such as
`agent_events.jsonl`, `agent_usage.json`, `agent_activity.json`,
`agent_last_message.txt`, workspace-delta manifests, and benchmark records, then
runs record synthesis, quality-signal extraction, aggregation, and report
generation without invoking an agent CLI.

Replay mode is for parser, record, and report development. It must not claim to
measure fresh agent performance, mutate the original job input, or refresh
token/cost data. Replay reports must identify their source run and mark
`agent_invocation` as `replayed`.

## Reporting Language

Reports use agent-neutral labels:

- `Agent events`;
- `Agent runtime`;
- `Agent usage`;
- `Agent final message`;
- `Agent status`;
- `Agent failure analysis`.

Reports always show agent name, model, CLI version when known, runtime image,
wheel variant, skill exposure mode, workflow, job metadata, and prompt hash.

Report code derives mode order and compared records from `scenario.json` and
comparison summaries. It must not hard-code a fixed three-mode ablation order or
evaluator-specific sections.
