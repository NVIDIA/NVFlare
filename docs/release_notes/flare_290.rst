**************************
What's New in FLARE v2.9.0
**************************

NVIDIA FLARE 2.9.0 focuses on simpler collaborative programming and more
predictable production execution. The release introduces the Pythonic Collab
API, three explicit Client API execution modes, HuggingFace Trainer support,
Slurm job execution, agent-ready workflows, standard Recipe metrics artifacts,
and major transport and launcher reliability improvements.

Release Highlights
==================

- **Pythonic Collab API**: publish ordinary Python client functions and invoke
  them from server workflows without manually building messages, ``FLModel``
  objects, or controller/executor pairs. Collab supports simulation and PoC
  execution, synchronous and asynchronous aggregation, and decentralized
  workflows such as Swarm.
- **Explicit Client API execution modes**: run training in the Client Job,
  launch and manage an external trainer process, or attach a trainer owned by
  another system. Attach supports the site's existing network listener and a
  protected shared-filesystem option.
- **HuggingFace Trainer Client API**: patch ``Trainer`` and ``SFTTrainer``
  workflows so FLARE handles round exchange, global-weight loading, local
  training budgets, checkpoint state, and rank-0 communication.
- **Recipe automation and metrics artifacts**: use public per-site and metadata
  helpers, a consistent ``--export`` / ``--export-dir`` interface, and standard
  ``metrics_summary.json`` and ``round_metrics.jsonl`` result artifacts for
  supported built-in training recipes.
- **Agent-ready FL workflows**: packaged NVFlare agent skills provide
  repeatable, reviewable workflows for orientation, conversion, diagnosis, and
  Auto-FL experiment planning and evaluation.
- **Slurm and Kubernetes operations**: prepare Slurm startup kits, run workers
  through Apptainer, Pyxis, or a bare environment, and use stronger
  Kubernetes/Slurm credential, child-exit, timeout, and cleanup behavior.
- **Large-model transport improvements**: F3 adds receiver-confirmed progress,
  bounded receiver budgets, retry-aware completion, larger default streaming
  capacity, and benchmark-selected window and ACK settings.
- **PyTorch Lightning integration**: automatic SCAFFOLD and FedProx support and
  corrected multi-rank result handling reduce custom federated training code.

Collab API
==========

The new Collab API separates collaborative algorithm code from transport and
runtime wiring. Client functions use ``@collab.publish`` and server workflows
use ``@collab.main``; calls such as ``collab.clients.train(model)`` return
ordinary Python results associated with each site. This keeps aggregation and
training logic directly readable and locally testable while FLARE supplies the
distributed runtime.

See :ref:`api_evolution` for API-selection guidance and the
:github_nvflare_link:`Hello Collab example <examples/hello-world/hello-collab>`
for a runnable FULL/DIFF FedAvg workflow. Advanced callback, asynchronous,
split-learning, and Swarm examples are under
:github_nvflare_link:`examples/advanced/collab <examples/advanced/collab>`.

Client API and Framework Integrations
=====================================

``ClientAPIExecutor`` now provides explicit ``in_process``,
``external_process``, and ``attach`` modes. Choose a mode based on who owns the
trainer process: NVFlare owns in-process and external-process trainers, while
an external system owns an attached trainer. Network Attach reuses the site's
Client Parent listener; protected shared-file Attach uses a Client Job-owned
listener.

The HuggingFace Client API patches an existing ``Trainer`` or ``SFTTrainer``
instead of requiring an application to implement the Client API receive/send
loop manually. Lightning workflows add automatic SCAFFOLD and FedProx support,
and nonzero distributed ranks no longer send duplicate results.

See :ref:`client_api_usage`, :ref:`client_api_attach`, and
:ref:`hf_client_api`.

Recipes, Metrics, and Agent Workflows
=====================================

Recipe APIs now expose supported per-site configuration, generated-job
metadata, execution-environment settings, and consistent job export. Built-in
training aggregation recipes also emit standard summary and per-round metrics
artifacts when the workflow reports aggregation metrics, so automation can
consume results without scraping logs.

NVFlare-owned agent skills package deterministic helpers and reviewable
contracts for common coding-agent workflows. The Auto-FL skill guides dataset
inspection, experiment planning, bounded trial execution, metric validation,
and final reporting.

See :ref:`job_recipe`, :ref:`recipe_api`, :ref:`recipe_metrics_artifacts`,
:ref:`recipe_command`, and :ref:`autofl_skill`.

Deployment and Transport
========================

The new Slurm launcher prepares a stable shared workspace and runs workers in
Apptainer, Pyxis, or a site-managed bare environment. It supports shared-file
Attach, configurable scheduler timeouts, multi-node worker launch, result-aware
heartbeat cleanup, and propagation of useful worker exit codes.

Kubernetes deployment adds per-study job-pod templates and ConfigMap/Secret
staging, and now propagates child failures more reliably. Job-process bootstrap
credentials move from process arguments into the environment; Kubernetes uses
a per-job Secret.

F3 streaming now uses a 64 MiB sender window and 16 MiB ACK interval selected
for a nominal 25 Gbit/s network. Receiver-confirmed transfer progress, bounded
pending data, retry settlement, and larger blob support strengthen transfer of
large models and results.

See :ref:`slurm_job_launcher`, :ref:`deploy_prepare_command`, and
:ref:`communication_configuration`.

Compatibility and Migration Notes
=================================

- F3 now uses a 64 MiB streaming-window and 16 MiB ACK-interval sender default,
  matching ``dev_tools/f3/comm_config.yml`` and the values selected for a
  nominal 25 Gbit/s network. The sender supplies these values to the receiver;
  4 MiB remains only the receiver-local ACK fallback when a legacy peer does
  not provide streaming parameters. TCP connections now enable
  ``TCP_NODELAY`` by default to reduce request/ACK latency. Review aggregate
  stream concurrency and memory budgets when upgrading from the smaller 2.8
  defaults.
- Job-process bootstrap credentials (auth token, token signature, session ID)
  are no longer passed as command-line arguments. Launchers deliver them
  through the job process environment; on Kubernetes they ride a per-job
  Secret referenced via ``env[].valueFrom.secretKeyRef``. There is no fallback
  machinery, so Docker and Kubernetes job images must run NVFlare 2.9 or
  newer: a job image pinned to 2.8 or earlier fails immediately at argument
  parsing when launched by a 2.9 CP/SP. The CLI path is retained, so an older
  parent launching a newer job image is unaffected. Custom launchers that
  render worker commands from ``generate_client_command`` /
  ``generate_server_command`` and implement ``launch_job`` directly must also
  export ``get_credential_env(job_args)`` into the child environment. Launcher
  Kubernetes RBAC now needs the ``patch`` and ``delete`` verbs on Secrets
  (included in the generated Helm role templates).
- Patched PyTorch Lightning clients now report ``NUM_STEPS_CURRENT_ROUND`` as
  the actual per-round change in ``trainer.global_step`` instead of
  ``trainer.estimated_stepping_batches``. This corrects cumulative aggregation
  over-weighting in later rounds when ``update_fit_loop=True``. Because
  ``global_step`` counts completed optimizer steps across optimizers, a
  multi-optimizer FedAvg client reports their combined step count unless it
  supplies ``NUM_STEPS_CURRENT_ROUND`` explicitly; explicit client metadata is
  still preserved.
- Recipe discovery now exposes the concrete PyTorch ``FedProxRecipe`` as
  ``fedprox-pt`` and no longer advertises the ``fedprox-tf`` manual pattern as
  a concrete recipe. TensorFlow clients can continue to combine a FedAvg
  recipe with ``TFFedProxLoss`` explicitly.
- CellPipe cell names now keep the runtime token and pipe mode in one
  explicitly marked, ``~``-delimited FQCN leaf segment
  (``site-1.cellpipe~plain~<job-id>~active``, or
  ``<relay>.cellpipe~alias~<site>~<job-id>~active`` behind a relay) so a
  pipe cell's FQCN parent matches the cell it actually connects to and pipe
  names can never be confused with other cell names. As part of this change,
  CellPipe validates tokens at construction: tokens must be non-empty, may
  not contain the reserved ``~`` separator, and may not contain ``.`` when
  the pipe connects to the site's own CP or a relay. Custom
  ``FlareAgentWithCellPipe`` agent ids that violate these rules now fail fast
  with a ``ValueError`` instead of producing unroutable cell names.
- Both ends of a CellPipe pair derive each other's cell names independently,
  so a Client Job process and an external training process must run the same
  NVFlare naming scheme. A training environment pinned to an older NVFlare
  fails with "peer FQCN mismatch" when paired with a 2.9 CJ; align the
  training environment's NVFlare version with the site's. Only the flat
  whole-FQCN alias used by NVFlare 2.8 and earlier (a root-connected pipe
  named ``<site>_<token>_<mode>``) is still recognized for backward
  compatibility. The forms used through 2.8 when nested under a CP or relay
  (``<parent>.<site>_<token>_<mode>``) are not, because an unmarked leaf
  inside a longer FQCN is indistinguishable from a real cell of that name.
  When upgrading to 2.9, upgrade a site and its relay together, including
  sites currently running NVFlare 2.8.
- ``ScriptRunner`` now exports ``ClientAPIExecutor`` for both in-process and
  external-process execution. Jobs generated with FLARE 2.9 therefore require
  a client runtime that provides this executor and are not runnable on older
  client runtimes. ``ScriptRunner`` no longer performs a build-time PyTorch or
  TensorFlow import check; ensure the required framework dependencies are
  available in the execution environment. Code that explicitly passes
  ``pipe_connect_type`` (including its former default value) or supplies a
  custom ``task_pipe`` must use ``BaseScriptRunner``. A client app may contain
  only one ``ClientAPIExecutor``; configurations that previously added multiple
  script runners to one site must combine the scripts behind one entry point
  and dispatch on the Client API task name.
- PyTorch swarm learning can now combine client-to-client tensor streaming with
  aggregation-client tensor disk offload. Set
  ``aggregation_format=ExchangeFormat.PYTORCH`` and
  ``enable_tensor_disk_offload=True`` on ``SwarmLearningRecipe``. The same
  offload flag is available on ``SwarmClientConfig`` for Job API users.
- The deprecated FL HUB implementation under ``nvflare.app_common.hub`` has
  been removed. Configurations that reference its deployer, controller, or
  executor components must migrate to the
  :ref:`current hierarchical FL architecture <flare_hierarchical_architecture>`.
- ``ModelLearner`` and the NumPy ``NPTrainer`` variants remain available for
  backward compatibility, but now emit deprecation warnings when
  instantiated. Use :ref:`Job Recipes <job_recipe>` with the
  :ref:`Client API <client_api>` for new applications.

See the :ref:`migration_guide` for version-by-version upgrade guidance.
