:orphan:

**************************
What's New in FLARE v2.9.0
**************************

Compatibility and Migration Notes
=================================

- ``nvflare poc start`` and ``nvflare poc stop`` now preserve every repeated
  ``-p`` / ``--service`` and ``-ex`` / ``--exclude`` value. Earlier versions
  silently kept only the last value. ``poc stop`` now also honors participant
  exclusions. A bare ``poc start`` continues to start the server and clients
  without an admin console.
- ``nvflare poc stop`` now waits for targeted and exclusion-based shutdowns to
  complete before returning ``status: stopped``. Use ``--no-wait`` for
  fire-and-forget behavior.
- Docker jobs may now control only ``image``, ``python_path``, ``entrypoint``,
  ``num_of_gpus``, and ``shm_size`` through their launcher metadata. Selecting
  ``image``, ``python_path``, or ``entrypoint`` requires BYOC authorization at
  each receiving site. Previously job-controlled Docker SDK options such as
  ``ipc_mode`` and ``device_requests`` are now site-owned and can be configured
  through ``default_job_container_kwargs`` or a study's ``docker_kwargs``;
  launcher-owned options such as mounts and networks remain fixed. New jobs
  containing unsupported options are rejected at submission; jobs stored before
  an upgrade are checked again and can fail at launch until their metadata is
  migrated.
- The flat ``resource_spec`` names ``num_of_gpus``, ``num_of_cpus``, and ``memory`` are now reserved portable fields
  and must use the documented portable types. Custom resource managers that previously interpreted these names
  differently must migrate to the portable types or rename their custom fields. Legacy nested resource specifications
  without ``@default`` remain unchanged.
- Collab calls now carry a versioned authorization envelope and are accepted
  only from authenticated participants in the same job. All sites that run a
  Collab job must use NVFlare 2.9 or newer; a 2.8 or older Collab sender lacks
  this envelope and its calls are rejected. A 2.9 sender receives an immediate
  ``COMM_ERROR``; a 2.8 or older sender can instead observe a request timeout
  because it lacks the dedicated stream-error correlation used by 2.9. The
  receiving site logs that the peer may be running an older NVFlare version to
  make this mixed-version failure diagnosable.
- F3 retains its 16 MiB streaming-window and 4 MiB ACK-interval defaults for
  compatibility and bounded per-stream memory. High-bandwidth deployments may
  opt into larger values on all endpoints; ``dev_tools/f3/comm_config.yml``
  provides a 64 MiB/16 MiB tuning example. TCP connections now enable
  ``TCP_NODELAY`` by default to reduce request/ACK latency.
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
- Patched PyTorch Lightning clients now transmit validation metrics captured by
  an explicit ``trainer.validate()`` call before ``trainer.fit()``, regardless
  of ``train_with_evaluation``. Sanity-check and in-fit validation metrics are
  not transmitted as global-model scores. Setting
  ``train_with_evaluation=True`` continues to require validation metrics;
  otherwise metrics remain optional, but ``False`` no longer suppresses metrics
  from an explicit pre-fit validation. This enables ``IntimeModelSelector`` and
  best-global-model persistence for recipe-based Lightning jobs. Applications
  that must keep such metrics local should omit the explicit pre-fit validation;
  if they still require it locally, a custom task-result filter must remove
  ``MetaKey.INITIAL_METRICS`` before the result reaches the server.
- ``SimpleIntimeModelSelector`` (CCWF) now selects a scalar ``key_metric``
  (default ``val_accuracy``) from dict-valued ``INITIAL_METRICS`` payloads
  instead of failing with a logged ``TypeError`` that silently disabled
  best-model selection. Swarm jobs whose metric dicts contain the configured
  key change from selection-inert to active best-global-model tracking without
  a config change; dict payloads lacking the key, and non-numeric values, are
  skipped with a warning, so configure ``key_metric`` to match the reported
  metric name. A new ``negate_key_metric`` argument supports lower-is-better
  metrics such as losses.
- Recipe discovery now exposes the concrete PyTorch ``FedProxRecipe`` as
  ``fedprox-pt`` and no longer advertises the ``fedprox-tf`` manual pattern as
  a concrete recipe. TensorFlow clients can continue to combine a FedAvg
  recipe with ``TFFedProxLoss`` explicitly.
- The legacy Client API execution stack has been removed. This includes
  ``ParamsConverter``, the framework-specific converter components,
  ``InProcessClientAPIExecutor``, ``ClientAPILauncherExecutor``,
  ``LauncherExecutor``, ``SubprocessLauncher``, ``TaskExchanger``,
  ``FlareAgent``, ``BaseScriptRunner``, ``ExternalConfigurator``, and the
  ``Pipe``/``PipeHandler`` implementations (including ``FilePipe`` and
  ``CellPipe``). Use ``ClientAPIExecutor`` with
  ``in_process``, ``external_process``, or ``attach`` execution mode. Custom
  parameter transformations belong in trainer code around
  ``flare.receive()``/``flare.send()``; common functions remain available in
  ``nvflare.client.converter_utils``.
- Recipe-level ``pipe_type`` and ``pipe_root_path`` options have been removed.
  Transport is selected through site communication configuration. The F3
  ``FileDriver`` remains available as scheme ``shared-file`` for either a
  launched external process or an attached trainer.
- ``ScriptRunner`` now exports ``ClientAPIExecutor`` for both in-process and
  external-process execution. Jobs generated with FLARE 2.9 therefore require
  a client runtime that provides this executor and are not runnable on older
  client runtimes. ``ScriptRunner`` no longer performs a build-time PyTorch or
  TensorFlow import check; ensure the required framework dependencies are
  available in the execution environment. A client app may contain only one
  ``ClientAPIExecutor``; configurations that previously added multiple
  script runners to one site must combine the scripts behind one entry point
  and dispatch on the Client API task name.
- PyTorch swarm learning can now combine client-to-client tensor streaming with
  aggregation-client tensor disk offload. Set
  ``aggregation_format=ExchangeFormat.PYTORCH`` and
  ``enable_tensor_disk_offload=True`` on ``SwarmLearningRecipe``. The same
  offload flag is available on ``SwarmClientConfig`` for Job API users.
- Auto-FL campaigns now honor the job's native metric direction
  (``key_metric_mode`` or a matching same-metric ``stop_cond``) instead of
  assuming maximization, so raw lower-is-better objectives no longer need to be
  negated. Campaign admission fails closed in three new cases: an obvious
  lower-is-better metric such as ``val_loss`` that relies only on NVFlare's
  implicit ``max`` default is rejected until the job declares
  ``key_metric_mode="min"``, and jobs passing a custom ``model_selector`` are
  rejected because that component supersedes ``key_metric_mode``, so its
  selection direction cannot be imported deterministically. A requested metric
  that differs from the job's key metric is also rejected unless
  ``mutation_schema.yaml`` declares the requested and optimization metric
  bridge. Remove the custom selector and expose its criterion as a declared
  ``key_metric`` with ``key_metric_mode``, or declare the alternate metric
  bridge, before initializing a campaign. Experimental legacy minimization
  campaigns without direction provenance must be re-initialized in a fresh
  workspace. Job constructor calls that pass ``**kwargs`` now also fail closed
  when the splat could hide ``key_metric``, ``key_metric_mode``,
  ``model_selector``, or ``stop_cond``; spell out the safety-critical keywords
  in the call before initializing.
- In external-process Client API mode, losing a trainer after its lazy result
  envelope has been accepted now fails the run as ``EXECUTION_EXCEPTION`` even
  if a controller's ``min_responses`` threshold could otherwise tolerate a
  missing client. The accepted envelope may already have exposed references to
  downstream consumers and cannot be safely withdrawn. An explicit job abort
  that wins the terminal-state race remains ``ABORTED``.
- ``SwarmLearningRecipe`` now configures client-side best-model selection by
  default. Use ``key_metric`` to select a dictionary-valued validation metric
  and ``key_metric_mode="min"`` for lower-is-better metrics. Clients must report
  a pre-training validation metric with the configured name for selection to
  occur; jobs without that metric continue to persist the last global model but
  do not create ``best_FL_global_model.pt``. Set ``key_metric=None`` to opt out
  and preserve the pre-2.9 last-model-only behavior. Selection skips round 0,
  so a one-round job does not create a best-model checkpoint. With
  ``key_metric_mode="min"``, Swarm best-metric logs and records expose the
  negated comparison value (for example, a loss of 2.31 is shown as -2.31).
  ``client_config_overrides`` can no longer replace ``model_selector``: migrate
  the former ``{"model_selector": None}`` opt-out to ``key_metric=None``, and
  use ``BaseSwarmLearningRecipe`` with an explicit ``SwarmClientConfig`` for a
  custom selector.
