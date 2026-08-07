:orphan:

**************************
What's New in FLARE v2.9.0
**************************

Compatibility and Migration Notes
=================================

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
- Recipe discovery now exposes the concrete PyTorch ``FedProxRecipe`` as
  ``fedprox-pt`` and no longer advertises the ``fedprox-tf`` manual pattern as
  a concrete recipe. TensorFlow clients can continue to combine a FedAvg
  recipe with ``TFFedProxLoss`` explicitly.
- The legacy Client API execution stack has been removed. This includes
  ``ParamsConverter``, the framework-specific converter components,
  ``InProcessClientAPIExecutor``, ``ClientAPILauncherExecutor``,
  ``LauncherExecutor``, ``SubprocessLauncher``, ``TaskExchanger``,
  ``FlareAgent``, ``BaseScriptRunner``, ``MetricRelay``, ``MetricsSender``,
  ``ExternalConfigurator``, and the ``Pipe``/``PipeHandler`` implementations
  (including ``FilePipe`` and ``CellPipe``). Use ``ClientAPIExecutor`` with
  ``in_process``, ``external_process``, or ``attach`` execution mode. Custom
  parameter transformations belong in trainer code around
  ``flare.receive()``/``flare.send()``; common functions remain available in
  ``nvflare.client.converter_utils``.
- Flower metric streaming now uses ``FlowerMetricsReceiver`` and a direct local
  Cell Client API session. Legacy ``MetricRelay``/``MetricsSender`` component
  IDs, metrics ``CellPipe`` configuration, and selector overrides should be
  removed from custom Flower job templates. ``FLARE_CLIENT_API_TYPE`` is now a
  reserved ``FlowerJob.extra_env`` key and should also be removed; Flower
  configures the metrics session automatically.
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
