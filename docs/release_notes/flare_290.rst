:orphan:

**************************
What's New in FLARE v2.9.0
**************************

Compatibility and Migration Notes
=================================

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
