.. _migration_guide:

################
Migration Guide
################

This guide covers API and configuration changes when upgrading between FLARE releases.

Upgrading from 2.9 to 2.10
==========================

2.9 and 2.10 sites do not interoperate, whichever 2.10 features you use. Every
2.10 connection handshake carries a transport protocol version, and a peer with
a missing or different version, including every 2.9 peer, is rejected before it
attaches. Upgrade the server, clients, relays, and admin environments together,
using the procedure below, and review the behavior changes that follow it
before you restart a federation.

Upgrade Procedure
-----------------

Copy each existing client kit to a **separate workspace** and start it in a
2.10 environment, keeping the 2.9 client running. Shared workspaces conflict
on PID and shutdown files. No certificate or endpoint changes are needed.
The staged client probes under a distinct name without registering. It waits
for a compatible server, retrying every 60 seconds (configurable via
``client.upgrade_probe_interval``). Stop the process to cancel waiting.

At cutover, finish or abort jobs and issue ``shutdown all`` from the 2.9 admin
console. Exit the old server process, then start 2.10 using its existing kit
and workspace. Upgrade the admin environment and any relays; start upgraded
relays **after the 2.10 server is ready**. A relay handshake identifies the
relay's protocol, not the root server's protocol.

Staged clients join on a subsequent probe; late sites join when installed.
Verify with ``nvflare system version``. Leftover 2.9 peers are rejected before
endpoint attachment and cannot replace a 2.10 connection. This procedure
includes a server outage and does not resume running jobs across the upgrade.

Re-provision Secure Projects for Per-Job Certificates
-----------------------------------------------------

This is the one security change in 2.10 that is required. Single sign-on and
certificate-derived study memberships are optional and need no action.

In secure mode, 2.10 gives each job process (server job and client job) its
own short-lived credential, issued by a job-signing CA that ``CertBuilder``
generates into the server startup kit. The site's long-lived key is no longer
used by job processes.

Impact (secure mode only; non-secure deployments are not affected):

- A secure-mode server kit without a job CA, or with one about to expire,
  cannot deploy jobs; the reason is reported in the job deploy detail. There
  is no fallback to the site certificate.
- The server and clients must run the same release.
- In secure mode, the Docker and Slurm launchers refuse a clear-text parent
  connection. Configure the client's internal listener with ``stcp`` or
  ``mtls``.

Action: re-provision existing secure-mode projects with 2.10 and distribute
the new startup kits. ``CertBuilder`` generates the job CA by default. If your
deployment manages its own CA material, mark the server ``external_job_ca``
and install the job-CA credentials at deployment; setting
``enable_job_ca: false`` alone leaves a secure server unable to deploy jobs.

Process Jobs Now Apply ``local/study_runtime.yaml``
---------------------------------------------------

Process jobs previously ignored ``local/study_runtime.yaml``. They now apply
the selected study's ``env`` and ``secret_env`` before each launch. Study
values override inherited settings, and a missing secret source fails the
launch before the process starts. Container and mount settings are rejected in
Process mode.

Action: review any existing ``local/study_runtime.yaml`` on Process sites
before upgrading, because values that used to have no effect now take effect.

Aggregation Rejects Incompatible Parameter Shapes
-------------------------------------------------

Weighted aggregation previously accepted client parameters with different
shapes by broadcasting them (for example ``(2,)`` with ``(1,)``, or a
zero-dimensional value with a singleton array). 2.10 validates exact shapes
before accumulating and rejects a mismatched contribution.

Impact:

- Built-in FedAvg continues with the remaining valid clients. A metric shape
  mismatch rejects the whole client contribution.
- A round with no accepted contributions fails before updating or saving.
- Jobs that relied on silent broadcasting now see rejected contributions in
  the logs.

Action: make client parameter shapes match the global model.

Docker GPU Reservation
----------------------

For GPU-managed Docker clients, 2.10 reserves specific devices through the
client's GPU resource manager and passes those device IDs to Docker, so
concurrent jobs no longer receive the same GPUs.

Impact:

- Omitted job GPU counts inherit the site or study defaults at admission.
  Explicit job counts, including zero and legacy process counts, override the
  defaults.
- A failed job start now cancels pending client reservations immediately.
- Passthrough clients keep their previous behavior: job GPU counts and
  site or study ``device_requests`` (including pinned ``DeviceIDs``) go
  directly to Docker, with no FLARE reservation, and need no re-provisioning.
  Server containers keep native Docker GPU requests.

Action: review the GPU settings of Docker client sites that are GPU-managed.

Dashboard Changes
-----------------

- Public registration and user self-updates accept only an unset role,
  ``member``, or ``lead``. Administrative, custom, and malformed roles are
  rejected, and project administrators can still assign administrative roles.
- Client creation and PATCH requests that contain ``props`` are rejected with
  HTTP 400, and client-kit generation ignores previously stored ``props``.

Action: API integrations that send ``props`` must omit it.

Collab API (Technical Preview)
------------------------------

The ``async_aggregation`` example is renamed ``response_callbacks``, and its
entry point is ``job.py``. Update any paths or commands that refer to the old
name.

Task-Scoped Execution (Opt-In)
------------------------------

No action is needed for existing jobs; job lifetime remains the default. If
you adopt ``execution_lifetime="task"`` with PyTorch, note that unfiltered
Client Job input hooks receive local tensor references that support metadata
and ``nbytes``, not tensor attributes such as ``.shape`` or ``.numel()``.
Workers receive tensors.

Task lifetime requires an explicit Kubernetes or Slurm task launcher in the
site configuration; Process and Docker sites reject a task-lifetime app, so a
recipe-wide ``execution_lifetime="task"`` does not downgrade them. Keep those
sites at job lifetime through per-site client apps.

Kubernetes: Pod Patch Permission
--------------------------------

Kubernetes job Pods use a settlement finalizer, so the server's Role needs
``patch`` permission on Pods. The server Helm chart adds it; add it to any
custom Role.

Internal Renames With Compatibility Aliases
-------------------------------------------

The client terminal-report constant, class, and handler are renamed from job
failure to job outcome (``REPORT_JOB_FAILURE``, ``JobFailureMsgKey``, and
``process_job_failure`` remain as aliases, and ``report_job_failure`` remains
the wire value). Existing peers and callers keep working; no action is needed
unless you subclass these.

New in 2.10 (No Migration Required)
-----------------------------------

The following features are opt-in or additive:

- Single sign-on for administrators through ``admin_cert_provider`` with the
  built-in ``step_ca`` provider (optional; off unless configured). Static admin
  startup kits are unchanged and can coexist with provider-backed kits.
- Disk-backed FedAvg and FedProx aggregation (``enable_disk_aggregation``).
- Certificate-derived study memberships (optional; registry-based study
  access is unchanged).
- Outbound HTTP proxy support for the HTTPS transport.

See :doc:`release_notes/flare_2100` for details, and see the job clone
deprecation under "Upcoming Main-Branch Changes" below.

Upgrading from 2.7.2 to 2.8.0
=============================

Python and Removed Legacy Surfaces
----------------------------------

FLARE 2.8.0 targets Python 3.10 through 3.14. Python 3.9 is no longer listed as
a supported development target.

The deprecated FLAdminAPI surface has been removed. Use the FLARE API, Recipe
API, Client API, and ``nvflare`` CLI workflows for new automation.

HA/Overseer code has also been removed from the 2.8 branch.

Client API Subprocess Timeout Validation
----------------------------------------

Subprocess-mode Client API jobs now validate two large-model safety settings at
job initialization:

- ``download_complete_timeout`` must not be ``None``. The subprocess must stay
  alive after ``send_to_peer()`` ACKs so the server can finish downloading
  tensors from the subprocess ``DownloadService``.
- ``max_resends`` must not be ``None`` when using ``ClientAPILauncherExecutor``.
  Unlimited resends can turn one delayed large-model transfer into an unbounded
  series of replacement download transactions.

If your 2.7.x job explicitly set either value to ``None``, update it before
running on 2.8.0. Recipe-based external-process jobs already serialize the
default ``max_resends=3`` in executor args, so the following setting is only
needed when overriding a previous explicit ``None`` or choosing a different
retry budget:

.. code-block:: python

   recipe.add_client_config({
       "download_complete_timeout": 1800,
       "max_resends": 3,  # finite non-negative integer; 0 disables retries
   })

For large tensor or NumPy payloads, also keep the related streaming timeouts
consistent. If you explicitly raise ``tensor_streaming_per_request_timeout`` or
``np_streaming_per_request_timeout``, set ``PEER_READ_TIMEOUT`` and
``download_complete_timeout`` to values at least as large as the configured
streaming per-request timeout, and keep ``tensor_min_download_timeout`` or
``np_min_download_timeout`` at least as large as the same value.

.. code-block:: python

   recipe.add_client_config({
       "tensor_streaming_per_request_timeout": 600,
       "tensor_min_download_timeout": 600,
       "PEER_READ_TIMEOUT": 600,
       "download_complete_timeout": 1800,
       "max_resends": 3,
   })

Late Retry Handling for Finished Download Refs
----------------------------------------------

FLARE 2.8.0 makes finished ``DownloadService`` refs retry-safe for the same
requester. If a client completed a large download but retries because the final
EOF response was delayed, the server returns the same terminal status instead of
``INVALID_REQUEST`` / ``no ref found``. This is an internal reliability fix and
does not require job-code changes, but it is most effective when the subprocess
timeouts above are configured consistently for very large models.

Upcoming Main-Branch Changes
============================

Job Clone Deprecation
---------------------

The ``nvflare job clone`` command, the legacy interactive Admin CLI
``clone_job`` command, and the Python FLARE API ``clone_job()`` method are
deprecated for NVFlare 2.10.0. They remain functional during the compatibility
period, but cloning copies the stored job artifact without running the
client-side signing path. The clone therefore retains the original
``.__nvfl_sig.json`` signatures and embedded ``.__nvfl_submitter.crt``
certificate, including its signer identity and absolute expiration. Cloning
does not renew or replace the original signing certificate.

To retry or retrigger a job, re-export or reuse the original local job folder
and submit it with current credentials so the artifact is signed with the
current submitter certificate:

.. code-block:: shell

   nvflare job submit -j JOB_FOLDER

If you no longer have the original local job folder, the clone command remains
available during the compatibility period. For a finished job, you can download
the job, reuse the job definition automatically extracted from ``job.zip``, and
submit that folder with current credentials. Downloading currently requires the
job to have finished, so this recovery path does not cover every case supported
by cloning.

Legacy Client API Stack Removal
-------------------------------

The legacy converter, launcher, agent, exchanger, and pipe APIs have been
removed. This includes ``BaseScriptRunner`` and ``ExternalConfigurator`` in
addition to the legacy executor and pipe classes. Migrate client jobs to
:class:`ClientAPIExecutor<nvflare.app_common.executors.client_api_executor.ClientAPIExecutor>`
using ``in_process``, ``external_process``, or ``attach`` mode.

Recipe-level ``pipe_type`` and ``pipe_root_path`` settings are no longer
accepted. Select transport in site ``comm_config.json`` instead; the F3
``FileDriver`` remains available as scheme ``shared-file`` for an attached
trainer. A launched external-process trainer instead requires a clear TCP
listener bound to loopback. For custom model representation logic,
transform parameters explicitly around ``flare.receive()`` and
``flare.send()`` or use helpers in :mod:`nvflare.client.converter_utils`.

FLARE API Compatibility Note
----------------------------

On the current ``main`` branch, :class:`NoConnection<nvflare.fuel.flare_api.api_spec.NoConnection>`
now subclasses Python's built-in ``ConnectionError`` instead of directly subclassing
``Exception``.

Impact:

- Existing code that catches ``ConnectionError`` will now also catch
  ``NoConnection``.
- Existing code that catches ``NoConnection`` continues to work unchanged.

If your application distinguishes FLARE connection failures from broader OS or
network exceptions, review any broad ``except ConnectionError:`` handlers before
upgrading to the next release built from ``main``.

FLARE API Lifecycle Restriction
-------------------------------

On the current ``main`` branch, :meth:`Session.shutdown<nvflare.fuel.flare_api.api_spec.SessionSpec.shutdown>`
and :meth:`Session.restart<nvflare.fuel.flare_api.api_spec.SessionSpec.restart>`
are now restricted to ``TargetType.SERVER`` only.

Impact:

- Existing callers that pass ``TargetType.ALL`` or ``TargetType.CLIENT`` will now fail.
- Server-scoped lifecycle control continues to work unchanged.
- :meth:`Session.shutdown_system<nvflare.fuel.flare_api.api_spec.SessionSpec.shutdown_system>`
  is unchanged and still supports whole-system shutdown.

For whole local PoC lifecycle control, use the PoC start/stop flow instead of
the general system admin API.

FLARE API Abort of a Starting Job
---------------------------------

On the current ``main`` branch, :meth:`Session.abort_job<nvflare.fuel.flare_api.api_spec.SessionSpec.abort_job>`
retries for up to 30 seconds when the server replies that the job is still
starting ("is starting; retry abort."). Previously this reply raised
``InternalError`` immediately.

Impact:

- Callers no longer need their own retry loop around ``abort_job`` for a job
  that was just submitted. This includes the Recipe ``Run.abort()``, the POC and
  production environments, and the ``nvflare job abort`` command.
- A call can now block for up to 30 seconds before raising if the job is still
  starting, for example while another thread holds the same ``Run`` lock.
- All other errors, including authorization failures, are raised immediately and
  are not retried. Existing handlers for them continue to work unchanged.

CLI Startup Kit Resolution Change
---------------------------------

On the current ``main`` branch, server-connected CLI commands use a shared
active startup kit registry in ``~/.nvflare/config.conf``.

Impact:

- Use ``nvflare config add <id> <startup-kit-dir>`` and
  ``nvflare config use <id>`` to register and activate a startup kit.
- ``nvflare config -d/--startup_kit_dir`` remains accepted for compatibility
  with 2.7.x scripts, but is deprecated.
- ``NVFLARE_STARTUP_KIT_DIR`` remains an automation override and takes
  precedence over the active registry entry when set.
- ``nvflare config -jt/--job_templates_dir`` remains accepted for compatibility
  with 2.7.x scripts, but job template config is deprecated.
- Root ``nvflare config`` continues to manage local settings such as the POC
  workspace. Startup kit paths are managed by the ``nvflare config``
  subcommands.

If you use shell profiles or CI settings that export ``NVFLARE_STARTUP_KIT_DIR``,
review them before upgrading because they override the active registry entry.

CLI Config Flag Compatibility
-----------------------------

On the current ``main`` branch, ``nvflare config`` keeps the 2.7.x POC
workspace flag names.

Impact:

- ``-pw`` and ``--poc_workspace_dir`` remain the supported flags for setting
  the POC workspace.
- The interim development-only ``--poc.workspace`` spelling is not part of the
  public compatibility contract.

If you have older scripts that use ``-pw`` or ``--poc_workspace_dir``, they
continue to work.

Client Disable Semantics
------------------------

``nvflare system remove-client`` is not exposed as a supported public CLI
command. The legacy interactive-console ``remove_client`` command is hidden
from normal help and remains a registry cleanup operation only: it releases
the active token so the client can register again. It does not stop the client
process, revoke credentials, or prevent reconnect.

Use the new durable access-control commands when the intent is to keep a client
out of the federation:

- ``nvflare system disable-client <client> --force`` persists a disabled flag
  in the server workspace, removes any active registry entry, and rejects
  later registration or heartbeat from that client.
- ``nvflare system enable-client <client> --force`` clears the disabled flag so
  the client can rejoin on the next registration or heartbeat.

This is operational disablement, not certificate revocation.

Study Name Validation Relaxation
--------------------------------

On the current ``main`` branch, study names now allow underscores in internal
positions, so names such as ``my_study`` are valid.

Impact:

- ``project.yml`` validation now accepts study names with internal underscores.
- Login and study-scoped authorization paths will accept the same names.

If you maintain external validation or naming policy around study identifiers,
update those checks to match the new rule before upgrading.

Site Log Configuration Restriction
----------------------------------

On the current ``main`` branch, :meth:`Session.configure_site_log<nvflare.fuel.flare_api.api_spec.SessionSpec.configure_site_log>`
and the corresponding ``nvflare system log-config`` path now accept only simple
log levels and built-in log modes.

Impact:

- JSON ``dictConfig`` payloads are no longer accepted for site-wide log changes.
- File-path based logging configs are no longer accepted for site-wide log changes.
- Supported values remain the standard log levels plus built-in modes such as
  ``concise``, ``progress``, ``msg_only``, ``full``, ``verbose``, and ``reload``.

If you previously used advanced JSON/file-based configs with
``configure_site_log``, switch to the supported level/mode values before
upgrading to the next release built from ``main``.
For dict-based or file-path logging, use ``configure_job_log`` on a running job instead.

POC Start Defaults and Repeatable Selection
-------------------------------------------

On the current ``main`` branch, the documented default behavior of
``nvflare poc start`` is clarified to reflect the actual runtime behavior:
the default start set is the server plus client services, not every
participant directory under the workspace.

Impact:

- Running ``nvflare poc start`` with no explicit ``-p`` / ``--service`` starts
  the server and clients.
- Admin consoles are not started unless explicitly selected.
- ``-p`` / ``--service`` and ``-ex`` / ``--exclude`` can be repeated for
  ``poc start`` and ``poc stop``. Earlier versions silently kept only the last
  value when an option was repeated.
- ``poc stop -ex <participant>`` now leaves each excluded participant running;
  earlier versions ignored exclusions on the default coordinated shutdown path.

The default-start behavior is a documentation/help clarification. Preserving
repeated participant options is new CLI behavior on ``main``.

Upgrading from 2.7.0/2.7.1 to 2.7.2
======================================

Recipe API Changes
------------------

**initial_model renamed to model**

The ``initial_model`` parameter in all recipes has been renamed to ``model`` for clarity:

.. code-block:: python

    # Before (2.7.0/2.7.1)
    recipe = FedAvgRecipe(
        ...
        initial_model=SimpleNetwork(),
    )

    # After (2.7.2)
    recipe = FedAvgRecipe(
        ...
        model=SimpleNetwork(),
    )

The ``model`` parameter now also accepts dict-based configuration with optional pretrained checkpoint:

.. code-block:: python

    recipe = FedAvgRecipe(
        ...
        model={"path": "my_module.MyModel", "args": {"hidden_size": 256}},
        initial_ckpt="pretrained.pt",
    )

**PTFedAvgEarlyStopping merged into PTFedAvg**

``PTFedAvgEarlyStopping`` has been merged into ``PTFedAvg`` with InTime aggregation support.
A backward-compatible alias is provided, but new code should use ``PTFedAvg``:

.. code-block:: python

    # Before
    from nvflare.app_opt.pt.fedavg_early_stopping import PTFedAvgEarlyStopping
    controller = PTFedAvgEarlyStopping(...)

    # After
    from nvflare.app_opt.pt.fedavg import PTFedAvg
    controller = PTFedAvg(...)

MONAI Integration
------------------

The separate ``nvflare-monai`` wheel package is deprecated. Use the Client API directly
for MONAI integration. See the updated examples in ``examples/advanced/monai/`` and the
`MONAI Migration Guide <https://github.com/NVIDIA/NVFlare/blob/main/integration/monai/MIGRATION.md>`_.

New Features (No Migration Required)
--------------------------------------

The following 2.7.2 features work automatically with no code changes:

- **TensorDownloader**: Transparent memory optimization for PyTorch model weight transfer.
  See :ref:`tensor_downloader`.
- **Server-side memory cleanup**: Automatic garbage collection and heap trimming.
  See :doc:`/programming_guide/memory_management`.

Backward Compatibility
-----------------------

- **Job Config API**: Existing ``FedJob``-based configurations continue to work alongside the new Recipe API.
- **Config-based Jobs**: JSON/YAML configuration-based jobs continue to work as before.
- **Executor/ModelLearner APIs**: Still functional but no longer the recommended pattern. Use Recipe API + Client API for new projects.

For the full list of changes, see the :doc:`What's New in 2.7.2 </release_notes/flare_272>` release notes.

Upgrading from 2.5/2.6 to 2.7
================================

FLARE 2.7.0 introduced several major changes:

- **Job Recipe API**: A higher-level API for creating FL jobs. See :ref:`job_recipe`.
- **Client API** is now the recommended pattern for all new FL jobs.
- **Hierarchical FL**: New relay-based communication hierarchy for large-scale deployments.
  See :ref:`flare_hierarchical_architecture`.
- **Edge & Mobile**: Federated training on mobile devices (iOS/Android) with ExecuTorch.
  See :ref:`mobile_training`.
- **File Streaming**: Pull-based file download for large model transfers.
  See :ref:`file_streaming`.

For migrating from the older FLAdminAPI to the Client API, see :doc:`Migrating to FLARE API </programming_guide/migrating_to_flare_api>`.

For the full list of 2.7.0 changes, see :doc:`/release_notes/flare_270`.
