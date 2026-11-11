.. _migration_guide:

################
Migration Guide
################

This guide covers API and configuration changes when upgrading between FLARE releases.

Upgrading from 2.9.0 to 2.9.1
=============================

Review the changes below before upgrading.

**Required actions**, if your jobs, deployments, or custom integrations use the
affected feature:

- Removed tensor-streaming APIs and their timeout settings. See
  :ref:`tensor_transfer_migration_291`.
- Control-request authorization for custom code that sends internal CellNet
  control messages. Supported administration and job APIs need no changes.
- Client-log staging paths for custom log collectors and storage providers.
- Checkpoint path restrictions for ``PTFileModelPersistor`` and the Dashboard
  client API change.
- Updated security dependencies: refresh locked environments and package
  mirrors.
- For confidential computing, provisioning changes and the rejection of
  launcher-defined bring-your-own-code (BYOC) jobs. Azure Container Instances
  (ACI) deployments also require an approved policy and signed external kit.

**Behavior changes and fixes** that need no action unless noted:

- Site component authorization and initialization failures in multiprocess
  sub-workers can now stop a job.
- In-time FedAvg stops on an accumulation exception and keeps the previous
  checkpoint.
- Client-log sender checks and deletion recovery are stricter, and recipe
  client log streaming in server jobs works again.
- Configuration-format deprecations (HOCON and YAML): plan to move to JSON
  before 2.10.0.

.. _tensor_transfer_migration_291:

Migrate Push-Based Tensor Streaming to Tensor Downloader
--------------------------------------------------------

The ``nvflare.app_opt.tensor_stream`` package, ``TensorServerStreamer``,
``TensorClientStreamer``, ``Recipe.enable_tensor_streaming()``, and the
``examples/advanced/tensor-stream`` example are removed in 2.9.1. Existing
exported jobs referencing those classes cannot load them on 2.9.1.

Nothing replaces the enable call. For standard PyTorch recipes, delete
``recipe.enable_tensor_streaming(...)``: the Tensor Downloader (see
:ref:`tensor_downloader`) is used automatically, with no per-recipe or per-task
opt-in.

.. code-block:: python

    # 2.9.0
    recipe = FedAvgRecipe(
        name="my-job",
        min_clients=2,
        num_rounds=10,
        model=MyLargeModel(),
        train_script="client.py",
        server_expected_format=ExchangeFormat.PYTORCH,
    )
    recipe.enable_tensor_streaming(
        format="pytorch",
        tasks=["train"],
        tensor_send_timeout=30.0,
        wait_send_task_data_all_clients_timeout=300.0,
    )

    # 2.9.1: delete the enable call. Keep server_expected_format only if you
    # want PyTorch tensors on the wire; see "Memory-saving use cases" below.
    recipe = FedAvgRecipe(
        name="my-job",
        min_clients=2,
        num_rounds=10,
        model=MyLargeModel(),
        train_script="client.py",
        server_expected_format=ExchangeFormat.PYTORCH,
    )

Jobs that were already exported contain the streamer components
(``nvflare.app_opt.tensor_stream.server.TensorServerStreamer`` on the server and
``nvflare.app_opt.tensor_stream.client.TensorClientStreamer`` on the clients).
Remove them from the server and client configurations, then re-export and
resubmit the job.

Old use case and migration action
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Previous use
     - 2.9.1 action
   * - ``recipe.enable_tensor_streaming(...)``
     - Delete the call. There is no replacement.
   * - ``TensorServerStreamer`` or ``TensorClientStreamer`` in an exported job
     - Remove the components, then re-export and resubmit.
   * - ``format="pytorch"``
     - Set ``server_expected_format=ExchangeFormat.PYTORCH`` on the recipe. For
       direct Client API configuration, set both formats (see below).
   * - ``tasks=[...]``
     - No per-task option. See "Scope" below.
   * - Reduce server memory for large models
     - Keep the PyTorch wire format and set
       ``enable_tensor_disk_offload=True`` (see below).
   * - ``tensor_send_timeout`` or ``wait_send_task_data_all_clients_timeout``
     - Remove. There is no one-to-one mapping (see below).
   * - ``get_task_timeout`` overrides added to satisfy the streamer
     - Remove them if they exist only for that coupling.
   * - ``examples/advanced/tensor-stream`` (GPT-2)
     - No direct replacement. Use the
       :github_nvflare_link:`Hello HuggingFace example
       <examples/hello-world/hello-huggingface>`, which uses PyTorch tensors
       with ``enable_tensor_disk_offload=True`` for large models.

Memory-saving use cases
^^^^^^^^^^^^^^^^^^^^^^^

Two separate mechanisms cover what tensor streaming was used for:

- The automatic Tensor Downloader reduces memory during transfer, because
  tensors are pulled in chunks and the payload is not serialized into one
  buffer.
- ``enable_tensor_disk_offload=True`` additionally lowers server aggregation
  memory: incoming tensors are staged in temporary files and materialized
  lazily. It is available on the PyTorch recipes that expose the argument
  (for example FedAvg, FedProx, Scaffold, FedCE, and Swarm) and applies only to
  PyTorch tensors. It needs ``server_expected_format=ExchangeFormat.PYTORCH``;
  with the default ``NUMPY`` format the recipes warn that nothing is offloaded.

.. code-block:: python

    from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
    from nvflare.client.config import ExchangeFormat

    recipe = FedAvgRecipe(
        name="my-job",
        min_clients=2,
        num_rounds=10,
        model=MyLargeModel(),
        train_script="client.py",
        server_expected_format=ExchangeFormat.PYTORCH,  # keep tensors as PyTorch on the wire
        enable_tensor_disk_offload=True,  # stage incoming tensors on the server's disk
    )

For direct ``ClientAPIExecutor`` or ``ScriptRunner`` configuration,
``params_exchange_format`` is the representation the training script sees from
``flare.receive()`` and passes to ``flare.send()``, and ``server_expected_format``
is what the server expects. The PyTorch tensor decomposer is registered for the
wire only when ``server_expected_format`` is ``"pytorch"`` (or the pair is
``"raw"`` with ``"pytorch"``). Setting only ``params_exchange_format="pytorch"``
leaves ``server_expected_format`` at its default ``"numpy"``, so tensors are
converted to NumPy on the wire. Set both to ``"pytorch"`` to keep tensors.

Timeout behavior
^^^^^^^^^^^^^^^^

The old push streamer made the server wait until every client had received the
task tensors, and advertised a minimum ``get_task_timeout`` of
``wait_send_task_data_all_clients_timeout + 60`` seconds. Both the all-clients
barrier and that automatic minimum are gone. Remove ``get_task_timeout`` or
related overrides that you added only for that coupling.

Other waits still apply and must be sized for the actual delivery path: the
Tensor Downloader's per-request and minimum download timeouts, and the Client
API ``task_wait_timeout`` and ``result_wait_timeout``. There is no one-to-one
mapping from ``tensor_send_timeout`` or
``wait_send_task_data_all_clients_timeout``; do not copy the old values assuming
the same semantics. See :doc:`/programming_guide/timeouts` and the
:ref:`tensor_downloader` configuration section, which says which process reads
each setting.

Scope and custom integrations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There is no per-task ``tasks=`` opt-in and no ``format=`` argument. Downloader
behavior follows serialization: any task payload that contains a type with a
registered download decomposer is transferred by download. PyTorch tensors use
``TensorDecomposer`` (settings prefixed ``tensor_``) and NumPy arrays use the
built-in ``NumpyArrayDecomposer`` (settings prefixed ``np_``). TensorFlow-specific
payloads use normal serialization.

Standard PyTorch recipes, ``PTFedAvg``, ``PTFileModelPersistor``, and Client API
jobs that declare the PyTorch wire format register ``TensorDecomposer``
themselves. A custom job or component that sends or receives PyTorch tensors
outside those paths must register it in every process that serializes or
deserializes them, typically from the component's initialization:

.. code-block:: python

    from nvflare.app_opt.pt.decomposers import TensorDecomposer
    from nvflare.fuel.utils import fobs

    fobs.register(TensorDecomposer)

Register it in the server job process (for a custom controller or persistor) and
in each client process that handles the tensors (the executor, or the external
trainer process). See the Low-Level API section of :ref:`tensor_downloader` for
custom decomposers.

Control Requests Must Follow Authenticated Caller and Job Boundaries
--------------------------------------------------------------------

2.9.1 checks control requests against the transport endpoint, upstream route,
and job identity. Supported administration and job APIs need no call-site
changes. Custom code that sends internal CellNet control messages must obey
these restrictions:

- Client administration originates from the server through the configured
  upstream path. A client job may notify only its own parent of its status.
  Client-job commands originate from the server or the job's parent.
- Server-job controls originate from the parent server. Client traffic
  forwarded through that parent does not gain server-control authority.
- Full participant data and run-status updates on the server-parent channel
  are restricted to the matching server job. A same-job client recovery query
  receives an empty participant map when the server job exists, or ``None``
  when it does not; it no longer receives participant credentials. Cross-job
  queries are rejected.
- Server-parent ``HEARTBEAT`` requests are accepted only from the matching
  server job or the simulator's own parent-server heartbeat. Other identities
  receive ``UNAUTHENTICATED``.
- Server-job and workspace-transfer credentials are scoped to their matching
  job identities. Changing claimed origin headers does not grant access to
  another job or to the parent-server identity.

Action: use supported admin and job APIs for lifecycle operations, and update
custom internal integrations or test harnesses that relied on peer-supplied
identity headers, cross-job access, or participant tokens returned to a client.
Upgrade the server, clients, relays, and job runtime images to receive the
checks at each affected boundary.

Updated Security Dependencies
------------------------------

2.9.1 requires ``certifi>=2026.10.19`` and ``uv>=0.13.6,<0.14.0`` following
the eMudhra/emSign CA incident. Normal dependency resolution applies these
requirements when installing or upgrading NVFlare.

Action: update dependency locks, constraints, offline mirrors, and container
images to satisfy both requirements, including environments previously built
without ``uv``. Do not retain older versions through a dependency-free upgrade.

.. _configuration_format_migration_291:

HOCON and OmegaConf Configuration Deprecation
---------------------------------------------

HOCON/PyHOCON and OmegaConf-based runtime and job configurations are deprecated
in 2.9.1. They remain supported in 2.9.1 and will be removed in 2.10.0. Use JSON
for new configurations and migrate existing files before upgrading to 2.10.0.
This applies to ``.conf``, ``.yml``, and ``.yaml`` files loaded by these
configuration loaders, including their ``.default`` variants.

Action:

- Convert runtime and job configuration contents to valid JSON. For example,
  migrate ``meta.conf`` to ``meta.json`` and ``config_fed_client.yaml`` to
  ``config_fed_client.json``.
- Replace HOCON includes/substitutions and OmegaConf interpolations/resolvers
  with explicit, reviewed JSON values. Renaming the extension alone does not
  convert the contents or these format-specific features.
- Update references to the migrated files, validate the resulting configuration,
  and remove obsolete copies so configuration lookup selects the intended file.
  For ``.default`` templates, use the corresponding ``.json.default`` name.

Provisioning YAML files, including ``project.yml``, ``cc_project.yml``, and
participant ``cc_config`` files, use separate provisioning loaders and are
outside this deprecation. See :ref:`configurations` for the configuration
formats and search order.

Confidential Computing Provisioning (Technical Preview)
-------------------------------------------------------

Confidential computing was introduced as a technical preview in 2.7.0, with
its own provisioning syntax and a HashiCorp Vault plus Trustee KBS key backend
(also present in 2.8 and 2.9.0). In 2.9.1 all confidential deployments use one
unified provisioning interface: a participant selects ``cc_deployment_mode``
(``bare_metal_cvm``, ``coco``, or ``azure_cc``) in its ``cc_config``, and the
project supplies a shared ``cc_project_config``. On-premises CVMs are built
with CVM Builder, which stores binding-addressed vault keys directly in Trustee
resource storage, so HashiCorp Vault is no longer part of the key path.

Impact:

- The earlier on-premises provisioning syntax and packager entry points are
  rejected; there is no compatibility mode. Convert the whole project and its
  examples together. The ``examples/advanced/cc_provision`` example is
  replaced by ``examples/advanced/cvm_builder``.
- Existing HashiCorp Vault secrets, measurements and deliveries are not
  imported. There is no in-place upgrade of an existing deployment; build
  fresh.
- Before switching a project to ``bare_metal_cvm``, deploy the supported
  upstream Trustee, publish the approved generic CVM references and policies,
  and build fresh application vaults.
- Plan to rebuild and redeliver CVMs. The old and new key backends are not
  wire- or storage-compatible, and 2.9.1 does not import 2.9.0 deliveries.

See :ref:`cc_deployment` for the provisioning configuration, :ref:`cvm_builder`
for bare-metal CVMs, :ref:`cc_architecture` for the resulting trust model, and
the :github_nvflare_link:`Trustee deployment guide
<nvflare/lighter/cc/image_builder/TRUSTEE_GUIDE.md>` for the current backend and
the migration boundary.

.. _aci_migration_291:

Azure Container Instances Require an Approved Policy and Signed External Kit
----------------------------------------------------------------------------

For ``azure_cc`` with ``deployment_target: confidential_container``, 2.9.1
supports one protected ACI server with confidential-VM or Confidential
Containers clients. Existing ACI deployments must be reprovisioned. Evidence
signature validation alone no longer suffices: the signed policy fingerprint
must match the exact CCE policy approved during provisioning, and the evidence
must identify the supported hardware, Azure compliant container environment,
and debugging disabled.

Action:

1. Create a separate deployment-signing key and build the fixed server image
   with its public key and the startup verification program. Keep the final
   startup kit outside the policy-measured image.
2. Generate and decode the CCE policy for that image. Set ``cce_policy_file``
   and ``startup_kit_signing_key_file`` in the participant's ``azure_cc`` block;
   relative paths resolve against the participant configuration file.
3. Reprovision the project. This approves the exact policy bytes and configures
   every ACI verifier with their SHA-256 fingerprint (``expected_policy_hash``).
   Redistribute the affected kits so peers check the same approved policy.
4. Deploy with the packaged policy, ``startup-kit.tar``, and ``startup-kit.sig``.
   Deliver the archive and signature through the read-only Azure secret volume
   using private deployment parameters. Do not rebuild the image or regenerate
   the policy after provisioning.

See :ref:`confidential_azure_container_instances_deployment` for the complete
workflow and positive and negative deployment checks. A policy change requires
reprovisioning and redistribution; a deployment-signing key change also requires
rebuilding the fixed image and regenerating its policy. Previously signed
archives remain valid: reprovisioning alone does not revoke credentials or old
archives. Follow the guide's recovery procedure after a compromise.

This change does not alter confidential-VM configuration. The separate legacy
``AZCVMAuthorizer`` still does not require a statement identifying the evidence
as an Azure compliant confidential VM; see the deployment guide for that
limitation and the distinction from ACI and CoCo verification.

Confidential Computing Job Policy
---------------------------------

Launcher-defined bring-your-own-code (BYOC) content, meaning executable content
in Docker, Kubernetes, and Slurm launcher specifications, is now classified as
BYOC at submission time and rejected in confidential computing, for every confidential computing role.
Package the code in the approved workload image and add only exact, reviewed
classes to ``class_allow_list``.

PyTorch Checkpoint Paths Must Stay Within the Application
---------------------------------------------------------

``PTFileModelPersistor`` now confines checkpoint output paths in every deployment,
including deployments that do not use confidential computing. The checkpoint
directory (``APP_CKPT_DIR`` in ``config/environment.json`` or ``AppConstants.LOG_DIR``
in the FL context) must be relative to the application directory and resolve
within it. ``global_model_file_name`` and ``best_global_model_file_name`` must
be relative to the checkpoint directory and must not traverse above it.
Symlinks that resolve outside the application directory are also rejected.
Invalid settings stop job initialization with ``Invalid checkpoint path
configuration``.

Action: replace absolute paths and paths that escape these directories with
relative paths inside the application. For example, change
``global_model_file_name: /shared/model.pt`` to
``global_model_file_name: model.pt`` and set ``APP_CKPT_DIR`` to
``checkpoints`` to save under ``<application directory>/checkpoints/model.pt``.
An external symlink is not a workaround; copy or export checkpoints separately
if they are needed outside the application directory. Default checkpoint paths
require no change. These restrictions concern checkpoint output paths, not the
explicit ``source_ckpt_file_full_name`` input used to load an initial model.

Dashboard Client ``props`` Are Rejected
---------------------------------------

Dashboard client creation and PATCH requests that contain ``props`` are
rejected with HTTP 400, and client-kit generation ignores previously stored
``props``. The Dashboard UI's client name and capacity requests are not
affected.

Action: API integrations that send ``props`` must omit it.

Aggregation, Multiprocess Execution, and Log Retention
-------------------------------------------------------

In-time FedAvg now terminates the workflow on an accumulation exception instead
of saving a model that may contain part of a rejected contribution. Investigate
the failed contribution and restart from the previous valid checkpoint rather
than expecting the round to continue after this error.

Multiprocess sub-workers now initialize the required services and consult the
parent site's existing component authorization callback for runtime builds.
Custom components must satisfy that site policy; unavailable authorization or
initialization failures stop execution.

When ``JobLogReceiver.dest_dir`` is unset, default client-log staging moves
from the system temporary directory in 2.9.0 to
``{workspace.get_log_root()}/.client_log_streams/{job_id}/{client_name}/{log_file_name}``
in 2.9.1. Update scripts that tail or collect live logs from the old temporary
location. Without a workspace, the receiver uses a private temporary directory.
An explicit ``dest_dir`` continues to select
``{dest_dir}/{job_id}/{client_name}/{log_file_name}``; its retention remains
operator-managed.

2.9.1 also requires the stream owner to match the authenticated sender and
rejects invalid path components and symlinks. Custom log producers must retain
the authenticated identity through dispatch; peer-supplied identity fields do
not establish ownership.

Default managed staging retains active logs across job cleanup. Explicit job
deletion now records durable deletion intent before changing the job store,
removes closed managed logs, and lets active streams drain before discarding
them. Recovery checks authoritative job presence: a confirmed existing job
clears interrupted deletion intent and permits uploads, confirmed absence
permits cleanup, and unavailable checks preserve both logs and intent. Custom
job managers and storage providers need authoritative ``job_exists`` and
``object_exists`` support for automatic recovery. See
:doc:`/programming_guide/live_log_streaming`.

Examples
--------

The Hello World examples and their requirements are updated for NVFlare 2.9.1.
Reinstall an example's requirements if you created its environment for 2.9.0.

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
  ``concise``, ``msg_only``, ``full``, ``verbose``, and ``reload``.

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
