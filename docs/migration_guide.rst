.. _migration_guide:

################
Migration Guide
################

This guide covers API and configuration changes when upgrading between FLARE releases.

Upcoming Main-Branch Changes
============================

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

For whole local PoC lifecycle control, use the PoC start/stop flow instead of
the general system admin API.

CLI Startup Kit Resolution Change
---------------------------------

On the current ``main`` branch, the ``NVFLARE_STARTUP_KIT_DIR`` environment
variable now takes precedence over the persisted CLI config when resolving the
startup kit for server-connected CLI commands.

Impact:

- If both ``NVFLARE_STARTUP_KIT_DIR`` and the CLI config specify startup kit
  paths, the environment variable wins.
- Shell profiles that export ``NVFLARE_STARTUP_KIT_DIR`` may override
  ``poc.startup_kit`` or ``prod.startup_kit`` from ``~/.nvflare/config.conf``.

If you rely on the persisted CLI config, review your shell environment before
upgrading to the next release built from ``main``.

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

POC Start Default Service Clarification
---------------------------------------

On the current ``main`` branch, the documented default behavior of
``nvflare poc start`` is clarified to reflect the actual runtime behavior:
the default start set is the server plus client services, not every
participant directory under the workspace.

Impact:

- Running ``nvflare poc start`` with no explicit ``-p`` / ``--service`` starts
  the server and clients.
- Admin consoles are not started unless explicitly selected.

This is a documentation/help clarification, not a runtime behavior change.

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

- **Job Recipe API** (technical preview): A higher-level API for creating FL jobs. See :ref:`job_recipe`.
- **Client API** is now the recommended pattern for all new FL jobs.
- **Hierarchical FL**: New relay-based communication hierarchy for large-scale deployments.
  See :ref:`flare_hierarchical_architecture`.
- **Edge & Mobile**: Federated training on mobile devices (iOS/Android) with ExecuTorch.
  See :ref:`mobile_training`.
- **File Streaming**: Pull-based file download for large model transfers.
  See :ref:`file_streaming`.

For migrating from the older FLAdminAPI to the Client API, see :doc:`Migrating to FLARE API </programming_guide/migrating_to_flare_api>`.

For the full list of 2.7.0 changes, see :doc:`/release_notes/flare_270`.
