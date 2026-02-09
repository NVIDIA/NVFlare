.. _migration_guide:

################
Migration Guide
################

This guide covers API and configuration changes when upgrading between FLARE releases.

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

For migrating from the older FLARE API to the Client API, see :ref:`migrating_to_flare_api`.

For the full list of 2.7.0 changes, see :doc:`/release_notes/flare_270`.
