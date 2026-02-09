.. _mobile_training:

###############################################
Mobile Federated Training (iOS / Android)
###############################################

FLARE 2.7 introduces federated learning on mobile devices (iOS and Android) using
`ExecuTorch <https://github.com/pytorch/executorch>`_. The key advantage:
**no device-side programming is needed** -- you develop your model in standard PyTorch,
and FLARE handles the export, deployment, and federated training orchestration.

How It Works
============

1. **Design your model** in standard PyTorch (keep it lightweight for mobile)
2. **Wrap it** in a ``DeviceModel`` that includes loss and prediction logic for ExecuTorch
3. **Use the ETFedBuffRecipe** to create a FLARE job -- FLARE handles everything else

The mobile SDKs (Android and iOS) communicate with the FLARE server via HTTP,
following the :ref:`Edge Device Interaction Protocol (EDIP) <flare_edge>`.

Step 1 -- Design Model Architecture
------------------------------------

Design your model using PyTorch as you would for single-machine training. Keep in mind
that mobile devices have limited computational resources. Refer to the
`ExecuTorch documentation <https://github.com/pytorch/executorch>`_ for supported layers,
as they may differ from standard PyTorch.

Step 2 -- Create DeviceModel
-----------------------------

ExecuTorch requires the model to return both the loss and predictions during training.
Wrap your model into a ``DeviceModel``:

.. code-block:: python

   from nvflare.edge.models.model import DeviceModel

   class TrainingNet(DeviceModel):
       def __init__(self):
           super().__init__(MyCifar10Net())

The ``DeviceModel`` base class includes ``CrossEntropyLoss`` by default. You can override
the loss function as needed.

Step 3 -- Create FLARE Job with ETFedBuffRecipe
-------------------------------------------------

Use the ``ETFedBuffRecipe`` to create a federated training job for mobile devices:

.. code-block:: python

   recipe = ETFedBuffRecipe(
       job_name=job_name,
       device_model=device_model,
       input_shape=input_shape,
       output_shape=output_shape,
       model_manager_config=ModelManagerConfig(
           max_model_version=3,
           update_timeout=1000.0,
           num_updates_for_model=total_num_of_devices,
       ),
       device_manager_config=DeviceManagerConfig(
           device_selection_size=total_num_of_devices,
           min_hole_to_fill=total_num_of_devices,
       ),
       evaluator_config=evaluator_config,
       simulation_config=(
           SimulationConfig(
               task_processor=task_processor,
               num_devices=num_of_simulated_devices_on_each_leaf,
           )
           if num_of_simulated_devices_on_each_leaf > 0
           else None
       ),
       device_training_params={"epoch": 3, "lr": 0.0001, "batch_size": batch_size},
   )

Key parameters:

- **device_model**: The ``DeviceModel`` wrapper from Step 2
- **input_shape, output_shape**: Tensor shapes for ExecuTorch model export
- **device_training_params**: Training hyperparameters passed to each device

Mobile SDK Guides
=================

For detailed SDK integration and API references:

- :doc:`FLARE Mobile Development Guide <flare_mobile>` -- SDK architecture, setup, and best practices for both Android and iOS
- :doc:`Android SDK API Reference <mobile_android>` -- Kotlin/Java API reference for Android

Examples
========

See the `edge examples <https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/edge>`_
for complete working examples of mobile federated training.

Reference
=========

.. [1] Nguyen, J., Malik, K., Zhan, H., Yousefpour, A., Rabbat, M., Malek, M., & Huba, D. (2023).
   Asynchronous Federated Learning with Bidirectional Quantized Communications and Buffered Aggregation.
   arXiv preprint arXiv:2308.00263. https://arxiv.org/pdf/2308.00263

.. toctree::
   :maxdepth: 1
   :hidden:

   flare_mobile
   mobile_android
