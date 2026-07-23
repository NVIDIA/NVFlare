.. _hf_client_api:

##########################
HuggingFace Client API
##########################

The HuggingFace Client API lets you federate an existing HuggingFace
``Trainer`` or TRL ``SFTTrainer`` script without manually calling
``flare.receive()`` and ``flare.send()``. Import ``nvflare.client.hf``,
patch the trainer, and keep the familiar ``trainer.evaluate()`` and
``trainer.train()`` calls in the training loop.

Use this API when your local training script already uses HuggingFace
``Trainer`` style code and you want FLARE to handle the FL task exchange,
global-weight loading, local-budget stopping, checkpoint state, and
rank-0 communication.

Minimal Client Change
=====================

Start from a normal HuggingFace trainer script. After constructing the
trainer, call ``flare.patch(trainer)`` and run one evaluate/train pair per
FL round:

.. code-block:: python

    import nvflare.client.hf as flare

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    flare.patch(trainer)

    while flare.is_running():
        trainer.evaluate()
        trainer.train()

``nvflare.client.hf`` re-exports the standard Client API symbols, so the
same module can be used for ``FLModel``, ``get_job_id()``, ``log()``, and
other Client API calls. Its ``is_running()`` implementation is HuggingFace
aware and coordinates with the patched trainer state.

What ``patch()`` Does
=====================

``flare.patch(trainer)`` wraps the trainer methods and registers FLARE
callbacks. During each FL round it:

* initializes the FLARE Client API with the global process rank;
* receives the FL task and global model on rank 0;
* broadcasts task metadata and parameters to other distributed ranks;
* loads received parameters into the HuggingFace model;
* limits each local round to the configured local step or epoch budget;
* restores Trainer checkpoint state between rounds when enabled;
* captures evaluation metrics for server-side model selection;
* sends trained parameters and metrics back to FLARE from rank 0.

The training script still owns model construction, dataset loading,
``TrainingArguments``/``SFTConfig``, optimizers, schedulers, and any normal
HuggingFace callbacks.

Patch Options
=============

The most common options are:

``restore_state``
   Whether to restore HuggingFace Trainer state across FL rounds. This is
   enabled by default so optimizer, scheduler, and global-step state continue
   across rounds. When enabled, ``save_only_model=True`` is not supported.

``params_scope``
   Which model parameters to exchange. ``"auto"`` is the default. It uses
   adapter-only parameters for PEFT ``PeftModel`` trainers and full-model
   parameters otherwise. Use ``"model"`` to force full-model exchange or
   ``"adapter"`` to require PEFT adapter exchange.

``server_key_prefix``
   Optional key prefix to strip from server parameters when loading into the
   trainer, and to add back when sending results. Use this when the server
   model wrapper has a different state-dict namespace than the trainer model.

``local_epochs`` / ``local_steps``
   Optional explicit local training budget. Specify at most one. If neither
   is set, the wrapper uses ``TrainingArguments.max_steps`` when positive;
   otherwise it converts ``TrainingArguments.num_train_epochs`` to optimizer
   steps from the trainer dataloader.

``load_state_dict_strict``
   Whether incoming parameters must match the local model keyspace strictly.
   Keep this enabled unless you intentionally expect partial parameter
   exchange.

``stream_metrics``
   Whether to stream finite scalar HuggingFace logs through the FLARE
   tracking API on rank 0.

Example:

.. code-block:: python

    flare.patch(
        trainer,
        params_scope="auto",
        server_key_prefix="model.",
        local_epochs=1,
        stream_metrics=True,
    )

Metrics And Model Selection
===========================

Call ``trainer.evaluate()`` before ``trainer.train()`` when the server should
receive validation metrics before local training:

.. code-block:: python

    while flare.is_running():
        metrics = trainer.evaluate()
        trainer.train()

The HuggingFace Client API reports the metrics returned by the trainer. Configure
the server-side recipe or selector with a higher-is-better metric key. If you do
not want the recipe to select a best model, set ``key_metric=""``:

.. code-block:: python

    from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe

    recipe = FedAvgRecipe(
        name="hf_sft",
        model=model,
        min_clients=2,
        num_rounds=3,
        train_script="client.py",
        launch_external_process=True,
        key_metric="",
    )

If the server sends an evaluation task, ``trainer.evaluate()`` sends metrics
without running training. If the server requests ``submit_model``, either
``trainer.evaluate()`` or ``trainer.train()`` can trigger submission of the
last completed checkpoint, falling back to the current in-memory model.

Parameter Scope And Key Prefixes
================================

Use ``params_scope="auto"`` for most jobs:

* Full-model SFT uses full model parameters.
* PEFT/LoRA jobs using a ``PeftModel`` use adapter parameters only.

Use ``server_key_prefix`` only when the server-side model and trainer model
state-dict keys differ. For example, if a server wrapper stores the base model
under ``self.model``, server keys may look like ``model.transformer...`` while
the local trainer keys omit ``model.``. In that case:

.. code-block:: python

    flare.patch(trainer, server_key_prefix="model.")

For PEFT jobs where the server model already exposes adapter-shaped keys,
leave the prefix unset:

.. code-block:: python

    flare.patch(trainer, params_scope="auto", server_key_prefix=None)

Job Recipe Setup
================

Use a recipe to package the trainer script and configure the Client API
executor. LLM jobs commonly run in an external process so ``torchrun``,
``accelerate``, CUDA, and other training runtime settings stay isolated from
the FLARE client job process.

.. code-block:: python

    from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
    from nvflare.client.config import ExchangeFormat
    from nvflare.recipe import SimEnv

    recipe = FedAvgRecipe(
        name="hf_sft",
        model={"class_path": "model.CausalLMModel", "args": {"model_name_or_path": "gpt2"}},
        min_clients=2,
        num_rounds=3,
        train_script="client.py",
        train_args="--model_name_or_path gpt2 --local_epoch 1",
        server_expected_format=ExchangeFormat.PYTORCH,
        launch_external_process=True,
        key_metric="",
    )

    env = SimEnv(num_clients=2)
    recipe.execute(env)

Use ``server_expected_format=ExchangeFormat.PYTORCH`` to keep PyTorch tensors
and preserve tensor dtypes such as ``bfloat16``. Use
``ExchangeFormat.NUMPY`` when the server-side workflow expects NumPy arrays;
half-precision tensors are cast to ``float32`` before they are sent to a
NumPy server.

Distributed Training
====================

For multi-GPU or multi-node HuggingFace training, launch the script with
``torchrun`` or another launcher that initializes ``torch.distributed`` and
sets global ``RANK``/``WORLD_SIZE``.

All ranks must call the same patched Trainer methods in the same order:

.. code-block:: python

    while flare.is_running():
        trainer.evaluate()
        trainer.train()

Rank 0 is the only process that calls the FLARE Client API receive/send path.
Other ranks receive task data through ``torch.distributed`` broadcast.
If one rank calls ``trainer.evaluate()`` while another calls
``trainer.train()`` for the same task, the wrapper raises an error instead of
allowing a distributed deadlock.

For large distributed parameter payloads, the wrapper may stage rank-0
parameters under ``<output_dir>/_fl_exchange`` and have the other ranks read
that file. In multi-node jobs, make sure ``TrainingArguments.output_dir`` is
on shared storage. If shared storage is not available, force object broadcast:

.. code-block:: bash

    export NVFLARE_HF_PARAMS_EXCHANGE_STRATEGY=object

You can also force file staging with ``NVFLARE_HF_PARAMS_EXCHANGE_STRATEGY=file``.
The automatic threshold is controlled by
``NVFLARE_HF_PARAMS_FILE_EXCHANGE_MIN_BYTES``.

Checkpoint Behavior
===================

With ``restore_state=True``:

* the wrapper resumes from the last FLARE-recorded HuggingFace checkpoint for
  later rounds;
* ``TrainingArguments.save_total_limit`` is set to ``2`` when it is unset, so
  the current and previous FL checkpoints can be retained;
* ``load_best_model_at_end=True`` is rejected because global best-model
  selection belongs to the FL server;
* ``save_only_model=True`` is rejected because optimizer and scheduler state
  are needed for Trainer resume.

If you pass your own ``resume_from_checkpoint`` to ``trainer.train()``, FLARE
uses that checkpoint for the train call and applies received global weights in
memory instead of mutating the user-provided checkpoint.

Unsupported Configurations
==========================

The first HuggingFace Client API implementation intentionally does not support:

* DeepSpeed;
* FSDP;
* ``load_best_model_at_end=True``;
* ``save_only_model=True`` with ``restore_state=True``;
* more than one patched HuggingFace ``Trainer`` in the same Python process.

Troubleshooting
===============

``Previous HuggingFace FL task ... is still pending``
   The loop called ``flare.is_running()`` again before completing the current
   task. Call the expected ``trainer.train()`` or ``trainer.evaluate()`` first.

``Divergent HuggingFace Trainer call across ranks``
   Distributed ranks called different Trainer methods. Make every rank execute
   the same evaluate/train sequence.

``rank > 0, but torch.distributed is not initialized``
   A nonzero global rank was detected without an initialized distributed
   process group. Launch distributed jobs with ``torchrun`` or remove the
   stale ``RANK`` environment variable for single-process runs.

``None of the model parameters matched``
   The server and trainer state-dict keyspaces differ. Set the correct
   ``server_key_prefix`` or adjust ``params_scope``.

Complete Example
================

See the full HuggingFace SFT/PEFT example:

* :github_nvflare_link:`examples/advanced/hf_client_api <examples/advanced/hf_client_api>`

For the general Client API concepts, see :ref:`client_api_usage`.
