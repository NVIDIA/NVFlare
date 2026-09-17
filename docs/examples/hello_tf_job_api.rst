.. _hello_tf_job_api:

Hello TensorFlow with Job API
==============================

This example uses TensorFlow/Keras and NVIDIA FLARE's Job and Client APIs to
train an MNIST classifier with federated averaging. The runnable source is in
the :github_nvflare_link:`hello-tf directory <examples/hello-world/hello-tf/>`.

The setup has one server and two clients. In each round, every client evaluates
the received global model, trains it on its local half of MNIST, and sends its
updated layer weights and accuracy to the server. The server aggregates the
updates with :class:`FedAvg<nvflare.app_common.workflows.fedavg.FedAvg>`.

Running the Example
-------------------

From the repository root, create an environment, install the example
requirements, and run the job recipe:

.. code-block:: shell

   $ cd examples/hello-world/hello-tf
   $ python3 -m pip install -r requirements.txt
   $ python3 job.py

Use ``--n_clients`` and ``--num_rounds`` to change the defaults:

.. code-block:: shell

   $ python3 job.py --n_clients 3 --num_rounds 5

The recipe executes with :class:`SimEnv<nvflare.recipe.sim_env.SimEnv>`. Results
and logs are written under ``/tmp/nvflare/simulation`` by default.

Job Recipe
----------

The example's ``job.py`` constructs a TensorFlow
:class:`FedAvgRecipe<nvflare.app_opt.tf.recipes.fedavg.FedAvgRecipe>` with the
actual model object and client training script:

.. literalinclude:: ../../examples/hello-world/hello-tf/job.py
   :language: python
   :linenos:
   :lines: 14-

The important inputs are ``model=Net()`` and ``train_script="client.py"``.
``add_experiment_tracking`` also adds TensorBoard event handling to the job.

Model
-----

``model.py`` defines the Keras ``Net`` class used to initialize and persist the
global model:

.. literalinclude:: ../../examples/hello-world/hello-tf/model.py
   :language: python
   :linenos:
   :lines: 14-

Client Training
---------------

``client.py`` is ordinary TensorFlow training code with a small Client API loop:

.. literalinclude:: ../../examples/hello-world/hello-tf/client.py
   :language: python
   :linenos:
   :lines: 14-

The essential Client API calls are:

- ``flare.init()`` to initialize the trainer in the Client Job process.
- ``flare.receive()`` to receive the current global model.
- ``flare.send()`` to return updated weights and metrics.

Generated Application Configuration
-----------------------------------

When the recipe is exported, the server persistor refers to the model that the
example actually supplies in ``model.py``:

.. code-block:: json

   {
     "id": "persistor",
     "path": "nvflare.app_opt.tf.model_persistor.TFModelPersistor",
     "args": {
       "model": {
         "path": "model.Net",
         "args": {}
       }
     }
   }

The generated client configuration uses the unified
``ClientAPIExecutor`` in ``in_process`` mode and points to the actual bundled
training file, ``client.py``:

.. code-block:: json

   {
     "tasks": ["*"],
     "executor": {
       "path": "nvflare.app_common.executors.client_api_executor.ClientAPIExecutor",
       "args": {
         "execution_mode": "in_process",
         "task_script_path": "client.py",
         "params_exchange_format": "keras_layer_weights",
         "server_expected_format": "numpy"
       }
     }
   }

The trainer runs inside the Client Job process in this example. Use
:ref:`client_api_attach` only when the trainer process is started and owned
independently of NVFLARE.

Notes on Running with GPUs
--------------------------

TensorFlow may allocate most available GPU memory at startup. When simulating
multiple clients on one host, enable memory growth and asynchronous allocation:

.. code-block:: shell

   $ TF_FORCE_GPU_ALLOW_GROWTH=true TF_GPU_ALLOCATOR=cuda_malloc_async python3 job.py

For GPU environments, the
`NVIDIA TensorFlow container <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow>`_
is recommended.

Previous Versions of Hello TensorFlow
-------------------------------------

- `2.0 <https://github.com/NVIDIA/NVFlare/tree/2.0/examples/hello-tf2>`_
- `2.1 <https://github.com/NVIDIA/NVFlare/tree/2.1/examples/hello-tf2>`_
- `2.2 <https://github.com/NVIDIA/NVFlare/tree/2.2/examples/hello-tf2>`_
- `2.3 <https://github.com/NVIDIA/NVFlare/tree/2.3/examples/hello-world/hello-tf2>`_
- `2.4 <https://github.com/NVIDIA/NVFlare/tree/2.4/examples/hello-world/hello-tf2>`_
