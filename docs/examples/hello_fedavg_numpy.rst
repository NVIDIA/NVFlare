.. _hello_fedavg_numpy:

Hello FedAvg with NumPy
=======================

Before You Start
----------------

Before jumping into this guide, make sure you have an environment with
`NVIDIA FLARE <https://pypi.org/project/nvflare/>`_ installed.

You can follow :ref:`getting_started` on the general concept of setting up a
Python virtual environment (the recommended environment) and how to install NVIDIA FLARE.


Introduction
-------------

This tutorial is meant solely to demonstrate how the NVIDIA FLARE system works, without introducing any actual deep
learning concepts.

Through this exercise, you will learn how to use NVIDIA FLARE with numpy to perform basic
computations across two clients with the included :class:`FedAvg<nvflare.app_common.workflows.fedavg.FedAvg>` workflow,
which sends the model to the clients then aggregates the results that come back.

Due to the simplified weights, you will be able to clearly see and understand
the results of the FL aggregation and the model persistor process.

The setup of this exercise consists of one **server** and two **clients**.
The model is set to the starting weights ``[[1, 2, 3], [4, 5, 6], [7, 8, 9]]``.

The following steps compose one cycle of weight updates, called a **round**:

 #. Clients are responsible for adding a delta to the weights to calculate new weights for the model.
 #. These updates are then sent to the server which will aggregate them to produce a model with new weights.
 #. Finally, the server sends this updated version of the model back to each client, so the clients can continue to calculate the next model weights in future rounds.

For this exercise, we will be working with the ``hello-fedavg-numpy`` in the examples folder.

Let's get started. First clone the repo, if you haven't already:

.. code-block:: shell

  $ git clone https://github.com/NVIDIA/NVFlare.git

Remember to activate your NVIDIA FLARE Python virtual environment from the installation guide.
Ensure numpy is installed.

.. code-block:: shell

  (nvflare-env) $ python3 -m pip install numpy

Now that you have all your dependencies installed, let's look into the ``fedavg_script_executor_hello-numpy.py`` script which
builds the job with the Job API.


NVIDIA FLARE Job API
--------------------

The ``fedavg_script_executor_hello-numpy.py`` script builds the job with the Job API. The following sections are the key lines to focus on:

Define a FedJob
^^^^^^^^^^^^^^^^
:class:`FedJob<nvflare.job_config.api.FedJob>` allows you to generate job configurations in a Pythonic way. It is initialized with the
name for the job, which will also be used as the directory name if the job is exported.

.. code-block:: python

   from nvflare import FedAvg, FedJob, ScriptExecutor

   job = FedJob(name="hello-fedavg-numpy")

Define the Controller Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Define the controller workflow and send to server. We use :class:`FedAvg<nvflare.app_common.workflows.fedavg.FedAvg>` and specify the number of
clients and rounds, then use the :func:`to<nvflare.job_config.api.FedJob.to>` routine to send the component to the server for the job.

.. code-block:: python

   n_clients = 2
   num_rounds = 3

   controller = FedAvg(
      num_clients=n_clients,
      num_rounds=num_rounds,
   )
   job.to(controller, "server")

Add Clients
^^^^^^^^^^^^
Next, we can use the :class:`ScriptExecutor<nvflare.app_common.executors.script_executor.ScriptExecutor>` and send it to each of the
clients to run our training script. We will examine the training script ``hello-numpy_fl.py`` in the next main section.

The :func:`to<nvflare.job_config.api.FedJob.to>` routine sends the component to the specified client for the job. Here, our clients
are named "site-0" and "site-1" and we are using the same training script for both.

.. code-block:: python

   from nvflare.client.config import ExchangeFormat

   train_script = "src/hello-numpy_fl.py"

   for i in range(n_clients):
      executor = ScriptExecutor(
         task_script_path=train_script, task_script_args="", params_exchange_format=ExchangeFormat.NUMPY
      )
      job.to(executor, f"site-{i}")


Optionally Export the Job or Run in Simulator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
With all the components needed for the job, you can export the job to a directory with :func:`export<nvflare.job_config.api.FedJob.export>`
if you want to look at what is built and configured for each client. You can use the exported job to submit it to a real NVFlare deployment
using the :ref:`FLARE Console <operating_nvflare>` or :ref:`flare_api`.

.. code-block:: python

   job.export_job("/tmp/nvflare/jobs/job_config")

This is optional if you just want to run the job in a simulator environment directly, as :class:`FedJob<nvflare.job_config.api.FedJob>` has
a :func:`simulator_run<nvflare.job_config.api.FedJob.simulator_run>` function.

.. code-block:: python

   job.simulator_run("/tmp/nvflare/jobs/workdir")

The results are saved in the specified directory provided as an argument to the :func:`simulator_run<nvflare.job_config.api.FedJob.simulator_run>` function.


NVIDIA FLARE Client Training Script
------------------------------------
The training script ``hello-numpy_fl.py`` is the main script that will be run on the clients. It contains print statements to
help you follow the output while the FL system is running.

On the client side, the training workflow is as follows:

   1. Receive the model from the FL server (for this example we initialize the model in the client code to the numpy array [[1, 2, 3], [4, 5, 6], [7, 8, 9]] if the model params are empty).
   2. Perform training on the received global model and calculate metrics.
   3. Send the new model back to the FL server.

Using NVFlare's Client API, there are three essential methods to help achieve this workflow:

   - `init()`: Initializes NVFlare Client API environment.
   - `receive()`: Receives model from the FL server.
   - `send()`: Sends the model to the FL server.

The following code snippet highlights how these methods are used in the training script:

.. code-block:: python

   import nvflare.client as flare

   flare.init() # 1. Initializes NVFlare Client API environment.
   input_model = flare.receive() # 2. Receives model from the FL server.
   params = input_model.params # 3. Obtain the required information from the received model.

   # original local training code
   new_params = train(params)

   output_model = flare.FLModel(params=new_params) # 4. Put the results in a new `FLModel`
   flare.send(output_model) # 5. Sends the model to the FL server. 

This has been simplified to ignore dealing with data formats to focus on the NVFlare Client API, but you can find the full training
script ``hello-numpy_fl.py`` in the ``src`` directory of :github_nvflare_link:`examples/hello-world/hello-fedavg-numpy <examples/hello-world/hello-fedavg-numpy>`.


Running the Job API Script
---------------------------
Now that you have a good understanding of the training script, you can run the job with the ``fedavg_script_executor_hello-numpy.py`` script:

.. code-block:: shell

   (nvflare-env) $ python3 fedavg_script_executor_hello-numpy.py

This will run the job in a simulator environment and you should be able to see the output as the job proceeds to completion.

You've successfully run your first numpy federated learning system.

You now have a decent grasp of the main FL concepts, and are ready to start exploring how NVIDIA FLARE can be applied to many other tasks.

The full application for this exercise can be found in
:github_nvflare_link:`examples/hello-world/hello-fedavg-numpy <examples/hello-world/hello-fedavg-numpy>`.

Previous Versions of this Example (previously Hello Scatter and Gather)
-----------------------------------------------------------------------

   - `hello-numpy-sag for 2.0 <https://github.com/NVIDIA/NVFlare/tree/2.0/examples/hello-numpy-sag>`_
   - `hello-numpy-sag for 2.1 <https://github.com/NVIDIA/NVFlare/tree/2.1/examples/hello-numpy-sag>`_
   - `hello-numpy-sag for 2.2 <https://github.com/NVIDIA/NVFlare/tree/2.2/examples/hello-numpy-sag>`_
   - `hello-numpy-sag for 2.3 <https://github.com/NVIDIA/NVFlare/tree/2.3/examples/hello-world/hello-numpy-sag>`_
   - `hello-numpy-sag for 2.4 <https://github.com/NVIDIA/NVFlare/tree/2.4/examples/hello-world/hello-numpy-sag>`_
