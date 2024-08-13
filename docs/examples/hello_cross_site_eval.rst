.. _hello_cross_val:

Hello Cross-Site Validation
===========================

Before You Start
----------------

Before jumping into this guide, make sure you have an environment
with `NVIDIA FLARE <https://pypi.org/project/nvflare/>`_ installed.

You can follow :ref:`getting_started` on the general concept of setting up a
Python virtual environment (the recommended environment) and how to install NVIDIA FLARE.

Prerequisite
-------------

This example introduces :class:`CrossSiteEval<nvflare.app_common.workflows.cross_site_eval.CrossSiteEval>` and builds
on the :doc:`Hello PyTorch <hello_pt>` example
based on the :class:`FedAvg<nvflare.app_common.workflows.fedavg.FedAvg>` workflow.

Introduction
-------------
In this exercise, you will learn how to use NVIDIA FLARE to perform cross site validation
after training.

The training process is similar to the train script uesd in the :doc:`Hello PyTorch <hello_pt>` example. This example does not
use the Job API to construct the job but instead has the job in the ``jobs`` folder of the example so you can see the server and
client configurations.

The setup of this exercise consists of one **server** and two **clients**.
The server side model starts with the default weights when the model is loaded with :class:`PTFileModelPersistor<nvflare.app_opt.pt.file_model_persistor.PTFileModelPersistor>`.

Cross site validation consists of the following steps:

    - The :class:`CrossSiteEval<nvflare.app_common.workflows.cross_site_eval.CrossSiteEval>` workflow
      gets the client models with the ``submit_model`` task.
    - The ``validate`` task is broadcast to the all participating clients with the model shareable containing the model data,
      and results from the ``validate`` task are saved.

During this exercise, we will see how NVIDIA FLARE takes care of most of the above steps with little work from the user.
We will be working with the ``hello-cross-val`` application in the examples folder.
Custom FL applications can contain the folders:

 #. **custom**: contains the custom components including our training script (``train.py``, ``net.py``)
 #. **config**: contains client and server configurations (``config_fed_client.conf``, ``config_fed_server.conf``)
 #. **resources**: can optionally contain the logger config (``log.config``)

Let's get started. First clone the repo, if you haven't already:

.. code-block:: shell

  $ git clone https://github.com/NVIDIA/NVFlare.git

Remember to activate your NVIDIA FLARE Python virtual environment from the installation guide.

Ensure PyTorch and torchvision are installed: 

.. code-block:: shell

  (nvflare-env) $ python3 -m pip install torch torchvision

Now that you have all your dependencies installed, let's take a look at the job.


Training
--------------------------------
 
In the :doc:`Hello PyTorch <hello_pt>` example, we implemented the setup and the training script in ``hello-pt_cifar10_fl.py``.
In this example, we start from the same basic setup and training script but extend it to process the ``validate`` and ``submit_model`` tasks to
work with the :class:`CrossSiteEval<nvflare.app_common.workflows.cross_site_eval.CrossSiteEval>`
workflow to get the client models.

Note that the server also produces a global model.
The :class:`CrossSiteEval<nvflare.app_common.workflows.cross_site_eval.CrossSiteEval>`
workflow submits the server model for evaluation after the client models.

Implementing the Validator
--------------------------

The code for processing the ``validate`` task during
the :class:`CrossSiteEval<nvflare.app_common.workflows.cross_site_eval.CrossSiteEval>` workflow is added to the
``while flare.is_running():`` loop in the training script.

.. code-block:: python

  elif flare.is_evaluate():
      accuracy = evaluate(input_model.params)
      print(f"({client_id}) accuracy: {accuracy}")
      flare.send(flare.FLModel(metrics={"accuracy": accuracy}))

It handles the ``validate`` task by performing calling the ``evaluate()`` method we have added to the ``train.py`` training script in our custom folder.

Application Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^

Inside the config folder there are two files, ``config_fed_client.json`` and ``config_fed_server.json``.

.. literalinclude:: ../../examples/hello-world/hello-cross-val/jobs/hello-cross-val/app/config/config_fed_server.conf
   :language: conf
   :linenos:
   :caption: config_fed_server.conf

The server now has a second workflow, :class:`CrossSiteEval<nvflare.app_common.workflows.cross_site_eval.CrossSiteEval>`, configured after Scatter and
Gather (:class:`ScatterAndGather<nvflare.app_common.workflows.scatter_and_gather.ScatterAndGather>` is an implementation of the :class:`FedAvg<nvflare.app_common.workflows.fedavg.FedAvg>` workflow).


.. literalinclude:: ../../examples/hello-world/hello-cross-val/jobs/hello-cross-val/app/config/config_fed_client.conf
   :language: conf
   :linenos:
   :caption: config_fed_client.conf

The client configuration now uses the Executor :class:`PTClientAPILauncherExecutor<nvflare.app_opt.pt.client_api_launcher_executor.PTClientAPILauncherExecutor>`
configured to launch the train script ``train.py`` with :class:`SubprocessLauncher<nvflare.app_common.launchers.subprocess_launcher.SubprocessLauncher>`.
The "train", "validate", and "submit_model" tasks have been configured for the added to the ``PTClientAPILauncherExecutor`` Executor to
work with the :class:`CrossSiteEval<nvflare.app_common.workflows.cross_site_eval.CrossSiteEval>` workflow.

Cross site validation!
----------------------

To run the application, you can use a POC environment or a real provisioned environment and use the FLARE Console or the FLARE API to submit the job,
or you can run quickly run it with the FLARE Simulator with the following command:

.. code-block:: shell

  (nvflare-env) $ nvflare simulator -w /tmp/nvflare/ -n 2 -t 1 examples/hello-world/hello-cross-val/jobs/hello-cross-val

During the first phase, the model will be trained.

During the second phase, cross site validation will happen.

The workflow on the client will change to :class:`CrossSiteEval<nvflare.app_common.workflows.cross_site_eval.CrossSiteEval>`
as it enters this second phase.

During cross site evaluation, every client validates other clients' models and server models (if present).
This can produce a lot of results. All the results will be kept in the job's workspace when it is completed.

Understanding the Output
^^^^^^^^^^^^^^^^^^^^^^^^

After running the job, you should begin to see outputs tracking the progress of the FL run.
As each client finishes training, it will start the cross site validation process.
During this you'll see several important outputs the track the progress of cross site validation.

The server shows the log of each client requesting models, the models it sends and the results received.
Since the server could be responding to many clients at the same time, it may
require careful examination to make proper sense of events from the jumbled logs.


.. include:: access_result.rst

.. note::
    You could see the cross-site validation results
    at ``[DOWNLOAD_DIR]/[JOB_ID]/workspace/cross_site_val/cross_val_results.json``

The full source code for this exercise can be found in
:github_nvflare_link:`examples/hello-world/hello-numpy-cross-val <examples/hello-world/hello-numpy-cross-val/>`.

Previous Versions of Hello Cross-Site Validation
------------------------------------------------

  - `hello-numpy-cross-val for 2.0 <https://github.com/NVIDIA/NVFlare/tree/2.0/examples/hello-numpy-cross-val>`_
  - `hello-numpy-cross-val for 2.1 <https://github.com/NVIDIA/NVFlare/tree/2.1/examples/hello-numpy-cross-val>`_
  - `hello-numpy-cross-val for 2.2 <https://github.com/NVIDIA/NVFlare/tree/2.2/examples/hello-numpy-cross-val>`_
  - `hello-numpy-cross-val for 2.3 <https://github.com/NVIDIA/NVFlare/tree/2.3/examples/hello-world/hello-numpy-cross-val/>`_
  - `hello-numpy-cross-val for 2.4 <https://github.com/NVIDIA/NVFlare/tree/2.4/examples/hello-world/hello-numpy-cross-val/>`_
