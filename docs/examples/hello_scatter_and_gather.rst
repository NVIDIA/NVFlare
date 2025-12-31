.. _hello_scatter_and_gather:

Hello Federated Learning with NumPy
====================================

Before You Start
----------------

Before jumping into this guide, make sure you have an environment with
`NVIDIA FLARE <https://pypi.org/project/nvflare/>`_ installed.

You can follow :ref:`getting_started` on the general concept of setting up a
Python virtual environment (the recommended environment) and how to install NVIDIA FLARE.


Introduction
-------------

This tutorial demonstrates how NVIDIA FLARE's federated learning system works using a simple NumPy example,
without introducing complex deep learning concepts.

Through this exercise, you will learn the basic federated learning workflow:

 #. A server coordinates training across multiple clients
 #. Clients train on their local data
 #. The server aggregates the results to produce an updated global model
 #. This process repeats for multiple rounds

The current ``hello-numpy`` example uses NVIDIA FLARE's **Recipe API**, a modern Python-based approach
to defining federated learning jobs. It demonstrates the core FL concepts with simple NumPy arrays as model weights.

Getting Started
---------------

First, clone the repo if you haven't already:

.. code-block:: shell

  $ git clone https://github.com/NVIDIA/NVFlare.git

Remember to activate your NVIDIA FLARE Python virtual environment from the installation guide.
Ensure numpy is installed.

.. code-block:: shell

  (nvflare-env) $ python3 -m pip install numpy

Running the Hello NumPy Example
--------------------------------

The ``hello-numpy`` example uses NVIDIA FLARE's Recipe API, which provides a Python-based approach to defining FL jobs.

To run the example with 2 clients in simulation mode:

.. code-block:: shell

  (nvflare-env) $ cd examples/hello-world/hello-numpy
  (nvflare-env) $ python job.py --n_clients 2

The job will run in a simulated environment (SimEnv) and you'll see the federated learning process in action,
with the server coordinating rounds of training across the clients.

Results are saved to ``/tmp/nvflare/simulation/hello-numpy/`` by default.

Understanding the Code
-----------------------

The ``job.py`` file defines the federated learning job using the Recipe API. It specifies:

 * The workflow controller (how rounds are coordinated)
 * The client-side training logic
 * The server-side aggregation strategy
 * Model persistors and other FL components

For detailed code examples and explanations, refer to:

 * The ``job.py`` file in :github_nvflare_link:`examples/hello-world/hello-numpy <examples/hello-world/hello-numpy>`
 * The :ref:`programming guide <programming_guide>` for core FL concepts
 * The :ref:`hello_fedavg_numpy` example for a similar approach with FedAvg workflow

Congratulations!
----------------

You've successfully run your first NumPy federated learning system.

You now have a decent grasp of the main FL concepts, and are ready to start exploring how NVIDIA FLARE can be applied to many other tasks.

The full application for this exercise can be found in
:github_nvflare_link:`examples/hello-world/hello-numpy <examples/hello-world/hello-numpy>`.

Previous Versions of Hello Scatter and Gather
---------------------------------------------

   - `hello-numpy-sag for 2.0 <https://github.com/NVIDIA/NVFlare/tree/2.0/examples/hello-numpy-sag>`_
   - `hello-numpy-sag for 2.1 <https://github.com/NVIDIA/NVFlare/tree/2.1/examples/hello-numpy-sag>`_
   - `hello-numpy-sag for 2.2 <https://github.com/NVIDIA/NVFlare/tree/2.2/examples/hello-numpy-sag>`_
   - `hello-numpy-sag for 2.3 <https://github.com/NVIDIA/NVFlare/tree/2.3/examples/hello-world/hello-numpy-sag>`_
