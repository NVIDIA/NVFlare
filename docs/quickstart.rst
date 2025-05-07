.. _quickstart:

##########
Quickstart
##########

This guide will get you running a simple federated learning example in minutes.
Make sure you have completed the :ref:`installation` steps before proceeding.

Prerequisites
=============
- Python 3.9+
- pip
- Git
- NVFlare installed (see :ref:`installation`)

Run Your First Example
======================
1. Clone the examples:

.. code-block:: shell

   $ git clone https://github.com/NVIDIA/NVFlare.git
   $ cd NVFlare/examples/hello-world/hello-pt

2. Install example dependencies:

.. code-block:: shell

   $ pip install -r requirements.txt

3. Run the example:

.. code-block:: shell

   $ python fedavg_script_runner_pt.py

That's it! You should see the federated learning simulation running with two clients training a model together.

Understanding the Example
=========================
This example demonstrates a simple federated learning scenario using PyTorch. For a detailed explanation of:

- How the example works
- The neural network architecture
- The federated learning workflow
- PyTorch integration details

See the :doc:`Hello PyTorch with Job API <examples/hello_pt_job_api>` guide and the :doc:`FedJob API <programming_guide/fed_job_api>` documentation.

Next Steps
==========
Now that you've run your first example:

1. Learn more about different ways to run NVFlare in the :ref:`getting_started` guide
2. Explore more examples in the :ref:`example_applications` section
3. When ready for production, see :ref:`real_world_fl` for deployment guidance
4. For development, see :ref:`programming_guide`
