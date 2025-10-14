.. _getting_started:

###############
Getting Started
###############

This guide will help you understand the different ways to run NVIDIA FLARE and get started with your federated learning journey.
Before proceeding, make sure you have completed the :ref:`quickstart` guide to run your first example.

Ways to Run NVFlare
==================
NVFlare supports different running modes to accommodate various use cases, from development to production:

.. list-table:: NVIDIA FLARE Modes
   :header-rows: 1

   * - **Mode**
     - **Documentation**
     - **Description**
   * - Simulator
     - :ref:`fl_simulator`
     - | The FL Simulator is a light weight simulation where the job run is automated on a 
       | single system. Useful for quickly running a job or experimenting with research 
       | or FL algorithms.
   * - POC
     - :ref:`poc_command`
     - | POC mode establishes and connects distinct server and client "systems" which can 
       | then be orchestrated using the FLARE Console all from a single machine. Users can 
       | also experiment with various deployment options (project.yml), which can be used 
       | in production modes.
   * - Production
     - :ref:`provisioned_setup`
     - | Real world production mode involves a distributed deployment with generated startup 
       | kits from the provisioning process. Provides provisioning tool, dashboard, and 
       | various deployment options.

Next Steps
==========
Now that you understand the different ways to run NVFlare:

1. Try the getting started `tutorials <https://github.com/NVIDIA/NVFlare/tree/2.6/examples/getting_started>`__ to learn more about each mode
2. Explore more advanced examples and `step-by-step <https://github.com/NVIDIA/NVFlare/tree/2.6/examples/hello-world/step-by-step>`__ walk-throughs
3. Learn how to convert your standalone/centralized training code to `federated learning code <https://github.com/NVIDIA/NVFlare/tree/2.6/examples/hello-world/ml-to-fl>`__
4. When ready for production, see :ref:`real_world_fl` for deployment guidance
5. For development, see :ref:`programming_guide` for detailed programming instructions
