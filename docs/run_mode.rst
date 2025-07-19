.. _run_mode:

   **Modes to Run NVFLARE**
   =========================
   NVFLARE supports various modes to suit different needs, from development to production. Here’s an overview of the available modes:

   .. list-table:: NVIDIA FLARE Modes
      :header-rows: 1

      * - **Mode**
        - **Documentation**
        - **Description**
      * - Simulator
        - :ref:`fl_simulator`
        - | The FL Simulator is a lightweight simulation tool that automates job runs on a
          | single system. It is ideal for quickly running jobs or experimenting with research
          | and FL algorithms.
      * - POC
        - :ref:`poc_command`
        - | POC mode simulates deployment on a single host. It uses separate processes for Clients and Server,
          | allowing you to test the "provision" process locally with all pointing to localhost.
          | You can also interact with the FLARE system locally via POC mode.
      * - Production
        - :ref:`provisioned_setup`
        - | The production mode involves a distributed deployment with startup kits generated
          | from the provisioning process. It provides tools for provisioning, a dashboard, and
          | various deployment options.