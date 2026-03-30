:orphan:

.. _run_mode:

Modes to Run NVFLARE
====================

NVFLARE supports several modes for different use cases:

+-------------+--------------------------------------------------------------+
| Mode        | Description                                                  |
+=============+==============================================================+
| Simulator   | :ref:`fl_simulator`                                          |
|             | Lightweight, runs jobs on a single system for fast testing   |
|             | and FL algorithm experiments.                                |
+-------------+--------------------------------------------------------------+
| POC         | :ref:`poc_command`                                           |
|             | Simulates deployment on one host with separate processes     |
|             | for Clients/Server. Enables local "provision" testing and    |
|             | FLARE system interaction.                                    |
+-------------+--------------------------------------------------------------+
| Production  | :ref:`provisioned_setup`                                     |
|             | Distributed deployment using startup kits from provisioning. |
|             | Includes provisioning tools, dashboard, and deployment       |
|             | options.                                                     |
+-------------+--------------------------------------------------------------+