.. _tutorials:

Tutorials
=========

Learn NVFLARE in the order that matches your goal. If you are new to NVFLARE,
start with the current onboarding sequence. Experienced users can go directly
to the focused feature tutorials, the searchable example catalog, or the
structured training resources.

Current Onboarding
------------------

This short sequence is tested against the current release.

#. :ref:`quickstart` — run your first federated learning job in the simulator,
   inspect the result, and choose your next step.
#. :github_nvflare_link:`Hello World examples <examples/hello-world>` —
   introductory federated applications across frameworks (PyTorch, PyTorch
   Lightning, TensorFlow, JAX, NumPy, Hugging Face, Flower). Each example runs
   with a single ``python job.py``.

Feature Tutorials
-----------------

Focused Jupyter notebooks for learning one NVFLARE capability, run locally
from the :github_nvflare_link:`tutorials directory <examples/tutorials>`.
The README in that directory is the authoritative index: it records each
notebook's outcome, prerequisites, entry point, expected completion signal,
tested environment, and last verification.

**Build and test applications**

- `Intro to the FL Simulator <https://github.com/NVIDIA/NVFlare/blob/main/examples/tutorials/flare_simulator.ipynb>`_ —
  run jobs on a single system with the simulator-managed server and simulated
  clients.
- `Job Recipe <https://github.com/NVIDIA/NVFlare/blob/main/examples/tutorials/job_recipe.ipynb>`_ —
  build and run federated jobs with the high-level Recipe API instead of
  hand-written job configurations.

**Operate and deploy systems**

- `NVFLARE in POC Mode <https://github.com/NVIDIA/NVFlare/blob/main/examples/tutorials/setup_poc.ipynb>`_ —
  configure and operate a full local FL system with separate server, client,
  and admin processes.
- `NVFlare CLI <https://github.com/NVIDIA/NVFlare/blob/main/examples/tutorials/nvflare_cli.ipynb>`_ —
  manage setup, recipes, jobs, systems, studies, and deployment from the
  ``nvflare`` command groups.
- `Hello FLARE API <https://github.com/NVIDIA/NVFlare/blob/main/examples/tutorials/flare_api.ipynb>`_ —
  connect an admin session and drive a running FL system programmatically with
  the FLARE API.

**Observe and troubleshoot runs**

- `FLARE Logging <https://github.com/NVIDIA/NVFlare/blob/main/examples/tutorials/logging.ipynb>`_ —
  configure console and file logging for different use cases and run modes.

Example Catalog
---------------

- Browse and filter the complete set of runnable examples by experience level,
  framework, algorithm, application, industry, API, and privacy algorithm in
  the `NVFLARE Catalog <https://nvidia.github.io/NVFlare/catalog/>`_.
- See :ref:`example_applications` for the major example applications and
  algorithms in this repository.
- `Video tutorials <https://nvidia.github.io/NVFlare/>`_ give quick
  introductions to FLARE for deep learning and traditional machine learning,
  including CIFAR10 classification and Kaplan-Meier survival analysis.

Structured Training
-------------------

.. note::

   The self-paced course was developed with NVFLARE 2.6, and the hosted DLI
   courses are versioned independently of the current NVFLARE release and may
   use earlier versions. Both remain useful structured paths, but not all
   content reflects the latest APIs; some examples may need adjustments for
   newer versions.

- :ref:`self_paced_training` — a five-part, 12-chapter course with over 100
  notebooks and videos covering federated learning with NVFLARE, from
  fundamentals to system architecture, privacy and security, advanced topics,
  and industry applications.
- Hosted NVIDIA DLI courses for guided, hands-on FLARE training
  (independently versioned; may use earlier NVFLARE releases):
  `Introduction to Federated Learning with NVIDIA FLARE <https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-FX-28+V1>`_
  and
  `Decentralized AI at Scale with NVIDIA FLARE <https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-FX-29+V1>`_.

Visit the `NVIDIA FLARE developer portal <https://developer.nvidia.com/flare>`_
for product information, tutorials, documentation, and developer resources.
