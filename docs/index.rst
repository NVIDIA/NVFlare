############
NVIDIA FLARE
############

.. toctree::
   :maxdepth: -1
   :hidden:
   :caption: Introduction

   fl_introduction
   flare_overview
   whats_new
   Getting Started <quickstart>

.. toctree::
   :maxdepth: -1
   :hidden:
   :caption: Guides

   example_applications_algorithms
   real_world_fl
   user_guide
   programming_guide
   best_practices

.. toctree::
   :maxdepth: -1
   :hidden:
   :caption: Miscellaneous

   faq
   publications_and_talks
   contributing
   API <apidocs/modules>
   glossary

NVIDIA FLARE (NVIDIA Federated Learning Application Runtime Environment) is a domain-agnostic, open-source, extensible SDK that allows
researchers and data scientists to adapt existing ML/DL workflows (PyTorch, RAPIDS, Nemo, TensorFlow) to a federated paradigm; and enables
platform developers to build a secure, privacy preserving offering for a distributed multi-party collaboration.

NVIDIA FLARE is built on a componentized architecture that gives you the flexibility to take federated learning workloads from research
and simulation to real-world production deployment.  Some of the key components of this architecture include:

 - **FL Simulator** for rapid development and prototyping
 - **FLARE Dashboard** for simplified project management and deployment
 - **Reference FL algorithms** (e.g., FedAvg, FedProx) and workflows (e.g., Scatter and Gather, Cyclic)
 - **Privacy preservation** with differential privacy, homomorphic encryption, and more
 - **Management tools** for secure provisioning and deployment, orchestration, and management
 - **Specification-based API** for extensibility

Learn more about FLARE features in the :ref:`FLARE Overview <flare_overview>` and :ref:`What's New <whats_new>`.

Getting Started
===============
For first-time users and FL researchers, FLARE provides the :ref:`FL Simulator <fl_simulator>` that allows you to build, test, and deploy applications locally.
The :ref:`Getting Started <getting_started>` guide covers installation and walks through an example application using the FL Simulator.
Additional examples can be found at the :ref:`Examples Applications <example_applications>`, which showcase different federated learning workflows and algorithms on various machine learning and deep learning tasks.

FLARE for Users
===============
If you want to learn how to interact with the FLARE system, please refer to the :ref:`User Guide <user_guide>`.
When you are ready for a secure, distributed deployment, the :ref:`Real World Federated Learning <real_world_fl>` section covers the tools and processes required to deploy and operate a secure, real-world FLARE project.

FLARE for Developers
====================
When you're ready to build your own application, the :ref:`Programming Guide <programming_guide>`, :ref:`Programming Best Practices <best_practices>`, :ref:`FAQ <faq>`, and :ref:`API Reference <apidocs/modules>`
provide an in-depth look at the FLARE platform and APIs.
