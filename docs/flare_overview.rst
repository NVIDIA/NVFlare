.. _flare_overview:

#####################
NVIDIA FLARE Overview
#####################

.. toctree::
   :maxdepth: 1

   flare_overview/design_principles
   flare_overview/system_architecture
   flare_overview/unique_features

FLARE is a Federated Runtime Environment
----------------------------------------
At its core, FLARE provides a plugable runtime environment that supports federated computing.
The FLARE runtime supports client and server communication over gRPC.  The orchestration,
control flow, aggregation logic on the server can be customized to support
any collaborative computing workflow. Similarly, the client side compute logic can be constructed
to support any federated workflow.  In the future, the communication protocol will be abstracted,
so that any protocol can be used to facilitate the federated computing workflow. This flexible,
customizable framework allows the end user to address any real-world workflow by adapting the environment to their specific needs.


FLARE is about Federated Learning
---------------------------------
FLARE provides reference Federated Learning algorithms that reflect
the state of the art (SOTA) for deep learning and machine learning.

  * Basic Federated Learning Algorithms (FedAvg, FedProx, FedOpt)
  * Federated Analysis
  * Deep Learning (CIFAR10, ...)
  * Personalized Federated Learning (Ditto)
  * Non-IID (Scaffold)
  * Medical Applications (MONAI, ...)
  * Federated XGBoost (coming soon)

With FLARE's pluggable component architecture, you can extend these reference implementations with customized algorithms to fit your needs.

FLARE is a SDK, not a platform
------------------------------
We want to enable more people to adopt Federated Learning, whether the user is

  * a machine learning researcher -- interested in experimenting the latest FL algorithms, or
  * a data scientist -- interested in applying FL to a real world use case, or
  * a system integrator -- interested in building a platform to enable Federated Learning for others.

For a researcher, FLARE provides an easy to use enviornment that allows rapid experimentation with different FL algorithms.

For a data scientist, FLARE makes it easy to use common FL algorithms to build federated applications for real-world deployments.

For system integrator, FLARE's modular architecture makes it easy to extend or replace components and customize to your needs.  Whether it's communication, authentication, storage, workflow, or DL/ML frameworks, FLARE should be easily embedded into your system.

