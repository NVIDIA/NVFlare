.. _flare_overview:

#####################
NVIDIA FLARE Overview
#####################

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

With FLARE's pluggable component architecture, you can extend these reference implementations
with customized algorithms to fit your needs.

FLARE is a SDK, not a platform
------------------------------
We want to enable more people to levage Federated Learning, weather the user is
* a machine learning researcher -- interested in experimenting the latest FL algorithms, or
* a data scientist -- interested in applying FL to a real world use case, or
* a system integrator -- interested in building a platform to enable Federated Learning for others.

For researcher, FLARE will provide an easy to use enviornment that quickly experiments different FL algorithms.
For data scientist, we like to make it easy to take FL algorithms and deploy to the real world without much change.
For system integrator, you should be easily replace most any components and customize to your needs,
weather its communication, authentication, storage, workflow, deep learning framework.
FLARE should be easily embedded into your system.

.. toctree::
   :maxdepth: 1
   flare_overview/flare_unique_features
   flare_overview/flare_design_principles
   flare_overview/flare_system_architecture