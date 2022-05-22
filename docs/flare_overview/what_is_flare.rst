----------------------
What is NVIDIA FLARE ?
----------------------

NVIDIA FLARE (NVIDIA Federated Learning Application Runtime Environment) is a domain-agnostic, open-source, extensible
SDK that allows researchers and data scientists to adapt existing ML/DL workflows (PyTorch, RAPIDS, Nemo, TensorFlow) to
a federated paradigm; and enables platform developers to build a secure, privacy preserving offering for a distributed
multi-party collaboration.

Federated learning allows for multiple clients each with their own data to collaborate together without
having to share their actual data. This allows for different parties located any place in the world to use their local
secure protected data to perform tasks coordinated by an FL server set up in the secure cloud to achieve better learning
results.

NVIDIA FLARE is built on a componentized architecture, which allows researchers to customize workflows to their
liking and experiment with different ideas quickly.


FLARE is a Federated Runtime Environment
----------------------------------------
At the deepest core, FLARE provides a plugable runtime environment that supports federated computing.
FLARE provides different clients and server communicate over internet exchange communications.
The orchestration, control flow, aggregation logics at server can be customized to your needs. The compute logics on the clide sites
can be customized to support all kinds of federated computing. In the future, the communication protocol can be pluggable.

Such flexible, customizable framework allows one to fits all kinds of real-world use cases


FLARE is about Federated Learning
---------------------------------
** NOTE: NEED MORE WORK **

Flare provides out of box Federated learning algorithms that reflects
the state of art (SOT) deep learning and macnine learning algorithm.
  * Basic Federated Learning Algorithms (FedAvg, ...)
  * Federated Analysis
  * Deep Learning ( CIFAIR10 ...)
  * Personalized Federated Learning ( Ditto)
  * non-IID (Scaffolding)
  * Medical Applications ( Prostate )
  * xxxx : FedSM
  * Federated XgBoost ( coming soon)

You can build your own customized algorithms fair easily due to the FLARE's pluggable component design.

FLARE is a SDK, not a platform
------------------------------

We want to enable more people to levage Federated Learning, weather the user is
* a machine learning researcher -- interested in experimenting the latest FL algorithms, or
* a data scientist -- interested to apply FL to the real world use case, or
* a system integrator -- interested to build a platform that enable Federated Learning for others

For researcher, FLARE will provide an easy to use enviornment that quickly experiments different FL algorithms.
For data scientist, we like to make it easy to take FL algorithms and deploy to the real world without much change.
For system integrator, you should be easily replace most any components and customize to your needs,
weather its communication, authentication, storage, workflow, deep learning framework.
FLARE should be easily embedded into your system.

