############
NVIDIA FLARE
############

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

Collaborative computing is not limited to deep learning, and with NVIDIA FLARE 2.0, you can develop any FL workflows
with the newly provided `Controller API <programming_guide/controllers>`_. A full reference implementation of the
`previous default workflow <programming_guide/controllers/scatter_and_gather_workflow>`_ is included along with the
source code.

.. toctree::
   :maxdepth: 1

   highlights
   installation
   quickstart
   example_applications
   user_guide
   programming_guide
   best_practices
   faq
   API <apidocs/modules>
   appendix
