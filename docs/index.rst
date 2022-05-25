############
NVIDIA FLARE
############


NVIDIA FLARE (NVIDIA Federated Learning Application Runtime Environment) is a domain-agnostic, open-source, extensible SDK that allows researchers and data scientists to adaptexisting ML/DL workflows (PyTorch, RAPIDS, Nemo, TensorFlow) to a federated paradigm; and enables platform developers to build a secure, privacy preserving offering for a distributed multi-party collaboration.

Federated learning allows multiple clients, each with their own data, to collaborate without sharing data.  Different parties or institutions located throughtout the world can perform a set of tasks on their own local data, coordinated by a secure, central Federated Learning server, to build a global model.  NVIDIA FLARE enables this collaborative workflow without ever needing to give external access to a participants' local data.

NVIDIA FLARE is built on a componentized architecture that allows researchers to customize workflows to their liking and experiment with different ideas quickly.

With NVIDIA FLARE 2.1.0, :ref:`High Availability (HA) <high_availability>` and :ref:`Multi-Job Execution <multi_job>` introduce new concepts and change the way the system needs to be configured and operated. See `conversion from 2.0 <appendix/converting_from_previous.html>`_ for details.

.. toctree::
   :maxdepth: 1

   highlights
   flare_overview
   quickstart
   example_applications
   user_guide
   programming_guide
   best_practices
   faq
   contributing
   API <apidocs/modules>
   appendix
