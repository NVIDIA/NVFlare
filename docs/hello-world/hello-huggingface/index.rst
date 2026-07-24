.. _hello_huggingface:

Hello HuggingFace
=================

This example demonstrates how to use NVIDIA FLARE with the HuggingFace Client
API to run federated supervised fine-tuning with a Qwen causal language model.
The complete example code is in
:github_nvflare_link:`examples/hello-world/hello-huggingface <examples/hello-world/hello-huggingface>`.

Install NVFLARE and Dependencies
--------------------------------

For complete installation instructions, see :doc:`Installation </installation>`.
On a released branch:

.. code-block:: text

   pip install nvflare

The HuggingFace Client API is introduced for NVFlare 2.9.0. Until that package
is published, install NVFlare from this repository and install the remaining
example dependencies separately:

.. code-block:: bash

   git clone https://github.com/NVIDIA/NVFlare.git
   cd NVFlare
   python -m pip install -e .
   python -m pip install torch transformers accelerate datasets peft trl safetensors

The ``nvflare~=2.9.0rc`` entry in ``requirements.txt`` records the first
compatible release. After NVFlare 2.9.0 is published,
``python -m pip install -r requirements.txt`` installs the complete
environment.

Code Structure
--------------

.. code-block:: text

   hello-huggingface
   |
   |-- client.py        # HuggingFace/TRL local training script
   |-- model.py         # Qwen server-side model definitions
   |-- prepare_data.py  # writes synthetic per-site JSONL data
   |-- job.py           # job recipe for simulation and export
   |-- requirements.txt
   |-- README.md

Data
----

Prepare the default two-client synthetic JSONL dataset:

.. code-block:: bash

   python prepare_data.py

By default this writes:

.. code-block:: text

   /tmp/nvflare/hello-huggingface/data
   |
   |-- site-1
   |   |-- train.jsonl
   |   |-- valid.jsonl
   |-- site-2
   |   |-- train.jsonl
   |   |-- valid.jsonl

You can use your own prepared data by passing ``--data_root`` to
``prepare_data.py`` and ``job.py``.

Client Code
-----------

The client script is a normal HuggingFace/TRL ``SFTTrainer`` script. The
federated adaptation is intentionally small:

.. code-block:: python

   import nvflare.client.hf as flare

   flare.patch(trainer)

   while flare.is_running():
       trainer.evaluate()
       trainer.train()

``flare.patch(trainer)`` wraps the trainer methods so the script can receive
global parameters, evaluate, run the local training budget, and send the result
back to the FL server.

Run Job
-------

After preparing data, run the simulation:

.. code-block:: bash

   python job.py

The default job runs two simulated clients for two FL rounds using PEFT/LoRA.
To export the job configuration without running simulation:

.. code-block:: bash

   python job.py --export_config

Use ``--train_mode sft`` to run full-model SFT instead of the default LoRA
mode.

Learn More
----------

For the HuggingFace Client API contract and options, see :ref:`hf_client_api`.
