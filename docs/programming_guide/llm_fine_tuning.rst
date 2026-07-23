.. _llm_fine_tuning:

######################################
Federated LLM Fine-Tuning
######################################

NVIDIA FLARE supports federated fine-tuning of large language models (LLMs) using
popular frameworks including HuggingFace Transformers and NVIDIA NeMo. Multiple
fine-tuning strategies are supported:

- **SFT (Supervised Fine-Tuning)** -- Full or partial model fine-tuning on task-specific data
- **PEFT (Parameter-Efficient Fine-Tuning)** -- LoRA and other adapter-based methods that train only a small subset of parameters

All approaches use the FLARE Client API, so you can convert existing
single-machine fine-tuning scripts to federated with minimal code changes.

HuggingFace Integration
=======================

FLARE provides direct support for federated fine-tuning of HuggingFace models
with ``nvflare.client.hf``. This facade patches a HuggingFace ``Trainer`` or
TRL ``SFTTrainer`` so existing ``trainer.evaluate()`` and ``trainer.train()``
calls can participate in FL rounds.

**Federated SFT** fine-tunes the full model (or selected layers) across sites:

.. code-block:: python

   # client.py -- standard HuggingFace training, federated via HuggingFace Client API
   import nvflare.client.hf as flare

   trainer = SFTTrainer(...)
   flare.patch(trainer, params_scope="auto")

   while flare.is_running():
       trainer.evaluate()
       trainer.train()

**Federated PEFT (LoRA)** trains only adapter parameters, dramatically reducing
communication costs -- ideal for large models where transmitting full weights is impractical.

See the complete examples:

- `HuggingFace Client API Qwen Example <https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/hf_client_api>`_
- :ref:`hf_client_api`

NVIDIA NeMo Integration
=======================

For NVIDIA NeMo models, FLARE provides tight integration for multiple fine-tuning strategies:

- `Federated SFT with NeMo <https://github.com/NVIDIA/NVFlare/tree/main/integration/nemo/examples/supervised_fine_tuning>`_ -- Supervised fine-tuning of NeMo models across sites
- `Federated PEFT with NeMo <https://github.com/NVIDIA/NVFlare/tree/main/integration/nemo/examples/peft>`_ -- Parameter-efficient LoRA fine-tuning with NeMo

Self-Paced Training
===================

For a structured learning path covering federated LLM training:

- `Chapter 8: Federated LLM Training <https://github.com/NVIDIA/NVFlare/tree/main/examples/tutorials/self-paced-training/part-4_advanced_federated_learning/chapter-8_federated_LLM_training>`_

  - 8.1 Federated BERT
  - 8.2 Federated SFT
  - 8.3 Federated PEFT
  - 8.4 LLM Quantization for Communication
  - 8.5 LLM Streaming
