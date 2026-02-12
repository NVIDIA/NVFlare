.. _llm_fine_tuning:

######################################
Federated LLM Fine-Tuning
######################################

NVIDIA FLARE supports federated fine-tuning of large language models (LLMs) using
popular frameworks including HuggingFace Transformers and NVIDIA NeMo. Multiple
fine-tuning strategies are supported:

- **SFT (Supervised Fine-Tuning)** -- Full or partial model fine-tuning on task-specific data
- **PEFT (Parameter-Efficient Fine-Tuning)** -- LoRA and other adapter-based methods that train only a small subset of parameters
- **Prompt Learning** -- Learning soft prompts while keeping the base model frozen

All approaches use the standard FLARE Client API, so you can convert existing
single-machine fine-tuning scripts to federated with minimal code changes.

HuggingFace Integration
=======================

FLARE provides direct support for federated fine-tuning of HuggingFace models.

**Federated SFT** fine-tunes the full model (or selected layers) across sites:

.. code-block:: python

   # client.py -- standard HuggingFace training, federated via Client API
   from nvflare.client.tracking import SummaryWriter
   import nvflare.client as flare

   flare.init()
   while flare.is_running():
       input_model = flare.receive()
       # Load weights into your HuggingFace model
       # Run SFT training loop
       # Send updated weights back
       flare.send(output_model)

**Federated PEFT (LoRA)** trains only adapter parameters, dramatically reducing
communication costs -- ideal for large models where transmitting full weights is impractical.

See the complete examples:

- `HuggingFace SFT & PEFT Examples <https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/llm_hf>`_

NVIDIA NeMo Integration
=======================

For NVIDIA NeMo models, FLARE provides tight integration for multiple fine-tuning strategies:

- `Federated SFT with NeMo <https://github.com/NVIDIA/NVFlare/tree/main/integration/nemo/examples/supervised_fine_tuning>`_ -- Supervised fine-tuning of NeMo models across sites
- `Federated PEFT with NeMo <https://github.com/NVIDIA/NVFlare/tree/main/integration/nemo/examples/peft>`_ -- Parameter-efficient fine-tuning (LoRA, P-Tuning) with NeMo
- `Federated Prompt Learning with NeMo <https://github.com/NVIDIA/NVFlare/tree/main/integration/nemo/examples/prompt_learning>`_ -- Learning soft prompts while keeping base model frozen

Self-Paced Training
===================

For a structured learning path covering federated LLM training:

- `Chapter 8: Federated LLM Training <https://github.com/NVIDIA/NVFlare/tree/main/examples/tutorials/self-paced-training/part-4_advanced_federated_learning/chapter-8_federated_LLM_training>`_

  - 8.1 Federated BERT
  - 8.2 Federated SFT
  - 8.3 Federated PEFT
  - 8.4 LLM Quantization for Communication
  - 8.5 LLM Streaming
