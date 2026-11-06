# Examples of NeMo-NVFlare Integration

### [Parameter-Efficient Fine-Tuning (PEFT) with NeMo](./peft/README.md)
In this example, we fine-tune LoRA adapters for Nemotron 3 Nano or Nemotron 3.5 Lightning with NeMo AutoModel PEFT
using NVFlare's Recipe API and explicit Client API calls. The financial sentiment task shows how to adapt a large
language model to a downstream task while exchanging only its adapters.

### [Supervised fine-tuning (SFT) with NeMo and NVFlare](./supervised_fine_tuning/README.md)
In this example, we fine-tune all trainable weights of a Nemotron 3 Nano model with NeMo AutoModel SFT using NVFlare's
Recipe API, full-model FedAvg transfer, and explicit Client API calls.
