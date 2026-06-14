# Examples of NeMo-NVFlare Integration

### [Parameter-Efficient Fine-Tuning (PEFT) with NeMo](./peft/README.md)
In this example, we fine-tune Nemotron 3 Nano LoRA adapters with NeMo AutoModel PEFT using NVFlare's Recipe API and
explicit Client API calls.
methods to showcase how to adapt a large language model (LLM) to 
a downstream task, such as financial sentiment predictions. 

### [Supervised fine-tuning (SFT) with NeMo and NVFlare](./supervised_fine_tuning/README.md)
An example of using NVIDIA FLARE
with NeMo for [supervised fine-tuning (SFT)](https://github.com/NVIDIA/NeMo-Megatron-Launcher#5152-sft-training) 
to fine-tune all parameters of a large language model (LLM) on supervised data to teach the model how to follow user specified instructions. 

### [Prompt learning with NeMo and NVFlare](./prompt_learning/README.md)
An example of using NVIDIA FLARE
with NeMo for [prompt learning](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/v1.17.0/nlp/nemo_megatron/prompt_learning.html) using NVFlare's Learner API
to adapt a large language model (LLM) to a downstream task. 
