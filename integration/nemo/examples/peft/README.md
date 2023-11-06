## Parameter-Efficient Fine-Tuning (PEFT) with NeMo

In this example, we utilize NeMo's [PEFT](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/nemo_megatron/peft/landing_page.html)
methods to showcase how to adapt a large language model (LLM) to 
a downstream task, such as financial sentiment predictions. 

With one line configuration change, you can try different PEFT techniques such as [p-tuning](https://arxiv.org/abs/2103.10385), [adapters](https://proceedings.mlr.press/v97/houlsby19a.html), or [LoRA](https://arxiv.org/abs/2106.09685), which add a small number of trainable parameters to the LLM
that condition the model to produce the desired output for the downstream task.

For more details, see the [PEFT script](https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/language_modeling/tuning/megatron_gpt_peft_tuning.py) in NeMo, which we adapt using NVFlare's Lightning client API to run in a federated scenario.

## Dependencies
We assume you followed the instructions [here](../../README.md#requirements) 
to install the NeMo, NVFlare, and the NeMo-NVFlare package. 

The example was tested with the main branch of [NeMo](https://github.com/NVIDIA/NeMo).

## Examples
### 1. Federated p-tuning using a 345 million parameter GPT model
This example requires a GPU with at least 16GB memory to run three clients in parallel on the same GPU.
We use [JupyterLab](https://jupyterlab.readthedocs.io) for this example.
To start JupyterLab, run
```
jupyter lab .
```
and open [peft.ipynb](./peft.ipynb).
