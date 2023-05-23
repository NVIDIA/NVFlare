## Prompt Learning with NeMo

In this example, we utilize NeMo's [prompt learning](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/nemo_megatron/prompt_learning.html)
feature to showcase how to adapt a large language model (LLM) to 
a downstream task such as financial sentiment predictions. 
As the prompt learning technique shown in the example is p-tuning which adds a small prompt encoder network to the LLM
to produce virtual tokens that guide the model toward the desired output of the downstream task.

## Dependencies
We assume you followed the instructions [here](../../README.md#requirements) 
to install the NeMo framework and the NeMo-NVFlare package. 

## Run the example
We use [JupyterLab](https://jupyterlab.readthedocs.io) for this example.
To start JupyterLab, run
```
jupyter lab .
```
and open [prompt_learning.ipynb](./prompt_learning.ipynb).
