## Prompt Learning with NeMo

In this example, we utilize NeMo's [prompt learning](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/nemo_megatron/prompt_learning.html)
feature to showcase how to adapt a large language model (LLM) to 
a downstream task such as financial sentiment predictions. 
As the prompt learning technique shown in the example is p-tuning which adds a small prompt encoder network to the LLM
to produce virtual tokens that guide the model toward the desired output of the downstream task.

<img src="./figs/p-tuning.svg"  width="60%" height="60%">

In our federated implementation, the LLM parameters stay fixed. Prompt encoder parameters are trained/updated and averaged on the server.

<img src="./figs/fed_p-tuning.svg"  width="90%" height="90%">

## Dependencies
We assume you followed the instructions [here](../../README.md#requirements) 
to install the NeMo, NVFlare, and the NeMo-NVFlare package. 

## Examples
### 1. Federated p-tuning using a 345 million parameter GPT model
This example requires a GPU with at least 16GB memory to run three clients in parallel on the same GPU.
We use [JupyterLab](https://jupyterlab.readthedocs.io) for this example.
To start JupyterLab, run
```
jupyter lab .
```
and open [prompt_learning.ipynb](./prompt_learning.ipynb).

### 2. Federated p-tuning using a 20 billion parameter GPT model
This example running a 20B GPT model requires more computational resources. 
To run three clients in parallel, we require at least six GPUs with 64 GB memory or more each 
(Ampere or later GPU architecture).

To run the example, follow the instructions in [prompt_learning_20B.md](prompt_learning_20B.md).
