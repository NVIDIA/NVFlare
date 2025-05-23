## Parameter-Efficient Fine-Tuning (PEFT) with NeMo

In this example, we utilize NeMo's [PEFT](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/v1.22.0/nlp/nemo_megatron/peft/landing_page.html)
methods to showcase how to adapt a large language model (LLM) to 
a downstream task, such as financial sentiment predictions. 

With one line configuration change, you can try different PEFT techniques such as [p-tuning](https://arxiv.org/abs/2103.10385), [adapters](https://proceedings.mlr.press/v97/houlsby19a.html), or [LoRA](https://arxiv.org/abs/2106.09685), which add a small number of trainable parameters to the LLM
that condition the model to produce the desired output for the downstream task.

For more details, see the [PEFT script](https://github.com/NVIDIA/NeMo/blob/v1.22.0/examples/nlp/language_modeling/tuning/megatron_gpt_peft_tuning.py) in NeMo, which we adapt using NVFlare's Lightning client API to run in a federated scenario.

## Dependencies
The example was tested with the [NeMo 23.10 container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo).
In the following, we assume this example folder of the container is mounted to `/workspace` and all downloading, etc. operations are based on this root path.

> Note in the following, mount both the [current directory](./) and the [job_templates](../../../../job_templates) 
> directory to locations inside the docker container. Please make sure you have cloned the full NVFlare repo. 

Start the docker container from **this directory** using
```
# cd NVFlare/integration/nemo/examples/peft
DOCKER_IMAGE="nvcr.io/nvidia/nemo:23.10"
docker run --runtime=nvidia -it --rm --shm-size=16g -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit stack=67108864 \
-v ${PWD}/../../../../job_templates:/job_templates -v ${PWD}:/workspace -w /workspace ${DOCKER_IMAGE}
```

Next, install NVFlare.
```
pip install nvflare~=2.5.0rc
```

## Examples
### 1. Federated PEFT using a 345 million parameter GPT model
We use [JupyterLab](https://jupyterlab.readthedocs.io) for this example.
To start JupyterLab, run
```
jupyter lab .
```
and open [peft.ipynb](./peft.ipynb).

#### Hardware requirement
This example requires a GPU with at least 24GB memory to run three clients in parallel on the same GPU.
