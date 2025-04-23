# Hello Cyclic Weight Transfer

["Cyclic Weight Transfer"](https://pubmed.ncbi.nlm.nih.gov/29617797/
) (CWT) is an alternative to [FedAvg](https://arxiv.org/abs/1602.05629). CWT uses the [Cyclic Controller](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.app_common.workflows.cyclic.html) to pass the model weights from one site to the next for repeated fine-tuning.

> **_NOTE:_** This example uses the [MNIST](http://yann.lecun.com/exdb/mnist/) handwritten digits dataset and will load its data within the trainer code.

We recommend to use [NVIDIA TensorFlow docker](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow) if you want to use GPU.
If you don't need to run using GPU, you can just use python virtual environment.

To run this example with the FLARE API, you can follow the [hello_world notebook](../hello_world.ipynb).

## Run NVIDIA TensorFlow container

Please install the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) first.
Then run the following command:

```bash
docker run --gpus=all -it --rm -v [path_to_NVFlare]:/NVFlare nvcr.io/nvidia/tensorflow:xx.xx-tf2-py3
```

## Install NVFlare

```bash
pip3 install nvflare
```

## Run the experiment

Prepare the data first:

```
bash ./prepare_data.sh
```

Run the script using the job API to create the job and run it with the simulator:

```
python3 cyclic_script_runner.py
```

## Access the logs and results

You can find the running logs and results inside the simulator's workspace:

```bash
$ ls /tmp/nvflare/jobs/workdir
```

### Notes on running with GPUs

If you choose to run the example using GPUs, it is important to note that,
by default, TensorFlow will attempt to allocate all available GPU memory at the start.
In scenarios where multiple clients are involved, you have to prevent TensorFlow from allocating all GPU memory 
by setting the following flags.
```bash
TF_FORCE_GPU_ALLOW_GROWTH=true TF_GPU_ALLOCATOR=cuda_malloc_async
```

If you possess more GPUs than clients, a good strategy is to run one client on each GPU.
This can be achieved by using the `--gpu` argument during simulation, e.g., `nvflare simulator -n 2 --gpu 0,1 [job]`.
