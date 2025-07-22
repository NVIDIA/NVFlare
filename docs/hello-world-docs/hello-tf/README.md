# Hello TensorFlow

Example of using [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) to train an image classifier
using federated averaging ([FedAvg](https://arxiv.org/abs/1602.05629))
and [TensorFlow](https://tensorflow.org/) as the deep learning training framework.

> **_NOTE:_** This example uses the [MNIST](https://www.tensorflow.org/datasets/catalog/mnist) handwritten digits dataset and will load its data within the trainer code.

See the [Hello TensorFlow](https://nvflare.readthedocs.io/en/main/examples/hello_tf_job_api.html#hello-tf-job-api) example documentation page for details on this
example.

We recommend to use [NVIDIA TensorFlow docker](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow) if you want to use GPU.
If you don't need to run using GPU, you can just use python virtual environment.

To run this example with the FLARE API, you can follow the [hello_world notebook](../hello-world/hello_world.ipynb).

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

Run the script using the job API to create the job and run it with the simulator:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true TF_GPU_ALLOCATOR=cuda_malloc_async python3 fedavg_script_runner_tf.py
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
