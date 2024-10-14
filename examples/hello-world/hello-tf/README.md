# Hello TensorFlow

Example of using [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) to train an image classifier
using federated averaging ([FedAvg](https://arxiv.org/abs/1602.05629))
and [TensorFlow](https://tensorflow.org/) as the deep learning training framework.

> **_NOTE:_** This example uses the [MNIST](https://www.tensorflow.org/datasets/catalog/mnist) handwritten digits dataset and will load its data within the trainer code.

See the [Hello TensorFlow](https://nvflare.readthedocs.io/en/main/examples/hello_tf_job_api.html#hello-tf-job-api) example documentation page for details on this
example.

To run this example with the FLARE API, you can follow the [hello_world notebook](../hello_world.ipynb), or you can quickly get
started with the following:

### 1. Install NVIDIA FLARE

Follow the [Installation](../../getting_started/README.md) instructions to install NVFlare.

Install additional requirements (if you already have a specific version of nvflare installed in your environment, you may want to remove nvflare in the requirements to avoid reinstalling nvflare):

```
pip3 install tensorflow
```

### 2. Run the experiment

Run the script using the job API to create the job and run it with the simulator:

```
python3 fedavg_script_runner_tf.py
```

### 3. Access the logs and results

You can find the running logs and results inside the simulator's workspace:

```bash
$ ls /tmp/nvflare/jobs/workdir
```

#### Notes on running with GPUs

For running with GPUs, we recommend using
[NVIDIA TensorFlow docker](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow)

If you choose to run the example using GPUs, it is important to note that by default, TensorFlow will attempt to allocate all available GPU memory at the start.
In scenarios where multiple clients are involved, you have to prevent TensorFlow from allocating all GPU memory 
by setting the following flags.

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true TF_GPU_ALLOCATOR=cuda_malloc_async python3 fedavg_script_runner_tf.py
```

If you possess more GPUs than clients, a good strategy is to run one client on each GPU.
This can be achieved using the `-gpu` argument if using the nvflare simulator command, e.g., `nvflare simulator -n 2 -gpu 0,1 [job]`.
