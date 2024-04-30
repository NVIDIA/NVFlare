# Hello Cyclic Weight Transfer

["Cyclic Weight Transfer"](https://pubmed.ncbi.nlm.nih.gov/29617797/
) (CWT) is an alternative to the scatter-and-gather approach used in [FedAvg](https://arxiv.org/abs/1602.05629). CWT uses the [CyclicController](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.app_common.workflows.cyclic_ctl.html) to pass the model weights from one site to the next for repeated fine-tuning.

> **_NOTE:_** This example uses the [MNIST](http://yann.lecun.com/exdb/mnist/) handwritten digits dataset and will load its data within the trainer code.

You can follow the [hello_world notebook](../hello_world.ipynb) or the following:

### 1. Install NVIDIA FLARE

Follow the [Installation](https://nvflare.readthedocs.io/en/main/quickstart.html) instructions to install NVFlare.

Install additional requirements:

```
pip3 install tensorflow
```

### 2. Run the experiment

Prepare the data first:

```
bash ./prepare_data.sh
```

Use nvflare simulator to run the hello-examples:

```bash
nvflare simulator -w /tmp/nvflare/ -n 2 -t 2 ./jobs/hello-cyclic
```

### 3. Access the logs and results

You can find the running logs and results inside the simulator's workspace/simulate_job

```bash
$ ls /tmp/nvflare/simulate_job/
app_server  app_site-1  app_site-2  log.txt

```

### 4. Notes on running with GPUs

For running with GPUs, we recommend using
[NVIDIA TensorFlow docker](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow)

If you choose to run the example using GPUs, it is important to note that,
by default, TensorFlow will attempt to allocate all available GPU memory at the start.
In scenarios where multiple clients are involved, you have a couple of options to address this.

One approach is to include specific flags to prevent TensorFlow from allocating all GPU memory.
For instance:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true nvflare simulator -w /tmp/nvflare/ -n 2 -t 2 ./jobs/hello-cyclic
```

If you possess more GPUs than clients,
an alternative strategy is to run one client on each GPU.
This can be achieved as illustrated below:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true nvflare simulator -w /tmp/nvflare/ -n 2 -gpu 0,1 ./jobs/hello-cyclic
```
