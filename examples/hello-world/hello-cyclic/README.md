# Hello Cyclic Weight Transfer

["Cyclic Weight Transfer"](https://pubmed.ncbi.nlm.nih.gov/29617797/
) (CWT) is an alternative to [FedAvg](https://arxiv.org/abs/1602.05629). CWT uses the [Cyclic Controller](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.app_common.workflows.cyclic.html) to pass the model weights from one site to the next for repeated fine-tuning.

> **_NOTE:_** This example uses the [MNIST](http://yann.lecun.com/exdb/mnist/) handwritten digits dataset and will load its data within the trainer code.

To run this example with the FLARE API, you can follow the [hello_world notebook](../hello_world.ipynb), or you can quickly get
started with the following:

### 1. Install NVIDIA FLARE

Follow the [Installation](../../getting_started/README.md) instructions to install NVFlare.

Install additional requirements (if you already have a specific version of nvflare installed in your environment, you may want to remove nvflare in the requirements to avoid reinstalling nvflare):

```
pip3 install tensorflow
```

### 2. Run the experiment

Prepare the data first:

```
bash ./prepare_data.sh
```

Run the script using the job API to create the job and run it with the simulator:

```
python3 cyclic_script_runner.py
```

### 3. Access the logs and results

You can find the running logs and results inside the simulator's workspace:

```bash
$ ls /tmp/nvflare/jobs/workdir
```

### 4. Notes on running with GPUs

For running with GPUs, we recommend using the
[NVIDIA TensorFlow docker](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow)

If you choose to run the example using GPUs, it is important to note that by default, TensorFlow will attempt to allocate all available GPU memory at the start.
In scenarios where multiple clients are involved, you have a couple of options to address this.

One approach is to include specific flags to prevent TensorFlow from allocating all GPU memory.
For instance:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true python3 cyclic_script-executor-hello-cyclic.py
```
