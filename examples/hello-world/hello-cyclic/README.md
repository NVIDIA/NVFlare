# Hello Cyclic Weight Transfer

["Cyclic Weight Transfer"](https://pubmed.ncbi.nlm.nih.gov/29617797/
) (CWT) is an alternative to the scatter-and-gather approach used in [FedAvg](https://arxiv.org/abs/1602.05629). CWT uses the [CyclicController](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.app_common.workflows.cyclic_ctl.html) to pass the model weights from one site to the next for repeated fine-tuning.

> **_NOTE:_** This example uses the [MNIST](http://yann.lecun.com/exdb/mnist/) handwritten digits dataset and will load its data within the trainer code.

To run this example with the FLARE API, you can follow the [hello_world notebook](../hello_world.ipynb), or you can quickly get
started with the following:

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

Run the script using the job API to create the job and run it with the simulator:

```
python3 cyclic_script_executor_hello-cyclic.py
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
For instance if you are running the job with the simulator after exporting it with `job.export_job("/tmp/nvflare/jobs/job_config")` uncommented at the end of script in this directory:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true nvflare simulator -w /tmp/nvflare/ -n 2 -t 2 /tmp/nvflare/jobs/job_config/hello-tf_cyclic
```

If you possess more GPUs than clients,
an alternative strategy is to run one client on each GPU.
This can be achieved as illustrated below:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true nvflare simulator -w /tmp/nvflare/ -n 2 -gpu 0,1 /tmp/nvflare/jobs/job_config/hello-tf_cyclic
```
