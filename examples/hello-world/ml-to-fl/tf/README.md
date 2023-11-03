# TensorFlow Deep Learning to Federated Learning transition with NVFlare

Please install the requirements first, it is suggested to install inside a virtual environment:

```bash
pip install -r requirements.txt
```

Note that for running with GPUs, we recommend using [NVIDIA TensorFlow docker](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow)

We will demonstrate how to transform an existing DL code into an FL application step-by-step:

1. [How to modify an existing training script using DL2FL Client API](#transform-cifar10-tensorflow-training-code-to-fl-with-nvflare-client-api)

2. [How to modify an existing multi GPU training script using DL2FL Client API](#transform-cifar10-tensorflow-multi-gpu-training-code-to-fl-with-nvflare-client-api)

## Transform CIFAR10 TensorFlow training code to FL with NVFLARE Client API

Given a TensorFlow CIFAR10 example: [./code/cifar10_tf_original.py](./code/cifar10_tf_original.py).

You can run it using

```bash
python3 ./code/cifar10_tf_original.py
```

To transform the existing code to FL training code, we made the following changes:

1. Import NVFlare Client API: ```import nvflare.client as flare```
2. Initialize NVFlare Client API: ```flare.init()```
3. Receive aggregated/global FLModel from NVFlare side: ```input_model = flare.receive()```
4. Load the received aggregated/global model weights into the model structure: ```model.get_layer(k).set_weights(v)```
5. Evaluate on the model with aggregated/global weights loaded to get the metrics for model selection: via regular ```evaluate()``` function
6. Construct the FLModel to be returned to the NVFlare side: ```output_model = flare.FLModel(params={layer.name: layer.get_weights() for layer in model.layers}, xxx)```
7. Send the model back to NVFlare: ```flare.send(output_model)```

Notice that we need to get / load the model weights as a ``dict`` of arrays because we want to reuse existing NVFlare components.

The modified code can be found here: [./code/cifar10_tf_fl.py](./code/cifar10_tf_fl.py), [./code/tf_net.py](./code/tf_net.py).

After we modify our training script, we need to put it into a [job structure](https://nvflare.readthedocs.io/en/latest/real_world_fl/job.html) so that NVFlare system knows how to deploy and run the job.

Please refer to [JOB CLI tutorial](../../../tutorials/job_cli.ipynb) on how to generate a job easily from our existing job templates.


We choose the [tensorflow job template](../../../../job_templates/sag_tf/) and run the following command to create the job:

```bash
nvflare config -jt ../../../../job_templates
nvflare job create -force -j ./jobs/tensorflow -w sag_tf -sd ./code/ -f config_fed_client.conf app_script=cifar10_tf_fl.py
```

Then we can run the job using the simulator:

```bash
bash ./prepare_data.sh
nvflare simulator -n 2 -t 2 ./jobs/tensorflow -w tensorflow_workspace
```


## Transform CIFAR10 TensorFlow multi GPU training code to FL with NVFLARE Client API

Following the [official documentation](https://www.tensorflow.org/guide/keras/distributed_training#single-host_multi-device_synchronous_training), we modified the single 
device TensorFlow CIFAR10 example: [./code/cifar10_tf_original.py](./code/cifar10_tf_original.py) to
a multi-device version: [./code/cifar10_tf_multi_gpu_original.py](./code/cifar10_tf_multi_gpu_original.py)

You can run it using

```bash
python3 ./code/cifar10_tf_multi_gpu_original.py
```

To transform the existing multi-gpu code to FL training code, we can apply the same changes as in [single GPU case](#transform-cifar10-tensorflow-training-code-to-fl-with-nvflare-client-api).

The modified code can be found here: [./code/cifar10_tf_multi_gpu_fl.py](./code/cifar10_tf_multi_gpu_fl.py).

After we modify our training script, we need to put it into a [job structure](https://nvflare.readthedocs.io/en/latest/real_world_fl/job.html) so that NVFlare system knows how to deploy and run the job.

Please refer to [JOB CLI tutorial](../../../tutorials/job_cli.ipynb) on how to generate a job easily from our existing job templates.


We choose the [tensorflow job template](../../../../job_templates/sag_tf/) and run the following command to create the job:

```bash
nvflare config -jt ../../../../job_templates
nvflare job create -force -j ./jobs/tensorflow_multi_gpu -w sag_tf -sd ./code/ -f config_fed_client.conf app_script=cifar10_tf_multi_gpu_fl.py
```

Then we can run the job using the simulator:

```bash
bash ./prepare_data.sh
TF_GPU_ALLOCATOR=cuda_malloc_async nvflare simulator -n 2 -t 2 ./jobs/tensorflow_multi_gpu -w tensorflow_multi_gpu_workspace
```

Note that the flag "TF_GPU_ALLOCATOR=cuda_malloc_async" is only needed if you are going to run more than one process in the same GPU.
