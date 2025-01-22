# TensorFlow Deep Learning to Federated Learning transition with NVFlare

We will demonstrate how to transform an existing DL code into an FL application step-by-step:

1. [How to modify an existing training script using the DL2FL Client API](#transform-cifar10-tensorflow-training-code-to-fl-with-nvflare-client-api)

2. [How to modify an existing multi GPU training script using DL2FL Client API](#transform-cifar10-tensorflow-multi-gpu-training-code-to-fl-with-nvflare-client-api)

## Software Requirements

Please install the requirements first. It is suggested to install them inside a virtual environment.

```bash
pip install -r requirements.txt
```

## Minimum Hardware Requirements

| Example name | minimum requirements |
| ------------ | -------------------- |
| [How to modify an existing training script using DL2FL Client API](#transform-cifar10-tensorflow-training-code-to-fl-with-nvflare-client-api) | 1 CPU or 1 GPU* |
| [How to modify an existing multi GPU training script using DL2FL Client API](#transform-cifar10-tensorflow-multi-gpu-training-code-to-fl-with-nvflare-client-api) | 2 CPUs or 2 GPUs* |

\* depends on whether TF can found GPU or not


## Notes on running with GPUs

For running with GPUs, we recommend using
[NVIDIA TensorFlow docker](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow)

If you choose to run the example using GPUs, it is important to note that,
by default, TensorFlow will attempt to allocate all available GPU memory at the start.
In scenarios where multiple clients are involved, you have to prevent TensorFlow from allocating all GPU memory 
by setting the following flags.
```bash
TF_FORCE_GPU_ALLOW_GROWTH=true TF_GPU_ALLOCATOR=cuda_malloc_async
```

If you possess more GPUs than clients, a good strategy is to run one client on each GPU.
This can be achieved by using the `-gpu` argument during simulation, e.g., `nvflare simulator -n 2 -gpu 0,1 [job]`.


## Transform CIFAR10 TensorFlow training code to FL with NVFLARE Client API

Given a TensorFlow CIFAR-10 example: [./src/cifar10_tf_original.py](./src/cifar10_tf_original.py).

You can run it using

```bash
python3 ./src/cifar10_tf_original.py
```

To transform the existing code into FL training code, we made the following changes:

1. Import NVFlare Client API: ```import nvflare.client as flare```
2. Initialize NVFlare Client API: ```flare.init()```
3. Receive aggregated/global FLModel from NVFlare side: ```input_model = flare.receive()```
4. Load the received aggregated/global model weights into the model structure: ```model.get_layer(k).set_weights(v)```
5. Evaluate on the model with aggregated/global weights loaded to get the metrics for model selection: via regular ```evaluate()``` function
6. Construct the FLModel to be returned to the NVFlare side: ```output_model = flare.FLModel(params={layer.name: layer.get_weights() for layer in model.layers}, xxx)```
7. Send the model back to NVFlare: ```flare.send(output_model)```

Notice that we need to get / load the model weights as a ``dict`` of arrays because we want to reuse existing NVFlare components.

The modified code can be found here: [./src/cifar10_tf_fl.py](./src/cifar10_tf_fl.py), [./src/tf_net.py](./src/tf_net.py).

After we modify our training script, we can create a job using the in-process ScriptRunner: [tf_client_api_job.py](tf_client_api_job.py).
(Please refer to [FedJob API](https://nvflare.readthedocs.io/en/main/programming_guide/fed_job_api.html) for more details on formulating a job)

Then we can run the job using the simulator:

```bash
bash ./prepare_data.sh
TF_FORCE_GPU_ALLOW_GROWTH=true TF_GPU_ALLOCATOR=cuda_malloc_async python3 tf_client_api_job.py --script src/cifar10_tf_fl.py
```


## Transform CIFAR10 TensorFlow multi GPU training code to FL with NVFLARE Client API

Following the [official documentation](https://www.tensorflow.org/guide/keras/distributed_training#single-host_multi-device_synchronous_training), we modified the single 
device TensorFlow CIFAR10 example: [./src/cifar10_tf_original.py](./src/cifar10_tf_original.py) to
a multi-device version: [./src/cifar10_tf_multi_gpu_original.py](./src/cifar10_tf_multi_gpu_original.py)

You can run it using

```bash
python3 ./src/cifar10_tf_multi_gpu_original.py
```

To transform the existing multi-gpu code to FL training code, we can apply the same changes as in [single GPU case](#transform-cifar10-tensorflow-training-code-to-fl-with-nvflare-client-api).

The modified code can be found here: [./src/cifar10_tf_multi_gpu_fl.py](./src/cifar10_tf_multi_gpu_fl.py).

After we modify our training script, we can create a job using the ScriptRunner to launch our script: [tf_client_api_job.py](tf_client_api_job.py).

Then we can run the job using the simulator:

```bash
bash ./prepare_data.sh
TF_FORCE_GPU_ALLOW_GROWTH=true TF_GPU_ALLOCATOR=cuda_malloc_async python3 tf_client_api_job.py --script src/cifar10_tf_multi_gpu_fl.py --launch_process
```
