# PyTorch Deep Learning to Federated Learning transition with NVFlare

We will demonstrate how to transform an existing DL code into an FL application step-by-step:

  1. [Show a baseline training script](#the-baseline)
  2. [How to modify an existing training script using DL2FL Client API](#transform-cifar10-dl-training-code-to-fl-including-best-model-selection-using-client-api)
  3. [How to modify a PyTorch Lightning script using DL2FL Lightning Client API](#transform-cifar10-pytorch-lightning-training-code-to-fl-with-nvflare-client-lightning-integration-api)

If you have multi GPU please refer to the following examples:

  1. [How to modify a PyTorch DDP training script using DL2FL Client API](#transform-cifar10-pytorch--ddp-training-code-to-fl-using-client-api)
  2. [How to modify a PyTorch Lightning DDP training script using DL2FL Lightning Client API](#transform-cifar10-pytorch-lightning--ddp-training-code-to-fl-with-nvflare-client-lightning-integration-api)

## Software Requirements

Please install the requirements first, it is suggested to install inside a virtual environment:

```bash
pip install -r requirements.txt
```

## Minimum Hardware Requirements

Each example has different requirements:

| Example name | minimum requirements |
| ------------ | -------------------- |
| [Show a baseline training script](#the-baseline) | 1 CPU or 1 GPU* |
| [How to modify an existing training script using DL2FL Client API](#transform-cifar10-dl-training-code-to-fl-including-best-model-selection-using-client-api) | 1 CPU or 1 GPU* |
| [How to modify a PyTorch Lightning script using DL2FL Lightning Client API](#transform-cifar10-pytorch-lightning-training-code-to-fl-with-nvflare-client-lightning-integration-api) | 1 CPU or 1 GPU* |
| [How to modify a PyTorch DDP training script using DL2FL Client API](#transform-cifar10-pytorch--ddp-training-code-to-fl-using-client-api) | 2 GPUs |
| [How to modify a PyTorch Lightning DDP training script using DL2FL Lightning Client API](#transform-cifar10-pytorch-lightning--ddp-training-code-to-fl-with-nvflare-client-lightning-integration-api) | 2 CPUs or 2 GPUs** |


\* it depends on you use `device=cpu` or `device=cuda`
\*\* it depends on whether `torch.cuda.is_available()` is True or not

## The baseline

We take a CIFAR10 example directly from [PyTorch website](https://github.com/pytorch/tutorials/blob/main/beginner_source/blitz/cifar10_tutorial.py) and do the following cleanups to get [cifar10_original.py](./src/cifar10_original.py):

1. Remove the comments
2. Move the definition of Convolutional Neural Network to [net.py](./src/net.py)
3. Wrap the whole code inside a main method (https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods)
4. Add the ability to run on GPU to speed up the training process (optional)

You can run the baseline using

```bash
python3 ./src/cifar10_original.py
```

It will run for 2 epochs.
Then we will see something like this:

```bash
Extracting ./data/cifar-10-python.tar.gz to ./data
Files already downloaded and verified
[1,  2000] loss: 2.127
[1,  4000] loss: 1.826
[1,  6000] loss: 1.667
[1,  8000] loss: 1.568
[1, 10000] loss: 1.503
[1, 12000] loss: 1.455
[2,  2000] loss: 1.386
[2,  4000] loss: 1.362
[2,  6000] loss: 1.348
[2,  8000] loss: 1.329
[2, 10000] loss: 1.327
[2, 12000] loss: 1.275
Finished Training
Accuracy of the network on the 10000 test images: 55 %
```

## Transform CIFAR10 DL training code to FL including best model selection using Client API

Now we have a CIFAR10 DL training code, let's transform it to FL with NVFLARE Client API.


We made the following changes:

1. Import NVFlare Client API: ```import nvflare.client as flare```
2. Initialize NVFlare Client API: ```flare.init()```
3. Receive aggregated/global FLModel from NVFlare side each round: ```input_model = flare.receive()```
4. Load the received aggregated/global model weights into the model structure: ```net.load_state_dict(input_model.params)```
5. Wrap evaluation logic into a method to re-use for evaluation on both trained and received aggregated/global model
6. Evaluate on received aggregated/global model to get the metrics for model selection
7. Construct the FLModel to be returned to the NVFlare side: ```output_model = flare.FLModel(xxx)```
8. Send the model back to NVFlare: ```flare.send(output_model)```

Optional: Change the data path to an absolute path and use ```./prepare_data.sh``` to download data

The modified code can be found in [./src/cifar10_fl.py](./src/cifar10_fl.py)

After we modify our training script, we can create a job using the in-process ScriptRunner: [pt_client_api_job.py](pt_client_api_job.py).
(Please refer to [FedJob API](https://nvflare.readthedocs.io/en/main/programming_guide/fed_job_api.html) for more details on formulating a job)

Then we can run it using the NVFlare Simulator:

```bash
bash ./prepare_data.sh
python3 pt_client_api_job.py --script src/cifar10_fl.py
```

Congratulations! You have finished an FL training!


## Transform CIFAR10 PyTorch Lightning training code to FL with NVFLARE Client lightning integration API

If you are using [PyTorch Lightning](https://lightning.ai/) to write your training scripts, you can use our NVFlare lightning client API to convert it into FL.

Given a CIFAR10 PyTorch Lightning code example: [./src/cifar10_lightning_original.py](./src/cifar10_lightning_original.py).
Notice we wrap the [Net class](./src/net.py) into LightningModule: [LitNet class](./src/lit_net.py)

You can run it using

```bash
python3 ./src/cifar10_lightning_original.py
```


To transform the existing code to FL training code, we made the following changes:

1. Import NVFlare Lightning Client API: ```import nvflare.client.lightning as flare```
2. Patch the PyTorch Lightning trainer ```flare.patch(trainer)```
3. Receive aggregated/global FLModel from NVFlare side each round: ```input_model = flare.receive()```
4. Call trainer.evaluate() method to evaluate newly received aggregated/global model. The resulting evaluation metric will be used for the best model selection

The modified code can be found in [./src/cifar10_lightning_fl.py](./src/cifar10_lightning_fl.py)

After we modify our training script, we can create a job using the in-process ScriptRunner: [pt_client_api_job.py](pt_client_api_job.py).

Note that for PyTorch Lightning we pass the "key_metric"="val_acc_epoch" (this name originates from the code [here](./src/lit_net.py#L58))
which means the validation accuracy for that epoch.

And we use `lit_net.LitNet` instead of `net.Net` for model class.

Then we run it using the NVFlare simulator:

```bash
bash ./prepare_data.sh
python3 pt_client_api_job.py --script src/cifar10_lightning_fl.py --key_metric val_acc_epoch
```

## Transform CIFAR10 PyTorch + DDP training code to FL using Client API

We follow the official [PyTorch documentation](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#initialize-ddp-with-torch-distributed-run-torchrun) and write a [./src/cifar10_ddp_original.py](./src/cifar10_ddp_original.py).

Note that we wrap the evaluation logic into a method for better usability.

It can be run using the torch distributed run:

```bash
python3 -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=6666 ./src/cifar10_ddp_original.py
```

To modify this multi-GPU code to be used in FL.
We made the following changes:

1. Import NVFlare Client API: ```import nvflare.client as flare```
2. Initialize NVFlare Client API: ```flare.init()```
3. Receive aggregated/global FLModel from NVFlare side each round: ```input_model = flare.receive()```
4. Load the received aggregated/global model weights into the model structure: ```net.load_state_dict(input_model.params)```
5. Evaluate on received aggregated/global model to get the metrics for model selection
6. Construct the FLModel to be returned to the NVFlare side: ```output_model = flare.FLModel(xxx)```
7. Send the model back to NVFlare: ```flare.send(output_model)```

Note that we only do flare receive and send on the first process (rank 0).
Because all the worker processes launched by torch distributed will have the same model in the end, we don't need to send duplicate
models back.

The modified code can be found in [./src/cifar10_ddp_fl.py](./src/cifar10_ddp_fl.py)

After we modify our training script, we can create a job using the ex-process ScriptRunner and set the command to the torch.distributed.run: [pt_client_api_job.py](pt_client_api_job.py).

Then we run it using the NVFlare simulator with the ```torch.distributed.run``` with different ports:

```bash
bash ./prepare_data.sh
python3 pt_client_api_job.py --script src/cifar10_ddp_fl.py --launch_process --launch_command 'python3 -m torch.distributed.run --nnodes\=1 --nproc_per_node\=2 --master_port\={PORT}' --ports 7777,8888
```

This will start 2 clients and each client will start 2 worker processes.

Note that you might need to change the ports if they are already taken on your machine.


## Transform CIFAR10 PyTorch Lightning + ddp training code to FL with NVFLARE Client lightning integration API

After we finish the [single GPU case](#transform-cifar10-pytorch-lightning-training-code-to-fl-with-nvflare-client-lightning-integration-api), we will
show how to convert multi GPU training as well.

We just need to change the Trainer initialize to add extra options: `strategy="ddp", devices=2` 

The modified Lightning + DPP code can be found in [./src/cifar10_lightning_ddp_original.py](./src/cifar10_lightning_ddp_original.py)

You can execute it using:

```bash
python3 ./src/cifar10_lightning_ddp_original.py
```

The modified FL code can be found in [./src/cifar10_lightning_ddp_fl.py](./src/cifar10_lightning_ddp_fl.py)

After we modify our training script, we can create a job using the ScriptRunner to launch our script: [pt_client_api_job.py](pt_client_api_job.py).

Note that we pass the "key_metric"="val_acc_epoch" (this name originates from the code [here](./src/lit_net.py#L58))
which means the validation accuracy for that epoch.

And we use `lit_net.LitNet` instead of `net.Net` for model class.

Then we run it using the NVFlare simulator:

```bash
bash ./prepare_data.sh
python3 pt_client_api_job.py --script src/cifar10_lightning_ddp_fl.py --key_metric val_acc_epoch --launch_process
```
