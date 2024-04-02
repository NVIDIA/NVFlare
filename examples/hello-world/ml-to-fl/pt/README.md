# PyTorch Deep Learning to Federated Learning transition with NVFlare

We will demonstrate how to transform an existing DL code into an FL application step-by-step:

  1. [Show a baseline training script](#the-baseline)
  2. [How to modify an existing training script using DL2FL Client API](#transform-cifar10-dl-training-code-to-fl-including-best-model-selection-using-client-api)
  3. [How to modify a structured script using DL2FL decorator](#the-decorator-use-case)
  4. [How to modify a PyTorch Lightning script using DL2FL Lightning Client API](#transform-cifar10-pytorch-lightning-training-code-to-fl-with-nvflare-client-lightning-integration-api)

If you have multi GPU please refer to the following examples:

  1. [How to modify a PyTorch DDP training script using DL2FL Client API](#transform-cifar10-pytorch--ddp-training-code-to-fl-using-client-api)
  2. [How to modify a PyTorch Lightning DDP training script using DL2FL Lightning Client API](#transform-cifar10-pytorch-lightning--ddp-training-code-to-fl-with-nvflare-client-lightning-integration-api)

## Software Requirements

Please install the requirements first, it is suggested to install inside a virtual environment:

```bash
pip install -r requirements.txt
```

Please also configure the job templates folder:

```bash
nvflare config -jt ../../../../job_templates/
nvflare job list_templates
```

## Minimum Hardware Requirements

Each example has different requirements:

| Example name | minimum requirements |
| ------------ | -------------------- |
| [Show a baseline training script](#the-baseline) | 1 CPU or 1 GPU* |
| [How to modify an existing training script using DL2FL Client API](#transform-cifar10-dl-training-code-to-fl-including-best-model-selection-using-client-api) | 1 CPU or 1 GPU* |
| [How to modify a structured script using DL2FL decorator](#the-decorator-use-case) | 1 CPU or 1 GPU* |
| [How to modify a PyTorch Lightning script using DL2FL Lightning Client API](#transform-cifar10-pytorch-lightning-training-code-to-fl-with-nvflare-client-lightning-integration-api) | 1 CPU or 1 GPU* |
| [How to modify a PyTorch DDP training script using DL2FL Client API](#transform-cifar10-pytorch--ddp-training-code-to-fl-using-client-api) | 2 GPUs |
| [How to modify a PyTorch Lightning DDP training script using DL2FL Lightning Client API](#transform-cifar10-pytorch-lightning--ddp-training-code-to-fl-with-nvflare-client-lightning-integration-api) | 2 CPUs or 2 GPUs** |


\* it depends on you use `device=cpu` or `device=cuda`
\*\* it depends on whether `torch.cuda.is_available()` is True or not

## The baseline

We take a CIFAR10 example directly from [PyTorch website](https://github.com/pytorch/tutorials/blob/main/beginner_source/blitz/cifar10_tutorial.py) and do the following cleanups to get [cifar10_original.py](./code/cifar10_original.py):

1. Remove the comments
2. Move the definition of Convolutional Neural Network to [net.py](./code/net.py)
3. Wrap the whole code inside a main method (https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods)
4. Add the ability to run on GPU to speed up the training process (optional)

You can run the baseline using

```bash
python3 ./code/cifar10_original.py
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

The modified code can be found in [./code/cifar10_fl.py](./code/cifar10_fl.py)

After we modify our training script, we need to put it into a [job structure](https://nvflare.readthedocs.io/en/latest/real_world_fl/job.html) so that NVFlare system knows how to deploy and run the job.

Please refer to [JOB CLI tutorial](../../../tutorials/job_cli.ipynb) on how to generate a job easily from our existing job templates.

We will use the in-process client API, we choose the [sag_pt in_proc job template](../../../../job_templates/sag_pt_in_proc) and run the following command to create the job:

```bash
nvflare job create -force -j ./jobs/client_api -w sag_pt_in_proc -sd ./code/ \
    -f config_fed_client.conf app_script=cifar10_fl.py
```

Then we can run it using the NVFlare Simulator:

```bash
bash ./prepare_data.sh
nvflare simulator -n 2 -t 2 ./jobs/client_api -w client_api_workspace
```

Congratulations! You have finished an FL training!


## The Decorator use case

The above case shows how you can change an existing DL code to FL.

Usually, people have already put their codes into "train", "evaluate", and "test" methods, so they can reuse them.
In that case, the NVFlare DL2FL decorator is the way to go.

To structure the code, we make the following changes to [./code/cifar10_original.py](./code/cifar10_original.py):

1. Wrap training logic into a ``train`` method
2. Wrap evaluation logic into an ``evaluate`` method
3. Call train method and evaluate method

The result is [./code/cifar10_structured_original.py](./code/cifar10_structured_original.py)

To modify this structured code to be used in FL.
We made the following changes:

1. Import NVFlare Client API: ```import nvflare.client as flare```
2. Initialize NVFlare Client API: ```flare.init()```
3. Modify the ``train`` method:
    - Decorate with ```@flare.train```
    - Take additional argument in the beginning
    - Load the received aggregated/global model weights into the model structure: ```net.load_state_dict(input_model.params)```
    - Return an FLModel object
4. Add an ```fl_evaluate``` method:
    - Decorate with ```@flare.evaluate```
    - The first argument is input FLModel
    - Return a float number of metric
5. Receive aggregated/global FLModel from NVFlare side each round: ```input_model = flare.receive()```
6. Call ```fl_evaluate``` method before training to get metrics on the received aggregated/global model

Optional: Change the data path to an absolute path and use ```./prepare_data.sh``` to download data

The modified code can be found in [./code/cifar10_structured_fl.py](./code/cifar10_structured_fl.py)


We choose the [sag_pt job template](../../../../job_templates/sag_pt) and run the following command to create the job:

```bash
nvflare job create -force -j ./jobs/decorator -w sag_pt -sd ./code/ -f config_fed_client.conf app_script=cifar10_structured_fl.py
```

Then we can run it using the NVFlare simulator:

```bash
bash ./prepare_data.sh
nvflare simulator -n 2 -t 2 ./jobs/decorator -w decorator_workspace
```

## Transform CIFAR10 PyTorch Lightning training code to FL with NVFLARE Client lightning integration API

If you are using [PyTorch Lightning](https://lightning.ai/) to write your training scripts, you can use our NVFlare lightning client API to convert it into FL.

Given a CIFAR10 PyTorch Lightning code example: [./code/cifar10_lightning_original.py](./code/cifar10_lightning_original.py).
Notice we wrap the [Net class](./code/net.py) into LightningModule: [LitNet class](./code/lit_net.py)

You can run it using

```bash
python3 ./code/cifar10_lightning_original.py
```


To transform the existing code to FL training code, we made the following changes:

1. Import NVFlare Lightning Client API: ```import nvflare.client.lightning as flare```
2. Patch the PyTorch Lightning trainer ```flare.patch(trainer)```
3. Receive aggregated/global FLModel from NVFlare side each round: ```input_model = flare.receive()```
4. Call trainer.evaluate() method to evaluate newly received aggregated/global model. The resulting evaluation metric will be used for the best model selection

The modified code can be found in [./code/cifar10_lightning_fl.py](./code/cifar10_lightning_fl.py)

Then we can create the job using sag_pt_in_proc template:

```bash
nvflare job create -force -j ./jobs/lightning -w sag_pt_in_proc -sd ./code/ \
    -f config_fed_client.conf app_script=cifar10_lightning_fl.py \
    -f config_fed_server.conf key_metric=val_acc_epoch model_class_path=lit_net.LitNet
```

Note that we pass the "key_metric"="val_acc_epoch" (this name originates from the code [here](./code/lit_net.py#L58))
which means the validation accuracy for that epoch.

And we use "lit_net.LitNet" instead of "net.Net" for model class.

Then we run it using the NVFlare simulator:

```bash
bash ./prepare_data.sh
nvflare simulator -n 2 -t 2 ./jobs/lightning -w lightning_workspace
```


## Transform CIFAR10 PyTorch + DDP training code to FL using Client API

We follow the official [PyTorch documentation](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#initialize-ddp-with-torch-distributed-run-torchrun) and write a [./code/cifar10_ddp_original.py](./code/cifar10_ddp_original.py).

Note that we wrap the evaluation logic into a method for better usability.

It can be run using the torch distributed run:

```bash
python3 -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=6666 ./code/cifar10_ddp_original.py
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

The modified code can be found in [./code/cifar10_ddp_fl.py](./code/cifar10_ddp_fl.py)

In this example, we are going to to use a different job template, where we leverage Client API with sub-process launcher
instead of in-process launcher in other examples. Here is the command we use to create the job: 

```bash
nvflare job create -force -j ./jobs/client_api_ddp -w sag_pt_deploy_map -sd ./code/ \
    -f app_1/config_fed_client.conf script="python3 -m torch.distributed.run --nnodes\=1 --nproc_per_node\=2 --master_port\=7777 custom/cifar10_ddp_fl.py" \
    -f app_2/config_fed_client.conf script="python3 -m torch.distributed.run --nnodes\=1 --nproc_per_node\=2 --master_port\=8888 custom/cifar10_ddp_fl.py"
```


Then we run it using the NVFlare simulator:

```bash
bash ./prepare_data.sh
nvflare simulator -n 2 -t 2 ./jobs/client_api_ddp -w client_api_ddp_workspace
```


This will start 2 clients and each client will start 2 worker processes.

Note that you might need to change the "master_port" in the "config_fed_client.conf"
 if those ports are already taken on your machine.


## Transform CIFAR10 PyTorch Lightning + ddp training code to FL with NVFLARE Client lightning integration API

After we finish the [single GPU case](#transform-cifar10-pytorch-lightning-training-code-to-fl-with-nvflare-client-lightning-integration-api), we will
show how to convert multi GPU training as well.

We just need to change the Trainer initialize to add extra options: `strategy="ddp", devices=2` 

The modified Lightning + DPP code can be found in [./code/cifar10_lightning_ddp_original.py](./code/cifar10_lightning_ddp_original.py)

You can execute it using:

```bash
python3 ./code/cifar10_lightning_ddp_original.py
```

The modified FL code can be found in [./code/cifar10_lightning_ddp_fl.py](./code/cifar10_lightning_ddp_fl.py)

Then we can create the job using sag_pt template:

```bash
nvflare job create -force -j ./jobs/lightning_ddp -w sag_pt -sd ./code/ \
    -f config_fed_client.conf app_script=cifar10_lightning_ddp_fl.py \
    -f config_fed_server.conf key_metric=val_acc_epoch model_class_path=lit_net.LitNet
```

Note that we pass the "key_metric"="val_acc_epoch" (this name originates from the code [here](./code/lit_net.py#L58))
which means the validation accuracy for that epoch.

And we use "lit_net.LitNet" instead of "net.Net" for model class.

Then we run it using the NVFlare simulator:

```bash
bash ./prepare_data.sh
nvflare simulator -n 2 -t 2 ./jobs/lightning_ddp -w lightning_ddp_workspace
```
