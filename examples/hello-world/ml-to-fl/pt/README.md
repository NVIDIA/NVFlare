# PyTorch Deep Learning to Federated Learning transition with NVFlare

We will demonstrate how to transform an existing DL code into an FL application step-by-step:

  1. [Show a baseline training script](#the-baseline)
  2. [How to modify an existing training script using DL2FL Client API](#transform-cifar10-dl-training-code-to-fl-including-best-model-selection-using-client-api)
  3. [How to modify a structured script using DL2FL decorator](#the-decorator-use-case)
  4. [How to modify a PyTorch Lightning script using DL2FL Lightning Client API](#transform-cifar10-pytorch-lightning-training-code-to-fl-with-nvflare-client-lightning-integration-api)

Please install the requirements first, it is suggested to install inside a virtual environment:

```bash
pip install -r requirements.txt
```

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

It will run for 2 epochs and you will see something like this:

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
3. Receive aggregated/global FLModel from NVFlare side: ```input_model = flare.receive()```
4. Load the received aggregated/global model weights into the model structure: ```net.load_state_dict(input_model.params)```
5. Wrap evaluation logic into a method to re-use for evaluation on both trained and received aggregated/global model
6. Evaluate on received aggregated/global model to get the metrics for model selection
7. Construct the FLModel to be returned to the NVFlare side: ```output_model = flare.FLModel(xxx)```
8. Send the model back to NVFlare: ```flare.send(output_model)```

Optional: Change the data path to an absolute path and use ```../prepare_data.sh``` to download data

The modified code can be found in [./code/cifar10_fl.py](./code/cifar10_fl.py)

After we modify our training script, we need to put it into a [job structure](https://nvflare.readthedocs.io/en/latest/real_world_fl/job.html) so that NVFlare system knows how to deploy and run the job.

Please refer to [JOB CLI tutorial](../../../tutorials/job_cli.ipynb) on how to generate a job easily from our existing job templates.

We choose the [sag_pt job template](../../../../job_templates/sag_pt/) and run the following command to create the job:

```bash
nvflare config -jt ../../../../job_templates/
nvflare job list_templates
nvflare job create -force -j ./jobs/client_api -w sag_pt -sd ./code/ -s ./code/cifar10_fl.py
```

Then we can run it using the NVFlare Simulator:

```bash
bash ../prepare_data.sh
nvflare simulator -n 2 -t 2 ./jobs/client_api -w client_api_workspace
```

Congratulations! You have finished an FL training!

## The Decorator use case

The above case shows how you can change an existing DL code to FL.

Usually, people have already put their codes into "train", "evaluate", and "test" methods so they can reuse them.
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
5. Call ```fl_evaluate``` method before training to get metrics on the received aggregated/global model

Optional: Change the data path to an absolute path and use ```../prepare_data.sh``` to download data

The modified code can be found in [./code/cifar10_structured_fl.py](./code/cifar10_structured_fl.py)


We choose the [sag_pt job template](../../../../job_templates/sag_pt/) and run the following command to create the job:

```bash
nvflare job create -force -j ./jobs/decorator -w sag_pt -sd ./code/ -s ./code/cifar10_structured_fl.py
```

Then we can run it using the NVFlare simulator:

```bash
bash ../prepare_data.sh
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
3. Call trainer.evaluate() method to evaluate newly received aggregated/global model. The resulting evaluation metric will be used for the best model selection

The modified code can be found in [./code/cifar10_lightning_fl.py](./code/cifar10_lightning_fl.py)

Then we can create the job using sag_pt template:

```bash
nvflare job create -force -j ./jobs/lightning -w sag_pt -sd ./code/ -s ./code/cifar10_lightning_fl.py
```

We need to modify the "key_metric" in "config_fed_server.conf" from "accuracy" to "val_acc_epoch" (this name originates from the code [here](./code/lit_net.py#L56)) which means the validation accuracy for that epoch:

```
{
  id = "model_selector"
  name = "IntimeModelSelector"
  args {
    key_metric = "val_acc_epoch"
  }
}
```

And we modify the model architecture to use the LitNet class:

```
{
  id = "persistor"
  path = "nvflare.app_opt.pt.file_model_persistor.PTFileModelPersistor"
  args {
    model {
      path = "lit_net.LitNet"
    }
  }
}
```

Then we run it using the NVFlare simulator:

```bash
bash ../prepare_data.sh
nvflare simulator -n 2 -t 2 ./jobs/lightning -w lightning_workspace
```
