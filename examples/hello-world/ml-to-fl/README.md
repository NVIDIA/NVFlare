# Machine Learning/Deep Learning to Federated Learning transition with NVFlare

Converting Machine Learning (ML) or Deep Learning (DL) to Federated Learning (FL) is not easy, as it involves:

1. Algorithms formulation, how to formulate an ML/DL to FL algorithm and what information needs to be pass between Client and Server

2. Convert existing standalone, centralized ML/DL code to FL code.

3. Configure the workflow to use the newly changed code.

In this example, we assume algorithm formulation is fixed (FedAvg).
We are showing how to quickly convert the centralized DL to FL.
We will demonstrate different techniques depending on the existing code structure and preferences.

For configuring the workflows, one can reference to the config we have here and the documentation.

In this directory, we are providing job configurations to showcase how to utilize 
`LauncherExecutor`, `Launcher` and several NVFlare interfaces to simplify the
transition from your ML/DL code to FL with NVFlare.

We will demonstrate how to transfer an existing DL code into FL application step-by-step:

  1. Show a base line training script [the base line](#the-base-line)
  2. How to modify a non-structured script using ML2FL Client API [the api call use case](#the-api-call-use-case)
  3. How to modify a structured script using ML2FL decorator [the decorator use case](#the-decorator-use-case)
  4. How to modify a structured "lightning" script using ML2FL Lightning Client API [the lightning use case](#the-lightning-use-case)

## The base line

We take a CIFAR10 example directly from [PyTorch website](https://github.com/pytorch/tutorials/blob/main/beginner_source/blitz/cifar10_tutorial.py) and do the following cleanups to get [cifar10_tutorial_clean.py](./cifar10_tutorial_clean.py):

1. Remove the comments
2. Move the definition of Convolutional Neural Network to [net.py](./net.py)
3. Wrap the whole code inside a main method (https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods)
4. Add the ability to run on GPU to speed up the training process (optional)

You can run the baseline using

```bash
python3 cifar10_tutorial_clean.py
```

It will run for 2 epochs and you will see something like:

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

## The API call use case

Now we have a simple DL training script, let's modify it using NVFlare ML-to-FL Client API.

We make the following changes:

1. Import NVFlare Client API: ```import nvflare.client as flare```
2. Initialize NVFlare Client API: ```flare.init()```
3. Receive aggregated/global FLModel from NVFlare side: ```input_model = flare.receive()```
4. Load the received aggregated/global model weights into net: ```net.load_state_dict(input_model.params)``` 
5. Construct the FLModel to be returned to the NVFlare side: ```output_model = flare.FLModel(xxx)```
6. Send the model back to NVFlare: ```flare.send(output_model)```

Optional: Change the data path to an absolute path and use ```prepare_data.sh``` to download data

The modified code can be found in [./jobs/client_api/app/custom/cifar10.py](./jobs/client_api/app/custom/cifar10.py)

After we modify our training script, we need to put it into a [job structure](https://nvflare.readthedocs.io/en/latest/real_world_fl/job.html) so that NVFlare system knows how to deploy and run the job.

Please refer to [JOB CLI example](TODO) on how to generate a job easily from our existing job templates.

We choose the "PTFilePipeLauncherExecutor" job template and make the following changes:

1. Copy the job template as name ```client_api```
2. Copy cifar10.py and net.py inside the ```client_api/app/custom``` folder
3. Update the "script" inside the components part in ```client_api/app/config/config_fed_client.json```:
```json
"components": [
  {
    "id": "launcher",
    "name": "SubprocessLauncher",
    "args": {
      "script": "python custom/cifar10.py"
    }
  }
]
```

Now we have re-write our code and created the [client_api job folder](./jobs/client_api/), we can run it using NVFlare Simulator:

```bash
./prepare_data.sh
nvflare simulator -n 2 -t 2 ./jobs/client_api
```

Congradulations! You have finished an FL training!

## The API call use case with model selection

Continuing on the API call use case, let's add some functionalities that is usually used in FL:

1. Model/Weight difference: Instead of passing the full model weights, it is a common practice to pass just the weight difference
    for privacy protection.

2. Best aggregated/global model selection: We want to know what is the best aggregated model. We could ask each of the clients to evaluate the aggregated model and send back the metrics, we then utilize "IntimeModelSelector" to select the best model based on these reported metrics.

Let's copy the job folder into a new job folder called "client_api_with_model_select".

To pass just the model/weight difference, we need to modify two places in the configuration:

1. Modify the "transfer_type" from "FULL" to "DIFF" in [config_exchange.json](./jobs/client_api_with_model_select/app/config/config_exchange.json) 
2. Modify the "expected_data_kind" from "WEIGHTS" to "WEIGHT_DIFF" in [config_fed_server.json](./jobs/client_api_with_model_select/app/config/config_fed_server.json))


To add best model selection, we need to do the following:

- Add the aggregated/global model evaluation logic to the custom code [cifar10.py](./jobs/client_api_with_model_select/app/custom/cifar10.py):
    1. Wrap evaluation logic into a method to re-use for evaluation on both trained and received aggregated/global model
    2. Evaluate on received aggregated/global model to get the metrics for model selection
    3. Construct the FLModel to be returned to the NVFlare side with metrics set 

- Add "IntimeModelSelector" to [config_fed_server.json](./jobs/client_api_with_model_select/app/config/config_fed_server.json)

Then we can run this example using:

```bash
nvflare simulator -n 2 -t 2 ./jobs/client_api_with_model_select
```

## The Decorator use case

The above 2 cases show how you can change non-structured ML/DL code to FL.

Usually people have already put their codes into "train", "evaluate", "test" methods so they can reuse.
In that case, the NVFlare ML2FL decorator is the way to go.

[cifar10_tutorial_structured.py](./cifar10_tutorial_structured.py) is an example of structured code. It wraps the training and evaluation logic into 2 methods.

To modify this structured code to be used in FL.
We do the following changes:

1. Import NVFlare Client API: ```import nvflare.client as flare```
2. Initialize NVFlare Client API: ```flare.init()```
3. Modify the train method:
    - Decorate with ```@flare.train```
    - Take additional argument in the beginning
    - Load the received aggregated/global model weights into net: ```net.load_state_dict(input_model.params)```
    - Return an FLModel object
4. Add an fl_evaluate method:
    - Decorate with ```@flare.evaluate```
    - First argument is input FLModel
    - Return a float number of metric
5. Call fl_evaluate method before training to get metrics on received aggregated/global model

Optional: Change the data path to an absolute path and use ```prepare_data.sh``` to download data

We have done the above changes in [jobs/decorator](./jobs/decorator/)
Then we can run it using simulator:

```bash
nvflare simulator -n 2 -t 2 ./jobs/decorator
```

## The lightning use case
If you are using lightning framework to write your training scripts, you can use our NVFlare lightning client API to convert it into FL.

Let's take [cifar10_lightning.py](./cifar10_lightning.py) as our base example.

We make the following changes to it:

1. Import NVFlare Client API: ```import nvflare.client.lightning as flare```
2. Patch the lightning trainer ```flare.patch(trainer)```
3. Evaluate the received aggregated/global model to allow server-side model selection

We have done the above changes in [jobs/lightning](./jobs/lightning/)
Then we can run it using simulator:

```bash
nvflare simulator -n 2 -t 2 ./jobs/lightning
```
