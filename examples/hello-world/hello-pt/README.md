# Hello PyTorch

Example of using [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) to train an image classifier using federated averaging ([FedAvg]([FedAvg](https://arxiv.org/abs/1602.05629))) and [PyTorch](https://pytorch.org/) as the deep learning training framework.

> **_NOTE:_** This example uses the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset and will load its data within the trainer code.

### 1. Install NVIDIA FLARE

Follow the [Installation](https://nvflare.readthedocs.io/en/main/quickstart.html) instructions.
Install additional requirements:

```
pip3 install torch torchvision
```

### 2. Run the experiment

Use nvflare simulator to run the hello-examples:

```
nvflare simulator -w /tmp/nvflare/ -n 2 -t 2 hello-pt
```

### 3. Access the logs and results

You can find the running logs and results inside the simulator's workspace/simulate_job

```bash
$ ls /tmp/nvflare/simulate_job/
app_server  app_site-1  app_site-2  log.txt

```

### 4. Global Model Initialization Approaches

There are multiple approaches to initialize a global model, either on FL server side or on the client side. 

When the model is initialized at the FL **server-side**, the initialized global model will then be broadcast to all the FL clients.
Clients will take this initial model and start training. The benefits of the server-side model initialization 
allows the model to only initialize once in one place (server) and then distributed to all clients, 
so all clients have the same initial model. The potential issue with server-side model initialization might
involve security risk of running custom-python code on server. leveraging a predefined model file as initialization model can be another server-side approach. 

In this example, we are using **client-side** model initialization approach. 

The client-side model initialization avoids server-side custom code as well as without extra setup in the model-file based approach.
client side initialization asks every client to send the initialized model as a pre-task in the workflow, before the training starts.
On the server side, once the server receive the initial models from clients, server can choose different strategies to leverage the models
from different clients:
* Select one model randomly among all clients’ models, then use it as the global initial model
* Apply aggregation function to generate the global initial model

In this example,we use _InitializeGlobalWeights_ controller, which have implemented the following strategies ( weight_method)
* Weight_method = "first" , then use the weights reported from the first client;
* weight_method = "client", then only use the weights reported from the specified client.

If one’s use case demands a different strategy, then you can implement a new controller.


Looking at the job workflow, we have defined three workflows in config_fed_server.json
  * pre_train ( get_weights )  with _InitializeGlobalWeights_ controller
  * scatter_and_gather (train) with _ScatterAndGather_ controller
  * cross_site_validate (cross validation) with _CrossSiteModelEval_

```
 "workflows": [
      {
        "id": "pre_train",
        "name": "InitializeGlobalWeights",
        "args": {
          "task_name": "get_weights"
        }
      },
      {
        "id": "scatter_and_gather",
        "name": "ScatterAndGather",
        "args": {
            ... skip arguments ...
        }
      },
      {
        "id": "cross_site_validate",
        "name": "CrossSiteModelEval",
        "args": {
          "model_locator_id": "model_locator"
        }
      }
  ]
```
 



