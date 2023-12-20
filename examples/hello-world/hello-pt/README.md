# Hello PyTorch

Example of using [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) to train an image classifier
using federated averaging ([FedAvg](https://arxiv.org/abs/1602.05629))
and [PyTorch](https://pytorch.org/) as the deep learning training framework.

> **_NOTE:_** This example uses the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset and will load its data within the trainer code.

You can follow the [hello_world notebook](../hello_world.ipynb) or the following:


### 1. Install NVIDIA FLARE

Follow the [Installation](https://nvflare.readthedocs.io/en/main/quickstart.html) instructions to install NVFlare.

Install additional requirements:

```
pip3 install -r requirements.txt
```

### 2. Run the experiment

Prepare the data first:

```
bash ./prepare_data.sh
```

Use nvflare simulator to run the hello-examples:

```
nvflare simulator -w /tmp/nvflare/ -n 2 -t 2 hello-pt/jobs/hello-pt
```

### 3. Access the logs and results

You can find the running logs and results inside the simulator's workspace/simulate_job

```bash
$ ls /tmp/nvflare/simulate_job/
app_server  app_site-1  app_site-2  log.txt

```

### 4. Global Model Initialization Approaches

There are various methods for initializing a global model in federated learning, which can be done either on the FL server or on the client side. The choice of model initialization approach depends on the specific use case of the user.

When the global model is initialized on the FL server-side, it is then broadcasted to all the FL clients. 
The clients can use this initial model as a starting point for training. 
The advantage of server-side model initialization is that the model only needs to be initialized once in one location 
(the server), and then distributed to all clients, ensuring that all clients have the same initial model. 
However, it is important to note that server-side model initialization may present potential security risks 
if custom Python code is run on the server. An alternative approach for server-side initialization is to use 
a predefined model file as the initialization model. The ScatterAndGather controller is using persistor to reads / init
model from server-side. 

In this example, we are using **client-side** model initialization approach. 

The client-side model initialization avoids server-side custom code as well as without extra setup in the model-file based approach.
client side initialization asks every client to send the initialized model as a pre-task in the workflow, before the training starts.
On the server side, once the server receive the initial models from clients, server can choose different strategies to leverage the models
from different clients:
* Select one model randomly among all clients' models, then use it as the global initial model
* Apply aggregation function to generate the global initial model

In this example,we use _InitializeGlobalWeights_ controller, which have implemented the following strategies ( weight_method)
* Weight_method = "first" , then use the weights reported from the first client;
* weight_method = "client", then only use the weights reported from the specified client.

If one's use case demands a different strategy, then you can implement a new model initialization controller.

Looking at the job workflow, we have defined three workflows in config_fed_server.json
  * pre_train ( model initialization )  with _InitializeGlobalWeights_ controller
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

Once the global model is initialized, it is set to the fl_ctx as sticky property and then pass to the 
next controller (ScatterAndGather) in the training step. The sticky property allows properties pass cross controllers. 
The ScatterAndGather still leverage _persistor_ to load the initial global model, but since there is no model file 
or server-side initialized model, the ScatterAndGather then try to load the model from fl_ctx's _"global_model"_ property, 
which is initialized from the client-side and set by the previous controller in the workflow. 
 



