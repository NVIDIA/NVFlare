
# Hello Pytorch
This example demonstrates how to use NVIDIA FLARE with PyTorch to train an image classifier using federated averaging (FedAvg).The complete example code can be found in the`hello-pt directory <examples/hello-world/hello-pt/>`. It is recommended to create a virtual environment and run everything within a virtualenv.

## NVIDIA FLARE Installation
for the complete installation instructions, see <../../installation.html>
```
pip install nvflare

```
Install the dependency

```
pip install -r requirements.txt
```
## Code Structure
first get the example code from github:

```
git clone https://github.com/NVIDIA/NVFlare.git
```
then navigate to the hello-pt directory:

```
git switch <release branch>
cd examples/hello-world/hello-pt
```
``` bash
hello-pt
|
|-- client.py         # client local training script
|-- model.py          # model definition
|-- job.py            # job recipe that defines client and server configurations
|-- requirements.txt  # dependencies
|-- doc.py            # documentation file to generate the current documentation
```
doc.py is this file, it is used for documentation generation, it is not part of the fl code.

## Data
This example uses the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset

In a real FL experiment, each client would have their own dataset used for their local training. 
You can download the CIFAR10 dataset from the Internet via torchvision’s datasets module, 
You can split the datasets for different clients, so that each client has its own dataset. 
Here for simplicity’s sake, the same dataset we will be using on each client.

## Model
In PyTorch, neural networks are implemented by defining a class (e.g., SimpleNetwork) that extends nn.Module. 
The network’s architecture is set up in the __init__ method,# while the forward method determines how input data flows
through the layers. For faster computations, the model is transferred to a hardware accelerator (such as CUDA GPUs) if available; otherwise, it runs on the CPU. The implementation of this model can be found in model.py.
 
## Client Code
The client code ```client.py``` is responsible for Notice the training code is almost identical to the pytorch standard training code. 
The only difference is that we added a few lines to receive and send data to the server.

## Server Code
In federated averaging, the server code is responsible for aggregating model updates from clients, the workflow pattern is similar to scatter-gather. In this example, we will directly use the default federated averaging algorithm provided by NVFlare. 
The FedAvg class is defined in nvflare.app_common.workflows.fedavg.FedAvg
There is no need to defined a customized server code for this example.

## Job Recipe Code
Job Recipe contains the client.py and built-in Fed average algorithm.
```
from nvflare.app_opt.pt.job_config.Job_recipe import FedAvgRecipe

if __name__ == "__main__":
n_clients = 2
num_rounds = 2
train_script = "src/client.py"
client_script_args = ""

    recipe = FedAvgRecipe(clients=n_clients,
                          num_rounds=num_rounds,
                          model= SimpleNetwork(),
                          client_script=train_script,
                          client_script_args= client_script_args)
    recipe.execute()
```
 
## Run Job
from terminal try to run the code


```
    python job.py
```
# output

```

    # 2025-07-20 21:41:23,489 - INFO - model selection weights control: {}
    # 2025-07-20 21:41:24,357 - INFO - Tensorboard records can be found in /tmp/nvflare/sim/workspace/server/simulate_job/tb_events you can view it using `tensorboard --logdir=/tmp/nvflare/sim/workspace/server/simulate_job/tb_events`
    # 2025-07-20 21:41:24,358 - INFO - Initializing BaseModelController workflow.
    # 2025-07-20 21:41:24,359 - INFO - Beginning model controller run.
    # 2025-07-20 21:41:24,359 - INFO - Start FedAvg.
    # 2025-07-20 21:41:24,360 - INFO - loading initial model from persistor
    # 2025-07-20 21:41:24,360 - INFO - Both source_ckpt_file_full_name and ckpt_preload_path are not provided. Using the default model weights initialized on the persistor side.
    # 2025-07-20 21:41:24,361 - INFO - Round 0 started.
    # 2025-07-20 21:41:24,361 - INFO - Sampled clients: ['site-1', 'site-2']
    # 2025-07-20 21:41:24,361 - INFO - Sending task train to ['site-1', 'site-2']
    # 2025-07-20 21:41:28,049 - INFO - start task run() with full path: /tmp/nvflare/sim/workspace/site-2/simulate_job/app_site-2/custom/client.py
    # 2025-07-20 21:41:28,052 - INFO - start task run() with full path: /tmp/nvflare/sim/workspace/site-1/simulate_job/app_site-1/custom/client.py
    # 2025-07-20 21:41:28,060 - INFO - execute for task (train)
    # 2025-07-20 21:41:28,060 - INFO - send data to peer
    # 2025-07-20 21:41:28,061 - INFO - sending payload to peer
    # 2025-07-20 21:41:28,061 - INFO - Waiting for result from peer
    # 2025-07-20 21:41:28,063 - INFO - execute for task (train)
    # 2025-07-20 21:41:28,063 - INFO - send data to peer
    # 2025-07-20 21:41:28,064 - INFO - sending payload to peer
    # 2025-07-20 21:41:28,064 - INFO - Waiting for result from peer
    # 2025-07-20 21:41:29,128 - INFO - Files already downloaded and verified
    # 2025-07-20 21:41:29,168 - INFO - Files already downloaded and verified
    # 2025-07-20 21:41:29,566 - INFO - site = site-1, current_round=0
    # 2025-07-20 21:41:29,599 - INFO - site = site-2, current_round=0
    # 2025-07-20 21:41:29,987 - INFO - site=site-1, Epoch: 0/2, Iteration: 0, Loss: 4.805266360441844e-05
    # 2025-07-20 21:41:30,002 - INFO - site=site-2, Epoch: 0/2, Iteration: 0, Loss: 4.810774823029836e-05
    # 2025-07-20 21:41:39,339 - INFO - site=site-1, Epoch: 0/2, Iteration: 3000, Loss: 0.10355195295189817
    # 2025-07-20 21:41:39,348 - INFO - site=site-2, Epoch: 0/2, Iteration: 3000, Loss: 0.1036934811597069
    # 2025-07-20 21:41:39,737 - INFO - site=site-1, Epoch: 1/2, Iteration: 0, Loss: 2.9984918733437858e-05
    # 2025-07-20 21:41:39,746 - INFO - site=site-2, Epoch: 1/2, Iteration: 0, Loss: 2.9017041126887005e-05
    # 2025-07-20 21:41:49,111 - INFO - site=site-1, Epoch: 1/2, Iteration: 3000, Loss: 0.08605644813800852
    # 2025-07-20 21:41:49,120 - INFO - site=site-2, Epoch: 1/2, Iteration: 3000, Loss: 0.08581393839791417
    # 2025-07-20 21:41:49,497 - INFO - Finished Training for site-1
    # 2025-07-20 21:41:49,506 - INFO - Finished Training for site-2
    # 2025-07-20 21:41:49,507 - INFO - site: site-1, sending model to server.
    # 2025-07-20 21:41:49,508 - INFO - site: site-2, sending model to server.
    # 2025-07-20 21:41:49,990 - INFO - aggregating 2 update(s) at round 0
    # 2025-07-20 21:41:49,992 - INFO - Start persist model on server.
    # 2025-07-20 21:41:49,995 - INFO - End persist model on server.
    # 2025-07-20 21:41:49,995 - INFO - Round 1 started.
    # 2025-07-20 21:41:49,995 - INFO - Sampled clients: ['site-1', 'site-2']
    # 2025-07-20 21:41:49,996 - INFO - Sending task train to ['site-1', 'site-2']
    # 2025-07-20 21:41:51,695 - INFO - execute for task (train)
    # 2025-07-20 21:41:51,696 - INFO - send data to peer
    # 2025-07-20 21:41:51,696 - INFO - sending payload to peer
    # 2025-07-20 21:41:51,697 - INFO - Waiting for result from peer
    # 2025-07-20 21:41:51,754 - INFO - execute for task (train)
    # 2025-07-20 21:41:51,754 - INFO - send data to peer
    # 2025-07-20 21:41:51,755 - INFO - sending payload to peer
    # 2025-07-20 21:41:51,756 - INFO - Waiting for result from peer
    # 2025-07-20 21:41:52,010 - INFO - site = site-1, current_round=1
    # 2025-07-20 21:41:52,011 - INFO - site = site-2, current_round=1
    # 2025-07-20 21:41:52,023 - INFO - site=site-1, Epoch: 0/2, Iteration: 0, Loss: 3.0566091338793434e-05
    # 2025-07-20 21:41:52,023 - INFO - site=site-2, Epoch: 0/2, Iteration: 0, Loss: 2.9468193650245666e-05
    # 2025-07-20 21:42:05,812 - INFO - site=site-2, Epoch: 0/2, Iteration: 3000, Loss: 0.08019066233622531
    # 2025-07-20 21:42:06,095 - INFO - site=site-1, Epoch: 0/2, Iteration: 3000, Loss: 0.08063799119119842
    # 2025-07-20 21:42:06,218 - INFO - site=site-2, Epoch: 1/2, Iteration: 0, Loss: 2.4138346314430236e-05
    # 2025-07-20 21:42:06,494 - INFO - site=site-1, Epoch: 1/2, Iteration: 0, Loss: 2.6129357516765596e-05
    # 2025-07-20 21:42:15,555 - INFO - site=site-2, Epoch: 1/2, Iteration: 3000, Loss: 0.07604475673598547
    # 2025-07-20 21:42:15,841 - INFO - site=site-1, Epoch: 1/2, Iteration: 3000, Loss: 0.07583552108642956
    # 2025-07-20 21:42:15,944 - INFO - Finished Training for site-2
    # 2025-07-20 21:42:15,945 - INFO - site: site-2, sending model to server.
    # 2025-07-20 21:42:16,217 - INFO - Finished Training for site-1
    # 2025-07-20 21:42:16,220 - INFO - site: site-1, sending model to server.
    # 2025-07-20 21:42:16,236 - WARNING - validation metric not existing in site-1
    # 2025-07-20 21:42:16,300 - WARNING - validation metric not existing in site-2
    # 2025-07-20 21:42:16,427 - INFO - aggregating 2 update(s) at round 1
    # 2025-07-20 21:42:16,428 - INFO - Start persist model on server.
    # 2025-07-20 21:42:16,431 - INFO - End persist model on server.
    # 2025-07-20 21:42:16,431 - INFO - Finished FedAvg.
```