
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
    recipe = FedAvgRecipe(
        name="hello-pt",
        min_clients=n_clients,
        num_rounds=num_rounds,
        initial_model=SimpleNetwork(),
        train_script="client.py",
        train_args=f"--batch_size {batch_size}",
    )

    env = SimEnv(num_clients=n_clients, num_threads=n_clients)
    recipe.execute(env=env)
```
 
## Run Job
from terminal try to run the code


```
    python job.py
```
# output

```
2025-08-09 18:36:50,912 - INFO - model selection weights control: {}
2025-08-09 18:36:51,756 - INFO - Tensorboard records can be found in /tmp/nvflare/simulation/hello-pt/server/simulate_job/tb_events you can view it using `tensorboard --logdir=/tmp/nvflare/simulation/hello-pt/server/simulate_job/tb_events`
2025-08-09 18:36:51,757 - INFO - Initializing BaseModelController workflow.
2025-08-09 18:36:51,757 - INFO - Beginning model controller run.
2025-08-09 18:36:51,758 - INFO - 
================================================================================
                                 Start FedAvg.                                  
================================================================================

2025-08-09 18:36:51,758 - INFO - loading initial model from persistor
2025-08-09 18:36:51,758 - INFO - Both source_ckpt_file_full_name and ckpt_preload_path are not provided. Using the default model weights initialized on the persistor side.
2025-08-09 18:36:51,759 - INFO - 
--------------------------------------------------------------------------------
                                Round 0 started.                                
--------------------------------------------------------------------------------

2025-08-09 18:36:51,759 - INFO - Sampled clients: ['site-1', 'site-2']
2025-08-09 18:36:51,760 - INFO - Sending task train to ['site-1', 'site-2']
2025-08-09 18:36:55,498 - INFO - start task run() with full path: /tmp/nvflare/simulation/hello-pt/site-1/simulate_job/app_site-1/custom/client.py
2025-08-09 18:36:55,502 - INFO - NP: dumping 10 arrays to file /tmp/nvflare/datums/NumpyArrayDecomposer_a91c1f47-5bd9-4699-b570-f0551e406569.npz
2025-08-09 18:36:55,505 - INFO - start task run() with full path: /tmp/nvflare/simulation/hello-pt/site-2/simulate_job/app_site-2/custom/client.py
2025-08-09 18:36:55,509 - INFO - NP: dumping 10 arrays to file /tmp/nvflare/datums/NumpyArrayDecomposer_8ddbad42-3f27-4491-afcf-f038bb3adf77.npz
2025-08-09 18:36:55,584 - INFO - execute for task (train)
2025-08-09 18:36:55,585 - INFO - send data to peer
2025-08-09 18:36:55,585 - INFO - sending payload to peer
2025-08-09 18:36:55,586 - INFO - Waiting for result from peer
2025-08-09 18:36:55,657 - INFO - execute for task (train)
2025-08-09 18:36:55,658 - INFO - send data to peer
2025-08-09 18:36:55,658 - INFO - sending payload to peer
2025-08-09 18:36:55,659 - INFO - Waiting for result from peer
2025-08-09 18:36:56,646 - INFO - Files already downloaded and verified
2025-08-09 18:36:56,660 - INFO - Files already downloaded and verified
2025-08-09 18:36:57,097 - INFO - site = site-1, current_round=0
2025-08-09 18:36:57,098 - INFO - site = site-2, current_round=0
2025-08-09 18:36:57,505 - INFO - site=site-1, Epoch: 0/2, Iteration: 0, Loss: 4.821970065434774e-05
2025-08-09 18:36:57,531 - INFO - site=site-2, Epoch: 0/2, Iteration: 0, Loss: 4.755312701066335e-05
2025-08-09 18:37:06,999 - INFO - site=site-1, Epoch: 0/2, Iteration: 3000, Loss: 0.10494570057094098
2025-08-09 18:37:07,055 - INFO - site=site-2, Epoch: 0/2, Iteration: 3000, Loss: 0.10441589770093561
2025-08-09 18:37:07,397 - INFO - site=site-1, Epoch: 1/2, Iteration: 0, Loss: 2.5529302656650544e-05
2025-08-09 18:37:07,453 - INFO - site=site-2, Epoch: 1/2, Iteration: 0, Loss: 2.9990196228027343e-05
2025-08-09 18:37:16,856 - INFO - site=site-1, Epoch: 1/2, Iteration: 3000, Loss: 0.08704691271235546
2025-08-09 18:37:16,909 - INFO - site=site-2, Epoch: 1/2, Iteration: 3000, Loss: 0.08581113628298044
2025-08-09 18:37:17,251 - INFO - Finished Training for site-1
2025-08-09 18:37:17,252 - INFO - site: site-1, sending model to server.
2025-08-09 18:37:17,300 - INFO - Finished Training for site-2
2025-08-09 18:37:17,302 - INFO - site: site-2, sending model to server.
2025-08-09 18:37:17,623 - INFO - NP: dumping 10 arrays to file /tmp/nvflare/datums/NumpyArrayDecomposer_c0af9186-1ae3-42d6-8819-7d23676cea94.npz
2025-08-09 18:37:17,714 - INFO - NP: dumping 10 arrays to file /tmp/nvflare/datums/NumpyArrayDecomposer_af01d59a-bc27-4b53-a5e3-c24e5cf160df.npz
2025-08-09 18:37:18,051 - INFO - aggregating 2 update(s) at round 0
2025-08-09 18:37:18,053 - INFO - Start persist model on server.
2025-08-09 18:37:18,055 - INFO - End persist model on server.
2025-08-09 18:37:18,056 - INFO - 
--------------------------------------------------------------------------------
                                Round 1 started.                                
--------------------------------------------------------------------------------

2025-08-09 18:37:18,056 - INFO - Sampled clients: ['site-1', 'site-2']
2025-08-09 18:37:18,057 - INFO - Sending task train to ['site-1', 'site-2']
2025-08-09 18:37:19,862 - INFO - NP: dumping 10 arrays to file /tmp/nvflare/datums/NumpyArrayDecomposer_8f4ff13f-9ac6-40f1-b053-408f835d8e41.npz
2025-08-09 18:37:19,983 - INFO - execute for task (train)
2025-08-09 18:37:19,984 - INFO - send data to peer
2025-08-09 18:37:19,984 - INFO - sending payload to peer
2025-08-09 18:37:19,984 - INFO - Waiting for result from peer
2025-08-09 18:37:20,019 - INFO - NP: dumping 10 arrays to file /tmp/nvflare/datums/NumpyArrayDecomposer_00fe5d07-7802-4fff-b44f-74048970464c.npz
2025-08-09 18:37:20,137 - INFO - execute for task (train)
2025-08-09 18:37:20,138 - INFO - send data to peer
2025-08-09 18:37:20,138 - INFO - sending payload to peer
2025-08-09 18:37:20,138 - INFO - Waiting for result from peer
2025-08-09 18:37:20,256 - INFO - site = site-1, current_round=1
2025-08-09 18:37:20,266 - INFO - site=site-1, Epoch: 0/2, Iteration: 0, Loss: 2.7689563731352487e-05
2025-08-09 18:37:20,306 - INFO - site = site-2, current_round=1
2025-08-09 18:37:20,318 - INFO - site=site-2, Epoch: 0/2, Iteration: 0, Loss: 2.8927244246006012e-05
2025-08-09 18:37:36,577 - INFO - site=site-1, Epoch: 0/2, Iteration: 3000, Loss: 0.08340640442073345
2025-08-09 18:37:36,971 - INFO - site=site-1, Epoch: 1/2, Iteration: 0, Loss: 2.2170931100845337e-05
2025-08-09 18:37:38,679 - INFO - site=site-2, Epoch: 0/2, Iteration: 3000, Loss: 0.08276068011609217
2025-08-09 18:37:39,308 - INFO - site=site-2, Epoch: 1/2, Iteration: 0, Loss: 3.102670609951019e-05
2025-08-09 18:37:46,149 - INFO - site=site-1, Epoch: 1/2, Iteration: 3000, Loss: 0.07720229349657894
2025-08-09 18:37:46,531 - INFO - Finished Training for site-1
2025-08-09 18:37:46,533 - INFO - site: site-1, sending model to server.
2025-08-09 18:37:47,020 - INFO - NP: dumping 10 arrays to file /tmp/nvflare/datums/NumpyArrayDecomposer_1d4143bf-96e6-430c-87dc-65be9cd88087.npz
2025-08-09 18:37:47,166 - WARNING - validation metric not existing in site-1
2025-08-09 18:37:54,165 - INFO - site=site-2, Epoch: 1/2, Iteration: 3000, Loss: 0.07727636773263415
2025-08-09 18:37:54,778 - INFO - Finished Training for site-2
2025-08-09 18:37:54,781 - INFO - site: site-2, sending model to server.
2025-08-09 18:37:55,183 - INFO - NP: dumping 10 arrays to file /tmp/nvflare/datums/NumpyArrayDecomposer_b2e0ac0a-e2e9-474d-8633-37d28be306db.npz
2025-08-09 18:37:55,344 - WARNING - validation metric not existing in site-2
2025-08-09 18:37:55,519 - INFO - aggregating 2 update(s) at round 1
2025-08-09 18:37:55,521 - INFO - Start persist model on server.
2025-08-09 18:37:55,523 - INFO - End persist model on server.
2025-08-09 18:37:55,523 - INFO - 
================================================================================
                                Finished FedAvg.                                
================================================================================

```