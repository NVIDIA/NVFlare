# Flower App (PyTorch) in NVIDIA FLARE

In this example, we run 2 Flower clients and Flower Server in parallel using NVFlare's simulator.

## Preconditions

To run Flower code in NVFlare, we created a job, including an app with the following custom folder content 
```bash
$ tree jobs/hello-flwr-pt
.
├── client.py           # <-- contains `ClientApp`
├── server.py           # <-- contains `ServerApp`
├── task.py             # <-- task-specific code (model, data)
```
Note, this code is directly copied from Flower's [app-pytorch](https://github.com/adap/flower/tree/main/examples/app-pytorch) example.

## 1. Install dependencies
To run this job with NVFlare, we first need to install the dependencies.
```bash
pip install -r requirements.txt
```

## 2.1 Run a simulation

Next, we run 2 Flower clients and Flower Server in parallel using NVFlare's simulator.
```bash
nvflare simulator jobs/hello-flwr-pt -n 2 -t 2 -w /tmp/nvflare/flwr
```

## 2.2 Run a simulation with TensorBoard streaming

Next, we run 2 Flower clients and Flower Server in parallel using NVFlare while streaming 
the TensorBoard metrics to the server at each iteration using NVFlare's metric streaming.

```bash
CLIENT_API_TYPE="EX_PROCESS_API" nvflare simulator jobs/hello-flwr-pt_tb_streaming -n 2 -t 2 -w /tmp/nvflare/flwr_tb_streaming
```
