# Flower App (PyTorch) in NVIDIA FLARE

In this example, we run 2 Flower clients and Flower Server in parallel using NVFlare's simulator.

## Preconditions

To run Flower code in NVFlare, we created a job, including an app with the following custom folder content 
```bash
$ tree jobs/hello-flwr-pt/app/custom

├── flwr_pt
│   ├── client.py   # <-- contains `ClientApp`
│   ├── __init__.py # <-- to register the python module
│   ├── server.py   # <-- contains `ServerApp`
│   └── task.py     # <-- task-specific code (model, data)
└── pyproject.toml  # <-- Flower project file
```
Note, this code is adapted from Flower's [app-pytorch](https://github.com/adap/flower/tree/main/examples/app-pytorch) example.

## 1. Install dependencies
If you haven't already, we recommend creating a virtual environment.
```bash
python3 -m venv nvflare_flwr
source nvflare_flwr/bin/activate
```
To run this job with NVFlare, we first need to install its dependencies.
```bash
pip install ./jobs/hello-flwr-pt/app/custom
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
