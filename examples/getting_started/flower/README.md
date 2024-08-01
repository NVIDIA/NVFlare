# Flower App (PyTorch) in NVIDIA FLARE

In this example, we run 2 Flower clients and Flower Server in parallel using NVFlare's simulator with the JobAPI.

## Preconditions

To run Flower code in NVFlare, simply install the requirements and run the below script. 

Note, this code is directly copied from Flower's [app-pytorch](https://github.com/adap/flower/tree/main/examples/app-pytorch) example.

## Install dependencies
To run this job with NVFlare, we first need to install the dependencies.
```bash
pip install -r requirements.txt
```

## Run a simulation

Next, we run 2 Flower clients and Flower Server in parallel using NVFlare's simulator.
```bash
python fedavg_flwr_cifar10.py
```
