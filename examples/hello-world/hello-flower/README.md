# Flower App (PyTorch) in NVIDIA FLARE

In this example, we run 2 clients and 1 server using NVFlare's simulator.

## Preconditions

Following https://github.com/adap/flower/tree/main/examples/quickstart-pytorch we prepare the following flower app: 

```bash
$ tree flwr-pt

├── flwr_pt
│   ├── client.py   # <-- contains `ClientApp`
│   ├── __init__.py # <-- to register the python module
│   ├── server.py   # <-- contains `ServerApp`
│   └── task.py     # <-- task-specific code (model, data)
└── pyproject.toml  # <-- Flower project file
```

To be run inside NVFlare, we need to add the following sections to "pyproject.toml":
```
[tool.flwr.app.config]
num-server-rounds = 3

[tool.flwr.federations]
default = "local-simulation"

[tool.flwr.federations.local-simulation]
options.num-supernodes = 2
address = "127.0.0.1:9093"
insecure = true
```

You can adjust the num-server-rounds.
The number `options.num-supernodes` should match the number of NVFlare clients defined in [job.py](./job.py), e.g., `job.simulator_run(args.workdir, gpu="0", n_clients=2)`.

## 1. Install dependencies
If you haven't already, we recommend creating a virtual environment.
```bash
python3 -m venv nvflare_flwr
source nvflare_flwr/bin/activate
pip install -r ./requirements.txt
```

## 2.1 Run flwr-pt with NVFlare simulation

We run 2 Flower clients and Flower Server in parallel using NVFlare's simulator.
```bash
python job.py --job_name "flwr-pt" --content_dir "./flwr-pt"
```

## 2.2 Run flwr-pt with NVFlare simulation and NVFlare's TensorBoard streaming

We run 2 Flower clients and Flower Server in parallel using NVFlare while streaming 
the TensorBoard metrics to the server at each iteration using NVFlare's metric streaming.

```bash
python job.py --job_name "flwr-pt-tb" --content_dir "./flwr-pt-tb" --stream_metrics
```

You can visualize the metrics streamed to the server using TensorBoard.
```bash
tensorboard --logdir /tmp/nvflare/hello-flower
```
![tensorboard training curve](./train.png)


## 3. Run with real deployment

First, check real-world deployment guide: https://nvflare.readthedocs.io/en/2.6/real_world_fl/overview.html

Second, export the corresponding NVFlare job:
```bash
python job.py --job_name "flwr-pt" --content_dir "./flwr-pt" --export_job --export_dir "./jobs"
```

An NVFlare job will be generated at "./jobs" folder.

Then you can copy it inside the admin console's transfer folder and then run:
```bash
submit_job flwr-pt
```
