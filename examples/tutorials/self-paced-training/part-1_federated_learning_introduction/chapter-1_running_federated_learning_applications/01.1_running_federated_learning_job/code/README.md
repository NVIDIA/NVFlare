# Run your first federated job

A two-client, three-round FedAvg recipe with evaluation of the saved global model.

`job.py` starts `client.py` at each site. The PyTorch training loop and evaluation functions are directly in `client.py`, alongside the Client API loop. This first-run example does not need a standalone `train.py` program. That comparison is introduced in the [conversion lesson](../../01.2_convert_deep_learning_to_federated_learning/convert_dl_to_fl.ipynb).

Complete the course [setup](../../01.1_running_federated_learning_job/setup.ipynb), then run from this `code` directory:

```bash
python job.py
```

This section is self-contained: [job.py](job.py) defines its Recipe, [client.py](client.py) runs at each site, and [model.py](model.py) and [prepare_data.py](prepare_data.py) define its model and synthetic dataset. The training and evaluation functions are directly in the client script. Edit these files to explore this lesson's variation.

The default runtime workspace is `/tmp/nvflare/simulation/chapter1-basic`. Set `NVFLARE_SIMULATOR_WORKSPACE_ROOT` to use a different parent directory, and give the notebook kernel the same value. Exported jobs go under `./exported` when using `--export --export-dir ./exported`.
