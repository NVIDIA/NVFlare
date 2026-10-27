# Convert local training to federated learning

`train.py` runs ordinary PyTorch training. `client.py` adds the training-only Client API loop; `client_evaluate.py` adds evaluation-task support.

Run `train.py` directly for the local baseline. `job.py` launches the selected federated client, which contains its own training and evaluation functions. Compare those functions and the main loop directly with the standalone program.

Complete the course [setup](../../01.1_running_federated_learning_job/setup.ipynb), then run from this `code` directory:

```bash
python train.py
python job.py
python job.py --client-script client_evaluate.py
```

The evaluation-capable variation writes to `/tmp/nvflare/simulation/chapter1-conversion-evaluate`, separately from the training-only run.

This section is self-contained: [job.py](job.py) defines its Recipe, [client.py](client.py) runs at each site, and [model.py](model.py) and [data.py](data.py) define its model and synthetic dataset. The training and evaluation functions are directly in the client script. Edit these files to explore this lesson's variation.

The default runtime workspace is `/tmp/nvflare/simulation/chapter1-conversion`. Set `NVFLARE_SIMULATOR_WORKSPACE_ROOT` to use a different parent directory, and give the notebook kernel the same value. Exported jobs go under `./exported` when using `--export --export-dir ./exported`.
