# Track experiments with MLflow

`client.py` uses `MLflowWriter`; `job.py` stores the server-received analytics in this directory's `mlflow.db`.

Complete the course [setup](../../../01.1_running_federated_learning_job/setup.ipynb), then run from this `code` directory:

```bash
python -m pip install mlflow
python job.py
```

This section is self-contained: [job.py](job.py) defines its Recipe, [client.py](client.py) runs at each site, and [model.py](model.py) and [data.py](data.py) define its model and synthetic dataset. The training and evaluation functions are directly in the client script. Edit these files to explore this lesson's variation.

The default runtime workspace is `/tmp/nvflare/simulation/chapter1-mlflow`. Set `NVFLARE_SIMULATOR_WORKSPACE_ROOT` to use a different parent directory, and give the notebook kernel the same value. Exported jobs go under `./exported` when using `--export --export-dir ./exported`.
