# Track experiments with MLflow

`client.py` uses `MLflowWriter`; `job.py` stores the server-received analytics in this directory's `mlflow.db`. The default run uses **20 federated rounds**, producing 20 points per metric per site (steps 0–19).

Complete the course [setup](../../../01.1_running_federated_learning_job/setup.ipynb), then run from this `code` directory:

```bash
python -m pip install mlflow
python job.py
```

To browse the results, open the [notebook](../experiment_tracking_mlflow.ipynb), run its **Open MLflow UI** cell, and click the browser link. The cell launches MLflow with the same database used by the job. Select the latest site runs and view the histories of `global_accuracy_before_train`, `local_accuracy_after_train`, and `local_train_loss` against the round step. Loss can continue decreasing after accuracy reaches 100%. Stop the UI with the notebook toolbar's **Stop/Interrupt** button when finished.

This section is self-contained: [job.py](job.py) defines its Recipe, [client.py](client.py) runs at each site, and [model.py](model.py) and [prepare_data.py](prepare_data.py) define its model and synthetic dataset. The training and evaluation functions are directly in the client script. Edit these files to explore this lesson's variation.

The default runtime workspace is `/tmp/nvflare/simulation/chapter1-mlflow`. Set `NVFLARE_SIMULATOR_WORKSPACE_ROOT` to use a different parent directory, and give the notebook kernel the same value. Exported jobs go under `./exported` when using `--export --export-dir ./exported`.
