# Customize the server

Implement server aggregation in [aggregator.py](aggregator.py), then pass `MedianAggregator()` to `FedAvgRecipe` in [job.py](job.py). The custom component collects client models, computes the median at each parameter coordinate, and clears its state between rounds. It also averages the clients' received-model accuracy for stopping and model selection.

Complete the course [setup](../../01.1_running_federated_learning_job/setup.ipynb), then run from this `code` directory:

```bash
python job.py
```

This lesson uses **three sites**: for one parameter coordinate with client values `1, 2, 100`, the median is `2`, while the equal-weight mean is about `34.3`. With only two sites, NumPy's median would equal their mean. The custom rule ignores optimizer-step weights; each site contributes once. This small-model teaching example retains all client models for a round in memory.

Compare with the built-in step-weighted averaging, using the same clients:

```bash
python job.py --aggregation fedavg
```

Both runs keep the same round limit, stopping policy, and model persistence. `--num_rounds`, `--stop_accuracy`, and `--patience` change those settings. To see early stopping immediately and its reason in the console, use `python job.py --stop_accuracy 0 --log_config full`. Progress mode shows round progress and the final summary; detailed stopping messages are in `server/log.txt` under the workspace.

This section is self-contained: [job.py](job.py) defines its recipe and selects the server aggregator, [client.py](client.py) runs at each site, and [model.py](model.py) and [prepare_data.py](prepare_data.py) define its model and synthetic dataset. The training and evaluation functions are directly in the client script. Client training code does not change when you switch aggregators.

The default runtime workspace is `/tmp/nvflare/simulation/chapter1-custom-server`; the built-in averaging comparison uses `/tmp/nvflare/simulation/chapter1-custom-server-fedavg`. Set `NVFLARE_SIMULATOR_WORKSPACE_ROOT` to a fresh parent directory to preserve repeated runs, and give the notebook kernel the same value. Exported jobs go under `./exported` when using `--export --export-dir ./exported`.
