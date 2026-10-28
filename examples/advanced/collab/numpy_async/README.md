# Small AsyncRecipe example

This NumPy example adds a constant local delta using buffered asynchronous
training. It requires no dataset, PyTorch, or KD components.

From this directory:

```bash
python job.py --num-clients 2 --num-updates 4 --buffer-size 2
```

`ArrayClient` returns the standard `(model, metrics, metadata)` tuple. `ArrayApplication`
supplies initial weights. `AsyncRecipe` handles assignments, versions, buffering,
and lifecycle; `NumpyModelAdapter` handles arithmetic. `--num-updates` counts
global publications. Results use the normal Collab run result mechanism.

See [the contracts](../../../../docs/design/async_recipe.md) for independent
imports, application hooks, weighting, transformations, and failure behavior.

Each successful run saves `schedule.json` and `final_weights.npy`. Actual arrival
order depends on execution timing. For optional fixed replay, logical-client
simulation and isolated seed streams, use the separate
[async research example](../pt_async_cifar10_reproducible/README.md).
