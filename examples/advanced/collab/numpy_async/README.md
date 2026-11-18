# Small FedBuffRecipe example

This NumPy example adds a constant local delta using buffered asynchronous
training. It requires no dataset, PyTorch, or KD components.

From this directory:

```bash
python job.py --num-clients 2 --num-rounds 4 --buffer-size 2
```

`ArrayClient` returns the standard `(model, metrics, metadata)` tuple. `ArrayModelLoader`
supplies initial weights and `ArrayObserver` records results. `FedBuffRecipe` handles
assignments, versions, buffering, and lifecycle; `NumpyModelOps` handles arithmetic.
`--num-rounds` counts global publications; clients may remain in flight across
rounds. Results use the normal Collab run result mechanism.

See [the FedBuff recipe guide](../../../../docs/user_guide/data_scientist_guide/fedbuff_recipe.rst) for independent
imports, observer hooks, weighting, transformations, and failure behavior.

Each successful run saves `schedule.json` and `final_weights.npy`. Actual arrival
order depends on execution timing.
