# Small AsyncRecipe example

This NumPy example adds a constant local delta using buffered asynchronous
training. It requires no dataset, PyTorch, or KD components.

From this directory:

```bash
python job.py --num-clients 2 --num-updates 4 --buffer-size 2
```

`ArrayClient` returns the standard `(model, metrics, metadata)` tuple. `ArrayApplication`
supplies initial weights. `AsyncRecipe` handles assignments, versions, buffering,
and lifecycle; `NumpyAsyncModelAdapter` handles arithmetic. `--num-updates` counts
global publications. Results use the normal Collab run result mechanism.

See [the contracts](../../../../docs/design/async_recipe.md) for independent
imports, application hooks, weighting, transformations, and failure behavior.
