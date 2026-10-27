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

To use a predefined arrival sequence, select assignment IDs in acceptance order:

```bash
python job.py --num-clients 2 --num-updates 4 --buffer-size 1 --arrival-sequence 1 0 3 2
```

The recipe holds early results until their turn, preserving subsequent dispatch
and model-base decisions. IDs must cover all contributions and respect K/B/O;
an incompatible sequence fails explicitly. The default remains actual-arrival
execution. Each successful run saves `schedule.json` alongside `final_weights.npy`;
filter its `accept` events to inspect or record the arrival sequence.
The [extension contracts](../../../../docs/design/async_recipe.md#explicit-extension-points-on-asyncrecipe)
cover logical jobs over fewer physical workers, assignment seeds, recording,
custom simulated-time schedulers, and optional storage offload.
