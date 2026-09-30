# Hello FedAvg with the Collab API

This example demonstrates federated averaging as a custom workflow written
with NVIDIA FLARE's Collab API. The complete application is self-contained in
[`job.py`](job.py): a server coordinates training rounds, clients train a small
synthetic PyTorch model, and the server averages their model weights.

## What you'll learn

- Defining the server workflow with `@collab.main`
- Publishing client training with `@collab.publish`
- Calling every client with `collab.clients.train(...)`
- Passing PyTorch state dictionaries and scalar losses directly
- Setting client-specific values with `set_per_site_config(...)`

The example configures two clients by default. `site-1` trains for two local
epochs per round and the other sites train for five, illustrating how each
client receives only its own configuration.

## Run it

Install the example dependencies, then run the job in simulation:

```bash
cd examples/hello-world/hello-collab
python -m pip install -r requirements.txt
python job.py
```

You can change the number of clients or federated rounds:

```bash
python job.py --num-clients 3 --num-rounds 5
```

The same `CollabRecipe` can also run with `PocEnv` or `ProdEnv` from
`nvflare.recipe`; the application does not need a Collab-specific runner.

For response callbacks, split learning, and decentralized workflows, continue
with the [advanced Collab examples](../../advanced/collab/README.md).
