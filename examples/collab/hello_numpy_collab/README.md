# Hello NumPy Collab

This is a Collab API rewrite of
[`examples/hello-world/hello-numpy`](../../hello-world/hello-numpy/README.md).
It uses the same Recipe API and performs the same small federated averaging
experiment:

- Start with the NumPy weights `[[1, 2, 3], [4, 5, 6], [7, 8, 9]]`.
- Each client simulates local training by adding one to every weight.
- The server averages the client models and repeats for three rounds.
- Each client returns its full model or model difference and mean as an ordinary Python tuple.

## What Collab simplifies

Both examples create a recipe and execute it with `SimEnv`. The original
example separates that Recipe API setup from a Client API training script.
The Collab version keeps the complete application in
[`hello_numpy_collab.py`](hello_numpy_collab.py):

| Original `hello-numpy` | Collab rewrite |
|---|---|
| Execute `NumpyFedAvgRecipe` with `SimEnv` | Execute `CollabRecipe` with the same `SimEnv` Recipe API |
| Initialize the Client API and run a `receive()` / `send()` loop | Decorate the ordinary `train()` function with `@collab.publish` |
| Exchange updates through `FLModel` | Pass NumPy arrays, tuples, and floats directly—no `FLModel`, `Shareable`, or `DXO` |
| Rely on a framework-specific FedAvg implementation | Write the averaging loop as ordinary Python under `@collab.main` |
| Connect separate server and client entry points | Call every client with `collab.clients.train(model)` |

The decorators describe which code runs on the server and clients. The
training and aggregation code remains regular Python, so it can be read,
called, and tested without learning a messaging API.

The complete client interaction is a normal function call:

```python
client_results = collab.clients.train(model, update_type)
model_updates = [model_update for _, (model_update, _) in client_results]
averaged_update = np.mean(model_updates, axis=0)
model = model + averaged_update if update_type == "diff" else averaged_update
```

## Run it

Install NVFlare from this repository, then run the module from the `examples`
directory:

```bash
python -m collab.hello_numpy_collab.hello_numpy_collab
```

The Collab rewrite accepts the same core experiment options as the original.
For example, send model differences or export the job configuration with:

```bash
python -m collab.hello_numpy_collab.hello_numpy_collab --update_type diff
python -m collab.hello_numpy_collab.hello_numpy_collab --export_config
```

Other options are `--n_clients`, `--num_rounds`, and `--log_config`. After
three rounds the final model is:

```text
[[ 4.  5.  6.]
 [ 7.  8.  9.]
 [10. 11. 12.]]
```
