# Collab API Examples

The catalog is organized into core examples and advanced/integration
examples. `hello_collab` directly uses the local in-process runner.
`hello_numpy_collab` mirrors the Recipe API and `SimEnv` flow of
`hello-world/hello-numpy`. The remaining core examples are built around a
`make_recipe()` function; their execution environment is an option
(`--runtime in_process | multi_process | prod | export`), never a separate
example — the recipe is identical across all of them:

```bash
cd examples   # the examples import each other via the collab.* package path

python -m collab.hello_collab.hello_collab                           # minimal local-only NumPy example
python -m collab.hello_collab.hello_collab_functions                 # plain functions; module auto-wrapped
python -m collab.hello_numpy_collab.hello_numpy_collab                # same Recipe API as hello-numpy, less client plumbing
python -m collab.hello_fedavg.hello_fedavg                            # threads in this process
python -m collab.split_learning.split_learning                          # CIFAR-10 split learning
python -m collab.hello_fedavg.hello_fedavg --runtime multi_process   # real FLARE processes (local POC)
python -m collab.hello_fedavg.hello_fedavg --runtime prod \
    --startup-kit /path/to/prod_00/admin@example.com                  # provisioned deployment
python -m collab.hello_fedavg.hello_fedavg --runtime export --job-root /tmp/jobs
```

## Core examples

| Example | Demonstrates |
|---|---|
| `hello_collab` | Minimal local NumPy collab with classes or an automatically wrapped function module |
| `hello_numpy_collab` | Same Recipe API/`SimEnv` flow as `hello-world/hello-numpy`, using direct calls instead of Client API messaging |
| `hello_fedavg` | The Collab API in one file: `@collab.main`, `@collab.publish`, `collab.clients.train(...)`, per-site config |
| `call_patterns` | `--pattern seq \| cyclic` server-to-client invocation styles (parallel group call is in `hello_fedavg`) |
| `async_filters_metrics` | In-time (asynchronous) aggregation, call/result filter chains, metrics tracking |
| `swarm_events` | Decentralized swarm learning with client-to-client calls and events |
| `workflow_composition` | Chaining workflows in one `@collab.main`; resource dirs, artifacts, `@collab.final` |
| `split_learning` | Computation-equivalent CIFAR-10 SplitNN using direct function calls |

## Advanced and integration examples

These examples focus on integration with existing training APIs and
distributed execution rather than the core Pythonic Collab programming model.

| Example | Demonstrates |
|---|---|
| `client_api` | Standard Client API training code under collab: `--mode in_process \| subprocess` |
| `client_api_ddp` | torchrun/DDP per client: `--sync checkpoint \| broadcast` |

Shared support code lives in `common/` (numpy trainer and strategies, torch
helpers, and the `--runtime` selector). Support modules
used by a single example live in that example's directory.

The NumPy core examples run in a base installation; `hello_fedavg`,
`async_filters_metrics --flavor pt`, and `split_learning` need PyTorch
(`split_learning` also needs torchvision). In the
advanced/integration section, `client_api` and `client_api_ddp` need PyTorch.

For the design behind the API see `docs/design/collab_api_design.md`; for a
step-by-step migration from local training to collab see
`docs/design/collab_api_migration_tutorial.md`.
