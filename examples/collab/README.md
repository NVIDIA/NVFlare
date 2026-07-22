# Collab API Examples

The catalog is organized into core examples and advanced/integration
examples. `hello_numpy_collab` mirrors the Recipe API and `SimEnv` flow of
`hello-world/hello-numpy`. The remaining core examples are built around a
`make_recipe()` function; their execution environment is an option
(`--runtime in_process | multi_process | prod | export`), never a separate
example — the recipe is identical across all of them:

```bash
cd examples   # makes the collab package available to module imports

python -m collab.hello_numpy_collab.hello_numpy_collab                # same Recipe API as hello-numpy, less client plumbing
python -m collab.hello_fedavg.hello_fedavg                            # threads in this process
python -m collab.async_aggregation.async_aggregation                  # aggregate responses as they arrive
python -m collab.hello_fedavg.hello_fedavg --runtime multi_process   # real FLARE processes (local POC)
python -m collab.hello_fedavg.hello_fedavg --runtime prod \
    --startup-kit /path/to/prod_00/admin@example.com                  # provisioned deployment
python -m collab.hello_fedavg.hello_fedavg --runtime export --job-root /tmp/jobs
```

## Core examples

| Example | Demonstrates |
|---|---|
| `hello_numpy_collab` | Same Recipe API/`SimEnv` flow as `hello-world/hello-numpy`, using direct calls instead of Client API messaging |
| `hello_fedavg` | The Collab API in one file: `@collab.main`, `@collab.publish`, `collab.clients.train(...)`, per-site config |
| `call_patterns` | `--pattern seq \| cyclic` server-to-client invocation styles (parallel group call is in `hello_fedavg`) |
| `async_aggregation` | In-time aggregation with a response callback |
| `swarm_events` | Decentralized swarm learning with client-to-client calls and events |
| `workflow_composition` | Chaining workflows in one `@collab.main`; resource dirs, artifacts, `@collab.final` |

Every server object or module must define exactly one `@collab.main` entry
point. `workflow_composition` demonstrates how to call multiple workflow stages
from that single entry point.

## Advanced and integration examples

These examples focus on integration with existing training APIs and
distributed execution rather than the core Pythonic Collab programming model.

| Example | Demonstrates |
|---|---|
| `client_api` | Standard Client API training code running in each client's FLARE process |

Each example is self-contained. Its entry point and any runner, trainer,
strategy, widget, or utility modules that it needs live together in that
example's directory; there is no shared `common` package. A helper used by more
than one example is intentionally kept with each consumer so an example can be
copied or adapted on its own.

The example-local `runner.py` files only provide the `--runtime` command-line
convenience shown above. They are not part of the public Collab API. Application
code should execute `CollabRecipe` with `SimEnv`, `PocEnv`, or `ProdEnv` from
`nvflare.recipe`.

The NumPy core examples run in a base installation; `hello_fedavg` needs
PyTorch. In the advanced/integration section, `client_api` needs PyTorch.

For the design behind the API see the
[Collab API design](../../docs/design/collab_api_design.md). For a step-by-step
migration from local training to Collab see the
[migration tutorial](../../docs/design/collab_api_migration_tutorial.md).
