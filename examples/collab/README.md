# Collab API Examples

The catalog contains self-contained examples of the core Collab programming
model. `hello_numpy_collab` mirrors the Recipe API and `SimEnv` flow of
`hello-world/hello-numpy`. Every entry point builds a `CollabRecipe` and runs it
directly with the standard `SimEnv`:

```bash
cd examples   # makes the collab package available to module imports

python -m collab.hello_numpy_collab.hello_numpy_collab
python -m collab.hello_fedavg.hello_fedavg
python -m collab.simple_split_learning.simple_split_learning
python -m collab.async_aggregation.async_aggregation
python -m collab.swarm.swarm --num-clients 3
```

## Examples

| Example | Demonstrates |
|---|---|
| `hello_numpy_collab` | Same Recipe API/`SimEnv` flow as `hello-world/hello-numpy`, using direct calls instead of Client API messaging |
| `hello_fedavg` | The Collab API in one file: `@collab.main`, `@collab.publish`, `collab.clients.train(...)`, per-site config |
| `simple_split_learning` | Split learning on MNIST with client-side images and bottom model, server-side labels and top model, and direct activation/gradient exchange |
| `async_aggregation` | In-time aggregation with a response callback |
| `swarm` | Decentralized swarm learning with client-to-client calls |

Every server object or module must define exactly one `@collab.main` entry
point. A workflow with multiple stages should call them from that single entry
point.

Each example is self-contained. Its entry point and any trainer,
strategy, widget, or utility modules that it needs live together in that
example's directory; there is no shared `common` package. A helper used by more
than one example is intentionally kept with each consumer so an example can be
copied or adapted on its own.

To use another deployment mode, execute the same recipe with `PocEnv` or
`ProdEnv` from `nvflare.recipe`; Collab has no separate runner or environment
abstraction.

The NumPy examples run in a base installation; `hello_fedavg` needs PyTorch.
`simple_split_learning` needs PyTorch and torchvision and downloads MNIST on
its first run.

For the design behind the API see the
[Collab API design](../../docs/design/collab_api_design.md). For a step-by-step
migration from local training to Collab see the
[migration tutorial](../../docs/design/collab_api_migration_tutorial.md).
