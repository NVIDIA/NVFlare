# Collab API Design

## Status and Scope

The Collab API is a Python function-call interface for writing federated and
distributed workflows. A user publishes client functions, writes a server
workflow that calls those functions through proxies, and packages both sides in
a normal NVFlare `FedJob` through `CollabRecipe`.

This version deliberately has one execution contract:

- `CollabRecipe` always finalizes to a regular `FedJob`.
- The job runs with the standard Recipe environments: `SimEnv`, `PocEnv`, or
  `ProdEnv`.
- Server and client code runs inside its FLARE site process.
- External worker processes and multi-rank/DDP execution are not part of the
  current public API.

This boundary keeps deployment, lifecycle, call context, and failure handling
consistent across simulation, POC, and production. A future external-process
feature must define a typed, per-site execution policy and preserve the full
Collab lifecycle before it is added to the public contract.

## User Model

The smallest function-based program has a published client function and one
server `@collab.main` function:

```python
from nvflare.collab import CollabRecipe, collab
from nvflare.recipe import SimEnv


@collab.publish
def train(value):
    return value + 1


@collab.main
def run():
    results = collab.clients.train(value=1)
    return list(results)


recipe = CollabRecipe(
    job_name="collab_quickstart",
    min_clients=2,
)
recipe.execute(SimEnv(num_clients=2))
```

Modules containing decorated functions and class instances containing decorated
methods are both supported. Modules are wrapped by `ModuleWrapper` so the same
object model can be serialized into a job.

## How Collab Differs from Ordinary RPC

Collab calls resemble local Python calls, but the contract is richer than a
request/response transport:

- A proxy can represent one site, a named object at that site, or a group of
  sites.
- A group call broadcasts one logical invocation and returns results associated
  with their target sites.
- `@collab.init`, `@collab.main`, and `@collab.final` define application
  lifecycle behavior.
- Every invocation has a `Context` carrying caller, callee, abort state,
  workspace access, application properties, and group information.
- Calls support options such as timeout, optional delivery, result expectation,
  security, and maximum parallelism.
- Group response callbacks can run concurrently on backend result-delivery
  threads. Callbacks that mutate shared state must synchronize that access.
- A failed single-site call raises by default so workflow code can recover with
  `try`/`except CollabCallError`. An `optional=True` call logs a warning and
  returns `None` instead. Group iteration yields successful results only;
  `results.failures` maps failed sites to their `CollabCallError` objects.

## Public API

The top-level `nvflare.collab` package exports:

- `collab`: decorators, context accessors, proxies, and application properties.
- `CollabCallError`: structured site/function/cause details for failed calls.
- `CollabRecipe`: creates the normal NVFlare job.
- `simple_logging`: convenience logging setup for examples.

Execution environments come from `nvflare.recipe`, not `nvflare.collab`:

```python
from nvflare.recipe import PocEnv, ProdEnv, SimEnv
```

Transport classes and transport-type flags are intentionally not public. User
code describes logical calls and lets the selected standard environment choose
the deployment topology.

## Decorators and Lifecycle

### `@collab.publish`

Publishes a function or method so a remote proxy can invoke it. The generated
publish interface records the accepted argument names and is exchanged when a
job starts.

### `@collab.init`

Runs after a site app is set up with its workspace, proxies, abort signal, and
properties, but before normal calls are handled. An object may define multiple
initializers; all of them run once in method-name order.

### `@collab.main`

Defines the server workflow. A server application must define exactly one
`@collab.main` function or method. The workflow can call `collab.clients`,
`collab.server`, named child objects, or selected groups. Applications with
multiple stages should use one wrapper `@collab.main` and invoke those stages
explicitly in the required order.

### `@collab.final`

Runs when the site completes so the application can release resources and
persist final state.

## Context and Application Properties

A published function may request the current context explicitly:

```python
from nvflare.collab import collab
from nvflare.collab.api import Context


@collab.publish
def train(value, context: Context = None):
    site_name = context.app.name
    workspace = context.workspace
    learning_rate = context.app.get_prop("learning_rate", 0.01)
    if context.is_aborted():
        return None
    return value * learning_rate
```

The facade provides shorter access when passing the context is unnecessary:

```python
site_name = collab.site_name
caller = collab.caller
workspace = collab.workspace
fl_ctx = collab.fl_ctx
learning_rate = collab.get_app_prop("learning_rate", 0.01)
```

`collab.workspace` is the site's standard `nvflare.apis.workspace.Workspace`.
For example, the current job's run directory is
`collab.workspace.get_run_dir(collab.fl_ctx.get_job_id())`.

`collab.fl_ctx` is the live `FLContext` for the current site. Because it is a
runtime object, `CollabRecipe` adds it to the site's application properties
when the server or client app starts rather than serializing it into the job.

Per-site values use the standard Recipe configuration mechanism:

```python
recipe.set_per_site_config(
    {
        "site-1": {"learning_rate": 0.01},
        "site-2": {"learning_rate": 0.02},
    }
)
```

`CollabRecipe` stores the map in the client job configuration. Each
`CollabExecutor` resolves only its own site's values into application
properties at run start.

## Architecture

```text
User workflow and published functions
                 |
                 v
        Proxy / ProxyList / Group
                 |
                 v
     private invocation dispatcher
                 |
                 v
         Cell dispatcher
          (FLARE CellNet)
                 |
                 v
       target App + lifecycle
```

### Application layer

`App`, `ServerApp`, and `ClientApp` own:

- the primary user object and named Collab objects;
- publish interfaces;
- lifecycle functions;
- application properties;
- site proxies, workspace, and abort signal.

### Proxy layer

`Proxy` validates calls against the target publish interface and represents a
single logical target. `ProxyList` and `Group` coordinate calls to multiple
targets, including concurrency limits and target-attributed results.
`collab.other_clients` returns the client group with the current client removed,
which is useful for decentralized client-to-client workflows.

### Private invocation layer

`_InvocationDispatcher` is an internal strategy used by proxies. It is private
because applications should not choose transport or branch on transport type.
`CellDispatcher` serializes a logical invocation onto CellNet and is used by
`CollabController` and `CollabExecutor` in every standard environment.

Invocation dispatch is separate from execution placement. The current execution
placement is always the FLARE site process.

### FLARE runtime layer

`CollabRecipe.finalize()` adds:

- a `CollabController` and server application components to the server app;
- a `CollabExecutor` and client application components to each client app;
- user source files to each job's `custom` directory.

At startup, the controller and executors exchange publish interfaces. Each side
registers a CellNet callback, builds proxies for the other sites, sets up the
application, and enters the Collab lifecycle.

## Standard Environments

The recipe is independent of where it runs:

```python
# Fast local simulation
recipe.execute(SimEnv(num_clients=2))

# Local multi-process POC deployment
env = PocEnv(num_clients=2)
try:
    recipe.execute(env)
finally:
    env.stop(clean_up=True)

# Existing provisioned deployment
recipe.execute(ProdEnv(startup_kit_location="/path/to/admin/startup-kit"))
```

All three paths deploy the same finalized `FedJob`. There are no Collab-specific
environment classes and no recipe-wide subprocess command fields.

## Source Layout

```text
nvflare/collab/
├── api/                 programming model and private dispatcher contract
├── recipe.py            CollabRecipe -> FedJob
└── runtime/
    ├── controller.py    server workflow integration
    ├── executor.py      client site integration
    ├── cell_dispatcher.py / dispatch.py
    └── lifecycle.py     init/main/final orchestration
```

Examples are under [`examples/collab`](../../examples/collab/README.md). Each
example owns the trainer, strategy, and utility modules that it uses;
there is no shared `examples/collab/common` package. This keeps every example
self-contained and makes its dependencies visible in one directory.

Each example executes its recipe with `SimEnv` directly. Applications select
`SimEnv`, `PocEnv`, or `ProdEnv` from `nvflare.recipe`; there is no additional
Collab runner.

## Deferred Work

Application-supplied call/result middleware is not a security filter boundary.
A future direct-call filter design must make policy site-owned and enforce it
consistently for native Collab calls and task-plane Executor clients before any
`CallFilter` or `ResultFilter` API is published.

Bridging Collab to the Client API is also deferred. The bridge should schedule a
normal NVFlare `Task` whose configured `Shareable -> Shareable` Executor runs
through `ClientRunner`, preserving site filters, lifecycle, abort handling,
accounting, external-process execution, and DDP. It must not host an Executor or
a callback behind a client-side Collab RPC.

Collab-specific events are deferred; published functions provide the current
notification mechanism without introducing a second event system.

External-process execution is intentionally deferred. Before it can become a
public feature, its design must specify:

- a typed execution policy selectable independently per server/client site;
- object reconstruction and `init`/published-call/`final` lifecycle parity;
- exception, abort, timeout, workspace, and application-property propagation;
- multi-rank invocation semantics that call every participating rank safely;
- tracking and result aggregation without a control-rank deadlock.

Free-form launcher commands and global recipe flags are not a durable substitute
for that contract.
