# FLARE Collab API - Architecture Design Document

## Overview

FLARE Collab API, code name Fox (FLARE Object Exchange), is NVIDIA FLARE's collaborative API that simplifies federated learning development by allowing researchers to write natural Python code while the framework handles distribution, communication, and orchestration.

### Design Goals

1. **Simplicity**: Write federated learning code as if it were local Python
2. **Flexibility**: Support in-process simulation, multi-process POC, and distributed production
3. **Transparency**: Hide infrastructure complexity from data scientists
4. **Compatibility**: Minimal code changes when moving from simulation to production
5. **Extensibility**: Plugin architecture for backends, tracking, and execution environments

---

## Architecture Layers

The Fox architecture consists of layered components. The **Subprocess Layer is OPTIONAL**
and only used for multi-GPU, multi-node, or custom script execution. The **Tracking Layer
works in BOTH in-process and subprocess modes**.

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                   USER CODE LAYER                                    │
│                                                                                      │
│   @fox.algo, @fox.collab, @fox.init decorators                                      │
│   User's training logic, aggregation functions                                       │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                    API LAYER                                         │
│                                                                                      │
│   FoxRecipe, ModuleWrapper, App, Proxy, ProxyList, Group, CallOption                │
│   Handles method decoration, remote call abstraction, result aggregation            │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                               EXECUTION ENVIRONMENT LAYER                            │
│                                                                                      │
│   SimEnv (in-process threads)     │    PocEnv (multi-process local)                 │
│   FoxSimulator                    │    POC Infrastructure                           │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                   BACKEND LAYER                                      │
│                                                                                      │
│   SimBackend (thread-based)       │    FlareBackend (CellNet-based)                 │
│   Direct function calls           │    Network communication                        │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    │                                           │
                    ▼                                           ▼
┌───────────────────────────────────────┐   ┌───────────────────────────────────────────┐
│        IN-PROCESS EXECUTION           │   │   SUBPROCESS EXECUTION (OPTIONAL)         │
│        (Default - inprocess=True)     │   │   (For multi-GPU/node - inprocess=False)  │
│                                       │   │                                           │
│   @fox.collab methods run directly    │   │   SubprocessLauncher spawns FoxWorker    │
│   in FoxExecutor process              │   │   via torchrun or custom launcher        │
│                                       │   │                                           │
│   ┌─────────────────────────────┐     │   │   ┌─────────────────────────────────┐     │
│   │ InProcessWriter             │     │   │   │ SubprocessWriter                │     │
│   │ (direct event firing)       │     │   │   │ (CellNet relay to parent)       │     │
│   └─────────────────────────────┘     │   │   └─────────────────────────────────┘     │
└───────────────────────────────────────┘   └───────────────────────────────────────────┘
                    │                                           │
                    └─────────────────────┬─────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                          TRACKING LAYER (Works in Both Modes)                        │
│                                                                                      │
│   ┌───────────────────────────────────────────────────────────────────────────────┐ │
│   │ API-Compatible Writers (TensorBoardWriter, MLflowWriter, WandbWriter)         │ │
│   │                                                                               │ │
│   │   ↓ Auto-detects execution mode via AutoWriter ↓                              │ │
│   │                                                                               │ │
│   │   In-Process Mode          │    Subprocess Mode                               │ │
│   │   InProcessWriter          │    SubprocessWriter                              │ │
│   │   → fire_event directly    │    → CellNet → MetricsRelay → fire_event         │ │
│   └───────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                      │
│   → TrackingReceiver (TensorBoard/MLflow/W&B) on Server                             │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                               COMMUNICATION LAYER                                    │
│                                                                                      │
│   CellNet (Channels: fox/call, fox_worker, fox_metrics)                             │
│   FOBS Serialization, TensorDecomposer                                              │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### Execution Mode Decision Tree

```
                        ┌─────────────────────────┐
                        │  FoxExecutor starts     │
                        └───────────┬─────────────┘
                                    │
                        ┌───────────▼───────────┐
                        │  inprocess=True?      │
                        └───────────┬───────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │ YES                 │                     │ NO
              ▼                     │                     ▼
┌─────────────────────────┐         │     ┌─────────────────────────────────┐
│ IN-PROCESS MODE         │         │     │ SUBPROCESS MODE                 │
│                         │         │     │                                 │
│ • @fox.collab runs in   │         │     │ • SubprocessLauncher spawns     │
│   same process          │         │     │   FoxWorker process             │
│ • Direct method calls   │         │     │ • Uses launcher_cmd (torchrun)  │
│ • InProcessWriter for   │         │     │ • CellNet for communication     │
│   tracking              │         │     │ • SubprocessWriter for tracking │
└─────────────────────────┘         │     └─────────────────────────────────┘
                                    │
                        Use Cases:  │
                                    │
    ┌─────────────────────────────────────────────────────────────────────┐
    │                                                                     │
    │  IN-PROCESS (Default)          │  SUBPROCESS (Optional)            │
    │  ────────────────────          │  ─────────────────────            │
    │  • Single-GPU training         │  • Multi-GPU (torchrun DDP)       │
    │  • Simple algorithms           │  • Multi-node training            │
    │  • Quick prototyping           │  • Custom launchers               │
    │  • SimEnv simulation           │  • Isolated execution             │
    │                                │                                   │
    └─────────────────────────────────────────────────────────────────────┘
```

---

## Layer 1: User Code Layer

### Decorators

| Decorator | Purpose | Location |
|-----------|---------|----------|
| `@fox.algo` | Marks server-side orchestration methods | Server module |
| `@fox.collab` | Marks client-side collaborative methods | Client module |
| `@fox.init` | Marks one-time initialization methods | Both |

### Two API Patterns

Fox supports two distinct API patterns for client-side training:

| Pattern | Description | Use Case |
|---------|-------------|----------|
| **Collab API** | Decorator-based (`@fox.collab`) | Simple training, server-driven workflow |
| **Client API** | Receive/send pattern (`flare.receive()`, `flare.send()`) | DDP training, client-driven workflow |

### Collab API Example

```python
# Server-side (aggregation)
@fox.algo
def fed_avg(self):
    weights = self.model.state_dict()
    for round in range(self.num_rounds):
        results = fox.clients.train(weights)  # Parallel client calls
        weights = weighted_avg(results)
    return weights

# Client-side (training)
@fox.collab
def train(self, weights=None):
    model = SimpleModel()
    if weights:
        model.load_state_dict(weights)
    # ... training loop ...
    return model.state_dict(), len(dataset)
```

### Client API Example

```python
# Server-side (server.py)
@fox.algo
def fed_avg(self):
    weights = self.model.state_dict()
    for round in range(self.num_rounds):
        input_model = FLModel(params=weights, current_round=round)
        results = fox.clients.execute(fl_model=input_model)  # Calls execute() on FoxClientAPI
        weights = aggregate(results)
    fox.clients.stop()  # Signal clients to stop

# Client-side (client.py) - runs top-to-bottom, no decorators
from nvflare.fox.sys.worker import get_client_api
flare = get_client_api()
flare.init()

while flare.is_running():
    input_model = flare.receive()
    if input_model is None:
        break
    # ... training ...
    output_model = FLModel(params=new_weights, metrics={"loss": loss})
    flare.send(output_model)
```

---

## Layer 2: API Layer

### Core Components

#### FoxRecipe (`nvflare/fox/sys/recipe.py`)

Job configuration and orchestration entry point.

```python
# Collab API (auto-detects training module from client)
recipe = FoxRecipe(
    job_name="fedavg",
    min_clients=5,
    server=server_module,      # Module with @fox.algo
    client=client_module,      # Module with @fox.collab
    inprocess=True,            # or False for subprocess
    run_cmd="torchrun --nproc_per_node=4",
    tracking_type="tensorboard",
)

# Client API (explicit training_module for subprocess)
from nvflare.client.in_process.fox_api import FoxClientAPI

recipe = FoxRecipe(
    job_name="fedavg",
    min_clients=2,
    server=FedAvg(),           # Server with @fox.algo calling execute()/stop()
    client=FoxClientAPI(),     # Client API bridge
    inprocess=False,
    run_cmd="torchrun --nproc_per_node=2",
    training_module="my_package.client",  # Explicit path to client script
)
```

**Key Parameters:**

| Parameter | Description |
|-----------|-------------|
| `server` | Server object with `@fox.algo` methods |
| `client` | Client object with `@fox.collab` methods, or `FoxClientAPI()` |
| `inprocess` | True for in-process, False for subprocess |
| `run_cmd` | Subprocess launcher command (e.g., `torchrun --nproc_per_node=4`) |
| `training_module` | Explicit training module path (required for Client API) |

**Key Responsibilities:**
- Auto-detect and wrap modules with `ModuleWrapper`
- Configure execution mode (in-process vs subprocess)
- Set up tracking configuration
- Generate FLARE job configuration
- Pass `training_module` to subprocess launcher

#### ModuleWrapper (`nvflare/fox/api/module_wrapper.py`)

Enables Python modules to be used as Fox client/server objects.

```python
# Wraps a module containing @fox.collab or @fox.algo functions
wrapper = ModuleWrapper(my_training_module)
```

**Key Features:**
- Discovers decorated functions in a module
- Creates callable bound methods
- Handles serialization (only stores module name)
- Resolves `__main__` modules to importable paths

#### App (`nvflare/fox/api/app.py`)

Base class for `ServerApp` and `ClientApp`.

```
App
├── ServerApp  - Wraps server object, discovers @fox.algo methods
└── ClientApp  - Wraps client object, discovers @fox.collab methods
```

#### Proxy & ProxyList (`nvflare/fox/api/proxy.py`)

Remote callable abstraction.

```python
# Single proxy - represents one remote target
proxy = Proxy(backend, target_name="site-1", call_opt=CallOption())
result = proxy.train(weights)  # Calls train() on site-1

# ProxyList - represents multiple targets
clients = ProxyList([proxy1, proxy2, proxy3])
results = clients.train(weights)  # Parallel calls to all clients
```

#### Group (`nvflare/fox/api/group.py`)

Handles parallel invocation of methods on proxy lists.

```python
group = Group(targets, backend, call_opt)
results = group.train(weights)  # Returns list of results
```

#### CallOption (`nvflare/fox/api/call_opt.py`)

Configuration for remote calls.

```python
call_opt = CallOption(
    expect_result=True,    # Wait for result
    blocking=True,         # Synchronous call
    timeout=60.0,          # Call timeout
    secure=False,          # Use secure channel
    parallel=0,            # Parallelism level
)
```

#### FoxClientAPI (`nvflare/client/in_process/fox_api.py`)

Implements the standard NVFlare Client API (`APISpec`) using Fox Collab API internally.
**FoxClientAPI is an adapter** that converts the push-based Collab API to the pull-based
Client API pattern.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Push ↔ Pull Adapter Pattern                               │
│                                                                              │
│   Collab API (Push)                    Client API (Pull)                     │
│   ─────────────────                    ─────────────────                     │
│   Server pushes task to client         Client pulls task from server         │
│                                                                              │
│   fox.clients.execute(model) ────┐ ┌──── model = flare.receive()             │
│                                  │ │                                         │
│                                  ▼ ▼                                         │
│                         ┌────────────────┐                                   │
│                         │  FoxClientAPI   │                                  │
│                         │   (Adapter)     │                                  │
│                         └────────────────┘                                   │
│                                  │ │                                         │
│   result = execute() ◀───────────┘ └────── flare.send(result)                │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Two Modes:**

| Mode | Mechanism | Use Case |
|------|-----------|----------|
| **In-Process** | Direct variable storage | Simple simulation, no subprocess |
| **Subprocess** | Queue-based bridging | Multi-GPU (torchrun), DDP |

**In-Process Mode (Simple Adapter):**

```python
# No queues, no threading - just variable assignment
class FoxClientAPI:
    @collab
    def execute(self, fl_model, ...):
        self._current_model = fl_model    # Store for receive()
        self._training_func()             # User code runs here
        return self._result               # Return what send() stored
    
    def receive(self):
        return self._current_model        # Direct access
    
    def send(self, model):
        self._result = model              # Direct storage
```

**Subprocess Mode (Queue-Based):**

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    Subprocess Mode: Queue-Based Bridging                      │
│                                                                               │
│   Server                              Client (Subprocess)                     │
│   ┌─────────────────────┐             ┌──────────────────────────────────┐   │
│   │ @fox.algo           │             │         FoxClientAPI             │   │
│   │                     │   CellNet   │                                  │   │
│   │ fox.clients.execute │────────────▶│ @fox.collab execute()            │   │
│   │   (fl_model)        │             │   → puts model in _call_queue    │   │
│   │                     │             │   → waits on _result_queue       │   │
│   │                     │             │                                  │   │
│   │                     │             │ ┌────────────────────────────┐   │   │
│   │                     │             │ │   User's Training Script   │   │   │
│   │                     │             │ │                            │   │   │
│   │                     │             │ │   flare.receive()          │   │   │
│   │                     │             │ │     ← gets from _call_queue│   │   │
│   │                     │             │ │                            │   │   │
│   │                     │             │ │   flare.send(result)       │   │   │
│   │ receives result     │◀────────────│ │     → puts in _result_queue│   │   │
│   │                     │             │ └────────────────────────────┘   │   │
│   │                     │             │                                  │   │
│   │ fox.clients.stop()  │────────────▶│ @fox.collab stop()               │   │
│   │                     │             │   → sets _running = False        │   │
│   └─────────────────────┘             └──────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Key Methods:**

| Method | Role | Description |
|--------|------|-------------|
| `init()` | Client | Initialize the API (called by user script) |
| `receive()` | Client | Block until server sends model via `execute()` |
| `send(model)` | Client | Send result back, unblocking `execute()` |
| `is_running()` | Client | Returns False after `stop()` is called |
| `execute()` | Server (`@fox.collab`) | Called by server, bridges to receive/send |
| `stop()` | Server (`@fox.collab`) | Signal client to exit its loop |
| `make_client_app()` | Factory | Creates new instance for each client site |

---

## Layer 3: Execution Environment Layer

### ExecEnv Interface

Abstract base class for execution environments.

```python
class ExecEnv:
    def deploy(self, recipe: FoxRecipe) -> Run:
        """Deploy and execute the job"""
        pass
    
    def stop(self):
        """Stop the environment"""
        pass
```

### SimEnv (`nvflare/fox/sim/sim_env.py`)

In-process thread-based simulation.

```
┌─────────────────────────────────────────────────────────────────┐
│                          SimEnv                                  │
│                                                                  │
│  ┌─────────────────┐    ┌─────────────────────────────────────┐ │
│  │   FoxSimulator  │    │          ThreadPoolExecutor          │ │
│  │                 │    │                                      │ │
│  │  - Simulates    │    │  Thread-1: site-1 ClientApp          │ │
│  │    server &     │    │  Thread-2: site-2 ClientApp          │ │
│  │    clients      │    │  Thread-3: site-3 ClientApp          │ │
│  │                 │    │  ...                                 │ │
│  └─────────────────┘    └─────────────────────────────────────┘ │
│                                                                  │
│  Backend: SimBackend (direct function calls)                     │
└─────────────────────────────────────────────────────────────────┘
```

**Characteristics:**
- Single process, multiple threads
- Fast iteration for development
- No network overhead
- Shared memory between components

### PocEnv (`nvflare/fox/sys/poc_env.py`)

Multi-process **local** execution using POC infrastructure.

> **Note**: PocEnv runs everything on a **single machine** for local development and testing.
> Server and clients are co-located only for convenience - in production, they are distributed.

```
┌─────────────────────────────────────────────────────────────────┐
│                    PocEnv (Single Machine)                       │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    POC Infrastructure                     │   │
│  │                                                           │   │
│  │  Process-1: Server (FoxController)                        │   │
│  │  Process-2: site-1 (FoxExecutor)                          │   │
│  │  Process-3: site-2 (FoxExecutor)                          │   │
│  │  ...                                                      │   │
│  │                                                           │   │
│  │  (All processes on same machine for local testing)        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Backend: FlareBackend (CellNet communication)                   │
└─────────────────────────────────────────────────────────────────┘
```

**Characteristics:**
- Multiple processes on **single machine** (for development/testing)
- Real FLARE runtime components
- CellNet for inter-process communication
- Validates production behavior locally before distributed deployment

---

## Layer 4: Backend Layer

### Backend Interface

```python
class Backend:
    def call(self, target, func_name, *args, **kwargs) -> Any:
        """Call a method on a target"""
        pass
```

### SimBackend (`nvflare/fox/sim/backend.py`)

Thread-based backend for simulation.

```
┌─────────────────────────────────────────────────────────────────┐
│                        SimBackend                                │
│                                                                  │
│  Server Thread                    Client Threads                 │
│  ┌──────────────┐                ┌──────────────┐               │
│  │  fed_avg()   │   Direct       │  site-1      │               │
│  │              │───call───────▶ │  train()     │               │
│  │              │◀──result────── │              │               │
│  └──────────────┘                └──────────────┘               │
│                                  ┌──────────────┐               │
│                                  │  site-2      │               │
│                                  │  train()     │               │
│                                  └──────────────┘               │
│                                                                  │
│  Communication: Thread-safe queues, direct function calls        │
└─────────────────────────────────────────────────────────────────┘
```

### FlareBackend (`nvflare/fox/sys/backend.py`)

CellNet-based backend for distributed execution.

> **Note**: In production, server and clients are on **different machines/locations**.
> The diagram shows logical connectivity, not physical co-location.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            FlareBackend (Distributed)                            │
│                                                                                  │
│  Server (Location A)                     Clients (Different Locations)          │
│  ┌──────────────────┐                   ┌──────────────────┐                    │
│  │  FoxController   │      CellNet      │  FoxExecutor     │                    │
│  │  (Cloud/VM)      │ ════(fox/call)═══▶│  site-1          │  (Site B)          │
│  │                  │ ◀═══response═════ │  (Hospital)      │                    │
│  └──────────────────┘                   └──────────────────┘                    │
│           │                             ┌──────────────────┐                    │
│           │                             │  FoxExecutor     │                    │
│           └═══════(fox/call)═══════════▶│  site-2          │  (Site C)          │
│                                         │  (Research Lab)  │                    │
│                                         └──────────────────┘                    │
│                                                                                  │
│  Communication: CellNet over network, FOBS serialization, mTLS                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Layer 5: Subprocess Layer (OPTIONAL)

> **Note:** This layer is **OPTIONAL** and only used when:
> - Multi-GPU training with `torchrun` or similar launchers
> - Multi-node distributed training
> - Custom script execution requiring isolated processes
>
> For single-GPU training or simple algorithms, the default in-process mode
> (`inprocess=True`) is sufficient and this layer is not used.

### Components

#### SubprocessLauncher (`nvflare/fox/sys/subprocess_launcher.py`)

Runs within `FoxExecutor` to manage subprocess.

```python
launcher = SubprocessLauncher(
    site_name="site-1",
    training_module="my_training",
    parent_cell=cell,
    launcher_cmd="torchrun --nproc_per_node=4",
    subprocess_timeout=300.0,
)
launcher.start()
result = launcher.call("train", args=(weights,))
launcher.stop()
```

**Responsibilities:**
- Spawn subprocess with environment variables
- Set up CellNet listener for worker connection
- Forward method calls to worker
- Manage subprocess lifecycle

#### FoxWorker (`nvflare/fox/sys/worker.py`)

Runs in the subprocess, connects back to parent. Supports two execution modes:

**Mode 1: Collab API (RPC-based)**
```python
# Worker loads module with @fox.collab methods
# Parent calls methods via CellNet RPC
```

**Mode 2: Client API (Script-based)**
```python
# Worker instantiates FoxClientAPI
# Worker runs user's training script directly via runpy
# Script uses get_client_api() to access the API
```

```python
# Environment variables set by SubprocessLauncher:
#   FOX_PARENT_URL, FOX_PARENT_FQCN, FOX_SITE_NAME, FOX_WORKER_ID
#   FOX_CLIENT_CLASS (determines mode: "FoxClientAPI" or module class)
```

**Responsibilities:**
- Connect to parent via CellNet
- Detect execution mode from `FOX_CLIENT_CLASS`
- For Collab API: Load module, wait for RPC calls
- For Client API: Instantiate `FoxClientAPI`, run user script
- Handle DDP rank coordination (only rank 0 communicates)

### Subprocess Architecture

The subprocess layer supports two distinct execution patterns:

#### Pattern 1: Collab API (RPC-based)

Server calls `@fox.collab` methods on the client. Worker waits for RPC calls.

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│                           Collab API Subprocess Flow                                │
│                                                                                     │
│  FoxExecutor                           Subprocess (torchrun)                        │
│  ┌─────────────────┐                   ┌────────────────────────────────────────┐  │
│  │ SubprocessLaunch│    fox_worker     │              FoxWorker                 │  │
│  │                 │    /call          │                                        │  │
│  │  call("train")  │──────────────────▶│  Load module                           │  │
│  │                 │                   │  Find @fox.collab method               │  │
│  │                 │                   │  Execute train()                       │  │
│  │  receives result│◀──────────────────│  Return result                         │  │
│  └─────────────────┘                   └────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────────────┘
```

#### Pattern 2: Client API (Script-based)

Server calls `execute()`/`stop()` on FoxClientAPI. Worker runs user script directly.

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│                           Client API Subprocess Flow                                │
│                                                                                     │
│  FoxExecutor                           Subprocess (torchrun)                        │
│  ┌─────────────────┐                   ┌────────────────────────────────────────┐  │
│  │ SubprocessLaunch│    fox_worker     │              FoxWorker                 │  │
│  │                 │    /call          │                                        │  │
│  │                 │                   │  1. Detect client_class=FoxClientAPI   │  │
│  │                 │                   │  2. Create FoxClientAPI instance       │  │
│  │                 │                   │  3. runpy.run_module(client.py)        │  │
│  │                 │                   │                                        │  │
│  │ call("execute") │──────────────────▶│  4. execute() puts model in queue      │  │
│  │                 │                   │                                        │  │
│  │                 │                   │  ┌──────────────────────────────────┐  │  │
│  │                 │                   │  │  User's client.py (running)      │  │  │
│  │                 │                   │  │                                  │  │  │
│  │                 │                   │  │  flare.receive()  ← gets model   │  │  │
│  │                 │                   │  │  # ... training ...              │  │  │
│  │                 │                   │  │  flare.send(result) → puts queue │  │  │
│  │                 │                   │  └──────────────────────────────────┘  │  │
│  │                 │                   │                                        │  │
│  │ receives result │◀──────────────────│  5. execute() returns from queue       │  │
│  │                 │                   │                                        │  │
│  │ call("stop")    │──────────────────▶│  6. stop() sets _running = False       │  │
│  │                 │                   │  7. User script exits while loop       │  │
│  └─────────────────┘                   └────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────────────┘
```

#### Full Architecture with DDP

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              FoxExecutor (Parent Process)                            │
│                                                                                      │
│  ┌────────────────────────────────────────────────────────────────────────────────┐ │
│  │                          SubprocessLauncher                                     │ │
│  │                                                                                 │ │
│  │  1. Set ENV vars: FOX_PARENT_URL, FOX_PARENT_FQCN, FOX_CLIENT_CLASS, etc.      │ │
│  │  2. Spawn: torchrun --nproc_per_node=4 -m nvflare.fox.sys.worker my_training   │ │
│  │  3. Wait for ready signal                                                       │ │
│  │  4. Forward calls via CellNet (fox_worker/call)                                │ │
│  │                                                                                 │ │
│  └────────────────────────────────────────────────────────────────────────────────┘ │
│                                          │                                           │
│                                          │ CellNet                                   │
│                                          ▼                                           │
└──────────────────────────────────────────┼───────────────────────────────────────────┘
                                           │
┌──────────────────────────────────────────┼───────────────────────────────────────────┐
│                              Subprocess (torchrun)                                    │
│                                          │                                           │
│  ┌───────────────────────────────────────▼───────────────────────────────────────┐  │
│  │                              FoxWorker (Rank 0)                                │  │
│  │                                                                                │  │
│  │  1. Read ENV vars, connect to parent                                           │  │
│  │  2. Detect mode from FOX_CLIENT_CLASS                                          │  │
│  │  3. Signal ready                                                               │  │
│  │  4. Collab API: Wait for RPC calls                                             │  │
│  │     Client API: Run user script, handle execute()/stop()                       │  │
│  │  5. Return results to parent                                                   │  │
│  │                                                                                │  │
│  │  ┌─────────────────────────────────────────────────────────────────────────┐  │  │
│  │  │  User's Training Code                                                    │  │  │
│  │  │                                                                          │  │  │
│  │  │  # Collab API                      │  # Client API                       │  │  │
│  │  │  @fox.collab                       │  flare = get_client_api()           │  │  │
│  │  │  def train(weights):               │  while flare.is_running():          │  │  │
│  │  │      model = DDP(model)            │      model = flare.receive()        │  │  │
│  │  │      # training...                 │      # training with DDP...         │  │  │
│  │  │      return weights                │      flare.send(result)             │  │  │
│  │  └─────────────────────────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                       │
│  ┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐             │
│  │  Rank 1 (Worker)    │ │  Rank 2 (Worker)    │ │  Rank 3 (Worker)    │             │
│  │  Participates in    │ │  Participates in    │ │  Participates in    │             │
│  │  DDP, no parent     │ │  DDP, no parent     │ │  DDP, no parent     │             │
│  │  communication      │ │  communication      │ │  communication      │             │
│  └─────────────────────┘ └─────────────────────┘ └─────────────────────┘             │
│                                                                                       │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Layer 6: Tracking Layer

The Tracking Layer works in **BOTH in-process and subprocess modes**. Writers
auto-detect the execution mode and use the appropriate underlying mechanism.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                     TRACKING LAYER (Works in Both Modes)                             │
└─────────────────────────────────────────────────────────────────────────────────────┘

User Code (same API regardless of execution mode):
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  from nvflare.fox.tracking import SummaryWriter  # or mlflow, wandb                 │
│  writer = SummaryWriter()                                                            │
│  writer.add_scalar("loss", 0.5, global_step=100)                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  API-Compatible Writers                                                              │
│                                                                                      │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐          │
│  │  TensorBoardWriter  │  │    MLflowWriter     │  │    WandbWriter      │          │
│  │  (SummaryWriter)    │  │                     │  │                     │          │
│  │                     │  │  log_metric()       │  │  log()              │          │
│  │  add_scalar()       │  │  log_param()        │  │  init()             │          │
│  │  add_scalars()      │  │  log_metrics()      │  │  config             │          │
│  └──────────┬──────────┘  └──────────┬──────────┘  └──────────┬──────────┘          │
│             │                        │                        │                      │
│             └────────────────────────┼────────────────────────┘                      │
│                                      ▼                                               │
│                          ┌─────────────────────────────────────────────────────────┐ │
│                          │              AutoWriter (Mode Detection)                │ │
│                          │                                                         │ │
│                          │  Checks FOX_PARENT_URL env var to detect mode          │ │
│                          └──────────────────────┬──────────────────────────────────┘ │
│                                                 │                                    │
│                    ┌────────────────────────────┼────────────────────────────┐       │
│                    │                            │                            │       │
│                    ▼                            │                            ▼       │
│     ┌──────────────────────────┐               │          ┌──────────────────────────┐│
│     │  IN-PROCESS MODE         │               │          │  SUBPROCESS MODE         ││
│     │                          │               │          │                          ││
│     │  InProcessWriter         │               │          │  SubprocessWriter        ││
│     │  → fire_event directly   │               │          │  → CellNet relay         ││
│     │    to event_manager      │               │          │    (fox_metrics/log)     ││
│     └────────────┬─────────────┘               │          └────────────┬─────────────┘│
│                  │                             │                       │              │
└──────────────────┼─────────────────────────────┼───────────────────────┼──────────────┘
                   │                             │                       │
                   │                             │                       │ CellNet
                   │                             │                       ▼
                   │                             │    ┌───────────────────────────────┐
                   │                             │    │  MetricsRelay (FoxExecutor)   │
                   │                             │    │                               │
                   │                             │    │  Receives from subprocess     │
                   │                             │    │  → fire_event                 │
                   │                             │    └───────────────┬───────────────┘
                   │                             │                    │
                   └─────────────────────────────┼────────────────────┘
                                                 │
                                                 ▼
                              ┌─────────────────────────────────────────┐
                              │         FLARE Event System              │
                              │    event_manager.fire_event             │
                              │         (TOPIC_LOG_DATA)                │
                              └────────────────────┬────────────────────┘
                                                   │
                                                   ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  Tracking Receiver (on Server - configured via add_experiment_tracking)             │
│                                                                                      │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐          │
│  │ TensorBoardReceiver │  │   MLflowReceiver    │  │   WandBReceiver     │          │
│  │                     │  │                     │  │                     │          │
│  │  Writes to actual   │  │  Logs to MLflow     │  │  Logs to W&B        │          │
│  │  TensorBoard        │  │  server             │  │  server             │          │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Purpose | Used When |
|-----------|---------|-----------|
| `AutoWriter` | Auto-detects execution mode | Always (internal) |
| `InProcessWriter` | Fires events directly | inprocess=True |
| `SubprocessWriter` | Sends via CellNet | inprocess=False |
| `MetricsRelay` | Receives from subprocess | inprocess=False |
| `TensorBoardWriter` | TensorBoard API | User choice |
| `MLflowWriter` | MLflow API | User choice |
| `WandbWriter` | W&B API | User choice |

### Usage Patterns

```python
# TensorBoard (same API as torch.utils.tensorboard)
from nvflare.fox.tracking import SummaryWriter
writer = SummaryWriter()
writer.add_scalar("train/loss", loss, global_step=step)

# MLflow (same API as mlflow)
from nvflare.fox.tracking import mlflow
mlflow.log_metric("loss", 0.5, step=100)
mlflow.log_param("lr", 0.001)

# W&B (same API as wandb)
from nvflare.fox.tracking import wandb
wandb.init(project="my-project")
wandb.log({"loss": 0.5, "accuracy": 0.9})
```

---

## Layer 7: Communication Layer

### CellNet Channels

| Channel | Topic | Purpose |
|---------|-------|---------|
| `fox/call` | `call` | Server ↔ Client method invocation |
| `fox_worker` | `call` | FoxExecutor ↔ FoxWorker method calls |
| `fox_worker` | `ready` | Worker signals ready to parent |
| `fox_worker` | `shutdown` | Graceful worker shutdown |
| `fox_metrics` | `log` | Metrics from subprocess to executor |

### Cell FQCN Hierarchy

The CellNet uses Fully Qualified Cell Names (FQCN) for message routing:

```
server                      # Server cell
├── app                     # Server app

site-1                      # Client 1 cell
├── app                     # Client 1 app (FoxExecutor)
│   └── worker.0            # Worker subprocess (rank 0)

site-2                      # Client 2 cell
├── app                     # Client 2 app
│   └── worker.0            # Worker subprocess
```

**Example FQCNs:**
- `site-1` - Client site cell
- `site-1.app` - Client app cell (where FoxExecutor runs)
- `site-1.app.worker.0` - Worker subprocess cell (child of app)

### FOBS Serialization

Fox uses FOBS (FLARE Object Serialization) for data transfer:

- **TensorDecomposer**: Efficient PyTorch tensor serialization
- **NumPy arrays**: Native support
- **Python primitives**: dict, list, str, int, float, etc.

---

## Configuration Flow

### Recipe to Job

```
FoxRecipe
    │
    ├── Wraps modules with ModuleWrapper
    ├── Configures FoxController (server)
    ├── Configures FoxExecutor (clients)
    ├── Sets subprocess options
    ├── Configures tracking
    │
    └── Generates FLARE Job
            │
            ├── config_fed_server.json
            │       └── FoxController component
            │
            └── config_fed_client.json
                    └── FoxExecutor component
                            ├── inprocess: true/false
                            └── run_cmd: "torchrun ..."
```

---

## Execution Modes

### Mode 1: In-Process Simulation (SimEnv)

```python
recipe = FoxRecipe(job_name="fedavg", min_clients=5)
env = SimEnv(num_clients=5)
run = recipe.execute(env)
```

- Single process, multiple threads
- Fast iteration
- No subprocess, no CellNet

### Mode 2: Multi-Process POC (PocEnv)

```python
recipe = FoxRecipe(job_name="fedavg", min_clients=3)
env = PocEnv(num_clients=3)
run = recipe.execute(env)
```

- Multiple processes, single machine
- Real FLARE runtime
- CellNet communication

### Mode 3: Subprocess with Multi-GPU (PocEnv + subprocess)

```python
recipe = FoxRecipe(
    job_name="fedavg",
    min_clients=2,
    inprocess=False,
    run_cmd="torchrun --nproc_per_node=4",
    # training_module auto-detected from client
)
env = PocEnv(num_clients=2)
run = recipe.execute(env)
```

- FoxExecutor spawns subprocess
- torchrun manages DDP across GPUs
- FoxWorker coordinates with parent

### Mode 4: Production Distributed

```python
recipe = FoxRecipe(...)
# Server: Deploy to cloud VM or on-premise server
# Clients: Deploy to each participating site (K8s, HPC, VM)
# Connection: mTLS, certificates provisioned for each site
```

- **Server and clients on different machines/networks**
- Full security features (mTLS, authentication)
- Production monitoring
- Data never leaves client sites

---

## Directory Structure

```
nvflare/fox/
├── DESIGN.md              # This document
├── __init__.py            # Package exports
│
├── api/                   # API Layer
│   ├── app.py             # App, ServerApp, ClientApp
│   ├── call_opt.py        # CallOption
│   ├── constants.py       # API constants
│   ├── context.py         # Execution context
│   ├── dec.py             # Decorators (@fox.algo, @fox.collab)
│   ├── group.py           # Group (parallel invocation)
│   ├── module_wrapper.py  # ModuleWrapper
│   ├── proxy.py           # Proxy, ProxyList
│   └── run_server.py      # Server execution logic
│
├── sim/                   # Simulation Layer
│   ├── backend.py         # SimBackend
│   ├── foxsimulator.py    # FoxSimulator
│   └── sim_env.py         # SimEnv
│
├── sys/                   # System/Runtime Layer
│   ├── adaptor.py         # FoxAdaptor
│   ├── backend.py         # FlareBackend
│   ├── controller.py      # FoxController
│   ├── executor.py        # FoxExecutor
│   ├── poc_env.py         # PocEnv
│   ├── recipe.py          # FoxRecipe
│   ├── subprocess_launcher.py  # SubprocessLauncher
│   ├── utils.py           # System utilities
│   ├── worker.py          # FoxWorker
│   └── ws.py              # Workspace utilities
│
├── tracking/              # Tracking Layer
│   ├── auto_writer.py     # AutoWriter (mode detection)
│   ├── base_writer.py     # BaseWriter abstract class
│   ├── constants.py       # Tracking constants
│   ├── inprocess_writer.py   # InProcessWriter (direct event)
│   ├── metrics_handler.py # MetricsRelay (subprocess→executor)
│   ├── mlflow_writer.py   # MLflow-compatible API
│   ├── subprocess_writer.py  # SubprocessWriter (CellNet)
│   ├── tensorboard_writer.py # TensorBoard-compatible API
│   └── wandb_writer.py    # W&B-compatible API
│
└── examples/              # Examples
    └── pt/
        ├── collab_api/               # Decorator-based (@fox.collab) examples
        │   ├── in_process/
        │   │   ├── collab_fedavg_train.py
        │   │   ├── collab_fedavg_no_class.py
        │   │   └── simulate_fedavg_train.py
        │   └── sub_process/
        │       └── distributed_train.py
        │
        └── client_api/               # Receive/send pattern examples
            └── sub_process/
                ├── server.py         # @fox.algo server with execute()/stop()
                ├── client.py         # Client script using receive()/send()
                ├── job.py            # FoxRecipe with FoxClientAPI
                └── sim_distributed_fedavg_train.py  # Standalone DDP simulation
```

---

## Implementation Notes

### Client API Mode: Key Implementation Details

1. **`make_client_app()` Method**: Required for Client API mode because `FoxClientAPI` contains
   unpickleable objects (threading `Queue`, `Lock`). The simulator calls this method to create
   fresh instances for each client instead of using `deepcopy()`.

2. **`get_client_api()` Function**: Enables user scripts to access the `FoxClientAPI` instance
   created by `FoxWorker`. This bridges the gap between the worker's internal state and the
   user's training script.

3. **Module Import Handling**: When Python runs with `-m nvflare.fox.sys.worker`, the module
   may be pre-imported. The worker explicitly updates `sys.modules['nvflare.fox.sys.worker']._client_api`
   to ensure `get_client_api()` returns the correct instance.

4. **DDP Rank Coordination**: In Client API mode with DDP, only rank 0 communicates with the
   server (via `receive()`/`send()`). Other ranks must participate in collective operations
   (like `dist.broadcast`) but don't directly interact with the FL system.

### Key Environment Variables

| Variable | Description |
|----------|-------------|
| `FOX_PARENT_URL` | CellNet URL for connecting to parent |
| `FOX_PARENT_FQCN` | Parent cell's FQCN for message routing |
| `FOX_SITE_NAME` | Client site name (e.g., `site-1`) |
| `FOX_WORKER_ID` | Worker ID within the site |
| `FOX_CLIENT_CLASS` | Client class name (determines execution mode) |

---

## Future Work: HPC, Multi-Node, and Kubernetes

The Client API pattern (`FoxClientAPI` with `receive()`/`send()`) is the foundation for all
advanced deployment scenarios. This unified approach ensures consistent client-side training
code across multi-GPU, HPC, multi-node, and Kubernetes environments.

### Unified Client API Strategy

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    Client API: One Pattern for All Environments                      │
│                                                                                      │
│   Environment          Launcher                   Client Training Code              │
│   ─────────────────────────────────────────────────────────────────────────────     │
│   Multi-GPU (local)    torchrun                   flare.receive() / flare.send()   │
│   HPC (SLURM)          srun + torchrun            flare.receive() / flare.send()   │
│   Multi-Node           mpirun / torchrun          flare.receive() / flare.send()   │
│   Kubernetes           Job/Pod                    flare.receive() / flare.send()   │
│                                                                                      │
│   → Same client.py works everywhere, only launcher configuration changes            │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### HPC Integration (SLURM, PBS, LSF)

**Goal**: Enable Fox clients to run on HPC clusters with job schedulers.

> **Note**: In real federated learning, the server is typically **external** to the HPC cluster
> (e.g., cloud VM, on-premise server at coordinating institution). The diagram below shows
> two scenarios: POC/simulation (server on login node) and production (external server).

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         HPC Architecture: Production                                 │
│                                                                                      │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                    EXTERNAL SERVER (Cloud/On-Premise)                        │   │
│   │                                                                              │   │
│   │    ┌───────────────────────────────────────────────────────────────────┐    │   │
│   │    │  FoxController                                                     │    │   │
│   │    │  • Orchestrates FL rounds                                          │    │   │
│   │    │  • Aggregates model updates                                        │    │   │
│   │    └───────────────────────────────────────────────────────────────────┘    │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                          │                                           │
│                                    Network/WAN                                       │
│                                          │                                           │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                         HPC CLUSTER (Site A)                                 │   │
│   │                                                                              │   │
│   │   Login Node                         Compute Nodes (GPU)                     │   │
│   │   ┌─────────────────┐               ┌────────────────────────────────────┐   │   │
│   │   │ Job Submission  │    SLURM      │  Allocated via sbatch/srun          │   │   │
│   │   │ (sbatch script) │    Job        │                                     │   │   │
│   │   │                 │──────────────▶│  FoxExecutor                        │   │   │
│   │   │                 │               │  └─ srun torchrun ... client.py     │   │   │
│   │   │                 │               │     (8 nodes × 8 GPUs = 64 GPUs)    │   │   │
│   │   └─────────────────┘               └────────────────────────────────────┘   │   │
│   │                                                                              │   │
│   │   Local Data: Never leaves the cluster                                       │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                      │
│   (Other HPC clusters at Site B, Site C connect similarly to the same server)       │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

**Key Components:**

| Component | Description |
|-----------|-------------|
| `HpcEnv` | New execution environment for HPC clusters |
| `SlurmLauncher` | Submits jobs via `sbatch`, monitors via `squeue` |
| `HpcWorkerAdapter` | Connects worker subprocess to parent across nodes |

**Configuration Example:**

```python
recipe = FoxRecipe(
    job_name="fedavg_hpc",
    server=FedAvg(),
    client=FoxClientAPI(),
    training_module="my_training.client",
    inprocess=False,
    # HPC-specific configuration
    run_cmd="srun torchrun --nproc_per_node=8 --nnodes=$SLURM_NNODES",
    hpc_config={
        "scheduler": "slurm",
        "partition": "gpu",
        "nodes_per_client": 2,
        "gpus_per_node": 8,
        "time_limit": "4:00:00",
    },
)
env = HpcEnv(num_clients=5)
recipe.execute(env)
```

**Implementation Considerations:**

1. **Cross-Node Communication**: Use CellNet with TCP transport for inter-node messaging
2. **Job Lifecycle**: Handle SLURM job states (PENDING, RUNNING, COMPLETED, FAILED)
3. **Resource Allocation**: Request GPUs, memory, and time limits per client
4. **Environment Propagation**: Pass `FOX_*` environment variables through SLURM

---

### Multi-Node Distributed Training

**Goal**: Support training where each client spans multiple nodes (e.g., 16 nodes × 8 GPUs = 128 GPUs per client).

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         Multi-Node Client Architecture                               │
│                                                                                      │
│   FoxExecutor (site-1)                                                              │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                      SubprocessLauncher                                      │   │
│   │                                                                              │   │
│   │   torchrun --nnodes=4 --nproc_per_node=8 --rdzv_backend=c10d                │   │
│   │            --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT                         │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                          │                                           │
│              ┌───────────────────────────┼───────────────────────────┐              │
│              │                           │                           │              │
│              ▼                           ▼                           ▼              │
│   ┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐      │
│   │   Node 1            │   │   Node 2            │   │   Node 3 & 4        │      │
│   │   Rank 0-7          │   │   Rank 8-15         │   │   Rank 16-31        │      │
│   │                     │   │                     │   │                     │      │
│   │   Rank 0: FoxWorker │   │   (DDP workers)     │   │   (DDP workers)     │      │
│   │   ↔ Parent CellNet  │   │                     │   │                     │      │
│   └─────────────────────┘   └─────────────────────┘   └─────────────────────┘      │
│                                                                                      │
│   Only Rank 0 communicates with server; all ranks participate in DDP collectives    │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

**Key Features:**

1. **Rendezvous Coordination**: Use PyTorch's elastic launch (`torchrun`) with `c10d` backend
2. **Single Communication Point**: Only global rank 0 connects to FoxExecutor
3. **Broadcast Weights**: Rank 0 broadcasts received model to all nodes via `dist.broadcast`
4. **Aggregate Gradients**: DDP handles gradient synchronization across all nodes

**Client Training Pattern:**

```python
# client.py - works for single-node and multi-node
from nvflare.fox.sys.worker import get_client_api
import torch.distributed as dist

flare = get_client_api()
flare.init()

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

while True:
    if rank == 0:
        if not flare.is_running():
            dist.broadcast(torch.tensor([1]), src=0)  # Signal stop
            break
        input_model = flare.receive()
        dist.broadcast(torch.tensor([0]), src=0)  # Signal continue
    else:
        signal = torch.tensor([0])
        dist.broadcast(signal, src=0)
        if signal.item() == 1:
            break
    
    # Broadcast weights to all ranks
    weights = broadcast_state_dict(input_model.params if rank == 0 else None, src=0)
    
    # All ranks train with DDP
    new_weights, loss = train_ddp(weights, rank, world_size)
    
    if rank == 0:
        flare.send(FLModel(params=new_weights, metrics={"loss": loss}))

dist.destroy_process_group()
```

---

### Kubernetes Native Deployment

**Goal**: Deploy Fox clients as Kubernetes-native workloads. Server and clients are
**distributed across different locations** - this is the essence of federated learning.

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    Federated Learning: Distributed Architecture                      │
│                                                                                      │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                         SERVER (Central Aggregator)                          │   │
│   │                                                                              │   │
│   │   Options:                                                                   │   │
│   │   • Cloud VM (AWS/GCP/Azure)                                                 │   │
│   │   • On-premise server                                                        │   │
│   │   • K8s pod in separate cluster                                              │   │
│   │                                                                              │   │
│   │   ┌─────────────────────────────────────────────────────────────────────┐   │   │
│   │   │  FoxController                                                       │   │   │
│   │   │  • Orchestrates FL rounds                                            │   │   │
│   │   │  • Aggregates client updates                                         │   │   │
│   │   │  • Exposes endpoint for client connections                           │   │   │
│   │   └─────────────────────────────────────────────────────────────────────┘   │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                          │                                           │
│                                    Internet/WAN                                      │
│                                          │                                           │
│        ┌─────────────────────────────────┼─────────────────────────────────┐        │
│        │                                 │                                 │        │
│        ▼                                 ▼                                 ▼        │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐       │
│  │  Site A (Hospital)   │  │  Site B (Research)   │  │  Site C (Enterprise) │       │
│  │  K8s Cluster         │  │  K8s Cluster         │  │  K8s Cluster         │       │
│  │                      │  │                      │  │                      │       │
│  │  ┌────────────────┐  │  │  ┌────────────────┐  │  │  ┌────────────────┐  │       │
│  │  │ Fox Operator   │  │  │  │ Fox Operator   │  │  │  │ Fox Operator   │  │       │
│  │  └───────┬────────┘  │  │  └───────┬────────┘  │  │  └───────┬────────┘  │       │
│  │          │           │  │          │           │  │          │           │       │
│  │          ▼           │  │          ▼           │  │          ▼           │       │
│  │  ┌────────────────┐  │  │  ┌────────────────┐  │  │  ┌────────────────┐  │       │
│  │  │ Client Pod     │  │  │  │ Client Pod     │  │  │  │ Client Pod     │  │       │
│  │  │                │  │  │  │                │  │  │  │                │  │       │
│  │  │ FoxExecutor    │  │  │  │ FoxExecutor    │  │  │  │ FoxExecutor    │  │       │
│  │  │ └─ FoxWorker   │  │  │  │ └─ FoxWorker   │  │  │  │ └─ FoxWorker   │  │       │
│  │  │    (8 GPUs)    │  │  │  │    (4 GPUs)    │  │  │  │    (16 GPUs)   │  │       │
│  │  │                │  │  │  │                │  │  │  │                │  │       │
│  │  │ Local Data     │  │  │  │ Local Data     │  │  │  │ Local Data     │  │       │
│  │  │ (never leaves) │  │  │  │ (never leaves) │  │  │  │ (never leaves) │  │       │
│  │  └────────────────┘  │  │  └────────────────┘  │  │  └────────────────┘  │       │
│  │                      │  │                      │  │                      │       │
│  │  PVC: checkpoints    │  │  PVC: checkpoints    │  │  PVC: checkpoints    │       │
│  └──────────────────────┘  └──────────────────────┘  └──────────────────────┘       │
│                                                                                      │
│   Key: Data stays local, only model updates traverse the network                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

**Server Deployment Options:**

| Option | Description | Use Case |
|--------|-------------|----------|
| Cloud VM | Simple EC2/GCE instance | Quick setup, development |
| On-premise | Dedicated server | Enterprise, security requirements |
| K8s Pod | Separate K8s cluster | Cloud-native, managed |
| Managed Service | NVIDIA hosted | Production, SaaS model |

**Client-Side CRD (FoxClient):**

```yaml
# Deployed in EACH client's K8s cluster
apiVersion: nvflare.nvidia.com/v1
kind: FoxClient
metadata:
  name: site-a-client
  namespace: federated-learning
spec:
  # Server connection (external)
  server:
    endpoint: "grpcs://fl-server.example.com:8443"
    certificate: server-cert-secret
  
  # Client identity
  siteName: "hospital-a"
  clientCert: client-cert-secret
  
  # Training configuration
  training:
    image: nvcr.io/nvidia/nvflare:latest
    trainingModule: my_training.client
    launcher: "torchrun --nproc_per_node=8"
    resources:
      cpu: "32"
      memory: "128Gi"
      nvidia.com/gpu: "8"
  
  # Local storage (data never leaves)
  storage:
    dataPVC: local-training-data
    checkpointPVC: model-checkpoints
```

**Key Components:**

| Component | Location | Description |
|-----------|----------|-------------|
| `FoxController` | Server (any) | Central aggregation, orchestration |
| `FoxOperator` | Each client K8s | Manages client pod lifecycle |
| `FoxClient CRD` | Each client K8s | Defines client configuration |
| `FoxExecutor` | Client pod | Runs training, communicates with server |

**Implementation Phases:**

1. **Phase 1: Helm Charts**
   - Server Helm chart (deploy anywhere)
   - Client Helm chart (deploy in client K8s)
   - Manual certificate management

2. **Phase 2: Client Operator**
   - `FoxClient` CRD for each site
   - Automatic pod lifecycle management
   - Health monitoring and restart

3. **Phase 3: Security & Connectivity**
   - mTLS between server and clients
   - Certificate rotation
   - Network policies for client isolation

4. **Phase 4: Multi-Cluster Management**
   - Central dashboard for all clients
   - Client registration workflow
   - Cross-site observability

---

### Shared Infrastructure Components

These components are common across HPC, multi-node, and K8s environments:

#### 1. Elastic Client Management

```python
class ElasticClientManager:
    """Handle dynamic client join/leave during training."""
    
    def on_client_join(self, client_id: str):
        """New client joined mid-training."""
        pass
    
    def on_client_leave(self, client_id: str):
        """Client left (crashed or preempted)."""
        pass
    
    def redistribute_work(self):
        """Rebalance work among remaining clients."""
        pass
```

#### 2. Checkpoint/Restart

```python
class CheckpointManager:
    """Save and restore training state for fault tolerance."""
    
    def save_checkpoint(self, round_num: int, global_model: dict, client_states: dict):
        """Save checkpoint to persistent storage (S3, PVC, HDFS)."""
        pass
    
    def restore_checkpoint(self) -> Tuple[int, dict, dict]:
        """Restore from last checkpoint after failure."""
        pass
```

#### 3. Object Storage Integration

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    Large Model Transfer via Object Storage                           │
│                                                                                      │
│   Server                    S3/MinIO/GCS                    Clients                 │
│   ┌──────────────┐         ┌──────────────┐         ┌──────────────────────────┐   │
│   │ Upload model │────────▶│ s3://bucket/ │◀────────│ Download model           │   │
│   │              │         │ round-5.pt   │         │                          │   │
│   │ Send ref     │─────────────────────────────────▶│ Receive ref, fetch model │   │
│   └──────────────┘         └──────────────┘         └──────────────────────────┘   │
│                                                                                      │
│   Benefits: Reduced CellNet load, parallel downloads, resumable transfers           │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

#### 4. Observability Stack

| Component | Purpose |
|-----------|---------|
| Prometheus | Metrics collection (GPU utilization, training loss, round times) |
| Grafana | Dashboards and alerting |
| Jaeger/Zipkin | Distributed tracing for debugging |
| ELK/Loki | Log aggregation across all pods/nodes |

---

### Implementation Checklist

**Client API Enhancements:**
- [x] Subprocess mode with queue-based bridging (Done)
- [ ] In-process mode with direct variable adapter (no queues/threading)
- [ ] `set_training_func()` for registering training callback
- [ ] Mode auto-detection based on `inprocess` flag

**HPC Support:**
- [ ] `HpcEnv` execution environment
- [ ] `SlurmLauncher` for SLURM clusters
- [ ] PBS and LSF launcher support
- [ ] Cross-node CellNet communication
- [ ] Environment variable propagation through job schedulers

**Multi-Node DDP:**
- [ ] PyTorch elastic launch integration
- [ ] Rendezvous coordination (`c10d` backend)
- [ ] Multi-node weight broadcasting utilities
- [ ] NCCL backend optimization

**Kubernetes:**
- [ ] Helm charts for manual deployment
- [ ] `FoxJob` Custom Resource Definition
- [ ] `FoxOperator` Kubernetes controller
- [ ] StatefulSet for client pods
- [ ] Service discovery and DNS
- [ ] PersistentVolumeClaim integration

**Fault Tolerance:**
- [ ] `CheckpointManager` for state persistence
- [ ] Automatic restart from checkpoint
- [ ] `ElasticClientManager` for dynamic join/leave
- [ ] Pod disruption budget support

**Infrastructure:**
- [ ] Object storage integration (S3/MinIO/GCS)
- [ ] Prometheus metrics exporter
- [ ] Grafana dashboard templates
- [ ] Distributed tracing (Jaeger/Zipkin)

---

## Future Enhancement: Flexible Client-Server Contract

### Current Limitation

The Client API currently uses `FLModel` as the fixed contract between server and client:

```python
# Server sends FLModel
input_model = FLModel(params=weights, current_round=round)
results = fox.clients.execute(fl_model=input_model)

# Client receives/sends FLModel
input_model = flare.receive()  # Returns FLModel
flare.send(output_model)       # Expects FLModel
```

This design enables:
- **Independent development**: Server and client code evolve separately
- **Standardization**: All FL algorithms use the same data structure
- **Tooling**: Metrics, logging, and debugging work consistently

However, this becomes limiting in advanced scenarios:

| Scenario | Limitation |
|----------|------------|
| **Personalized + Global models** | Need to send/receive multiple models |
| **Hierarchical FL** | Different model types at different levels |
| **Split learning** | Activations/gradients instead of weights |
| **Multi-task learning** | Multiple task-specific heads |
| **Auxiliary data** | Additional tensors (embeddings, prototypes) |

### Proposed Solutions

#### Option 1: Extended FLModel with Flexible Payload

Extend `FLModel` to carry additional named payloads without breaking existing code:

```python
class FLModel:
    params: dict              # Primary model weights (existing)
    metrics: dict             # Metrics (existing)
    current_round: int        # Round number (existing)
    
    # New: flexible additional payloads
    extra: dict[str, Any]     # Named additional data
```

**Usage:**

```python
# Server
input_model = FLModel(
    params=global_weights,
    extra={
        "personal_weights": personal_weights,
        "prototype": class_prototypes,
    }
)

# Client
input_model = flare.receive()
global_weights = input_model.params
personal_weights = input_model.extra.get("personal_weights")
```

**Pros:** Backward compatible, minimal API change
**Cons:** Untyped `extra` dict, no schema validation

---

#### Option 2: Generic Type Parameter

Make the Client API generic over the message type:

```python
class FoxClientAPI(Generic[T]):
    def receive(self) -> T: ...
    def send(self, model: T) -> None: ...

# Standard usage (backward compatible)
flare: FoxClientAPI[FLModel] = get_client_api()

# Advanced usage
@dataclass
class PersonalizedModel:
    global_params: dict
    personal_params: dict
    prototypes: torch.Tensor

flare: FoxClientAPI[PersonalizedModel] = get_client_api()
```

**Pros:** Type-safe, IDE support, custom schemas
**Cons:** Requires explicit type annotation, more complex

---

#### Option 3: Multi-Channel Communication

Support multiple named channels for different data types:

```python
# Server
fox.clients.execute(
    global_model=FLModel(params=global_weights),
    personal_model=FLModel(params=personal_weights),
    prototypes=prototype_tensor,
)

# Client
global_model = flare.receive("global_model")
personal_model = flare.receive("personal_model")
prototypes = flare.receive("prototypes")

# ... training ...

flare.send("global_model", updated_global)
flare.send("personal_model", updated_personal)
```

**Pros:** Clear separation, parallel transfers possible
**Cons:** More complex API, ordering considerations

---

#### Option 4: Streaming/Chunked Transfer

For very large models, support streaming:

```python
# Server
async for chunk in fox.clients.execute_stream(large_model):
    process_chunk(chunk)

# Client
async for chunk in flare.receive_stream():
    accumulate(chunk)
# ... training ...
async for chunk in generate_chunks(result):
    await flare.send_chunk(chunk)
```

**Pros:** Memory efficient, handles arbitrarily large models
**Cons:** Significant complexity, async required

---

### Recommended Approach

**Phase 1: Extend FLModel (Option 1)**
- Add `extra: dict` field to `FLModel`
- Zero breaking changes
- Immediate benefit for personalization use cases

**Phase 2: Multi-Channel (Option 3)**
- Implement for cases where `extra` dict is insufficient
- Enables parallel transfer of independent data
- Better semantics for split learning

```python
# Backward compatible: single channel (default)
input_model = flare.receive()  # Returns FLModel from default channel

# New: named channels
global_model = flare.receive(channel="global")
personal_model = flare.receive(channel="personal")
```

### Example: Personalized Federated Learning

```python
# server.py
@fox.algo
def personalized_fedavg(self):
    global_weights = self.global_model.state_dict()
    
    for round in range(self.num_rounds):
        input_model = FLModel(
            params=global_weights,
            current_round=round,
            extra={"lr_schedule": self.get_lr(round)}
        )
        
        results = fox.clients.execute(fl_model=input_model)
        
        # Aggregate only global weights
        global_updates = [r.params for r in results]
        global_weights = weighted_avg(global_updates)
        
        # Personal weights stay on clients (not aggregated)
    
    fox.clients.stop()

# client.py
flare = get_client_api()
flare.init()

# Initialize personal model locally
personal_model = PersonalHead()

while flare.is_running():
    input_model = flare.receive()
    if input_model is None:
        break
    
    global_weights = input_model.params
    lr = input_model.extra.get("lr_schedule", 0.01)
    
    # Train both models
    new_global, new_personal = train_personalized(
        global_weights, 
        personal_model.state_dict(),
        lr=lr
    )
    
    # Only send global weights back
    output_model = FLModel(
        params=new_global,
        metrics={"loss": loss, "personal_acc": personal_acc}
    )
    flare.send(output_model)
    
    # Keep personal weights locally
    personal_model.load_state_dict(new_personal)

# Save personal model for this client
torch.save(personal_model.state_dict(), f"personal_{site_name}.pt")
```

---

## Additional Future Considerations

1. **S3-API Model Transfer**: Offload large model transfers to object storage

