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

### Example User Code

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

---

## Layer 2: API Layer

### Core Components

#### FoxRecipe (`nvflare/fox/sys/recipe.py`)

Job configuration and orchestration entry point.

```python
recipe = FoxRecipe(
    job_name="fedavg",
    min_clients=5,
    server=server_module,      # Module with @fox.algo
    client=client_module,      # Module with @fox.collab
    inprocess=True,            # or False for subprocess
    subprocess_launcher="torchrun --nproc_per_node=4",
    tracking_type="tensorboard",
)
```

**Key Responsibilities:**
- Auto-detect and wrap modules with `ModuleWrapper`
- Configure execution mode (in-process vs subprocess)
- Set up tracking configuration
- Generate FLARE job configuration

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

Multi-process local execution using POC infrastructure.

```
┌─────────────────────────────────────────────────────────────────┐
│                          PocEnv                                  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    POC Infrastructure                     │   │
│  │                                                           │   │
│  │  Process-1: Server (FoxController)                        │   │
│  │  Process-2: site-1 (FoxExecutor)                          │   │
│  │  Process-3: site-2 (FoxExecutor)                          │   │
│  │  ...                                                      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Backend: FlareBackend (CellNet communication)                   │
└─────────────────────────────────────────────────────────────────┘
```

**Characteristics:**
- Multiple processes on single machine
- Real FLARE runtime components
- CellNet for inter-process communication
- Validates production behavior locally

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

```
┌─────────────────────────────────────────────────────────────────┐
│                       FlareBackend                               │
│                                                                  │
│  Server Process                   Client Processes               │
│  ┌──────────────┐                ┌──────────────┐               │
│  │ FoxController│   CellNet      │ FoxExecutor  │               │
│  │              │───(fox/call)──▶│  site-1      │               │
│  │              │◀──response──── │              │               │
│  └──────────────┘                └──────────────┘               │
│                                  ┌──────────────┐               │
│                                  │ FoxExecutor  │               │
│                                  │  site-2      │               │
│                                  └──────────────┘               │
│                                                                  │
│  Communication: CellNet channels, FOBS serialization             │
└─────────────────────────────────────────────────────────────────┘
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

Runs in the subprocess, connects back to parent.

```python
# Automatically launched by SubprocessLauncher
# Reads connection info from environment variables:
#   FOX_PARENT_URL, FOX_PARENT_FQCN, FOX_SITE_NAME, FOX_WORKER_ID
```

**Responsibilities:**
- Connect to parent via CellNet
- Load user's training module
- Execute @fox.collab methods
- Handle DDP rank coordination (only rank 0 communicates)

### Subprocess Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              FoxExecutor (Parent Process)                            │
│                                                                                      │
│  ┌────────────────────────────────────────────────────────────────────────────────┐ │
│  │                          SubprocessLauncher                                     │ │
│  │                                                                                 │ │
│  │  1. Set ENV vars: FOX_PARENT_URL, FOX_PARENT_FQCN, etc.                        │ │
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
│  │  2. Load training module                                                       │  │
│  │  3. Signal ready                                                               │  │
│  │  4. Execute @fox.collab methods on call                                        │  │
│  │  5. Return results to parent                                                   │  │
│  │                                                                                │  │
│  │  ┌─────────────────────────────────────────────────────────────────────────┐  │  │
│  │  │  User's Training Code                                                    │  │  │
│  │  │                                                                          │  │  │
│  │  │  @fox.collab                                                             │  │  │
│  │  │  def train(weights):                                                     │  │  │
│  │  │      # DDP training with all ranks                                       │  │  │
│  │  │      dist.init_process_group(...)                                        │  │  │
│  │  │      model = DDP(model)                                                  │  │  │
│  │  │      ...                                                                 │  │  │
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
                            ├── subprocess_launcher: "torchrun ..."
                            └── training_module: "my_training"
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
    subprocess_launcher="torchrun --nproc_per_node=4",
    training_module="my_training",
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
# Deploy via FLARE admin or Helm charts
```

- Multiple machines
- Full security features
- Production monitoring

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
        └── collab_api/
            ├── in_process/
            │   ├── collab_fedavg_train.py
            │   ├── collab_fedavg_no_class.py
            │   └── simulate_fedavg_train.py
            └── sub_process/
                └── distributed_train.py
```

---

## Future Considerations

1. **Kubernetes Native**: Integration with K8s operators for job pods
2. **S3-API Model Transfer**: Offload large model transfers to object storage
3. **Enhanced Observability**: Prometheus metrics, distributed tracing
4. **Fault Tolerance**: Checkpoint/restart for long-running jobs
5. **Resource Scheduling**: GPU-aware scheduling and allocation

