# FLARE Collab API -- Internal Design Specification

*Target audience: NVIDIA FLARE engineers who will design and implement the Collab API.*

*For the user-facing API specification, see [collab_api_spec.md](collab_api_spec.md).*

## 1. Scope

This document covers internal architecture, bridge mechanics, execution environment internals, development requirements, and open design questions for the Collab API. It assumes familiarity with the user-facing spec and existing FLARE internals (Controller/Executor APIs, CellNet, ModelController, Job/Recipe infrastructure).

---

## 2. API Layering

NVFlare has three levels of server-side API:

```
High-level:    Collab API  (@collab.main, @collab.publish)   <-- promoted API for new development
Mid-level:     ModelController API                            <-- no longer promoted once Collab API ships
Low-level:     Controller API + Executor API                  <-- internal plumbing
```

The Collab API is **built on top of** the Controller and Executor APIs. It is not a separate system -- it is a higher-level abstraction over the same underlying infrastructure.

Today's standard recipes (`FedAvgRecipe`, `CyclicRecipe`, etc.) were built using the **ModelController API** (mid-level). Going forward, **new FL algorithms and recipes should be developed using the Collab API** (high-level).

### Class Hierarchy

```
Recipe (base class -- nvflare.recipe.spec)
  |
  |-- CollabRecipe (nvflare.collab.sys.recipe)
  |     Server: user-written @collab.main method  [Collab API -- high-level, promoted]
  |     Client: @collab.publish methods or Client API train_script
  |
  |-- FedAvgRecipe (nvflare.app_opt.pt.recipes.fedavg)
  |     Server: built-in scatter-and-gather  [ModelController API -- mid-level, legacy]
  |     Client: Client API train_script
  |
  |-- CyclicRecipe, SimpleSwarmLearningRecipe, ...
  |-- KMeansFedAvgRecipe, SVMFedAvgRecipe, ... (sklearn)
  |-- NumpyFedAvgRecipe, FedAvgLrRecipe, ... (numpy)
```

---

## 3. CollabClientAPI Bridge Architecture

The Collab server uses a **push** model (`collab.clients.execute(fl_model)`) while the Client API uses a **pull** model (`flare.receive()`). The `CollabClientAPI` bridge connects the two.

### Bridge Internals

```
Server (@collab.main)                       Client (Client API)
========================                    ========================

collab.clients.execute(fl_model)            flare.receive()
         |                                       ^
         v                                       |
  +-------------------------------------------------+
  |              CollabClientAPI (bridge)             |
  |                                                   |
  |  execute() is @collab.publish:                    |
  |    1. puts fl_model in _call_queue                |
  |    2. blocks on _result_queue                     |
  |                                                   |
  |  receive():                                       |
  |    gets fl_model from _call_queue                 |
  |                                                   |
  |  send(result):                                    |
  |    puts result in _result_queue                   |
  |    -> unblocks execute() -> returns to server     |
  |                                                   |
  |  stop():                                          |
  |    sets _stopped=True                             |
  |    -> is_running() returns False                  |
  +-------------------------------------------------+
         |                                       |
         v                                       v
collab.clients.stop()                       flare.is_running() -> False
```

### Two Execution Modes

| Mode | When | Mechanism |
|------|------|-----------|
| **In-process** (`launch_external_process=False`) | Single GPU, `SimEnv` | `CollabRecipe` sets `train_script` on `CollabClientAPI`. On first `execute()` call, the script is launched in a **background thread**. The script calls `flare.receive()` / `flare.send()` which route to the `CollabClientAPI` instance via **`contextvars`**. Queue-based handshake between `execute()` and `receive()`/`send()`. |
| **Subprocess** (`launch_external_process=True`) | Multi-GPU (torchrun DDP) | `SubprocessLauncher` spawns the training script as a child process (e.g. via `torchrun`). Communication between the `CollabClientAPI` (in the parent FLARE process) and the client script (in the subprocess) uses **CellNet channels**. Only rank 0 communicates with the server; other ranks sync via `dist.broadcast`. |

### Wiring Sequence

When `train_script="client.py"` is specified in `CollabRecipe`:

1. `CollabRecipe` creates a `CollabClientAPI()` instance.
2. It calls `client_api.set_train_script("client.py", ...)` to register the script path.
3. The `CollabClientAPI` is passed to the executor as the client object.
4. When the server calls `collab.clients.execute(fl_model=...)`, this invokes `CollabClientAPI.execute()` (a `@collab.publish` method).
5. `CollabClientAPI.execute()` starts the training script (first call only) and bridges `fl_model` to `flare.receive()` via queues.
6. The training script's `flare.send(result)` puts the result back in the queue, which `execute()` returns to the server.

---

## 4. Execution Environment Internals

### 4.1 Requirements

1. **Backward compatible:** Existing recipes (no Collab API) must continue to work exactly as before.
2. **Collab Backend instantiation:** When the Collab API is involved, the environment must instantiate a **Collab Backend** to handle `@collab.main` / `@collab.publish` dispatch.
3. **Explicit, not automatic:** There is **no auto-matching or auto-detection magic**. The use of Collab API must be **explicitly specified** by the recipe (not inferred by scanning code for annotations). The recipe carries the information about which API it uses. The user writes the same `SimEnv(...)` / `PocEnv(...)` for all recipes -- the environment consults the recipe to determine which backend to instantiate.

The mechanism by which the recipe indicates its API type (e.g. a flag, a base class check, metadata, or another mechanism) is an **open question** (OQ-1). What is **not** acceptable: any form of implicit detection or code scanning.

### 4.2 Backend Selection

The user writes a single `SimEnv` / `PocEnv`. Behind the scenes:

| Environment | Standard Recipe | Collab Recipe |
|------------|-----------------|---------------|
| **`SimEnv`** | Full FLARE simulator (multi-threaded, job submission, task dispatch, component lifecycle) | **Collab simulation backend**: pure function calls within the same process. `@collab.main` calls `@collab.publish` directly. No simulator overhead. |
| **`PocEnv`** | Existing FLARE multi-process (CellNet) | Same `PocEnv` + **Collab FLARE Backend**: dispatches `@collab.main`/`@collab.publish` over CellNet |
| **`ProdEnv`** | Existing FLARE deployment | Same `ProdEnv` + **Collab FLARE Backend** |

There is **no separate `CollabSimEnv`/`CollabPocEnv`/`CollabProdEnv`**. The Collab backend is an internal implementation detail.

### 4.3 SimEnv Backend Comparison

| | Standard Backend | Collab Backend |
|---|---|---|
| **Execution model** | Full FLARE simulator with job submission, task dispatch, component lifecycle | Pure function calls: `@collab.main` calls `@collab.publish` directly |
| **Speed** | Slower (simulator overhead) | Much faster (no simulator infrastructure) |
| **Fidelity** | High: exercises full FLARE job pipeline | Lower: no job submission, no CellNet, no component lifecycle |
| **Use case** | Testing standard recipes, integration testing | Rapid algorithm prototyping, unit testing custom logic |
| **Multi-GPU** | Supported via subprocess launcher | Supported via `launch_external_process=True` |

---

## 5. Recipe Packaging (Open Design)

### The Problem

Today, creating a new named recipe from Collab API server logic + Client API client requires using the Job API to build every piece from scratch (similar to how `CollabRecipe` itself was built). Researchers need a **packaging tool or base class** that makes it easy to turn their `@collab.main` server + `train_script` client into a reusable recipe.

### Progression Model

```
Phase 1 (Research):    CollabRecipe + @collab.main + @collab.publish  (SimEnv, pure function calls)
Phase 2 (Hybrid):      @collab.main server + Client API client        (wiring mechanism TBD -- OQ-3)
Phase 3 (New Recipe):  Job API packages server + client -> SplitLearningRecipe(...)  (end-user-facing)
```

**Open question (OQ-2):** What does the recipe packaging utility look like? (base class, builder, code generator, or other)

**Open question (OQ-3):** How are Collab server + Client API client wired in Phase 2? (`CollabRecipe` with `train_script`, per-algorithm recipe, or new mechanism)

---

## 6. Decorator and Runtime Infrastructure

### Components to Implement

- **`@collab.main`** -- marks the server-side orchestration method. Called once per job.
- **`@collab.publish`** -- marks client-side methods for remote invocation by the server.
- **`collab.clients`** -- `ProxyList` that dispatches method calls to all clients in parallel and collects results.
- **`collab.site_name`** -- per-client site name, injected by the framework.
- **Thread isolation** -- `contextvars`-based isolation so multiple clients can run concurrently in threaded simulation.
- **Error handling** -- client failure detection, timeout management, partial result handling.

---

## 7. Development Requirements

| # | Requirement | Description | Dependencies | Priority |
|---|-------------|-------------|--------------|----------|
| 1 | **SimEnv Collab Backend (Simulation)** | Extend `SimEnv` with a Collab simulation backend that dispatches `@collab.main` and `@collab.publish` calls as pure function calls within the same process (no old simulator overhead). Handles `collab.clients` proxy, `collab.site_name` injection, and result aggregation. Must support all Collab API usage patterns: single file, multiple files, with classes, without classes (standalone functions). From the user's perspective there is only one `SimEnv` -- the Collab backend is an internal implementation detail selected based on the recipe. | -- | P0 |
| 2 | **Collab FLARE Backend (POC / Production)** | Implement the Collab FLARE Backend that enables `PocEnv`/`ProdEnv` to support Collab API. Dispatches `@collab.main` and `@collab.publish` calls over CellNet (multi-process / multi-node). Must support all Collab API usage patterns. Must integrate with existing FLARE communication infrastructure. There is no separate `CollabPocEnv`/`CollabProdEnv` -- the Collab FLARE Backend is what makes `PocEnv`/`ProdEnv` Collab-aware. | 8 | P0 |
| 3 | **CollabClientAPI bridge (Simulation)** | In `SimEnv` (Collab backend), support Collab API on the server side with Client API on the client side, connected via the `CollabClientAPI` bridge. Must support both in-process (single GPU) and subprocess (`launch_external_process=True` for DDP) modes. | 1 | P0 |
| 4 | **CollabClientAPI bridge (POC / Production)** | In `PocEnv`/`ProdEnv`, support Collab API on the server side with Client API on the client side, connected via the `CollabClientAPI` bridge over CellNet. Must support subprocess mode for multi-GPU / multi-node DDP. | 2, 3 | P0 |
| 5 | **Recipe packaging utility** | Create a utility, base class, or tooling that makes it easy to package Collab API server logic + Client API client into a new named recipe via the Job API. Design is an open question (OQ-2). | 3, 4 | P1 |
| 6 | **CollabRecipe** | Create `CollabRecipe` that works in both simulation (`SimEnv`) and POC/Production environments. Should support all usage patterns: Collab API on both sides, and Collab API server + Client API client (via `train_script`). Initial simulation support requires #1 only; full POC/Production support requires #2, #3, #4. | 1 (initial), 2, 3, 4 (full) | P0 |
| 7 | **ExecEnv enhancement** | Enhance `ExecEnv` to support Collab API while remaining backward compatible. Requires an explicit mechanism for the recipe/job to indicate API type (not code scanning). Design is an open question (OQ-1). | 1, 2 | P0 |
| 8 | **Decorator and runtime infrastructure** | Implement `@collab.main`, `@collab.publish` decorators and the runtime infrastructure: `collab.clients` (ProxyList), `collab.site_name`, `contextvars`-based thread isolation, error handling for client failures. | -- | P0 |
| 9 | **Multi-GPU / DDP support** | Support `launch_external_process=True` with `torchrun` for multi-GPU DDP training. Subprocess launcher spawns the training script; rank 0 communicates with server via `CollabClientAPI`; other ranks sync via `dist.broadcast`. | 3, 4 | P1 |

### Suggested Implementation Order

```
Phase 1 (Foundation):   #8 (decorators/runtime) + #1 (SimEnv Collab Backend)
Phase 2 (Research):     #6 (CollabRecipe, initial sim-only) + #7 (ExecEnv enhancement)
Phase 3 (Bridge):       #3 (CollabClientAPI in Sim) + #9 (multi-GPU/DDP) + #6 (CollabRecipe + Client API bridge)
Phase 4 (Production):   #2 (Collab FLARE Backend for Poc/Prod) + #4 (CollabClientAPI in Poc/Prod) + #6 (CollabRecipe full)
Phase 5 (Packaging):    #5 (Recipe packaging utility)
```

---

## 8. Open Design Questions

| # | Question | Context |
|---|----------|---------|
| OQ-1 | **How does the recipe indicate its API type to the environment?** Flag, base class check, metadata, or other mechanism. No code scanning or auto-detection. | Section 4.1 |
| OQ-2 | **What does the recipe packaging utility look like?** Base class, builder, code generator, or other. | Section 5 |
| OQ-3 | **How are Collab server + Client API client wired in Phase 2 (hybrid)?** `CollabRecipe` with `train_script`, per-algorithm recipe, or new mechanism. | Section 5 |

---

## 9. What Each Recipe Controls (Reference)

| Layer | CollabRecipe | Standard Recipes (FedAvg, Cyclic, etc.) |
|-------|-------------|-------------------------------------|
| **Server algorithm** | User-written `@collab.main` -- full control | Built-in workflow -- user only sets parameters |
| **Client training** | `@collab.publish` methods **or** `train_script` (Client API) | `train_script` only (Client API) |
| **Aggregation** | User code inside `@collab.main` | Built-in aggregator (e.g. `InTimeAccumulateWeightedAggregator`) |
| **Model handling** | User manages weights directly | Recipe manages persistence (e.g. `PTFileModelPersistor`) |
| **Multi-GPU** | `launch_external_process=True` + `command="torchrun ..."` | Built-in launcher configuration |
| **Execution envs** | `SimEnv`, `PocEnv`, `export()` | `SimEnv`, `PocEnv`, `export()` |

### Shared Capabilities (from Base Recipe)

| Capability | Method |
|-----------|--------|
| Execute locally | `recipe.execute(env)` |
| Export for production | `recipe.export(path)` |
| Server filters | `recipe.add_server_output_filter(...)` |
| Client filters | `recipe.add_client_input_filter(...)` |
| Config | `recipe.add_server_config(...)` |
| Decomposers | `recipe.add_decomposers(...)` |
| Environment processing | `recipe.process_env(env)` |
