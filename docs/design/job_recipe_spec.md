# Recipe API  -  User Guide & Interface Design

This document is the **API interface initial design** for the NVFlare Recipe API. It merges the Recipe User Guide (see RST: `user_guide/data_scientist_guide/job_recipe.rst`, `available_recipes.rst`) with the public API spec into one reference. The RST files are unchanged.

A **recipe** defines *what* to run (e.g. FedAvg, Cyclic); an **execution environment** defines *where* (simulation, proof-of-concept, or production). You create a recipe with high-level parameters, then either **export** it to a job directory for submission or **execute** it in an environment. The same recipe runs everywhere without code changes.

---

## 1. Why Job Recipes

The Job API lets you define FL workflows in Python without editing config files, but it still requires understanding controllers, executors, and workflows. **Job Recipes** simplify this:

- **Only the arguments that matter:** e.g. `min_clients`, `num_rounds`, `train_script`, model.
- **Consistent entry points** for FedAvg, Cyclic, FedOpt, and other patterns.
- **One recipe, many environments:** SimEnv (threads), PocEnv (processes on one machine), ProdEnv (real deployment).

Start with a recipe; move to the full Job API when you need to customize. Recipe support is a technical preview; not all algorithms have recipes yet.

---

## 2. Concepts

| Concept | Description |
|--------|--------------|
| **Recipe** | A federated job definition (plus optional filters, config overrides, decomposers). You use a concrete recipe (e.g. `FedAvgRecipe`) and call `export()` or `execute()`. No internal job type required. |
| **ExecEnv** | Where the job runs. NVFlare provides **SimEnv**, **PocEnv**, and **ProdEnv**; each deploys the job and returns status/results via a `Run` handle. |
| **Run** | Returned by `recipe.execute(env)`. Use it for job ID, status, result path, and abort. |

---

## 3. Recipe API

**Module:** `nvflare.recipe.spec.Recipe`. You use concrete recipes (e.g. `FedAvgRecipe`) or subclass `Recipe` for custom jobs.

### 3.1 Constructor

Concrete recipes take high-level parameters and build the job internally:

```python
recipe = FedAvgRecipe(
    name="my_fedavg",
    min_clients=2,
    num_rounds=5,
    train_script="train.py",
    train_args="--epochs 2",
    model=my_model,
)
```

Optional **`per_site_config`** lets you override the training script and/or arguments (e.g. learning rate, batch size) per site; see **3.4 Per-site config** below.


### 3.2 Export and execute

**Export** writes the job to disk for later submission (e.g. via Job CLI):

```python
recipe.export(job_dir="/path/to/job_root")
# Optional: server_exec_params, client_exec_params, env for process_env()
```

**Execute** runs the job in a given environment and returns a `Run`:

```python
run = recipe.execute(env=env, server_exec_params=None, client_exec_params=None)
run.get_job_id()
run.get_status()
result_path = run.get_result(timeout=3600.0)
run.abort()  # if needed
```

### 3.3 Other recipe methods

All methods below are on the base `Recipe` class and work with any concrete recipe.

#### process_env(env)

Override to apply environment-specific configuration before deploy. Called automatically by `export(..., env=...)` and `execute(env)`. Default is a no-op; concrete recipes can override to adjust settings based on the target environment (e.g. different timeouts for simulation vs production).

#### Filters

Add task-data or task-result filters to server or clients. Filters transform data in transit (e.g. precision conversion, parameter exclusion, differential privacy).

| Method | Direction | Scope |
|--------|-----------|-------|
| `add_server_output_filter(filter, tasks=None)` | Server → clients (task data) | Server |
| `add_server_input_filter(filter, tasks=None)` | Clients → server (task result) | Server |
| `add_client_input_filter(filter, tasks=None, clients=None)` | Server → client (task data) | All or specific clients |
| `add_client_output_filter(filter, tasks=None, clients=None)` | Client → server (task result) | All or specific clients |

**Example (BioNeMo  -  `examples/advanced/bionemo/downstream/tap/job.py`):**

```python
recipe = FedAvgRecipe(name="bionemo_tap", min_clients=3, train_script="train.py", ...)

# Convert incoming params to the right precision on each client
recipe.add_client_input_filter(BioNeMoParamsFilter(precision="bf16"), tasks=["train", "validate"])

# Convert outgoing state dict format
recipe.add_client_output_filter(BioNeMoStateDictFilter(), tasks=["train", "validate"])

# Exclude the regression head from aggregation (personal endpoint per client)
recipe.add_client_output_filter(
    BioNeMoExcludeParamsFilter(exclude_vars="regression_head"),
    tasks=["train", "validate"],
)
```

#### Config

`add_server_config(config: dict)` and `add_client_config(config: dict, clients=None)` add **top-level key-value pairs** to the generated `config_fed_server.json` or `config_fed_client.json`. At runtime, controllers and executors read these values via `nvflare.utils.configs.get_server_config_value(fl_ctx, key)` and `get_client_config_value(fl_ctx, key)`. This is the general-purpose mechanism for passing configuration from the recipe to the runtime without new constructor parameters.

**Example (BioNeMo  -  override client executor timeout):**

```python
recipe = FedAvgRecipe(name="bionemo_job", min_clients=3, train_script="train.py", ...)

# Override the default pre-init timeout for heavy library imports (e.g. BioNeMo/NeMo)
recipe.add_client_config({"EXTERNAL_PRE_INIT_TIMEOUT": 600})

# The ClientAPILauncherExecutor reads this at initialize() via:
#   get_client_config_value(fl_ctx, EXTERNAL_PRE_INIT_TIMEOUT)
```

#### File bundling

`add_client_file(file_path, clients=None)` and `add_server_file(file_path)` bundle extra files or directories (scripts, configs, resources) into generated client/server apps. Use these for launcher wrappers, helper scripts, and static assets needed at runtime.

| Method | Scope | Purpose |
|--------|-------|---------|
| `add_client_file(file_path, clients=None)` | All or specific clients | Bundle files into client app `custom/` |
| `add_server_file(file_path)` | Server | Bundle files into server app `custom/` |

**Example:**

```python
recipe = FedAvgRecipe(name="fedavg", min_clients=2, train_script="train.py", ...)
recipe.add_client_file("submit.sh")
recipe.add_client_file("env_setup.sh", clients=["site-1"])
recipe.add_server_file("server_hooks.py")
```

#### Decomposers

`add_decomposers(decomposers: List[Union[str, Decomposer]])` registers custom serialization decomposers on both server and clients. Pass class name strings or `Decomposer` instances.

**Example (PyTorch tensor decomposer for BioNeMo-style jobs):**

```python
recipe = FedAvgRecipe(name="bionemo_job", min_clients=3, train_script="train.py", ...)

# Register TensorDecomposer so PyTorch tensors can be serialized in transit
recipe.add_decomposers(["nvflare.app_opt.pt.decomposers.TensorDecomposer"])
```

**Migration note:** The BioNeMo examples in the repository still use the deprecated pattern `recipe.job.to_server(DecomposerRegister(...))` / `recipe.job.to_clients(DecomposerRegister(...))` directly. These should be migrated to `recipe.add_decomposers(...)` as shown above. The `add_decomposers` API is the recommended method — it wraps the same logic internally without exposing the internal job object.

### 3.4 Per-site config

To use **different training scripts or hyperparameters per site**, pass **`per_site_config`** on the recipe constructor. Keys are site names (e.g. `"site-1"`, `"site-2"`); values are dicts that override the default `train_script`, `train_args`, and optionally other recipe-specific options (e.g. `command`, `data_loader` for XGBoost). Common use: different data paths, learning rates, or batch sizes per site. Do not put credentials or secrets in `train_args` or per-site config; use environment variables or the execution environment's secret mechanism (see Section 6).

**Supported on:** FedAvg (base and PT/TF/Sklearn/NumPy variants), FedEval, XGBoost (vertical, histogram, bagging), and related recipes. Each recipe documents which keys it accepts in the per-site dict (e.g. `train_script`, `train_args`; XGBoost may add `data_loader`, `lr_scale`).

```python
recipe = FedAvgRecipe(
    name="fedavg",
    min_clients=2,
    num_rounds=5,
    train_script="train.py",
    train_args="--epochs 2",
    model=model,
    per_site_config={
        "site-1": {"train_script": "train_site1.py", "train_args": "--lr 0.01 --batch_size 32"},
        "site-2": {"train_args": "--lr 0.02 --batch_size 64"},
    },
)
```

**Examples:** `examples/advanced/sklearn-linear/job.py`, `sklearn-kmeans/job.py`, `sklearn-svm/job.py`; `xgboost/fedxgb/job.py`; `multi-gpu/pt/job.py`; `llm_hf/job.py`; `gnn/protein/job.py`, `gnn/finance/job.py`.

### 3.5 Model (dict config)

Recipes that need an initial model (e.g. FedAvg, FedOpt, SCAFFOLD) accept a **model** (or **initial_model**). Instead of passing a model *instance*, you can pass a **dict config** so the model class is resolved at runtime (useful when the class is not importable in the recipe process or you want to keep the job serializable). The dict must have:

- **`class_path`** (str): Fully qualified class name (e.g. `"my_package.models.MyNet"`). The runtime will import and instantiate this class.
- **`args`** (dict, optional): Constructor arguments for the model class. Default `{}`.

```python
# Dict config: no need to instantiate the model in the recipe process
model_config = {
    "class_path": "my_project.model.MyNet",
    "args": {"in_features": 784, "hidden": 128, "num_classes": 10},
}
recipe = FedAvgRecipe(
    name="fedavg",
    min_clients=2,
    num_rounds=5,
    train_script="train.py",
    train_args="--epochs 2",
    model=model_config,
)
```

You can still pass a model *instance* (e.g. `nn.Module`, `tf.keras.Model`) when you have it in memory; dict config is for when you prefer a declarative, serializable form.

### 3.6 Checkpoint usage

Recipes that support an initial model (e.g. FedAvg, FedOpt, SCAFFOLD) can take **`initial_ckpt`** and often **`save_filename`** to control where weights are loaded from and where the FL global model is saved.

- **`initial_ckpt`** (optional): Path to a pre-trained checkpoint file used to **load initial weights** on the server.
  - **Absolute path** (e.g. `/workspace/pretrained.pt`): Treated as a **server-side** path. The file must exist on the server at runtime; it is not bundled into the job. Use for production or when the checkpoint lives on shared storage.
  - **Relative path** (e.g. `start.pt`): Treated as a **local** file. The file is **bundled** into the job (server app's `custom/` directory). At runtime the persistor sees the file by basename. Use for small checkpoints or when submitting a self-contained job.
- **`save_filename`** (recipe parameter, e.g. FedAvg): Filename under which the **FL global model** is saved on the server after aggregation (e.g. `"FL_global_model.pt"`). The server persistor writes the best/current global model to this file in the job workspace.

Typical patterns:

- **Cold start (architecture + optional weights):** Pass `model=model_config` (see Section 3.5). Optionally pass `initial_ckpt` to load pre-trained weights into that model.
- **Resume from a previous FL run:** Pass `initial_ckpt` with the path to the saved global model (e.g. from a previous job's workspace). For PyTorch, you still usually need the model architecture: use `model=model_config` (Section 3.5) and `initial_ckpt=path_to_weights`.
- **Save location:** Use `save_filename` to control where the aggregated model is written each round or at the end.

Do not put secrets (e.g. credentials to download checkpoints) in paths or config; use environment variables or runtime secret injection (see Section 6).

---

## 4. ExecEnv API

**Module:** `nvflare.recipe.spec.ExecEnv`. Abstract base; you use **SimEnv**, **PocEnv**, or **ProdEnv**.

**Constructor:** `ExecEnv(extra: dict = None)`.

**Required methods (for custom envs):** `deploy(job) -> str`, `get_job_status(job_id)`, `abort_job(job_id)`, `get_job_result(job_id, timeout=0.0) -> Optional[str]`. **Optional:** `stop(clean_up=False)`.

### 4.1 SimEnv (simulation)

Best for quick experiments and debugging. Result under `/tmp/nvflare/simulation/<job_name>`.

**Arguments:** `num_clients`, `clients` (list; length must match `num_clients` if both set), `num_threads`, `gpu_config` (comma-separated), `log_config`, `workspace_root` (default `/tmp/nvflare/simulation`).

```python
from nvflare.recipe.sim_env import SimEnv
env = SimEnv(num_clients=2, num_threads=2)
run = recipe.execute(env)
run.get_result()
```

### 4.2 PocEnv (proof-of-concept)

Server and clients as separate processes on one machine. Good for demos and validation before production.

**Arguments:** `num_clients`, `clients`, `gpu_ids`, `use_he`, `docker_image`, `project_conf_path`, `username`. Set `NVFLARE_POC_WORKSPACE` if needed.

```python
from nvflare.recipe.poc_env import PocEnv
env = PocEnv(num_clients=2)
run = recipe.execute(env)
run.get_result()
```

### 4.3 ProdEnv (production)

Uses an existing NVFlare deployment; connects via the admin startup kit.

**Arguments:** `startup_kit_location`, `login_timeout`, `username`.

```python
from nvflare.recipe.prod_env import ProdEnv
env = ProdEnv(startup_kit_location="/path/to/admin/startup_kit")
run = recipe.execute(env)
run.get_result()
```

**Workflow:** Develop in SimEnv → validate in PocEnv → deploy in ProdEnv. Same recipe, no code changes.

---

## 5. Run API

**Module:** `nvflare.recipe.run.Run`; returned by `recipe.execute(env)`.

| Method | Description |
|--------|-------------|
| `get_job_id()` | Job ID from deploy. |
| `get_status()` | Current status (cached after stop). |
| `get_result(timeout=0.0)` | Wait for completion; returns result workspace path or None. After this, env is stopped and result cached. |
| `abort()` | Abort the job (no-op if already stopped). |

---

## 6. Usage flow and best practices

1. **Create a recipe** with the parameters for your algorithm (name, min_clients, num_rounds, train_script, model, etc.).
2. **Export or execute:** use `recipe.export(job_dir)` to write a job for CLI/submission, or `recipe.execute(env)` to run in SimEnv, PocEnv, or ProdEnv.
3. **Develop in SimEnv** to iterate quickly, **validate in PocEnv** for multi-process behavior, **deploy in ProdEnv** for production.
4. Start with a built-in recipe; use consistent naming; monitor execution. Add filters or config with the recipe methods if needed.

**Edge applications** (hierarchical/edge) are not supported by the simulator; use ProdEnv. See `examples/advanced/edge`.

**Kubernetes (k8s):** The same recipe should run in a Kubernetes environment with **no recipe changes**. Use an ExecEnv that deploys to k8s (e.g. a K8sEnv or k8s-backed deployment). **Note:** `deploy_map`, `resource_spec`, `client_scripts`, and `client_launcher` are discussed here as **proposed enhancements** (see Sections **10.6**, **10.7**, **10.8**, and **10.11**) and are not all available in the current Recipe API.

**Secrets and credentials (dataset access, API keys, etc.):** Recipe parameters such as **train_args**, **per_site_config**, and **add_client_config** / **add_server_config** are part of the job definition and may be written to disk or logs. **Do not put actual credentials or secrets there.** Use environment variables, mounted secrets (e.g. k8s Secrets, vault), or the execution environment's secret-injection mechanism so that the training script receives credentials at runtime. The recipe can reference secret *identifiers* (e.g. env var names or secret keys) rather than values; the runtime resolves them when the job runs.

---

## 7. Available recipes (reference)

Full list and code samples: **`user_guide/data_scientist_guide/available_recipes.rst`**.

**Categories:** FedAvg (PyTorch, TensorFlow, NumPy, Sklearn, with HE), FedProx, FedOpt, SCAFFOLD, Cyclic, XGBoost (horizontal, bagging, vertical), Sklearn (KMeans, SVM, Logistic Regression), Federated Statistics, Fed Eval, Cross-Site Eval, PSI, Flower, Swarm, Edge. **Utils:** `add_experiment_tracking(recipe, tracking_type=...)`, `add_cross_site_evaluation(recipe)`.

**Recipe coverage and limitations:** For **Swarm (swarming learning)** and **Cyclic**, several controller/config parameters exist in the Job API (`SwarmClientConfig`, `SwarmServerConfig`, etc.) but are **not yet exposed** on the recipe - e.g. **learn_task_timeout** (e.g. 10800), **max_concurrent_submissions** (e.g. 3), server/client timeouts. See Section 9 (requirement 9) and Section 10.9.

#### 7.1 Memory management (existing recipes)

**Server-side memory management:** Several recipes expose **server_memory_gc_rounds** (constructor parameter) so the server runs memory cleanup every N rounds. The recipe passes this value to the server controller as **memory_gc_rounds**; the controller (e.g. `BaseFedAvg`, `ScatterAndGather`, `CyclicController`, `Cyclic`) calls `nvflare.fuel.utils.memory_utils.cleanup_memory()` at the end of each round when the round index is a multiple of N. That function runs `gc.collect()`, `malloc_trim` (Linux/glibc), and does not use PyTorch CUDA on the server. Set **server_memory_gc_rounds** to **0** to disable.

**Recipes that support server_memory_gc_rounds:** FedAvgRecipe (unified and PT/TF variants; default **0**), CyclicRecipe (base, PT, and TF variants; default **1**), FedOptRecipe (PT default **1**, TF default **0**), ScaffoldRecipe (PT/TF; default **0**), FedAvgRecipeWithHE (PT; default **1**).

**Best practice for long-running server jobs:** set `server_memory_gc_rounds` (e.g. 5) and use `MALLOC_ARENA_MAX=4` in the server environment.

**Example (server_memory_gc_rounds):**

```python
recipe = FedAvgRecipe(
    name="fedavg",
    min_clients=2,
    num_rounds=10,
    train_script="train.py",
    model=model,
    server_memory_gc_rounds=5,   # run gc.collect + malloc_trim every 5 rounds on server
)
```

**Client-side memory management:** Client-side memory options (e.g. **client_memory_gc_rounds**, **cuda_empty_cache**) are **not yet exposed on recipes**. They are covered as **requirement 14** in Section 9 and proposed in Section 10.14. In the meantime, use `MALLOC_ARENA_MAX=2` in the client environment and call `nvflare.fuel.utils.memory_utils.cleanup_memory(cuda_empty_cache=True)` directly in your training script if needed.

---

## 8. Quick reference

| Goal | Use |
|------|-----|
| Write job to disk | `recipe.export(job_dir)` |
| Run in an environment | `recipe.execute(env)` → `Run`; then `get_job_id()`, `get_status()`, `get_result()`, `abort()` |
| Server/client config | `recipe.add_server_config(config)`, `recipe.add_client_config(config, clients=None)` |
| Per-site script/args | `per_site_config={"site-1": {"train_script": "...", "train_args": "..."}, ...}` on recipe constructor |
| Model (dict config) / checkpoint | `model={"class_path": "module.Class", "args": {...}}`; `initial_ckpt` (absolute = server-side, relative = bundled); `save_filename` for FL global model (Sections 3.5 and 3.6). |
| Task/result filters | `add_server_output_filter`, `add_server_input_filter`, `add_client_input_filter`, `add_client_output_filter` |
| Custom serialization | `recipe.add_decomposers([...])` |
| Environment-specific setup | Override `recipe.process_env(env)` |
| Server/client memory | `server_memory_gc_rounds` (constructor, e.g. FedAvgRecipe); client options see req. 14. |
| Job scope (policy) | Not yet exposed; see req. 13 (avoid `recipe.job.job.meta_props` hack). |

---

## 9. API Enhancements

Feature enhancements for the Recipe API, grouped by target release.

### Release 2.7.2

| # | Requirement | Current exposure |
|---|-------------|------------------|
| 9 | **Algorithm-specific config (timeouts, concurrency).** Recipes for Swarm, Cyclic, and similar workflows should expose or allow overriding of controller/config parameters (e.g. Swarm: learn_task_timeout, max_concurrent_submissions; server/client timeouts) so users do not need to drop to the Job API. | Swarm/Cyclic configs (SwarmClientConfig, SwarmServerConfig, etc.) exist in Job API; SimpleSwarmLearningRecipe and Cyclic recipe do not expose them. |
| 14 | **Client-side memory management in the Recipe API.** Clients need configurable memory cleanup (e.g. gc.collect + malloc_trim, and optionally torch.cuda.empty_cache) every N rounds to control RSS. Recipe API should expose **client_memory_gc_rounds** (and related options) so client executors/runners are configured for cleanup without dropping to job config. Align with `nvflare.fuel.utils.memory_utils` best practices (e.g. cleanup every round on client, MALLOC_ARENA_MAX=2). | memory_utils.cleanup_memory exists; client executors need recipe-level config. |

### Release 2.8.0

| # | Requirement | Current exposure |
|---|-------------|------------------|
| 6 | **Deploy map in the Recipe API.** Users need to specify which app goes to which site via a **deploy_map** (app name to list of sites). deploy_map is written to the job's **meta.json**; it only defines app-to-site placement, not resources. Recipe API should expose a way to set deploy_map without dropping to raw job config. | Deploy map exists at job/meta level; Recipe API should expose it. |
| 7 | **Multi-script / launcher workflows (e.g. Slurm).** Support training that uses `client.py` but depends on Slurm or similar mechanisms requiring multiple additional scripts (submit scripts, wrappers, env setup). | Today recipes typically assume a single train script (+ args). Need a way to attach extra scripts or a "launcher bundle" (or launcher type like `slurm`) so that client execution can run in cluster/slurm environments with the right scripts deployed. |
| 8 | **Kubernetes (k8s) deployment.** Recipe API must support deployment in Kubernetes without recipe changes. deploy_map (req. 6), launcher/scripts (req. 7), and **resource_spec in meta.json** (GPU/resource requirements; req. 11) should be environment-agnostic so a K8s ExecEnv can translate them to Pod/Job specs, resource requests/limits, and node selectors. | No new recipe parameters required; design existing ones so k8s backend can consume them. |
| 10 | **Optional integrations (contribution estimation, personal model, etc.).** For commonly used recipes (especially DL/FedAvg-related), flags or config to turn on optional features (e.g. contribution estimation via fed_ce; personal model via FedSM or Ditto). When enabled, the recipe auto-adds the default component/algorithm behind the scenes. | Not yet in recipe API. |
| 11 | **Resource (GPU) requirements in the Recipe API.** GPU and other resource requirements are **not** in deploy_map; they are in the job's **meta.json** under **resource_spec** (site name to resource dict, e.g. `num_gpus`, `mem_per_gpu_in_GiB`). Recipe API should expose resource_spec so the generated job's meta.json includes per-site resource requirements. | resource_spec exists in meta.json; Recipe API should expose it. |
| 12 | **Client-side experiment tracking.** Recipe API should support client-side experiment tracking as a first-class option (e.g. MLflow, TensorBoard, WandB on each client). Today `add_experiment_tracking(recipe, tracking_type, ..., client_side=True)` exists but should be documented and consistently exposed so users can enable client-side tracking without dropping to job config. | add_experiment_tracking has client_side=True; needs to be documented and stable. |
| 13 | **Job scope (meta_props) in the Recipe API.** Policy-related features (e.g. privacy scope) require setting the job scope in the job's meta. Currently users set it via the internal hack `recipe.job.job.meta_props = {"scope": "foo"}`. Recipe API should expose a way to set **scope** (or meta_props such as `scope`) so the generated job's meta.json includes it without accessing internal job types. | Job meta has scope (JobMetaKey.SCOPE); recipe does not expose it. |

---

## 10. Proposed API changes

The following are **proposed** API changes to address the requirements in Section 9. These are design proposals, not commitments.

> **Note:** Sections 10.1-10.5 were removed; original requirements 1-5 (hiding the internal job type) were deemed unnecessary after re-evaluation.

### 10.6 Requirement 6  -  Deploy map in the Recipe API

**Clarification:** **deploy_map** (in meta.json) only defines *which apps are deployed to which sites* (app name to list of site names). It does **not** carry GPU or resource information; see Requirement 11 (resource_spec) for that.

**Proposal:** Expose **deploy_map** at the recipe level so users can specify which app runs on which sites.
- **Option A  -  Constructor parameter:** e.g. `deploy_map: Optional[Dict[str, List[str]]] = None` where keys are app names and values are lists of site names (e.g. `{"app": ["server", "site-1", "site-2"]}` or `{"app_server": ["server"], "app_1": ["site-1"], "app_2": ["site-2"]}`). Default could be "single app on server and all clients" (current behavior).
- **Option B  -  Recipe method:** e.g. `recipe.set_deploy_map(deploy_map: Dict[str, List[str]])` so users can set or override the map after construction. Recipe validates that app names match the apps the recipe defines.
- When generating the job, the recipe writes this deploy_map into the job's **meta.json** (not into any other structure).

```python
# Proposed (Option A) - deploy_map parameter does not exist yet; see Requirement 6
recipe = FedAvgRecipe(
    name="multi_app",
    min_clients=2,
    train_script="train.py",
    model=model,
    deploy_map={
        "app_server": ["server"],
        "app_1": ["site-1"],
        "app_2": ["site-2"],
    },
)
```

**Recommendation (ease of use):** Keep **deploy_map** strictly for app-to-site placement. Prefer a **constructor parameter** so placement is defined in one place. Document clearly that deploy_map does not carry resource/GPU info (that is Requirement 11).

### 10.7 Requirement 7  -  Multi-script / launcher workflows (e.g. Slurm)

**Proposal:** Support **extra scripts** and/or a **launcher type** so client execution can use Slurm or similar mechanisms.

- **Option A  -  Base Recipe method:** `recipe.add_client_scripts(scripts: List[str])` defined on the **base `Recipe` class** so it works with **all** recipes without each concrete recipe needing its own constructor parameter. The base method stores the script paths and bundles them into the client app at export/execute time. Script bundling is a cross-cutting concern (not algorithm-specific) — it should be handled once in the base class rather than duplicated across every recipe's constructor.

  > **Implementation note:** The current base `Recipe` class has `add_client_file(file_path, clients=None)` which adds a single file. This proposal extends the concept to a batch method `add_client_scripts(scripts: List[str])` for convenience. The final API may unify these (e.g. keep `add_client_file` and add `add_client_scripts` as a batch wrapper, or rename to a single method). Verify the implementation matches this proposal before release.
- **Option B  -  Launcher type:** `client_launcher: Optional[Union[str, LauncherConfig]] = None` where `str` is a preset (e.g. `"subprocess"`, `"slurm"`, `"custom"`) and `LauncherConfig` (or dict) specifies `type="slurm"` plus options (e.g. `submit_script`, `wrapper_script`). This configures the client to run via `sbatch` (or similar) instead of directly invoking `train_script`. This is a separate concern from script bundling and can be added later.

Concrete recipes document how `client_scripts` and `client_launcher` are applied (e.g. which executor or launcher class is used) so that Slurm or other cluster mechanisms are supported without dropping to raw job config.

```python
# Proposed (Option A) - add_client_scripts() does not exist yet; use add_client_file() today
recipe = FedAvgRecipe(name="slurm_job", min_clients=2, train_script="client.py", model=model)
recipe.add_client_scripts(["submit.sh", "run_wrapper.sh", "env_setup.sh"])
# Same method works with any recipe:
recipe = SimpleSwarmLearningRecipe(name="swarm", model=model_config, num_rounds=10, train_script="train.py")
recipe.add_client_scripts(["submit.sh", "env_setup.sh"])

# Note: current SimpleSwarmLearningRecipe expects train_args as a dict
# (expanded into ScriptRunner via **train_args), unlike FedAvgRecipe's
# commonly used string-style train_args.

# Option B: launcher type (separate concern, can be added later)
recipe = FedAvgRecipe(name="slurm_job", min_clients=2, train_script="client.py", model=model)
recipe.add_client_scripts(["submit.sh", "env_setup.sh"])  # bundle scripts
# client_launcher="slurm"  # future: configure launcher type
```

**Recommendation (ease of use):** Prefer **Option A** (`recipe.add_client_scripts()`) on the **base `Recipe` class**. Script bundling is a cross-cutting concern (not algorithm-specific) — defining it once on the base class means every recipe (current and future) supports it automatically with no per-recipe constructor changes. For **Slurm or another launcher**, **Option B** (launcher type/preset) can be added later as a separate enhancement; it is independent of script bundling. Start with Option A for the common "bundle extra scripts" use case.

### 10.8 Requirement 8  -  Kubernetes (k8s) deployment

**Proposal:** No new recipe parameters. The same recipe runs in k8s with no changes. Ensure **deploy_map** (app→site placement), **resource_spec** in the job's **meta.json** (GPU/resource requirements per site; Section 10.11), and **client_scripts** / **client_launcher** (Section 10.7) are defined in an **environment-agnostic** way so that:

- A **K8s ExecEnv** (or k8s-backed deployment) can consume the job produced by `recipe.export()` and translate (a) **deploy_map** into which Pods run which app, and (b) **resource_spec** from meta.json into k8s resource requests/limits, node selectors, or tolerations for GPU nodes.
- Launcher/script needs (e.g. entrypoint, extra scripts) can be satisfied in k8s via container image and volume mounts rather than a new recipe API.

**Recommendation (ease of use):** Keep the recipe API environment-agnostic. Users write one recipe and choose the ExecEnv (SimEnv, PocEnv, ProdEnv, or a future K8sEnv). No "k8s mode" or k8s-specific recipe parameters - the k8s backend reads **deploy_map** and **resource_spec** from the job’s meta.json (and launcher metadata) the same way Slurm or other backends do.

### 10.9 Requirement 9  -  Algorithm-specific config (Swarm, Cyclic timeouts and concurrency)

**Context (from code):** The Job API defines **SwarmClientConfig** / **SwarmClientController** with e.g. learn_task_timeout (max time for a training task; None = no limit), max_concurrent_submissions (default 1), learn_task_abort_timeout, learn_task_ack_timeout, final_result_ack_timeout, min_responses_required, request_to_submit_result_*. **SwarmServerConfig** has start_task_timeout, configure_task_timeout, max_status_report_interval, progress_timeout (defaults: 10, 300, 90, 3600). **CyclicServerConfig** / **CyclicClientConfig** have analogous timeouts. **SimpleSwarmLearningRecipe** builds these configs without exposing these parameters.

**Important distinction  -  two kinds of config:**

| Layer | Mechanism | Where it lives | Example keys |
|-------|-----------|-----------------|--------------|
| **Job-level config** | Constructor args of `SwarmServerConfig` / `SwarmClientConfig` (or `CyclicServerConfig` / `CyclicClientConfig`) | Baked into the workflow/executor component at job-build time | `progress_timeout`, `learn_task_timeout`, `max_concurrent_submissions`, `configure_task_timeout` |
| **Top-level config** | `recipe.add_server_config()` / `recipe.add_client_config()` (Section 3.3) | Written as key-value pairs in `config_fed_server.json` / `config_fed_client.json`; read at runtime via `get_server_config_value()` / `get_client_config_value()` | `EXTERNAL_PRE_INIT_TIMEOUT`, custom app-defined keys |

The Swarm/Cyclic timeouts (e.g. `progress_timeout`, `learn_task_timeout`, `max_concurrent_submissions`) are **job-level**: they are constructor arguments of `SwarmServerConfig` / `SwarmClientConfig`, passed when the workflow is built. They are **not** top-level config keys and therefore **cannot** be set via `add_server_config()` / `add_client_config()`.

**Proposal:**

- **Option A  -  Explicit recipe constructor parameters:** Surface the most common job-level settings (e.g. `learn_task_timeout`, `max_concurrent_submissions`, `progress_timeout`) as named constructor parameters on `SimpleSwarmLearningRecipe`. The recipe passes them through to `SwarmServerConfig` / `SwarmClientConfig`. Type-safe, discoverable, documented.

- **Option B  -  Recipe method for job-level overrides:** Add `recipe.set_server_job_config(config: dict)` and `recipe.set_client_job_config(config: dict)` on the base `Recipe` class (or on algorithm-specific recipe subclasses). These methods accept a dict of constructor-argument overrides and apply them when building the underlying config objects (e.g. `SwarmServerConfig(**overrides)`). This avoids adding many constructor parameters while still allowing users to override any job-level setting.

```python
# Option A: explicit recipe parameters (discoverable, documented)
recipe = SimpleSwarmLearningRecipe(
    name="swarm_job",
    model=model_config,
    num_rounds=10,
    train_script="train.py",
    learn_task_timeout=10800,        # max seconds per training task (e.g. 3 hours)
    max_concurrent_submissions=3,    # allow 3 concurrent submissions on aggregation client
    progress_timeout=3600,           # server: max time without workflow progress
)

# Option B: recipe method for job-level overrides
recipe = SimpleSwarmLearningRecipe(
    name="swarm_job",
    model=model_config,
    num_rounds=10,
    train_script="train.py",
)
recipe.set_server_job_config({
    "progress_timeout": 3600,
    "configure_task_timeout": 600,
})
recipe.set_client_job_config({
    "learn_task_timeout": 10800,
    "max_concurrent_submissions": 3,
})
```

**Note  -  top-level config (separate concern):** `recipe.add_server_config()` / `recipe.add_client_config()` (Section 3.3) is a **separate mechanism** for top-level config keys in the JSON config files (e.g. `EXTERNAL_PRE_INIT_TIMEOUT`). It already works today and is unrelated to job-level component constructor arguments. See the BioNeMo example in Section 3.3.

**Conflict resolution  -  Option A vs Option B overlap:** If both Option A and Option B are implemented, a user could specify the same key in both places with different values (e.g. `progress_timeout=3600` in the constructor **and** `set_server_job_config({"progress_timeout": 7200})`). To prevent silent surprises, **`set_server_job_config` / `set_client_job_config` must raise a `ValueError`** if the caller passes a key that was already set via an Option A constructor parameter. The error message should tell the user to use the constructor parameter instead. This makes conflicts fail-fast and unambiguous — there is exactly one place to set each value.

**Recommendation (ease of use):** **Option A** (explicit constructor parameters) is best for the top 3-5 most commonly tuned settings (`learn_task_timeout`, `max_concurrent_submissions`, `progress_timeout`) — it is discoverable, type-safe, and IDE-friendly. **Option B** (`set_server_job_config` / `set_client_job_config`) complements Option A by providing an escape hatch for less common settings without bloating the constructor. Option B rejects any key that Option A already exposes (raises `ValueError`), so there is no ambiguity about which value takes effect. Start with Option A for the common settings; add Option B if users need to override additional job-level parameters.

### 10.10 Requirement 10  -  Optional integrations (contribution estimation, personal model, etc.)

**Proposal:** For selected recipes (primarily **FedAvg-related** and other commonly used DL recipes), support **flags or config** to turn on optional features that the recipe integrates behind the scenes by auto-adding the corresponding default component/algorithm. Consider multiple integration points under one design:

- **Contribution estimation:** User turns on a flag or config (e.g. contribution_estimation=True or "fed_ce"); recipe adds the default contribution-estimation component (e.g. **fed_ce**).
- **Personal model:** User turns on a flag or config (e.g. personal_model=True or "FedSM"); recipe adds the default personal-model component - e.g. **FedSM** as default, or **Ditto** as an alternative. Enables per-client personalized models behind the scenes.

Same pattern for each: boolean flag or string/config to select algorithm; recipe wires in the default (or chosen) implementation without the user touching the low-level Job API. Other possible options (e.g. differential privacy, compression) can follow the same pattern.

- **Option A  -  Boolean flags per feature:** contribution_estimation=True, personal_model=True. Each adds its default (fed_ce, FedSM). Simple and discoverable.
- **Option B  -  Config per feature (built-ins only):** contribution_estimation="fed_ce" | None, personal_model="FedSM" | "Ditto" | None. Choose among built-in algorithms or disable. Custom (user-provided) algorithm support is not defined yet and is left for later.

```python
# Option A: flags (defaults: fed_ce, FedSM)
recipe = FedAvgRecipe(name="fedavg", min_clients=3, num_rounds=10, train_script="train.py", model=model, contribution_estimation=True, personal_model=True)

# Option B: config (e.g. Ditto instead of FedSM)
recipe = FedAvgRecipe(..., contribution_estimation="fed_ce", personal_model="Ditto")
```

**Recommendation (ease of use):** **Start with Option A only** (boolean flags) for the first iteration. Support contribution_estimation and personal_model as flags; recipe adds built-in defaults (fed_ce, FedSM). Do not require supporting both Option A and Option B initially. **Option B** (config to choose among built-ins, e.g. personal_model="Ditto" instead of FedSM) is harder and can be added later when needed. Custom algorithm support (user-provided component) is not yet defined - we do not know how to support it in the recipe API - so leave it out of scope until we have a clear design.

### 10.11 Requirement 11  -  Resource (GPU) requirements in the Recipe API (meta.json resource_spec)

**Clarification:** GPU and other resource requirements are **not** in **deploy_map**. They are specified in the job's **meta.json** under **resource_spec**: a map from *site name* to a resource dict (e.g. `{"num_gpus": 1, "mem_per_gpu_in_GiB": 4}`). The recipe (or job generator) should write resource_spec into meta.json when producing a job; the runtime (and K8s/Slurm backends) read it for scheduling and resource allocation.

**Proposal:** Expose **resource_spec** at the recipe level so users can specify GPU (or other resource) requirements per site.
- **Option A  -  Constructor parameter:** e.g. `resource_spec: Optional[Dict[str, Dict]] = None` on each concrete recipe. Convenient when resource requirements are known at construction time, but must be duplicated across every recipe class.
- **Option B  -  Base Recipe method (proposed):** `recipe.set_resource_spec(resource_spec: Dict[str, Dict])` defined on the **base `Recipe` class** so it works with **all** recipes (FedAvg, Cyclic, Swarm, XGBoost, etc.) without each concrete recipe needing its own constructor parameter. The base method stores resource_spec and writes it into the job's **meta.json** at export/execute time. This keeps constructors focused on algorithm parameters and makes resource_spec a cross-cutting concern handled once.

  > **Implementation note:** `set_resource_spec()` does not exist in the current `Recipe` base class. This is a **design proposal** for future implementation, not current API documentation. The method name and signature may change during implementation.
- When generating the job, the recipe writes resource_spec into the job's **meta.json** alongside deploy_map.

```python
# Option A: constructor parameter (must be added to each recipe)
recipe = FedAvgRecipe(
    name="gpu_job",
    min_clients=2,
    train_script="train.py",
    model=model,
    resource_spec={
        "site-1": {"num_gpus": 1, "mem_per_gpu_in_GiB": 4},
        "site-2": {"num_gpus": 2, "mem_per_gpu_in_GiB": 8},
    },
)

# Option B: base Recipe method (works with any recipe, no constructor changes needed)
recipe = FedAvgRecipe(name="gpu_job", min_clients=2, train_script="train.py", model=model)
recipe.set_resource_spec({
    "site-1": {"num_gpus": 1, "mem_per_gpu_in_GiB": 4},
    "site-2": {"num_gpus": 2, "mem_per_gpu_in_GiB": 8},
})
# Same method works with any recipe:
recipe = CyclicRecipe(name="cyclic_gpu", min_clients=3, train_script="train.py", model=model)
recipe.set_resource_spec({"site-1": {"num_gpus": 2}})
```

**Recommendation (ease of use):** Prefer **Option B** (`recipe.set_resource_spec()`) on the **base `Recipe` class**. Resource requirements are a cross-cutting concern (meta.json, not algorithm-specific) so they belong on the base class rather than duplicated in every concrete recipe's constructor. This keeps constructors focused on algorithm parameters and ensures every recipe (current and future) supports resource_spec automatically with zero per-recipe work. Document clearly: **resource_spec** (in meta.json) = GPU/resource requirements per site; **deploy_map** (Requirement 6) = which app runs where.

### 10.12 Requirement 12  -  Client-side experiment tracking

**Context:** `add_experiment_tracking(recipe, tracking_type, tracking_config=None, client_side=False, server_side=True)` already supports **client_side=True**, which adds a tracking receiver to all clients so each client logs metrics locally (e.g. MLflow, TensorBoard, WandB). This is useful for per-site experiments and debugging.

**Proposal:** (1) Document client-side tracking in the main Recipe User Guide and this spec. (2) Ensure the API is stable: `client_side=True` adds the receiver to clients with appropriate event config (e.g. `ANALYTIC_EVENT_TYPE`). (3) Optionally add a recipe constructor parameter or method, e.g. `experiment_tracking={"type": "mlflow", "client_side": True, "server_side": True, "config": {...}}`, so users can enable tracking without a separate `add_experiment_tracking` call. Prefer documenting the existing helper as the primary path.

**Recommendation (ease of use):** **Document `add_experiment_tracking(..., client_side=True)`** as the supported way to enable client-side experiment tracking. Add a short subsection in the user guide and in Section 7 (Available recipes) so users discover it. If demand grows, add a recipe-level parameter in a later release.

### 10.13 Requirement 13  -  Job scope (meta_props) in the Recipe API

**Context:** Job **scope** is used for policy-related features (e.g. privacy filters, scope-based authorization). The job's meta.json carries a **scope** key (see `JobMetaKey.SCOPE`). Today users who need to set scope use the internal hack: `recipe.job.job.meta_props = {"scope": "foo"}`. This depends on the internal job type and is brittle.

**Proposal:** Expose **scope** (and optionally other meta_props) at the recipe level so the generated job's meta.json includes them.
- **Option A  -  Constructor parameter:** e.g. `scope: Optional[str] = None` or `meta_props: Optional[Dict[str, Any]] = None` (with `scope` as a key). When generating the job, the recipe writes these into the job's meta (e.g. meta.json) so the runtime and policy layer can read them.
- **Option B  -  Recipe method:** e.g. `recipe.set_scope(scope: str)` or `recipe.set_meta_props(props: Dict[str, Any])`.

**Recommendation (ease of use):** Add **scope** as a constructor parameter (e.g. `scope: Optional[str] = None`) so policy-related jobs can set it in one place. If other meta_props are needed later, add **meta_props** dict or a small set of named params. Document that scope is used for policy/privacy and must not contain secrets.

### 10.14 Requirement 14  -  Client-side memory management in the Recipe API

**Context:** Long-running client tasks can grow RSS. `nvflare.fuel.utils.memory_utils.cleanup_memory(cuda_empty_cache=True)` runs `gc.collect`, `malloc_trim` (Linux/glibc), and optionally `torch.cuda.empty_cache()`. Best practice: run cleanup every round on clients; use `MALLOC_ARENA_MAX=2` and `cuda_empty_cache=True` for PyTorch GPU clients. Server-side **server_memory_gc_rounds** is already exposed on FedAvgRecipe; client-side equivalent is not yet exposed on recipes.

**Proposal:** Expose **client_memory_gc_rounds** (and optionally **client_cuda_empty_cache**) at the recipe level so client executors/runners run memory cleanup every N rounds (and optionally clear CUDA cache).
- **Constructor parameter:** e.g. `client_memory_gc_rounds: int = 1` (0 = disable), `client_cuda_empty_cache: bool = False` (enable for PyTorch GPU). Recipe passes these into client config or script-runner so the client executor calls `cleanup_memory(cuda_empty_cache=...)` at the end of each round when configured.
- **Per-site override:** Allow `per_site_config[site]["client_memory_gc_rounds"]` and `client_cuda_empty_cache` so some sites can disable or use different values.

**Recommendation (ease of use):** Add **client_memory_gc_rounds** (default 1 for every round) and **client_cuda_empty_cache** (default False; enable for PyTorch GPU) to recipes that run client training (e.g. FedAvgRecipe, CyclicRecipe). Document in Section 7.1 (Memory management) and align with `memory_utils` best practices. Implementation can wire these into the client executor or script-runner config used by the recipe.

