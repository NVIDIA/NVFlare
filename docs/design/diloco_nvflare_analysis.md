# Decoupled DiLoCo: Analysis and NVFlare Implementation Design

**Paper:** "Decoupled DiLoCo for Resilient Distributed Pre-training" (arXiv 2604.21428)  
**Authors:** Douillard, Rush, Donchev, Charles, et al. (Google DeepMind / Google)

---

## 1. Paper Summary

### Problem

Standard distributed LLM pre-training uses tightly coupled SPMD (Single Program Multiple Data) with global all-reduce synchronization at every step. A single hardware failure or straggler stalls the entire cluster. At scale (millions of chips, weeks of training), failures become routine — not exceptional.

Existing mitigations (elastic data-parallel, slice-granular reconfiguration) degrade goodput significantly. At 2.4M chips, elastic DP drops to ~40% goodput. Prior DiLoCo and Streaming DiLoCo reduced bandwidth but remained fundamentally synchronous.

### Core Algorithm: Decoupled DiLoCo

The algorithm separates training into two roles:

**Learners (M workers):**
- Each owns an independent data shard and isolated accelerators
- Run inner optimizer (AdamW) for H steps
- Push lightweight metadata (step counters, token counters) to syncer asynchronously
- Receive updated model fragments from syncer non-blocking — never wait for peers or syncer
- Track: `t_m` (local steps), `c_steps` (steps since last sync), `c_tokens` (tokens since last sync)

**Syncer (CPU-only coordinator):**
- Maintains global model fragments `{Θ_p}` and outer optimizer state
- Every H steps per fragment, waits for a minimum quorum of K ≤ M learners
- Pulls fragments from available learners, computes weighted outer gradient
- Applies outer optimizer (Nesterov SGD) to produce updated global fragment
- Broadcasts updated fragment to all learners

**Key innovations over standard DiLoCo:**

| Property | Standard DiLoCo | Decoupled DiLoCo |
|---|---|---|
| Synchronization | Lock-step all-reduce | Quorum-based, async |
| Failure handling | One failure stalls all | Absent learners skipped |
| Communication | Blocking barrier | Non-blocking metadata + pull |
| Uptime at 2.4M TPU chips | ~58% goodput (elastic data-parallel) | 88–100% goodput |
| Bandwidth vs data-parallel | Lower | 60–242× lower |

### Key Mechanisms

**Minimum Quorum (K):** Syncer proceeds when ≥ K of M learners report. Stragglers excluded from that sync round automatically.

**Token-weighted merging:** Each learner's contribution is weighted by `w = c_tokens × (c_tokens / c_steps)`. Faster learners processed more data and get proportional influence.

**Radial-Directional Averaging (RDA):** Per-learner outer gradients are nearly orthogonal. Simple averaging shrinks the norm by √M. RDA separately averages directions and norms so scale is invariant to M, eliminating the need to re-tune outer LR when adding learners.

**Adaptive Grace Window (ξ_grace):** When network bandwidth is available while waiting for quorum, the syncer adaptively extends the wait window to incorporate more learners — trading slack bandwidth for better sample efficiency.

**Model Fragmentation:** Model is split into P fragments with balanced tensor partitioning (greedy bin-packing). Fragment p syncs every H steps, staggered so bandwidth is spread evenly over time (not bursty).

**Learner Recovery:** When a failed learner restarts, it connects to the syncer, receives the current vector clock, then pulls a checkpoint from a healthy peer learner. Peers only respond after observing the same vector clock, ensuring consistency. Staleness is bounded by H steps.

### Experimental Results

- **Models:** 2B, 5B, 9B dense; 2.8B/3.8B MoE (Gemma 4 architecture)
- **Token budgets:** up to 1.3T tokens
- **Key results:**
  - ML quality matches standard DP at 2B–9B params
  - At long training (1.3T tokens): DiLoCo *outperforms* DP (64.9 vs 64.5 text avg)
  - Under hardware failures at 2.4M chips: 88% goodput (vs 58% elastic data-parallel)
  - Post-training (RLHF/SFT) quality preserved
  - Bandwidth: 1.7 Gbits/s vs 104 Gbits/s for data-parallel at 1s step time
  - Scavenging opportunistic compute: 0.80× wall-clock time with +300% burst capacity

### Hyperparameters

| Parameter | Value Used | Description |
|---|---|---|
| H | 24 | Inner steps between syncs (per fragment) |
| P | 24 | Number of model fragments |
| τ | 2 | Steps of comm/compute overlap |
| K | 1–8 | Minimum quorum size |
| Inner opt | AdamW | Per-learner local optimizer |
| Outer opt | Nesterov SGD | Applied by syncer to pseudo-gradients |

---

## 2. Mapping to NVFlare Concepts

### Conceptual Mapping

| Decoupled DiLoCo | NVFlare Equivalent | Notes |
|---|---|---|
| Learner (worker) | `Executor` on each client | Runs inner training loop |
| Syncer (coordinator) | `Controller` on server | Manages aggregation and outer opt |
| Inner optimizer loop (H steps) | Local training in executor | Standard NVFlare pattern |
| Pseudo-gradient (Δ = local − global) | `DXO(data_kind=DataKind.WEIGHT_DIFF)` | Already supported |
| Outer optimizer (Nesterov SGD) | Custom server-side optimizer in Controller | Similar to `FedOptCtl` |
| Token-weighted merging | Custom `Aggregator` with weighted accumulation | Extends `WeightedAggregationHelper` |
| Minimum quorum K | `broadcast_and_wait(min_responses=K)` | Built into NVFlare |
| Model fragments {Θ_p} | Controller-managed parameter partitions | New concept, manual partitioning needed |
| Non-blocking learner loop | Background task threads via NVFlare executor | Achievable with async executor pattern |
| Adaptive grace window | `wait_time_after_min_received` in broadcast | Configurable in NVFlare |
| RDA merging | Custom aggregation method in Aggregator | New implementation needed |
| Learner recovery (peer transfer) | Custom executor recovery logic | No direct equivalent, needs implementation |
| Chandy-Lamport checkpointing | NVFlare checkpoint + persistor | Partial support via `PTFileModelPersistor` |
| Model fragmentation (P parts) | Custom ShareableGenerator | New: partition and distribute fragments |

### Closest Existing Algorithm: FedOpt + SCAFFOLD

DiLoCo is closest to **FedOpt** (`nvflare/app_opt/pt/fedopt_ctl.py`):
- FedOpt: clients return weight diffs → server applies optimizer to aggregated diff
- DiLoCo: same, but clients do many more local steps (H=24 rounds of training per sync) and the outer optimizer is critical for convergence

SCAFFOLD provides the pattern for global correction state maintained on the server (`AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL`), which is analogous to the outer optimizer momentum state.

### Simplifications for NVFlare FL Context

NVFlare is designed for federated learning (not pre-training distribution), so a realistic NVFlare implementation would:

1. **Skip model fragmentation** (fragments are a bandwidth optimization for pre-training at scale; NVFlare FL clients typically have full models)
2. **Use synchronous quorum** (NVFlare's `broadcast_and_wait(min_responses=K)` achieves quorum semantics)
3. **Implement Standard DiLoCo** first, then layer on resilience features
4. **Target fine-tuning / domain adaptation** use cases, not raw LLM pre-training

---

## 3. NVFlare Implementation Design

### Component Architecture

```
Server (Syncer role)
├── DiLoCoController          ← orchestrates rounds, applies outer optimizer
├── DiLoCoAggregator          ← weighted pseudo-gradient aggregation + RDA
├── PTFileModelPersistor      ← checkpoint global model
└── FullModelShareableGenerator

Client (Learner role)
└── PTClientAPILauncherExecutor → train.py (user training script)
                                   uses nvflare.client API
                                   trains H inner steps, returns WEIGHT_DIFF
```

### Component 1: `DiLoCoController`

**File:** `nvflare/app_opt/pt/diloco_ctl.py`  
**Extends:** `BaseFedAvg` or `ModelController`

```python
class DiLoCoController(ModelController):
    def __init__(
        self,
        num_rounds: int,
        H: int = 10,                          # inner steps per round
        outer_lr: float = 0.7,
        outer_momentum: float = 0.9,           # Nesterov SGD params
        min_clients: int = 1,                  # quorum K
        wait_time_after_min_received: float = 0.0,
        persistor_id: str = "persistor",
        aggregator_id: str = "aggregator",
    ):
        ...

    def run(self):
        self.load_model()                      # load from persistor
        for round_num in range(self.num_rounds):
            # 1. Send global model to clients
            model = self.shareable_generator.learnable_to_shareable(self._global_weights, fl_ctx)
            model.set_header("H", self.H)      # tell clients how many inner steps

            # 2. Broadcast and wait for quorum
            results = self.broadcast_and_wait(
                task=Task(name="train", data=model),
                targets=self._participating_clients,
                min_responses=self.min_clients,
                wait_time_after_min_received=self.wait_time_after_min_received,
            )

            # 3. Aggregate pseudo-gradients
            for result in results:
                self.aggregator.accept(result, fl_ctx)
            aggr_result = self.aggregator.aggregate(fl_ctx)

            # 4. Apply outer optimizer (Nesterov SGD) to aggregated pseudo-gradient
            self._outer_optimizer_step(aggr_result)

            # 5. Save checkpoint
            self.save_model()
```

**Outer optimizer step:**
```python
def _outer_optimizer_step(self, pseudo_gradient_dxo):
    # Nesterov SGD applied to aggregated pseudo-gradients
    # pseudo_gradient = mean(local_params - global_params) across clients
    for name, grad in pseudo_gradient_dxo.data.items():
        # velocity update (Nesterov)
        self._velocity[name] = self.momentum * self._velocity[name] + grad
        nesterov_update = self.momentum * self._velocity[name] + grad
        self._global_weights[name] -= self.outer_lr * nesterov_update
```

### Component 2: `DiLoCoAggregator`

**File:** `nvflare/app_opt/pt/diloco_aggregator.py`  
**Extends:** `Aggregator`

```python
class DiLoCoAggregator(Aggregator):
    """
    Accumulates pseudo-gradients (WEIGHT_DIFF) from clients.
    Supports:
    - Token-weighted averaging (weight by c_tokens * quality)
    - RDA (Radial-Directional Averaging) for norm-invariant merging
    - Simple weighted averaging as fallback
    """
    def __init__(self, aggregation_method: str = "weighted_avg"):
        # aggregation_method: "weighted_avg" | "rda"
        ...

    def accept(self, shareable: Shareable, fl_ctx: FLContext) -> bool:
        dxo = DXO.from_shareable(shareable)
        num_steps = dxo.meta.get(MetaKey.NUM_STEPS_CURRENT_ROUND, 1)
        num_tokens = dxo.meta.get("num_tokens", num_steps)  # DiLoCo extension
        weight = num_tokens * (num_tokens / num_steps)       # token-weighted quality
        self._accumulate(dxo.data, weight)
        return True

    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        if self.aggregation_method == "rda":
            merged = self._rda_merge()
        else:
            merged = self._weighted_avg()

        result_dxo = DXO(data_kind=DataKind.WEIGHT_DIFF, data=merged)
        return result_dxo.to_shareable()

    def _rda_merge(self):
        """Radial-Directional Averaging: average direction and norm separately."""
        # For each parameter tensor:
        #   direction = mean(v_i / ||v_i||)
        #   norm = mean(||v_i||)
        #   result = norm * normalize(direction)
        ...
```

### Component 3: Client Training Script

**File:** `train.py` (user-provided, launched by `PTClientAPILauncherExecutor`)

```python
import nvflare.client as flare
from nvflare.client.tracking import SummaryWriter

flare.init()

model = MyModel()
inner_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

while flare.is_running():
    # Receive global model from server
    fl_model = flare.receive()
    H = fl_model.meta.get("H", 10)

    # Load global weights
    model.load_state_dict(fl_model.params)
    global_params = {k: v.clone() for k, v in model.state_dict().items()}

    # Inner training loop: H steps
    token_count = 0
    for step in range(H):
        batch = next(dataloader)
        loss = model(batch)
        inner_optimizer.zero_grad()
        loss.backward()
        inner_optimizer.step()
        token_count += batch["input_ids"].numel()

    # Compute pseudo-gradient (delta = local - global)
    pseudo_grad = {
        k: model.state_dict()[k] - global_params[k]
        for k in global_params
    }

    # Send back pseudo-gradient with token metadata
    result = FLModel(
        params_type=ParamsType.DIFF,
        params=pseudo_grad,
        meta={
            MetaKey.NUM_STEPS_CURRENT_ROUND: H,
            "num_tokens": token_count,
        }
    )
    flare.send(result)
```

### Component 4: Job Configuration

**File:** `nvflare/app_opt/pt/recipes/diloco.py`

```python
class DiLoCoJob(BaseFedJob):
    def __init__(
        self,
        name: str,
        model: nn.Module,
        num_rounds: int,
        H: int = 10,
        outer_lr: float = 0.7,
        outer_momentum: float = 0.9,
        min_clients: int = 2,
        train_script: str = "train.py",
        aggregation_method: str = "weighted_avg",  # or "rda"
    ):
        super().__init__(name=name, min_clients=min_clients)

        # Server components
        persistor = PTFileModelPersistor(model=model)
        self.to_server(persistor, id="persistor")

        aggregator = DiLoCoAggregator(aggregation_method=aggregation_method)
        self.to_server(aggregator, id="aggregator")

        controller = DiLoCoController(
            num_rounds=num_rounds,
            H=H,
            outer_lr=outer_lr,
            outer_momentum=outer_momentum,
            min_clients=min_clients,
        )
        self.to_server(controller)

        # Client components
        executor = PTClientAPILauncherExecutor(
            train_script=train_script,
            params_transfer_type="DIFF",
            params_exchange_format=ExchangeFormat.NUMPY,
        )
        self.to_clients(executor)
```

---

## 4. Resilience Features: What's Easy vs What Needs Work

### Available in NVFlare Today

| Feature | How |
|---|---|
| Minimum quorum K | `broadcast_and_wait(min_responses=K)` |
| Client dropout tolerance | `wait_time_after_min_received` + NVFlare fault handling |
| Checkpoint/restart | `PTFileModelPersistor` saves global model each round |
| Weighted aggregation | `WeightedAggregationHelper` with `NUM_STEPS_CURRENT_ROUND` |
| Privacy filters | Differential privacy filters in filter chain |

### Requires New Implementation

| Feature | Effort | Notes |
|---|---|---|
| RDA merging | Low | New aggregation math, ~100 LoC |
| Token-weighted quality metric | Low | Extra metadata in DXO |
| Nesterov SGD outer optimizer | Low | Same pattern as `FedOptCtl` |
| Model fragmentation (P parts) | Medium | New ShareableGenerator; needed only for bandwidth optimization at large scale |
| Async learner (non-blocking) | High | NVFlare uses synchronous rounds; true async requires executor threading changes |
| Adaptive grace window | Medium | Extend `broadcast_and_wait` timing logic |
| Peer-to-peer learner recovery | High | No existing peer channel; would need Cell Network direct routing |

---

## 5. Recommended Implementation Phasing

### Phase 1: Standard DiLoCo (Synchronous)

Goal: Federated fine-tuning with local-SGD + Nesterov outer optimizer.

- `DiLoCoController` with Nesterov SGD outer step
- `DiLoCoAggregator` with weighted averaging
- Client script using `nvflare.client` API, H inner steps, returns `WEIGHT_DIFF`
- Job recipe `DiLoCoJob`

Maps to: the original DiLoCo paper (not decoupled). Useful for FL fine-tuning with infrequent aggregation.

### Phase 2: Resilient DiLoCo (Quorum-based)

Goal: Tolerate client dropouts and stragglers.

- Set `min_clients=K < num_clients` in `broadcast_and_wait`
- Add token-weighted quality metric to aggregation
- Add configurable `wait_time_after_min_received` (adaptive grace window approximation)

### Phase 3: RDA + Scaling

Goal: Neutral ML performance as M (number of clients) scales.

- Implement RDA merging in `DiLoCoAggregator`
- Validate that text/vision benchmark scores remain stable as M increases from 2→4→8

### Phase 4: Decoupled (Asynchronous)

Goal: True non-blocking learner loop with overlapping communication.

- Requires architectural change: executor runs training continuously, not in NVFlare's request-response loop
- Server aggregates on its own schedule (not waiting for all clients to finish a round)
- This is a significant departure from NVFlare's current round-based paradigm

---

## 6. Limitations and Gaps

1. **NVFlare is round-based; DiLoCo (decoupled) is fully async.** Standard DiLoCo fits naturally. Decoupled DiLoCo requires a continuous training loop that is outside NVFlare's current executor model.

2. **No model fragmentation support.** NVFlare passes full models. Fragment-based communication is only needed for bandwidth optimization at pre-training scale (100k+ chips), not typical FL deployments.

3. **No peer-to-peer channel between clients.** Learner recovery in Decoupled DiLoCo requires a healthy peer to push model state to a recovering learner. NVFlare clients only communicate via the server.

4. **MoE load-balancing.** Global load-balancing loss across learners is not supported in a federated setting. Per-learner local balance only.

5. **Scale gap.** Paper targets 150K–2.4M TPU chips for resilience experiments. NVFlare FL targets 10–1000 clients. The resilience story is most compelling at pre-training scale.

---

## 7. Missing Components

This section catalogs what does not yet exist in NVFlare and must be built before DiLoCo can run end-to-end. Components are grouped by layer: framework gaps (changes to NVFlare core), new algorithm files (no core changes needed), and minor extensions (small additions to existing files).

---

### 7.1 Framework Gaps (Core NVFlare Changes Required)

These are capabilities the DiLoCo algorithm needs that NVFlare does not currently expose. Each requires a PR to core NVFlare, not just a new algorithm file.

#### Fox/Collab API (`collab_api` branch, unmerged)

The Fox API exists on `origin/collab_api` but has not been merged to `main`. Without it, the simpler Fox-based implementation (Section 8) cannot be used; the Controller/Executor path (Section 3) is the only option.

- **Status:** Functional on branch, unmerged
- **Blocker for:** Section 8 Fox implementation (not Section 3)
- **Files:** `nvflare/fox/` (~18,000 LoC on the branch)

#### Async Executor / Non-blocking Learner Loop

NVFlare executors follow a strict request-response cycle: the server sends a task, the client processes it, returns a result, then waits for the next task. True Decoupled DiLoCo requires learners to train continuously without waiting for the server's round boundary.

- **Status:** Not implemented
- **Needed for:** Phase 4 (Decoupled / fully async)
- **Approach:** Executor spawns a background training thread; server communicates via an async channel rather than task dispatch. Alternatively, Fox subprocess mode with a long-running `@fox.collab` loop approximates this.
- **Effort:** High — architectural change to executor lifecycle

#### Peer-to-Peer Client Channel

Learner recovery in Decoupled DiLoCo requires a recovering learner to pull a checkpoint directly from a healthy peer, not through the server. NVFlare's Cell Network only routes client↔server; there is no client↔client channel.

- **Status:** Not implemented
- **Needed for:** Learner recovery (Phase 4 resilience)
- **Approach:** Add a direct cell route between client cells using existing CellNet infrastructure (`nvflare/fuel/f3/cellnet/`). The server acts as a directory to announce peer addresses; the transfer itself is peer-to-peer.
- **Effort:** Medium-High

#### Adaptive Grace Window in `broadcast_and_wait`

`broadcast_and_wait` supports a fixed `wait_time_after_min_received`. The paper's adaptive grace window extends this dynamically based on available network bandwidth — if bandwidth is idle while waiting for quorum, wait longer to absorb more learners.

- **Status:** Fixed timeout only; adaptive behavior not implemented
- **Needed for:** Phase 2+ (resilient DiLoCo)
- **Approach:** Add a `grace_window_cb` hook to `broadcast_and_wait` that receives current bandwidth utilization and returns an extension delta. The DiLoCo controller supplies the callback.
- **Effort:** Medium (`nvflare/app_common/workflows/model_controller.py`)

#### Model Fragmentation in ShareableGenerator

The paper partitions the model into P fragments (greedy bin-packing by tensor size) and syncs them on a staggered schedule to spread bandwidth evenly. NVFlare's `FullModelShareableGenerator` sends the full model each round.

- **Status:** Not implemented
- **Needed for:** Bandwidth optimization at pre-training scale (not needed for FL fine-tuning)
- **Approach:** New `FragmentedModelShareableGenerator` that partitions `nn.Module` parameters into P balanced buckets; controller iterates over fragment indices instead of full-model rounds.
- **Effort:** Medium

---

### 7.2 New Algorithm Files (No Core Changes)

These files implement DiLoCo on top of existing NVFlare APIs. They require no changes to framework code and can be developed independently.

#### Controller/Executor path (Section 3)

| File | Description |
|---|---|
| `nvflare/app_opt/pt/diloco_ctl.py` | `DiLoCoController`: round orchestration, Nesterov SGD outer step |
| `nvflare/app_opt/pt/diloco_aggregator.py` | `DiLoCoAggregator`: token-weighted accumulation, RDA merge |
| `nvflare/app_opt/pt/recipes/diloco.py` | `DiLoCoJob`: wires controller + aggregator + persistor |
| `examples/diloco/train.py` | Client training script: H inner steps, returns WEIGHT_DIFF |
| `examples/diloco/run_job.py` | Job launcher |

#### Fox/Collab API path (Section 8, requires Fox merge)

| File | Description |
|---|---|
| `nvflare/app_opt/pt/fox/diloco_syncer.py` | `DiLoCoSyncer`: `@fox.algo run()`, outer optimizer, RDA |
| `nvflare/app_opt/pt/fox/diloco_learner.py` | `DiLoCoLearner`: `@fox.collab train_H_steps()`, H inner steps |
| `nvflare/app_opt/pt/fox/recipes/diloco.py` | `make_diloco_recipe()`: FoxRecipe factory |

#### Tests

| File | Description |
|---|---|
| `tests/unit_test/app_opt/pt/diloco_ctl_test.py` | Controller round logic, outer step math |
| `tests/unit_test/app_opt/pt/diloco_aggregator_test.py` | RDA merge, token weighting, quorum threshold |

---

### 7.3 Minor Extensions to Existing Files

Small additions that touch existing NVFlare files but do not restructure them.

| File | Change | Reason |
|---|---|---|
| `nvflare/apis/dxo.py` or `nvflare/app_common/app_constant.py` | Add `MetaKey.NUM_TOKENS` constant | Token-weighted aggregation needs a standard metadata key; avoids string literals |
| `nvflare/app_common/pt/pt_fed_utils.py` | Add `nesterov_sgd_step()` helper | Reusable outer optimizer step shared by DiLoCoController and any future momentum-based algorithm |

---

## 9. Files to Create (Summary)

```
nvflare/app_opt/pt/
├── diloco_ctl.py                      # DiLoCoController (Section 3, Phase 1+)
├── diloco_aggregator.py               # DiLoCoAggregator with RDA (Section 3, Phase 1+)
├── recipes/diloco.py                  # DiLoCoJob recipe (Section 3)
└── fox/
    ├── diloco_syncer.py               # DiLoCoSyncer @fox.algo (Section 8, requires Fox merge)
    ├── diloco_learner.py              # DiLoCoLearner @fox.collab (Section 8)
    └── recipes/diloco.py             # make_diloco_recipe() (Section 8)

examples/diloco/
├── README.md
├── train.py                           # Client training script (Section 3 path)
└── run_job.py                         # Job launcher

tests/unit_test/app_opt/pt/
├── diloco_ctl_test.py
└── diloco_aggregator_test.py
```

---

## 8. Alternative: Fox/Collab API Implementation

### Why Fox Simplifies DiLoCo

The Controller/Executor approach in Section 3 requires manually writing `Task` scheduling, `Shareable` encoding/decoding, and `DXO` packing/unpacking. The Fox/Collab API (code name FOX — FLARE Object Exchange, on branch `collab_api`) eliminates all of that boilerplate. The DiLoCo algorithm maps directly onto Fox's natural programming model:

| DiLoCo concept | Fox mechanism |
|---|---|
| Learner (H inner steps) | `@fox.collab` method |
| Syncer (outer optimizer) | `@fox.algo` method |
| K-of-N quorum with grace window | `fox.clients(blocking=False, timeout=grace_window).train_H_steps(...)` |
| Multi-GPU inner training (DDP) | `FoxRecipe(run_cmd="torchrun --nproc_per_node=N")` |
| Simulation / POC / production | Same code, different `env` passed to `recipe.execute(env)` |

The DiLoCo-specific logic (RDA, quorum filtering, Nesterov step) is plain Python sitting on top of Fox. The round orchestration, serialization, and communication are handled by the framework.

---

### Component 1: Syncer (server-side)

**File:** `nvflare/app_opt/pt/fox/diloco_syncer.py`

```python
import torch
from nvflare.fox import fox
from nvflare.fuel.utils.log_utils import get_obj_logger


class DiLoCoSyncer:
    """
    Syncer role: outer optimizer (Nesterov SGD) applied to aggregated pseudo-gradients.
    Uses Fox @fox.algo to orchestrate rounds, @fox.clients for quorum collection.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        num_rounds: int,
        H: int = 10,
        outer_lr: float = 0.7,
        outer_momentum: float = 0.9,
        min_clients: int = 1,           # quorum K
        grace_window: float = 0.0,      # extra seconds to wait after K respond
        aggregation: str = "rda",       # "rda" | "weighted_avg"
    ):
        self.model = model
        self.num_rounds = num_rounds
        self.H = H
        self.outer_lr = outer_lr
        self.outer_momentum = outer_momentum
        self.min_clients = min_clients
        self.grace_window = grace_window
        self.aggregation = aggregation
        self._velocity = {k: torch.zeros_like(v) for k, v in model.named_parameters()}
        self.logger = get_obj_logger(self)

    @fox.algo
    def run(self):
        global_weights = {k: v.clone() for k, v in self.model.named_parameters()}

        for round_num in range(self.num_rounds):
            self.logger.info(f"Round {round_num}/{self.num_rounds}")

            # Broadcast global weights to all clients; collect with grace window.
            # blocking=False + timeout = fire to all, accept any that respond within timeout.
            results = fox.clients(
                blocking=False,
                timeout=self.grace_window if self.grace_window > 0 else 60.0,
            ).train_H_steps(round_num, global_weights, self.H)

            # Filter errors; enforce quorum K
            pseudo_grads = []
            token_counts = []
            for site_name, val in results:
                if isinstance(val, Exception):
                    self.logger.warning(f"Client {site_name} failed: {val}")
                    continue
                grad, num_tokens = val
                pseudo_grads.append(grad)
                token_counts.append(num_tokens)

            if len(pseudo_grads) < self.min_clients:
                self.logger.warning(f"Only {len(pseudo_grads)} responses, need {self.min_clients}. Skipping round.")
                continue

            # Aggregate pseudo-gradients
            if self.aggregation == "rda":
                merged = self._rda_merge(pseudo_grads, token_counts)
            else:
                merged = self._weighted_avg(pseudo_grads, token_counts)

            # Outer optimizer: Nesterov SGD applied to aggregated pseudo-gradient
            global_weights = self._outer_step(global_weights, merged)

        self.logger.info("Training complete")
        return global_weights

    def _weighted_avg(self, pseudo_grads, token_counts):
        total_tokens = sum(token_counts)
        merged = {}
        for key in pseudo_grads[0]:
            merged[key] = sum(
                g[key] * (t / total_tokens)
                for g, t in zip(pseudo_grads, token_counts)
            )
        return merged

    def _rda_merge(self, pseudo_grads, token_counts):
        """Radial-Directional Averaging: average direction and norm separately.
        Prevents norm shrinkage by sqrt(M) that occurs with plain averaging."""
        total_tokens = sum(token_counts)
        merged = {}
        for key in pseudo_grads[0]:
            tensors = [g[key] for g in pseudo_grads]
            weights = [t / total_tokens for t in token_counts]
            norms = [t.norm() for t in tensors]
            avg_norm = sum(n * w for n, w in zip(norms, weights))
            # Weighted average of unit-normalized directions
            avg_dir = sum(
                (t / (n + 1e-12)) * w
                for t, n, w in zip(tensors, norms, weights)
            )
            dir_norm = avg_dir.norm()
            merged[key] = avg_norm * (avg_dir / (dir_norm + 1e-12))
        return merged

    def _outer_step(self, global_weights, pseudo_gradient):
        new_weights = {}
        for name, param in global_weights.items():
            grad = pseudo_gradient[name]
            v = self._velocity[name]
            # Nesterov SGD: v ← momentum*v + grad; param ← param - lr*(momentum*v + grad)
            v_new = self.outer_momentum * v + grad
            self._velocity[name] = v_new
            new_weights[name] = param - self.outer_lr * (self.outer_momentum * v_new + grad)
        return new_weights
```

---

### Component 2: Learner (client-side)

**File:** `nvflare/app_opt/pt/fox/diloco_learner.py`

```python
import copy
import torch
from nvflare.fox import fox
from nvflare.fuel.utils.log_utils import get_obj_logger


class DiLoCoLearner:
    """
    Learner role: runs H inner steps of AdamW, returns pseudo-gradient + token count.
    Decorated with @fox.collab so Fox routes server calls here automatically.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        dataset,
        inner_lr: float = 1e-4,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.dataset = dataset
        self.inner_lr = inner_lr
        self.device = device
        self.inner_optimizer = torch.optim.AdamW(model.parameters(), lr=inner_lr)
        self.logger = get_obj_logger(self)

    @fox.collab
    def train_H_steps(self, round_num, global_weights, H):
        if fox.is_aborted:
            return None

        self.logger.info(f"[{fox.call_info}] Round {round_num}: running {H} inner steps")

        # Load global weights
        self.model.load_state_dict(global_weights)
        global_snapshot = copy.deepcopy(global_weights)

        # Inner training loop: H steps of AdamW
        token_count = 0
        dataloader = iter(self.dataset)
        for step in range(H):
            try:
                batch = next(dataloader)
            except StopIteration:
                dataloader = iter(self.dataset)
                batch = next(dataloader)

            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            loss = outputs.loss

            self.inner_optimizer.zero_grad()
            loss.backward()
            self.inner_optimizer.step()
            token_count += batch["input_ids"].numel()

        # Pseudo-gradient: Δ = local_params − global_params
        pseudo_grad = {
            k: self.model.state_dict()[k].cpu() - global_snapshot[k]
            for k in global_snapshot
        }
        self.logger.info(f"[{fox.call_info}] Round {round_num}: done, tokens={token_count}")
        return pseudo_grad, token_count
```

---

### Component 3: Job Recipe

**File:** `nvflare/app_opt/pt/fox/recipes/diloco.py`

```python
from nvflare.fox.sys.recipe import FoxRecipe
from .diloco_syncer import DiLoCoSyncer
from .diloco_learner import DiLoCoLearner


def make_diloco_recipe(
    model,
    dataset_factory,         # callable: () → Dataset
    num_rounds: int = 100,
    H: int = 10,
    outer_lr: float = 0.7,
    outer_momentum: float = 0.9,
    min_clients: int = 2,    # quorum K
    grace_window: float = 5.0,
    aggregation: str = "rda",
    gpus_per_client: int = 1,
) -> FoxRecipe:

    syncer = DiLoCoSyncer(
        model=model,
        num_rounds=num_rounds,
        H=H,
        outer_lr=outer_lr,
        outer_momentum=outer_momentum,
        min_clients=min_clients,
        grace_window=grace_window,
        aggregation=aggregation,
    )

    learner = DiLoCoLearner(
        model=model,
        dataset=dataset_factory(),
        inner_lr=1e-4,
    )

    recipe = FoxRecipe(
        job_name="diloco",
        min_clients=min_clients,
        server=syncer,
        client=learner,
        # For multi-GPU inner training uncomment and set gpus_per_client > 1:
        # inprocess=False,
        # run_cmd=f"torchrun --nproc_per_node={gpus_per_client}",
    )
    return recipe
```

**Usage:**

```python
from nvflare.fox.sys.env import SimEnv, PocEnv
from nvflare.app_opt.pt.fox.recipes.diloco import make_diloco_recipe

recipe = make_diloco_recipe(model=MyModel(), dataset_factory=make_dataset, num_rounds=50)

# Simulation (single process, fast iteration):
recipe.execute(SimEnv(num_clients=4))

# POC (real CellNet, multi-process on one machine):
recipe.execute(PocEnv(num_clients=4))

# Production (multi-GPU, torchrun DDP per client):
recipe = make_diloco_recipe(..., gpus_per_client=8)
# set inprocess=False, run_cmd="torchrun --nproc_per_node=8" in FoxRecipe
recipe.execute(PocEnv(num_clients=4))
```

---

### Comparison: Fox vs Controller/Executor

| Aspect | Controller/Executor | Fox/Collab API |
|---|---|---|
| Orchestration | `DiLoCoController(ModelController)` — ~150 LoC | `@fox.algo run()` — ~50 LoC |
| Client logic | `DiLoCoExecutor(Executor)` + `train.py` — ~100 LoC | `@fox.collab train_H_steps()` — ~50 LoC |
| Serialization | `DXO(DataKind.WEIGHT_DIFF)` + `Shareable` | Plain Python dict, handled by Fox/FOBS |
| Quorum | `broadcast_and_wait(min_responses=K)` | `fox.clients(blocking=False, timeout=T)` + filter |
| Multi-GPU | Separate executor config | `run_cmd="torchrun --nproc_per_node=N"` in recipe |
| Simulation | Separate simulator setup | `SimEnv(num_clients=N)` in same code |
| Lines of code (total) | ~500 LoC (all components) | ~200 LoC (all components) |

The Fox version is **~2.5× less code** because serialization, task routing, and execution-mode switching are handled by the framework rather than the algorithm author.

### Fox Gaps for Full Decoupled DiLoCo

Fox does not eliminate all implementation work for the fully decoupled (asynchronous) variant:

1. **Per-round quorum threshold** — `min_clients` in `FoxRecipe` gates job start, not per-round acceptance. The `len(pseudo_grads) < self.min_clients` guard in `_run()` fills this gap.
2. **Grace window** — `CallOption(timeout=...)` gives a hard cutoff. The paper's adaptive grace window (extends based on available bandwidth) requires custom logic above the Fox layer.
3. **RDA** — Fox returns raw results; `_rda_merge()` is user code (~30 LoC).
4. **Peer-to-peer recovery** — Learner restart requiring a checkpoint pull from a peer has no Fox equivalent; would need Cell Network direct routing, same as the Controller/Executor approach.

---

## Appendix A: Radial-Directional Averaging (RDA) Explained

### The Problem

When you average N pseudo-gradients with plain weighted averaging, the result has norm ≈ original_norm / √N. As you add more learners M, the effective outer gradient shrinks — you would have to re-tune the outer learning rate every time you scale M up. That is impractical for large distributed runs.

Geometrically: each learner's pseudo-gradient points in a slightly different direction (each has seen different data). When you sum N nearly-orthogonal unit vectors and divide by N, the magnitude collapses by √N even though the directions contained useful signal.

### What RDA Does

RDA averages **direction** and **norm** separately, then reconstructs:

```
result = avg_norm × normalize(avg_direction)
```

**Step 1 — Direction:** normalize each pseudo-gradient to unit length, then take a weighted average of those unit vectors.

**Step 2 — Norm:** take a weighted average of the original magnitudes.

**Step 3 — Reconstruct:** scale the averaged direction by the averaged norm.

This way the output magnitude is invariant to M. Adding more learners changes the direction of the merged gradient (more data coverage) but not its scale, so the outer learning rate and convergence behavior stay stable as you scale from 2 → 4 → 8 → 64 learners.

### Code (30 LoC)

```python
def _rda_merge(self, pseudo_grads, token_counts):
    total_tokens = sum(token_counts)
    merged = {}
    for key in pseudo_grads[0]:
        tensors = [g[key] for g in pseudo_grads]
        weights = [t / total_tokens for t in token_counts]

        norms = [t.norm() for t in tensors]
        avg_norm = sum(n * w for n, w in zip(norms, weights))

        # Weighted average of unit-normalized directions
        avg_dir = sum(
            (t / (n + 1e-12)) * w
            for t, n, w in zip(tensors, norms, weights)
        )
        dir_norm = avg_dir.norm()
        merged[key] = avg_norm * (avg_dir / (dir_norm + 1e-12))
    return merged
```

### When It Matters

RDA matters most when M is large (≥4 learners) or when the outer LR is sensitive to gradient scale (Nesterov SGD with momentum). For small M (2 learners), plain weighted averaging is usually fine. The paper shows RDA is necessary to maintain benchmark parity as M scales from 8 to 64+ learners.

---

## References

- Paper: https://arxiv.org/abs/2604.21428
- Original DiLoCo: Douillard et al. 2023 (arXiv 2311.08105)
- Streaming DiLoCo: arXiv 2403.xxxxx
- NVFlare FedOpt: `nvflare/app_opt/pt/fedopt_ctl.py`
- NVFlare SCAFFOLD: `nvflare/app_common/workflows/scaffold.py`
- NVFlare ModelController: `nvflare/app_common/workflows/model_controller.py`
