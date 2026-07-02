# Export Argument Fidelity (Reflection-Based Config Export)

Status: design proposal
Scope: `FedJobConfig` export / job config generation — **product-wide, independent
of any agent/skill work**. This proposal stands on its own; it must be fixed
whether or not the skills effort ships. See `agent_skill_operating_model.md` for the
separate skills proposal, which merely *consumes* the guarantee established here.

## Problem

Job export reconstructs each component's `args` block by **reflecting over the
instance** and matching constructor parameters to instance attributes. In
`FedJobConfig._get_args` (`nvflare/job_config/fed_job_config.py:414`):

- `get_component_init_parameters(component)` yields the constructor params +
  defaults (`nvflare/fuel/utils/class_utils.py:195`, walks the MRO signature).
- `attrs = component.__dict__` — the instance state.
- For each param, `attr_key = param if param in attrs else "_" + param`, and it
  serializes only when `attr_key in attrs` and the value differs from the default
  (or the param is in `_always_serialize_args`).

This encodes an assumption: **every constructor parameter is stored as an
instance attribute of the same name (`self.<param>` or `self._<param>`).** When a
class violates that — consumes a param without storing it, or stores it under a
different name, or fans it into derived fields — the parameter is **silently
dropped** from the exported config. The reconstructed job is then quietly wrong.

## Why this is a general problem, not a few classes

Export serializes the **entire component graph of a job**, and much of that graph
is **user-authored**: custom controllers, executors, persistors, filters,
shareable generators, aggregators, and other model-related components added
through the Job API. Users follow no attribute convention, and we cannot
enumerate their classes.

This proposal covers user-authored classes that participate in the FLARE
component contract. Arbitrary user ML classes that do not inherit from a FLARE
component base, such as a `torch.nn.Module`, sklearn estimator, or application
helper object, need an explicit config boundary such as `{path, args}` or a
FLARE wrapper component. They cannot be captured transparently through
`FLComponent.__init_subclass__()`.

Therefore the fidelity of reflection-based export cannot be established by auditing
FLARE's own classes. A sweep of FLARE components (below) is useful for gauging
migration risk, but it is **not** the answer to the problem — the problem is
structural to reflection and applies to every class we did not write.

### Evidence: FLARE component sweep (migration-risk gauge only)

A static sweep of 43 in-scope FLARE components (mirroring `_get_args`: MRO signature
for params, AST walk of every `__init__` for `self.<param>`/`self._<param>`) found:

- **41 / 43** store every constructor param → reflection recovers them today.
- **2** drop a param:
  - `FedXGBTreeExecutor` (`nvflare/app_opt/xgboost/tree_based/executor.py:85`):
    `learning_rate` stored as `self.base_lr` → dropped.
  - `MetricsArtifactWriter` (`nvflare/app_common/widgets/metrics_artifact_writer.py:57-66`):
    `limits` fanned into nine `self.max_*` fields → dropped.
- Enum-valued stored params (`ScriptRunner`, `FilePipe`) are **not** a problem:
  `_get_args` has an `Enum → .value` branch (`fed_job_config.py:432-433`), so they
  serialize as scalars and round-trip.

Read correctly, this says the known FLARE-side breakage is small enough to fix
manually. It does **not** justify automatic constructor capture: that mechanism
would affect every `FLComponent` subclass while still not solving arbitrary
non-`FLComponent` user classes such as `torch.nn.Module` or sklearn estimators.

## Design options

### Option A - Explicit export-args protocol (recommended)

Keep reflection as the default behavior, but let a component opt into an
authoritative export representation when reflection cannot recover the
constructor contract.

Add a small optional method recognized by `FedJobConfig._get_args()`:

```python
class ExportableConfigArgs:
    def export_config_args(self) -> dict:
        """Return constructor-compatible, config-serializable args."""
```

`FedJobConfig._get_args()` becomes:

```python
def _get_args(self, component, custom_dir):
    export_args = getattr(component, "export_config_args", None)
    if callable(export_args):
        return self._serialize(export_args(), custom_dir)

    return self._get_args_by_reflection(component, custom_dir)
```

This has isolated impact:

- no `__init__` wrapping;
- no metaclass or `__init_subclass__` behavior change;
- no memory retention from captured constructor args;
- no export change for existing classes unless they opt in;
- known FLARE classes can be fixed surgically;
- user-authored FLARE components have an explicit escape hatch.

The method must return the same argument names the constructor accepts, and the
values must already be config-shaped or serializable by the existing component
serializer. It must not return runtime-only objects such as open files, sockets,
live model instances, or secrets.

#### Usage examples

Example 1: `MetricsArtifactWriter` fans one constructor argument into several
derived fields. Store the original config-shaped value and expose it explicitly:

```python
class MetricsArtifactWriter(Widget):
    def __init__(self, limits: dict | None = None):
        super().__init__()
        self._limits = limits or {}
        self.max_json_files = self._limits.get("json", 100)
        self.max_tb_files = self._limits.get("tensorboard", 20)
        self.max_log_files = self._limits.get("logs", 50)

    def export_config_args(self):
        return {"limits": self._limits}
```

Export result:

```json
{
  "path": "nvflare.app_common.widgets.metrics_artifact_writer.MetricsArtifactWriter",
  "args": {
    "limits": {
      "json": 10,
      "tensorboard": 5
    }
  }
}
```

Example 2: `FedXGBTreeExecutor` stores `learning_rate` under `base_lr`. Preserve
the constructor name in the export contract:

```python
class FedXGBTreeExecutor(Executor):
    def __init__(self, learning_rate: float = 0.3, ...):
        super().__init__()
        self.base_lr = learning_rate
        ...

    def export_config_args(self):
        args = self._get_reflection_export_args()
        args["learning_rate"] = self.base_lr
        return args
```

The exact implementation should avoid duplicating the whole constructor argument
list by hand if possible. For example, a small shared helper can compute today's
reflection result and let the component patch only the fields reflection cannot
recover.

Example 3: a user-authored FLARE component can opt in without affecting any
other component:

```python
class SiteModelPersistor(ModelPersistor):
    def __init__(self, model_dir: str, strict: bool = True):
        super().__init__()
        self.root = Path(model_dir)
        self._strict_mode = strict

    def export_config_args(self):
        return {
            "model_dir": str(self.root),
            "strict": self._strict_mode,
        }
```

Example 4: arbitrary non-`FLComponent` model classes still use explicit recipe
or persistor config. They are not solved by export reflection:

```python
model = {
    "class_path": "my_project.models.Net",
    "args": {"num_classes": 10}
}
```

Recipes normalize this with `recipe_model_to_job_model()`:

```python
{"path": "my_project.models.Net", "args": {"num_classes": 10}}
```

For a live PyTorch model instance, export cannot infer how to reconstruct the
constructor. If reconstruction matters, pass `{path, args}`. If runtime state
matters, use the existing persistor/checkpoint path.

#### Decorator variant

A decorator can reduce boilerplate for classes that want opt-in capture without
changing every `FLComponent` subclass. The decorator should be a front end to
the same `export_config_args()` protocol, not a separate export path.

Component-class decorator:

```python
@export_config_args
class SiteModelPersistor(ModelPersistor):
    def __init__(self, model_dir: str, strict: bool = True):
        super().__init__()
        self.root = Path(model_dir)
        self._strict_mode = strict
```

Conceptual implementation:

```python
def export_config_args(cls):
    original_init = cls.__init__

    @functools.wraps(original_init)
    def wrapped_init(self, *args, **kwargs):
        sig = inspect.signature(original_init)
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()

        original_init(self, *args, **kwargs)

        captured = dict(bound.arguments)
        captured.pop("self", None)
        captured.pop("args", None)
        captured.pop("kwargs", None)
        self._nvflare_export_config_args = captured

    cls.__init__ = wrapped_init

    if not hasattr(cls, "export_config_args"):
        cls.export_config_args = lambda self: self._nvflare_export_config_args

    return cls
```

Export remains unchanged from Option A:

```python
export_args = getattr(component, "export_config_args", None)
if callable(export_args):
    return self._serialize(export_args(), custom_dir)
```

This decorator has a much smaller blast radius than base-class automatic
capture: only decorated classes change behavior. It still has constructor
wrapping side effects for that class, so it should be optional and documented.
Use an explicit hand-written `export_config_args()` method for classes with
heavy objects, secrets, normalization, or non-obvious config semantics.

For non-`FLComponent` model classes, prefer recipe-internal normalization rather
than wrapping the live class. The user can still write code in terms of class
`A`; the recipe turns `A` plus explicit args into the config representation that
export can preserve:

```python
recipe = FedAvgRecipe(
    model_class=MyNet,
    model_args={"num_classes": 10},
    ...
)
```

Inside the recipe:

```python
def _normalize_model(model=None, model_class=None, model_args=None):
    if model_class is not None:
        return {
            "class_path": f"{model_class.__module__}.{model_class.__qualname__}",
            "args": model_args or {},
        }

    if isinstance(model, dict):
        return recipe_model_to_job_model(model)

    # Existing live-instance path. This can preserve runtime state through a
    # persistor/checkpoint, but it cannot recover constructor args reliably.
    return model
```

This is the "decorator inside the recipe" pattern in spirit: the recipe owns the
adapter and the user still names class `A`. But it should not monkey-patch
`A.__init__` or create a generated subclass `class B(A)`. Those approaches change
class identity, module path, MRO, pickling behavior, and framework checks. They
also only work if the recipe instantiates the object; they cannot recover args
from an already-created `A(...)` instance.

The key rule is:

- if the recipe receives `A` plus args before construction, it can produce
  `{class_path/path, args}`;
- if the recipe receives a live `A(...)` instance, constructor args are already
  gone unless the class itself recorded them.

### Option B - Full component config contract (`to_config()`)

Let a component return the full `{path, args}` block instead of only args.
This is more flexible for exotic components, but it overlaps more with
`FedJobConfig` and gives component authors control over class-path serialization.
Keep this as a future escape hatch, not the first implementation.

### Option C - Automatic constructor capture

Record bound constructor args by wrapping `FLComponent.__init__` through a
metaclass or `__init_subclass__`. This fixes some user-authored FLARE component
cases automatically, but it has a large blast radius:

- changes constructor behavior for every `FLComponent` subclass;
- can affect multiple inheritance, introspection, decorators, stack traces, and
  test assumptions;
- can retain large objects, secrets, or runtime handles;
- still does not capture arbitrary non-`FLComponent` model/application classes.

Reject for now. Revisit only if explicit export args plus product boundaries do
not solve real user failures.

### Option D - Keep reflection, only fix current classes manually

This is the smallest immediate patch and should be done regardless:

- store `FedXGBTreeExecutor.learning_rate` in an export-recoverable field or
  implement `export_config_args()`;
- store `MetricsArtifactWriter._limits` or implement `export_config_args()`.

Alone, this leaves no general escape hatch for user-authored FLARE components,
so it is not enough as the design. It is part of Option A's rollout.

## Recommendation

Option A: explicit `export_config_args()` protocol + reflection fallback.

Immediate patches: fix the two dropped FLARE params with either stored
export-recoverable fields or `export_config_args()`:

- `FedXGBTreeExecutor.learning_rate`
- `MetricsArtifactWriter._limits`

Do not implement automatic constructor capture in the first version. It is too
impactful for the benefit, and it still does not address the non-`FLComponent`
user-class problem.

For arbitrary user model classes, keep the boundary explicit: recipe model config
or wrapper component with `{path, args}`, plus persistor/checkpoint handling for
runtime state.

## Tests

- Opt-in round-trip: a component with `export_config_args()` exports with the
  returned args and reconstructs equivalently.
- Non-serializable arg: a component constructed with an open handle / bare callable
  fails export loudly with a message naming component+param — not a silent drop.
- Fallback: a component without `export_config_args()` still exports via
  reflection unchanged.
- Guard (parametric over the in-scope FLARE catalog): every constructor param is
  export-recoverable either by reflection or `export_config_args()`.
- Regression: existing recipes export byte-comparable or behaviorally equivalent
  configs, except the two intentional fixes.
- Non-`FLComponent` model config: recipe `{class_path/path, args}` still normalizes
  through `recipe_model_to_job_model()` and exports as `{path, args}`.

## Work items

1. Immediate: store `FedXGBTreeExecutor.learning_rate` and
   `MetricsArtifactWriter._limits` as export-recoverable fields.
2. Add optional `export_config_args()` handling in `FedJobConfig._get_args()`,
   before reflection fallback.
3. Add a small shared serializer/helper so an opt-in component can reuse today's
   reflection result and patch only the missing fields.
4. Add loud-fail behavior for non-serializable values returned by
   `export_config_args()`.
5. Add tests above, including a user-authored FLARE component test double.
6. Keep non-`FLComponent` user classes on explicit `{path, args}` / wrapper
   boundaries; document that constructor recovery is not attempted for live
   model instances.

## Relationship to the skills proposal

None mechanically. The skills/plan proposal assumes exported jobs faithfully
reflect resolved configuration; this proposal provides that. If this ships and
skills never do, it is still correct and complete. Keep the two reviewed and
merged independently.
