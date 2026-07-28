# Agent Inspector Simplification Implementation Plan

Status: Implemented
Author: NVFlare Team
Date: 2026-07-27
Parent design: [Agent Inspector Simplification](agent_inspector_simplification.md)

## Implementation Principle

Prefer the simplest implementation that preserves common routing and safety
behavior. Do not replace one generic semantic engine with another abstraction
layer.

The implementation must be framework-neutral. PyTorch, Lightning, and Hugging
Face must be evaluated together at every routing change. Import-only frameworks
must remain stable.

The number and boundaries of pull requests should be decided after the baseline
and dependency analysis. This plan defines implementation order, not a fixed PR
stack.

## Scope

### In Scope

- Shared AST visitor simplification.
- PyTorch, Lightning, and Hugging Face detector adaptation.
- PyTorch-family routing and unresolved-owner behavior.
- Cross-framework routing, safety, CLI, and performance tests.
- Additive inspector output compatibility improvements.

### Out Of Scope

- New framework conversion detectors.
- Recipe or Client API changes.
- Generated-job behavior changes.
- Full cross-file semantic analysis.
- Adoption of a third-party static analyzer.
- Rewriting dataset or exported-job inspection unrelated to framework routing.

## Step 0: Establish The Baseline

### Work

1. Record the current inspector test count and runtime.
2. Capture representative JSON output for:
   - plain PyTorch manual training;
   - Lightning training;
   - HF Trainer training;
   - Lightning with Transformers model usage;
   - HF Trainer with ordinary PyTorch model/data plumbing;
   - active Lightning and HF owners;
   - evaluation/inference-only HF;
   - factory-built and cross-file unresolved Trainers;
   - partial and completed Client API conversion;
   - TensorFlow, sklearn, XGBoost, JAX, and NumPy import-only cases.
3. Identify public or benchmark consumers of evidence-kind strings and output
   fields.
4. Measure changed-file and source-line ownership of:
   - lexical binding support;
   - deferred callable support;
   - decorator-result inference;
   - generator/eager-consumer handling;
   - family routing.
5. Add real source fixtures for all six current `nvflare-orient` eval cases and
   every prompt-only negative routing case in the PyTorch, Lightning, and
   Hugging Face conversion evals.
6. Record repeated wall-time and peak-memory baselines for representative
   projects and an advanced-AST fixture.

### Deliverables

- A table-driven routing baseline in tests.
- Stored expected output fragments, avoiding brittle full JSON snapshots.
- A list of compatibility-sensitive output fields and evidence kinds.
- Fixture-backed routing evals for orientation and converter-negative cases.
- Approved maximum line counts: 2,140 lines for `inspector.py` and 4,570 lines
  for `agent_inspector_test.py`, from baselines of 2,543 and 5,248.
- A recorded performance protocol and baseline.

### Exit Criteria

- Each accepted routing behavior has a direct test.
- Routing evals exercise actual source rather than prompt-only framework claims.
- Existing unintentional behavior is not silently promoted into a requirement.
- Line-count ceilings and the performance baseline are approved before visitor
  changes begin.

## Step 1: Classify Existing AST Tests

### Work

Tag or group inspector tests into:

1. **Core syntax:** imports, aliases, calls, classes, assignments, normal
   lexical scopes.
2. **Safety correctness:** rebinding, shadowing, mixed-owner detection,
   non-executing inspection.
3. **Conservative ambiguity:** factories, unresolved cross-file receivers,
   attribute-held owners.
4. **Language-completeness modeling:** callable-return decorators, class-body
   execution timing, generator consumption, eager-container behavior.

For each category 4 test, answer:

- Does this represent a supported customer training pattern?
- Can direct evidence detect it without semantic execution?
- Is `nvflare-orient` an acceptable and safer result?
- Which implementation exists solely for this test?

### Deliverables

- Explicit keep/change/remove disposition for every advanced AST test.
- Review approval for routing expectations that change to unresolved.

### Exit Criteria

- No advanced engine code is removed before its tests have an intentional
  disposition.

## Step 2: Define The Shared Evidence Contract

### Work

1. Document detector-level meanings for import, candidate, training-owner, and
   integrated evidence.
2. Keep existing evidence-kind strings where possible.
3. Add small detector methods only when they encode a stable shared question,
   such as `is_training_owner_evidence`.
4. Ensure `pytorch_class` and data-plumbing evidence remain active framework
   evidence but are not automatically training-owner evidence.
5. Keep framework-specific symbol knowledge in detector modules.

### Target Files

- `nvflare/tool/agent/frameworks/base.py`
- `nvflare/tool/agent/frameworks/pytorch.py`
- `nvflare/tool/agent/frameworks/lightning.py`
- `nvflare/tool/agent/frameworks/huggingface.py`
- `nvflare/tool/agent/frameworks/registry.py`

### Exit Criteria

- All three active detectors answer the same ownership questions.
- Registry routing does not parse framework-specific evidence strings.
- Import-only mappings remain unchanged.

## Step 3: Reduce The Shared Visitor

### Work

Retain:

- normal AST traversal;
- scope stack and direct lexical declarations;
- imports and aliases;
- direct class bases;
- direct calls;
- assignments and rebinding;
- source-order finalization for direct evidence;
- per-file recursion and parse failure handling.

Remove:

- `_DeferredCallableBody`;
- `_LazyCallableResult`;
- `_YieldFinder`;
- `_class_deferred_bodies`;
- class callable and lazy binding tables;
- decorator-return callable inference;
- eager iterable consumer special handling;
- AST visitor return values used only for inferred execution;
- secondary callable-body replay.

Remove in small increments. After each increment, run the full cross-framework
routing matrix. Retaining a listed construct requires a design amendment backed
by a real source fixture and measured benefit.

### Constraints

- Do not add a replacement control-flow framework.
- Do not special-case HF or Lightning in `inspector.py`.
- Do not weaken straightforward rebinding and shadowing behavior.
- Do not preserve a synthetic test by introducing a new execution heuristic.

### Exit Criteria

- The visitor has one understandable traversal model.
- Supported inspection remains linear in AST size.
- Unsupported runtime semantics produce unresolved evidence or no
  recommendation.

## Step 4: Adapt Framework Detectors

### PyTorch

Preserve:

- import and alias detection;
- `nn.Module` candidate detection;
- DataLoader/data-plumbing distinction;
- optimizer/loss/manual-loop evidence;
- plain Client API integration evidence.

Verify that model definitions and data plumbing do not claim training ownership
without a direct manual-training signal.

### Lightning

Preserve:

- Lightning import variants;
- `LightningModule` candidate evidence;
- reachable `Trainer(...)` construction as the existing owner proxy;
- Lightning Client API patch evidence;
- Lightning ownership when Transformers models are embedded.

Do not rewrite or weaken Lightning conversion instructions as part of detector
simplification. Bound `Trainer.fit()` ownership is a potential follow-up, not
part of this behavior-preserving cut.

### Hugging Face

Preserve:

- Transformers and TRL Trainer imports and subclasses;
- Trainer/config candidate evidence;
- direct bound `.train()` owner evidence;
- evaluation/inference-only exclusion;
- HF Client API patch evidence;
- unresolved factory fallback to orientation.

### Exit Criteria

- Each detector passes its direct unit tests.
- Cross-framework tests pass before moving to registry changes.

## Step 5: Simplify Family Resolution

### Work

Implement routing directly from:

- entry-context reachability;
- training-owner evidence;
- candidate evidence;
- conversion state;
- explicit family membership.

Required outcomes:

| Scenario | Result |
| --- | --- |
| Manual PyTorch owner plus HF tokenizer/model | PyTorch converter |
| Lightning owner plus Transformers model | Lightning converter |
| HF Trainer owner plus DataLoader/`nn.Module` | HF converter |
| HF Trainer owner plus `LightningModule`, without Lightning Trainer | HF converter |
| Active Lightning and HF owners | Orient |
| HF Trainer candidate with unresolved owner | Orient |
| HF inference plus incidental `torch` import | No converter |
| Import-only recognized framework | Report only |

Avoid score or evidence-count behavior that can override ownership.

### Target Files

- `nvflare/tool/agent/frameworks/registry.py`
- `nvflare/tool/agent/inspector.py`
- framework registry and inspector tests.

### Exit Criteria

- Routing decisions can be explained from evidence strength and owner state.
- Adding duplicate import evidence cannot change the selected owner.

## Step 6: Output Compatibility

### Work

1. Preserve the existing `"schema_version": "1"`.
2. Add ownership detail without removing or changing existing fields.
3. Add a `framework_ownership` object containing `state`, `owners`, and
   `candidates`.
4. Preserve current fields and evidence locations.
5. Keep CLI success when usable evidence exists despite per-file skips.
6. Preserve clean failure for nonexistent paths or complete inspection failure.

### Exit Criteria

- Existing consumers continue to parse current fields.
- Schema version remains `"1"` for the additive change.
- New ownership state is machine-readable.
- CLI tests cover partial inspection and schema version.

## Implementation Results

| Measure | Baseline | Result | Change |
| --- | ---: | ---: | ---: |
| `inspector.py` lines | 2,543 | 2,135 | -16.0% |
| `agent_inspector_test.py` lines | 5,248 | 4,564 | -13.0% |
| Advanced-AST median time | 0.4334 s | 0.3635 s | -16.1% |
| Advanced-AST median peak memory | 17.05 MB | 15.07 MB | -11.6% |

The unchanged real-source fixture stayed effectively flat because parsing and
result construction dominate that workload. The advanced fixture concentrates
class callables, generators, and lazy-result patterns and therefore measures
the removed machinery directly.

The benchmark used the same Python interpreter and repository revision for
both implementations, five measured runs after one warm-up, median elapsed
wall time from `time.perf_counter()`, and median peak allocation from
`tracemalloc`. The controlled advanced fixture was one generated Python file
containing a class with 600 repeated groups of:

- a nested function called from the class body;
- a lambda assigned and called from the class body;
- a generator function consumed from the class body;
- framework-like calls inside those callable bodies.

The baseline was measured in a detached worktree at the pre-change commit. The
representative real-source scan used the same checked-in source tree for both
implementations.

## Step 7: Test And Evaluation Matrix

### Unit Tests

- Framework detector tests for direct evidence.
- Registry tests for all ownership combinations.
- Inspector tests for supported scopes, rebinding, and failure isolation.
- CLI JSON tests.
- Dataset and exported-job inspection regression tests.

### Cross-Framework Cases

- PyTorch only.
- Lightning only.
- HF only.
- PyTorch plus incidental Lightning imports.
- PyTorch plus incidental HF imports.
- Lightning with Transformers model.
- HF with PyTorch model and DataLoader.
- Lightning and HF active owners.
- Factory-built or cross-file unresolved owner.
- Inference/evaluation-only code.
- Already patched and partially patched code.

### Skill Evals

- All orientation cases use source fixtures.
- Converter-negative routing cases use source fixtures.
- Correct converter chosen for each clear owner.
- Orient chosen for mixed or unresolved ownership.
- No converter for inference-only/import-only input.

### Performance

Measure:

- representative single-file scripts;
- `examples/hello-world`;
- a 250-file source tree;
- a file near existing AST size/depth limits.

Use repeated runs under the same environment. The controlled advanced-AST
fixture must improve median wall time and peak memory by at least 10 percent.
Representative source-tree scans must not regress beyond normal benchmark
variance.

## Step 8: Validation Commands

Run incrementally:

```bash
python -m pytest tests/unit_test/tool/agent/frameworks_test.py -q
python -m pytest tests/unit_test/tool/agent/agent_inspector_test.py -q
python -m pytest tests/unit_test/tool/agent/agent_cli_test.py -q
python -m pytest tests/unit_test/tool/agent -q
python -m pytest tests/unit_test/tool/agent_skill_checks -q
./runtest.sh -s
```

Run the repository license check for newly added Python modules:

```bash
./runtest.sh -l
```

Run benchmark scenarios for PyTorch, Lightning, HF, and mixed-owner paths
before merge.

## Step 9: Documentation And Cleanup

### Work

- Update `docs/design/skills_architecture.md` to describe the inspector as a
  conservative static evidence engine.
- Document the supported static syntax boundary.
- Remove obsolete comments and tests that imply complete Python semantics.
- Document changed routing expectations in release notes or the PR description.
- Report deleted complexity as well as added code.

### Exit Criteria

- Maintainers can identify supported and unresolved syntax without reading the
  visitor implementation.
- No skill claims the inspector can establish runtime ownership for arbitrary
  Python.

## Risk Controls

### Risk: Lightning Or PyTorch Regression

Mitigation:

- pin cross-framework behavior before edits;
- review every expectation change explicitly;
- run framework tests after each visitor reduction;
- do not rewrite converter instructions as part of visitor simplification.

### Risk: More Orient Results

Mitigation:

- accept conservative orientation for genuinely unresolved ownership;
- measure the rate on examples and benchmark fixtures;
- restore only common direct patterns, not arbitrary language semantics.

Abort or revise the design if any fixture with a currently clear training owner
becomes unresolved, any fixture selects the wrong converter, or the orient rate
increases outside the tests intentionally reclassified in Step 1.

### Risk: External Evidence-Kind Consumers

Mitigation:

- preserve evidence-kind strings where possible;
- add schema version before incompatible changes;
- use additive owner state.

### Risk: Simplification Becomes Another Rewrite

Mitigation:

- delete in small increments;
- avoid new frameworks or abstractions;
- meet the approved line-count ceilings;
- stop once the supported contract and routing matrix pass.

## Suggested Review Checkpoints

1. Baseline and test classification approved.
2. Shared evidence contract approved.
3. Visitor reduction reviewed independently from framework routing.
4. Detector and family routing reviewed across all three active frameworks.
5. Fixture-backed orientation and converter-negative routing evals reviewed.
6. Full tests, lint, and benchmark matrix complete.

These are review checkpoints, not predetermined PR boundaries.

## Definition Of Done

- The design's supported syntax contract is implemented.
- Common PyTorch, Lightning, and HF routing remains correct.
- Mixed and unresolved ownership fails closed.
- Import-only framework behavior remains stable.
- The approved implementation and advanced-test line-count ceilings are met.
- `_DeferredCallableBody`, `_LazyCallableResult`, `_YieldFinder`, and their
  execution-replay machinery are removed.
- The controlled benchmark meets the wall-time and memory improvement target.
- Inspector JSON compatibility is preserved with `"schema_version": "1"`.
- Targeted, full agent, skill, style, and benchmark checks pass.
- No unrelated converter or product-runtime behavior changes are included.

## Validation Results

- `python -m pytest tests/unit_test/tool/agent tests/unit_test/tool/agent_skill_checks -q`:
  505 passed, with two pre-existing Lightning warnings.
- `./runtest.sh -s`: black, isort, flake8, and agent-skill lint passed.
- `./runtest.sh -l`: license-header validation passed.
- Advanced-AST benchmark: two paired repetitions produced 12.2% to 16.1%
  lower median wall time; median peak allocation was 11.6% lower in both.
