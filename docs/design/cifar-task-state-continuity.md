# Native PyTorch CIFAR training checkpoints

This draft builds on merged P4 (`94c5b0aebd3de9a6733926d5fc285144bfd335da`)
and is independent of the FedOpt recipe metadata change. It adds explicit native
optimizer, scheduler and main-process random-state checkpoints to the shipped
PyTorch CIFAR FedAvg, FedOpt and FedProx clients.

The standard PyTorch DataLoaders and the existing `create_data_loaders()` helper
are unchanged from the base revision. There is no custom loader, loader
checkpoint flag, batch interception or reproduction of PyTorch iterator seed
draws. No additional data-loading dependency is introduced.

The clients save and restore SGD momentum, cosine-scheduler progress and
Python/NumPy/PyTorch CPU RNG state through
`flare.system_info()["task_checkpoint"]`. The native file is scoped to the job,
site and an earlier successful round. The next global model still comes from
the server. P4 owns promotion of the candidate after validated completion and
process settlement; local promotion is separate from server receipt and
algorithm acceptance. Job lifetime performs no checkpoint file reads or writes.

FedAvg and FedOpt now dispatch evaluation without optimizer steps or checkpoint
candidates. All three job entry points expose `--execution_lifetime job|task`,
retain job lifetime by default, and bundle the shared `src` directory into the
exported client application. Existing recipe configuration remains available:

```python
recipe.add_client_config({"execution_lifetime": "task"})
```

## Remaining acceptance requirement

Main-process RNG state does not include RNG streams inside persistent
DataLoader workers. The native loader may also consume a fresh main-process
seed when its workers are recreated. Therefore this draft does not restore
persistent-worker state or establish numerical parity for the examples' default
`num_workers=2`. The default worker count is unchanged.

Complete state continuity with persistent data workers remains an unresolved
requirement in [#5472](https://github.com/NVIDIA/NVFlare/issues/5472), under the
broader task-execution acceptance in
[#5434](https://github.com/NVIDIA/NVFlare/issues/5434). This draft only partially
addresses that issue and must not close it or claim full compatibility. The
removed custom loader is not an accepted solution to that requirement.

## Validation scope

The native checkpoint unit cases cover optimizer momentum, enabled/disabled
schedulers, three main-process RNG sources, missed rounds, invalid
scope/version/rounds and unchanged job-mode checkpoint I/O. Existing FedProx
evaluation/submission checks remain included. The actual entry-point export
cases cover all three algorithms and both lifetimes, including shared helper
packaging.

The CPU Process integration cases execute the actual shipped client loops with
a small offline stochastic dataset/model, two sites, three rounds and two local
epochs. For both zero and two data workers, they compare each task's restored
native state with its preceding locally saved state, check scheduler progress,
checkpoint promotion, evaluation and fresh-process execution. With zero data
workers only, they additionally compare all per-round input/output/state and
final model/global evaluation exactly between job and task lifetimes. The
persistent-worker cases are execution and native-restore checks, not parity
qualification.

Local validation on macOS / Python 3.14.4 / PyTorch 2.12.1 passed:

- 19 focused native-checkpoint, existing FedProx dispatch and actual CLI/export
  checks, with no skips.
- Six Process cases in 416.51 seconds, covering 12 jobs. All three zero-worker
  pairs passed exact state/model/evaluation comparisons. The three two-worker
  pairs passed native checkpoint restoration and execution checks only.
- Full `./runtest.sh -s --skip-install` and whitespace checks. Notebook formatting
  was skipped because Black's Jupyter dependencies are absent.

The tested runtime/test sources were frozen before the Process run and verified
unchanged afterward. Earlier evidence from the custom-loader revision does not
qualify this revised implementation. Reproduce the checks with:

```bash
python -m pytest -q tests/unit_test/examples/cifar_task_checkpoint_test.py tests/unit_test/examples/cifar10_fedprox_client_test.py tests/unit_test/examples/cifar_task_export_test.py
python -m pytest -q tests/integration_test/fast/cifar_task_state_test.py
./runtest.sh -s --skip-install
```

This is bounded CPU SimEnv evidence. Full CIFAR, PoC/production, GPU and
performance qualification remain outstanding. SCAFFOLD, local-model submission,
Lightning restoration, TensorFlow application state and external task trainers
also remain broader release requirements.
