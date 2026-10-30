# Original versus native FLARE: author validation

Keep the original `research/decent-vla/` as the reference while testing the
independent `flare_native/` candidate. The baseline is PR #5566 revision
`41bd9659658b89f5184711584349959f8d54e59a`. Both jobs run through FLARE 2.8.1.

## Compare

Install the **native** package and the policy dependencies following
[README.md](README.md#setup-and-data). Run from `flare_native/` in that environment:

```bash
# CPU check; the final line uses a fresh comparison folder.
python job.py
python scripts/compare_training.py --reference-dir .. \
  --config outputs/mock_fl_nvflare/run.yaml \
  --output-dir /absolute/path/to/comparisons/mock-01 --wire-dtype fp32

# A real-policy check; requires two suitable GPUs and cached weights/data.
python job.py --policy pi05 --algorithm fedadam --smoke
python scripts/compare_training.py --reference-dir .. \
  --config outputs/pi05_libero_fullft_fedadam_nvflare_smoke/run.yaml \
  --output-dir /absolute/path/to/comparisons/pi05-adam-01 \
  --wire-dtype fp32 --gpu 0,1 --num-workers 0 --checkpoint-every 1
```

The runner executes the parent implementation's `decent_vla.cli.main` from
`--reference-dir`, then exports the native job with `job.py --config` and runs
`nvflare simulator`. Each source is selected through a separate subprocess and
`PYTHONPATH`. Install the native package once; both sources share the `decent-vla`
package name. After unfolding the candidate, point
`--reference-dir` at a separate original checkout. Generated pipeline paths
are shared by the two comparison jobs and packaged into exported deployments.

Use identical hardware, dependency versions, pretrained weights, dataset
snapshot, partition seed, batch size, local steps, optimizer/scheduler and
preprocessing. Site-local reconstruction of non-overlapping shares of a common
dataset is supported. Save `pip freeze` and `nvidia-smi` output with GPU results.

The comparison automatically requires all clients, disables resume/W&B, uses
fresh outputs, and sets deterministic Python/NumPy/Torch/CUDA seeds, fixed
`PYTHONHASHSEED`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, strict deterministic
algorithms, cuDNN benchmarking off and TF32 off. Data workers default to zero.
Only reference subprocesses load `scripts/reference_probe/sitecustomize.py`:
it resolves the original adapters, seeds backend construction and records
round history without changing training or aggregation formulas. The runner checks that it loaded;
no edits to the original source are required.

Unsupported deterministic CUDA operations should fail. Select the same supported
backend implementation in both versions rather than suppressing that failure.
Independent backend RNGs and worker/prefetch behavior may need additional controls;
see [PyTorch reproducibility](https://docs.pytorch.org/docs/2.7/notes/randomness.html).

## Results and longer runs

| Output | Check |
| --- | --- |
| `reference.yaml`, `native.yaml`, `comparison_meta.json` | Actual inputs, source paths, versions, expected participation and successful-check tolerances |
| `comparison.json` | Maximum per-client loss error each round and shared-tensor error at checkpoint rounds |
| `training_curves.csv` | Per-client and unweighted mean losses for plotting |
| `reference/`, `native/`, and their `.log` files | Manifests, raw curves, checkpoints, simulator workspaces and full logs |

Success requires equal partitions, complete round/client participation, finite
losses and matching shared parameter names. Every client loss and every shared
tensor at checked rounds must meet the tolerance. FedProx loss includes the
proximal term; aggregation retains dataset-frame/sample weighting. JSON rounds
are zero-based; CSV/checkpoints are one-based. A null weight error means that
round's weights were not checked, not that their error was zero.

Start with fp32 wire and `atol=1e-6, rtol=1e-5`, then test the original wire
precision, worker settings and several seeds. These tolerances are diagnostic
starting values, not a GPU convergence guarantee. Native reduction arrival order
and Adam arithmetic can round differently. If results diverge, check repeatability
of the same implementation first, then inspect the first divergent round.

For full curves, generate without `--smoke` and compare that config. All losses
are checked every round; `--checkpoint-every 10` checks weights every ten rounds
plus the final round. `--checkpoint-every 0` checks only final weights and saves
disk space. Final exports and any enabled client checkpoints still consume disk.
Weight comparison memory-maps tensors instead of loading unused Adam moments.
Recheck saved outputs without retraining:

```bash
python scripts/compare_training.py --output-dir /absolute/path/to/comparisons/pi05-adam-01 \
  --compare-only --atol 1e-6 --rtol 1e-5
```

Validate FedAvg, FedAdam and FedProx, full tuning and LoRA for each policy.
FedProx additionally needs both normalization settings; edit
`fl.strategy.extra.normalize` in generated YAML. Finish with matching checkpoint
evaluation protocols and full training curves. CPU mock agreement is insufficient.

## Clients, threads and GPUs

`job.py --num-clients N` sets the logical sites/shards. Simulator resources are
FLARE command options, not native YAML fields. For eight clients on two GPUs:

```bash
nvflare simulator jobs/smolvla_libero_lora_nvflare \
  -w outputs/smolvla_workspace -c c0,c1,c2,c3,c4,c5,c6,c7 -t 8 \
  -gpu 0,1,0,1,0,1,0,1
```

The simulator CLI uses `-t` / `--threads`; newer recipe examples call the
analogous `SimEnv` option `num_threads`. `-gpu` / `--gpu` takes a **group map**,
not simply a pool of available GPUs. The comparison runner's `--threads` and
`--gpu` forward identical resources to the original launcher and native FLARE CLI.
It defaults to N threads and requires resident clients for strict parity.

| FLARE 2.8.1 option, with eight logical clients | Effect |
| --- | --- |
| `-t 8`, no GPU map | Eight resident workers; set `device: cpu` for CPU execution |
| `-t 8 -gpu 0` | Eight resident clients on one GPU, if memory fits |
| `-gpu 0,1,0,1,0,1,0,1` | Eight groups/workers, four per physical GPU |
| `-gpu 0,1` | Stock simulator rotates clients through two workers; loses warm state |
| `-t 1 -gpu 0` | Sequential rotation; unsuitable for current strict comparisons |
| `-gpu '[0,1]'` | One group exposing two GPUs; does not enable model parallelism |

With multiple groups FLARE forces one worker **per group**, overriding threads.
Thus `-t 1` with an eight-entry map still gives eight workers. With one group,
threads determines concurrency. Always specify threads explicitly; the stock
CLI can default to one. `data.num_workers` is a separate per-client DataLoader
setting, and `fl.min_clients` controls required updates, not concurrent workers.
Do not lower the quorum to fit GPU memory.

Repeated IDs do not partition or enlarge VRAM. Resident models, optimizer state
and buffers must fit, with enough host RAM for aggregation/final export. Workers
set `CUDA_VISIBLE_DEVICES`; normally use `device: auto` / local `cuda:0`.
For over 100 logical clients, use the simulator CLI's `-m N` and matching
`-c c0,...,cN-1`. The automatic original/native comparison retains the reference
launcher's 100-client limit. See the [FLARE simulator guide](https://nvflare.readthedocs.io/en/main/user_guide/nvflare_cli/fl_simulator.html);
recheck behavior when changing the 2.8.1 pin.

## Sequential execution needs complete disk-backed state

The stock simulator supports fewer workers than clients, but swaps processes
out between execution turns. Current strict comparisons reject these layouts:
client-process state would be lost. `persist_client_state: true` with
`save_every_n_rounds: 1` saves weights, optimizer and scheduler, but omits RNG
and data-iterator progress. It is insufficient for equivalent sequential training.
Comparison jobs also disable resume deliberately.

For large models under limited GPU memory, implement disk-backed state using
FLARE `SWAP_OUT` / `SWAP_IN` lifecycle hooks. It must preserve all shared/personal
parameters and buffers, optimizer/scheduler/global step, AMP scaler where used,
Python/NumPy/Torch/CUDA/backend RNGs, and sampler order/epoch/batch position.
Workers/prefetch need a checkpointable or reproducibly replayable design.
Publish checkpoints atomically with completed-round identity before acknowledging
completion so retries cannot train twice. Server FedAdam state cannot replace
client state. Validate resident versus swapped runs of the same implementation
before comparing original versus native. This feature remains unimplemented.


## Final PR cleanup after alignment

After the authors confirm real-policy training curves and evaluation alignment,
save the comparison results as review evidence outside the contribution. Remove
these migration-only files before preparing the final PR contents. Paths below
are relative to `flare_native/`:

| File to remove | Temporary purpose |
| --- | --- |
| `AUTHOR_COMPARISON.md` | This migration/author-validation guide |
| `scripts/compare_training.py` | Launches the original and native implementations and compares their artifacts |
| `scripts/reference_probe/sitecustomize.py` | Instruments the original implementation for deterministic comparisons; remove the entire `scripts/reference_probe/` directory |
| `tests/unit_test/scripts/training_comparison_test.py` | Tests the temporary comparison runner; remove it together with that runner |

Before deleting this guide, retain the useful simulator GPU/thread guidance and
sequential-execution limitations in `README.md`. Remove its links to this guide,
the migration/unfolding instructions, and comparison-tool references in
`tests/README.md`. Check for references to the removed files and
`DVLA_REFERENCE_*` hooks.

Keep `job.py`, `evaluate.py`, policy/data adapters, native FLARE workflows,
persistence, deterministic seeding, training metrics, and the remaining native
unit/deployment tests. They support normal use of the example rather than the
original-versus-native comparison. Keep their test helpers and the policy
setup/evaluation scripts as well.

Remove the original implementation outside `flare_native/` from
`research/decent-vla/`, then move the remaining native contents, including
dotfiles, up into `research/decent-vla/`. Generated jobs, workspaces, downloaded
models/data and comparison outputs stay outside the final tracked contribution.
Re-export the job and rerun the native checks from the final location.
