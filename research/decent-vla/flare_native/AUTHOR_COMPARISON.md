# Original versus native FLARE: author validation

Keep the original `research/decent-vla/` as the reference while testing the
independent `flare_native/` candidate. The baseline is PR #5566 revision
`41bd9659658b89f5184711584349959f8d54e59a`. Both jobs run through FLARE 2.8.1.

## Compare

Install the **native** package and the policy dependencies following
[README.md](README.md#setup-and-data). Run from `flare_native/` in that environment:

```bash
# CPU check; the final line uses a fresh comparison folder.
python job.py --generate-only
python scripts/compare_training.py --reference-dir .. \
  --config outputs/mock_fl_nvflare/run.yaml \
  --output-dir /absolute/path/to/comparisons/mock-01 --wire-dtype fp32

# A real-policy check; requires two suitable GPUs and cached weights/data.
python job.py --policy pi05 --algorithm fedadam --smoke --generate-only
python scripts/compare_training.py --reference-dir .. \
  --config outputs/pi05_libero_fullft_fedadam_nvflare_smoke/run.yaml \
  --output-dir /absolute/path/to/comparisons/pi05-adam-01 \
  --wire-dtype fp32 --num-workers 0 --checkpoint-every 1
```

The runner executes reference then native, selecting each source through a
separate subprocess and `PYTHONPATH`. Both packages share the `decent-vla` name,
so do not alternately install them. After unfolding the candidate, point
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
it seeds backend construction and records the original round history without
changing training or aggregation formulas. The runner checks that it loaded;
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

`job.py --num-clients N --gpus 0,1` expands the available GPU pool into one
resident slot per client. It does not bound concurrency to two clients.
The generated YAML's `fl.nvflare.gpu` is the expanded **group map**. The YAML
key and `FedJob.simulator_run` argument are `threads`, not `num_threads`.
`SimEnv` in newer recipe-based examples calls the analogous setting `num_threads`.

| FLARE 2.8.1 setting | Effect |
| --- | --- |
| `data.num_clients: N` | N logical sites/shards (`c0`…`cN-1`) |
| `threads: N`, no GPU map | N resident workers; set `device: cpu` for CPU execution |
| `threads: 8`, `gpu: '0'` | Eight resident clients on one GPU, if memory fits |
| `gpu: '0,1,0,1,0,1,0,1'` | Eight groups/workers, four per physical GPU |
| `gpu: '0,1'`, eight clients | Stock simulator rotates clients through two workers; native launcher rejects this map |
| `threads: 1`, `gpu: '0'`, eight clients | Sequential rotation; unsuitable for current strict parity comparisons |
| `gpu: '[0,1]'` | One group exposing two GPUs; does not enable model parallelism |

With multiple groups FLARE forces one worker **per group**, overriding threads.
Thus `threads: 1` with an eight-entry map still gives eight workers. With one
group, threads determines concurrency. The example defaults to N threads;
the bare simulator CLI can default to one. `data.num_workers` is a separate
per-client DataLoader setting, and `fl.min_clients` controls required updates,
not concurrent workers. Do not lower the quorum to fit GPU memory.

Repeated IDs do not partition or enlarge VRAM. All resident models, optimizer
state and buffers must fit, with enough host RAM for aggregation/final export.
Workers set `CUDA_VISIBLE_DEVICES`; normally use `device: auto` / local `cuda:0`.
For over 100 logical clients, export and launch with the simulator CLI's `-m N`
and matching `-c c0,...,cN-1`. The current automatic comparison launcher retains
the 100-client limit. See the [FLARE simulator guide](https://nvflare.readthedocs.io/en/main/user_guide/nvflare_cli/fl_simulator.html);
recheck release-specific behavior when changing the 2.8.1 pin.

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
