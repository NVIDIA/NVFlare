# Multi-node PyTorch in an NVFlare Slurm job

This example starts one `torchrun` agent per node; together they form one static multi-node worker
group inside an NVFlare client job. It is example code and is not installed with NVFlare.

The site or study must use `sandbox: none`, and the job must request more than one node. Multi-node
GPU jobs must also specify `gpus_per_node`. Invoke the helper once from the node-0 NVFlare client
process. The Python interpreter, this helper, and the training script must be available at the same
paths on all allocated nodes; `srun` and `scontrol` must be available to the node-0 process. Their
command names can be overridden in trusted site or study setup with `NVFLARE_SLURM_SRUN` and
`NVFLARE_SLURM_SCONTROL`.

From the client training code, run:

```bash
python /shared/NVFlare/examples/advanced/slurm/slurm_torchrun_node.py \
  --nproc-per-node auto \
  -- train.py --your-training-arguments
```

`auto` uses `SLURM_GPUS_ON_NODE`, then PyTorch's visible CUDA device count, then one process when no
GPU is visible. PyTorch must be importable when `SLURM_GPUS_ON_NODE` is unavailable. The optional
`--rdzv-port-base` and `--rdzv-port-span` select the deterministic rendezvous-port range; `torchrun`
reports any port collision. Sites may instead use their own trusted multi-node launcher.
