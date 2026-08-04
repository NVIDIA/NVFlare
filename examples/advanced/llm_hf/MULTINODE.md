# Multi-node Slurm Training

This example uses NVFlare's Slurm Job Launcher to run one federated client
across multiple GPU nodes. The launcher owns the Slurm allocation and the
cross-node rendezvous; the application does not call `srun` itself.

## What changed

The old `client_wrapper.sh` and `nvflare.slurm` development harness are no
longer needed. The wrapper started a nested `srun`, assembled `torchrun`
arguments, and replaced `localhost` inside `client_api_config.json`. Current
NVFlare provides those responsibilities through two existing mechanisms:

- Job topology is declared in `launcher_spec[site]["slurm"]`.
- `nvflare.app_opt.pt.torchrun_node` translates the launcher-provided
  `NVFL_*` environment into a static multi-node `torchrun` rendezvous.

Only global rank 0 owns the NVFlare Client API session. The Hugging Face Client
API patched into `SFTTrainer` broadcasts task, model, lifecycle, and error state
through PyTorch distributed collectives. The Client API's rank-0-local
bootstrap address therefore does not need to be changed for other nodes.

## Prerequisites

- An NVFlare deployment whose client site uses `runtime: slurm`.
- A running resident Slurm parent for that site.
- A shared dataset path visible at the same location on every allocated node.
- An environment or job image containing this example's dependencies.

See the [Slurm Job Launcher guide](../../../docs/user_guide/admin_guide/deployment/slurm_job_launcher.rst)
for site preparation, parent startup, container, and scheduler configuration.

The `--client_ids` values passed to `job.py` must match the deployed client site
names. For example, the commands below assume a client site named `dolly`.

## Export a two-node job

From this directory, export a job that requests two nodes and eight GPUs on
each node:

```bash
python job.py \
    --client_ids dolly \
    --data_path /shared/datasets/llm_hf \
    --job_dir /shared/jobs/llm_hf_2x8 \
    --slurm_nodes 2 \
    --slurm_gpus_per_node 8 \
    --message_mode tensor \
    --export_config
```

The exported `meta.json` contains a site-specific launcher block like this:

```json
{
  "launcher_spec": {
    "dolly": {
      "slurm": {
        "nodes": 2,
        "gpus_per_node": 8,
        "additional_node_command": "python3 -m nvflare.app_opt.pt.torchrun_node --nproc-per-node=8 custom/client.py ..."
      }
    }
  }
}
```

`job.py` sets only `nodes`, `gpus_per_node`, and the site's normal client
command. During export, `ScriptRunner` derives the complete
`additional_node_command`, including `client.py` and its site-specific
arguments.

## Submit from an admin startup kit

To export and submit the same recipe directly through `ProdEnv`:

```bash
python job.py \
    --client_ids dolly \
    --data_path /shared/datasets/llm_hf \
    --job_dir /shared/jobs/llm_hf_2x8 \
    --startup_kit_location /path/to/admin/startup_kit \
    --username admin@nvidia.com \
    --slurm_nodes 2 \
    --slurm_gpus_per_node 8 \
    --message_mode tensor
```

Do not wrap this command in an allocation that also starts the server and
client. The resident client parent submits the client job to Slurm when the
server assigns work to the site.

## Execution flow

1. The server assigns the federated task to the resident client site.
2. The Slurm launcher requests the topology in `launcher_spec` and starts one
   task per allocated node.
3. Node rank 0 runs the normal client job process. Other node ranks run the
   generated `additional_node_command`.
4. `torchrun_node` uses `NVFL_NNODES`, `NVFL_NODE_RANK`,
   `NVFL_MASTER_ADDR`, `NVFL_MASTER_PORT`, and `NVFL_RUN_ID` to start one
   `torchrun` worker group across all nodes.
5. Global rank 0 exchanges models with NVFlare. The patched Hugging Face Client
   API broadcasts task and model state to the remaining ranks, and all ranks
   execute the same `trainer.evaluate()` and `trainer.train()` calls.

`torchrun_node` also supports `nodes=1`, where it starts standalone
single-node `torchrun` with the requested number of processes.

## Troubleshooting

- Inspect the client parent log for the Slurm job ID and launcher diagnostics.
- Use `squeue` and `sacct` to inspect allocation state.
- Inspect `<run-dir>/slurm-<job-id>.out` for `torchrun_node`, NCCL, and client
  output.
- Verify that the dataset and output paths are visible on every node.
- Verify that the site name in `--client_ids` has a matching explicit
  `launcher_spec` block in the exported `meta.json`.
- Do not edit `client_api_config.json`; a remote rank attempting to connect to
  that rank-0-local endpoint indicates the process was launched outside the
  supported node-group/Cell bootstrap path.
