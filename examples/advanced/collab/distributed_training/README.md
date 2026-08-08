# Distributed Collab Training with PyTorch DDP

This example runs each NVFlare client as a persistent PyTorch
`DistributedDataParallel` (DDP) rank group. It uses synthetic regression data,
so no dataset download is required.

The server still makes an ordinary Collab call:

```python
client_results = collab.clients.train(global_weights, round_number)
```

At each client, Collab broadcasts that call to every DDP rank. All ranks train
with a `DistributedSampler`, and only global rank zero's result is returned to
the server for federated averaging.

## Run

From `examples/advanced`:

```bash
python -m collab.distributed_training.distributed_training
```

The default starts one NVFlare client with two DDP processes and therefore
requires two CUDA GPUs. To run two federated clients with two DDP processes
each:

```bash
python -m collab.distributed_training.distributed_training \
    --num-clients 2 \
    --nproc-per-client 2
```

In simulation, those client rank groups run concurrently. Use your scheduler
or site environment to assign an appropriate set of visible GPUs to each
client in a real deployment.

To inspect the generated job without running it:

```bash
python -m collab.distributed_training.distributed_training --export-config
```

## How the launcher fits

The recipe enables external client execution and supplies a launcher prefix for
each site:

```python
recipe = CollabRecipe(
    job_name="collab_distributed_training",
    server=server,
    client=client,
    launch_external_process=True,
)

recipe.set_per_site_config(
    {
        "site-1": {
            "command": (
                "python3 -m torch.distributed.run --nnodes=1 "
                "--nproc_per_node=2 --master_port=29500"
            )
        }
    }
)
```

The value of `command` is a prefix, not a complete training-script command.
Collab appends its distributed worker module and private bootstrap arguments.
`torchrun` creates the ranks; the decorated client methods contain the DDP
application code.

## Client lifecycle

```python
class DDPTrainer:
    @collab.init
    def initialize(self):
        self.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(self.local_rank)
        dist.init_process_group("nccl")

    @collab.publish
    def train(self, weights, round_number):
        model = make_model().to(self.local_rank)
        model.load_state_dict(weights)
        model = DDP(model, device_ids=[self.local_rank])
        sampler = DistributedSampler(self.dataset)
        # Train on this rank's shard.
        return model.module.state_dict() if dist.get_rank() == 0 else None

    @collab.final
    def finalize(self):
        dist.destroy_process_group()
```

There are no rank-specific decorator options. Collab invokes every lifecycle
and published method on all ranks in the same order. Rank-specific behavior,
such as returning a value or making an outbound Collab call only on global rank
zero, remains ordinary PyTorch application logic.

For multi-node execution, replace the per-site launcher prefix with a launcher
that starts `torchrun` on every allocated node. Rendezvous configuration,
remote-node startup, networking, workspace availability, and GPU allocation
remain responsibilities of the launcher and site operator.

## Options

| Argument | Description | Default |
|---|---|---|
| `--num-clients` | Number of federated clients | `1` |
| `--num-rounds` | Federated averaging rounds | `3` |
| `--nproc-per-client` | DDP processes per client | `2` |
| `--local-epochs` | Local epochs per federated round | `2` |
| `--master-port` | First site's rendezvous port; later sites increment it | `29500` |
| `--export-config` | Export the NVFlare job without running | disabled |

## Requirements

- NVFlare installed from this source tree
- PyTorch with CUDA and NCCL support
- At least two CUDA GPUs for the default command
