# Disk-backed PyTorch FedAvg

```python
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe

recipe = FedAvgRecipe(
    name="large-fedavg",
    min_clients=2,
    num_rounds=3,
    train_script="client.py",
    model_storage="disk",
    initial_ckpt="/models/checkpoint",  # safetensors file or Hugging Face directory/index
)
```

The recipe selects PyTorch exchange, incoming tensor offload, and these components:

| Component | Responsibility |
|---|---|
| Existing FedAvg controller | Rounds, events, model updates and save/early-stop decisions |
| DiskFedAvgAggregator | Weighted FULL/DIFF reduction one tensor key at a time |
| DiskFedAvgPersistor | Load initial/saved weights and publish selected output |
| Existing tensor transport | Concurrent downloads and outgoing tensor serialization |

Both new components use existing interfaces. Generic FedAvg is unchanged. FedProx
also accepts this option; enable_tensor_disk_offload alone only selects incoming offload.

FULL/DIFF, partial keys, exclusions, scalar metrics/statistics and early stopping are
supported. Weights combine site weight and NUM_STEPS_CURRENT_ROUND. Shared keys must
match shape/dtype. Each key uses the existing FedAvg weighted aggregation helper,
preserving its dtype and rounding behavior; integer/bool averages become floating
point. DIFF accepts this promotion when matching the base and
preserves untouched keys. A rejected contribution fails the round without publication.
Initial weights may be monolithic or sharded safetensors; directories/indexes require
absolute server paths. Relative monolithic files use existing recipe bundling.

## File lifecycle and memory

Incoming client updates stay in temporary files. Aggregation writes one key at a time
to `.FL_global_model.current.safetensors.next`, then atomically replaces
`.FL_global_model.current.safetensors`. Saving renames current over
`FL_global_model.safetensors` and retargets live refs. With early stopping, saved holds
the best selected model while current may be unsaved. Reload chooses saved or initial.
Written checkpoints carry an ID checked on the opened file before each tensor read;
late downloads fail if their fixed slot has been replaced, instead of mixing models.

Outgoing refs materialize one tensor at a time and use the existing safetensors
serializer. Small tensors retain batching; disk-ref prefetch and shared byte caching
are disabled. Memory scales with tensor/batch size and concurrent receivers, including
serialization buffers. Native serialization without tensor streaming is unsupported.
Client files require roughly client_count × model_size disk space, plus checkpoint
slots. Use real disk for TMPDIR/workspace. Large-model RSS remains to be measured.

## Limits

Fixed slots require sequential rounds and completed transfers before reuse. Historical
snapshots, model inventory/locators, custom components/checkpoint names, persistor filters
and separate best-model-event files are unsupported. Only weights are persisted; full
HA resume and power-loss recovery are outside scope. Other algorithms' optimizer or
control state does not become disk-backed through this option.
