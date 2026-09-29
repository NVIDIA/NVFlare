# Disk-backed PyTorch FedAvg

```python
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe

recipe = FedAvgRecipe(
    name="large-fedavg",
    min_clients=2,
    num_rounds=3,
    train_script="client.py",
    enable_disk_aggregation=True,
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
also accepts `enable_disk_aggregation`. The separate `enable_tensor_disk_offload`
option controls incoming streamed tensors only:

| `enable_disk_aggregation` | `enable_tensor_disk_offload` | Behavior |
|---|---|---|
| `False` (default) | Omitted, `None`, or `False` | In-memory aggregation and global model |
| `False` | `True` | Incoming tensors on disk; aggregation and global model in memory |
| `True` | Omitted, `None`, or `True` | Disk-backed aggregation and safetensors model persistence |
| `True` | `False` | Configuration error |

Incoming offload requires streamed PyTorch exchange. When enabling incoming offload
alone, also set `server_expected_format=ExchangeFormat.PYTORCH`. Disk aggregation
selects that format automatically and rejects incompatible formats.

Clients may train with NumPy while exchanging PyTorch tensors through the existing
Client API converters. Set their per-site `framework` to `FrameworkType.NUMPY` and
retain PyTorch server exchange. The client runtime needs PyTorch for conversion;
parameters must have compatible shapes and NumPy-supported dtypes.

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
