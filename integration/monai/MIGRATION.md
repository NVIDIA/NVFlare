# MONAI-NVFlare Integration Migration Summary

This document summarizes the migration from the deprecated `monai_nvflare` package to the modern Client API + Recipe pattern.

## What Changed

### Deprecated (Old Pattern)
- ❌ `monai_nvflare` package
- ❌ `ClientAlgoExecutor` and `ClientAlgo` classes
- ❌ `MonaiBundlePersistor` custom persistor
- ❌ `ClientAlgoStatistics` for stats
- ❌ JSON configuration files for jobs

### Recommended (New Pattern)
- ✅ NVFlare **Client API** (`nvflare.client`)
- ✅ **FedAvgRecipe** for training jobs
- ✅ **FedStatsRecipe** for statistics jobs
- ✅ Pythonic job configuration
- ✅ Direct MONAI bundle integration with `MonaiAlgo`
- ✅ Built-in experiment tracking (MLflow, TensorBoard)

## Migration Guide

### Training Jobs

**Before:**
```python
# config_fed_client.json
{
  "executors": [{
    "executor": {
      "path": "monai_nvflare.client_algo_executor.ClientAlgoExecutor",
      "args": {"client_algo_id": "client_algo"}
    }
  }],
  "components": [{
    "id": "client_algo",
    "path": "monai.fl.client.MonaiAlgo",
    "args": {"bundle_root": "config/spleen_ct_segmentation"}
  }]
}
```

**After:**
```python
# client.py
import nvflare.client as flare
from monai.fl.client import MonaiAlgo
from monai.fl.utils.exchange_object import ExchangeObject

flare.init()

algo = MonaiAlgo(
    bundle_root=args.bundle_root,
    local_epochs=args.local_epochs,
    config_train_filename="configs/train.json",
)
algo.initialize(extra={"CLIENT_NAME": flare.get_site_name()})

while flare.is_running():
    input_model = flare.receive()
    global_weights = ExchangeObject(weights=input_model.params)
    algo.train(data=global_weights)
    updated_weights = algo.get_weights()
    output_model = flare.FLModel(params=updated_weights.weights)
    flare.send(output_model)

# job.py
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe

recipe = FedAvgRecipe(
    name="spleen_fedavg",
    min_clients=2,
    num_rounds=100,
    initial_model=FLUNet(...),
    train_script="client.py",
    train_args="--bundle_root bundles/spleen_ct_segmentation"
)
```

### Statistics Jobs

**Before:**
```python
# Using ClientAlgoStatistics with JSON config
{
  "components": [{
    "path": "monai_nvflare.client_algo_statistics.ClientAlgoStatistics",
    "args": {"client_algo_stats_id": "client_algo_stats"}
  }]
}
```

**After:**
```python
# Using FedStatsRecipe with MonaiAlgo
from nvflare.recipe.fedstats import FedStatsRecipe
from monai_stats import MonaiBundleStatistics

recipe = FedStatsRecipe(
    name="spleen_stats",
    stats_output_path="statistics/image_statistics.json",
    sites=["site-1", "site-2"],
    statistic_configs={"count": {}, "histogram": {"*": {"bins": 100}}},
    stats_generator=MonaiBundleStatistics(bundle_root="bundles/spleen_ct_segmentation")
)
```

## Updated Examples

All examples now use the modern pattern:

1. **mednist/** - Classification example with Client API
   - Direct MONAI training without MonaiAlgo
   - Manual data loading and model training
   - TensorBoard and MLflow tracking

2. **spleen_ct_segmentation/** - 3D segmentation with MONAI bundles
   - `client.py` - Client API with MonaiAlgo for bundle management
   - `job.py` - FedAvgRecipe with experiment tracking
   - `client_stats.py` - Statistics collection client
   - `job_stats.py` - FedStatsRecipe for federated statistics
   - Updated README with simulation instructions

## Benefits of Migration

1. **Simpler Code**: ~50% less code, no JSON configs
2. **Better Documentation**: Pythonic API with IDE support
3. **More Maintainable**: Standard NVFlare patterns
4. **Built-in Features**: Experiment tracking, validation, etc.
5. **Future-Proof**: Active development and support

## Backward Compatibility

The `monai_nvflare` package remains available but deprecated:
- Shows deprecation warning on import
- All classes still functional
- Will be removed in future release

Old job configurations in `job/` and `jobs/` directories are marked as deprecated but remain for reference.

## Resources

- [Client API Docs](https://nvflare.readthedocs.io/en/main/programming_guide/execution_api_type.html#client-api)
- [FedAvgRecipe API](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.app_opt.pt.recipes.fedavg.html)
- [FedStatsRecipe API](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.recipe.fedstats.html)
- [Recipe Pattern Guide](https://nvflare.readthedocs.io/en/main/programming_guide/fed_job_api.html)
