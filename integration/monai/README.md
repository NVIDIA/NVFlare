# MONAI Integration

## Objective
Integration with [MONAI](https://project-monai.github.io/) for federated learning using NVFlare's Client API and FedAvgRecipe.

### Goals:
Enable the use of MONAI [bundles](https://monai.readthedocs.io/en/latest/bundle.html) from the MONAI [model zoo](https://github.com/Project-MONAI/model-zoo) or custom configurations with NVFlare using modern Client API patterns.

## Description
MONAI allows the definition of AI models using the "bundle" concept for easy experimentation and sharing. This integration shows how to use MONAI bundles in federated learning scenarios using NVFlare's **Client API** and **FedAvgRecipe** for simplified, Pythonic configuration.

### Examples

For examples of using [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) with [MONAI Bundle](https://monai.readthedocs.io/en/latest/mb_specification.html) and federated averaging ([FedAvg](https://arxiv.org/abs/1602.05629)), see the [examples](./examples/README.md).

## Requirements

Follow the instructions for setting up a [virtual environment](../../examples/README.md#set-up-a-virtual-environment):

```bash
pip install -r examples/spleen_ct_segmentation/requirements.txt
```

## Migration Note

**Deprecated:** The `monai_nvflare` package and `ClientAlgoExecutor`/`ClientAlgo` classes are deprecated. 

**Use instead:** NVFlare's [Client API](https://nvflare.readthedocs.io/en/main/programming_guide/execution_api_type.html#client-api) with [FedAvgRecipe](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.app_opt.pt.recipes.fedavg.html) for simpler, more maintainable code.

See the updated examples for the new pattern.
