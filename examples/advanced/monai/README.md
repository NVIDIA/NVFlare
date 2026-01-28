# MONAI Integration Examples

## Objective
Integration with [MONAI](https://project-monai.github.io/) for federated learning using NVFlare's Client API and FedAvgRecipe.

### Goals:
Enable the use of MONAI [bundles](https://monai.readthedocs.io/en/latest/bundle.html) from the MONAI [model zoo](https://github.com/Project-MONAI/model-zoo) or custom configurations with NVFlare using modern Client API patterns.

## Description
MONAI allows the definition of AI models using the "bundle" concept for easy experimentation and sharing. These examples demonstrate how to use MONAI bundles in federated learning scenarios using NVFlare's **Client API** and **FedAvgRecipe** for simplified, Pythonic configuration.

## Requirements

Follow the instructions for setting up a [virtual environment](../../README.md#set-up-a-virtual-environment):

```bash
# For MedNIST example
pip install -r mednist/requirements.txt

# For Spleen CT Segmentation example
pip install -r spleen_ct_segmentation/requirements.txt
```

## Examples

### [Converting MONAI Code to Federated Learning](./mednist/README.md)

Tutorial showing how to run an end-to-end classification pipeline with MONAI and deploy it in federated learning using NVFlare's Client API and Job Recipes.

**Key Features:**
- Direct MONAI training without MonaiAlgo
- Manual data loading and model training
- TensorBoard and MLflow experiment tracking
- Classification task on MedNIST dataset

### [Federated Learning for 3D Spleen CT Segmentation](./spleen_ct_segmentation/README.md)

Example using NVFlare with [MONAI Bundle](https://monai.readthedocs.io/en/latest/mb_specification.html) for federated averaging ([FedAvg](https://arxiv.org/abs/1602.05629)) training and federated statistics collection using simulation.

**Key Features:**
- FedAvgRecipe for simplified job configuration
- Client API with MonaiAlgo for bundle management
- FedStatsRecipe for federated statistics
- TensorBoard and MLflow experiment tracking
- 3D medical image segmentation with MONAI bundles

### Client API Pattern

The Client API provides a simple, Pythonic interface for federated learning:

```python
import nvflare.client as flare

flare.init()

while flare.is_running():
    input_model = flare.receive()
    # Train your model
    output_model = flare.FLModel(params=updated_weights)
    flare.send(output_model)
```

### Recipe Pattern

Recipes provide high-level abstractions for common federated learning workflows:

```python
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe

recipe = FedAvgRecipe(
    name="my_fedavg_job",
    min_clients=2,
    num_rounds=100,
    train_script="client.py",
    train_args="--local_epochs 5"
)
```

## Quick Start

1. Choose an example based on your use case:
   - **New to federated learning?** Start with [mednist](./mednist/README.md)
   - **Using MONAI bundles?** See [spleen_ct_segmentation](./spleen_ct_segmentation/README.md)

2. Follow the example's README for detailed setup and execution instructions

3. Experiment with the code and adapt it to your own datasets and models

## Resources

- [MONAI Documentation](https://monai.readthedocs.io/)
- [MONAI Model Zoo](https://github.com/Project-MONAI/model-zoo)
- [MONAI Bundles](https://monai.readthedocs.io/en/latest/bundle.html)
- [NVFlare Client API Documentation](https://nvflare.readthedocs.io/en/main/programming_guide/execution_api_type.html#client-api)
- [FedAvgRecipe API](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.app_opt.pt.recipes.fedavg.html)
- [FedStatsRecipe API](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.recipe.fedstats.html)
- [Recipe Pattern Guide](https://nvflare.readthedocs.io/en/main/programming_guide/fed_job_api.html)
