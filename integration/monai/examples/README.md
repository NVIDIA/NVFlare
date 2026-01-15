# Examples of MONAI-NVFlare Integration

### [Converting MONAI Code to Federated Learning](./mednist/README.md)
Tutorial showing how to run an end-to-end classification pipeline with MONAI and deploy it in federated learning using NVFlare's Client API and Job Recipes.

### [Federated Learning for 3D Spleen CT Segmentation](./spleen_ct_segmentation/README.md)
Example using NVFlare with [MONAI Bundle](https://monai.readthedocs.io/en/latest/mb_specification.html) for federated averaging ([FedAvg](https://arxiv.org/abs/1602.05629)) training and federated statistics collection using simulation.

Features:
- FedAvgRecipe for simplified job configuration
- Client API with MonaiAlgo for bundle management
- FedStatsRecipe for federated statistics
- TensorBoard and MLflow experiment tracking

## Architecture

All examples now use:
- **Client API**: Simple Python API for FL (vs. deprecated Executor pattern)
- **FedAvgRecipe**: Pythonic job configuration (vs. JSON configs)
- **MONAI Bundles**: Integration with bundle configs via MonaiAlgo
- **Tracking**: Built-in TensorBoard and MLflow support

No custom package (`monai_nvflare`) required.
