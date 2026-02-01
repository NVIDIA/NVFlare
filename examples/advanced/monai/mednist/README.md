## Converting MONAI Code to Federated Learning

This tutorial shows how simple it is to run an end-to-end classification pipeline with MONAI and deploy it in federated learning using NVFlare's **Client API**.

### 0. Install Requirements

```bash
pip install -r requirements.txt
```

### 1. Standalone Training with MONAI

[monai_101.ipynb](./monai_101.ipynb) is based on the [MONAI 101 classification tutorial](https://github.com/Project-MONAI/tutorials/blob/main/2d_classification/monai_101.ipynb) and demonstrates:

- Dataset download
- Data pre-processing  
- DenseNet-121 model definition and training
- Evaluation on test dataset

### 2. Federated Learning with MONAI

[monai_101_fl.ipynb](./monai_101_fl.ipynb) shows how to convert the standalone code into a federated learning scenario using NVFlare.

The example uses:
- **[Client API](https://nvflare.readthedocs.io/en/main/programming_guide/execution_api_type.html#client-api)**: Simple Python API for FL training
- **[FedAvg Algorithm](https://arxiv.org/abs/1602.05629)**: Federated averaging for model aggregation
- **Training Script**: [client.py](./client.py)

### Federated Learning Pattern

This example demonstrates the recommended NVFlare pattern:

```python
import nvflare.client as flare

# Initialize FL
flare.init()

# FL training loop
while flare.is_running():
    # Receive global model
    input_model = flare.receive()
    
    # Train locally
    # ... your training code ...
    
    # Send updated model
    flare.send(output_model)
```

For more complex scenarios with MONAI bundles, see the [spleen CT segmentation example](../spleen_ct_segmentation/README.md).

We use **FedAvgRecipe** for Pythonic job configuration and simply run

```
python job.py
```

to execute an FL simulation.
