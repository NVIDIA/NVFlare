# ML to FL with TensorFlow

This example demonstrates federated learning with TensorFlow, supporting 2 training modes:

| Mode | Description | Requirements |
|------|-------------|--------------|
| `tf` | Standard TensorFlow | 1 GPU |
| `tf_multi` | TensorFlow with MirroredStrategy | 2+ GPUs |

## Quick Start

1. **Prepare data:**
```bash
bash ./prepare_data.sh
```

2. **Run federated learning:**
```bash
# Standard TensorFlow
python job.py --mode tf

# TensorFlow with MirroredStrategy (multi-GPU)
python job.py --mode tf_multi --launch_process
```

## Project Structure

```
tf/
├── job.py              # Unified job config (both modes)
├── model.py            # TFNet class
├── client.py           # Standard TensorFlow client
├── client_multi_gpu.py # Multi-GPU client with MirroredStrategy
├── prepare_data.sh
└── requirements.txt
```

## Command Line Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | Training mode: `tf`, `tf_multi` | `tf` |
| `--n_clients` | Number of clients | 2 |
| `--num_rounds` | Number of training rounds | 5 |
| `--use_tracking` | Enable TensorBoard tracking | False |
| `--launch_process` | Launch in external process | False |
| `--export_config` | Export job config only | False |

## Examples

```bash
# Basic TensorFlow with 3 clients, 10 rounds
python job.py --mode tf --n_clients 3 --num_rounds 10

# Multi-GPU with external process
python job.py --mode tf_multi --launch_process

# Export config for deployment
python job.py --mode tf --export_config
```

## Client Scripts

### `client.py` - Standard TensorFlow

Uses `nvflare.client` API:
```python
import nvflare.client as flare

flare.init()
while flare.is_running():
    input_model = flare.receive()
    
    # Load weights
    for k, v in input_model.params.items():
        model.get_layer(k).set_weights(v)
    
    # Train
    model.fit(train_images, train_labels, epochs=1)
    
    # Send back
    flare.send(flare.FLModel(
        params={layer.name: layer.get_weights() for layer in model.layers},
        metrics={"accuracy": accuracy}
    ))
```

### `client_multi_gpu.py` - Multi-GPU with MirroredStrategy

Uses TensorFlow's `MirroredStrategy` for multi-GPU training:
```python
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = TFNet()
    model.compile(...)
```

## Notes on Running with GPUs

By default, TensorFlow will attempt to allocate all available GPU memory. When running multiple clients, set these environment variables:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true TF_GPU_ALLOCATOR=cuda_malloc_async python job.py
```

## Requirements

We recommend using [NVIDIA TensorFlow docker](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow) for GPU support:

```bash
docker run --gpus=all -it --rm -v [path_to_NVFlare]:/NVFlare nvcr.io/nvidia/tensorflow:xx.xx-tf2-py3
pip install nvflare
```
