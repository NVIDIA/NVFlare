# Tensor Stream Example with Federated Learning

This example demonstrates how to use NVFlare's **Tensor Streaming** capabilities for efficient federated learning with large language models (LLMs). It showcases fine-tuning a GPT-2 model using the FedAvg algorithm with the IMDB dataset across multiple clients.

## Overview

The example uses:
- **Model**: GPT-2 (pretrained causal language model)
- **Dataset**: IMDB dataset for text classification
- **Algorithm**: Federated Averaging (FedAvg)
- **Training Framework**: Hugging Face Transformers with TRL (Transformer Reinforcement Learning) SFTTrainer
- **Key Feature**: Tensor Streaming for efficient communication of large model parameters

## What is Tensor Streaming?

Tensor Streaming is an NVFlare feature that optimizes the communication of large tensors (model parameters) between the server and clients. It's particularly useful for:
- Large language models with millions/billions of parameters
- Reducing memory overhead during model exchange
- Efficient bandwidth utilization in federated learning scenarios

## Project Structure

```
tensor-stream/
├── README.md           # This file
├── job.py             # Main script to define and execute the FL job
├── client.py          # Client-side training logic
├── trainer.py         # Standalone trainer (for testing)
├── model.py           # Model and tokenizer loading utilities
├── requirements.txt   # Python dependencies
└── results/          # Training outputs and checkpoints
```

## Prerequisites

- Python 3.8+
- NVFlare 2.7.1+ installed
- GPU (optional, but recommended for faster training)

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Ensure NVFlare is installed:

```bash
pip install nvflare==2.7.1
```

## Files Description

### `model.py`
Contains utility functions to load the GPT-2 model and tokenizer. The function `get_model()` loads a pretrained model from Hugging Face and configures the tokenizer.

### `trainer.py`
Provides shared training utilities used by both standalone and federated learning:
- `get_dataset()` - Loads the IMDB dataset
- `preprocess()` - Tokenizes and prepares text data
- `get_training_arguments()` - Creates SFTConfig with training parameters
- `train()` - Standalone training function for testing

### `client.py`
Implements the federated learning client logic:
- Imports shared utilities from `trainer.py`
- Receives global model from the server
- Fine-tunes the model locally using SFTTrainer
- Sends updated model back to the server
- Integrates with NVFlare's tracking system

### `job.py`
Using the FedAvg recipe creates the federated job:
- Configures the FedAvg recipe
- Sets up Tensor Streaming for both server and clients
- Defines simulation environment
- Executes the federated learning job

## Usage

### Running the Federated Learning Job

Execute the main job script with default parameters:

```bash
python job.py
```

### Customizing Parameters

You can customize the number of clients and training rounds:

```bash
python job.py --n_clients 3 --num_rounds 5
```

**Arguments:**
- `--n_clients`: Number of federated learning clients (default: 2)
- `--num_rounds`: Number of federated learning rounds (default: 2)

### Testing the Training Pipeline (Standalone)

To test the training pipeline without federated learning:

```bash
python trainer.py
```

This will train the model on the IMDB dataset locally, using the same configuration as the federated clients.

## Configuration Details

### Training Configuration

Training arguments are defined in `trainer.py` via the `get_training_arguments()` function:

```python
training_args = SFTConfig(
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    fp16=False,
    learning_rate=2e-4,
    loss_type="nll",
    lr_scheduler_type="cosine",
    gradient_checkpointing=True,
    dataset_text_field="text",
    output_dir=f"./results/{client_name}",
    report_to="none",
    use_cpu=not torch.cuda.is_available(),
)
```

This configuration is shared between standalone training and federated learning, ensuring consistency.

### Tensor Streaming Setup

The job uses `TensorServerStreamer` and `TensorClientStreamer` components:

```python
recipe.job.to_server(TensorServerStreamer(), "tensor_server_streamer")
recipe.job.to_clients(TensorClientStreamer(), "tensor_client_streamer")
```

These streamers handle the efficient transmission of large model tensors between server and clients.

## Output

After running the job, you'll find:
- Training results in `./results/{client_name}/` directories for each client
- Job status and results location printed to console
- TensorBoard logs (if enabled) for experiment tracking

## Monitoring

The example includes TensorBoard integration for experiment tracking:

```python
add_experiment_tracking(recipe, tracking_type="tensorboard")
```

To view training metrics, run:

```bash
tensorboard --logdir ./results
```

## Advanced Usage

### GPU Training

The code automatically detects GPU availability. To force CPU usage, modify the `use_cpu` parameter in `SFTConfig`:

```python
use_cpu=True  # Force CPU
```

### Dataset Customization

To use a different dataset, modify the `get_dataset()` function in `trainer.py`:

```python
def get_dataset() -> any:
    return load_dataset("your_dataset_name")
```

You may also need to adjust the `preprocess()` function and the `max_length` parameter to suit your dataset's characteristics.

### Modifying Training Parameters

To change training parameters, edit the `get_training_arguments()` function in `trainer.py`. Common modifications include:
- Batch size and gradient accumulation
- Learning rate and scheduler
- Number of epochs
- Output directory structure

## Troubleshooting

### Out of Memory Errors
- Reduce `per_device_train_batch_size` in `trainer.py`
- Increase `gradient_accumulation_steps` in `trainer.py`
- Reduce `max_length` in the `preprocess()` function (currently set to 128)
- Use CPU training by setting `use_cpu=True` in `get_training_arguments()`
- Consider using a smaller model variant

### Slow Training
- Enable GPU if available (detected automatically)
- Reduce dataset size for testing by slicing the dataset
- Reduce `max_length` in preprocessing to speed up tokenization

### Dataset Download Issues
- Ensure internet connectivity
- Check Hugging Face Hub status at https://status.huggingface.co
- Try downloading the dataset manually first using `datasets-cli`
- Check disk space, as IMDB dataset requires ~150MB

### Import Errors
If you encounter import errors related to `torch` in `trainer.py`, add the import:
```python
import torch
```
at the top of the file with other imports.

## Notes

- The example uses GPT-2 (small) by default, which has ~124M parameters
- Training on CPU will be significantly slower than GPU
- The IMDB dataset is automatically downloaded on first run
- Each client maintains its own output directory in `./results/`

## References

- [NVFlare Documentation](https://nvflare.readthedocs.io/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [TRL Library](https://huggingface.co/docs/trl/)
- [IMDB Dataset](https://huggingface.co/datasets/imdb)

## License

Copyright (c) 2025, NVIDIA CORPORATION. Licensed under the Apache License, Version 2.0.
