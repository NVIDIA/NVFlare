# HuggingFace Integration

## Objective
Execute [HuggingFace SFT Trainer](https://huggingface.co/docs/trl/sft_trainer) in federated environments.

### Goals:

Allow HuggingFace models to be trained and adapted with NVFlare.

### Non-goals:

n/a

## Background
HuggingFace is widely used platform, especially for LLM training. It hosts lots of models and enables easy training via their standardized APIs.

## Description
NVFlare utilizes features from HuggingFace trainer, such as SFT with PEFT options training to run LLM tasks in federated environments.

### Examples

Both supervised fine-tuning (SFT) and parameter-efficient fine-tuning (PEFT) are supported.

## Requirements

### Install related package
```
pip install requirements.txt
```

### Installation in a virtual environment

If preferred to install dependencies locally, 
we recommend following the instructions for setting up a 
[virtual environment](../../examples/README.md#set-up-a-virtual-environment),
and using it in [JupyterLab](../../examples/README.md#notebooks) for running 
the notebooks in the NeMo integration examples.

Follow the NeMo installation steps [here](https://github.com/NVIDIA/NeMo#installation)
before installing the NeMo-NVFlare package.
