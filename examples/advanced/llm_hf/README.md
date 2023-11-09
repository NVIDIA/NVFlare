# LLM Tuning via HuggingFace SFT Trainer
This example shows how to use [NVIDIA FLARE](https://nvidia.github.io/NVFlare) for Large Language Models (LLMs) tuning tasks.

## Introduction 
This example illustrates both supervised fine-tuning (SFT) and parameter-efficient fine-tuning (PEFT) using the [SFT Trainer](https://huggingface.co/docs/trl/sft_trainer) from [HuggingFace](https://huggingface.co/).
We used the [Llama-2-7b model](https://huggingface.co/meta-llama/Llama-2-7b) to showcase the functionality of federated SFT and PEFT, allowing HuggingFace models to be trained and adapted with NVFlare. 
Mainly on two fronts:
- Adapt local HF training scripts to federated application, this will be show-cases by code comparison
- Handling large model weights (~26 GB for Llama-2-7b model), this is supported by NVFlare infrastructure, and does not need any code change.

## Setup
Please make sure you set up virtual environment following [example root readme](../../README.md).

## Data Preparation
We used [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) for this example. 
We first download and preprocess (to be consistent with our [NeMo example](../../../integration/nemo/examples/supervised_fine_tuning), we used the same preprocessing) the dataset:
```
mkdir dataset
cd dataset
git clone https://huggingface.co/datasets/databricks/databricks-dolly-15k
cd ..
mkdir dataset/dolly
python utils/preprocess_dolly.py --training_file dataset/databricks-dolly-15k/databricks-dolly-15k.jsonl --output_dir dataset/dolly
```

## Job Template
We reuse the job templates from [sag_pt](../../../job_templates/sag_pt/), let's set the job template path with the following command.
```bash
nvflare config -jt ../../../job_templates/
```
Then we can check the available templates with the following command.
```bash
nvflare job list_templates
```
We can see the "sag_pt" template is available, with which we further generate job configs for SFT and PEFT as:
```
nvflare job create -force -j "./jobs/hf_sft" -w "sag_pt" -sd "code" \
  -f meta.conf min_clients=1 \
  -f config_fed_client.conf app_script="llama_sft_fl.py" \
  -f config_fed_server.conf model_class_path="hf_sft_model.CausalLMModel" components[0].args.model.args.model_path="/model/llama-2-7b-hf" min_clients=1 num_rounds=3 key_metric="eval_loss" negate_key_metric=True  
```
and 
```
nvflare job create -force -j "./jobs/hf_peft" -w "sag_pt" -sd "code" \
  -f meta.conf min_clients=1 \
  -f config_fed_client.conf app_script="llama_peft_fl.py" \
  -f config_fed_server.conf model_class_path="hf_peft_model.CausalLMPEFTModel" components[0].args.model.args.model_path="/model/llama-2-7b-hf" min_clients=1 num_rounds=3 key_metric="eval_loss" negate_key_metric=True
```
For both client and server configs, we only set the necessary parameters for the SFT and PEFT tasks, and leave the rest to the default values.