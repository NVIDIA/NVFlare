# LLM Tuning via HuggingFace SFT/PEFT APIs
This example shows how to use [NVIDIA FLARE](https://nvidia.github.io/NVFlare) for Large Language Models (LLMs) tuning tasks. It illustrates how to adapt a local training script with [HuggingFace](https://huggingface.co/) trainer to NVFlare.

## Introduction 
This example illustrates both supervised fine-tuning (SFT) and parameter-efficient fine-tuning (PEFT) using the [SFT Trainer](https://huggingface.co/docs/trl/sft_trainer) from [HuggingFace](https://huggingface.co/) with [PEFT library](https://github.com/huggingface/peft).

We used the [Llama-3.2-1B model](https://huggingface.co/meta-llama/Llama-3.2-1B) to showcase the functionality of federated SFT and PEFT, allowing HuggingFace models to be trained and adapted with NVFlare. All other models from HuggingFace can be easily adapted following the same steps.

For PEFT, we used LoRA method, other PEFT methods (e.g. p-tuning, prompt-tuning) can be easily adapted as well by modifying the configs following [PEFT](https://github.com/huggingface/peft) examples.

We would like to showcase three key points in this example:
- Adapt local HuggingFace training scripts, both SFT and PEFT, to federated application
- Handling large model weights (~6 GB for Llama-3.2-1B model with float32 precision for communication), which is beyond protobuf's 2 GB hard limit. It is supported by NVFlare infrastructure via streaming, and does not need any code change.
- Use NVFlare's filter functionality to enable model quantization and precision conversion for communication, which can significantly reduce the message size and is thus important for communicating LLM updates.  

We conducted these experiments on a single 48GB RTX 6000 Ada GPU. 

To use Llama-3.2-1B model, please request access to the model here https://huggingface.co/meta-llama/Llama-3.2-1B and login with an access token using huggingface-cli.

## Setup
Please make sure you set up virtual environment following [example root readme](../../README.md).
Install additional requirements (if you already have a specific version of nvflare installed in your environment, you may want to remove nvflare in the requirements to avoid reinstalling nvflare):
```
python3 -m pip install -r requirements.txt
```
Git LFS is also necessary for downloads, please follow the steps in this [link](https://github.com/git-lfs/git-lfs/blob/main/INSTALLING.md).

## Data Preparation
We download and preprocess (consistent with our [NeMo example](../../../integration/nemo/examples/supervised_fine_tuning/README.md), we follow the same preprocessing steps).
```
mkdir dataset
cd dataset
git clone https://huggingface.co/datasets/tatsu-lab/alpaca
git clone https://huggingface.co/datasets/databricks/databricks-dolly-15k
git clone https://huggingface.co/datasets/OpenAssistant/oasst1
cd ..
mkdir dataset/dolly
python ./utils/preprocess_dolly.py --training_file dataset/databricks-dolly-15k/databricks-dolly-15k.jsonl --output_dir dataset/dolly
python ./utils/preprocess_alpaca.py --training_file dataset/alpaca/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet --output_dir dataset/alpaca
python ./utils/preprocess_oasst1.py --training_file dataset/oasst1/data/train-00000-of-00001-b42a775f407cee45.parquet --validation_file dataset/oasst1/data/validation-00000-of-00001-134b8fd0c89408b6.parquet --output_dir dataset/oasst1
```

## Adaptation of Centralized Training Script to Federated
To illustrate the adaptation process, we use a single dataset [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k).
### One-call training
Centralized trainings, as the baseline for comparison with other results, are done with the following command:
```
python3 ./utils/hf_sft_peft.py --output_path ./workspace/dolly_cen_sft --train_mode SFT
python3 ./utils/hf_sft_peft.py --output_path ./workspace/dolly_cen_peft --train_mode PEFT
```

### Adaptation Step 1: iterative training
To adapt the centralized training script to federated application, we first need to "break" the single call to `trainer.train()` into iterative calls, one for each round of training.
For this purpose, we provided `utils/hf_sft_peft_iter.py` as an example, which is a modified version of `utils/hf_sft_peft.py`.
Their differences are highlighted below:

![diff](./figs/diff.png)

Note that the `trainer.train()` call is replaced by a `for` loop, and the three training epochs becomes three rounds, one epoch per round. 

This setting (1 epoch per round) is for simplicity of this example. In practice, we can set the number of rounds and local epoch per round according to the needs: e.g. 2 rounds with 2 epochs per round will result in 4 training epochs in total.

At the beginning of each round, we intentionally load a fixed model weights saved at the beginning, over-writing the previous round's saved model weights, then call `trainer.train(resume_from_checkpoint=True)` with `trainer.args.num_train_epochs` incremented by 1 so that previous logging results are not overwritten. 

The purpose of doing so is to tell if the intended weights are succesfully loaded at each round. Without using a fixed starting model, even if the model weights are not properly loaded, the training loss curve will still follow the one-call result, which is not what we want to see. 

If the intended model weights (serving as the starting point for each round, the "global model" for FL use case) is properly loaded, then we shall observe a "zig-zag" pattern in the training loss curve. This is because the model weights are reset to the same starting point at the beginning of each round, in contrast to the one-shot centralized training, where the model weights are updated continuously, and the training loss curve should follow an overall decreasing trend.

To run iterative training, we use the following command:
``` 
python3 ./utils/hf_sft_peft_iter.py --output_path ./workspace/dolly_cen_sft_iter --train_mode SFT
python3 ./utils/hf_sft_peft_iter.py --output_path ./workspace/dolly_cen_peft_iter --train_mode PEFT
```

The SFT curves are shown below, black for single call, blue for iterative. We can see the "zig-zag" pattern in the iterative training loss curve.
![sft](./figs/cen_sft.png)

Similar patterns can be observed from the PEFT curves, purple for single call, green for iterative.
![peft](./figs/cen_peft.png)

### Adaptation Step 2: federated with NVFlare
Once we have the iterative training script ready with "starting model" loading capability, it can be easily adapted to a NVFlare trainer by using [Client API](../../hello-world/ml-to-fl/pt/README.md).

The major code modifications are for receiving and returning the global model (replacing the constant one used by iterative training), as shown below:

![diff](./figs/diff_fl_1.png)
![diff](./figs/diff_fl_2.png)

### Federated Training Results
We run the federated training on a single client using NVFlare Simulator via [JobAPI](https://nvflare.readthedocs.io/en/main/programming_guide/fed_job_api.html).
```
python3 sft_job.py --client_ids dolly --data_path ${PWD}/dataset --workspace_dir ${PWD}/workspace/hf_sft --job_dir ${PWD}/workspace/jobs/hf_sft --train_mode SFT 
python3 sft_job.py --client_ids dolly --data_path ${PWD}/dataset --workspace_dir ${PWD}/workspace/hf_peft --job_dir ${PWD}/workspace/jobs/hf_peft --train_mode PEFT 
```
The SFT curves are shown below, black for centralized results, magenta for FL training. With some training randomness, the two SFT training loss curves align with each other. 
![sft](./figs/fl_sft.png)

Similar patterns can be observed from the PEFT curves, purple for centralized results, orange for FL training. Alignment better than SFT can be observed.
![peft](./figs/fl_peft.png)

## Model Quantization for Communication
In the above example, we used numpy in float32 for communication. To reduce the message size, we can use model precision conversion and quantization 
from float32 to 16-bit, 8-bit, and 4-bit for communication. Quantization is enabled by NVFlare's [filter mechanism](https://nvflare.readthedocs.io/en/main/programming_guide/filters.html). We can use the following command to run the federated training with model quantization.
16-bit is a direct precision conversion, while 8-bit, 4-bit quantization is performed by [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes/tree/main).
Note that 4-bit quantizations (`fp4` or `nf4`) need device support.
```
python3 sft_job.py --client_ids dolly --data_path ${PWD}/dataset --workspace_dir ${PWD}/workspace/hf_sft_16 --job_dir ${PWD}/workspace/jobs/hf_sft_16 --train_mode SFT --quantize_mode float16
python3 sft_job.py --client_ids dolly --data_path ${PWD}/dataset --workspace_dir ${PWD}/workspace/hf_sft_8 --job_dir ${PWD}/workspace/jobs/hf_sft_8 --train_mode SFT --quantize_mode blockwise8
python3 sft_job.py --client_ids dolly --data_path ${PWD}/dataset --workspace_dir ${PWD}/workspace/hf_sft_fp4 --job_dir ${PWD}/workspace/jobs/hf_sft_fp4 --train_mode SFT --quantize_mode float4
python3 sft_job.py --client_ids dolly --data_path ${PWD}/dataset --workspace_dir ${PWD}/workspace/hf_sft_nf4 --job_dir ${PWD}/workspace/jobs/hf_sft_nf4 --train_mode SFT --quantize_mode normfloat4
```
The SFT curves are shown below, magenta for centralized results, others for FL training with quantization. We can see it achieves similar alignment comparing to centralized result with training randomness (similar to previous figure).
![sft](./figs/fl_sft_comp.png)

These results show that model precision conversion / quantization does not significantly impact the training while reducing the message size to 1/2, 1/4, and even 1/8, which can significantly reduce the message size, making it crucial for transmitting LLM updates.

For message reduce, from float32 to 16-/8-/4-bit, the message size (in MB) of Llama-3.2-1B model are reduced to: 

| Quantization      | Raw Model Size | Quantized Model Size | Quantization Meta Size |
|-------------------|----------------|----------------------|------------------------|
| float16           | 5716.26        | 2858.13              | 0.00                   |
| blockwise8        | 5716.26        | 1429.06              | 1.54                   |
| float4            | 5716.26        | 714.53               | 89.33                  |
| normalized float4 | 5716.26        | 714.53               | 89.33                  |

Note that quantization will generate additional meta data, which can be significant for 4-bit cases.

## Model Communication with Tensor
In addition, since the model is trained with bf16, instead of first converting to numpy in float32, we can directly communicate with tensor in bf16 to avoid the message size inflation due to the conversion. 
We can use the following command to run the federated training with direct tensor communication.
```
python3 sft_job.py --client_ids dolly --data_path ${PWD}/dataset --workspace_dir ${PWD}/workspace/hf_sft_tensor --job_dir ${PWD}/workspace/jobs/hf_sft_tensor --train_mode SFT --message_mode tensor
```
Similarly, quantization can be applied to tensor communication as well.
```
python3 sft_job.py --client_ids dolly --data_path ${PWD}/dataset --workspace_dir ${PWD}/workspace/hf_sft_tensor_fp4 --job_dir ${PWD}/workspace/jobs/hf_sft_tensor_fp4 --train_mode SFT --message_mode tensor --quantize_mode float4
```
In this case, since the tensor is in bf16, and the quantization reduces it to float4, the message size change is thus:
```
Before quantization: 2858.13 MB. After quantization: 714.53 MB with meta: 89.33 MB.
```

## Federated Training with Multiple Clients
With the above example, we can easily extend the federated training to multiple clients. We can use the following command to run the federated training with multiple clients:
```
python3 sft_job.py --client_ids dolly alpaca oasst1 --data_path ${PWD}/dataset --workspace_dir ${PWD}/workspace/hf_sft_multi --job_dir ${PWD}/workspace/jobs/hf_sft_multi --train_mode SFT --threads 1
```

For comparison, we run the other two sites in centralized training mode:
```
python3 ./utils/hf_sft_peft.py --data_path_train ./dataset/alpaca/training.jsonl --data_path_valid ./dataset/alpaca/validation.jsonl --output_path ./workspace/alpaca_cen_sft --train_mode SFT
python3 ./utils/hf_sft_peft.py --data_path_train ./dataset/oasst1/training.jsonl --data_path_valid ./dataset/oasst1/validation.jsonl --output_path ./workspace/oasst1_cen_sft --train_mode SFT
```

The training loss curves are shown below:

Dolly:
![sft](./figs/fl_sft_dolly.png)
Alpaca:
![sft](./figs/fl_sft_alpaca.png)
Oasst1:
![sft](./figs/fl_sft_oasst1.png)

As shown, federated training with multiple clients (lines with three sections) can achieve comparable or better results w.r.t. training loss to individual site's centralized trainings (continuous curves), demonstrating the effectiveness of federated learning.

Similarly for PEFT, we can run the following command:
```
python3 ./utils/hf_sft_peft.py --data_path_train ./dataset/alpaca/training.jsonl --data_path_valid ./dataset/alpaca/validation.jsonl --output_path ./workspace/alpaca_cen_peft --train_mode PEFT
python3 ./utils/hf_sft_peft.py --data_path_train ./dataset/oasst1/training.jsonl --data_path_valid ./dataset/oasst1/validation.jsonl --output_path ./workspace/oasst1_cen_peft --train_mode PEFT
python3 sft_job.py --client_ids dolly alpaca oasst1 --data_path ${PWD}/dataset --workspace_dir ${PWD}/workspace/hf_peft_multi --job_dir ${PWD}/workspace/jobs/hf_peft_multi --train_mode PEFT --threads 1
```

The training loss curves are shown below:

Dolly:
![peft](./figs/peft_dolly.png)
Alpaca:
![peft](./figs/peft_alpaca.png)
Oasst1:
![peft](./figs/peft_oasst1.png)