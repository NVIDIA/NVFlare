# LLM Tuning via HuggingFace SFT Trainer
This example shows how to use [NVIDIA FLARE](https://nvidia.github.io/NVFlare) for Large Language Models (LLMs) tuning tasks. It illustrates how to adapt a local training script with [HuggingFace](https://huggingface.co/) trainer to NVFlare.

## Introduction 
This example illustrates both supervised fine-tuning (SFT) and parameter-efficient fine-tuning (PEFT) using the [SFT Trainer](https://huggingface.co/docs/trl/sft_trainer) from [HuggingFace](https://huggingface.co/) with [PEFT library](https://github.com/huggingface/peft).

We used the [Llama-3.2-1B model](https://huggingface.co/meta-llama/Llama-3.2-1B) to showcase the functionality of federated SFT and PEFT, allowing HuggingFace models to be trained and adapted with NVFlare. All other models from HuggingFace can be easily adapted following the same steps.

For PEFT, we used LoRA method, other PEFT methods (e.g. p-tuning, prompt-tuning) can be easily adapted as well by modifying the configs following [PEFT](https://github.com/huggingface/peft) examples.

We would like to showcase three key points in this example:
- Adapt local HuggingFace training scripts, both SFT and PEFT, to federated application
- Handling large model weights (~6 GB for Llama-3.2-1B model with float32 precision for communication), which is beyond protobuf's 2 GB hard limit. It is supported by NVFlare infrastructure via streaming, and does not need any code change.
- Use NVFlare's filter functionality to enable model compression and precision conversion for communication, which can significantly reduce the message size and is thus important for communicating LLM updates.  

We conducted these experiments on a single 48GB RTX 6000 Ada GPU. 

## Setup
Please make sure you set up virtual environment following [example root readme](../../README.md).
Install additional requirements (if you already have a specific version of nvflare installed in your environment, you may want to remove nvflare in the requirements to avoid reinstalling nvflare):
```
python3 -m pip install -r requirements.txt
```

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
python3 ./utils/hf_sft_peft.py --output_path ./workspace/llama-3.2-1b-dolly-cen_sft --mode 0
python3 ./utils/hf_sft_peft.py --output_path ./workspace/llama-3.2-1b-dolly-cen_peft --mode 1
```
### Pre: Launch Modes
Before we start adapting the local training script to federated application, we first need to understand the launch modes of NVFlare client API.
In our [client settings](../../../job_templates/sag_pt/config_fed_client.conf), we have two launch modes by switching the `--launch_once` flag:
* If launch_once is true, the SubprocessLauncher will launch an external process once for the whole job
* If launch_once is false, the SubprocessLauncher will launch an external process everytime it receives a task from server
So if it is false, the SubprocessLauncher will create new processes every round.
If it is true, the SubprocessLauncher will reuse the same process for all rounds.

Turning `launch_once` to `false` can be useful in some scenarios like quick prototyping, but for the application of LLM where setup stage can take significant resources, we would want to only setup once. Hence, the below steps are for `launch_once = true` scenario.

See [Client API](../../hello-world/ml-to-fl/pt/README.md) for more details.

### Adaptation Step 1: iterative training
To adapt the centralized training script to federated application, under `launch_once = true` setting, we first need to "break" the single call to `trainer.train()` into iterative calls, one for each round of training.
For this purpose, we provided `utils/hf_sft_peft_iter.py` as an example, which is a modified version of `utils/hf_sft_peft.py`.
Their differences are highlighted below:

![diff](./figs/diff_1.png)
![diff](./figs/diff_2.png)

Note that the `trainer.train()` call is replaced by a `for` loop, and the three training epochs becomes three rounds, one epoch per round. 

This setting (1 epoch per round) is for simplicity of this example. In practice, we can set the number of rounds and local epoch per round according to the needs: e.g. 2 rounds with 2 epochs per round will result in 4 training epochs in total.

At the beginning of each round, we intentionally load a fixed model weights saved at the beginning, over-writing the previous round's saved model weights, then call `trainer.train(resume_from_checkpoint=True)` with `trainer.args.num_train_epochs` incremented by 1 so that previous logging results are not overwritten. 

The purpose of doing so is to tell if the intended weights are succesfully loaded at each round. Without using a fixed starting model, even if the model weights are not properly loaded, the training loss curve will still follow the one-call result, which is not what we want to see. 

If the intended model weights (serving as the starting point for each round, the "global model" for FL use case) is properly loaded, then we shall observe a "zig-zag" pattern in the training loss curve. This is because the model weights are reset to the same starting point at the beginning of each round, in contrast to the one-shot centralized training, where the model weights are updated continuously, and the training loss curve should follow an overall decreasing trend.

To run iterative training, we use the following command:
``` 
python3 ./utils/hf_sft_peft_iter.py --output_path ./workspace/llama-3.2-1b-dolly-cen_sft-iter --mode 0
python3 ./utils/hf_sft_peft_iter.py --output_path ./workspace/llama-3.2-1b-dolly-cen_peft-iter --mode 1
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
We run the federated training on a single client using NVFlare Simulator via [JobAPI](../job_api/README.md).
```
python3 sft_job.py --data_path ${PWD}/dataset/dolly --workspace_dir ${PWD}/workspace/hf_sft --job_dir ${PWD}/workspace/jobs/hf_sft --train_mode 0 
python3 sft_job.py --data_path ${PWD}/dataset/dolly --workspace_dir ${PWD}/workspace/hf_peft --job_dir ${PWD}/workspace/jobs/hf_peft --train_mode 1 
```
The SFT curves are shown below, black for centralized results, magenta for FL training. With some training randomness, the two PEFT training loss curves align with each other. 
![sft](./figs/fl_sft.png)

Similar patterns can be observed from the PEFT curves, purple for centralized results, orange for FL training. Alignment better than SFT can be observed.
![peft](./figs/fl_peft.png)

## Model Precision Conversion for Communication
In the above example, we used float32 for communication. To reduce the message size, we can use model precision conversion for communication. Model conversion is enabled by NVFlare's [filter mechanism](https://nvflare.readthedocs.io/en/main/programming_guide/filters.html). We can use the following command to run the federated training with model precision conversion:
```
python3 sft_job_compress.py --data_path ${PWD}/dataset/dolly --workspace_dir ${PWD}/workspace/hf_sft_compress --job_dir ${PWD}/workspace/jobs/hf_sft_compress --train_mode 0
python3 sft_job_compress.py --data_path ${PWD}/dataset/dolly --workspace_dir ${PWD}/workspace/hf_peft_compress --job_dir ${PWD}/workspace/jobs/hf_peft_compress --train_mode 1
```
The SFT curves are shown below, black for centralized results, yellow for FL training with compression. We can see it achieves similar alignment with centralized result.
![sft](./figs/fl_sft_comp.png)

Similar patterns can be observed from the PEFT curves, purple for centralized results, black for FL training with compression.
![peft](./figs/fl_peft_comp.png)

These results show that model precision conversion does not significantly impact the training while reducing the message size and is important for communicating LLM updates.
We can also see that the PEFT training loss curves are more aligned than SFT, which is consistent with the results from the centralized training.

For message reduce, since we convert float32 to float16, the message size is reduced by 2 times. The message size is reduced from 6GB to 3GB for Llama-3.2-1B model according to the log.
```shell
Compressed all 147 params Before compression: 5993930752 bytes After compression: 2996965376 bytes
```
