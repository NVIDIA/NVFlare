# Embedding Model Tuning via SentenceTransformers Trainer
This example shows how to use [NVIDIA FLARE](https://nvidia.github.io/NVFlare) for embedding tuning tasks, a critical component of Retrieval-Augmented Generation (RAG). 

It illustrates how to adapt a local training script with [SentenceTransformers](https://github.com/UKPLab/sentence-transformers) trainer to NVFlare.

## Introduction 
[SentenceTransformers](https://sbert.net/) is a widely used framework for computing dense vector representations for texts. 
The models are based on transformer, achieving state-of-the-art performance in various tasks. 

One major application is to embed the text in vector space for later clustering and/or retrieval using similarity metrics.

This example illustrates a supervised fine-tuning (SFT) scheme for an embedding model with various training datasets.

## Setup
Please make sure you set up virtual environment following [example root readme](../../../README.md).
Install additional requirements (if you already have a specific version of nvflare installed in your environment, you may want to remove nvflare in the requirements to avoid reinstalling nvflare):
```
python3 -m pip install -r requirements.txt
```
Models and data will be loaded directly from Huggingface, so no need to download them manually.

## Centralized Training
### Single-session training
Centralized trainings, as the baseline for comparison with FL results, are done with the following command:
```
bash train_single_session.sh
```
### Pre: Launch Modes
Before we start adapting the local training script to federated application, we first need to understand the launch modes of NVFlare client API.
In our [client settings](../../../job_templates/sag_pt/config_fed_client.conf), we have two launch modes by switching the `--launch_once` flag:
* If launch_once is true, the SubprocessLauncher will launch an external process once for the whole job
* If launch_once is false, the SubprocessLauncher will launch an external process everytime it receives a task from server
So if it is false, the SubprocessLauncher will create new processes every round.
If it is true, the SubprocessLauncher will reuse the same process for all rounds.

Turning `launch_once` to `false` can be useful in some scenarios like quick prototyping, but for the application of LLM where setup stage can take significant resources, we would want to only setup once. Hence, the below steps are for `launch_once = true` scenario.

See [Client API](../../../hello-world/ml-to-fl/pt/README.md) for more details.

### Adaptation Step 1: iterative training
To adapt the centralized training script to federated application, under `launch_once = true` setting, we first need to "break" the single call to `trainer.train()` into iterative calls, one for each round of training.
For this purpose, we provided `utils/train_iterative.py` as an example, which is a modified version of `utils/train_single_session.py`.

In the iterative training script, the `trainer.train()` call is replaced by a `for` loop, and the training epochs are split into six rounds, `unit_train_epochs = 0.25` epoch per round, in total `0.25 * 6 = 1.5` epochs, same as single session setting. 

The first round is trained with `trainer.train()`, then from the second round, 
we call `trainer.train(resume_from_checkpoint=True)` with `args.num_train_epochs` incremented by `unit_train_epochs` to continue training from the last checkpoint.

To run iterative training, we use the following command:
``` 
bash train_iterative.sh
```

The training loss curves are shown below, single session and iterative scripts align with each other. 

![iter_single](./figs/iter_single.png)

### Adaptation Step 2: federated with NVFlare
Once we have the iterative training script ready with "starting model" loading capability, it can be easily adapted to a NVFlare trainer by using [Client API](../../hello-world/ml-to-fl/pt/README.md).

The major code modifications are for receiving the global model, set it as the starting point for each round's training, and returning the trained model after each local training round.

## Job for NVFlare FL Training
With the local training script ready, we can go ahead to generate the NVFlare job configs by reusing the job templates from [sag_pt](../../../job_templates/sag_pt/).

Let's set the job template path with the following command.
```bash
nvflare config -jt ./job_template/
```
Then we can check the available templates with the following command.
```bash
nvflare job list_templates
```
We can see the "sag_pt_deploy_map" template is available, with which we further generate job configs for embedding model training as:
```
nvflare job create -force \
  -j "/tmp/embed/nvflare/job" -w "sag_pt_deploy_map" -sd "code" \
  -f meta.conf min_clients=3 \
  -f app_1/config_fed_client.conf app_script="train_fl.py" app_config="--dataset_name nli" \
  -f app_2/config_fed_client.conf app_script="train_fl.py" app_config="--dataset_name squad" \
  -f app_3/config_fed_client.conf app_script="train_fl.py" app_config="--dataset_name quora" \
  -f app_server/config_fed_server.conf model_class_path="st_model.SenTransModel" components[0].args.model.args.model_name="microsoft/mpnet-base" min_clients=3 num_rounds=7 key_metric="eval_loss" negate_key_metric=True 
```


For both client and server configs, we only set the necessary task-related parameters tasks, and leave the rest to the default values.

## Federated Training
With the produced job, we run the federated training on a single client using NVFlare Simulator.
```
nvflare simulator -w /tmp/embed/nvflare/workspace -n 3 -t 3 /tmp/embed/nvflare/job
```

## Results
The evaluation on two test datasets - [stsb](https://huggingface.co/datasets/sentence-transformers/stsb) with embedding similarity evaluation, and [NLI](https://huggingface.co/datasets/sentence-transformers/all-nli) with triplet accuracy evaluation, are shown below.

 TrainData | STSB_pearson_cos | STSB_spearman_euc | NLI_cos_acc | NLI_euc_acc
--- |------------------|-------------------|-------------| ---
NLI | 0.7586           | 0.7895            | 0.8033      | 0.8045
Squad | 0.8206           | 0.8154            | 0.8051      | 0.8042
Quora | 0.8161           | 0.8121            | 0.7891      | 0.7854
All | 0.8497           | 0.8523            | 0.8426      | 0.8384
Federated | 0.8444           | 0.8368            | 0.8269      |  0.8246

As shown, the federated training results are better than individual site's, and can be close to the centralized training results, demonstrating the effectiveness of NVFlare in embedding model tuning tasks.