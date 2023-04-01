# Federated Natural Language Processing

## Introduction 
This example shows how to use [NVIDIA FLARE](https://nvidia.github.io/NVFlare) for Natural Language Processing (NLP) tasks.
It uses a [BERT](https://github.com/google-research/bert) model from [huggingface](https://huggingface.co/bert-base-uncased)  on a Named Entity Recognition (NER) task using the [NCBI disease](https://pubmed.ncbi.nlm.nih.gov/24393765/) dataset. 
In this example, we select [BERT-base-uncased](https://huggingface.co/bert-base-uncased) model. 

This example is adapted from this [repo](https://github.com/PL97/federated-multi-modality-learning/). 

## Setup
Install required packages for training
```
pip install --upgrade pip
pip install -r ./requirements.txt
```

## Download and Preprocess Data 

The raw data can be accessed from [official page](https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/). 
In this example, we use the preprocessed csv-files, which can be downloaded [here](https://drive.google.com/drive/folders/13wROtEAnMgWpLMIGHB5CY1BQ1Xe2XqhG). Please download three files `train.csv`, `dev.csv`, and `test.csv`.

We then use the preprocessed data to generate random splits for a 4-client experiment. 
Please modify the `DATASET_ROOT` below to point to folder containing the four downloaded csv-files.
```commandline
bash prepare_data.sh DATASET_ROOT
```
The expected output is
```
(7594, 5) (2531, 5)
(5063, 5) (2531, 5)
(2532, 5) (2531, 5)
(2532, 5) (2532, 5)
(950, 5) (316, 5)
(634, 5) (316, 5)
(318, 5) (316, 5)
(318, 5) (318, 5)
```
The task here is to categorize each word in the text into three classes specified by the label. For example, the sentence 
`Recent progress has resulted in part of the gene mutated in Duchenne and the milder Becker muscular dystrophies being cloned and has suggested that the gene itself extends over 1 , 000 to 2 , 000 kilobases ( kb ) .` into label vector `O O O O O O O O O O O B I I I I I I O O O O O O O O O O O O O O O O O O O O O O O`. `B` marks the beginning of an entity, `I` marks each entity word, and `O` represents other words.
Let's take a closer look at the word-label correspondence:
![data sample](./figs/sample.png)
As shown above, the task is to capture the keywords related to medical findings.

## Run automated experiments
We use the NVFlare [simulator](https://nvflare.readthedocs.io/en/latest/user_guide/fl_simulator.html) to run the FL training.
Set `PYTHONPATH` to include custom files of this example:
```
export PYTHONPATH=${PWD}
```
### Prepare local configs
Please modify the `DATASET_ROOT` within [config_fed_client.json](./jobs/bert_ncbi/app/config/config_fed_client.json)
### Use NVFlare simulator to run the experiments
We use the NVFlare simulator to run the FL training experiments, following the pattern:
```
nvflare simulator jobs/[job] -w ${workspace_path}/[job] -c [clients] -gpu [gpu] -t [thread]
```
`[job]` is the experiment job that will be submitted for the FL training. 
In this example, this is `bert_ncbi`.  
The combination of `-c` and `-gpu`/`-t` controls the resource allocation. 

## Results on four clients 
In this example, we run four clients on 2 GPUs with 4 threads. The minimum GPU memory requirement is 10 GB per GPU. We put the workspace in `/tmp` folder
```
nvflare simulator jobs/bert_ncbi -w /tmp/nvflare/workspaces/bert_ncbi -n 4 -gpu 0,1,0,1
```

### Validation curve on each site
In this example, each client computes their validation scores using their own
validation set. We recorded the loss, F1 score, precision, and recall. 
The curves can be viewed with TensorBoard.

The TensorBoard curves for 50 epochs (50 FL rounds, 1 local epoch per round) during training are shown below:
![All training curve](./figs/training.png)

### Testing score
The testing score is computed for the global model over the testing set.
We provide a script for performing validation on testing data. 
Please modify the `DATASET_ROOT` below:
```
bash test_global_model.sh DATASET_ROOT
```
The test precision is 0.96, recall is 0.97, and F1 score is 0.96.
