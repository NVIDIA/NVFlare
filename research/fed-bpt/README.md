# FedBPT: Efficient Federated Black-box Prompt Tuning for Large Language Models

This example shows how to run [FedBPT](https://arxiv.org/abs/2310.01467) on an example task and the [FL simulator](https://nvflare.readthedocs.io/en/latest/user_guide/nvflare_cli/fl_simulator.html).

###### Abstract:
> Pre-trained language models (PLM) have revolutionized the NLP landscape, achieving stellar performances across diverse tasks. These models, while benefiting from vast training data, often require fine-tuning on specific data to cater to distinct downstream tasks. However, this data adaptation process has inherent security and privacy concerns, primarily when leveraging user-generated, device-residing data. Federated learning (FL) provides a solution, allowing collaborative model fine-tuning without centralized data collection. However, applying FL to finetune PLMs is hampered by challenges, including restricted model parameter access, high computational requirements, and communication overheads. This paper introduces Federated Black-box Prompt Tuning (FedBPT), a framework designed to address these challenges. FedBPT does not require the clients to access the model parameters. By focusing on training optimal prompts and utilizing gradient-free optimization methods, FedBPT reduces the number of exchanged variables, boosts communication efficiency, and minimizes computational and storage costs. Experiments highlight the framework's ability to drastically cut communication and memory costs while maintaining competitive performance. Ultimately, FedBPT presents a promising solution for efficient, privacy-preserving fine-tuning of PLM in the age of large language models.

## License
The code in this directory is released under Apache v2 License.
The code is extended from [Black-Box-Tuning (BBT)](https://github.com/txsun1997/Black-Box-Tuning) which is released under MIT License.
The models code is copied from the [transformers](https://github.com/huggingface/transformers) library.

## 1. Setup
We recommend creating a [conda environment](https://www.anaconda.com) following [BBT](https://github.com/txsun1997/Black-Box-Tuning#prepare-your-environment) 
with the addition of installing NVFlare for running federated learning and some other updates:
```commandline
conda create --name fedbpt python=3.12
conda activate fedbpt
pip install -r requirements.txt
pip install -e ../..
```

## 2. Run a federated learning experiment
The example uses `job.py` to create and run the NVFlare job without registering global job templates.
We utilize the [SST-2 dataset](https://huggingface.co/datasets/stanfordnlp/sst2) and the RoBerTa-large model for training.
```commandline
N_CLIENTS=10
SEED=1234
python job.py \
  --num_clients ${N_CLIENTS} \
  --num_rounds 200 \
  --seed ${SEED} \
  --task_name sst2 \
  --n_prompt_tokens 50 \
  --intrinsic_dim 500 \
  --k_shot 200 \
  --device cuda:0 \
  --loss_type ce \
  --cat_or_add add \
  --local_iter 8 \
  --num_users ${N_CLIENTS} \
  --iid 1 \
  --local_popsize 5 \
  --perturb 1 \
  --model_name roberta-large \
  --eval_clients site-1 \
  --llama_causal 1
```
By default, we only evaluate the global model on client `site-1` as in our setting, the global test set is shared by clients.

The following setting requires a GPU with at least 24 GB memory and enough system memory to run the clients in parallel (we recommend at least 40 GB).
For a system with less resources, you can set `--threads` to be a lower number and simulate the clients running sequentially.
```commandline
python job.py --num_clients ${N_CLIENTS} --threads 2 --gpu 0 --workspace /tmp/nvflare/fedbpt
```
If you have more GPUs available on your system, you can use `--gpu` to run clients on different GPUs in parallel.

To export the job without running the simulator, add `--export`:
```commandline
python job.py --export --export-dir ./jobs --num_clients ${N_CLIENTS} --num_rounds 200 --seed ${SEED}
```
This writes the job to `./jobs/fedbpt`.

## 3. Example results
The training results showing the global testing accuracy over 200 rounds is shown below. 
The global learnt prompt using FedBPT achieves an accuracy of 0.8761 on the SST-2 test set. 
<img src="./figs/global_test_acc.png" alt="FedBPT results" width="600"/>

## Citation

> Sun, Jingwei, et al. "FedBPT: Efficient Federated Black-box Prompt Tuning for Large Language Models." arXiv preprint arXiv:2310.01467 (2023).

BibTeX
```
@article{sun2023fedbpt,
  title={FedBPT: Efficient Federated Black-box Prompt Tuning for Large Language Models},
  author={Sun, Jingwei and Xu, Ziyue and Yin, Hongxu and Yang, Dong and Xu, Daguang and Chen, Yiran and Roth, Holger R},
  journal={arXiv preprint arXiv:2310.01467},
  year={2023}
}
```
