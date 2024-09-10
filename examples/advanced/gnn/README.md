# Federated GNN on Graph Dataset using Inductive Learning
In this example, we will demonstrate how to train a classification model using Graph Neural Network (GNN). 

### Background of Graph Neural Network
Graph Neural Networks (GNNs) show a promising future in research and industry, with potential applications in various domains, including social networks, e-commerce, recommendation systems, and more.
GNNs excel in learning, modeling, and leveraging complex relationships within graph-structured data. They combine local and global information, incorporate structural knowledge, adapt to diverse tasks, handle heterogeneous data, support transfer learning, scale for large graphs, offer interpretable insights, and achieve impressive performance. 

### Tasks
In this example, we provide two tasks:
1. **Protein Classification**:
The aim is to classify protein roles based on their cellular functions from gene ontology. The dataset we are using is PPI
([protein-protein interaction](http://snap.stanford.edu/graphsage/#code)) graphs, where each graph represents a specific human tissue. Protein-protein interaction (PPI) dataset is commonly used in graph-based machine-learning tasks, especially in the field of bioinformatics. This dataset represents interactions between proteins as graphs, where nodes represent proteins and edges represent interactions between them.
2. **Financial Transaction Classification**:
The aim is to classify whether a given transaction is licit or illicit. For this financial application, we use the [Elliptic++](https://github.com/git-disl/EllipticPlusPlus) dataset. It consists of 203k Bitcoin transactions and 822k wallet addresses to enable both the detection of fraudulent transactions and the detection of illicit addresses (actors) in the Bitcoin network by leveraging graph data. For more details, please refer to this [paper](https://arxiv.org/pdf/2306.06108.pdf).

### Algorithm
Both tasks are for node classification. We used the inductive representation learning method [GraphSAGE](https://arxiv.org/pdf/1706.02216.pdf) based on [Pytorch Geometric](https://github.com/pyg-team/pytorch_geometric)'s examples. 
[Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/)  is  a library built upon PyTorch to easily write and train Graph Neural Networks (GNNs) for a wide range of applications related to structured data.

For protein classification task, we used it in an unsupervised manner, following [PyG's unsupervised PPI example](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/graph_sage_unsup_ppi.py).
For financial transaction classification task, we used it in a supervised manner, directly using the node labels with supervised classification loss.

Since the inductive learning mode is being used, the locally learnt model (a representation encoding / classification network) is irrelevant to the candidate graph, we are able to use the basic [FedAvg](https://arxiv.org/abs/1602.05629) as the federated learning algorithm. The workflow is Scatter and Gather (SAG).

###  Experiments
####  Install NVIDIA FLARE
Follow the [Installation](../../getting_started/README.md) instructions.
Install additional requirements (if you already have a specific version of nvflare installed in your environment, you may want to remove nvflare in the requirements to avoid reinstalling nvflare):
```
python3 -m pip install -r requirements.txt
```
To support functions of PyTorch Geometric necessary for this example, we need extra dependencies. Please refer to [installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) and install accordingly:
```
python3 -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
```

#### Job Template
We reuse the job templates from [sag_gnn](../../../job_templates/sag_gnn), let's set the job template path with the following command.
```bash
nvflare config -jt ../../../job_templates/
```
Then we can check the available templates with the following command.
```bash
nvflare job list_templates
```
We can see the "sag_gnn" template is available

#### Protein Classification
The PPI dataset is directly available via torch_geometric library, we randomly split the dataset to 2 subsets, one for each client (`--client_id 1` and `--client_id 2`).
First, we run the local training on each client, as well as the whole dataset with `--client_id 0`.
```
python3 code/graphsage_protein_local.py --client_id 0
python3 code/graphsage_protein_local.py --client_id 1
python3 code/graphsage_protein_local.py --client_id 2 
```
Then, we create NVFlare job based on GNN template.
```
nvflare job create -force -j "/tmp/nvflare/jobs/gnn_protein" -w "sag_gnn" -sd "code" \
  -f app_1/config_fed_client.conf app_script="graphsage_protein_fl.py" app_config="--client_id 1 --epochs 10" \
  -f app_2/config_fed_client.conf app_script="graphsage_protein_fl.py" app_config="--client_id 2 --epochs 10" \
  -f app_server/config_fed_server.conf num_rounds=7 key_metric="validation_f1" model_class_path="torch_geometric.nn.GraphSAGE" components[0].args.model.args.in_channels=50  components[0].args.model.args.hidden_channels=64 components[0].args.model.args.num_layers=2 components[0].args.model.args.out_channels=64  
```
For client configs, we set client_ids for each client, and the number of local epochs per round for each client's local training. 

For server configs, we set the number of rounds for federated training, the key metric for model selection, and the model class path with model hyperparameters.

With the produced job, we run the federated training on both clients via FedAvg using the NVFlare Simulator.
```
nvflare simulator -w /tmp/nvflare/gnn/protein_fl_workspace -n 2 -t 2 /tmp/nvflare/jobs/gnn_protein
```

#### Financial Transaction Classification
We first download the Elliptic++ dataset to `/tmp/nvflare/datasets/elliptic_pp` folder. In this example, we will use the following three files:
- `txs_classes.csv`: transaction id and its class (licit or illicit)
- `txs_edgelist.csv`: connections for transaction ids 
- `txs_features.csv`: transaction id and its features
Then, we run the local training on each client, as well as the whole dataset. Again, `--client_id 0` uses all data.
```
python3 code/graphsage_finance_local.py --client_id 0
python3 code/graphsage_finance_local.py --client_id 1
python3 code/graphsage_finance_local.py --client_id 2 
```
Similarly, we create NVFlare job based on GNN template.
```
nvflare job create -force -j "/tmp/nvflare/jobs/gnn_finance" -w "sag_gnn" -sd "code" \
  -f app_1/config_fed_client.conf app_script="graphsage_finance_fl.py" app_config="--client_id 1 --epochs 10" \
  -f app_2/config_fed_client.conf app_script="graphsage_finance_fl.py" app_config="--client_id 2 --epochs 10" \
  -f app_server/config_fed_server.conf num_rounds=7 key_metric="validation_auc" model_class_path="pyg_sage.SAGE" components[0].args.model.args.in_channels=165  components[0].args.model.args.hidden_channels=256 components[0].args.model.args.num_layers=3 components[0].args.model.args.num_classes=2  
```
And with the produced job, we run the federated training on both clients via FedAvg using the NVFlare Simulator.
```
nvflare simulator -w /tmp/nvflare/gnn/finance_fl_workspace -n 2 -t 2 /tmp/nvflare/jobs/gnn_finance
```

###  Results
We can access the results inside the local and fl workspaces under `/tmp/nvflare/gnn`.
#### Color Scheme
Local trainings: 
- Black curve: whole dataset
- Green curve: client 1
- Purple curve: client 2

Federated learning: 
- Blue curve: client 1
- Red curve: client 2

#### Protein Classification
The training losses is shown below: 
![All training curves](./figs/protein_train_loss.png)

We can notice the "bumps" due to global model aggregation and syncing.

The validation scores is shown below:
![All validation curves](./figs/protein_val_f1.png)

Since we validate the global model for each round, the two clients' validation scores are the same (blue and red overlapping). As shown in the figure, while below the centralized training result using whole dataset, federated learning can help the training by achieving better scores as compared with local training using individual site's data only.

#### Financial Transaction Classification
The training losses is shown below: 
![All training curves](./figs/finance_train_loss.png)

The validation scores is shown below:
![All validation curves](./figs/finance_val_auc.png)

Since we validate the global model for each round, the two clients' validation scores are the same (blue and red overlapping). As shown in the figure, federated learning can help the training by achieving better scores as compared with local training using individual site's data only, and comparable to the centralized training result using whole dataset


## Citation for Elliptic++ Dataset

> Youssef Elmougy and Ling Liu. 2023. Demystifying Fraudulent Transactions and Illicit Nodes in the Bitcoin Network for Financial Forensics. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’23), August 6–10, 2023, Long Beach, CA, USA. ACM, New York, NY, USA, 16 pages. https://doi.org/10.1145/3580305.3599803

BibTeX
```
@article{elmougy2023demystifying,
  title={Demystifying Fraudulent Transactions and Illicit Nodes in the Bitcoin Network for Financial Forensics},
  author={Elmougy, Youssef and Liu, Ling},
  journal={arXiv preprint arXiv:2306.06108},
  year={2023}
}
```