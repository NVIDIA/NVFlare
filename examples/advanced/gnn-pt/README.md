# GraphSAGE with PyTorch and PyG
In this example, we will demonstrate how to train protein classifier using Graph Neural Network (GNN). 

The dataset we are using is PPI [protein-protein interaction]([http://snap.stanford.edu/graphsage/#code]) graphs, where each graph represents a specific human tissue. 

The federated learning Algorithm used is FedAvg [FedAvg](https://arxiv.org/abs/1602.05629) with SAG (Scatter and Gather) workflow. The deep learning framework used is pytorch. 



### Background of Graph Neural Network

Graph Neural Networks (GNNs) show promissing future in reaserch and industry, with potential applications in various domains, including social networks, e-commerce, recommendation systems, and more.
GNNs excel in learning, modeling, and leveraging complex relationships within graph-structured data. They combine local and global information, incorporate structural knowledge, adapt to diverse tasks, handle heterogeneous data, support transfer learning, scale for large graphs, offer interpretable insights, and achieve impressive performance. 
[GraphSAGE](https://arxiv.org/pdf/1706.02216.pdf) is a widely used GNN framework due to its ability to perform inductive learning on graph-structured data and scalability. 
Here we give an example of classifying protein roles based on their cellular functions from gene ontology with unsupervised [GraphSAGE](https://arxiv.org/pdf/1706.02216.pdf). The dataset we are using is PPI( 
([protein-protein interaction])([http://snap.stanford.edu/graphsage/#code]) graphs, where each graph represents a specific human tissue. 

We are adopting the example from [Pytorch Geometrics](https://pytorch-geometric.readthedocs.io/en/latest/), which is  a library built upon PyTorch to easily write and train Graph Neural Networks (GNNs) for a wide range of applications related to structured data.
> **_NOTE:_** 
You can follow the https://github.com/pyg-team/pytorch_geometric/blob/master/examples/graph_sage_unsup_ppi.py for GraphSAGE code.


### 1. Install NVIDIA FLARE

Follow the [Installation](https://nvflare.readthedocs.io/en/main/quickstart.html) instructions.
Install additional requirements:

```
python3 -m pip install -r requirements.txt
```

### 2. Run the experiment

Use nvflare simulator to run the hello-examples:

```
nvflare simulator -w /tmp/nvflare/ -n 2 -t 2 gnn-pt/jobs/gnn-pt
```

### 3. Access the logs and results

You can find the running logs and results inside the simulator's workspace/simulate_job

```bash
$ ls /tmp/nvflare/simulate_job/
app_server  app_site-1  app_site-2  log.txt

```

