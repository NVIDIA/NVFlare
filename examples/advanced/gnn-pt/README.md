# GraphSAGE with PyTorch and PyG

Example of using [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) to train an image classifier
using federated averaging ([FedAvg](https://arxiv.org/abs/1602.05629))
and [PyTorch](https://pytorch.org/) as the deep learning training framework.



### Background of Graph Neural Network

Graph Neural Networks (GNNs) are a promising research topic and industry market, with potential applications in various domains, including social networks, e-commerce, recommendation systems, and more.
GNNs excel in learning, modeling, and leveraging complex relationships within graph-structured data. They combine local and global information, incorporate structural knowledge, adapt to diverse tasks, handle heterogeneous data, support transfer learning, scale for large graphs, offer interpretable insights, and achieve impressive performance. 
[GraphSAGE](https://arxiv.org/pdf/1706.02216.pdf) is a widely used GNN due to its ability to perform inductive learning on graph-structured data and scalability. 
Here we give an example of classifying protein roles based on their cellular functions from gene ontology with unsupervised [GraphSAGE](https://arxiv.org/pdf/1706.02216.pdf). The dataset we are using is 
[protein-protein interaction]([http://snap.stanford.edu/graphsage/#code]) graphs, where each graph represents a specific human tissue. 

We are adopting the example from Pytorch Geometrics (PyG), which is  a library built upon PyTorch to easily write and train Graph Neural Networks (GNNs) for a wide range of applications related to structured data.
> **_NOTE:_** 
You can follow the https://github.com/pyg-team/pytorch_geometric/blob/master/examples/graph_sage_unsup_ppi.py for GraphSAGE code.


### 1. Install NVIDIA FLARE

Follow the [Installation](https://nvflare.readthedocs.io/en/main/quickstart.html) instructions.
Install additional requirements:

```
pip3 install -r requirements.txt
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

