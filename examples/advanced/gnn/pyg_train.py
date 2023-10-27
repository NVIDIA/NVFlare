import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.nn import Dropout, LayerNorm, Linear, ReLU
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import ChebConv, DeepGCNLayer, DenseGCNConv, GATConv, GCNConv, GraphConv, NNConv

warnings.filterwarnings("ignore")

from pyg_sage import SAGE
from read_data import process_ellipitc

df_features = pd.read_csv("raw/elliptic_txs_features.csv", header=None)
df_edges = pd.read_csv("raw/elliptic_txs_edgelist.csv")
df_classes = pd.read_csv("raw/elliptic_txs_classes.csv")

node_features, classified_idx, edge_index, weights, labels, y_train = process_ellipitc(
    df_features, df_edges, df_classes
)
# converting data to PyGeometric graph data format
data_train = Data(
    x=node_features, edge_index=edge_index, edge_attr=weights, y=torch.tensor(labels, dtype=torch.double)
)  # , adj= torch.from_numpy(np.array(adj))
X_train, X_valid, y_train, y_valid, train_idx, valid_idx = train_test_split(
    node_features[classified_idx], y_train, classified_idx, test_size=0.15, random_state=42, stratify=y_train
)

print(X_train.size(), X_valid.size())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SAGE(data_train.num_node_features, hidden_channels=256, num_classes=2, num_layers=3).to(device)
model.double()
data_train = data_train.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

model.train()
for epoch in range(100):
    optimizer.zero_grad()
    out = model(data_train)
    # data_train.y.unsqueeze(1)
    # out = out.reshape((data_train.x.shape[0]))
    loss = F.nll_loss(out[train_idx], data_train.y[train_idx].T.to(torch.long))
    auc = roc_auc_score(
        data_train.y.detach().cpu().numpy()[train_idx], torch.argmax(out, dim=1).detach().cpu().numpy()[train_idx]
    )  # [train_idx]
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(" - loss: {} - roc: {}".format(loss.item(), auc))
