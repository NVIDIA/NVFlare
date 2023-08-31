import os.path

import torch
import torch.nn.functional as F
import tqdm
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier

from torch_geometric.data import Batch
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader, LinkNeighborLoader
from torch_geometric.nn import GraphSAGE

# (0) import nvflare client API
import nvflare.client as flare

# Create PPI dataset for training.
path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "PPI")
train_dataset = PPI(path, split="train")
train_data = Batch.from_data_list(train_dataset)
train_loader = LinkNeighborLoader(
    train_data,
    batch_size=2048,
    shuffle=True,
    neg_sampling_ratio=1.0,
    num_neighbors=[10, 10],
    num_workers=6,
    persistent_workers=True,
)

test_dataset = PPI(path, split="test")
test_loader = DataLoader(test_dataset, batch_size=2)

n_iterations = len(train_loader)

LR = 0.01
EPOCHS = 5


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# need to change 64 2 64 to json
net = GraphSAGE(
    in_channels=train_dataset.num_features,
    hidden_channels=64,
    num_layers=2,
    out_channels=64,
)


# (1) initializes NVFlare client API
flare.init()
# (2) gets FLModel from NVFlare
input_model = flare.receive()

# (3) loads model from NVFlare
net.load_state_dict(input_model.params)
net.to(device)

# (4) test received model for model selection
def encode(loader):
    net.eval()
    xs, ys = [], []
    for data in loader:
        data = data.to(device)
        xs.append(net(data.x, data.edge_index).cpu())
        ys.append(data.y.cpu())
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


with torch.no_grad():
    x, y = encode(train_loader)
    clf = MultiOutputClassifier(SGDClassifier(loss="log", penalty="l2"))
    clf.fit(x, y)
    train_f1 = f1_score(y, clf.predict(x), average="micro")

    # Evaluate on test set:
    x, y = encode(test_loader)
    test_f1 = f1_score(y, clf.predict(x), average="micro")

optimizer = torch.optim.Adam(net.parameters(), lr=LR)

total_loss = 0
total_examples = 0
for epoch in range(EPOCHS):
    for data in tqdm.tqdm(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        h = net(data.x, data.edge_index)

        h_src = h[data.edge_label_index[0]]
        h_dst = h[data.edge_label_index[1]]
        link_pred = (h_src * h_dst).sum(dim=-1)  # Inner product.

        loss = F.binary_cross_entropy_with_logits(link_pred, data.edge_label)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * link_pred.numel()
        total_examples += link_pred.numel()


# (5) construct the FLModel to returned back
output_model = flare.FLModel(params=net.cpu().state_dict(), params_type="FULL", metrics={"f1": test_f1})

# (6) send back model
flare.send(output_model)
