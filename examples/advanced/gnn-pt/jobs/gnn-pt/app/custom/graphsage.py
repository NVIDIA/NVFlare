import os.path
import os.path as osp

import torch
from pt_constants import PTConstants
from torch import nn
from torch.optim import SGD

import torch.nn.functional as F
from torch_geometric.loader import DataLoader,LinkNeighborLoader
from torch_geometric.datasets import PPI
from torch_geometric.data import Batch
from torch_geometric.nn import GraphSAGE
import tqdm

# (1.0) import nvflare client API
import nvflare.client as flare


# Create PPI dataset for training.
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PPI')
train_dataset = PPI(path, split='train')
train_data = Batch.from_data_list(train_dataset)
train_loader = LinkNeighborLoader(train_data, batch_size=2048, shuffle=True,
                                  neg_sampling_ratio=1.0, num_neighbors=[10, 10],
                                  num_workers=6, persistent_workers=True)

test_dataset = PPI(path, split='test')
test_loader = DataLoader(test_dataset, batch_size=2)

n_iterations = len(train_loader)

lr=0.01
epochs=5


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# need to change 64 2 64 to json
model = GraphSAGE(
            in_channels=train_dataset.num_features,
            hidden_channels=64,
            num_layers=2,
            out_channels=64,
            ).to(device)


# (1.1) initializes NVFlare client API
flare.init()
# (1.2) gets FLModel from NVFlare
input_model = flare.receive()

# (1.3) loads model from NVFlare
net.load_state_dict(input_model.params)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    running_loss = 0.0
    for data in tqdm.tqdm(train_loader):
        data = data.to(self.device)
        optimizer.zero_grad()
        h = self.model(data.x, data.edge_index)

        h_src = h[data.edge_label_index[0]]
        h_dst = h[data.edge_label_index[1]]
        link_pred = (h_src * h_dst).sum(dim=-1)  # Inner product.

        loss = F.binary_cross_entropy_with_logits(link_pred, data.edge_label)
        loss.backward()
        optimizer.step()
                
        total_loss += float(loss) * link_pred.numel()
        total_examples += link_pred.numel()


print("Finished Training")

PATH = "./graphsage.pth"
torch.save(net.state_dict(), PATH)

#test
model = GraphSAGE(in_channels=train_dataset.num_features,
            hidden_channels=64,
            num_layers=2,
            out_channels=64,
            )
model.load_state_dict(torch.load(PATH))
model.to(device)


def encode(loader):
    model.eval()
    xs, ys = [], []
    for data in loader:
        data = data.to(device)
        xs.append(model(data.x, data.edge_index).cpu())
        ys.append(data.y.cpu())
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)

with torch.no_grad():
    x, y = encode(train_loader)
    clf = MultiOutputClassifier(SGDClassifier(loss='log', penalty='l2'))
    clf.fit(x, y)
    train_f1 = f1_score(y, clf.predict(x), average='micro')

    # Evaluate on test set:
    x, y = encode(test_loader)
    test_f1 = f1_score(y, clf.predict(x), average='micro')

    print (train_f1, test_f1)
