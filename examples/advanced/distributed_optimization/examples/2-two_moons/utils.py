from torch import nn
from sklearn.datasets import make_moons
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class TwoMoonsDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders(data_seed:int):
    X, y = make_moons(n_samples=100, noise=0.1, random_state=data_seed)

    X_train = torch.from_numpy(X).float()
    y_train = torch.from_numpy(y).long()

    X, y = make_moons(n_samples=20, noise=0.1, random_state=42)

    X_test = torch.from_numpy(X).float()
    y_test = torch.from_numpy(y).long()

    train_dataset = TwoMoonsDataset(X_train, y_train)
    test_dataset = TwoMoonsDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64)
    return train_dataloader, test_dataloader


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(2, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def plot_results(job, num_clients):
    plt.style.use('ggplot')
    train_loss = {}
    test_loss = {}
    for i in range(num_clients):
        train_loss[f"site-{i + 1}"] = torch.load(
            f"./tmp/runs/{job}/site-{i + 1}/train_loss_sequence.pt"
        )
        test_loss[f"site-{i + 1}"] = torch.load(
            f"./tmp/runs/{job}/site-{i + 1}/test_loss_sequence.pt"
        )

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # First subplot: Evolution of training loss
    for i in range(num_clients):
        time = train_loss[f"site-{i + 1}"][:, 0]
        loss = train_loss[f"site-{i + 1}"][:, 1]
        axs[0].plot(time, loss, label=f"site-{i + 1}")
    axs[0].legend()
    axs[0].set_ylim(-0.1, 1)
    axs[0].set_ylabel("Loss")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_title("Evolution of Training Loss")

    # Second subplot: Evolution of test loss
    for i in range(num_clients):
        time = test_loss[f"site-{i + 1}"][:, 0]
        loss = test_loss[f"site-{i + 1}"][:, 1]
        axs[1].plot(time, loss, label=f"site-{i + 1}")
    axs[1].legend()
    axs[1].set_ylim(-0.1, 1)
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_title("Evolution of Test Loss")

    plt.tight_layout()
    plt.savefig(f"{job}_results.png")