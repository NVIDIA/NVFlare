import os

import torch
from network import SimpleNetwork
from torch import nn
from torch.optim import SGD
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor


DATASET_PATH = "/tmp/nvflare/data"


def main():
    batch_size = 4
    epochs = 1
    lr = 0.01
    model = SimpleNetwork()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
    transforms = Compose(
        [
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    data_path = os.path.join(DATASET_PATH, "site-1")
                 
    train_dataset = CIFAR10(root=data_path, transform=transforms, download=True, train=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    n_loaders = len(train_loader)
    
    model.to(device)

    print("number of loaders = ", n_loaders)

    steps = epochs * n_loaders
    for epoch in range(epochs):
            running_loss = 0.0

            for i, batch in enumerate(train_loader):
                images, labels = batch[0].to(device), batch[1].to(device)
      
                optimizer.zero_grad()

                predictions = model(images)
                cost = loss(predictions, labels)
                cost.backward()
                optimizer.step()

                running_loss += cost.cpu().detach().numpy() / batch_size
                
                if i % 3000 == 0:
                    print(f"Epoch: {epoch+1}/{epochs}, batch: {i+1}, Loss: {running_loss / 3000}")
                    running_loss = 0.0

            last_loss = {running_loss / (i+1)}
            print(f"Epoch: {epoch}/{epochs}, batch: {i+1}, Loss: {last_loss}")

    print("Finished Training")

    PATH = "./cifar_net.pth"
    torch.save(model.state_dict(), PATH)

    
    print(f"\n"
          f"Result Summary\n"
           "    Training parameters:\n" 
          f"       batch_size = {batch_size},\n"
          f"       epochs = {epochs},\n"
          f"       lr = {lr},\n"
          f"       total data batches = {n_loaders},\n"
          f"    Metrics: last_loss = {last_loss}\n")


if __name__ == "__main__":
    main()
