import torch
import torch.nn as nn
from torch.nn import functional as F

# Basic Net for XOR
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 10)
        self.linear2 = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear2(F.sigmoid(self.linear(x)))

# On device training requires the loss to be embedded in the model
class TrainingNet(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input, label):
        pred = self.net(input)
        return self.loss(pred, label), pred.detach().argmax(dim=1)

def export_model():
    # Create and initialize the model
    net = Net()
    training_net = TrainingNet(net)
    training_net.eval()

    # Create example inputs
    example_inputs = (torch.randn(1, 2), torch.tensor([0]))

    # Export the model
    torch.export.export(
        training_net,
        example_inputs,
        "xor.pte"
    )

if __name__ == "__main__":
    export_model() 