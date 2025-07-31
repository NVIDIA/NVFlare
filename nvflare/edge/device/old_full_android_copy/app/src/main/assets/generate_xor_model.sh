#!/bin/bash

# Create a temporary directory
TEMP_DIR=$(mktemp -d)
cd $TEMP_DIR

# Create the Python script
cat > export_xor_model.py << 'EOL'
import torch
import torch.nn as nn
import torch.optim as optim
from executorch.extension.training.pybindings._training_lib import export_model

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

class TrainingNet(nn.Module):
    def __init__(self):
        super(TrainingNet, self).__init__()
        self.net = Net()
        self.loss_fn = nn.MSELoss()

    def forward(self, x, y):
        pred = self.net(x)
        loss = self.loss_fn(pred, y)
        return loss

def export_model():
    model = TrainingNet()
    model.train()
    
    # Create example inputs
    x = torch.randn(4, 2)
    y = torch.randn(4, 1)
    
    # Export the model
    export_model(model, (x, y), "xor.pte")

if __name__ == "__main__":
    export_model()
EOL

# Install required packages
pip install torch executorch

# Run the script
python3 export_xor_model.py

# Copy the generated model to the assets directory
cp xor.pte /Users/kevlu/workspace/repos/mobile/NVFlare/nvflare/edge/android/app/src/main/assets/

# Clean up
cd -
rm -rf $TEMP_DIR 