import torch
import torch.nn as nn
import torch.nn.functional as F
from executorch.exir import to_edge
from executorch.exir import EdgeCompileConfig
from executorch.exir import ExecutorchProgram

class SimpleImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple CNN for 3x32x32 input images
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    # Create the model
    model = SimpleImageClassifier()
    model.eval()
    
    # Create example input (batch size 1, 3 channels, 32x32 image)
    example_input = torch.randn(1, 3, 32, 32)
    
    # Export to ExecuTorch format
    edge_model = to_edge(model, (example_input,))
    executorch_program = edge_model.to_executorch()
    
    # Save the model
    with open("image_classifier.pte", "wb") as f:
        f.write(executorch_program.buffer)
    
    print("Model exported successfully to image_classifier.pte")

if __name__ == "__main__":
    main() 