import torch
import torch.nn as nn
from executorch.exir import to_edge
from executorch.exir.backend.backend_api import to_backend
from executorch.exir.backend.backend_api import CompileSpec
from executorch.exir.backend.backend_api import BackendType

class SimpleAddition(nn.Module):
    def forward(self, x, y):
        return x + y

def main():
    # Create the model
    model = SimpleAddition()
    model.eval()
    
    # Create example inputs
    example_inputs = (torch.tensor([1.0]), torch.tensor([1.0]))
    
    # Export to ExecuTorch format
    edge_model = to_edge(model, example_inputs)
    executorch_program = edge_model.to_executorch()
    
    # Save the model
    with open("simple_addition.pte", "wb") as f:
        f.write(executorch_program.buffer)

if __name__ == "__main__":
    main() 