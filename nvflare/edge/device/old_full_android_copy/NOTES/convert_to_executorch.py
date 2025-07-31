import torch
from torch.export import export
from torch.export.exported_program import TS2EPConverter
import executorch.exir as exir
from executorch.exir import to_edge

def convert_model():
    # Load the TorchScript model
    model = torch.jit.load("image_classifier.pt")
    
    # Create example input
    example_inputs = (torch.randn(1, 3, 224, 224),)
    
    # Convert TorchScript to ExportedProgram
    converter = TS2EPConverter(model, example_inputs)
    exported_program = converter.convert()
    
    # Convert to ExecuTorch format
    edge_program = to_edge(exported_program)
    executorch_program = edge_program.to_executorch()
    
    # Save the model
    with open("image_classifier.pte", "wb") as f:
        f.write(executorch_program.buffer)
    
    print("Model converted successfully to image_classifier.pte")

if __name__ == "__main__":
    convert_model() 