import torch
import torchvision.models as models
from torch.export import export
import executorch.exir as exir
from executorch.exir import to_edge

def export_model():
    # Load pre-trained MobileNetV2
    model = models.mobilenet_v2(weights='IMAGENET1K_V1')
    model.eval()
    
    # Create example input
    example_inputs = (torch.randn(1, 3, 224, 224),)
    
    # Export the model
    print("Exporting model to ExportedProgram...")
    exported_program = export(model, example_inputs)
    
    # Convert to Edge dialect
    print("Converting to Edge dialect...")
    edge_program = to_edge(exported_program)
    
    # Convert to ExecuTorch format
    print("Converting to ExecuTorch format...")
    executorch_program = edge_program.to_executorch()
    
    # Save the model
    print("Saving model...")
    with open("image_classifier.pte", "wb") as f:
        f.write(executorch_program.buffer)
    
    print("Model exported successfully to image_classifier.pte")

if __name__ == "__main__":
    export_model() 