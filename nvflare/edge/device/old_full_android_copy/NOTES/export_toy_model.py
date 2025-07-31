import torch
from executorch.examples.models.toy_model.model import AddModule
from executorch.exir import to_edge
from torch.export import export

# Create the model
model = AddModule()

# Get example inputs
example_inputs = model.get_example_inputs()

# Export to aten
exported_model = export(model, example_inputs)

# Export to edge
edge_model = to_edge(exported_model)

# Convert to executorch
executorch_program = edge_model.to_executorch()

# Save the model
with open("add_model.pte", "wb") as f:
    f.write(executorch_program.buffer) 