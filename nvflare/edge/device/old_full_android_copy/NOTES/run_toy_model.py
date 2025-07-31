import torch
from executorch.extension.pybindings.portable_lib import _load_for_executorch_from_buffer

# Load the model
with open("add_model.pte", "rb") as f:
    buffer = f.read()
    model = _load_for_executorch_from_buffer(buffer)

# Create some test inputs (two tensors of ones)
input1 = torch.ones(1)
input2 = torch.ones(1)
inputs = (input1, input2)

# Run the model
print("Input 1:", input1)
print("Input 2:", input2)
output = model.run_method("forward", inputs)
print("Output (should be 2.0):", output[0]) 