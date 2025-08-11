# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from executorch.exir import to_edge
from torch.export import export
from torch.export.experimental import _export_forward_backward


def export_model_to_bytes(net: nn.Module, input_shape, output_shape):
    """Exports a PyTorch model to ExecuTorch PTE format to be used in embedded or edge environments.

    This function creates dummy input and label tensors based on the provided shapes,
    runs the model export pipeline (including lowering to Executorch), and returns
    the serialized model buffer.

    Args:
        net (nn.Module): The PyTorch model to export.
        input_shape (tuple): The shape of the input tensor, e.g., (batch_size, channels, height, width).
        output_shape (tuple): The shape of the output tensor, e.g., (batch_size, num_classes).

    Returns:
        The exported model (.pte) in bytes.
    """

    input_tensor = torch.randn(input_shape)
    label_tensor = torch.ones(output_shape, dtype=torch.int64)
    model_buffer = export_model(net, input_tensor, label_tensor).buffer
    return model_buffer


def export_model(net: nn.Module, input_tensor_example, label_tensor_example):
    """Runs the full export pipeline to convert a PyTorch model into an Executorch-compatible format.

    This includes:
      - Capturing the forward graph
      - Capturing backward graph for training (if applicable)
      - Lowering to Edge dialect
      - Lowering to Executorch format

    Args:
        net (nn.Module): The PyTorch model to export.
        input_tensor_example (torch.Tensor): An example input tensor for tracing.
        label_tensor_example (torch.Tensor): An example output/label tensor for training export.

    Returns:
        ExportedProgram: The final lowered and exported Executorch model.
    """
    # Captures the forward graph. The graph will look similar to the model definition now.
    # Will move to export_for_training soon which is the api planned to be supported in the long term.
    ep = export(net, (input_tensor_example, label_tensor_example), strict=True)
    # Captures the backward graph. The exported_program now contains the joint forward and backward graph.
    ep = _export_forward_backward(ep)
    # Lower the graph to edge dialect.
    ep = to_edge(ep)
    # Lower the graph to executorch.
    ep = ep.to_executorch()
    return ep


# On device training requires the loss to be embedded in the model (and be the first output).
class DeviceModel(nn.Module):
    """Model wrapper for classification with CrossEntropyLoss."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input, label):
        pred = self.net(input)
        return self.loss(pred, label), pred.detach().argmax(dim=1)
