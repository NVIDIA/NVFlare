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

from executorch.exir import to_edge
from torch.export import export
from torch.export.experimental import _export_forward_backward


def export_model(net, input_tensor_example, label_tensor_example):
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
