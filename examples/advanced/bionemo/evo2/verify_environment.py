# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Fail fast if the Evo2 native CUDA extension is incompatible with PyTorch."""

import torch
from causal_conv1d import causal_conv1d_fn


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside the Evo2 container.")
    input_tensor = torch.randn(2, 32, 600, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(32, 4, device="cuda", dtype=torch.float32, requires_grad=True)
    output = causal_conv1d_fn(input_tensor, weight)
    output.float().sum().backward()
    torch.cuda.synchronize()
    print("causal-conv1d CUDA forward/backward smoke passed")


if __name__ == "__main__":
    main()
