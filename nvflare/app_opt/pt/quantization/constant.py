# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

# Supported Input Data Type
# Message quantization is mainly for reducing the message that can be
# significantly large, e.g. LLMs. Thus, the supported input data types
# we consider are common ones during LLM training, including fp32, fp16, and bf16.
DATA_TYPE = [
    "FLOAT32",
    "FLOAT16",
    "BFLOAT16",
]

# Supported Quantization Type to reduce the above input data types
# The quantization types are mainly for reducing the model size,
# Hence, we support 16-, 8-, and 4-bits quantization.
# Note that 8- and 4-bits quantization needs GPU support.
QUANTIZATION_TYPE = [
    "FLOAT16",
    "BLOCKWISE8",
    "FLOAT4",
    "NORMFLOAT4",
]
