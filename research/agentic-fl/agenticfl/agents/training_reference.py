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

"""Reference-selection guidance for agent-generated FL training code."""

NVFLARE_EXAMPLE_SELECTION_ID = "agent_selected_nvflare_example"
NVFLARE_EXAMPLE_SELECTION_PROMPT = (
    "Search NVIDIA FLARE examples for the closest implementation pattern instead of assuming "
    "a fixed reference. Choose by modality (image, language, or multimodal/mixed), "
    "application domain (for medical imaging, prefer MONAI examples), task type "
    "(segmentation, classification, detection, generation, etc.), FL workflow "
    "(FedAvg or the closest compatible controller), and execution style "
    "(Client API, PyTorch, MONAI, or another suitable integration). Use the selected "
    "example as structural guidance only; adapt the model, dataset, transforms, loss, "
    "metrics, IO, and privacy-safe logging to the AgenticFL training plan and "
    "extracted-data contract."
)

NVFLARE_EXAMPLE_SELECTION_PATTERN = (
    "Agent selects the closest NVFlare example by modality, application domain, task type, "
    "FL workflow, and execution style."
)
