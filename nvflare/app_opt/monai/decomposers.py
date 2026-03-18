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

"""Registration helpers for MONAI types used by FOBS serialization."""

from nvflare.fuel.utils import fobs

MONAI_ENUM_TYPES: set[str] = {
    "monai.utils.enums.TraceKeys",
    "monai.utils.enums.TraceStatusKeys",
    "monai.utils.enums.CommonKeys",
    "monai.utils.enums.GanKeys",
    "monai.utils.enums.JITMetadataKeys",
    "monai.utils.enums.ProbMapKeys",
    "monai.utils.enums.PatchKeys",
    "monai.utils.enums.WSIPatchKeys",
    "monai.utils.enums.FastMRIKeys",
    "monai.utils.enums.SpaceKeys",
    "monai.utils.enums.MetaKeys",
    "monai.utils.enums.EngineStatsKeys",
    "monai.utils.enums.DataStatsKeys",
    "monai.utils.enums.ImageStatsKeys",
    "monai.utils.enums.LabelStatsKeys",
    "monai.utils.enums.AlgoKeys",
    "monai.utils.enums.AdversarialKeys",
    "monai.handlers.metric_logger.MetricLoggerKeys",
    "monai.apps.nuclick.transforms.NuclickKeys",
}


def register_monai_types() -> None:
    """Allow MONAI enum-like types to be deserialized by FOBS."""
    for monai_type in MONAI_ENUM_TYPES:
        fobs.add_type_name_whitelist(monai_type)
