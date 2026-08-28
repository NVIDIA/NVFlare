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

"""Canonical prepared-data contracts for AgenticFL.

These modules define the small, stable data shapes that agent-authored local
adapters must produce. The contracts are intentionally task-family level:
dataset-specific discovery and conversion remains agent-owned at runtime.
"""

from agenticfl.data.contracts.base import (
    CLASSIFICATION,
    OBJECT_DETECTION,
    SEGMENTATION,
    STANDARD_SPLITS,
    DataContract,
    available_contract_summaries,
    datalist_split_key,
    generated_contract_bbox_format,
    generated_contract_box_field,
    generated_contract_field_names,
    generated_contract_label_field,
    generated_contract_label_ids,
    generated_contract_materialized_field_names,
    generated_contract_record_type,
    generated_contract_visual_qc_required,
    generated_data_contract_validation_errors,
    generated_record_type_aliases,
    infer_record_type,
    manifest_record_type,
    normalize_record_type,
    task_family_from_policy,
)

__all__ = [
    "CLASSIFICATION",
    "OBJECT_DETECTION",
    "SEGMENTATION",
    "STANDARD_SPLITS",
    "DataContract",
    "available_contract_summaries",
    "datalist_split_key",
    "generated_contract_bbox_format",
    "generated_data_contract_validation_errors",
    "generated_contract_box_field",
    "generated_contract_field_names",
    "generated_contract_materialized_field_names",
    "generated_contract_visual_qc_required",
    "generated_contract_label_field",
    "generated_contract_label_ids",
    "generated_contract_record_type",
    "generated_record_type_aliases",
    "infer_record_type",
    "manifest_record_type",
    "normalize_record_type",
    "task_family_from_policy",
    "RUNTIME_CONTRACTS",
    "runtime_contract_for_record_type",
]

from . import classification, segmentation

RUNTIME_CONTRACTS = {
    SEGMENTATION: segmentation,
    CLASSIFICATION: classification,
}


def runtime_contract_for_record_type(record_type: str):
    return RUNTIME_CONTRACTS.get(record_type)
