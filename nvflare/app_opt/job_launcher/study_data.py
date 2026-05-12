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

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import yaml

DATA_MOUNT_ROOT = "/data"
MODE_RO = "ro"
MODE_RW = "rw"

_VALID_PATH_COMPONENT = re.compile(r"^[a-z0-9](?:[a-z0-9_-]{0,61}[a-z0-9])?$")


@dataclass(frozen=True)
class StudyDatasetMount:
    study: str
    dataset: str
    source: str
    mode: str

    @property
    def mount_path(self) -> str:
        return f"{DATA_MOUNT_ROOT}/{self.study}/{self.dataset}"

    @property
    def read_only(self) -> bool:
        return self.mode == MODE_RO


def _validate_path_component(value: str, label: str, file_path: str) -> None:
    if not isinstance(value, str) or not _VALID_PATH_COMPONENT.match(value):
        raise ValueError(f"{label} {value!r} in '{file_path}' is not a valid study-data path component.")


def load_study_data_file(file_path: str) -> dict:
    try:
        with open(file_path) as f:
            study_data = yaml.safe_load(f)
    except FileNotFoundError:
        return {}
    except OSError as e:
        raise ValueError(f"Could not read study data file '{file_path}': {e}") from e
    except yaml.YAMLError as e:
        raise ValueError(f"Could not parse study data file '{file_path}': {e}") from e

    if study_data is None:
        study_data = {}

    if not isinstance(study_data, dict):
        raise ValueError(f"file at study_data_file_path '{file_path}' does not contain a dictionary.")

    for study, datasets in study_data.items():
        _validate_path_component(study, "study name", file_path)
        if not isinstance(datasets, dict):
            raise ValueError(
                f"study_data.yaml uses study -> dataset -> {{source, mode}}; entry for study '{study}' "
                f"in '{file_path}' must be a dictionary."
            )
        for dataset, entry in datasets.items():
            _validate_path_component(dataset, "dataset name", file_path)
            if not isinstance(entry, dict):
                raise ValueError(
                    f"dataset entry '{study}/{dataset}' in '{file_path}' must be a dictionary with source and mode."
                )
            source = entry.get("source")
            if not isinstance(source, str) or not source:
                raise ValueError(f"dataset entry '{study}/{dataset}' in '{file_path}' must define a non-empty source.")
            mode = entry.get("mode")
            if mode not in (MODE_RO, MODE_RW):
                raise ValueError(f"dataset entry '{study}/{dataset}' in '{file_path}' must set mode to 'ro' or 'rw'.")

    return study_data


def should_mount_study_data(study: str | None) -> bool:
    return bool(study)


def resolve_study_dataset_mounts(study_data: dict, study: str, file_path: str) -> list[StudyDatasetMount]:
    datasets = study_data.get(study)
    if datasets is None:
        return []
    if not datasets:
        return []

    _validate_path_component(study, "study name", file_path)
    return [
        StudyDatasetMount(study=study, dataset=dataset, source=entry["source"], mode=entry["mode"])
        for dataset, entry in datasets.items()
    ]
