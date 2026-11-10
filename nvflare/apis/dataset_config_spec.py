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

"""Read-only access to the site's dataset configuration for the job's Study.

A site describes the data it offers each Study in ``<workspace>/local/datasets.yaml``. Components in a job read it
only through :func:`get_study_dataset_config`, which returns an immutable snapshot of the job's Study. In a container
job the launcher delivers only that Study's part and names it in ``NVFLARE_DATASET_CONFIG``.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

DATASET_CONFIG_ENV = "NVFLARE_DATASET_CONFIG"
DATASET_CONFIG_FILE = "datasets.yaml"  # under <workspace>/local


class DatasetConfigError(Exception):
    """The configuration is missing, unreadable or invalid, or a secret is unset. Messages never contain values."""


class UnknownDatasetError(DatasetConfigError):
    """The job's Study has no dataset with this name."""


@dataclass(frozen=True)
class DatasetConfig:
    """One dataset of the job's Study: ``studies.<study>.datasets.<name>`` in the file."""

    study: str  # the job's Study
    name: str  # the dataset name: the key in the file
    type: str  # source type, e.g. "parquet", "sql", "unity-catalog"
    uri: str  # absolute URI, required in the file, e.g. file:///data/oncology/omop
    path: str | None  # decoded local path of a file URI, e.g. /data/oncology/omop; None for a network URL
    options: Mapping[str, Any] = field(default_factory=dict)  # read-only, non-secret, type-specific
    secret_keys: tuple[str, ...] = ()  # keys of `secrets`, e.g. ("username", "password"); never values
    usage: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)  # read-only: component -> its rules

    def usage_rules(self, component: str) -> Mapping[str, Any] | None:
        """This dataset's rules for one component, e.g. "catalog"; None when it has none."""
        return self.usage.get(component)


class StudyDatasetConfig(ABC):
    """Read-only snapshot of the job's Study in the site's dataset configuration."""

    @property
    @abstractmethod
    def study(self) -> str:
        """The job's Study."""

    @property
    @abstractmethod
    def revision(self) -> str:
        """Opaque identifier of the snapshot; changes whenever the configuration changes."""

    @abstractmethod
    def list_datasets(self) -> list[str]:
        """The Study's dataset names, in file order; [] when the Study has none."""

    @abstractmethod
    def get_dataset(self, name: str) -> DatasetConfig:
        """One dataset. Raises UnknownDatasetError when the Study has no such dataset."""

    @abstractmethod
    def site_usage(self, component: str) -> Mapping[str, Any] | None:
        """Site-wide settings for one component (top-level usage.<component>); None when absent."""

    @abstractmethod
    def get_secret(self, name: str, key: str) -> str:
        """Value of one of the dataset's secrets, read from its environment variable at call time.

        Raises UnknownDatasetError for an unknown dataset, and DatasetConfigError when the key is not configured or
        the variable is unset. Never logs or returns the value in an error.
        """


def get_study_dataset_config(fl_ctx) -> StudyDatasetConfig:
    """Load and validate a snapshot of the job's Study.

    The Study comes from the job metadata in fl_ctx ("default" when the job names none). In a container job the file
    is the Study part the launcher delivered (NVFLARE_DATASET_CONFIG); in a process job it is
    <workspace>/local/datasets.yaml. Raises DatasetConfigError when the file is unreadable or invalid, or when
    NVFLARE_DATASET_CONFIG names a file that does not exist. Only a missing <workspace>/local/datasets.yaml yields a
    snapshot with no datasets.
    """
    from nvflare.app_common.dataset_config.api import load_job_study_config

    return load_job_study_config(fl_ctx)
