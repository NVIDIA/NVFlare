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

from typing import Optional


class StudyRegistry:
    def __init__(self, studies_config: dict):
        if not isinstance(studies_config, dict):
            raise ValueError(f"studies_config must be dict but got {type(studies_config)}")

        self._roles = {}
        self._sites = {}
        for study_name, study_def in studies_config.items():
            study_def = study_def or {}
            self._roles[study_name] = dict(study_def.get("admins", {}))
            self._sites[study_name] = set(study_def.get("sites", []))

    def get_role(self, user_name: str, study: str) -> Optional[str]:
        return self._roles.get(study, {}).get(user_name)

    def get_sites(self, study: str) -> Optional[set]:
        return self._sites.get(study)

    def has_study(self, study: str) -> bool:
        return study in self._roles or study in self._sites


class StudyRegistryService:
    _registry: Optional[StudyRegistry] = None

    @staticmethod
    def initialize(registry: Optional[StudyRegistry]):
        StudyRegistryService._registry = registry

    @staticmethod
    def get_registry() -> Optional[StudyRegistry]:
        return StudyRegistryService._registry
