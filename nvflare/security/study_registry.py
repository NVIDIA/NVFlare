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

import threading
from copy import deepcopy
from typing import Dict, Optional


class StudyRegistry:
    FORMAT_VERSION = "1.0"

    def __init__(self, studies_config: dict):
        if not isinstance(studies_config, dict):
            raise ValueError(f"studies_config must be dict but got {type(studies_config)}")

        format_version = studies_config.get("format_version")
        if format_version != self.FORMAT_VERSION:
            raise ValueError(f"missing or invalid study registry format_version: must be {self.FORMAT_VERSION}")

        studies = studies_config.get("studies")
        if not isinstance(studies, dict):
            raise ValueError(f"study registry 'studies' must be dict but got {type(studies)}")

        self._admins = {}
        self._site_orgs = {}
        self._sites = {}
        self._studies = {}
        for study_name, study_def in studies.items():
            study_def = study_def or {}
            admins = study_def.get("admins", [])
            if admins is None:
                admins = []
            if not isinstance(admins, list):
                raise ValueError(f"study '{study_name}' admins must be list but got {type(admins)}")
            site_orgs = study_def.get("site_orgs", {})
            if site_orgs is None:
                site_orgs = {}
            if not isinstance(site_orgs, dict):
                raise ValueError(f"study '{study_name}' site_orgs must be dict but got {type(site_orgs)}")
            admin_list = []
            seen_admins = set()
            for admin in admins:
                if not isinstance(admin, str):
                    raise ValueError(f"study '{study_name}' admin entries must be str but got {type(admin)}")
                if admin in seen_admins:
                    continue
                seen_admins.add(admin)
                admin_list.append(admin)

            normalized_site_orgs = {}
            sites = set()
            seen_sites = set()
            for org, org_sites in site_orgs.items():
                if not isinstance(org_sites, list):
                    raise ValueError(f"study '{study_name}' site_orgs[{org}] must be list but got {type(org_sites)}")
                normalized_sites = []
                for site in org_sites:
                    if not isinstance(site, str):
                        raise ValueError(
                            f"study '{study_name}' site entry for org '{org}' must be str but got {type(site)}"
                        )
                    if site in seen_sites:
                        raise ValueError(f"study '{study_name}' contains duplicate site '{site}' across org groups")
                    seen_sites.add(site)
                    normalized_sites.append(site)
                    sites.add(site)
                normalized_site_orgs[org] = normalized_sites

            self._admins[study_name] = set(admin_list)
            self._site_orgs[study_name] = normalized_site_orgs
            self._sites[study_name] = sites
            self._studies[study_name] = {
                "site_orgs": deepcopy(normalized_site_orgs),
                "sites": sorted(sites),
                "admins": list(admin_list),
            }

    def has_user(self, user_name: str, study: str) -> bool:
        return user_name in self._admins.get(study, set())

    def get_sites(self, study: str) -> set | None:
        return self._sites.get(study)

    def has_study(self, study: str) -> bool:
        return study in self._studies

    def has_org(self, study: str, org: str) -> bool:
        return org in self._site_orgs.get(study, {})

    def get_site_orgs(self, study: str) -> dict | None:
        site_orgs = self._site_orgs.get(study)
        return deepcopy(site_orgs) if site_orgs is not None else None

    def get_studies(self) -> dict[str, dict]:
        return deepcopy(self._studies)

    def get_study(self, study: str) -> dict | None:
        study_def = self._studies.get(study)
        return deepcopy(study_def) if study_def is not None else None


class StudyRegistryService:
    _registry: StudyRegistry | None = None
    _mutation_lock = threading.Lock()

    @staticmethod
    def initialize(registry: StudyRegistry | None):
        StudyRegistryService._registry = registry

    @staticmethod
    def get_registry() -> StudyRegistry | None:
        return StudyRegistryService._registry

    @staticmethod
    def reset():
        StudyRegistryService._registry = None

    @staticmethod
    def acquire_lock(timeout: float) -> bool:
        return StudyRegistryService._mutation_lock.acquire(timeout=timeout)

    @staticmethod
    def release_lock():
        StudyRegistryService._mutation_lock.release()
