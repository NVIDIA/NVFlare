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

import importlib


def _registry_module():
    return importlib.import_module("nvflare.security.study_registry")


def test_study_registry_returns_role_for_user_in_study():
    study_registry = _registry_module()
    registry = study_registry.StudyRegistry(
        {
            "cancer-research": {
                "sites": ["site-a", "site-b"],
                "admins": {"admin@nvidia.com": "lead"},
            }
        }
    )

    assert registry.get_role("admin@nvidia.com", "cancer-research") == "lead"


def test_study_registry_returns_none_for_missing_user_or_study():
    study_registry = _registry_module()
    registry = study_registry.StudyRegistry(
        {
            "cancer-research": {
                "sites": ["site-a"],
                "admins": {"admin@nvidia.com": "lead"},
            }
        }
    )

    assert registry.get_role("other@nvidia.com", "cancer-research") is None
    assert registry.get_role("admin@nvidia.com", "unknown-study") is None
    assert registry.get_sites("unknown-study") is None
    assert registry.has_study("unknown-study") is False


def test_study_registry_returns_enrolled_sites_as_a_set():
    study_registry = _registry_module()
    registry = study_registry.StudyRegistry(
        {
            "cancer-research": {
                "sites": ["site-a", "site-b"],
                "admins": {"admin@nvidia.com": "lead"},
            }
        }
    )

    assert registry.get_sites("cancer-research") == {"site-a", "site-b"}
    assert registry.has_study("cancer-research") is True


def test_study_registry_service_returns_initialized_registry():
    study_registry = _registry_module()
    registry = study_registry.StudyRegistry(
        {
            "cancer-research": {
                "sites": ["site-a"],
                "admins": {"admin@nvidia.com": "lead"},
            }
        }
    )

    study_registry.StudyRegistryService.initialize(registry)

    assert study_registry.StudyRegistryService.get_registry() is registry


def test_study_registry_service_reset_clears_registry():
    study_registry = _registry_module()
    study_registry.StudyRegistryService.initialize(study_registry.StudyRegistry({"study-a": {}}))

    study_registry.StudyRegistryService.reset()

    assert study_registry.StudyRegistryService.get_registry() is None
