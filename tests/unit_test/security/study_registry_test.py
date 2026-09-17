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


def _make_registry_config(studies):
    return {"format_version": "1.0", "studies": studies}


def test_study_registry_tracks_user_membership_for_study():
    study_registry = _registry_module()
    registry = study_registry.StudyRegistry(
        _make_registry_config(
            {
                "cancer-research": {
                    "site_orgs": {"org_a": ["site-a", "site-b"]},
                    "admins": ["admin@nvidia.com"],
                }
            }
        )
    )

    assert registry.has_user("admin@nvidia.com", "cancer-research") is True


def test_study_registry_rejects_missing_or_invalid_format_version():
    study_registry = _registry_module()

    try:
        study_registry.StudyRegistry({"studies": {"cancer-research": {}}})
        assert False, "expected ValueError for missing format_version"
    except ValueError as e:
        assert "format_version" in str(e)

    try:
        study_registry.StudyRegistry({"format_version": "2.0", "studies": {"cancer-research": {}}})
        assert False, "expected ValueError for invalid format_version"
    except ValueError as e:
        assert "format_version" in str(e)


def test_study_registry_rejects_missing_studies_mapping():
    study_registry = _registry_module()

    try:
        study_registry.StudyRegistry({"format_version": "1.0"})
        assert False, "expected ValueError for missing studies mapping"
    except ValueError as e:
        assert "studies" in str(e)


def test_study_registry_returns_false_for_missing_user_or_study():
    study_registry = _registry_module()
    registry = study_registry.StudyRegistry(
        _make_registry_config(
            {
                "cancer-research": {
                    "site_orgs": {"org_a": ["site-a"]},
                    "admins": ["admin@nvidia.com"],
                }
            }
        )
    )

    assert registry.has_user("other@nvidia.com", "cancer-research") is False
    assert registry.has_user("admin@nvidia.com", "unknown-study") is False
    assert registry.get_sites("unknown-study") is None
    assert registry.has_study("unknown-study") is False


def test_study_registry_returns_enrolled_sites_as_a_set():
    study_registry = _registry_module()
    registry = study_registry.StudyRegistry(
        _make_registry_config(
            {
                "cancer-research": {
                    "site_orgs": {"org_a": ["site-a"], "org_b": ["site-b"]},
                    "admins": ["admin@nvidia.com"],
                }
            }
        )
    )

    assert registry.get_sites("cancer-research") == {"site-a", "site-b"}
    assert registry.has_study("cancer-research") is True


def test_study_registry_service_returns_initialized_registry():
    study_registry = _registry_module()
    registry = study_registry.StudyRegistry(
        _make_registry_config(
            {
                "cancer-research": {
                    "site_orgs": {"org_a": ["site-a"]},
                    "admins": ["admin@nvidia.com"],
                }
            }
        )
    )

    study_registry.StudyRegistryService.initialize(registry)

    assert study_registry.StudyRegistryService.get_registry() is registry


def test_study_registry_service_reset_clears_registry():
    study_registry = _registry_module()
    study_registry.StudyRegistryService.initialize(study_registry.StudyRegistry(_make_registry_config({"study-a": {}})))

    study_registry.StudyRegistryService.reset()

    assert study_registry.StudyRegistryService.get_registry() is None


def test_study_registry_rejects_duplicate_site_across_org_groups():
    study_registry = _registry_module()

    try:
        study_registry.StudyRegistry(
            _make_registry_config(
                {
                    "cancer-research": {
                        "site_orgs": {
                            "org_a": ["site-shared"],
                            "org_b": ["site-shared"],  # duplicate
                        },
                        "admins": [],
                    }
                }
            )
        )
        assert False, "expected ValueError for duplicate site across org groups"
    except ValueError as e:
        assert "duplicate" in str(e).lower()


def test_study_registry_derived_flat_sites_union_of_all_org_groups():
    study_registry = _registry_module()
    registry = study_registry.StudyRegistry(
        _make_registry_config(
            {
                "cancer-research": {
                    "site_orgs": {
                        "org_a": ["site-a", "site-b"],
                        "org_b": ["site-c"],
                    },
                    "admins": [],
                }
            }
        )
    )

    sites = registry.get_sites("cancer-research")
    assert sites == {"site-a", "site-b", "site-c"}


def test_study_registry_flat_sites_excludes_sites_from_removed_org():
    study_registry = _registry_module()
    # Simulate what happens after remove-site removes all sites from org_b:
    # the flat sites set must not contain org_b's former sites.
    registry = study_registry.StudyRegistry(
        _make_registry_config(
            {
                "cancer-research": {
                    "site_orgs": {
                        "org_a": ["site-a"],
                        "org_b": [],  # org_b enrolled but no sites left
                    },
                    "admins": [],
                }
            }
        )
    )

    sites = registry.get_sites("cancer-research")
    assert sites == {"site-a"}
