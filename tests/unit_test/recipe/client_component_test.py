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

from unittest.mock import Mock

import pytest

from nvflare.app_common.filters.memorization_audit_filter import AuditModelSnapshot
from nvflare.job_config.api import FedApp, FedJob
from nvflare.job_config.fed_app_config import ClientAppConfig
from nvflare.recipe.spec import Recipe


def test_register_in_existing_site_apps_without_creating_an_all_sites_app():
    job = FedJob(name="components", min_clients=2)
    for site in ["site-1", "site-2"]:
        job._deploy_map[site] = FedApp(ClientAppConfig())
    recipe = Recipe(job)
    snapshot = AuditModelSnapshot()
    recipe.add_client_component(snapshot, "snapshot", clients=["site-1"])
    assert job._deploy_map["site-1"].app_config.components["snapshot"] is snapshot
    assert "snapshot" not in job._deploy_map["site-2"].app_config.components
    assert set(job._deploy_map) == {"site-1", "site-2"}
    with pytest.raises(ValueError, match="unknown client"):
        recipe.add_client_component(snapshot, "snapshot", clients=["site-3"])


def test_register_for_all_existing_sites():
    job = FedJob(name="components", min_clients=2)
    for site in ["site-1", "site-2"]:
        job._deploy_map[site] = FedApp(ClientAppConfig())
    recipe = Recipe(job)
    snapshot = AuditModelSnapshot()
    recipe.add_client_component(snapshot, "snapshot")
    for app in job._deploy_map.values():
        assert app.app_config.components["snapshot"] is snapshot


@pytest.mark.parametrize(
    "component,component_id,error",
    [(object(), "x", TypeError), (AuditModelSnapshot(), "", ValueError), (AuditModelSnapshot(), None, ValueError)],
)
def test_invalid_component_rejected_before_topology_changes(component, component_id, error):
    recipe = Recipe(FedJob(name="components", min_clients=1))
    recipe._add_to_client_apps = Mock()
    with pytest.raises(error):
        recipe.add_client_component(component, component_id)
    recipe._add_to_client_apps.assert_not_called()
