# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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
import re
from typing import Dict, List, Optional, Tuple

import pytest

from nvflare.apis.client import Client
from nvflare.apis.fl_context import FLContext, FLContextManager
from nvflare.apis.job_def import ALL_SITES, Job
from nvflare.apis.job_scheduler_spec import DispatchInfo
from nvflare.apis.resource_manager_spec import ResourceManagerSpec
from nvflare.apis.server_engine_spec import ServerEngineSpec
from nvflare.app_common.job_schedulers.job_scheduler import DefaultJobScheduler
from nvflare.app_common.resource_managers.list_resource_manager import ListResourceManager


class DummyResourceManager(ResourceManagerSpec):
    def __init__(self, name, resources):
        self.name = name
        self.resources = resources

    def check_resources(self, resource_requirement: dict, fl_ctx: FLContext) -> (bool, Optional[str]):
        print(f"{self.name}: checking resources with requirements {resource_requirement}")
        for k in resource_requirement:
            if k in self.resources:
                if self.resources[k] < resource_requirement[k]:
                    return False, None
        return True, None

    def cancel_resources(self, resource_requirement: dict, token: str, fl_ctx: FLContext):
        print(f"{self.name}: cancelling resources {resource_requirement}")

    def allocate_resources(self, resource_requirement: dict, token: str, fl_ctx: FLContext) -> dict:
        print(f"{self.name}: allocating resources {resource_requirement}")
        result = {}
        for k in resource_requirement:
            if k in self.resources:
                self.resources[k] -= resource_requirement[k]
                result[k] = resource_requirement[k]
        return result

    def free_resources(self, resources: dict, token: str, fl_ctx: FLContext):
        print(f"{self.name}: freeing resources {resources}")
        for k in resources:
            self.resources[k] += resources[k]


class Site:
    def __init__(self, name, resources, resource_manager=None):
        self.name = name
        if resource_manager:
            self.resource_manager = resource_manager
        else:
            self.resource_manager = DummyResourceManager(name=name, resources=resources)


class MockServerEngine(ServerEngineSpec):
    def __init__(self, clients: Dict[str, Site], run_name="exp1"):
        self.fl_ctx_mgr = FLContextManager(
            engine=self,
            identity_name="__mock_engine",
            job_id=run_name,
            public_stickers={},
            private_stickers={},
        )
        self.clients = clients

    def fire_event(self, event_type: str, fl_ctx: FLContext):
        pass

    def get_clients(self):
        return [Client(name=x, token="") for x in self.clients]

    def sync_clients_from_main_process(self):
        pass

    def validate_clients(self, client_names: List[str]):
        pass

    def new_context(self):
        return self.fl_ctx_mgr.new_context()

    def get_workspace(self):
        pass

    def get_component(self, component_id: str) -> object:
        pass

    def register_aux_message_handler(self, topic: str, message_handle_func):
        pass

    def send_aux_request(self, targets: [], topic: str, request, timeout: float, fl_ctx: FLContext) -> dict:
        pass

    def get_widget(self, widget_id: str):
        pass

    def persist_components(self, fl_ctx: FLContext, completed: bool):
        pass

    def restore_components(self, snapshot, fl_ctx: FLContext):
        pass

    def start_client_job(self, job_id, client_sites):
        pass

    def check_client_resources(self, resource_reqs: Dict[str, dict]) -> Dict[str, Tuple[bool, Optional[str]]]:
        result = {}
        with self.new_context() as fl_ctx:
            for site_name, requirements in resource_reqs.items():
                result[site_name] = self.clients[site_name].resource_manager.check_resources(requirements, fl_ctx)
        return result

    def get_client_name_from_token(self, token):
        return self.clients.get(token)

    def cancel_client_resources(
        self, resource_check_results: Dict[str, Tuple[bool, str]], resource_reqs: Dict[str, dict]
    ):
        with self.new_context() as fl_ctx:
            for site_name, result in resource_check_results.items():
                check_result, token = result
                if check_result and token:
                    self.clients[site_name].resource_manager.cancel_resources(
                        resource_requirement=resource_reqs[site_name], token=token, fl_ctx=fl_ctx
                    )


def create_servers(server_num, sites: List[Site]):
    servers = []
    for i in range(server_num):
        engine = MockServerEngine(clients={s.name: s for s in sites})
        servers.append(engine)
    return servers


def create_resource(cpu, gpu):
    return {"cpu": cpu, "gpu": gpu}


def create_job(job_id, resource_spec, deploy_map, min_sites, required_sites=None):
    return Job(
        job_id=job_id,
        resource_spec=resource_spec,
        deploy_map=deploy_map,
        min_sites=min_sites,
        required_sites=required_sites,
        meta={},
    )


def create_jobs(num_jobs, prefix="job", **kwargs):
    return [Job(job_id=f"{prefix}{i}", **kwargs) for i in range(num_jobs)]


job1 = create_job(
    job_id="job1",
    resource_spec={"site1": create_resource(1, 4), "site2": create_resource(1, 4), "site3": create_resource(2, 1)},
    deploy_map={"app1": ["server", "site1", "site2"], "app2": ["site3"]},
    min_sites=3,
)

job2 = create_job(
    job_id="job2",
    resource_spec={"site1": create_resource(2, 4), "site2": create_resource(2, 4), "site3": create_resource(12, 4)},
    deploy_map={"app3": ["server", "site1", "site2"], "app4": ["site3"]},
    min_sites=3,
)

job3 = create_job(
    job_id="job3",
    resource_spec={},
    deploy_map={"app5": [ALL_SITES]},
    min_sites=3,
)

job4 = create_job(
    job_id="job4",
    resource_spec={"site1": create_resource(2, 4), "site2": create_resource(5, 4), "site3": create_resource(12, 4)},
    deploy_map={"app7": ["server", "site1", "site2"], "app8": ["site3", "site4", "site5"]},
    min_sites=3,
)

job5 = create_job(
    job_id="job5",
    resource_spec={},
    deploy_map={"app9": [ALL_SITES], "app10": []},
    min_sites=3,
)


TEST_CASES = [
    (
        [job1],
        [
            Site(name="site1", resources=create_resource(16, 8)),
            Site(name="site2", resources=create_resource(16, 8)),
            Site(name="site3", resources=create_resource(32, 1)),
            Site(name="site4", resources=create_resource(2, 1)),
        ],
        job1,
        {
            "server": DispatchInfo(app_name="app1", resource_requirements={}, token=None),
            "site1": DispatchInfo(app_name="app1", resource_requirements=create_resource(1, 4), token=None),
            "site2": DispatchInfo(app_name="app1", resource_requirements=create_resource(1, 4), token=None),
            "site3": DispatchInfo(app_name="app2", resource_requirements=create_resource(2, 1), token=None),
        },
    ),
    (
        [job2, job1],
        [
            Site(name="site1", resources=create_resource(16, 8)),
            Site(name="site2", resources=create_resource(16, 8)),
            Site(name="site3", resources=create_resource(32, 1)),
            Site(name="site4", resources=create_resource(2, 1)),
        ],
        job1,
        {
            "server": DispatchInfo(app_name="app1", resource_requirements={}, token=None),
            "site1": DispatchInfo(app_name="app1", resource_requirements=create_resource(1, 4), token=None),
            "site2": DispatchInfo(app_name="app1", resource_requirements=create_resource(1, 4), token=None),
            "site3": DispatchInfo(app_name="app2", resource_requirements=create_resource(2, 1), token=None),
        },
    ),
    (
        [job3],
        [Site(name=f"site{i}", resources=create_resource(16, 8)) for i in range(8)],
        job3,
        {
            "server": DispatchInfo(app_name="app5", resource_requirements={}, token=None),
            "site0": DispatchInfo(app_name="app5", resource_requirements={}, token=None),
            "site1": DispatchInfo(app_name="app5", resource_requirements={}, token=None),
            "site2": DispatchInfo(app_name="app5", resource_requirements={}, token=None),
            "site3": DispatchInfo(app_name="app5", resource_requirements={}, token=None),
            "site4": DispatchInfo(app_name="app5", resource_requirements={}, token=None),
            "site5": DispatchInfo(app_name="app5", resource_requirements={}, token=None),
            "site6": DispatchInfo(app_name="app5", resource_requirements={}, token=None),
            "site7": DispatchInfo(app_name="app5", resource_requirements={}, token=None),
        },
    ),
    (
        [job4, job1],
        [
            Site(name="site1", resources=create_resource(16, 8)),
            Site(name="site2", resources=create_resource(16, 8)),
            Site(name="site3", resources=create_resource(32, 1)),
            Site(name="site4", resources=create_resource(2, 1)),
        ],
        job4,
        {
            "server": DispatchInfo(app_name="app7", resource_requirements={}, token=None),
            "site1": DispatchInfo(app_name="app7", resource_requirements=create_resource(2, 4), token=None),
            "site2": DispatchInfo(app_name="app7", resource_requirements=create_resource(5, 4), token=None),
            "site4": DispatchInfo(app_name="app8", resource_requirements={}, token=None),
        },
    ),
    (
        [job5],
        [Site(name=f"site{i}", resources=create_resource(16, 8)) for i in range(8)],
        job5,
        {
            "server": DispatchInfo(app_name="app9", resource_requirements={}, token=None),
            "site0": DispatchInfo(app_name="app9", resource_requirements={}, token=None),
            "site1": DispatchInfo(app_name="app9", resource_requirements={}, token=None),
            "site2": DispatchInfo(app_name="app9", resource_requirements={}, token=None),
            "site3": DispatchInfo(app_name="app9", resource_requirements={}, token=None),
            "site4": DispatchInfo(app_name="app9", resource_requirements={}, token=None),
            "site5": DispatchInfo(app_name="app9", resource_requirements={}, token=None),
            "site6": DispatchInfo(app_name="app9", resource_requirements={}, token=None),
            "site7": DispatchInfo(app_name="app9", resource_requirements={}, token=None),
        },
    ),
]


@pytest.fixture(
    params=[{"num_sites": 3}],
)
def setup_and_teardown(request):
    num_sites = request.param["num_sites"]
    sites = [Site(name=f"site{i}", resources=create_resource(1, 1)) for i in range(num_sites)]
    servers = create_servers(server_num=1, sites=sites)
    scheduler = DefaultJobScheduler(max_jobs=1)
    yield servers, scheduler, num_sites


class TestDefaultJobScheduler:
    def test_weird_deploy_map(self, setup_and_teardown):
        servers, scheduler, num_sites = setup_and_teardown
        candidate = create_job(
            job_id="test_job",
            resource_spec={},
            deploy_map={"app5": []},
            min_sites=1,
        )
        with servers[0].new_context() as fl_ctx:
            job, dispatch_info = scheduler.schedule_job(job_candidates=[candidate], fl_ctx=fl_ctx)
        assert job is None

    def test_missing_deploy_map(self, setup_and_teardown):
        servers, scheduler, num_sites = setup_and_teardown
        candidate = create_job(
            job_id="test_job",
            resource_spec={},
            deploy_map=None,
            min_sites=1,
        )
        with pytest.raises(
            RuntimeError, match=re.escape("Job (test_job) does not have deploy_map, can't be scheduled.")
        ):
            with servers[0].new_context() as fl_ctx:
                _, _ = scheduler.schedule_job(job_candidates=[candidate], fl_ctx=fl_ctx)

    def test_less_active_than_min(self, setup_and_teardown):
        servers, scheduler, num_sites = setup_and_teardown
        candidate = create_job(
            job_id="job",
            resource_spec={},
            deploy_map={"app5": [ALL_SITES]},
            min_sites=num_sites + 1,
        )
        with servers[0].new_context() as fl_ctx:
            job, dispatch_info = scheduler.schedule_job(job_candidates=[candidate], fl_ctx=fl_ctx)
        assert job is None

    def test_require_sites_not_active(self, setup_and_teardown):
        servers, scheduler, num_sites = setup_and_teardown
        candidate = create_job(
            job_id="job",
            resource_spec={},
            deploy_map={"app5": [ALL_SITES]},
            min_sites=1,
            required_sites=[f"site{num_sites}"],
        )
        with servers[0].new_context() as fl_ctx:
            job, dispatch_info = scheduler.schedule_job(job_candidates=[candidate], fl_ctx=fl_ctx)
        assert job is None

    def test_require_sites_not_enough_resource(self, setup_and_teardown):
        servers, scheduler, num_sites = setup_and_teardown
        candidate = create_job(
            job_id="job",
            resource_spec={"site2": create_resource(2, 2)},
            deploy_map={"app5": [ALL_SITES]},
            min_sites=1,
            required_sites=["site2"],
        )
        with servers[0].new_context() as fl_ctx:
            job, dispatch_info = scheduler.schedule_job(job_candidates=[candidate], fl_ctx=fl_ctx)
        assert job is None

    def test_not_enough_sites_has_enough_resource(self, setup_and_teardown):
        servers, scheduler, num_sites = setup_and_teardown
        candidate = create_job(
            job_id="job",
            resource_spec={f"site{i}": create_resource(2, 2) for i in range(num_sites)},
            deploy_map={"app5": [ALL_SITES]},
            min_sites=2,
            required_sites=[],
        )
        with servers[0].new_context() as fl_ctx:
            job, dispatch_info = scheduler.schedule_job(job_candidates=[candidate], fl_ctx=fl_ctx)
        assert job is None

    @pytest.mark.parametrize("job_candidates,sites,expected_job,expected_dispatch_info", TEST_CASES)
    def test_normal_case(self, job_candidates, sites, expected_job, expected_dispatch_info):
        servers = create_servers(server_num=1, sites=sites)
        scheduler = DefaultJobScheduler(max_jobs=10)
        with servers[0].new_context() as fl_ctx:
            job, dispatch_info = scheduler.schedule_job(job_candidates=job_candidates, fl_ctx=fl_ctx)
        assert job == expected_job
        assert dispatch_info == expected_dispatch_info

    @pytest.mark.parametrize("add_first_job", [True, False])
    def test_a_list_of_jobs(self, add_first_job):
        num_sites = 8
        num_jobs = 5
        max_jobs_allow = 4
        resource_on_each_site = {"gpu": [0, 1]}

        sites: Dict[str, Site] = {
            f"site{i}": Site(
                name=f"site{i}",
                resources=resource_on_each_site,
                resource_manager=ListResourceManager(resources=resource_on_each_site),
            )
            for i in range(num_sites)
        }
        first_job = create_jobs(
            num_jobs=1,
            prefix="weird_job",
            resource_spec={"site0": {"gpu": 1}},
            deploy_map={"app": ["server", "site0"]},
            min_sites=1,
            required_sites=["site0"],
            meta={},
        )
        jobs = create_jobs(
            num_jobs=num_jobs,
            resource_spec={f"site{i}": {"gpu": 1} for i in range(num_sites)},
            deploy_map={"app": ["server"] + [f"site{i}" for i in range(num_sites)]},
            min_sites=num_sites,
            required_sites=[f"site{i}" for i in range(num_sites)],
            meta={},
        )
        if add_first_job:
            jobs = first_job + jobs
        servers = create_servers(server_num=1, sites=list(sites.values()))
        scheduler = DefaultJobScheduler(max_jobs=max_jobs_allow)
        submitted_jobs = list(jobs)
        results = []
        for i in range(10):
            with servers[0].new_context() as fl_ctx:
                job, dispatch_infos = scheduler.schedule_job(job_candidates=submitted_jobs, fl_ctx=fl_ctx)
                if job:
                    submitted_jobs.remove(job)
                    results.append(job)
                    for site_name, dispatch_info in dispatch_infos.items():
                        if site_name != "server":
                            sites[site_name].resource_manager.allocate_resources(
                                dispatch_info.resource_requirements, token=dispatch_info.token, fl_ctx=fl_ctx
                            )
        assert results == [jobs[0], jobs[1]]
