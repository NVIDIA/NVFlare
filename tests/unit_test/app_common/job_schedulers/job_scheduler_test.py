# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
from unittest.mock import ANY, Mock

import pytest

import nvflare.app_common.job_schedulers.job_scheduler as job_scheduler_module
from nvflare.apis.client import Client
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext, FLContextManager
from nvflare.apis.job_def import ALL_SITES, Job, JobMetaKey, RunStatus
from nvflare.apis.job_def_manager_spec import JobDefManagerSpec
from nvflare.apis.job_scheduler_spec import DispatchInfo
from nvflare.apis.resource_manager_spec import ResourceManagerSpec
from nvflare.apis.server_engine_spec import ServerEngineSpec
from nvflare.app_common.job_schedulers.job_scheduler import DefaultJobScheduler
from nvflare.app_common.resource_managers.gpu_resource_manager import GPUResourceManager
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

    def report_resources(self, fl_ctx):
        return self.resources


class Site:
    def __init__(self, name, resources, resource_manager=None):
        self.name = name
        if resource_manager:
            self.resource_manager = resource_manager
        else:
            self.resource_manager = DummyResourceManager(name=name, resources=resources)


class MockServerEngine(ServerEngineSpec):
    def __init__(self, clients: dict[str, Site], run_name="exp1"):
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

    def validate_targets(self, client_names: list[str]):
        pass

    def new_context(self):
        return self.fl_ctx_mgr.new_context()

    def get_workspace(self):
        pass

    def add_component(self, component_id: str, component):
        pass

    def get_component(self, component_id: str) -> object:
        pass

    def register_aux_message_handler(self, topic: str, message_handle_func):
        pass

    def send_aux_request(
        self, targets: [], topic: str, request, timeout: float, fl_ctx: FLContext, optional=False, secure=False
    ) -> dict:
        pass

    def multicast_aux_requests(
        self,
        topic: str,
        target_requests,
        timeout: float,
        fl_ctx: FLContext,
        optional: bool = False,
        secure: bool = False,
    ) -> dict:
        pass

    def get_widget(self, widget_id: str):
        pass

    def persist_components(self, fl_ctx: FLContext, completed: bool):
        pass

    def restore_components(self, snapshot, fl_ctx: FLContext):
        pass

    def start_client_job(self, job, client_sites, fl_ctx: FLContext):
        pass

    def check_client_resources(
        self, job: Job, resource_reqs: dict[str, dict], fl_ctx: FLContext
    ) -> dict[str, tuple[bool, Optional[str]]]:
        result = {}
        with self.new_context() as fl_ctx:
            for site_name, requirements in resource_reqs.items():
                result[site_name] = self.clients[site_name].resource_manager.check_resources(requirements, fl_ctx)
        return result

    def get_client_name_from_token(self, token):
        return self.clients.get(token)

    def cancel_client_resources(
        self, resource_check_results: dict[str, tuple[bool, str]], resource_reqs: dict[str, dict], fl_ctx: FLContext
    ):
        # with self.new_context() as fl_ctx:
        for site_name, result in resource_check_results.items():
            check_result, token = result
            if check_result and token:
                self.clients[site_name].resource_manager.cancel_resources(
                    resource_requirement=resource_reqs[site_name], token=token, fl_ctx=fl_ctx
                )

    def update_job_run_status(self):
        pass


class _FakeStudyRegistry:
    def __init__(self, sites=None):
        self.sites = sites or {}

    def get_sites(self, study):
        return self.sites.get(study)


class _FakeStudyRegistryService:
    registry = None

    @staticmethod
    def get_registry():
        return _FakeStudyRegistryService.registry


def create_servers(server_num, sites: list[Site]):
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
    job_manager = Mock(spec=JobDefManagerSpec)
    yield servers, scheduler, num_sites, job_manager


class TestDefaultJobScheduler:
    def test_lifecycle_event_uses_explicit_job_id_over_sticky_job_id(self):
        scheduler = DefaultJobScheduler(max_jobs=20)
        fl_ctx = FLContext()
        fl_ctx.set_prop(FLContextKey.CURRENT_JOB_ID, "wrong-job")
        fl_ctx.set_prop(
            FLContextKey.EVENT_DATA,
            {JobMetaKey.JOB_ID.value: "right-job"},
            private=True,
            sticky=False,
        )

        scheduler.handle_event(EventType.JOB_STARTED, fl_ctx)
        assert scheduler.scheduled_jobs == ["right-job"]

        scheduler.handle_event(EventType.JOB_COMPLETED, fl_ctx)
        assert scheduler.scheduled_jobs == []

    def test_lifecycle_event_falls_back_to_sticky_job_id_without_event_data(self):
        scheduler = DefaultJobScheduler(max_jobs=20)
        fl_ctx = FLContext()
        fl_ctx.set_prop(FLContextKey.CURRENT_JOB_ID, "sticky-job")

        scheduler.handle_event(EventType.JOB_STARTED, fl_ctx)
        assert scheduler.scheduled_jobs == ["sticky-job"]

        scheduler.handle_event(EventType.JOB_ABORTED, fl_ctx)
        assert scheduler.scheduled_jobs == []

    def test_weird_deploy_map(self, setup_and_teardown):
        servers, scheduler, num_sites, job_manager = setup_and_teardown
        candidate = create_job(
            job_id="test_job",
            resource_spec={},
            deploy_map={"app5": []},
            min_sites=1,
        )
        with servers[0].new_context() as fl_ctx:
            job, dispatch_info = scheduler.schedule_job(
                job_manager=job_manager, job_candidates=[candidate], fl_ctx=fl_ctx
            )
        assert job is None

    def test_missing_deploy_map(self, setup_and_teardown):
        servers, scheduler, num_sites, job_manager = setup_and_teardown
        candidate = create_job(
            job_id="test_job",
            resource_spec={},
            deploy_map=None,
            min_sites=1,
        )

        with servers[0].new_context() as fl_ctx:
            _, _ = scheduler.schedule_job(job_manager=job_manager, job_candidates=[candidate], fl_ctx=fl_ctx)

            assert job_manager.set_status.called
            assert job_manager.set_status.call_args[0][1] == RunStatus.FINISHED_CANT_SCHEDULE

    def test_less_active_than_min(self, setup_and_teardown):
        servers, scheduler, num_sites, job_manager = setup_and_teardown
        candidate = create_job(
            job_id="job",
            resource_spec={},
            deploy_map={"app5": [ALL_SITES]},
            min_sites=num_sites + 1,
        )
        with servers[0].new_context() as fl_ctx:
            job, dispatch_info = scheduler.schedule_job(
                job_manager=job_manager, job_candidates=[candidate], fl_ctx=fl_ctx
            )
        assert job is None

    def test_require_sites_not_active(self, setup_and_teardown):
        servers, scheduler, num_sites, job_manager = setup_and_teardown
        candidate = create_job(
            job_id="job",
            resource_spec={},
            deploy_map={"app5": [ALL_SITES]},
            min_sites=1,
            required_sites=[f"site{num_sites}"],
        )
        with servers[0].new_context() as fl_ctx:
            job, dispatch_info = scheduler.schedule_job(
                job_manager=job_manager, job_candidates=[candidate], fl_ctx=fl_ctx
            )
        assert job is None

    def test_require_sites_duplicate_entries_are_treated_as_one(self, setup_and_teardown):
        servers, scheduler, num_sites, job_manager = setup_and_teardown
        candidate = create_job(
            job_id="job",
            resource_spec={},
            deploy_map={"app5": ["server", "site0"]},
            min_sites=1,
            required_sites=["site0", "site0"],
        )
        with servers[0].new_context() as fl_ctx:
            job, dispatch_info = scheduler.schedule_job(
                job_manager=job_manager, job_candidates=[candidate], fl_ctx=fl_ctx
            )
        assert job is candidate
        assert set(dispatch_info) == {"server", "site0"}

    @pytest.mark.parametrize(
        "required_sites",
        [
            pytest.param(1, id="invalid-container"),
            pytest.param([["site0"]], id="invalid-entry"),
        ],
    )
    def test_require_sites_invalid_metadata_does_not_interrupt_scheduling(
        self, monkeypatch, setup_and_teardown, required_sites
    ):
        servers, scheduler, num_sites, job_manager = setup_and_teardown
        monkeypatch.setattr(job_scheduler_module, "StudyRegistryService", _FakeStudyRegistryService, raising=False)
        monkeypatch.setattr(
            _FakeStudyRegistryService,
            "registry",
            _FakeStudyRegistry(sites={"cancer-research": {"site0"}}),
            raising=False,
        )
        malformed_candidate = create_job(
            job_id="malformed_job",
            resource_spec={},
            deploy_map={"app5": [ALL_SITES]},
            min_sites=1,
            required_sites=required_sites,
        )
        valid_candidate = create_job(
            job_id="valid_job",
            resource_spec={},
            deploy_map={"app5": ["server", "site0"]},
            min_sites=1,
        )
        malformed_candidate.meta[JobMetaKey.STUDY.value] = "cancer-research"
        valid_candidate.meta[JobMetaKey.STUDY.value] = "cancer-research"
        with servers[0].new_context() as fl_ctx:
            job, dispatch_info = scheduler.schedule_job(
                job_manager=job_manager,
                job_candidates=[malformed_candidate, valid_candidate],
                fl_ctx=fl_ctx,
            )
        assert job is valid_candidate
        assert set(dispatch_info) == {"server", "site0"}
        job_manager.set_status.assert_called_once_with(
            malformed_candidate.job_id, RunStatus.FINISHED_CANT_SCHEDULE, ANY
        )

    def test_require_sites_not_enough_resource(self, setup_and_teardown):
        servers, scheduler, num_sites, job_manager = setup_and_teardown
        candidate = create_job(
            job_id="job",
            resource_spec={"site2": create_resource(2, 2)},
            deploy_map={"app5": [ALL_SITES]},
            min_sites=1,
            required_sites=["site2"],
        )
        with servers[0].new_context() as fl_ctx:
            job, dispatch_info = scheduler.schedule_job(
                job_manager=job_manager, job_candidates=[candidate], fl_ctx=fl_ctx
            )
        assert job is None
        assert candidate.meta[JobMetaKey.SCHEDULE_COUNT.value] == 1
        assert (
            "required sites: ['site2'] don't have enough resources"
            in candidate.meta[JobMetaKey.SCHEDULE_HISTORY.value][0]
        )

    def test_not_enough_sites_has_enough_resource(self, setup_and_teardown):
        servers, scheduler, num_sites, job_manager = setup_and_teardown
        candidate = create_job(
            job_id="job",
            resource_spec={f"site{i}": create_resource(2, 2) for i in range(num_sites)},
            deploy_map={"app5": [ALL_SITES]},
            min_sites=2,
            required_sites=[],
        )
        with servers[0].new_context() as fl_ctx:
            job, dispatch_info = scheduler.schedule_job(
                job_manager=job_manager, job_candidates=[candidate], fl_ctx=fl_ctx
            )
        assert job is None

    def test_resource_manager_error_is_recorded_in_schedule_history(self):
        resource_manager = Mock(spec=ResourceManagerSpec)
        resource_manager.check_resources.return_value = (False, "resource check failed: unsupported license")
        candidate = create_job(
            job_id="job",
            resource_spec={"site1": {"license": 2}},
            deploy_map={"app": [ALL_SITES]},
            min_sites=1,
        )
        scheduler = DefaultJobScheduler(max_jobs=1, min_schedule_interval=0)

        with create_servers(1, [Site("site1", {}, resource_manager)])[0].new_context() as fl_ctx:
            job, _ = scheduler.schedule_job(Mock(spec=JobDefManagerSpec), [candidate], fl_ctx)

        assert job is None
        assert (
            "site1: resource check failed: unsupported license" in candidate.meta[JobMetaKey.SCHEDULE_HISTORY.value][0]
        )

    def test_unexpected_admission_error_cancels_resources_and_continues(self, monkeypatch):
        resource_manager = Mock(spec=ResourceManagerSpec)
        resource_manager.check_resources.side_effect = [(True, "first-token"), (True, "second-token")]
        server = create_servers(1, [Site("site1", {}, resource_manager)])[0]
        failed_candidate = create_job(
            job_id="failed-job",
            resource_spec={},
            deploy_map={"app": ["server", "site1"]},
            min_sites=1,
        )
        valid_candidate = create_job(
            job_id="valid-job",
            resource_spec={},
            deploy_map={"app": ["server", "site1"]},
            min_sites=1,
        )
        scheduler = DefaultJobScheduler(max_jobs=1, min_schedule_interval=0)
        job_manager = Mock(spec=JobDefManagerSpec)

        def fail_first_candidate(event_type, fl_ctx):
            if (
                event_type == EventType.AFTER_CHECK_CLIENT_RESOURCES
                and fl_ctx.get_prop(FLContextKey.CURRENT_JOB_ID) == failed_candidate.job_id
            ):
                raise RuntimeError("unexpected admission failure")

        monkeypatch.setattr(server, "fire_event", fail_first_candidate)

        with server.new_context() as fl_ctx:
            job, dispatch_info = scheduler.schedule_job(
                job_manager=job_manager,
                job_candidates=[failed_candidate, valid_candidate],
                fl_ctx=fl_ctx,
            )

        assert job is valid_candidate
        assert dispatch_info["site1"].token == "second-token"
        resource_manager.cancel_resources.assert_called_once_with(
            resource_requirement={}, token="first-token", fl_ctx=ANY
        )
        assert failed_candidate.meta[JobMetaKey.SCHEDULE_COUNT.value] == 1
        assert JobMetaKey.LAST_SCHEDULE_TIME.value in failed_candidate.meta
        assert (
            "unexpected admission error: RuntimeError: unexpected admission failure"
            in failed_candidate.meta[JobMetaKey.SCHEDULE_HISTORY.value][0]
        )
        job_manager.refresh_meta.assert_called_once_with(failed_candidate, scheduler._get_update_meta_keys(), ANY)

    @pytest.mark.parametrize(
        "failure_mode, expected_history",
        [
            ("resource-check-error", "failed before reservation results were available"),
            ("cancellation-error", "failed to cancel resources"),
        ],
    )
    def test_uncertain_admission_state_stops_candidate_scan(self, monkeypatch, failure_mode, expected_history):
        server = create_servers(1, [Site("site1", {})])[0]
        failed_candidate = create_job(
            job_id="failed-job",
            resource_spec={},
            deploy_map={"app": ["server", "site1"]},
            min_sites=1,
        )
        later_candidate = create_job(
            job_id="later-job",
            resource_spec={},
            deploy_map={"app": ["server", "site1"]},
            min_sites=1,
        )
        scheduler = DefaultJobScheduler(max_jobs=1, min_schedule_interval=0)
        job_manager = Mock(spec=JobDefManagerSpec)

        if failure_mode == "resource-check-error":
            check_client_resources = Mock(side_effect=RuntimeError("resource check failed"))
        else:
            check_client_resources = Mock(return_value={"site1": (True, "reservation-token")})
            monkeypatch.setattr(
                server,
                "cancel_client_resources",
                Mock(side_effect=RuntimeError("resource cancellation failed")),
            )

            def fail_admission(event_type, fl_ctx):
                if event_type == EventType.AFTER_CHECK_CLIENT_RESOURCES:
                    raise RuntimeError("unexpected admission failure")

            monkeypatch.setattr(server, "fire_event", fail_admission)

        monkeypatch.setattr(server, "check_client_resources", check_client_resources)

        with server.new_context() as fl_ctx:
            job, dispatch_info = scheduler.schedule_job(
                job_manager=job_manager,
                job_candidates=[failed_candidate, later_candidate],
                fl_ctx=fl_ctx,
            )

        assert job is None
        assert dispatch_info is None
        check_client_resources.assert_called_once()
        assert failed_candidate.meta[JobMetaKey.SCHEDULE_COUNT.value] == 1
        assert expected_history in failed_candidate.meta[JobMetaKey.SCHEDULE_HISTORY.value][0]
        assert JobMetaKey.SCHEDULE_COUNT.value not in later_candidate.meta
        job_manager.refresh_meta.assert_called_once_with(failed_candidate, scheduler._get_update_meta_keys(), ANY)

    def test_empty_resource_results_with_expected_replies_stop_candidate_scan(self, monkeypatch):
        server = create_servers(1, [Site("site1", {})])[0]
        failed_candidate = create_job(
            job_id="failed-job",
            resource_spec={},
            deploy_map={"app": ["server", "site1"]},
            min_sites=1,
        )
        later_candidate = create_job(
            job_id="later-job",
            resource_spec={},
            deploy_map={"app": ["server", "site1"]},
            min_sites=1,
        )
        scheduler = DefaultJobScheduler(max_jobs=1, min_schedule_interval=0)
        job_manager = Mock(spec=JobDefManagerSpec)
        check_client_resources = Mock(return_value={})
        monkeypatch.setattr(server, "check_client_resources", check_client_resources)

        with server.new_context() as fl_ctx:
            job, dispatch_info = scheduler.schedule_job(
                job_manager=job_manager,
                job_candidates=[failed_candidate, later_candidate],
                fl_ctx=fl_ctx,
            )

        assert job is None
        assert dispatch_info is None
        check_client_resources.assert_called_once()
        assert failed_candidate.meta[JobMetaKey.SCHEDULE_COUNT.value] == 1
        assert "returned no results" in failed_candidate.meta[JobMetaKey.SCHEDULE_HISTORY.value][0]
        assert JobMetaKey.SCHEDULE_COUNT.value not in later_candidate.meta
        job_manager.refresh_meta.assert_called_once_with(failed_candidate, scheduler._get_update_meta_keys(), ANY)

    def test_partial_resource_results_cancel_known_reservations_and_stop_candidate_scan(self, monkeypatch):
        resource_manager = Mock(spec=ResourceManagerSpec)
        server = create_servers(
            1,
            [
                Site("site1", {}, resource_manager),
                Site("site2", {}),
            ],
        )[0]
        failed_candidate = create_job(
            job_id="failed-job",
            resource_spec={},
            deploy_map={"app": ["server", "site1", "site2"]},
            min_sites=1,
        )
        later_candidate = create_job(
            job_id="later-job",
            resource_spec={},
            deploy_map={"app": ["server", "site1", "site2"]},
            min_sites=1,
        )
        scheduler = DefaultJobScheduler(max_jobs=1, min_schedule_interval=0)
        job_manager = Mock(spec=JobDefManagerSpec)
        check_client_resources = Mock(return_value={"site1": (True, "reservation-token")})
        monkeypatch.setattr(server, "check_client_resources", check_client_resources)

        with server.new_context() as fl_ctx:
            job, dispatch_info = scheduler.schedule_job(
                job_manager=job_manager,
                job_candidates=[failed_candidate, later_candidate],
                fl_ctx=fl_ctx,
            )

        assert job is None
        assert dispatch_info is None
        check_client_resources.assert_called_once()
        resource_manager.cancel_resources.assert_called_once_with(
            resource_requirement={}, token="reservation-token", fl_ctx=ANY
        )
        assert failed_candidate.meta[JobMetaKey.SCHEDULE_COUNT.value] == 1
        assert "missing sites: ['site2']" in failed_candidate.meta[JobMetaKey.SCHEDULE_HISTORY.value][0]
        assert JobMetaKey.SCHEDULE_COUNT.value not in later_candidate.meta
        job_manager.refresh_meta.assert_called_once_with(failed_candidate, scheduler._get_update_meta_keys(), ANY)

    def test_empty_resource_results_without_expected_replies_continue_candidate_scan(self):
        server = create_servers(1, [Site("site1", {})])[0]
        server_only_candidate = create_job(
            job_id="server-only-job",
            resource_spec={},
            deploy_map={"app": ["server"]},
            min_sites=0,
        )
        later_candidate = create_job(
            job_id="later-job",
            resource_spec={},
            deploy_map={"app": ["server", "site1"]},
            min_sites=1,
        )
        scheduler = DefaultJobScheduler(max_jobs=1, min_schedule_interval=0)
        job_manager = Mock(spec=JobDefManagerSpec)

        with server.new_context() as fl_ctx:
            job, dispatch_info = scheduler.schedule_job(
                job_manager=job_manager,
                job_candidates=[server_only_candidate, later_candidate],
                fl_ctx=fl_ctx,
            )

        assert job is later_candidate
        assert set(dispatch_info) == {"server", "site1"}
        assert server_only_candidate.meta[JobMetaKey.SCHEDULE_COUNT.value] == 1
        assert "error checking resources" in server_only_candidate.meta[JobMetaKey.SCHEDULE_HISTORY.value][0]
        job_manager.refresh_meta.assert_called_once_with(server_only_candidate, scheduler._get_update_meta_keys(), ANY)

    def test_unexpected_admission_error_honors_max_schedule_count(self, monkeypatch):
        resource_manager = Mock(spec=ResourceManagerSpec)
        resource_manager.check_resources.return_value = (True, "reservation-token")
        server = create_servers(1, [Site("site1", {}, resource_manager)])[0]
        candidate = create_job(
            job_id="failed-job",
            resource_spec={},
            deploy_map={"app": ["server", "site1"]},
            min_sites=1,
        )
        scheduler = DefaultJobScheduler(max_jobs=1, max_schedule_count=1, min_schedule_interval=0)
        job_manager = Mock(spec=JobDefManagerSpec)

        def fail_admission(event_type, fl_ctx):
            if event_type == EventType.AFTER_CHECK_CLIENT_RESOURCES:
                raise RuntimeError("unexpected admission failure")

        monkeypatch.setattr(server, "fire_event", fail_admission)

        for _ in range(2):
            with server.new_context() as fl_ctx:
                job, dispatch_info = scheduler.schedule_job(job_manager, [candidate], fl_ctx)

        assert job is None
        assert dispatch_info is None
        assert resource_manager.check_resources.call_count == 1
        job_manager.set_status.assert_called_once_with(candidate.job_id, RunStatus.FINISHED_CANT_SCHEDULE, ANY)

    def test_post_admission_bookkeeping_error_cancels_resources_and_continues(self, monkeypatch):
        resource_manager = Mock(spec=ResourceManagerSpec)
        resource_manager.check_resources.side_effect = [(True, "first-token"), (True, "second-token")]
        server = create_servers(1, [Site("site1", {}, resource_manager)])[0]
        failed_candidate = create_job(
            job_id="failed-job",
            resource_spec={},
            deploy_map={"app": ["server", "site1"]},
            min_sites=1,
        )
        later_candidate = create_job(
            job_id="later-job",
            resource_spec={},
            deploy_map={"app": ["server", "site1"]},
            min_sites=1,
        )
        scheduler = DefaultJobScheduler(max_jobs=1, min_schedule_interval=0)
        job_manager = Mock(spec=JobDefManagerSpec)
        update_schedule_history = scheduler._update_schedule_history
        update_calls = 0

        def fail_first_history_update(job, result, fl_ctx):
            nonlocal update_calls
            update_calls += 1
            if update_calls == 1:
                raise RuntimeError("history update failed")
            update_schedule_history(job, result, fl_ctx)

        monkeypatch.setattr(scheduler, "_update_schedule_history", fail_first_history_update)

        with server.new_context() as fl_ctx:
            job, dispatch_info = scheduler.schedule_job(
                job_manager=job_manager,
                job_candidates=[failed_candidate, later_candidate],
                fl_ctx=fl_ctx,
            )

        assert job is later_candidate
        assert dispatch_info["site1"].token == "second-token"
        resource_manager.cancel_resources.assert_called_once_with(
            resource_requirement={}, token="first-token", fl_ctx=ANY
        )
        assert failed_candidate.meta[JobMetaKey.SCHEDULE_COUNT.value] == 1
        assert (
            "unexpected admission error: RuntimeError: history update failed"
            in failed_candidate.meta[JobMetaKey.SCHEDULE_HISTORY.value][0]
        )
        job_manager.refresh_meta.assert_called_once_with(failed_candidate, scheduler._get_update_meta_keys(), ANY)

    @pytest.mark.parametrize("job_candidates,sites,expected_job,expected_dispatch_info", TEST_CASES)
    def test_normal_case(self, job_candidates, sites, expected_job, expected_dispatch_info):
        servers = create_servers(server_num=1, sites=sites)
        scheduler = DefaultJobScheduler(max_jobs=10, min_schedule_interval=0)
        job_manager = Mock(spec=JobDefManagerSpec)
        with servers[0].new_context() as fl_ctx:
            job, dispatch_info = scheduler.schedule_job(
                job_manager=job_manager, job_candidates=job_candidates, fl_ctx=fl_ctx
            )
        assert job == expected_job
        assert dispatch_info == expected_dispatch_info

    def test_portable_cpu_and_memory_bypass_gpu_resource_manager(self):
        sites = [
            Site(
                name="site1",
                resources={},
                resource_manager=GPUResourceManager(num_of_gpus=0, mem_per_gpu_in_GiB=0, ignore_host=True),
            ),
            Site(
                name="site2",
                resources={},
                resource_manager=GPUResourceManager(num_of_gpus=0, mem_per_gpu_in_GiB=0, ignore_host=True),
            ),
        ]
        candidate = create_job(
            job_id="portable",
            resource_spec={
                "@default": {"num_of_cpus": 2, "memory": "4Gi"},
                "site1": {"num_of_cpus": 4},
            },
            deploy_map={"app": [ALL_SITES]},
            min_sites=2,
        )
        scheduler = DefaultJobScheduler(max_jobs=1, min_schedule_interval=0)
        job_manager = Mock(spec=JobDefManagerSpec)

        with create_servers(1, sites)[0].new_context() as fl_ctx:
            job, dispatch_info = scheduler.schedule_job(job_manager, [candidate], fl_ctx)

        assert job == candidate
        assert dispatch_info["site1"].resource_requirements == {}
        assert dispatch_info["site2"].resource_requirements == {}

    def test_zero_gpu_with_gpu_memory_bypasses_gpu_resource_manager(self):
        sites = [
            Site(
                name="site1",
                resources={},
                resource_manager=GPUResourceManager(num_of_gpus=0, mem_per_gpu_in_GiB=0, ignore_host=True),
            )
        ]
        candidate = create_job(
            job_id="cpu-only",
            resource_spec={"site1": {"num_of_gpus": 0, "mem_per_gpu_in_GiB": 8}},
            deploy_map={"app": [ALL_SITES]},
            min_sites=1,
        )
        scheduler = DefaultJobScheduler(max_jobs=1, min_schedule_interval=0)

        with create_servers(1, sites)[0].new_context() as fl_ctx:
            job, dispatch_info = scheduler.schedule_job(Mock(spec=JobDefManagerSpec), [candidate], fl_ctx)

        assert job == candidate
        assert dispatch_info["site1"].resource_requirements == {}

    def test_cancellation_uses_resource_manager_requirements(self):
        sites = [
            Site(name="site1", resources={"license": 8}),
            Site(name="site2", resources={"license": 1}),
        ]
        candidate = create_job(
            job_id="portable-cancel",
            resource_spec={
                "@default": {"num_of_cpus": 2},
                "site1": {"license": 4},
                "site2": {"license": 2},
            },
            deploy_map={"app": [ALL_SITES]},
            min_sites=2,
        )
        scheduler = DefaultJobScheduler(max_jobs=1, min_schedule_interval=0)
        scheduler._cancel_resources = Mock()

        with create_servers(1, sites)[0].new_context() as fl_ctx:
            job, _ = scheduler.schedule_job(Mock(spec=JobDefManagerSpec), [candidate], fl_ctx)

        assert job is None
        assert scheduler._cancel_resources.call_args.kwargs["resource_reqs"] == {
            "site1": {"license": 4},
            "site2": {"license": 2},
        }

    @pytest.mark.parametrize("add_first_job", [True, False])
    def test_a_list_of_jobs(self, add_first_job):
        num_sites = 8
        num_jobs = 5
        max_jobs_allow = 4
        resource_on_each_site = {"gpu": [0, 1]}

        sites: dict[str, Site] = {
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
        scheduler = DefaultJobScheduler(max_jobs=max_jobs_allow, min_schedule_interval=0)
        job_manager = Mock(spec=JobDefManagerSpec)
        submitted_jobs = list(jobs)
        results = []
        for i in range(10):
            with servers[0].new_context() as fl_ctx:
                job, dispatch_infos = scheduler.schedule_job(
                    job_manager=job_manager, job_candidates=submitted_jobs, fl_ctx=fl_ctx
                )
                if job:
                    submitted_jobs.remove(job)
                    results.append(job)
                    for site_name, dispatch_info in dispatch_infos.items():
                        if site_name != "server":
                            sites[site_name].resource_manager.allocate_resources(
                                dispatch_info.resource_requirements, token=dispatch_info.token, fl_ctx=fl_ctx
                            )
        assert results == [jobs[0], jobs[1]]

    def test_failed_schedule_history(self, setup_and_teardown):
        servers, scheduler, num_sites, job_manager = setup_and_teardown
        candidate = create_job(
            job_id="job",
            resource_spec={},
            deploy_map={"app5": [ALL_SITES]},
            min_sites=num_sites + 1,
        )
        with servers[0].new_context() as fl_ctx:
            _, _ = scheduler.schedule_job(job_manager=job_manager, job_candidates=[candidate], fl_ctx=fl_ctx)
        assert candidate.meta[JobMetaKey.SCHEDULE_COUNT.value] == 1
        assert "connected sites (3) < min_sites (4)" in candidate.meta[JobMetaKey.SCHEDULE_HISTORY.value][0]

    def test_job_cannot_scheduled(self, setup_and_teardown):
        servers, scheduler, num_sites, job_manager = setup_and_teardown
        scheduler = DefaultJobScheduler(max_jobs=4, min_schedule_interval=0, max_schedule_count=2)
        candidate = create_job(
            job_id="job",
            resource_spec={},
            deploy_map={"app5": [ALL_SITES]},
            min_sites=num_sites + 1,
        )
        for i in range(3):
            with servers[0].new_context() as fl_ctx:
                _, _ = scheduler.schedule_job(job_manager=job_manager, job_candidates=[candidate], fl_ctx=fl_ctx)
        assert candidate.meta[JobMetaKey.SCHEDULE_COUNT.value] == 3
        assert job_manager.set_status.call_args[0][1] == RunStatus.FINISHED_CANT_SCHEDULE

    def test_all_sites_are_narrowed_to_study_enrolled_sites(self, monkeypatch, setup_and_teardown):
        servers, scheduler, num_sites, job_manager = setup_and_teardown
        monkeypatch.setattr(job_scheduler_module, "StudyRegistryService", _FakeStudyRegistryService, raising=False)
        monkeypatch.setattr(
            _FakeStudyRegistryService,
            "registry",
            _FakeStudyRegistry(sites={"cancer-research": {"site0", "site2"}}),
            raising=False,
        )

        candidate = create_job(
            job_id="study_job",
            resource_spec={"@default": {"num_of_cpus": 2}},
            deploy_map={"app5": [ALL_SITES]},
            min_sites=2,
        )
        candidate.meta[JobMetaKey.STUDY.value] = "cancer-research"

        with servers[0].new_context() as fl_ctx:
            job, dispatch_info = scheduler.schedule_job(
                job_manager=job_manager, job_candidates=[candidate], fl_ctx=fl_ctx
            )

        assert job == candidate
        assert set(dispatch_info) == {"server", "site0", "site2"}
        assert dispatch_info["site0"].resource_requirements == {}
        assert dispatch_info["site2"].resource_requirements == {}

    def test_required_out_of_study_site_blocks_job(self, monkeypatch, setup_and_teardown):
        servers, scheduler, num_sites, job_manager = setup_and_teardown
        monkeypatch.setattr(job_scheduler_module, "StudyRegistryService", _FakeStudyRegistryService, raising=False)
        monkeypatch.setattr(
            _FakeStudyRegistryService,
            "registry",
            _FakeStudyRegistry(sites={"cancer-research": {"site0"}}),
            raising=False,
        )

        candidate = create_job(
            job_id="blocked_job",
            resource_spec={"site0": create_resource(1, 1), "site2": create_resource(1, 1)},
            deploy_map={"app1": ["server", "site0", "site2"]},
            min_sites=1,
            required_sites=["site2"],
        )
        candidate.meta[JobMetaKey.STUDY.value] = "cancer-research"

        with servers[0].new_context() as fl_ctx:
            job, dispatch_info = scheduler.schedule_job(
                job_manager=job_manager, job_candidates=[candidate], fl_ctx=fl_ctx
            )

        assert job is None
        assert dispatch_info is None
        assert job_manager.set_status.called
