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

"""Deterministic daemon/Cell doubles; these do not qualify a Docker deployment."""

import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import TaskResultReceipt
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.apis.task_launcher_spec import (
    TaskExecutionPhase,
    TaskLaunchError,
    TaskLaunchRequest,
    TaskResourceRequest,
    TaskSettlementError,
    UnsupportedTaskResourceError,
)
from nvflare.app_opt.job_launcher import docker_launcher as backend
from nvflare.app_opt.task_launcher.docker_launcher import (
    CHANNEL,
    SESSION_ENV,
    DockerTaskLauncher,
    decode_control,
    encode_control,
)
from nvflare.app_opt.task_launcher.docker_service import OWNER_LABEL, DockerTaskService, validate_worker_config
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.utils import decode_payload, encode_payload
from nvflare.fuel.f3.message import Message
from nvflare.fuel.utils.fobs import fobs as fobs_registry
from nvflare.private.fed.client.task_execution_policy import TaskExecutionPolicy
from nvflare.private.fed.task_worker.artifacts import FileTaskArtifactStore, IncompleteTaskArtifactError
from nvflare.private.fed.task_worker.protocol import (
    TaskAttemptIdentity,
    WorkerBootstrap,
    read_bootstrap,
    write_bootstrap,
)
from nvflare.private.fed.task_worker.supervisor import TaskSupervisor
from nvflare.private.fed.utils.fed_utils import nvflare_fobs_initialize


class MissingContainer(Exception):
    pass


class NameConflict(Exception):
    status_code = 409


class Container:
    def __init__(self, daemon, kwargs):
        self.daemon, self.kwargs = daemon, kwargs
        self.id = f"id-{len(daemon.creates)}-" + kwargs["name"]
        self.status = "created"
        self.attrs = {"Config": {"Labels": kwargs["labels"]}, "State": {"ExitCode": 0}}
        self.remove_error = None
        self.stop_error = None

    def start(self):
        self.status = "running"
        if self.daemon.on_start:
            self.daemon.on_start(self)

    def stop(self, timeout):
        assert timeout == 0
        if self.stop_error:
            raise self.stop_error
        self.status = "exited"
        self.attrs["State"]["ExitCode"] = 137

    def logs(self, **kwargs):
        assert kwargs["tail"] == 1000
        return b"task container diagnostics\n"

    def remove(self, force, v):
        assert force and v
        if self.remove_error:
            raise self.remove_error
        self.daemon.removals.append(self.id)
        del self.daemon.items[self.id]


class Daemon:
    def __init__(self):
        self.containers = self
        self.items, self.creates, self.removals = {}, [], []
        self.on_start = None
        self.create_error = None

    def create(self, image, **kwargs):
        self.creates.append((image, kwargs))
        if self.create_error:
            raise self.create_error
        if any(item.kwargs["name"] == kwargs["name"] for item in self.items.values()):
            raise NameConflict("container name is already reserved")
        container = Container(self, kwargs)
        self.items[container.id] = container
        return container

    def get(self, identity):
        for container in self.items.values():
            if identity in (container.id, container.kwargs["name"]):
                return container
        raise backend.docker.errors.NotFound("absent")


@pytest.fixture
def profile(tmp_path, monkeypatch):
    # Keep deterministic coverage independent of the optional Docker SDK and
    # of the resident launcher's separately scoped SDK double.
    monkeypatch.setattr(
        backend, "docker", SimpleNamespace(errors=SimpleNamespace(NotFound=MissingContainer)), raising=False
    )
    nvflare_fobs_initialize()
    daemon = Daemon()
    cp = SimpleNamespace(get_fqcn=lambda: "site", register_request_cb=Mock())
    config = {
        "image": "site-worker@sha256:trusted",
        "python_path": "/image/python",
        "environment": {"OMP_NUM_THREADS": "1"},
    }
    service = DockerTaskService(cp, daemon, config)
    mounts = [{"Type": "bind", "Source": "/daemon/site/job", "Target": str(tmp_path / "job"), "ReadOnly": False}]
    session = service.add_job("job", "site", str(tmp_path), mounts, "501:20")
    identity = TaskAttemptIdentity("job", "site", "task", "train", "attempt")
    store = FileTaskArtifactStore(str(tmp_path / "job/.nvflare/task-execution/attempts"))
    bootstrap = WorkerBootstrap(
        identity, store.root_dir, str(tmp_path), {"path": "nvflare.app_common.np.np_trainer.NPTrainer"}
    )
    request = TaskLaunchRequest(
        "job", "site", "task", "attempt", ("/cj/python", "--must-not-run"), environment={"SECRET": "ignored"}
    )

    def send_request(**kwargs):
        assert kwargs["target"] == "site" and kwargs["channel"] == CHANNEL
        message = kwargs["request"]
        message.set_header(MessageHeaderKey.ORIGIN, "site.job")
        message.set_header(MessageHeaderKey.TOPIC, kwargs["topic"])
        return service.dispatch(message)

    cj = SimpleNamespace(get_fqcn=lambda: "site.job", send_request=Mock(side_effect=send_request))
    monkeypatch.setenv(SESSION_ENV, session.token)
    launcher = DockerTaskLauncher()
    launcher.handle_event(EventType.START_RUN, SimpleNamespace(get_engine=lambda: SimpleNamespace(get_cell=lambda: cj)))
    assert SESSION_ENV not in os.environ

    def stage():
        store.create_attempt(identity)
        store.write_input(identity, Shareable({"value": 1}))
        write_bootstrap(store.bootstrap_path(identity), bootstrap)

    def run(supervisor=None, abort_signal=None):
        return (supervisor or TaskSupervisor()).run(
            bootstrap=bootstrap,
            store=store,
            input_payload=Shareable({"value": 1}),
            runtime_root=str(tmp_path / "job/.nvflare/task-execution"),
            launcher=launcher,
            abort_signal=abort_signal or Signal(),
            request_factory=lambda *_: replace(request, resources=TaskResourceRequest(gpu_count=session.gpu_count)),
        )

    return SimpleNamespace(**locals())


def test_cpu_result_survives_removal_and_lost_receipt_does_not_rerun(profile):
    p = profile

    def complete(container):
        p.store.commit_result(p.identity, Shareable({"value": 2}))
        container.status = "exited"

    p.daemon.on_start = complete
    supervisor = TaskSupervisor()
    outcome = p.run(supervisor)
    assert outcome.result["value"] == 2 and outcome.execution_status.succeeded
    assert not p.daemon.items and len(p.daemon.removals) == 1
    directory = Path(p.store.attempt_dir(p.identity))
    assert (directory / "result.fobs").exists()
    assert (directory / "container.log").read_text() == "task container diagnostics\n"
    assert supervisor.read_saved_result(p.identity)["value"] == 2
    supervisor.record_receipt(p.identity, TaskResultReceipt.RETRY)
    assert (directory / "result.fobs").exists()
    supervisor.record_receipt(p.identity, TaskResultReceipt.RECEIVED)
    assert not (directory / "result.fobs").exists()
    assert (directory / "completion.json").exists() and (directory / "container.log").exists()
    assert len(p.daemon.creates) == 1


@pytest.mark.parametrize("gpu_count", [0, 1])
@pytest.mark.parametrize("cancel", [False, True])
def test_cpu_gpu_lifecycle_crosses_cell_wire_without_enum_registration(profile, monkeypatch, gpu_count, cancel):
    p = profile
    if gpu_count:
        _configure_gpu(p)
    phase_type = "nvflare.apis.task_launcher_spec.TaskExecutionPhase"
    monkeypatch.setattr(fobs_registry, "_enum_auto_registration", False)
    monkeypatch.setattr(
        fobs_registry, "_decomposers", {k: v for k, v in fobs_registry._decomposers.items() if k != phase_type}
    )
    monkeypatch.setattr(fobs_registry, "_type_name_whitelist", fobs_registry._type_name_whitelist - {phase_type})
    phases = []

    def through_cell_wire(**kwargs):
        encode_payload(kwargs["request"])
        decode_payload(kwargs["request"])
        reply = p.send_request(**kwargs)
        encode_payload(reply)
        decode_payload(reply)
        values = decode_control(reply.payload)
        assert type(values["status"]["phase"]) is str
        phases.append(values["status"]["phase"])
        return reply

    p.cj.send_request.side_effect = through_cell_wire
    p.stage()
    handle = p.launcher.launch_task(p.request)
    assert handle.poll().phase is TaskExecutionPhase.RUNNING
    if cancel:
        status = handle.cancel()
        assert status.cancel_requested
    else:
        _workers(p.daemon)[0].status = "exited"
        assert handle.poll().phase is TaskExecutionPhase.TERMINAL
        status = handle.wait_for_settlement(timeout=1)
    assert status.settled and status.phase is TaskExecutionPhase.TERMINAL
    assert not p.daemon.items
    assert phases[0] == "running" and phases[-1] == "terminal"
    assert phase_type not in fobs_registry._decomposers


def test_command_permissions_storage_and_environment_are_cp_owned(profile):
    p = profile
    p.stage()
    handle = p.launcher.launch_task(p.request)
    image, kwargs = p.daemon.creates[0]
    assert image == p.config["image"]
    assert kwargs["command"] == [
        "/image/python",
        "-u",
        "-m",
        "nvflare.private.fed.app.client.task_worker_container",
        "--bootstrap",
        p.store.bootstrap_path(p.identity),
    ]
    assert kwargs["entrypoint"] == []
    assert kwargs["user"] == "501:20" and kwargs["mounts"] == p.mounts
    assert kwargs["network_mode"] == "none" and kwargs["auto_remove"] is False
    assert kwargs["environment"]["NVIDIA_VISIBLE_DEVICES"] == "none"
    assert kwargs["environment"]["OMP_NUM_THREADS"] == "1"
    assert not {"SECRET", SESSION_ENV, "DOCKER_HOST"} & set(kwargs["environment"])
    assert not {"device_requests", "privileged", "pid_mode", "restart_policy", "volumes"} & set(kwargs)
    assert handle.execution_id.startswith("docker:id-")
    assert handle.cancel().settled
    with pytest.raises(RuntimeError, match="already launched"):
        p.launcher.launch_task(p.request)


@pytest.mark.parametrize("failure", ["remove", "stop", "disappear", "ownership"])
def test_cancel_response_or_disappearance_cannot_claim_settlement(profile, failure):
    p = profile
    p.stage()
    handle = p.launcher.launch_task(p.request)
    container = next(iter(p.daemon.items.values()))
    if failure == "remove":
        container.remove_error = RuntimeError("busy")
    elif failure == "stop":
        container.stop_error = RuntimeError("daemon unavailable")
    elif failure == "ownership":
        container.attrs["Config"]["Labels"][OWNER_LABEL] = "another owner"
    else:
        p.daemon.items.clear()
    with pytest.raises(TaskSettlementError):
        handle.cancel()
    assert not handle.poll().settled


def test_failed_worker_and_incomplete_output_retain_diagnostics(profile):
    p = profile

    def fail(container):
        p.store.stage_result(p.identity, Shareable({"partial": 1}))
        container.status = "exited"
        container.attrs["State"]["ExitCode"] = 7

    p.daemon.on_start = fail
    with pytest.raises(IncompleteTaskArtifactError):
        p.run()
    assert not p.daemon.items
    directory = Path(p.store.attempt_dir(p.identity))
    assert (directory / "result.fobs").exists() and (directory / "container.log").exists()
    assert not (directory / "completion.json").exists()


def test_create_response_loss_retains_handle_and_never_recreates(profile):
    p = profile
    p.stage()
    p.daemon.create_error = RuntimeError("create response lost")
    with pytest.raises(TaskLaunchError) as caught:
        p.launcher.launch_task(p.request)
    handle = caught.value.handle
    with pytest.raises(TaskSettlementError):
        handle.cancel()
    p.launcher._call("launch", p.request)
    assert len(p.daemon.creates) == 1


def test_lost_launch_rpc_response_can_cancel_the_retained_cp_execution(profile):
    p = profile
    p.stage()
    send = p.cj.send_request.side_effect

    def lose_response(**kwargs):
        reply = send(**kwargs)
        if kwargs["topic"] == "launch":
            raise TimeoutError("lost response")
        return reply

    p.cj.send_request.side_effect = lose_response
    with pytest.raises(TaskLaunchError) as caught:
        p.launcher.launch_task(p.request)
    assert caught.value.handle.cancel().settled
    assert not p.daemon.items and len(p.daemon.creates) == 1


def test_cancellation_before_delayed_launch_fences_physical_creation(profile):
    p = profile
    p.stage()
    assert p.launcher._call("cancel", p.request)["status"]["settled"]
    assert p.launcher.launch_task(p.request).poll().cancel_requested
    assert not p.daemon.creates


def test_late_completion_during_cancel_cannot_publish(profile):
    p = profile
    p.stage()
    handle = p.launcher.launch_task(p.request)
    container = next(iter(p.daemon.items.values()))

    def late_stop(timeout):
        p.store.commit_result(p.identity, Shareable({"too_late": 1}))
        container.status = "exited"

    container.stop = late_stop
    status = handle.cancel()
    assert status.settled and status.cancel_requested and not status.succeeded


@pytest.mark.parametrize("mutation", ["origin", "token", "site", "command", "traversal"])
def test_authority_rejects_forged_or_expanded_requests(profile, mutation):
    p = profile
    p.stage()
    message = Message(
        {MessageHeaderKey.ORIGIN: "site.job", MessageHeaderKey.TOPIC: "launch"},
        {"session": p.session.token, "identity": list(p.request.identity), "gpu_count": 0},
    )
    if mutation == "origin":
        message.set_header(MessageHeaderKey.ORIGIN, "site.other-job")
    elif mutation == "token":
        message.payload["session"] = "forged"
    elif mutation == "site":
        message.payload["identity"][1] = "other-site"
    elif mutation == "traversal":
        message.payload["identity"][3] = "../../other-attempt"
    else:
        message.payload["command"] = ["malicious"]
    message.payload = encode_control(message.payload)
    assert p.service.dispatch(message).get_header(MessageHeaderKey.RETURN_CODE) != ReturnCode.OK
    assert not p.daemon.creates


def test_unsupported_artifact_translation_fails_before_creation(profile):
    p = profile
    p.stage()
    path = Path(p.store.bootstrap_path(p.identity))
    bootstrap = replace(read_bootstrap(str(path)), artifact_root="/different/path")
    path.unlink()
    write_bootstrap(str(path), bootstrap)
    with pytest.raises(TaskLaunchError):
        p.launcher.launch_task(p.request)
    assert not p.daemon.creates


@pytest.mark.parametrize(
    "config",
    [
        {"image": "worker", "environment": {SESSION_ENV: "secret"}},
        {"image": "worker", "device_requests": []},
        {"image": "worker", "nano_cpus": True},
        {"image": "worker", "environment": {"CUDA_VISIBLE_DEVICES": "0"}},
    ],
)
def test_site_worker_profile_rejects_unsafe_or_invalid_fields(config):
    with pytest.raises(ValueError):
        validate_worker_config(config)


def test_gpu_requests_require_admission_and_do_not_create_containers(profile):
    with pytest.raises(TaskLaunchError):
        profile.launcher.launch_task(replace(profile.request, resources=TaskResourceRequest(gpu_count=1)))
    assert not profile.daemon.creates


def test_session_close_cancels_workers_and_fences_new_attempts(profile):
    p = profile
    p.stage()
    handle = p.launcher.launch_task(p.request)
    p.session.close()
    assert handle.poll().settled and not p.daemon.items
    request = replace(p.request, attempt_id="new")
    with pytest.raises(TaskLaunchError):
        p.launcher.launch_task(request)


def test_site_profile_requires_matching_cp_and_cj_backend(tmp_path):
    path = tmp_path / "resources.json"
    workspace = SimpleNamespace(get_resources_file_path=lambda: str(path))
    launcher = {"path": "nvflare.app_opt.task_launcher.docker_launcher.DockerTaskLauncher"}
    cp = {
        "path": "nvflare.app_opt.job_launcher.docker_launcher.ClientDockerJobLauncher",
        "args": {"task_worker_config": {"image": "worker"}},
    }
    path.write_text(json.dumps({"task_execution": {"launcher": launcher}}))
    with pytest.raises(ValueError, match="matching Docker"):
        TaskExecutionPolicy.from_workspace(workspace)
    path.write_text(json.dumps({"task_execution": {"launcher": launcher}, "components": [cp]}))
    assert TaskExecutionPolicy.from_workspace(workspace).launcher.launch_mode == "docker"
    path.write_text(json.dumps({"components": [cp]}))
    with pytest.raises(ValueError, match="must match"):
        TaskExecutionPolicy.from_workspace(workspace)


def test_shared_proxy_binds_once_for_multiple_executor_adapters(profile):
    p = profile
    context = SimpleNamespace(get_engine=lambda: SimpleNamespace(get_cell=lambda: p.cj))
    p.launcher.handle_event(EventType.START_RUN, context)
    assert p.launcher._session == p.session.token


def test_cancellation_after_commit_is_not_exposed_as_a_successful_outcome(profile):
    p = profile
    abort = Signal()

    def finish_after_abort(container):
        p.store.commit_result(p.identity, Shareable({"too_late": 1}))
        abort.trigger(True)
        container.status = "exited"

    p.daemon.on_start = finish_after_abort
    outcome = p.run(abort_signal=abort)
    assert outcome.aborted and outcome.result is None
    assert outcome.execution_status.settled and not p.daemon.items
    assert Path(p.store.attempt_dir(p.identity), "result.fobs").exists()


def test_retention_and_open_result_reader_survive_container_removal(profile):
    p = profile

    def complete(container):
        p.store.commit_result(p.identity, Shareable({"value": 2}))
        container.status = "exited"

    p.daemon.on_start = complete
    supervisor = TaskSupervisor(retain_artifacts=True)
    outcome = p.run(supervisor)
    payload = Path(p.store.attempt_dir(p.identity), "result.fobs")
    with payload.open("rb") as reader:
        supervisor.record_receipt(outcome.identity, TaskResultReceipt.TASK_CLOSED)
        assert reader.read() and payload.exists()
    assert not p.daemon.items and supervisor.read_saved_result(p.identity)["value"] == 2


def test_next_cpu_attempt_starts_while_previous_result_is_pending_receipt(profile):
    p = profile

    def complete(container):
        bootstrap = read_bootstrap(container.kwargs["command"][-1])
        p.store.commit_result(bootstrap.identity, Shareable({"attempt": bootstrap.identity.attempt_id}))
        container.status = "exited"

    p.daemon.on_start = complete
    first = p.run()
    assert first.execution_status.settled and not p.daemon.items
    previous = Path(p.store.attempt_dir(p.identity), "result.fobs")
    identity = replace(p.identity, attempt_id="second")
    supervisor = TaskSupervisor()
    outcome = supervisor.run(
        bootstrap=replace(p.bootstrap, identity=identity),
        store=p.store,
        input_payload=Shareable(),
        runtime_root=str(p.tmp_path / "job/.nvflare/task-execution"),
        launcher=p.launcher,
        abort_signal=Signal(),
        request_factory=lambda *_: replace(p.request, attempt_id="second"),
    )
    assert outcome.execution_status.settled and outcome.result["attempt"] == "second"
    assert previous.exists() and len(p.daemon.creates) == 2 and len(p.daemon.removals) == 2


def test_symlinked_attempt_diagnostics_cannot_write_into_another_job(profile):
    p = profile
    p.stage()
    handle = p.launcher.launch_task(p.request)
    other = p.tmp_path / "other-job"
    other.mkdir()
    attempt = Path(p.store.attempt_dir(p.identity))
    attempt.rename(attempt.with_name("saved"))
    attempt.symlink_to(other, target_is_directory=True)
    status = handle.cancel()
    assert status.settled and status.failure_reason and not (other / "container.log").exists()


def test_opaque_checkpoint_promotion_survives_container_and_attempt_cleanup(profile):
    p = profile

    def complete(container):
        Path(p.store.checkpoint_paths(p.identity)["output"]).write_bytes(b"native-opaque-state")
        reference = p.store.stage_checkpoint(p.identity)
        p.store.commit_result(p.identity, Shareable({"value": 2}), diagnostics={"checkpoint": reference.to_dict()})
        container.status = "exited"

    p.daemon.on_start = complete
    supervisor = TaskSupervisor()
    outcome = p.run(supervisor)
    checkpoint = p.tmp_path / "job/.nvflare/task-execution/checkpoints/native.checkpoint"
    p.store.commit_checkpoint(outcome.completion, str(checkpoint))
    supervisor.record_receipt(p.identity, TaskResultReceipt.RECEIVED)
    identity = replace(p.identity, attempt_id="next")
    p.store.create_attempt(identity)
    p.store.prepare_checkpoint(identity, str(checkpoint))
    assert Path(p.store.checkpoint_paths(identity)["input"]).read_bytes() == b"native-opaque-state"
    assert not p.daemon.items


def test_unsettled_execution_prevents_admission_of_another_attempt(profile):
    p = profile
    p.stage()
    first = p.launcher.launch_task(p.request)
    with pytest.raises(TaskLaunchError):
        p.launcher.launch_task(replace(p.request, attempt_id="another"))
    assert len(p.daemon.creates) == 1
    assert first.cancel().settled


def test_cleanup_recovery_does_not_reopen_a_fenced_job_session(profile):
    p = profile
    p.stage()
    handle = p.launcher.launch_task(p.request)
    container = next(iter(p.daemon.items.values()))
    container.remove_error = RuntimeError("daemon unavailable")
    with pytest.raises(TaskSettlementError):
        handle.cancel()
    container.remove_error = None
    assert handle.cancel().settled
    assert p.session.closed
    with pytest.raises(TaskLaunchError):
        p.launcher.launch_task(replace(p.request, attempt_id="after-recovery"))
    assert len(p.daemon.creates) == 1


def test_lost_create_acknowledgement_recovers_the_owned_native_container(profile):
    p = profile
    p.stage()
    create = p.daemon.create

    def lose_ack(*args, **kwargs):
        create(*args, **kwargs)
        raise RuntimeError("creation response lost")

    p.daemon.create = lose_ack
    with pytest.raises(TaskLaunchError) as caught:
        p.launcher.launch_task(p.request)
    status = caught.value.handle.poll()
    assert status.settled and status.cancel_requested and not p.daemon.items
    p.launcher._call("launch", p.request)
    assert len(p.daemon.creates) == 1


GPU_0 = "GPU-12345678-1234-1234-1234-123456789abc"
GPU_1 = "GPU-87654321-4321-4321-4321-cba987654321"


def _configure_gpu(profile, devices=(GPU_0,), count=1):
    from nvflare.app_opt.task_launcher.docker_service import validate_worker_config

    profile.session.config = validate_worker_config(
        {
            **profile.config,
            "gpu_partition": {"id": "approved-test-partition", "devices": list(devices), "exclusive": True},
        }
    )
    profile.session.gpu_count = count
    profile.request = replace(profile.request, resources=TaskResourceRequest(gpu_count=count))


def _workers(daemon):
    return [container for container in daemon.items.values() if OWNER_LABEL in container.kwargs["labels"]]


def _reservations(daemon):
    from nvflare.app_opt.task_launcher.docker_gpu import GPU_OWNER

    return [container for container in daemon.items.values() if GPU_OWNER in container.kwargs["labels"]]


def test_gpu_worker_uses_admitted_uuid_and_releases_it_after_whole_container_removal(profile):
    p = profile
    _configure_gpu(p)
    p.stage()
    handle = p.launcher.launch_task(p.request)
    worker = _workers(p.daemon)[0]
    reservation = _reservations(p.daemon)[0]
    assert reservation.status == "created"
    assert worker.kwargs["device_requests"] == [
        {
            "Driver": "nvidia",
            "DeviceIDs": [GPU_0],
            "Capabilities": [["gpu"]],
        }
    ]
    assert worker.kwargs["environment"]["NVIDIA_VISIBLE_DEVICES"] == GPU_0
    assert worker.kwargs["environment"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert not handle.poll().settled
    assert handle.cancel().settled and not p.daemon.items
    assert p.daemon.removals.index(worker.id) < p.daemon.removals.index(reservation.id)
    assert json.loads(Path(p.store.attempt_dir(p.identity), "resources.json").read_text())["gpu_devices"] == [GPU_0]


def test_gpu_reservation_release_is_part_of_settlement_and_later_cleanup_remains_fenced(profile):
    p = profile
    _configure_gpu(p)
    p.stage()
    handle = p.launcher.launch_task(p.request)
    reservation = _reservations(p.daemon)[0]
    reservation.remove_error = RuntimeError("reservation removal failed")
    with pytest.raises(TaskSettlementError):
        handle.cancel()
    assert not _workers(p.daemon) and _reservations(p.daemon)
    assert p.session.closed and not handle.poll().settled
    reservation.remove_error = None
    assert handle.cancel().settled and not p.daemon.items
    assert p.session.closed


def test_gpu_capacity_is_not_released_after_unconfirmed_worker_removal(profile):
    p = profile
    _configure_gpu(p)
    p.stage()
    handle = p.launcher.launch_task(p.request)
    worker = _workers(p.daemon)[0]
    worker.remove_error = RuntimeError("worker removal failed")
    with pytest.raises(TaskSettlementError):
        handle.cancel()
    assert _reservations(p.daemon) and _workers(p.daemon)
    worker.remove_error = None
    assert handle.cancel().settled and not p.daemon.items


def test_gpu_reservation_creation_response_loss_does_not_leak_or_relaunch(profile):
    p = profile
    _configure_gpu(p)
    p.stage()
    create = p.daemon.create

    def lose_ack(image, **kwargs):
        container = create(image, **kwargs)
        if kwargs["name"].startswith("nvflare-gpu-"):
            raise RuntimeError("reservation response lost")
        return container

    p.daemon.create = lose_ack
    handle = p.launcher.launch_task(p.request)
    assert len(_workers(p.daemon)) == 1 and len(_reservations(p.daemon)) == 1
    assert handle.cancel().settled and not p.daemon.items


def test_missing_reservation_ack_and_record_never_authorize_capacity_release(profile):
    p = profile
    _configure_gpu(p)
    p.stage()
    p.daemon.create_error = TimeoutError("reservation creation response lost")
    with pytest.raises(TaskLaunchError) as caught:
        p.launcher.launch_task(p.request)
    handle = caught.value.handle
    with pytest.raises(TaskSettlementError):
        handle.cancel()
    assert not handle.poll().settled
    assert not any(OWNER_LABEL in kwargs["labels"] for _, kwargs in p.daemon.creates)


def _competing_session(p, count=1):
    from nvflare.app_opt.task_launcher.docker_service import _JobSession

    session = _JobSession(p.daemon, "other-job", "site", str(p.tmp_path), p.mounts, p.session.config, "501:20", count)
    store = FileTaskArtifactStore(str(p.tmp_path / "other-job/.nvflare/task-execution/attempts"))

    def stage(attempt):
        identity = TaskAttemptIdentity("other-job", "site", "task", "train", attempt)
        store.create_attempt(identity)
        store.write_input(identity, Shareable())
        bootstrap = replace(p.bootstrap, identity=identity, artifact_root=store.root_dir)
        write_bootstrap(store.bootstrap_path(identity), bootstrap)
        return identity

    return session, store, stage


def test_daemon_wide_admission_and_reuse_before_the_previous_result_is_uploaded(profile):
    p = profile
    _configure_gpu(p)
    p.stage()
    handle = p.launcher.launch_task(p.request)
    session, _, stage = _competing_session(p)
    first = stage("competing")
    execution = session.launch((first.job_id, first.site_name, first.task_id, first.attempt_id))
    assert execution.poll().settled and execution.status.failure_reason
    assert len(_workers(p.daemon)) == 1 and len(_reservations(p.daemon)) == 1

    p.store.commit_result(p.identity, Shareable({"gpu_result": 1}))
    _workers(p.daemon)[0].status = "exited"
    assert handle.wait_for_settlement(timeout=1).settled
    assert Path(p.store.attempt_dir(p.identity), "result.fobs").exists()
    # No receipt/upload has occurred. A different CP/job can now own the same UUID.
    second = stage("after-release")
    execution = session.launch((second.job_id, second.site_name, second.task_id, second.attempt_id))
    assert execution.poll().phase.value == "running"
    assert len(_workers(p.daemon)) == 1
    assert _workers(p.daemon)[0].kwargs["device_requests"][0]["DeviceIDs"] == [GPU_0]
    assert execution.cancel().settled and not p.daemon.items


def test_multi_device_admission_rolls_back_partial_reservations_on_insufficient_capacity(profile):
    p = profile
    _configure_gpu(p, devices=(GPU_0, GPU_1))
    p.stage()
    handle = p.launcher.launch_task(p.request)
    session, _, stage = _competing_session(p, count=2)
    identity = stage("needs-two")
    execution = session.launch((identity.job_id, identity.site_name, identity.task_id, identity.attempt_id))
    assert execution.poll().settled and execution.status.failure_reason
    assert len(_reservations(p.daemon)) == 1
    assert len(_workers(p.daemon)) == 1
    assert handle.cancel().settled and not p.daemon.items


def test_lost_native_worker_creation_ack_keeps_gpu_reserved_until_worker_is_reconciled(profile):
    p = profile
    _configure_gpu(p)
    p.stage()
    create = p.daemon.create

    def lose_worker_ack(image, **kwargs):
        if OWNER_LABEL in kwargs["labels"]:
            raise TimeoutError("worker create may still be pending")
        return create(image, **kwargs)

    p.daemon.create = lose_worker_ack
    with pytest.raises(TaskLaunchError) as caught:
        p.launcher.launch_task(p.request)
    handle = caught.value.handle
    with pytest.raises(TaskSettlementError):
        handle.cancel()
    assert len(_reservations(p.daemon)) == 1 and not _workers(p.daemon)
    assert not handle.poll().settled


@pytest.mark.parametrize(
    "partition",
    [
        {"id": "unapproved", "devices": [GPU_0], "exclusive": False},
        {"id": "ordinal", "devices": ["0"], "exclusive": True},
        {"id": "all", "devices": ["all"], "exclusive": True},
        {"id": "duplicate", "devices": [GPU_0, GPU_0], "exclusive": True},
        {"id": "invalid", "devices": ["GPU-short"], "exclusive": True},
    ],
)
def test_gpu_partition_requires_approved_distinct_physical_uuids(partition):
    with pytest.raises(ValueError):
        validate_worker_config({"image": "worker", "gpu_partition": partition})


def test_cpu_job_does_not_reserve_gpus_even_when_a_partition_is_configured(profile):
    p = profile
    _configure_gpu(p, count=0)
    p.stage()
    handle = p.launcher.launch_task(p.request)
    assert not _reservations(p.daemon)
    assert "device_requests" not in _workers(p.daemon)[0].kwargs
    assert handle.cancel().settled


def test_gpu_count_is_bound_to_cp_metadata_not_the_cj_request(profile):
    p = profile
    _configure_gpu(p)
    p.stage()
    with pytest.raises(TaskLaunchError):
        p.launcher.launch_task(replace(p.request, resources=TaskResourceRequest(gpu_count=0)))
    assert not p.daemon.creates


def test_policy_resolves_docker_gpu_resources_but_process_keeps_its_existing_guard():
    meta = {"launcher_spec": {"default": {"docker": {"num_of_gpus": 1}}}}
    policy = TaskExecutionPolicy.from_config(
        {
            "launcher": {"path": "nvflare.app_opt.task_launcher.docker_launcher.DockerTaskLauncher"},
        }
    )
    assert policy.task_resources(meta, "site").gpu_count == 1
    override = {"launcher_spec": {**meta["launcher_spec"], "site": {"docker": {"num_of_gpus": 0}}}}
    assert policy.task_resources(override, "site").gpu_count == 0
    with pytest.raises(ValueError, match="GPUs"):
        TaskExecutionPolicy.from_config({}).task_resources(meta, "site")
    with pytest.raises(ValueError, match="portable GPUs reserve"):
        policy.task_resources({"resource_spec": {"@default": {"num_of_gpus": 1}}}, "site")


def test_cpu_memory_requests_do_not_claim_unimplemented_host_reservation(profile):
    with pytest.raises(UnsupportedTaskResourceError):
        profile.launcher.launch_task(replace(profile.request, resources=TaskResourceRequest(memory_mb=512)))
    assert not profile.daemon.creates


def test_gpu_resource_session_requires_a_partition_before_any_launch(profile):
    p = profile
    with pytest.raises(ValueError, match="approved exclusive gpu_partition"):
        p.service.add_job(
            "gpu-job",
            "site",
            str(p.tmp_path),
            p.mounts,
            "501:20",
            job_meta={"launcher_spec": {"site": {"docker": {"num_of_gpus": 1}}}},
        )
    assert not p.daemon.creates


def test_portable_gpu_reservation_is_rejected_before_cj_admission(profile):
    p = profile
    _configure_gpu(p)
    with pytest.raises(ValueError, match="portable GPUs reserve"):
        p.service.add_job(
            "portable-gpu-job",
            "site",
            str(p.tmp_path),
            p.mounts,
            "501:20",
            job_meta={"resource_spec": {"@default": {"num_of_gpus": 1}}},
        )
    assert "portable-gpu-job" not in p.service.sessions
    assert not p.daemon.creates


def test_resident_docker_jobs_cannot_bypass_the_opted_in_task_partition():
    from nvflare.app_opt.task_launcher.docker_gpu import validate_partition, validate_resident_access

    partition = validate_partition({"id": "approved", "devices": [GPU_0], "exclusive": True})
    for kwargs in (
        {"device_requests": [{"Count": 1, "Capabilities": [["gpu"]]}]},
        {"device_requests": [{"DeviceIDs": [GPU_0]}]},
        {"device_requests": [{"DeviceIDs": ["all"]}]},
        {"privileged": True},
        {"devices": ["/dev/nvidia0"]},
    ):
        with pytest.raises(ValueError):
            validate_resident_access(partition, kwargs)
    validate_resident_access(partition, {"device_requests": [{"DeviceIDs": [GPU_1], "Count": 0}]})
    validate_resident_access(partition, {})


def test_reservation_ownership_loss_never_authorizes_releasing_another_execution(profile):
    from nvflare.app_opt.task_launcher.docker_gpu import GPU_OWNER

    p = profile
    _configure_gpu(p)
    p.stage()
    handle = p.launcher.launch_task(p.request)
    reservation = _reservations(p.daemon)[0]
    reservation.kwargs["labels"][GPU_OWNER] = "another-owner"
    with pytest.raises(TaskSettlementError):
        handle.cancel()
    assert reservation.id in p.daemon.items
    assert reservation.id not in p.daemon.removals
    assert not _workers(p.daemon)


def test_gpu_failure_before_committed_result_releases_owned_devices_and_retains_diagnostics(profile):
    p = profile
    _configure_gpu(p)

    def fail(container):
        p.store.stage_result(p.identity, Shareable({"partial": True}))
        container.status = "exited"
        container.attrs["State"]["ExitCode"] = 7

    p.daemon.on_start = fail
    with pytest.raises(IncompleteTaskArtifactError):
        p.run()
    assert not p.daemon.items
    assert Path(p.store.attempt_dir(p.identity), "result.fobs").exists()
    assert Path(p.store.attempt_dir(p.identity), "resources.json").exists()


def test_failed_gpu_admission_cannot_accept_a_preexisting_completion(profile):
    from nvflare.app_opt.task_launcher.docker_gpu import DockerGpuReservations

    p = profile
    _configure_gpu(p)
    external = DockerGpuReservations(p.daemon, p.session.config["gpu_partition"], "worker", "501:20", "another-owner")
    external.acquire(1)
    launch = p.launcher.launch_task

    def premature_completion(request):
        p.store.commit_result(p.identity, Shareable({"must_not_publish": 1}))
        return launch(request)

    p.launcher.launch_task = premature_completion
    with pytest.raises(TaskLaunchError, match="insufficient free capacity"):
        p.run()
    assert not _workers(p.daemon)
    assert Path(p.store.attempt_dir(p.identity), "completion.json").exists()
    external.release()
    assert not p.daemon.items


def test_rejected_worker_create_releases_admitted_gpu_but_unknown_responses_do_not(profile):
    p = profile
    _configure_gpu(p)
    p.stage()
    create = p.daemon.create

    class Rejected(Exception):
        status_code = 400

    def reject_worker(image, **kwargs):
        if OWNER_LABEL in kwargs["labels"]:
            raise Rejected("invalid worker container configuration")
        return create(image, **kwargs)

    p.daemon.create = reject_worker
    with pytest.raises(TaskLaunchError) as caught:
        p.launcher.launch_task(p.request)
    assert caught.value.handle.poll().settled
    assert not p.daemon.items
    with pytest.raises(RuntimeError, match="already launched"):
        p.launcher.launch_task(p.request)
