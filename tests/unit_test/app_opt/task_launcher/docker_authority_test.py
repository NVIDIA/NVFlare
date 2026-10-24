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

"""CP authority and cleanup with stateful Docker/Cell fault injection."""

import json
import shutil
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from nvflare.apis.fl_constant import FLMetaKey
from nvflare.apis.job_launcher_spec import JobReturnCode
from nvflare.apis.task_launcher_spec import TaskExecutionPhase, TaskLaunchError, TaskSettlementError
from nvflare.app_opt.task_launcher.docker_launcher import MAX_CONTROL_BYTES, decode_control, encode_control
from nvflare.app_opt.task_launcher.docker_service import OWNER_LABEL, TaskDockerJobHandle, restrict_task_cj
from nvflare.fuel.common.exit_codes import ProcessExitCode
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.message import Message
from nvflare.private.fed.task_worker.artifacts import FileTaskArtifactStore
from nvflare.private.fed.task_worker.protocol import write_bootstrap
from nvflare.private.fed.utils.fed_utils import get_return_code
from tests.unit_test.app_opt.task_launcher import docker_launcher_test as fixtures

profile = fixtures.profile
_configure_gpu = fixtures._configure_gpu
_reservations = fixtures._reservations


def _cj_handle(p, monkeypatch, *, background=False):
    cj = p.daemon.create(
        "cp-cj-image", name="owned-cj", labels={OWNER_LABEL: p.session.owner, "nvflare.task.resource": "client-job"}
    )
    cj.status = "running"
    with monkeypatch.context() as patch:
        if not background:
            patch.setattr(TaskDockerJobHandle, "_start_cleanup_monitor", lambda self: None)
        handle = TaskDockerJobHandle(
            container_id=cj.id, container_name="owned-cj", docker_client=p.daemon, task_session=p.session
        )
    return handle, cj


def _drain(handle):
    for _ in range(6):
        if handle._cleanup_done.is_set():
            break
        handle._cleanup_once()
    assert handle._cleanup_done.is_set()


@pytest.mark.parametrize("gpu_count", [0, 1])
@pytest.mark.parametrize("operation", ["launch", "poll", "cancel", "settle"])
@pytest.mark.parametrize("credentials", ["own", "other-token", "other-origin"])
def test_job_cannot_control_another_jobs_execution(profile, gpu_count, operation, credentials):
    p = profile
    if gpu_count:
        _configure_gpu(p)
    other = p.service.add_job("other-job", "site", str(p.tmp_path), p.mounts, "501:20")
    other.config, other.gpu_count = p.session.config, gpu_count
    identity = replace(p.identity, job_id="other-job")
    store = FileTaskArtifactStore(str(p.tmp_path / "other-job/.nvflare/task-execution/attempts"))
    store.create_attempt(identity)
    write_bootstrap(
        store.bootstrap_path(identity), replace(p.bootstrap, identity=identity, artifact_root=store.root_dir)
    )
    other.launch((identity.job_id, identity.site_name, identity.task_id, identity.attempt_id))
    before = list(p.daemon.items)
    token = other.token if credentials == "other-token" else p.session.token
    origin = "site.other-job" if credentials == "other-origin" else "site.job"
    request = Message(
        {MessageHeaderKey.ORIGIN: origin, MessageHeaderKey.TOPIC: operation},
        encode_control(
            {"session": token, "identity": ["other-job", "site", "task", "attempt"], "gpu_count": gpu_count}
        ),
    )
    reply = p.service.dispatch(request)
    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.INVALID_REQUEST
    assert list(p.daemon.items) == before and not p.daemon.removals
    assert not other.closed


@pytest.mark.parametrize(
    "payload",
    [
        {},
        "[]",
        '{"session":"one","session":"two"}',
        '{"gpu_count":NaN}',
        "x" * (MAX_CONTROL_BYTES + 1),
    ],
)
def test_authority_rejects_unbounded_or_non_json_commands(profile, payload):
    reply = profile.service.dispatch(
        Message({MessageHeaderKey.ORIGIN: "site.job", MessageHeaderKey.TOPIC: "launch"}, payload)
    )
    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.INVALID_REQUEST
    assert not profile.daemon.creates


@pytest.mark.parametrize(
    "field,value",
    [
        ("pid_mode", "host"),
        ("pid_mode", "container:other-job"),
        ("ipc_mode", "host"),
        ("ipc_mode", "container:other-job"),
        ("userns_mode", "host"),
        ("network_mode", "host"),
        ("cap_add", ["SYS_ADMIN"]),
        ("group_add", ["docker"]),
        ("security_opt", ["seccomp=unconfined"]),
    ],
)
def test_cj_cannot_inherit_broad_host_authority(field, value):
    with pytest.raises(ValueError):
        restrict_task_cj({field: value}, {})


@pytest.mark.parametrize("name", ["DOCKER_HOST", "DOCKER_CERT_PATH", "DOCKER_CONFIG", "DOCKER_CONTEXT"])
def test_cj_cannot_receive_daemon_credentials(name):
    with pytest.raises(ValueError, match="Docker control"):
        restrict_task_cj({}, {name: "cp-only-control-setting"})


@pytest.mark.parametrize("parent", ["job", ".nvflare", "task-execution", "attempts", "attempt"])
def test_cp_refuses_every_symlinked_job_path_parent(profile, parent):
    p = profile
    p.stage()
    path = Path(p.store.bootstrap_path(p.identity)).parent
    while path.name != parent:
        path = path.parent
    target = p.tmp_path / "another-jobs-data"
    shutil.copytree(path, target)
    path.rename(path.with_name(path.name + ".original"))
    path.symlink_to(target, target_is_directory=True)
    before = sorted(str(item.relative_to(target)) for item in target.rglob("*"))
    with pytest.raises(TaskLaunchError):
        p.launcher.launch_task(p.request)
    assert not p.daemon.creates
    assert sorted(str(item.relative_to(target)) for item in target.rglob("*")) == before


@pytest.mark.parametrize("gpu_count", [0, 1])
def test_parent_replacement_between_bootstrap_read_and_publication_is_rejected(profile, monkeypatch, gpu_count):
    from nvflare.app_opt.task_launcher import docker_service

    p = profile
    if gpu_count:
        _configure_gpu(p)
    p.stage()
    attempt = Path(p.store.bootstrap_path(p.identity)).parent
    other = p.tmp_path / "other-job"
    other.mkdir()
    publish = docker_service._publish_exclusive

    def swap_parent(directory, name, writer):
        attempt.rename(attempt.with_name("saved-attempt"))
        attempt.symlink_to(other, target_is_directory=True)
        return publish(directory, name, writer)

    monkeypatch.setattr(docker_service, "_publish_exclusive", swap_parent)
    with pytest.raises(TaskLaunchError):
        p.launcher.launch_task(p.request)
    assert not list(other.iterdir())
    assert not p.daemon.items
    assert all(kwargs["labels"]["nvflare.task.resource"] != "worker" for _, kwargs in p.daemon.creates)


@pytest.mark.parametrize(
    "gpu_count,stage",
    [
        (0, "worker-create"),
        (0, "worker-start"),
        (1, "worker-create"),
        (1, "worker-start"),
        (1, "gpu-reservation"),
    ],
)
def test_cancellation_during_blocked_native_launch_retains_ownership(profile, gpu_count, stage):
    p = profile
    if gpu_count:
        _configure_gpu(p)
    p.stage()
    entered, release, cancelled = threading.Event(), threading.Event(), threading.Event()
    create = p.daemon.create
    replies, errors = [], []

    def blocked_create(*args, **kwargs):
        resource = kwargs["labels"]["nvflare.task.resource"]
        if (stage == "worker-create" and resource == "worker") or (
            stage == "gpu-reservation" and resource == "gpu-reservation"
        ):
            entered.set()
            assert release.wait(5)
        container = create(*args, **kwargs)
        if stage == "worker-start" and resource == "worker":
            start = container.start

            def blocked_start():
                entered.set()
                assert release.wait(5)
                start()

            container.start = blocked_start
        return container

    def launch():
        try:
            p.launcher._call("launch", p.request)
        except BaseException as error:
            errors.append(error)

    def cancel():
        try:
            replies.append(p.launcher._call("cancel", p.request))
        finally:
            cancelled.set()

    p.daemon.create = blocked_create
    launch_thread = threading.Thread(target=launch)
    cancel_thread = threading.Thread(target=cancel)
    launch_thread.start()
    try:
        assert entered.wait(3)
        cancel_thread.start()
        assert cancelled.wait(2), "cancel blocked behind the mutating Docker API"
        assert replies[0]["status"]["cancel_requested"] and not replies[0]["status"]["settled"]
        if gpu_count and stage != "gpu-reservation":
            assert _reservations(p.daemon)
    finally:
        release.set()
        launch_thread.join(3)
        if cancel_thread.ident:
            cancel_thread.join(3)
    assert not errors and not launch_thread.is_alive()
    assert p.launcher._call("settle", p.request)["status"]["settled"]
    assert not p.daemon.items


@pytest.mark.parametrize("gpu_count", [0, 1])
def test_cp_retries_worker_cleanup_after_cj_death_without_cj_requests(profile, monkeypatch, gpu_count):
    p = profile
    if gpu_count:
        _configure_gpu(p)
    p.stage()
    p.launcher.launch_task(p.request)
    worker = next(
        item for item in p.daemon.items.values() if item.kwargs["labels"]["nvflare.task.resource"] == "worker"
    )
    handle, cj = _cj_handle(p, monkeypatch)
    worker.remove_error = RuntimeError("daemon remove response unavailable")
    cj.status = "exited"
    requests = p.cj.send_request.call_count
    with pytest.raises(RuntimeError, match="settlement unconfirmed"):
        handle._cleanup_once()
    assert handle.poll() == JobReturnCode.UNKNOWN
    assert p.session.closed and worker.id in p.daemon.items
    if gpu_count:
        assert _reservations(p.daemon)
    worker.remove_error = None
    _drain(handle)
    assert not p.daemon.items
    assert handle.poll() == ProcessExitCode.INFRASTRUCTURE_ERROR
    (p.tmp_path / "job" / FLMetaKey.PROCESS_RC_FILE).write_text("0\n")
    assert get_return_code(handle, "job", str(p.tmp_path), handle.logger) == ProcessExitCode.INFRASTRUCTURE_ERROR
    assert p.cj.send_request.call_count == requests


def test_cp_waiter_does_not_finish_on_cj_stop_api_failure(profile, monkeypatch):
    handle, cj = _cj_handle(profile, monkeypatch)
    finished = threading.Event()
    waiter = threading.Thread(target=lambda: (handle.wait(), finished.set()))
    waiter.start()
    try:
        cj.stop_error = RuntimeError("Docker API down")
        handle.terminate()
        with pytest.raises(RuntimeError, match="API down"):
            handle._cleanup_once()
        assert not finished.wait(0.05)
        assert handle.poll() == JobReturnCode.UNKNOWN and cj.status == "running"
        cj.stop_error = None
        _drain(handle)
        assert finished.wait(1)
        assert handle.poll() == JobReturnCode.ABORTED
    finally:
        handle._cleanup_done.set()
        waiter.join(1)


@pytest.mark.parametrize("lost_identity", ["missing", "other-owner", "other-id"])
def test_cj_disappearance_or_replacement_does_not_authorize_control_or_settlement(profile, monkeypatch, lost_identity):
    p = profile
    handle, cj = _cj_handle(p, monkeypatch)
    if lost_identity == "missing":
        del p.daemon.items[cj.id]
    elif lost_identity == "other-owner":
        cj.attrs["Config"]["Labels"][OWNER_LABEL] = "other-job-owner"
    else:
        original = p.daemon.get
        replacement = p.daemon.create("other-image", name="other", labels={OWNER_LABEL: p.session.owner})
        p.daemon.get = lambda identity: replacement if identity == cj.id else original(identity)
    handle.terminate()
    with pytest.raises(RuntimeError):
        handle._cleanup_once()
    assert handle.poll() == JobReturnCode.UNKNOWN and not p.daemon.removals
    assert cj.status == "running" and p.session.closed


def test_cp_adopts_only_its_owned_cj_after_lost_create_reply(profile, monkeypatch):
    p = profile
    with monkeypatch.context() as patch:
        patch.setattr(TaskDockerJobHandle, "_start_cleanup_monitor", lambda self: None)
        handle = TaskDockerJobHandle(
            container_id=None, container_name="owned-cj", docker_client=p.daemon, task_session=p.session
        )

    def lose_create_reply():
        p.daemon.create("cj-image", name="owned-cj", labels={OWNER_LABEL: p.session.owner})
        raise TimeoutError("create reply lost")

    handle.launch(lose_create_reply)
    assert handle.poll() == JobReturnCode.UNKNOWN
    _drain(handle)
    assert handle.poll() == JobReturnCode.ABORTED and not p.daemon.items


def test_missing_cj_after_unacknowledged_create_keeps_cp_ownership(profile, monkeypatch):
    p = profile
    with monkeypatch.context() as patch:
        patch.setattr(TaskDockerJobHandle, "_start_cleanup_monitor", lambda self: None)
        handle = TaskDockerJobHandle(
            container_id=None, container_name="owned-cj", docker_client=p.daemon, task_session=p.session
        )

    def lose_create_reply():
        raise TimeoutError("create may still be pending")

    handle.launch(lose_create_reply)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="disappeared"):
            handle._cleanup_once()
        assert handle.poll() == JobReturnCode.UNKNOWN
    assert p.service.sessions["job"].job_handle is handle
    assert p.session.closed


def test_shutdown_before_cj_creation_prevents_native_create(profile, monkeypatch):
    p = profile
    p.service.close()
    with monkeypatch.context() as patch:
        patch.setattr(TaskDockerJobHandle, "_start_cleanup_monitor", lambda self: None)
        handle = TaskDockerJobHandle(
            container_id=None, container_name="owned-cj", docker_client=p.daemon, task_session=p.session
        )
    handle.launch(lambda: pytest.fail("CP closed admission before CJ creation"))
    _drain(handle)
    assert not p.daemon.creates and handle.poll() == JobReturnCode.ABORTED


@pytest.mark.parametrize("gpu_count", [0, 1])
def test_lost_launch_reply_and_cj_death_cannot_discard_cp_execution(profile, monkeypatch, gpu_count):
    p = profile
    if gpu_count:
        _configure_gpu(p)
    handle, cj = _cj_handle(p, monkeypatch, background=True)
    send = p.cj.send_request.side_effect

    def lose_reply(**kwargs):
        send(**kwargs)
        cj.status = "exited"
        raise TimeoutError("CJ lost its launch reply and exited")

    p.cj.send_request.side_effect = lose_reply
    p.stage()
    try:
        with pytest.raises(TaskLaunchError):
            p.launcher.launch_task(p.request)
        assert handle._cleanup_done.wait(3)
        assert not p.daemon.items and p.session.closed
        assert p.cj.send_request.call_count == 1
    finally:
        handle.terminate()


def test_background_cp_owner_settles_workers_after_cj_exits(profile, monkeypatch):
    p = profile
    p.stage()
    p.launcher.launch_task(p.request)
    handle, cj = _cj_handle(p, monkeypatch, background=True)
    cj.status = "exited"
    try:
        assert handle._cleanup_done.wait(3)
        assert not p.daemon.items and p.session.closed
    finally:
        handle.terminate()


def test_parent_shutdown_closes_every_job_and_rejects_late_admission(profile, monkeypatch):
    p = profile
    p.stage()
    p.launcher.launch_task(p.request)
    handle, cj = _cj_handle(p, monkeypatch)
    p.service.close()
    assert p.session.closed
    _drain(handle)
    assert handle.poll() == JobReturnCode.ABORTED and not p.daemon.items
    with pytest.raises(RuntimeError, match="closing"):
        p.service.add_job("later", "site", str(p.tmp_path), p.mounts, "501:20")


@pytest.mark.parametrize("gpu_count", [0, 1])
def test_cj_death_during_blocked_worker_create_keeps_cp_owner(profile, monkeypatch, gpu_count):
    p = profile
    if gpu_count:
        _configure_gpu(p)
    handle, cj = _cj_handle(p, monkeypatch, background=True)
    p.stage()
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    create = p.daemon.create
    close = p.session.request_close
    replies = []

    def record_close():
        close()
        closed.set()

    def blocked_create(*args, **kwargs):
        if kwargs["labels"]["nvflare.task.resource"] == "worker":
            entered.set()
            assert release.wait(5)
        return create(*args, **kwargs)

    monkeypatch.setattr(p.session, "request_close", record_close)
    p.daemon.create = blocked_create
    launcher = threading.Thread(target=lambda: replies.append(p.launcher._call("launch", p.request)))
    launcher.start()
    try:
        assert entered.wait(2)
        cj.status = "exited"
        assert closed.wait(2)
        assert handle.poll() == JobReturnCode.UNKNOWN
        assert p.session.executions[p.request.identity].launching
        if gpu_count:
            assert _reservations(p.daemon)
        release.set()
        launcher.join(2)
        assert handle._cleanup_done.wait(3)
        assert not p.daemon.items
        assert replies and not launcher.is_alive()
    finally:
        release.set()
        launcher.join(2)
        handle.terminate()


def test_control_reply_is_bounded_json(profile):
    p = profile
    p.stage()
    reply = p.send_request(
        target="site",
        channel="nvflare_docker_task",
        topic="launch",
        request=Message(
            payload=encode_control({"session": p.session.token, "identity": list(p.request.identity), "gpu_count": 0})
        ),
    )
    assert isinstance(reply.payload, str)
    assert decode_control(reply.payload)["execution_id"].startswith("docker:")
    assert json.loads(reply.payload)["status"]["phase"] == "running"


def test_cp_backend_error_details_do_not_cross_the_cj_boundary(profile, monkeypatch):
    p = profile
    p.stage()
    handle = p.launcher.launch_task(p.request)
    original = p.daemon.get
    secret = "private-daemon-credential-that-must-stay-in-cp"

    def fail_observation(*_):
        raise RuntimeError(secret)

    monkeypatch.setattr(p.daemon, "get", fail_observation)
    status = handle.poll()
    assert status.phase == TaskExecutionPhase.RUNNING and not status.settled
    assert "RuntimeError" in status.failure_reason and secret not in status.failure_reason
    monkeypatch.setattr(p.daemon, "get", original)
    assert handle.cancel().settled


@pytest.mark.parametrize("gpu_count", [0, 1])
def test_worker_ownership_loss_remains_fenced_even_if_label_reappears(profile, gpu_count):
    p = profile
    if gpu_count:
        _configure_gpu(p)
    p.stage()
    handle = p.launcher.launch_task(p.request)
    worker = next(
        item for item in p.daemon.items.values() if item.kwargs["labels"]["nvflare.task.resource"] == "worker"
    )
    owner = worker.attrs["Config"]["Labels"][OWNER_LABEL]
    worker.attrs["Config"]["Labels"][OWNER_LABEL] = "other-job"
    status = handle.poll()
    assert status.phase == TaskExecutionPhase.TERMINAL and not status.settled
    with pytest.raises(TaskSettlementError):
        handle.cancel()
    worker.attrs["Config"]["Labels"][OWNER_LABEL] = owner
    with pytest.raises(TaskSettlementError):
        handle.cancel()
    assert worker.status == "running" and not p.daemon.removals
    if gpu_count:
        assert _reservations(p.daemon)
