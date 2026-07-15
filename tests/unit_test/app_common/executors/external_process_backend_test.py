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

"""Tests for the external_process backend of ClientAPIExecutor (mirrors the in_process
backend tests).

Drives the real control-protocol round trip the design defines for external_process: the
backend writes the bootstrap config and launches the trainer command, a fake trainer reads
that config exactly like the real one and HELLOs over the (fake) cell, TASK_READY carries a
direct Shareable, RESULT_READY returns a direct Shareable, and execute() returns it. Also
covers the backend-contract obligations: initialize()
self-unwinding, finalize() idempotency + process-tree teardown, bounded result wait, LOG
routing through the executor-owned fire_log_analytics(), the launch-token/identity checks
of the HELLO handshake, and direct Cell payload handling.
"""

import os
import signal
import subprocess
import threading
import time
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from nvflare.apis.analytix import AnalyticsDataType
from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_constant import FLContextKey, FLMetaKey, ReservedKey, ReturnCode, ServerCommandNames
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import UnsafeJobError
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.executors.client_api import external_process_backend as ebp
from nvflare.app_common.executors.client_api.backend_spec import ClientAPIBackendContext
from nvflare.app_common.executors.client_api.external_process_backend import ExternalProcessBackend, bootstrap_file_name
from nvflare.app_common.executors.client_api_executor import ClientAPIExecutor
from nvflare.client.cell.bootstrap import (
    BOOTSTRAP_FILE_ENV_VAR,
    BOOTSTRAP_SCHEMA_VERSION,
    EXTERNAL_PROCESS_EXECUTION_MODE,
    BootstrapKey,
    read_bootstrap_config,
)
from nvflare.client.cell.defs import CHANNEL, PROTOCOL_VERSION, MsgKey, Topic
from nvflare.client.config import ConfigKey, ExchangeFormat, TransferType
from nvflare.fuel.f3.cellnet.defs import CellChannel, MessageHeaderKey
from nvflare.fuel.f3.cellnet.defs import ReturnCode as CellReturnCode
from nvflare.fuel.f3.cellnet.utils import make_reply as make_cell_reply
from nvflare.fuel.f3.cellnet.utils import new_cell_message
from nvflare.fuel.utils.fobs.decomposers.via_downloader import LazyDownloadRef

CJ_FQCN = "site-1.job-1"

PROTOCOL_TOPICS = (Topic.HELLO, Topic.RESULT_READY, Topic.LOG, Topic.HEARTBEAT)


def _task_accepted_reply():
    return make_cell_reply(CellReturnCode.OK, body={MsgKey.REPLY_TOPIC: Topic.TASK_ACCEPTED})


def _result_shareable() -> Shareable:
    return DXO(data_kind=DataKind.WEIGHTS, data={"w": [1.0]}).to_shareable()


class FakeProcess:
    """Stands in for the Popen handle of the launched trainer tree."""

    _next_pid = 90000

    def __init__(self, args, kwargs):
        self.args = args
        self.kwargs = kwargs
        FakeProcess._next_pid += 1
        self.pid = FakeProcess._next_pid
        self.returncode = None
        self.stdout = None
        self.term_calls = []
        self.wait_timeouts = []
        # workers of the launched tree that outlive the leader (torchrun shape)
        self.extra_group_members = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)
        # no natural exit unless a test set returncode; never really sleeps
        if self.returncode is None:
            raise subprocess.TimeoutExpired(cmd=self.args, timeout=timeout or 0)
        return self.returncode

    def terminate(self):
        self.term_calls.append("terminate")
        self.returncode = -signal.SIGTERM

    def kill(self):
        self.term_calls.append("kill")
        self.returncode = -signal.SIGKILL

    def exit(self, rc):
        self.returncode = rc


class FakeCell:
    """The CJ cell: records protocol registrations/sends and lets tests deliver messages."""

    def __init__(self):
        self.fqcn = CJ_FQCN
        self.decode_pass_through_channels = set()
        self.decode_pass_through_topics = set()
        self.listener_url = "tcp://127.0.0.1:56789"
        self.internal_listener_made = False
        self.cbs = {}
        self.sent = []  # (topic, target, payload, timeout)
        self.sent_kwargs = []  # extra kwargs per send_request (e.g. the cancel abort_signal)
        self.fired = []  # (topic, targets, payload)
        self.on_request = None  # fn(topic, target, request) -> reply Message
        self.on_shutdown = None  # optional SHUTDOWN-specific reply hook
        self.on_fire = None  # fn(topic, targets, message)
        self.disconnected = set()

    def get_fqcn(self):
        return self.fqcn

    def is_cell_connected(self, target_fqcn):
        return target_fqcn not in self.disconnected

    def make_internal_listener(self):
        self.internal_listener_made = True

    def get_internal_listener_url(self):
        return self.listener_url

    def register_request_cb(self, channel, topic, cb):
        assert channel == CHANNEL
        self.cbs[topic] = cb

    def send_request(self, channel, topic, target, request, timeout=None, **kwargs):
        self.sent.append((topic, target, request.payload, timeout))
        self.sent_kwargs.append(kwargs)
        if topic == Topic.SHUTDOWN:
            if self.on_shutdown is not None:
                return self.on_shutdown(topic, target, request)
            return make_cell_reply(CellReturnCode.OK, body={MsgKey.RESULT_SOURCE_LIVE: False})
        if self.on_request is not None:
            return self.on_request(topic, target, request)
        return _task_accepted_reply()

    def fire_and_forget(self, channel, topic, targets, message, **kwargs):
        self.fired.append((topic, tuple(targets), message.payload))
        if self.on_fire is not None:
            self.on_fire(topic, targets, message)

    def deliver(self, topic, origin, payload):
        """Invokes the backend's registered handler as an incoming cell request."""
        cb = self.cbs[topic]
        return cb(new_cell_message({MessageHeaderKey.ORIGIN: origin}, payload))


class FakeTrainerHarness:
    """Patched in as subprocess.Popen: 'launches' a trainer that reads the bootstrap
    config exactly like the real one and performs the HELLO handshake synchronously."""

    def __init__(self, cell):
        self.cell = cell
        self.processes = []
        self.bootstrap_configs = []
        self.hello_replies = []
        self.killpg_calls = []
        self.auto_hello = True
        self.hello_mutator = None  # fn(payload) mutating the HELLO payload
        self.origin_override = None
        self.exit_at_launch_rc = None  # simulate a command that dies before HELLO

    def popen(self, args, **kwargs):
        proc = FakeProcess(args, kwargs)
        self.processes.append(proc)
        if self.exit_at_launch_rc is not None:
            proc.returncode = self.exit_at_launch_rc
            return proc
        if self.auto_hello:
            env = kwargs.get("env") or {}
            config = read_bootstrap_config(env[BOOTSTRAP_FILE_ENV_VAR])
            self.bootstrap_configs.append(config)
            payload = {
                MsgKey.TRAINER_FQCN: config[BootstrapKey.TRAINER_FQCN],
                MsgKey.PROOF: config[BootstrapKey.LAUNCH_TOKEN],
                MsgKey.PROTOCOL_VERSION: config[BootstrapKey.PROTOCOL_VERSION],
                MsgKey.JOB_ID: config[BootstrapKey.JOB_ID],
                MsgKey.SITE_NAME: config[BootstrapKey.SITE_NAME],
                MsgKey.RANK: 0,
            }
            if self.hello_mutator is not None:
                self.hello_mutator(payload)
            origin = self.origin_override or config[BootstrapKey.TRAINER_FQCN]
            self.hello_replies.append(self.cell.deliver(Topic.HELLO, origin, payload))
        return proc

    def killpg(self, pgid, sig):
        """Models the group: alive while the leader runs OR extra members survive it;
        sig 0 is the liveness probe; a real signal takes the whole group down."""
        self.killpg_calls.append((pgid, sig))
        for proc in self.processes:
            if proc.pid == pgid:
                alive = proc.returncode is None or proc.extra_group_members
                if not alive:
                    raise ProcessLookupError(pgid)
                if sig == 0:
                    return
                if proc.returncode is None:
                    proc.returncode = -sig
                proc.extra_group_members = False
                return
        raise ProcessLookupError(pgid)

    def signals_sent(self):
        """The non-probe signals delivered to process groups (excludes sig-0 probes)."""
        return [(pgid, sig) for pgid, sig in self.killpg_calls if sig != 0]


@pytest.fixture
def env(tmp_path, monkeypatch):
    cell = FakeCell()
    harness = FakeTrainerHarness(cell)
    holder = SimpleNamespace(
        cell=cell,
        harness=harness,
        app_dir=str(tmp_path),
    )
    # late-bound so a test may swap harness.popen (e.g. multi-HELLO launch sequences)
    monkeypatch.setattr(ebp.subprocess, "Popen", lambda *args, **kwargs: harness.popen(*args, **kwargs))
    monkeypatch.setattr(ebp, "log_subprocess_output", lambda process, logger: None)
    # start_new_session gives pgid == pid; route group signals to the fake process table
    monkeypatch.setattr(ebp.os, "killpg", harness.killpg, raising=False)
    return holder


def _make_engine(cell):
    engine = MagicMock()
    engine.get_cell.return_value = cell
    engine.new_context.return_value.__enter__.return_value = FLContext()
    return engine


def _make_fl_ctx(engine, app_dir):
    fl_ctx = FLContext()
    fl_ctx.put(key=ReservedKey.ENGINE, value=engine, private=True, sticky=False)
    fl_ctx.put(key=ReservedKey.RUN_NUM, value="job-1", private=False, sticky=False)
    fl_ctx.put(key=ReservedKey.IDENTITY_NAME, value="site-1", private=False, sticky=False)
    workspace = Mock()
    workspace.get_app_dir.return_value = app_dir
    workspace.get_app_custom_dir.return_value = app_dir
    fl_ctx.put(key=FLContextKey.WORKSPACE_OBJECT, value=workspace, private=True, sticky=False)
    fl_ctx.put(key=FLContextKey.CURRENT_JOB_ID, value="job-1", private=False, sticky=False)
    return fl_ctx


def _make_context(executor=None, **overrides):
    kwargs = dict(
        executor=executor if executor is not None else MagicMock(),
        command="python -u custom/train.py",
        launch_once=True,
        launch_timeout=5.0,
        # bounded by default so a broken round trip FAILS the test instead of hanging it
        result_wait_timeout=10.0,
    )
    kwargs.update(overrides)
    return ClientAPIBackendContext(**kwargs)


def _initialized_backend(env, executor=None, **overrides):
    backend = ExternalProcessBackend()
    engine = _make_engine(env.cell)
    fl_ctx = _make_fl_ctx(engine, env.app_dir)
    backend.initialize(_make_context(executor=executor, **overrides), fl_ctx)
    return backend, fl_ctx


def _install_auto_result(env, lazy_result=False):
    """Makes the fake trainer reply TASK_ACCEPTED and immediately RESULT_READY.

    Runs synchronously on the execute() thread (send_request is synchronous), like the
    in_process test's fake trainer. Records instead of asserting: a raise in here would be
    converted to an EXECUTION_EXCEPTION reply by the backend and mask the real failure.
    """
    seen = SimpleNamespace(task_payloads=[], result_replies=[])

    def handler(topic, target, request):
        seen.task_payloads.append((topic, target, request.payload))
        if topic != Topic.TASK_READY:
            return make_cell_reply(CellReturnCode.INVALID_REQUEST)
        task_payload = request.payload
        result = _result_shareable()
        if lazy_result:
            result["lazy"] = LazyDownloadRef(target, "result-ref", "T0")
        result_payload = {
            MsgKey.SESSION_ID: task_payload[MsgKey.SESSION_ID],
            MsgKey.TASK_ID: task_payload[MsgKey.TASK_ID],
            MsgKey.RESULT_ID: uuid.uuid4().hex,
            MsgKey.RESULT: result,
        }
        seen.result_replies.append(env.cell.deliver(Topic.RESULT_READY, target, result_payload))
        return _task_accepted_reply()

    env.cell.on_request = handler
    env.cell.on_shutdown = lambda *_args: make_cell_reply(
        CellReturnCode.OK, body={MsgKey.RESULT_SOURCE_LIVE: lazy_result}
    )
    return seen


class TestFactory:
    def test_executor_factory_returns_external_process_backend(self):
        executor = ClientAPIExecutor(execution_mode="external_process", command="python custom/train.py")
        backend = executor._create_external_process_backend()
        assert isinstance(backend, ExternalProcessBackend)

    def test_initialize_rejects_missing_command(self, env):
        backend = ExternalProcessBackend()
        fl_ctx = _make_fl_ctx(_make_engine(env.cell), env.app_dir)
        with pytest.raises(ValueError, match="command"):
            backend.initialize(_make_context(command=None), fl_ctx)

    def test_initialize_requires_cell(self, env):
        backend = ExternalProcessBackend()
        engine = _make_engine(env.cell)
        engine.get_cell.return_value = None
        fl_ctx = _make_fl_ctx(engine, env.app_dir)
        with pytest.raises(RuntimeError, match="Cell"):
            backend.initialize(_make_context(), fl_ctx)


class TestInitializeAndFinalize:
    def test_initialize_launches_trainer_and_establishes_session(self, env):
        executor = MagicMock()
        backend, _ = _initialized_backend(env, executor=executor)
        try:
            session = backend._session
            assert session is not None and session.ready.is_set()
            assert session.session_id
            assert session.trainer_fqcn == f"{CJ_FQCN}.client_api_trainer_1"
            assert env.cell.internal_listener_made
            assert (CellChannel.SERVER_COMMAND, ServerCommandNames.GET_TASK) in env.cell.decode_pass_through_topics
            assert CellChannel.SERVER_COMMAND not in env.cell.decode_pass_through_channels
            for topic in PROTOCOL_TOPICS:
                assert topic in env.cell.cbs, f"backend must register a handler for {topic}"
            # LOG analytics go out federation-scoped (MetricRelay ex-process parity)
            executor.set_analytics_fire_fed_event.assert_called_once_with(True)
            # process launched in its own group with the bootstrap path in its env
            process = env.harness.processes[0]
            assert process.kwargs["start_new_session"] is (os.name == "posix")
            assert process.kwargs["cwd"] == env.app_dir
            assert BOOTSTRAP_FILE_ENV_VAR in process.kwargs["env"]
        finally:
            backend.finalize(FLContext())
        assert not backend._session
        assert (CellChannel.SERVER_COMMAND, ServerCommandNames.GET_TASK) not in env.cell.decode_pass_through_topics
        # launch-token file is removed after teardown
        assert not os.path.exists(os.path.join(env.app_dir, bootstrap_file_name(1)))

    def test_bootstrap_config_contents_and_permission(self, env):
        backend, _ = _initialized_backend(
            env,
            train_task_name="my_train",
            evaluate_task_name="my_eval",
            submit_model_task_name="my_submit",
            memory_gc_rounds=3,
            cuda_empty_cache=True,
        )
        try:
            path = os.path.join(env.app_dir, bootstrap_file_name(1))
            assert os.stat(path).st_mode & 0o777 == 0o600, "launch token must be owner-readable only"
            config = read_bootstrap_config(path)
            assert config[BootstrapKey.SCHEMA_VERSION] == BOOTSTRAP_SCHEMA_VERSION
            assert config[BootstrapKey.EXECUTION_MODE] == EXTERNAL_PROCESS_EXECUTION_MODE
            assert config[BootstrapKey.CONNECT_URL] == env.cell.listener_url
            assert config[BootstrapKey.CJ_FQCN] == CJ_FQCN
            assert config[BootstrapKey.TRAINER_FQCN] == f"{CJ_FQCN}.client_api_trainer_1"
            assert config[BootstrapKey.LAUNCH_TOKEN]
            assert config[BootstrapKey.PROTOCOL_VERSION] == PROTOCOL_VERSION
            assert config[BootstrapKey.JOB_ID] == "job-1"
            assert config[BootstrapKey.SITE_NAME] == "site-1"
            assert config[BootstrapKey.MEMORY_GC_ROUNDS] == 3
            assert config[BootstrapKey.CUDA_EMPTY_CACHE] is True
            exchange = config[BootstrapKey.TASK_EXCHANGE]
            assert exchange[ConfigKey.EXCHANGE_FORMAT] == ExchangeFormat.RAW
            assert exchange[ConfigKey.SERVER_EXPECTED_FORMAT] == ExchangeFormat.NUMPY
            assert exchange[ConfigKey.TRANSFER_TYPE] == TransferType.FULL
            assert exchange[ConfigKey.TRAIN_TASK_NAME] == "my_train"
            assert exchange[ConfigKey.EVAL_TASK_NAME] == "my_eval"
            assert exchange[ConfigKey.SUBMIT_MODEL_TASK_NAME] == "my_submit"
        finally:
            backend.finalize(FLContext())

    def test_initialize_unwinds_when_hello_never_arrives(self, env):
        env.harness.auto_hello = False
        backend = ExternalProcessBackend()
        fl_ctx = _make_fl_ctx(_make_engine(env.cell), env.app_dir)

        start = time.monotonic()
        with pytest.raises(RuntimeError, match="HELLO handshake within launch_timeout"):
            backend.initialize(_make_context(launch_timeout=0.3), fl_ctx)
        elapsed = time.monotonic() - start

        assert elapsed < 5.0
        # contract: self-unwinding - the launched process tree must not leak
        process = env.harness.processes[0]
        assert process.returncode is not None, "unwind must terminate the launched trainer"
        assert backend._session is None

    def test_initialize_fails_fast_when_trainer_exits_before_hello(self, env):
        env.harness.auto_hello = False
        env.harness.exit_at_launch_rc = 7
        backend = ExternalProcessBackend()
        fl_ctx = _make_fl_ctx(_make_engine(env.cell), env.app_dir)

        start = time.monotonic()
        # a large launch_timeout must NOT be waited out: process death bounds the wait
        with pytest.raises(RuntimeError, match=r"exited \(rc=7\)"):
            backend.initialize(_make_context(launch_timeout=30.0), fl_ctx)
        assert time.monotonic() - start < 5.0

    def test_initialize_fails_fast_when_hello_is_rejected(self, env):
        def bad_token(payload):
            payload[MsgKey.PROOF] = "wrong-token"

        env.harness.hello_mutator = bad_token
        backend = ExternalProcessBackend()
        fl_ctx = _make_fl_ctx(_make_engine(env.cell), env.app_dir)

        with pytest.raises(RuntimeError, match="rejected.*launch token mismatch"):
            backend.initialize(_make_context(launch_timeout=30.0), fl_ctx)
        assert env.harness.processes[0].returncode is not None

    def test_finalize_idempotent_and_stops_process_tree(self, env):
        backend, _ = _initialized_backend(env)
        process = env.harness.processes[0]
        session = backend._session

        backend.finalize(FLContext())
        backend.finalize(FLContext())  # contract: idempotent, must not raise

        shutdowns = [request for request in env.cell.sent if request[0] == Topic.SHUTDOWN]
        assert len(shutdowns) == 1, "idempotency means no repeated side effects"
        assert process.returncode is not None
        # launch-scoped token invalidation
        assert session.token == "" and session.session_id is None
        assert backend._session is None

    def test_finalize_prefers_natural_exit_over_termination(self, env):
        backend, _ = _initialized_backend(env)
        process = env.harness.processes[0]

        # a well-behaved trainer exits on SHUTDOWN before the natural-exit bound
        def exit_on_shutdown(topic, target, message):
            if topic == Topic.SHUTDOWN:
                process.exit(0)
                return make_cell_reply(CellReturnCode.OK)
            return _task_accepted_reply()

        env.cell.on_shutdown = exit_on_shutdown
        backend.finalize(FLContext())

        assert process.returncode == 0
        assert env.harness.signals_sent() == [], "no signals for a trainer that exited naturally"

    def test_finalize_does_not_kill_an_accepted_lazy_result_source(self, env):
        backend, _ = _initialized_backend(env, shutdown_timeout=0.2)
        process = env.harness.processes[0]
        session = backend._session
        session.result_source_live.set()
        env.cell.on_shutdown = lambda *_args: make_cell_reply(CellReturnCode.OK, body={MsgKey.RESULT_SOURCE_LIVE: True})

        finalize_done = threading.Event()

        def finalize():
            backend.finalize(FLContext())
            finalize_done.set()

        finalize_thread = threading.Thread(target=finalize)
        finalize_thread.start()

        deadline = time.monotonic() + 1.0
        while session.reaper_thread is None and time.monotonic() < deadline:
            time.sleep(0.005)
        assert session.reaper_thread is not None
        assert not finalize_done.wait(0.03), "END_RUN must preserve the CJ while the accepted result source is live"

        assert process.returncode is None
        assert len([request for request in env.cell.sent if request[0] == Topic.SHUTDOWN]) == 1
        assert env.harness.signals_sent() == []

        # The trainer exits itself only after its real downstream transfer waiter settles;
        # the reaper then performs ordinary launch-artifact cleanup and releases END_RUN.
        process.exit(0)
        finalize_thread.join(timeout=1.0)
        assert finalize_done.is_set()
        assert backend._session is None
        assert session.token == ""

    def test_finalize_stops_source_when_shutdown_ack_says_send_already_settled(self, env):
        backend, _ = _initialized_backend(env, shutdown_timeout=0.2)
        process = env.harness.processes[0]
        session = backend._session
        session.result_source_live.set()

        def settled_shutdown(topic, target, message):
            if topic == Topic.SHUTDOWN:
                return make_cell_reply(CellReturnCode.OK, body={MsgKey.RESULT_SOURCE_LIVE: False})
            return _task_accepted_reply()

        env.cell.on_shutdown = settled_shutdown
        backend.finalize(FLContext())

        assert not session.result_source_live.is_set()
        assert process.returncode is not None
        assert backend._session is None
        assert session.token == ""

    def test_settled_ack_still_allows_user_thread_natural_exit_grace(self, env):
        backend, _ = _initialized_backend(env, shutdown_timeout=0.0)
        process = env.harness.processes[0]
        session = backend._session
        session.result_source_live.set()
        wait_entered = threading.Event()
        release_exit = threading.Event()

        def natural_wait(timeout=None):
            wait_entered.set()
            assert timeout == pytest.approx(ebp._DEFAULT_SHUTDOWN_TIMEOUT)
            assert release_exit.wait(2.0)
            process.exit(0)
            return 0

        process.wait = natural_wait

        def settled_shutdown(topic, target, message):
            if topic == Topic.SHUTDOWN:
                return make_cell_reply(CellReturnCode.OK, body={MsgKey.RESULT_SOURCE_LIVE: False})
            return _task_accepted_reply()

        env.cell.on_shutdown = settled_shutdown
        finalize_thread = threading.Thread(target=backend.finalize, args=(FLContext(),))
        finalize_thread.start()

        assert wait_entered.wait(1.0)
        assert env.harness.signals_sent() == []
        assert finalize_thread.is_alive()

        release_exit.set()
        finalize_thread.join(timeout=2.0)
        assert not finalize_thread.is_alive()
        assert env.harness.signals_sent() == []
        assert backend._session is None

    @pytest.mark.skipif(os.name != "posix", reason="process-group semantics are POSIX")
    def test_settled_ack_graces_workers_after_launcher_exit(self, env):
        backend, _ = _initialized_backend(env, shutdown_timeout=0.2)
        process = env.harness.processes[0]
        session = backend._session
        session.result_source_live.set()
        process.exit(0)
        process.extra_group_members = True
        env.cell.on_shutdown = lambda *_args: make_cell_reply(
            CellReturnCode.OK, body={MsgKey.RESULT_SOURCE_LIVE: False}
        )

        finalize_thread = threading.Thread(target=backend.finalize, args=(FLContext(),))
        finalize_thread.start()

        time.sleep(0.03)
        assert finalize_thread.is_alive()
        assert process.extra_group_members
        assert env.harness.signals_sent() == []

        # The launcher is already gone, but its worker gets the full natural-exit grace
        # and finishes by itself while returning from send().
        process.extra_group_members = False
        finalize_thread.join(timeout=1.0)
        assert not finalize_thread.is_alive()
        assert env.harness.signals_sent() == []
        assert backend._session is None

    def test_shutdown_timeout_does_not_cut_off_accepted_result_source(self, env):
        backend, _ = _initialized_backend(env, shutdown_timeout=0.03)
        process = env.harness.processes[0]
        session = backend._session
        session.result_source_live.set()
        env.cell.on_shutdown = lambda *_args: make_cell_reply(CellReturnCode.OK, body={MsgKey.RESULT_SOURCE_LIVE: True})
        finalize_done = threading.Event()

        def finalize():
            backend.finalize(FLContext())
            finalize_done.set()

        finalize_thread = threading.Thread(target=finalize)
        finalize_thread.start()

        assert not finalize_done.wait(0.08), "ordinary shutdown_timeout must not cut off a live result source"
        assert process.returncode is None
        assert session.reaper_thread is not None and session.reaper_thread.is_alive()
        assert env.harness.signals_sent() == []

        process.exit(0)
        finalize_thread.join(timeout=1.0)
        assert finalize_done.is_set()
        assert backend._session is None

    def test_zero_shutdown_live_result_uses_backend_grace_and_releases_on_truth(self, env, monkeypatch):
        monkeypatch.setattr(ebp, "_DEFAULT_SHUTDOWN_TIMEOUT", 0.1)
        monkeypatch.setattr(ebp, "_NATURAL_EXIT_REAP_INTERVAL", 0.005)
        backend, _ = _initialized_backend(env, shutdown_timeout=0.0)
        process = env.harness.processes[0]
        session = backend._session
        session.result_source_live.set()
        env.cell.on_shutdown = lambda *_args: make_cell_reply(CellReturnCode.OK, body={MsgKey.RESULT_SOURCE_LIVE: True})
        finalize_done = threading.Event()

        def finalize():
            backend.finalize(FLContext())
            finalize_done.set()

        finalize_thread = threading.Thread(target=finalize)
        finalize_thread.start()
        assert not finalize_done.wait(0.02), "ScriptRunner's zero default must not erase live-result grace"

        process.exit(0)
        finalize_thread.join(timeout=1.0)
        assert finalize_done.is_set()
        assert env.harness.signals_sent() == []
        assert backend._session is None

    def test_zero_heartbeat_and_shutdown_do_not_make_disconnect_immediately_terminal(self, env, monkeypatch):
        monkeypatch.setattr(ebp, "_DEFAULT_SHUTDOWN_TIMEOUT", 0.06)
        monkeypatch.setattr(ebp, "_NATURAL_EXIT_REAP_INTERVAL", 0.005)
        backend, _ = _initialized_backend(env, heartbeat_timeout=0.0, shutdown_timeout=0.0)
        process = env.harness.processes[0]
        session = backend._session
        session.result_source_live.set()
        env.cell.on_shutdown = lambda *_args: make_cell_reply(CellReturnCode.OK, body={MsgKey.RESULT_SOURCE_LIVE: True})
        env.cell.disconnected.add(session.trainer_fqcn)

        finalize_thread = threading.Thread(target=backend.finalize, args=(FLContext(),))
        finalize_thread.start()
        time.sleep(0.02)

        # Reconnecting inside the nonzero source grace resets disconnect evidence. END_RUN
        # remains blocked because the source is still live, regardless of shutdown_timeout.
        env.cell.disconnected.discard(session.trainer_fqcn)
        time.sleep(0.08)
        assert finalize_thread.is_alive()
        assert process.returncode is None
        assert session.reaper_thread is not None and session.reaper_thread.is_alive()
        assert env.harness.signals_sent() == []

        process.exit(0)
        finalize_thread.join(timeout=1.0)
        assert not finalize_thread.is_alive()
        assert backend._session is None

    def test_lazy_result_reaper_retries_shutdown_then_requires_sustained_disconnect(self, env, monkeypatch, caplog):
        monkeypatch.setattr(ebp, "_NATURAL_EXIT_REAP_INTERVAL", 0.005)
        monkeypatch.setattr(ebp, "_SHUTDOWN_RETRY_INTERVAL", 0.01)
        backend, _ = _initialized_backend(env, heartbeat_timeout=0.03, shutdown_timeout=0.01)
        process = env.harness.processes[0]
        session = backend._session
        session.result_source_live.set()
        attempts = []

        def flaky_shutdown(topic, target, message):
            if topic != Topic.SHUTDOWN:
                return _task_accepted_reply()
            attempts.append((topic, target))
            if len(attempts) == 1:
                return None
            # The retry reaches the trainer. It closes Cell after its terminal result
            # waiter; the reaper still requires a sustained disconnect before stopping.
            env.cell.disconnected.add(session.trainer_fqcn)
            return make_cell_reply(CellReturnCode.OK)

        env.cell.on_shutdown = flaky_shutdown
        finalize_thread = threading.Thread(target=backend.finalize, args=(FLContext(),))
        finalize_thread.start()

        deadline = time.monotonic() + 0.5
        while len(attempts) < 2 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert len(attempts) >= 2

        # A first disconnected sample is not terminal.
        time.sleep(0.01)
        assert process.returncode is None
        env.cell.disconnected.discard(session.trainer_fqcn)
        time.sleep(0.04)
        assert process.returncode is None, "reconnect must reset the disconnect grace"
        env.cell.disconnected.add(session.trainer_fqcn)
        deadline = time.monotonic() + 1.0
        while process.returncode is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert process.returncode is not None
        finalize_thread.join(timeout=1.0)
        assert not finalize_thread.is_alive()
        assert "trainer SHUTDOWN was not acknowledged (rc=None)" in caplog.text

    def test_finalize_escalates_sigterm_to_sigkill(self, env, monkeypatch):
        backend, _ = _initialized_backend(env, stop_grace_period=0.01)
        # a wedged tree: SIGTERM/SIGKILL are ignored (fake killpg only records; sig-0
        # probes report the group alive), so the escalation must run its full course
        monkeypatch.setattr(env.harness, "killpg", lambda pgid, sig: env.harness.killpg_calls.append((pgid, sig)))
        monkeypatch.setattr(ebp.os, "killpg", env.harness.killpg, raising=False)
        monkeypatch.setattr(ebp, "_LOG_THREAD_JOIN_TIMEOUT", 0.2)  # bound the post-SIGKILL group wait

        backend.finalize(FLContext())

        if os.name == "posix":
            assert env.harness.signals_sent() == [
                (env.harness.processes[0].pid, signal.SIGTERM),
                (env.harness.processes[0].pid, signal.SIGKILL),
            ]

    @pytest.mark.skipif(os.name != "posix", reason="process-group semantics are POSIX")
    def test_workers_terminated_when_launcher_already_exited(self, env):
        """torchrun shape: the launcher (group leader) exits while workers keep the group
        alive — teardown must still signal the GROUP, not skip on the dead leader."""
        backend, _ = _initialized_backend(env, shutdown_timeout=0.0)
        process = env.harness.processes[0]
        process.exit(0)
        process.extra_group_members = True

        backend.finalize(FLContext())

        assert (process.pid, signal.SIGTERM) in env.harness.signals_sent()
        assert not process.extra_group_members, "surviving workers were signaled"

    def test_command_split_is_platform_appropriate(self, monkeypatch):
        """Each platform's tokenizer preserves its paths and resolves secrets as one arg."""
        monkeypatch.setattr(ebp.os, "name", "posix")
        assert ExternalProcessBackend._split_command("python -u custom/train.py") == [
            "python",
            "-u",
            "custom/train.py",
        ]
        monkeypatch.setenv("EXTERNAL_BACKEND_TEST_SECRET", "resolved value with spaces")
        assert ExternalProcessBackend._split_command(
            "python custom/train.py --token ${secret:EXTERNAL_BACKEND_TEST_SECRET}"
        ) == ["python", "custom/train.py", "--token", "resolved value with spaces"]
        with pytest.raises(ValueError, match="nested interpreter command strings"):
            ExternalProcessBackend._split_command("bash -c 'echo ${secret:EXTERNAL_BACKEND_TEST_SECRET}'")

        monkeypatch.setattr(ebp.os, "name", "nt")
        win_command = "python C:\\work\\train.py"
        assert ExternalProcessBackend._split_command(win_command) == ["python", "C:\\work\\train.py"]
        assert ExternalProcessBackend._split_command(
            "python C:\\work\\train.py --token ${secret:EXTERNAL_BACKEND_TEST_SECRET}"
        ) == ["python", "C:\\work\\train.py", "--token", "resolved value with spaces"]

    def test_signal_fallback_uses_terminate_then_kill(self, env, monkeypatch):
        """When group signaling is unavailable (non-POSIX, or killpg failure), the stop
        escalation falls back to Popen.terminate()/kill() on the process itself."""
        backend, _ = _initialized_backend(env, stop_grace_period=0.01)
        process = env.harness.processes[0]

        def no_killpg(pgid, sig):
            raise RuntimeError("killpg unavailable")

        monkeypatch.setattr(ebp.os, "killpg", no_killpg, raising=False)
        monkeypatch.setattr(ebp, "_LOG_THREAD_JOIN_TIMEOUT", 0.2)  # unprobeable group: bound the waits
        # a wedged trainer: terminate() is delivered but the process does not die
        process.terminate = lambda: process.term_calls.append("terminate")

        backend.finalize(FLContext())

        assert process.term_calls == ["terminate", "kill"]
        assert process.returncode is not None

    def test_finalize_does_not_raise_when_shutdown_send_fails(self, env, caplog):
        backend, _ = _initialized_backend(env)
        process = env.harness.processes[0]

        def boom(channel, topic, target, request, **kwargs):
            raise RuntimeError("cell send failed")

        env.cell.send_request = boom
        backend.finalize(FLContext())  # must not raise

        assert "cell send failed" in caplog.text
        assert process.returncode is not None, "a failing SHUTDOWN send must not skip process teardown"

    def test_shutdown_ack_time_is_charged_against_natural_exit_bound(self, env, monkeypatch):
        clock = [100.0]
        monkeypatch.setattr(ebp.time, "monotonic", lambda: clock[0])
        backend, _ = _initialized_backend(env, shutdown_timeout=2.0)
        process = env.harness.processes[0]

        def acknowledge_after_delay(topic, target, request):
            if topic == Topic.SHUTDOWN:
                clock[0] += 0.75
                return make_cell_reply(CellReturnCode.OK)
            return _task_accepted_reply()

        env.cell.on_shutdown = acknowledge_after_delay

        def exit_during_remaining_wait(timeout=None):
            process.wait_timeouts.append(timeout)
            process.exit(0)
            return 0

        process.wait = exit_during_remaining_wait
        backend.finalize(FLContext())

        shutdown = [request for request in env.cell.sent if request[0] == Topic.SHUTDOWN]
        assert len(shutdown) == 1
        assert shutdown[0][3] == pytest.approx(2.0)
        assert process.wait_timeouts == [pytest.approx(1.25)]
        assert 0.75 + process.wait_timeouts[0] == pytest.approx(2.0)

    def test_zero_natural_exit_bound_keeps_immediate_fire_and_forget(self, env):
        backend, _ = _initialized_backend(env, shutdown_timeout=0.0)
        backend.finalize(FLContext())

        assert [request for request in env.cell.sent if request[0] == Topic.SHUTDOWN] == []
        assert len([message for message in env.cell.fired if message[0] == Topic.SHUTDOWN]) == 1


class TestHello:
    def test_stale_process_hello_is_rejected_without_failing_session(self, env):
        backend, _ = _initialized_backend(env)
        try:
            session = backend._session
            stale_fqcn = f"{CJ_FQCN}.client_api_trainer_99"
            reply = env.cell.deliver(
                Topic.HELLO,
                stale_fqcn,
                {
                    MsgKey.TRAINER_FQCN: stale_fqcn,
                    MsgKey.PROOF: session.token,
                    MsgKey.PROTOCOL_VERSION: PROTOCOL_VERSION,
                    MsgKey.JOB_ID: "job-1",
                    MsgKey.SITE_NAME: "site-1",
                    MsgKey.RANK: 0,
                },
            )
            assert reply.payload[MsgKey.REPLY_TOPIC] == Topic.HELLO_REJECTED
            # the current, already-authenticated session is untouched
            assert session.ready.is_set() and session.reject_reason is None
        finally:
            backend.finalize(FLContext())

    def test_duplicate_hello_is_idempotent(self, env):
        backend, _ = _initialized_backend(env)
        try:
            session = backend._session
            first_session_id = session.session_id
            config = env.harness.bootstrap_configs[0]
            reply = env.cell.deliver(
                Topic.HELLO,
                session.trainer_fqcn,
                {
                    MsgKey.TRAINER_FQCN: config[BootstrapKey.TRAINER_FQCN],
                    MsgKey.PROOF: config[BootstrapKey.LAUNCH_TOKEN],
                    MsgKey.PROTOCOL_VERSION: PROTOCOL_VERSION,
                    MsgKey.JOB_ID: "job-1",
                    MsgKey.SITE_NAME: "site-1",
                    MsgKey.RANK: 0,
                },
            )
            assert reply.payload[MsgKey.REPLY_TOPIC] == Topic.HELLO_ACCEPTED
            assert reply.payload[MsgKey.SESSION_ID] == first_session_id, "duplicate HELLO must not mint a new session"
        finally:
            backend.finalize(FLContext())

    @pytest.mark.parametrize(
        "mutation,expect_reason",
        [
            ({MsgKey.PROTOCOL_VERSION: 99}, "protocol version"),
            ({MsgKey.JOB_ID: "job-2"}, "job id mismatch"),
            ({MsgKey.SITE_NAME: "site-2"}, "site name mismatch"),
            ({MsgKey.RANK: 1}, "rank"),
        ],
        ids=["bad_version", "bad_job_id", "bad_site_name", "nonzero_rank"],
    )
    def test_invalid_hello_fields_rejected(self, env, mutation, expect_reason):
        env.harness.hello_mutator = lambda payload: payload.update(mutation)
        backend = ExternalProcessBackend()
        fl_ctx = _make_fl_ctx(_make_engine(env.cell), env.app_dir)
        with pytest.raises(RuntimeError):
            backend.initialize(_make_context(launch_timeout=0.3), fl_ctx)
        reply = env.harness.hello_replies[0]
        assert reply.payload[MsgKey.REPLY_TOPIC] == Topic.HELLO_REJECTED
        assert expect_reason in reply.payload[MsgKey.REASON]

    def test_nonzero_rank_does_not_latch_launch_failure(self, env):
        """A non-zero rank's HELLO must not fail the launch wait: the real rank 0 may
        still connect (rank contract)."""
        ranks = iter([1, 0])
        env.harness.hello_mutator = lambda payload: payload.update({MsgKey.RANK: next(ranks)})
        original_popen = env.harness.popen

        def popen_with_two_hellos(args, **kwargs):
            process = original_popen(args, **kwargs)  # HELLO with rank 1 -> rejected
            config = env.harness.bootstrap_configs[-1]
            payload = {
                MsgKey.TRAINER_FQCN: config[BootstrapKey.TRAINER_FQCN],
                MsgKey.PROOF: config[BootstrapKey.LAUNCH_TOKEN],
                MsgKey.PROTOCOL_VERSION: PROTOCOL_VERSION,
                MsgKey.JOB_ID: "job-1",
                MsgKey.SITE_NAME: "site-1",
                MsgKey.RANK: next(ranks),
            }
            env.cell.deliver(Topic.HELLO, config[BootstrapKey.TRAINER_FQCN], payload)  # rank 0 -> accepted
            return process

        env.harness.popen = popen_with_two_hellos
        backend, _ = _initialized_backend(env, launch_timeout=5.0)
        try:
            assert backend._session.ready.is_set()
        finally:
            backend.finalize(FLContext())

    def test_hello_with_non_dict_payload_is_a_protocol_error(self, env):
        backend, _ = _initialized_backend(env)
        try:
            reply = env.cell.deliver(Topic.HELLO, "anywhere", "not-a-dict")
            assert reply.get_header(MessageHeaderKey.RETURN_CODE) == CellReturnCode.INVALID_REQUEST
        finally:
            backend.finalize(FLContext())

    def test_hello_after_finalize_is_rejected(self, env):
        backend, _ = _initialized_backend(env)
        config = env.harness.bootstrap_configs[0]
        backend.finalize(FLContext())

        reply = env.cell.deliver(
            Topic.HELLO,
            config[BootstrapKey.TRAINER_FQCN],
            {
                MsgKey.TRAINER_FQCN: config[BootstrapKey.TRAINER_FQCN],
                MsgKey.PROOF: config[BootstrapKey.LAUNCH_TOKEN],
                MsgKey.PROTOCOL_VERSION: PROTOCOL_VERSION,
                MsgKey.JOB_ID: "job-1",
                MsgKey.RANK: 0,
            },
        )
        assert reply.payload[MsgKey.REPLY_TOPIC] == Topic.HELLO_REJECTED
        # by the _closed gate specifically, not merely because the session is gone
        assert reply.payload[MsgKey.REASON] == "backend is closed"


class TestHeartbeatAndOperationalLiveness:
    def test_hello_advertises_heartbeat_policy(self, env):
        backend, _ = _initialized_backend(env, heartbeat_interval=0.02, heartbeat_timeout=0.2)
        try:
            body = env.harness.hello_replies[0].payload
            assert body[MsgKey.HEARTBEAT_INTERVAL] == 0.02
            assert body[MsgKey.HEARTBEAT_TIMEOUT] == 0.2
        finally:
            backend.finalize(FLContext())

    def test_heartbeat_is_bound_to_origin_and_session(self, env):
        backend, _ = _initialized_backend(env)
        try:
            session = backend._session
            with session._activity_lock:
                session._last_peer_activity = time.monotonic() - 5.0

            rejected = env.cell.deliver(
                Topic.HEARTBEAT,
                "foreign.cell",
                {MsgKey.SESSION_ID: session.session_id},
            )
            assert rejected.payload[MsgKey.REPLY_TOPIC] == Topic.ERROR
            assert session.peer_silent_for() >= 4.0

            accepted = env.cell.deliver(
                Topic.HEARTBEAT,
                session.trainer_fqcn,
                {MsgKey.SESSION_ID: session.session_id},
            )
            assert accepted.payload == {
                MsgKey.REPLY_TOPIC: Topic.HEARTBEAT,
                MsgKey.SESSION_ID: session.session_id,
            }
            assert session.peer_silent_for() < 0.5
        finally:
            backend.finalize(FLContext())

    def test_missing_heartbeat_bounds_unlimited_result_wait(self, env):
        backend, fl_ctx = _initialized_backend(
            env,
            heartbeat_interval=0.02,
            heartbeat_timeout=0.08,
            result_wait_timeout=None,
        )
        try:
            start = time.monotonic()
            result = backend.execute("train", Shareable(), fl_ctx, Signal())
            elapsed = time.monotonic() - start

            assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
            assert elapsed < 1.0
            assert "heartbeat timed out" in backend._abort_reason
            assert [f for f in env.cell.fired if f[0] == Topic.ABORT]
        finally:
            backend.finalize(FLContext())

    def test_live_task_payload_transaction_suppresses_heartbeat_expiry(self, env, monkeypatch):
        monkeypatch.setattr(ebp, "_RESULT_POLL_INTERVAL", 0.01)
        transfer_settled = threading.Event()
        waiter = SimpleNamespace(done=lambda: transfer_settled.is_set())
        monkeypatch.setattr(ebp.DownloadService, "get_transfer_waiter", lambda _tx_id: waiter)
        backend, fl_ctx = _initialized_backend(
            env,
            heartbeat_interval=0.02,
            heartbeat_timeout=0.05,
            result_wait_timeout=2.0,
        )
        try:

            def slow_materialization(topic, target, request):
                assert topic == Topic.TASK_READY
                tx_created = env.cell.sent_kwargs[-1]["fobs_ctx_props"][ebp.RESULT_UPLOAD_TX_CREATED_CB_CTX_KEY]
                tx_created(SimpleNamespace(tx_id="task-payload-tx"))
                time.sleep(0.1)  # longer than the heartbeat timeout
                transfer_settled.set()
                payload = request.payload
                env.cell.deliver(
                    Topic.RESULT_READY,
                    target,
                    {
                        MsgKey.SESSION_ID: payload[MsgKey.SESSION_ID],
                        MsgKey.TASK_ID: payload[MsgKey.TASK_ID],
                        MsgKey.RESULT_ID: "result-after-materialization",
                        MsgKey.RESULT: _result_shareable(),
                    },
                )
                return _task_accepted_reply()

            env.cell.on_request = slow_materialization
            env.cell.on_shutdown = lambda *_args: make_cell_reply(
                CellReturnCode.OK, body={MsgKey.RESULT_SOURCE_LIVE: False}
            )
            result = backend.execute("train", Shareable(), fl_ctx, Signal())
            assert result.get_return_code() == ReturnCode.OK
        finally:
            backend.finalize(FLContext())

    def test_inline_task_ready_pending_is_bounded_by_heartbeat(self, env, monkeypatch):
        monkeypatch.setattr(ebp, "_RESULT_POLL_INTERVAL", 0.01)
        backend, fl_ctx = _initialized_backend(
            env,
            heartbeat_interval=0.02,
            heartbeat_timeout=0.05,
            task_wait_timeout=None,
            result_wait_timeout=None,
        )
        try:

            def wedged_inline_request(topic, target, request):
                assert topic == Topic.TASK_READY
                cancel = env.cell.sent_kwargs[-1]["abort_signal"]
                while not cancel.triggered:
                    time.sleep(0.005)
                return _task_accepted_reply()

            env.cell.on_request = wedged_inline_request
            start = time.monotonic()
            result = backend.execute("train", Shareable(), fl_ctx, Signal())
            elapsed = time.monotonic() - start

            assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
            assert elapsed < 0.5
            assert "heartbeat timed out" in backend._abort_reason
            assert "TASK_READY was pending" in backend._abort_reason
        finally:
            backend.finalize(FLContext())

    @pytest.mark.skipif(os.name != "posix", reason="process-group semantics are POSIX")
    def test_surviving_worker_group_is_operationally_alive(self, env):
        backend, fl_ctx = _initialized_backend(env, shutdown_timeout=0.0)
        process = env.harness.processes[0]
        process.exit(0)
        process.extra_group_members = True
        try:
            _install_auto_result(env)
            result = backend.execute("train", Shareable(), fl_ctx, Signal())
            assert result.get_return_code() == ReturnCode.OK
            assert backend._abort is False
        finally:
            process.extra_group_members = False
            backend.finalize(FLContext())


class TestClosedGate:
    """The _closed gate must hold on its own: cell callbacks stay registered after
    finalize() (cellnet has no unregister), so a zombie trainer's messages must be
    refused even if backend session state were somehow still present. These tests
    resurrect the session object after finalize to isolate the gate from the
    session-teardown checks (the in_process analog is TestClosedApiOutgoingGate)."""

    @staticmethod
    def _resurrect_session(backend, session, token, session_id):
        session.token = token
        session.session_id = session_id
        backend._session = session

    def _finalized_backend_with_session(self, env, executor=None):
        backend, _ = _initialized_backend(env, executor=executor)
        session = backend._session
        token, session_id = session.token, session.session_id
        backend.finalize(FLContext())
        self._resurrect_session(backend, session, token, session_id)
        return backend, session

    def test_closed_gate_rejects_authenticated_hello(self, env):
        backend, session = self._finalized_backend_with_session(env)
        reply = env.cell.deliver(
            Topic.HELLO,
            session.trainer_fqcn,
            {
                MsgKey.TRAINER_FQCN: session.trainer_fqcn,
                MsgKey.PROOF: session.token,
                MsgKey.PROTOCOL_VERSION: PROTOCOL_VERSION,
                MsgKey.JOB_ID: "job-1",
                MsgKey.RANK: 0,
            },
        )
        assert reply.payload[MsgKey.REPLY_TOPIC] == Topic.HELLO_REJECTED
        assert reply.payload[MsgKey.REASON] == "backend is closed"

    def test_closed_gate_rejects_valid_result_ready(self, env):
        backend, session = self._finalized_backend_with_session(env)
        task = ebp._TaskContext(task_id="task-1")
        backend._current_task = task

        reply = env.cell.deliver(
            Topic.RESULT_READY,
            session.trainer_fqcn,
            {
                MsgKey.SESSION_ID: session.session_id,
                MsgKey.TASK_ID: "task-1",
                MsgKey.RESULT_ID: "res-1",
                MsgKey.RESULT: _result_shareable(),
            },
        )
        assert reply.payload[MsgKey.REPLY_TOPIC] == Topic.RESULT_REJECTED
        assert reply.payload[MsgKey.REASON] == "backend is closed"
        assert not task.result_ready.is_set()

    def test_closed_gate_drops_log(self, env):
        executor = MagicMock()
        backend, session = self._finalized_backend_with_session(env, executor=executor)
        executor.fire_log_analytics.reset_mock()

        env.cell.deliver(
            Topic.LOG,
            session.trainer_fqcn,
            {
                MsgKey.SESSION_ID: session.session_id,
                "key": "accuracy",
                "value": 0.9,
                "data_type": AnalyticsDataType.SCALAR,
            },
        )
        executor.fire_log_analytics.assert_not_called()


class TestExecute:
    def test_execute_round_trip(self, env):
        backend, fl_ctx = _initialized_backend(env)
        try:
            seen = _install_auto_result(env)

            task = Shareable()
            task.set_header(AppConstants.CURRENT_ROUND, 3)
            result = backend.execute("train", task, fl_ctx, Signal())

            assert isinstance(result, Shareable)
            assert result.get_return_code() == ReturnCode.OK
            # the round travels back on the result for workflow bookkeeping
            assert result.get_header(AppConstants.CURRENT_ROUND) == 3

            # TASK_READY carried the full correlation envelope
            topic, target, payload = seen.task_payloads[0]
            session = backend._session
            assert topic == Topic.TASK_READY and target == session.trainer_fqcn
            # the streaming request path requires a numeric timeout: the default
            # task_wait_timeout=None must be normalized, never passed through
            assert env.cell.sent[0][3] == ebp._TASK_READY_NO_PROGRESS_TIMEOUT
            assert payload[MsgKey.SESSION_ID] == session.session_id
            assert payload[MsgKey.TASK_NAME] == "train"
            assert payload[MsgKey.TASK_ID]
            assert payload[MsgKey.MODEL] is task
            # Call-scoped FOBS metadata tells Cell/ViaDownloader who must materialize
            # the direct task Shareable; there is no second payload-attempt envelope.
            assert env.cell.sent_kwargs[0]["receiver_ids"] == (session.trainer_fqcn,)
            assert "fobs_ctx_props" in env.cell.sent_kwargs[0]
            assert task.get_header(FLMetaKey.JOB_ID) == "job-1"
            assert task.get_header(FLMetaKey.SITE_NAME) == "site-1"
            # RESULT_READY carried a direct Shareable and was control-acked.
            assert seen.result_replies[0].payload[MsgKey.REPLY_TOPIC] == Topic.RESULT_ACCEPTED
        finally:
            backend.finalize(FLContext())

    def test_next_task_does_not_clear_a_prior_send_barrier(self, env):
        backend, fl_ctx = _initialized_backend(env)
        session = backend._session
        session.result_source_live.set()  # prior round accepted; trainer send still settling
        barrier_seen = []

        def handler(topic, target, request):
            barrier_seen.append(session.result_source_live.is_set())
            payload = request.payload
            env.cell.deliver(
                Topic.RESULT_READY,
                target,
                {
                    MsgKey.SESSION_ID: payload[MsgKey.SESSION_ID],
                    MsgKey.TASK_ID: payload[MsgKey.TASK_ID],
                    MsgKey.RESULT_ID: "result-after-prior-send",
                    MsgKey.RESULT: _result_shareable(),
                },
            )
            return _task_accepted_reply()

        env.cell.on_request = handler
        env.cell.on_shutdown = lambda *_args: make_cell_reply(
            CellReturnCode.OK, body={MsgKey.RESULT_SOURCE_LIVE: False}
        )
        try:
            result = backend.execute("train", Shareable(), fl_ctx, Signal())

            assert result.get_return_code() == ReturnCode.OK
            assert barrier_seen == [True]
        finally:
            backend.finalize(FLContext())

    def test_result_arriving_asynchronously_wakes_the_wait(self, env):
        """The production shape: a cell dispatcher thread sets the result event while
        execute() is parked in the wait loop (every other test delivers synchronously).

        The bound is deliberately BELOW _RESULT_POLL_INTERVAL: a purely poll-based wait
        (e.g. replacing task.result_ready.wait with time.sleep) would only observe the
        event at the next 0.5s poll and blow this bound — so the assertion actually
        verifies the event-driven wake, not merely 'a result eventually arrives'."""
        assert ebp._RESULT_POLL_INTERVAL >= 0.4, "the sub-poll-interval bound below depends on this"
        backend, fl_ctx = _initialized_backend(env, result_wait_timeout=30.0)
        try:

            def handler(topic, target, request):
                payload = request.payload
                result_payload = {
                    MsgKey.SESSION_ID: payload[MsgKey.SESSION_ID],
                    MsgKey.TASK_ID: payload[MsgKey.TASK_ID],
                    MsgKey.RESULT_ID: "res-async",
                    MsgKey.RESULT: _result_shareable(),
                }
                threading.Timer(0.05, env.cell.deliver, args=(Topic.RESULT_READY, target, result_payload)).start()
                return _task_accepted_reply()

            env.cell.on_request = handler
            env.cell.on_shutdown = lambda *_args: make_cell_reply(
                CellReturnCode.OK, body={MsgKey.RESULT_SOURCE_LIVE: False}
            )
            start = time.monotonic()
            result = backend.execute("train", Shareable(), fl_ctx, Signal())
            elapsed = time.monotonic() - start

            assert result.get_return_code() == ReturnCode.OK
            assert elapsed < 0.4, "the result event must WAKE the wait (< poll interval), not be polled up to 0.5s"
        finally:
            backend.finalize(FLContext())

    def test_client_api_channel_takes_the_streaming_request_path(self):
        """Locks the routing assumption behind the TASK_READY timeout normalization and
        native cancel: cellnet treats every non-excluded channel as streaming-capable, so
        our channel goes through Cell._send_request (numeric no-progress timeout,
        abort_signal support) — NOT CoreCell.send_request."""
        from nvflare.fuel.f3.cellnet.cell import _is_stream_channel

        assert _is_stream_channel(CHANNEL) is True

    def test_configured_task_wait_timeout_passes_through(self, env):
        backend, fl_ctx = _initialized_backend(env, task_wait_timeout=7.5)
        try:
            _install_auto_result(env)
            assert backend.execute("train", Shareable(), fl_ctx, Signal()).get_return_code() == ReturnCode.OK
            assert env.cell.sent[0][3] == 7.5
        finally:
            backend.finalize(FLContext())

    def test_task_ready_send_does_not_hang_past_abort(self, env):
        """The plain-channel Cell request has no cancellation and a None task_wait_timeout
        falls back to the cell's hour-long max: a wedged trainer handler must not pin
        execute() past abort (the send runs on a helper thread; the wait polls)."""
        backend, fl_ctx = _initialized_backend(env, result_wait_timeout=30.0)
        release = threading.Event()
        try:
            abort_signal = Signal()

            def wedged_handler(topic, target, request):
                release.wait(10.0)  # the trainer's TASK_READY handler never replies
                return _task_accepted_reply()

            env.cell.on_request = wedged_handler
            threading.Timer(0.3, abort_signal.trigger, args=["stop"]).start()

            start = time.monotonic()
            result = backend.execute("train", Shareable(), fl_ctx, abort_signal)
            elapsed = time.monotonic() - start

            assert result.get_return_code() == ReturnCode.TASK_ABORTED
            assert elapsed < 5.0, "abort must end the TASK_READY wait, not the cell max_timeout"
            assert [f for f in env.cell.fired if f[0] == Topic.ABORT]
            # the request itself was cancelled through the streaming path's native
            # abort_signal — the cell waiter is released, not abandoned until timeout
            cancel = env.cell.sent_kwargs[0].get("abort_signal")
            assert cancel is not None and cancel.triggered
        finally:
            release.set()
            backend.finalize(FLContext())

    def test_task_ready_send_detects_process_death(self, env):
        backend, fl_ctx = _initialized_backend(env, result_wait_timeout=30.0)
        release = threading.Event()
        try:

            def wedged_handler(topic, target, request):
                release.wait(10.0)
                return _task_accepted_reply()

            env.cell.on_request = wedged_handler
            threading.Timer(0.3, env.harness.processes[0].exit, args=[1]).start()

            start = time.monotonic()
            result = backend.execute("train", Shareable(), fl_ctx, Signal())
            elapsed = time.monotonic() - start

            assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
            assert elapsed < 5.0, "a dead trainer must end the TASK_READY wait"
            assert "TASK_READY was pending" in backend._abort_reason
        finally:
            release.set()
            backend.finalize(FLContext())

    def test_task_ready_send_detects_backend_close(self, env):
        """finalize() racing an in-flight TASK_READY send (the teardown window): the send
        wait must observe _closed and fail the task rather than pin execute() on the
        wedged handler until the streaming request's no-progress timeout."""
        backend, fl_ctx = _initialized_backend(env, result_wait_timeout=30.0)
        release = threading.Event()
        try:

            def wedged_handler(topic, target, request):
                release.wait(10.0)
                return _task_accepted_reply()

            env.cell.on_request = wedged_handler
            # END_RUN teardown flips _closed on another thread while the send is parked
            threading.Timer(0.3, lambda: setattr(backend, "_closed", True)).start()

            start = time.monotonic()
            result = backend.execute("train", Shareable(), fl_ctx, Signal())
            elapsed = time.monotonic() - start

            assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
            assert elapsed < 5.0, "backend close must end the TASK_READY wait"
            # the parked request was cancelled through the streaming path's native signal
            cancel = env.cell.sent_kwargs[0].get("abort_signal")
            assert cancel is not None and cancel.triggered
        finally:
            release.set()
            backend._closed = False  # let finalize run its normal teardown
            backend.finalize(FLContext())

    def test_concurrent_execute_rejects_second_without_disturbing_first(self, env):
        """One active task per session: a second execute() racing the first must be
        rejected without overwriting _current_task, so the first task's RESULT_READY is
        still accepted and the first returns OK. (ClientRunner does not serialize executor
        calls, so the backend must.)"""
        backend, fl_ctx = _initialized_backend(env, result_wait_timeout=30.0)
        first_in_flight = threading.Event()
        let_first_finish = threading.Event()
        results = {}

        def handler(topic, target, request):
            payload = request.payload
            # park the FIRST task inside execute (current task installed, awaiting result)
            # so the second arrives concurrently; then release it with a real result
            first_in_flight.set()
            let_first_finish.wait(10.0)
            env.cell.deliver(
                Topic.RESULT_READY,
                target,
                {
                    MsgKey.SESSION_ID: payload[MsgKey.SESSION_ID],
                    MsgKey.TASK_ID: payload[MsgKey.TASK_ID],
                    MsgKey.RESULT_ID: "res-concurrent",
                    MsgKey.RESULT: _result_shareable(),
                },
            )
            return _task_accepted_reply()

        env.cell.on_request = handler
        try:
            t1 = threading.Thread(
                target=lambda: results.__setitem__("first", backend.execute("train", Shareable(), fl_ctx, Signal()))
            )
            t1.start()
            assert first_in_flight.wait(5.0), "first task did not reach in-flight state"
            active_task = backend._current_task

            # second task arrives while the first is active -> must be rejected, not admitted
            second = backend.execute("evaluate", Shareable(), fl_ctx, Signal())
            assert second.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
            assert backend._current_task is active_task, "the rejected task must not replace or clear the active one"

            let_first_finish.set()
            t1.join(10.0)
            assert results["first"].get_return_code() == ReturnCode.OK, "first task's result must still be accepted"
        finally:
            let_first_finish.set()
            backend.finalize(FLContext())

    def test_busy_rejection_honors_triggered_abort(self, env):
        """A concurrent second task whose abort_signal is already triggered must get
        TASK_ABORTED (contract), not the busy EXECUTION_EXCEPTION — and must not touch the
        active task's session."""
        backend, fl_ctx = _initialized_backend(env, result_wait_timeout=30.0)
        first_in_flight = threading.Event()
        let_first_finish = threading.Event()

        def handler(topic, target, request):
            payload = request.payload
            first_in_flight.set()
            let_first_finish.wait(10.0)
            env.cell.deliver(
                Topic.RESULT_READY,
                target,
                {
                    MsgKey.SESSION_ID: payload[MsgKey.SESSION_ID],
                    MsgKey.TASK_ID: payload[MsgKey.TASK_ID],
                    MsgKey.RESULT_ID: "res-busy",
                    MsgKey.RESULT: _result_shareable(),
                },
            )
            return _task_accepted_reply()

        env.cell.on_request = handler
        try:
            t1 = threading.Thread(target=lambda: backend.execute("train", Shareable(), fl_ctx, Signal()))
            t1.start()
            assert first_in_flight.wait(5.0)
            active_task = backend._current_task

            aborted = Signal()
            aborted.trigger("stop")
            second = backend.execute("evaluate", Shareable(), fl_ctx, aborted)
            assert second.get_return_code() == ReturnCode.TASK_ABORTED
            # the active task's session was untouched: no ABORT was sent for the rejected call
            assert backend._current_task is active_task

            let_first_finish.set()
            t1.join(10.0)
        finally:
            let_first_finish.set()
            backend.finalize(FLContext())

    def test_raising_per_task_stop_still_releases_the_gate(self, env, monkeypatch):
        """A raising per-task _stop_session must not leak the admission gate — else every
        later task would be busy-rejected forever."""
        backend, fl_ctx = _initialized_backend(env, launch_once=False)
        _install_auto_result(env)
        real_stop = backend._stop_session
        calls = {"n": 0}

        def raising_stop(session, natural_exit_wait):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("stop boom")
            return real_stop(session, natural_exit_wait)

        monkeypatch.setattr(backend, "_stop_session", raising_stop)
        try:
            first = backend.execute("train", Shareable(), fl_ctx, Signal())
            assert first.get_return_code() == ReturnCode.OK  # a raising stop must not mask the result
            # the gate was released despite the raising stop: a second task is admitted
            second = backend.execute("train", Shareable(), fl_ctx, Signal())
            assert second.get_return_code() == ReturnCode.OK
        finally:
            backend.finalize(FLContext())

    def test_finalize_barriers_no_process_started_after_close(self, env, monkeypatch):
        """END_RUN racing a per-task launch: finalize() sets _closed and barriers on the
        execute gate, and the pre-Popen _closed check bails the launch — so no trainer
        process is ever started after finalize began."""
        backend, fl_ctx = _initialized_backend(env, launch_once=False)
        in_write = threading.Event()
        let_write = threading.Event()
        real_write = ebp.write_bootstrap_config

        def hooked_write(path, config):
            in_write.set()
            let_write.wait(5.0)
            return real_write(path, config)

        monkeypatch.setattr(ebp, "write_bootstrap_config", hooked_write)

        box = {}
        t = threading.Thread(
            target=lambda: box.__setitem__("r", backend.execute("train", Shareable(), fl_ctx, Signal()))
        )
        t.start()
        assert in_write.wait(5.0), "execute did not reach the bootstrap write (session installed, pre-Popen)"

        f = threading.Thread(target=lambda: backend.finalize(FLContext()))
        f.start()
        # finalize sets _closed synchronously, then barriers on the gate the execute holds
        deadline = time.monotonic() + 5.0
        while not backend._closed and time.monotonic() < deadline:
            time.sleep(0.01)
        assert backend._closed

        let_write.set()  # let the launch proceed to the pre-Popen _closed check -> bail
        t.join(5.0)
        f.join(5.0)

        assert env.harness.processes == [], "no trainer process may be started after finalize set _closed"
        assert box["r"].get_return_code() == ReturnCode.EXECUTION_EXCEPTION

    def test_abort_signal_mid_wait_returns_task_aborted(self, env):
        backend, fl_ctx = _initialized_backend(env, result_wait_timeout=30.0)
        try:
            abort_signal = Signal()

            def accept_then_abort(topic, target, request):
                abort_signal.trigger("stop")
                return _task_accepted_reply()

            env.cell.on_request = accept_then_abort
            start = time.monotonic()
            result = backend.execute("train", Shareable(), fl_ctx, abort_signal)
            elapsed = time.monotonic() - start

            assert result.get_return_code() == ReturnCode.TASK_ABORTED
            assert elapsed < 10.0, "abort must not wait out the result bound"
            assert [f for f in env.cell.fired if f[0] == Topic.ABORT]
        finally:
            backend.finalize(FLContext())

    def test_execute_returns_task_aborted_on_triggered_signal(self, env):
        backend, fl_ctx = _initialized_backend(env)
        try:
            signal_ = Signal()
            signal_.trigger("stop")
            result = backend.execute("train", Shareable(), fl_ctx, signal_)

            assert result.get_return_code() == ReturnCode.TASK_ABORTED
            # the trainer was told the task is aborted
            assert [f for f in env.cell.fired if f[0] == Topic.ABORT]
        finally:
            backend.finalize(FLContext())

    def test_execute_bounded_by_result_wait_timeout(self, env):
        backend, fl_ctx = _initialized_backend(env, result_wait_timeout=0.05)
        try:
            # trainer accepts the task but never sends a result
            start = time.monotonic()
            result = backend.execute("train", Shareable(), fl_ctx, Signal())
            elapsed = time.monotonic() - start

            assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
            assert elapsed < 0.5, "timeout must not be rounded up to the polling interval"
            # the trainer was told to stop the task
            assert [f for f in env.cell.fired if f[0] == Topic.ABORT]
        finally:
            backend.finalize(FLContext())

    def test_execute_after_timeout_fails_fast_with_accurate_rc(self, env):
        """One result-wait timeout ends the launch_once session for good (the wire ABORT
        ends the trainer's Client API loop); later tasks must fail FAST with
        EXECUTION_EXCEPTION - mirrors the in_process latched-abort behavior."""
        backend, fl_ctx = _initialized_backend(env, result_wait_timeout=0.05)
        try:
            first = backend.execute("train", Shareable(), fl_ctx, Signal())
            assert first.get_return_code() == ReturnCode.EXECUTION_EXCEPTION

            start = time.monotonic()
            second = backend.execute("train", Shareable(), fl_ctx, Signal())
            elapsed = time.monotonic() - start
            assert second.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
            assert elapsed < 1.0, "post-abort tasks must fail at entry, not wait the poll loop"
        finally:
            backend.finalize(FLContext())

    def test_execute_fails_fast_when_trainer_process_dies(self, env):
        backend, fl_ctx = _initialized_backend(env, result_wait_timeout=30.0)
        try:

            def accept_then_die(topic, target, request):
                env.harness.processes[0].exit(1)
                return _task_accepted_reply()

            env.cell.on_request = accept_then_die
            start = time.monotonic()
            result = backend.execute("train", Shareable(), fl_ctx, Signal())
            elapsed = time.monotonic() - start

            assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
            assert elapsed < 5.0, "a dead trainer must be detected, not waited out"
            assert "exited" in backend._abort_reason
        finally:
            backend.finalize(FLContext())

    def test_direct_task_delivery_failure_fails_task(self, env):
        backend, fl_ctx = _initialized_backend(env, result_wait_timeout=30.0)
        try:
            env.cell.on_request = lambda topic, target, request: (_ for _ in ()).throw(
                RuntimeError("direct Cell payload delivery failed")
            )
            start = time.monotonic()
            result = backend.execute("train", Shareable(), fl_ctx, Signal())
            elapsed = time.monotonic() - start

            assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
            assert elapsed < 10.0, "a failed direct Cell send must not wait out the result bound"
        finally:
            backend.finalize(FLContext())

    @pytest.mark.parametrize(
        "reply_factory",
        [
            lambda: None,
            lambda: make_cell_reply(CellReturnCode.TIMEOUT),
            lambda: make_cell_reply(CellReturnCode.OK, body={MsgKey.REPLY_TOPIC: Topic.TASK_FAILED}),
            lambda: make_cell_reply(CellReturnCode.OK, body="bad-body"),
        ],
        ids=["no_reply", "cell_timeout", "not_accepted", "bad_body"],
    )
    def test_task_ready_not_accepted_returns_execution_exception(self, env, reply_factory):
        backend, fl_ctx = _initialized_backend(env)
        try:
            env.cell.on_request = lambda topic, target, request: reply_factory()
            result = backend.execute("train", Shareable(), fl_ctx, Signal())
            assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
        finally:
            backend.finalize(FLContext())

    def test_multi_round_sequential_execute(self, env):
        """The same backend and session serve consecutive rounds (launch_once)."""
        backend, fl_ctx = _initialized_backend(env)
        try:
            _install_auto_result(env)
            for round_num in (0, 1, 2):
                task = Shareable()
                task.set_header(AppConstants.CURRENT_ROUND, round_num)
                result = backend.execute("train", task, fl_ctx, Signal())
                assert result.get_return_code() == ReturnCode.OK
                assert result.get_header(AppConstants.CURRENT_ROUND) == round_num
            assert len(env.harness.processes) == 1, "launch_once means one process for the whole job"
            task_ids = [payload[MsgKey.TASK_ID] for _, _, payload, _ in env.cell.sent]
            assert len(set(task_ids)) == 3
            assert all(isinstance(payload[MsgKey.MODEL], Shareable) for _, _, payload, _ in env.cell.sent)
        finally:
            backend.finalize(FLContext())

    def test_unsafe_job_error_propagates(self, env, monkeypatch):
        backend, fl_ctx = _initialized_backend(env)
        try:
            monkeypatch.setattr(backend, "_send_task_ready", Mock(side_effect=UnsafeJobError("unsafe")))
            with pytest.raises(UnsafeJobError, match="unsafe"):
                backend.execute("train", Shareable(), fl_ctx, Signal())
        finally:
            backend.finalize(FLContext())

    def test_execute_exception_returns_execution_exception(self, env, monkeypatch):
        backend, fl_ctx = _initialized_backend(env)
        try:
            monkeypatch.setattr(backend, "_send_task_ready", Mock(side_effect=RuntimeError("boom")))
            result = backend.execute("train", Shareable(), fl_ctx, Signal())
            assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
        finally:
            backend.finalize(FLContext())


class TestResultReadyHandler:
    def _task(self, backend):
        task = ebp._TaskContext(task_id="task-1")
        with backend._task_lock:
            backend._current_task = task
        return task

    def _payload(self, backend, **overrides):
        payload = {
            MsgKey.SESSION_ID: backend._session.session_id,
            MsgKey.TASK_ID: "task-1",
            MsgKey.RESULT_ID: "res-1",
            MsgKey.RESULT: _result_shareable(),
        }
        payload.update(overrides)
        return payload

    def test_second_result_ready_is_rejected(self, env):
        backend, _ = _initialized_backend(env)
        try:
            task = self._task(backend)
            origin = backend._session.trainer_fqcn
            first = env.cell.deliver(Topic.RESULT_READY, origin, self._payload(backend))
            second = env.cell.deliver(Topic.RESULT_READY, origin, self._payload(backend))

            assert first.payload[MsgKey.REPLY_TOPIC] == Topic.RESULT_ACCEPTED
            assert second.payload[MsgKey.REPLY_TOPIC] == Topic.RESULT_REJECTED
            assert "already accepted" in second.payload[MsgKey.REASON]
            assert isinstance(task.result, Shareable)
        finally:
            backend.finalize(FLContext())

    def test_finalize_wins_after_result_validation_before_acceptance(self, env, monkeypatch):
        """Closure is rechecked in the acceptance critical section.

        RESULT_READY may pass authentication just before END_RUN. If finalize wins the
        task lock afterward, the callback must not create a newly live result source that
        teardown has already classified.
        """
        backend, _ = _initialized_backend(env, shutdown_timeout=0.0, stop_grace_period=0.0)
        task = self._task(backend)
        session = backend._session
        validation_done = threading.Event()
        release_validation = threading.Event()
        reply_box = {}
        real_validate = backend._validate_session_msg

        def validate_then_pause(request, payload):
            validated = real_validate(request, payload)
            validation_done.set()
            assert release_validation.wait(5.0)
            return validated

        monkeypatch.setattr(backend, "_validate_session_msg", validate_then_pause)
        result = _result_shareable()
        result["lazy"] = LazyDownloadRef("trainer", "ref-1", "T0")
        delivery = threading.Thread(
            target=lambda: reply_box.setdefault(
                "reply",
                env.cell.deliver(
                    Topic.RESULT_READY,
                    session.trainer_fqcn,
                    self._payload(backend, **{MsgKey.RESULT: result}),
                ),
            )
        )
        delivery.start()
        assert validation_done.wait(5.0)

        backend.finalize(FLContext())
        release_validation.set()
        delivery.join(timeout=5.0)

        assert not delivery.is_alive()
        assert reply_box["reply"].payload[MsgKey.REPLY_TOPIC] == Topic.RESULT_REJECTED
        assert reply_box["reply"].payload[MsgKey.REASON] == "backend is closed"
        assert task.result_id is None
        assert not task.result_ready.is_set()
        assert not session.result_source_live.is_set()

    def test_accepted_result_marks_the_trainer_send_as_pending(self, env):
        backend, _ = _initialized_backend(env)
        try:
            self._task(backend)
            reply = env.cell.deliver(
                Topic.RESULT_READY,
                backend._session.trainer_fqcn,
                self._payload(backend),
            )

            assert reply.payload[MsgKey.REPLY_TOPIC] == Topic.RESULT_ACCEPTED
            assert backend._session.result_source_live.is_set()
        finally:
            # This test does not model the trainer's send-completion acknowledgement.
            backend._session.result_source_live.clear()
            backend.finalize(FLContext())

    @pytest.mark.parametrize(
        "origin_override,payload_overrides,expect",
        [
            (None, {MsgKey.TASK_ID: "task-99"}, "no current task"),
            (None, {MsgKey.SESSION_ID: "stale"}, "session id"),
            ("site-1.job-1.client_api_trainer_99", {}, "origin"),
            (None, {MsgKey.RESULT: "not-a-shareable"}, "invalid result"),
        ],
        ids=["wrong_task", "stale_session", "wrong_origin", "empty_manifest"],
    )
    def test_invalid_result_ready_is_rejected(self, env, origin_override, payload_overrides, expect):
        backend, _ = _initialized_backend(env)
        try:
            task = self._task(backend)
            origin = origin_override or backend._session.trainer_fqcn
            reply = env.cell.deliver(Topic.RESULT_READY, origin, self._payload(backend, **payload_overrides))

            assert reply.payload[MsgKey.REPLY_TOPIC] == Topic.RESULT_REJECTED
            assert expect in reply.payload[MsgKey.REASON]
            assert not task.result_ready.is_set()
        finally:
            backend.finalize(FLContext())


class TestLaunchPerTask:
    def test_initialize_does_not_launch(self, env):
        backend, _ = _initialized_backend(env, launch_once=False)
        try:
            assert env.harness.processes == []
            assert backend._session is None
        finally:
            backend.finalize(FLContext())

    def test_each_task_gets_fresh_process_token_and_fqcn(self, env):
        backend, fl_ctx = _initialized_backend(env, launch_once=False)
        try:
            _install_auto_result(env, lazy_result=True)

            first = backend.execute("train", Shareable(), fl_ctx, Signal())
            assert first.get_return_code() == ReturnCode.OK
            assert len(env.harness.processes) == 1
            first_process = env.harness.processes[0]
            assert first_process.returncode is None, "CJ must not kill a trainer that still owns lazy result refs"
            first_process.exit(0)  # trainer exits itself after its result transfer settles
            deadline = time.monotonic() + 1.0
            while backend._session is not None and time.monotonic() < deadline:
                time.sleep(0.01)
            assert backend._session is None

            second = backend.execute("train", Shareable(), fl_ctx, Signal())
            assert second.get_return_code() == ReturnCode.OK
            assert len(env.harness.processes) == 2
            env.harness.processes[1].exit(0)
            deadline = time.monotonic() + 1.0
            while backend._session is not None and time.monotonic() < deadline:
                time.sleep(0.01)

            config_1, config_2 = env.harness.bootstrap_configs
            assert config_1[BootstrapKey.LAUNCH_TOKEN] != config_2[BootstrapKey.LAUNCH_TOKEN]
            assert config_1[BootstrapKey.TRAINER_FQCN].endswith("_1")
            assert config_2[BootstrapKey.TRAINER_FQCN].endswith("_2")
            assert config_1[BootstrapKey.TASK_EXCHANGE][ConfigKey.LAUNCH_ONCE] is False
            assert config_2[BootstrapKey.TASK_EXCHANGE][ConfigKey.LAUNCH_ONCE] is False

            # Bootstrap files are launch-scoped and removed after natural exit: a survivor
            # of launch 1 can never re-read a file and find launch 2's valid credentials.
            path_1 = env.harness.processes[0].kwargs["env"][BOOTSTRAP_FILE_ENV_VAR]
            path_2 = env.harness.processes[1].kwargs["env"][BOOTSTRAP_FILE_ENV_VAR]
            assert path_1 != path_2
            assert not os.path.exists(path_1) and not os.path.exists(path_2)
        finally:
            backend.finalize(FLContext())

    def test_inline_result_uses_the_send_completion_reaper(self, env):
        backend, fl_ctx = _initialized_backend(env, launch_once=False, shutdown_timeout=0.0)
        _install_auto_result(env)
        try:
            result = backend.execute("train", Shareable(), fl_ctx, Signal())
            process = env.harness.processes[0]
            session = backend._session

            assert result.get_return_code() == ReturnCode.OK
            assert process.returncode is None
            assert session.reaper_thread is not None and session.reaper_thread.is_alive()

            # Even an inline result can still be returning through RESULT_ACCEPTED when
            # execute() finishes. The real trainer exits after send() returns; model that
            # truthful completion instead of expecting the CJ to kill it synchronously.
            process.exit(0)
            deadline = time.monotonic() + 1.0
            while backend._session is not None and time.monotonic() < deadline:
                time.sleep(0.01)
            assert backend._session is None
        finally:
            backend.finalize(FLContext())

    def test_finalize_stops_retired_per_task_launch_replaced_by_next_session(self, env, monkeypatch):
        monkeypatch.setattr(ebp, "_NATURAL_EXIT_REAP_INTERVAL", 0.005)
        backend, fl_ctx = _initialized_backend(env, launch_once=False, heartbeat_timeout=0.03)
        _install_auto_result(env, lazy_result=True)

        first = backend.execute("train", Shareable(), fl_ctx, Signal())
        retired_session = backend._session
        second = backend.execute("train", Shareable(), fl_ctx, Signal())
        current_session = backend._session
        assert first.get_return_code() == ReturnCode.OK
        assert second.get_return_code() == ReturnCode.OK
        assert len(env.harness.processes) == 2
        assert all(process.returncode is None for process in env.harness.processes)
        assert retired_session is not current_session

        finalize_thread = threading.Thread(target=backend.finalize, args=(FLContext(),), daemon=True)
        finalize_thread.start()
        try:
            deadline = time.monotonic() + 1.0
            while (
                retired_session.reaper_thread is None or current_session.reaper_thread is None
            ) and time.monotonic() < deadline:
                time.sleep(0.005)
            assert retired_session.reaper_thread is not None
            assert current_session.reaper_thread is not None
            assert finalize_thread.is_alive(), "END_RUN must wait for every owned result source"

            # Terminal truth for only the newest source is insufficient: the older
            # launch is no longer in backend._session but still owns download refs.
            env.cell.disconnected.add(current_session.trainer_fqcn)
            deadline = time.monotonic() + 1.0
            while env.harness.processes[1].returncode is None and time.monotonic() < deadline:
                time.sleep(0.005)
            assert env.harness.processes[1].returncode is not None
            assert env.harness.processes[0].returncode is None
            assert finalize_thread.is_alive()

            env.cell.disconnected.add(retired_session.trainer_fqcn)
            finalize_thread.join(timeout=1.0)
            assert not finalize_thread.is_alive()
            assert all(process.returncode is not None for process in env.harness.processes)
            assert backend._result_reapers == set()
        finally:
            env.cell.disconnected.update((retired_session.trainer_fqcn, current_session.trainer_fqcn))
            finalize_thread.join(timeout=1.0)

    def test_abort_during_per_task_launch_returns_task_aborted(self, env):
        """The HELLO wait must not hang past abort (execute contract): with no
        launch_timeout and a live-but-silent trainer, only the abort bounds the wait."""
        env.harness.auto_hello = False
        backend, fl_ctx = _initialized_backend(env, launch_once=False, launch_timeout=None)
        try:
            abort_signal = Signal()
            threading.Timer(0.3, abort_signal.trigger, args=["stop"]).start()

            start = time.monotonic()
            result = backend.execute("train", Shareable(), fl_ctx, abort_signal)
            elapsed = time.monotonic() - start

            assert result.get_return_code() == ReturnCode.TASK_ABORTED
            assert elapsed < 10.0, "abort must end the HELLO wait"
            assert env.harness.processes[0].returncode is not None, "the aborted launch was unwound"
        finally:
            backend.finalize(FLContext())

    def test_execute_after_finalize_fails_without_launching(self, env):
        """A task racing END_RUN teardown must not launch a trainer after finalize()."""
        backend, fl_ctx = _initialized_backend(env, launch_once=False)
        backend.finalize(FLContext())

        result = backend.execute("train", Shareable(), fl_ctx, Signal())

        assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
        assert env.harness.processes == [], "no trainer may be launched after teardown"

    def test_stale_launch_hello_is_rejected_after_stop(self, env):
        backend, fl_ctx = _initialized_backend(env, launch_once=False)
        try:
            _install_auto_result(env)
            assert backend.execute("train", Shareable(), fl_ctx, Signal()).get_return_code() == ReturnCode.OK

            # a zombie of launch 1 tries to authenticate with its (now invalidated) token
            config_1 = env.harness.bootstrap_configs[0]
            reply = env.cell.deliver(
                Topic.HELLO,
                config_1[BootstrapKey.TRAINER_FQCN],
                {
                    MsgKey.TRAINER_FQCN: config_1[BootstrapKey.TRAINER_FQCN],
                    MsgKey.PROOF: config_1[BootstrapKey.LAUNCH_TOKEN],
                    MsgKey.PROTOCOL_VERSION: PROTOCOL_VERSION,
                    MsgKey.JOB_ID: "job-1",
                    MsgKey.RANK: 0,
                },
            )
            assert reply.payload[MsgKey.REPLY_TOPIC] == Topic.HELLO_REJECTED
        finally:
            backend.finalize(FLContext())

    def test_task_timeout_does_not_poison_next_task(self, env):
        """A previous task's dead session must not fail a per-task launch (fresh process,
        fresh state): the backend abort latch only fail-fasts launch_once sessions."""
        backend, fl_ctx = _initialized_backend(env, launch_once=False, result_wait_timeout=0.05)
        try:
            # task 1: trainer accepts but never sends a result -> timeout
            first = backend.execute("train", Shareable(), fl_ctx, Signal())
            assert first.get_return_code() == ReturnCode.EXECUTION_EXCEPTION

            # task 2: fresh launch works
            _install_auto_result(env)
            second = backend.execute("train", Shareable(), fl_ctx, Signal())
            assert second.get_return_code() == ReturnCode.OK
        finally:
            backend.finalize(FLContext())

    def test_per_task_launch_failure_returns_error_reply(self, env):
        backend, fl_ctx = _initialized_backend(env, launch_once=False)
        try:
            env.harness.exit_at_launch_rc = 3
            result = backend.execute("train", Shareable(), fl_ctx, Signal())
            assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
        finally:
            backend.finalize(FLContext())


class TestLogRouting:
    def _log_payload(self, backend, **overrides):
        payload = {
            MsgKey.SESSION_ID: backend._session.session_id,
            "key": "accuracy",
            "value": 0.9,
            "data_type": AnalyticsDataType.SCALAR,
        }
        payload.update(overrides)
        return payload

    def test_log_routes_through_executor_fire_log_analytics(self, env):
        executor = MagicMock()
        backend, _ = _initialized_backend(env, executor=executor)
        try:
            env.cell.deliver(Topic.LOG, backend._session.trainer_fqcn, self._log_payload(backend))

            assert executor.fire_log_analytics.call_count == 1
            dxo = executor.fire_log_analytics.call_args[0][1]
            # the key->tag rename happened and the DXO carries the exact metric
            assert dxo.data == {"track_key": "accuracy", "track_value": 0.9}
        finally:
            backend.finalize(FLContext())

    def test_stale_session_log_is_dropped(self, env, caplog):
        executor = MagicMock()
        backend, _ = _initialized_backend(env, executor=executor)
        try:
            env.cell.deliver(
                Topic.LOG,
                backend._session.trainer_fqcn,
                self._log_payload(backend, **{MsgKey.SESSION_ID: "stale-session"}),
            )
            executor.fire_log_analytics.assert_not_called()
            assert "dropping LOG data" in caplog.text
        finally:
            backend.finalize(FLContext())

    def test_invalid_log_data_is_logged_and_ignored(self, env, caplog):
        executor = MagicMock()
        backend, _ = _initialized_backend(env, executor=executor)
        try:
            env.cell.deliver(Topic.LOG, backend._session.trainer_fqcn, None)
            executor.fire_log_analytics.assert_not_called()
            assert "invalid LOG data format" in caplog.text
        finally:
            backend.finalize(FLContext())

    def test_log_processing_error_is_logged_and_ignored(self, env, caplog):
        executor = MagicMock()
        executor.fire_log_analytics.side_effect = RuntimeError("analytics failed")
        backend, _ = _initialized_backend(env, executor=executor)
        try:
            env.cell.deliver(Topic.LOG, backend._session.trainer_fqcn, self._log_payload(backend))
            assert "failed to process trainer LOG data" in caplog.text
        finally:
            backend.finalize(FLContext())
