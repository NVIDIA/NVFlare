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
that config exactly like the real one and HELLOs over the (fake) cell, TASK_READY goes out
with the minted transfer_id and payload refs, RESULT_READY comes back, and execute()
returns the pulled result. Also covers the backend-contract obligations: initialize()
self-unwinding, finalize() idempotency + process-tree teardown, bounded result wait, LOG
routing through the executor-owned fire_log_analytics(), the launch-token/identity checks
of the HELLO handshake, and the one-live-attempt-per-transfer_id payload discipline.

The payload seam (payload_transfer.py) is replaced by fakes here — its own contract
behavior is covered in tests/unit_test/client/cell/payload_transfer_test.py.
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
from nvflare.apis.fl_constant import FLContextKey, FLMetaKey, ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import UnsafeJobError
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.executors.client_api import external_process_backend as ebp
from nvflare.app_common.executors.client_api.backend_spec import ClientAPIBackendContext
from nvflare.app_common.executors.client_api.external_process_backend import ExternalProcessBackend, bootstrap_file_name
from nvflare.app_common.executors.client_api_executor import ClientAPIExecutor
from nvflare.client.cell.bootstrap import BOOTSTRAP_FILE_ENV_VAR, BootstrapKey, read_bootstrap_config
from nvflare.client.cell.defs import CHANNEL, PROTOCOL_VERSION, MsgKey, Topic
from nvflare.client.cell.payload_transfer import PayloadTransferError
from nvflare.client.config import ConfigKey, ExchangeFormat, TransferType
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.defs import ReturnCode as CellReturnCode
from nvflare.fuel.f3.cellnet.utils import make_reply as make_cell_reply
from nvflare.fuel.f3.cellnet.utils import new_cell_message

CJ_FQCN = "site-1.job-1"

PROTOCOL_TOPICS = (Topic.HELLO, Topic.RESULT_READY, Topic.TASK_FAILED, Topic.LOG, Topic.BYE)


class _HookOnFirstEnter:
    """A context manager that runs a hook the first time it is entered, then behaves as a
    plain lock. Used to force a teardown at a handler's commit-lock boundary (a two-phase
    per-task lock); inert against a handler that commits under the backend's _task_lock."""

    def __init__(self, hook):
        self._hook = hook
        self._lock = threading.Lock()
        self._fired = False

    def __enter__(self):
        if not self._fired:
            self._fired = True
            self._hook()
        self._lock.acquire()
        return self

    def __exit__(self, *exc):
        self._lock.release()


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
        # workers of the launched tree that outlive the leader (torchrun shape)
        self.extra_group_members = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
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
        self.listener_url = "tcp://127.0.0.1:56789"
        self.internal_listener_made = False
        self.cbs = {}
        self.sent = []  # (topic, target, payload, timeout)
        self.sent_kwargs = []  # extra kwargs per send_request (e.g. the cancel abort_signal)
        self.fired = []  # (topic, targets, payload)
        self.on_request = None  # fn(topic, target, request) -> reply Message
        self.on_fire = None  # fn(topic, targets, message)

    def get_fqcn(self):
        return self.fqcn

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


class FakeAttempt:
    """Stands in for payload_transfer.TaskPayloadAttempt (the producer-side seam)."""

    instances = []
    fail_on_create = None
    create_hook = None  # runs mid-construction (models teardown racing attempt creation)

    def __init__(self, cell, obj, receiver_fqcn):
        if FakeAttempt.fail_on_create is not None:
            raise FakeAttempt.fail_on_create
        FakeAttempt.instances.append(self)
        if FakeAttempt.create_hook is not None:
            FakeAttempt.create_hook()
        n = len(FakeAttempt.instances)
        self.cell = cell
        self.obj = obj
        self.receiver_fqcn = receiver_fqcn
        self.tx_id = f"tx-{n}"
        self.ref_id = f"ref-{n}"
        self._completed = False
        self._failed = False
        self._reason = None
        self.terminated = False
        self.wait_timeouts = []
        # test hook: the transfer certifies during the retire-path settle wait
        self.settle_on_wait = False

    @classmethod
    def reset(cls):
        cls.instances = []
        cls.fail_on_create = None
        cls.create_hook = None

    def completed(self):
        return self._completed

    def failed(self):
        return self._failed

    def failure_reason(self):
        return self._reason

    def wait(self, timeout=None, linger=None):
        self.wait_timeouts.append(timeout)
        if self.settle_on_wait:
            self._completed = True
        return self._completed

    def terminate(self):
        self.terminated = True

    # test hooks
    def mark_delivered(self):
        self._completed = True

    def mark_failed(self, reason):
        self._failed = True
        self._reason = reason


@pytest.fixture
def env(tmp_path, monkeypatch):
    cell = FakeCell()
    harness = FakeTrainerHarness(cell)
    holder = SimpleNamespace(
        cell=cell,
        harness=harness,
        app_dir=str(tmp_path),
        fetch_calls=[],
        fetch_error=None,
        fetch_results=None,
        fetch_hook=None,
    )
    # late-bound so a test may swap harness.popen (e.g. multi-HELLO launch sequences)
    monkeypatch.setattr(ebp.subprocess, "Popen", lambda *args, **kwargs: harness.popen(*args, **kwargs))
    monkeypatch.setattr(ebp, "log_subprocess_output", lambda process, logger: None)
    # start_new_session gives pgid == pid; route group signals to the fake process table
    monkeypatch.setattr(ebp.os, "killpg", harness.killpg, raising=False)
    FakeAttempt.reset()
    monkeypatch.setattr(ebp, "TaskPayloadAttempt", FakeAttempt)

    def fake_fetch(cell_, from_fqcn, ref_ids, abort_signal=None):
        holder.fetch_calls.append((from_fqcn, list(ref_ids)))
        if holder.fetch_hook is not None:
            holder.fetch_hook()
        if holder.fetch_error is not None:
            raise holder.fetch_error
        if holder.fetch_results is not None:
            return holder.fetch_results
        return [_result_shareable()]

    monkeypatch.setattr(ebp, "fetch_result_payload", fake_fetch)
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
        execution_mode="external_process",
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


def _install_auto_result(env, mark_delivered=True, result_transfer_id=None):
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
        if mark_delivered and FakeAttempt.instances:
            FakeAttempt.instances[-1].mark_delivered()
        task_payload = request.payload
        result_payload = {
            MsgKey.SESSION_ID: task_payload[MsgKey.SESSION_ID],
            MsgKey.TASK_ID: task_payload[MsgKey.TASK_ID],
            MsgKey.RESULT_ID: uuid.uuid4().hex,
            MsgKey.TRANSFER_ID: result_transfer_id or uuid.uuid4().hex,
            MsgKey.MANIFEST: {MsgKey.REF_IDS: [f"res-ref-{task_payload[MsgKey.TASK_ID]}"]},
        }
        seen.result_replies.append(env.cell.deliver(Topic.RESULT_READY, target, result_payload))
        return _task_accepted_reply()

    env.cell.on_request = handler
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
            for topic in PROTOCOL_TOPICS:
                assert topic in env.cell.cbs, f"backend must register a handler for {topic}"
            # #4865's TASK_PAYLOAD_READY is acked cleanly (stateless, no-op materialization
            # signal in this heartbeat-less backend)
            assert Topic.TASK_PAYLOAD_READY in env.cell.cbs
            ack = env.cell.deliver(Topic.TASK_PAYLOAD_READY, backend._session.trainer_fqcn, {MsgKey.TASK_ID: "t-1"})
            assert ack.get_header(MessageHeaderKey.RETURN_CODE) == CellReturnCode.OK
            # LOG analytics go out federation-scoped (MetricRelay ex-process parity)
            executor.set_analytics_fire_fed_event.assert_called_once_with(True)
            # process launched in its own group with the bootstrap path in its env
            process = env.harness.processes[0]
            assert process.kwargs["start_new_session"] is (os.name == "posix")
            assert process.kwargs["cwd"] == env.app_dir
            assert BOOTSTRAP_FILE_ENV_VAR in process.kwargs["env"]
            # pgid breadcrumb for orphan reaping by the CP (design: "CJ Failure")
            pgid_path = os.path.join(env.app_dir, ebp.pgid_file_name(1))
            with open(pgid_path) as f:
                assert f.read() == str(process.pid)
        finally:
            backend.finalize(FLContext())
        assert not backend._session
        # the tree is confirmed stopped: breadcrumb and token file are gone
        assert not os.path.exists(os.path.join(env.app_dir, ebp.pgid_file_name(1)))
        assert not os.path.exists(os.path.join(env.app_dir, bootstrap_file_name(1)))

    def test_bootstrap_config_contents_and_permission(self, env):
        backend, _ = _initialized_backend(
            env, train_task_name="my_train", evaluate_task_name="my_eval", submit_model_task_name="my_submit"
        )
        try:
            path = os.path.join(env.app_dir, bootstrap_file_name(1))
            assert os.stat(path).st_mode & 0o777 == 0o600, "launch token must be owner-readable only"
            config = read_bootstrap_config(path)
            assert config[BootstrapKey.CONNECT_URL] == env.cell.listener_url
            assert config[BootstrapKey.CJ_FQCN] == CJ_FQCN
            assert config[BootstrapKey.TRAINER_FQCN] == f"{CJ_FQCN}.client_api_trainer_1"
            assert config[BootstrapKey.LAUNCH_TOKEN]
            assert config[BootstrapKey.PROTOCOL_VERSION] == PROTOCOL_VERSION
            assert config[BootstrapKey.JOB_ID] == "job-1"
            assert config[BootstrapKey.SITE_NAME] == "site-1"
            exchange = config[BootstrapKey.TASK_EXCHANGE]
            # FLARE-2698: pass-through boundary - no converter formats in the frozen surface
            assert exchange[ConfigKey.EXCHANGE_FORMAT] == ExchangeFormat.RAW
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

    def test_second_backend_on_same_cell_is_rejected(self, env):
        """V1: one external_process backend per CJ cell — a second one would silently
        overwrite the first's protocol handlers and launch namespace, cross-wiring both
        trainers. It must be rejected deterministically at initialize (-> panic)."""
        backend_1, _ = _initialized_backend(env)
        try:
            backend_2 = ExternalProcessBackend()
            fl_ctx = _make_fl_ctx(_make_engine(env.cell), env.app_dir)
            with pytest.raises(RuntimeError, match="already active"):
                backend_2.initialize(_make_context(), fl_ctx)

            # the loser's unwind must not evict the winner: the slot still points at
            # backend_1 (its _release_cell owner-guard held), handlers and session intact
            assert ebp._GUARD.owner_of(env.cell) is backend_1
            assert env.cell.cbs[Topic.HELLO].__self__ is backend_1
            assert backend_1._session.ready.is_set()
            assert len(env.harness.processes) == 1, "the rejected backend must not have launched"

            # and a THIRD backend must still be rejected — a rejected loser cannot have
            # silently freed the live winner's slot on its way out
            backend_3 = ExternalProcessBackend()
            with pytest.raises(RuntimeError, match="already active"):
                backend_3.initialize(_make_context(), _make_fl_ctx(_make_engine(env.cell), env.app_dir))
            assert ebp._GUARD.owner_of(env.cell) is backend_1
        finally:
            backend_1.finalize(FLContext())
        # winner released at finalize
        assert ebp._GUARD.owner_of(env.cell) is None

    def test_cell_slot_released_at_finalize_for_the_next_job(self, env):
        """Sequential jobs on one process (simulator) reuse the pattern: finalize must
        release the one-backend-per-cell slot."""
        backend_1, _ = _initialized_backend(env)
        backend_1.finalize(FLContext())

        backend_2, _ = _initialized_backend(env)
        try:
            assert backend_2._session is not None and backend_2._session.ready.is_set()
        finally:
            backend_2.finalize(FLContext())

    def test_slot_held_until_teardown_completes_not_merely_closed(self, env, monkeypatch):
        """The slot must be held until _release_cell(), NOT merely until _closed is set: a
        backend mid-teardown still owns launch seq 1's FQCN/artifacts and its handlers, so
        a second initialize racing that window must still be rejected (it would otherwise
        relaunch at seq 1 and let the old teardown delete the new launch's artifacts)."""
        backend_1, _ = _initialized_backend(env)
        in_teardown = threading.Event()
        release_teardown = threading.Event()
        real_stop = backend_1._stop_session

        def blocking_stop(session, natural_exit_wait):
            # _closed is already True here (finalize set it before calling _stop_session)
            in_teardown.set()
            release_teardown.wait(10.0)
            return real_stop(session, natural_exit_wait)

        monkeypatch.setattr(backend_1, "_stop_session", blocking_stop)
        finalizer = threading.Thread(target=lambda: backend_1.finalize(FLContext()))
        finalizer.start()
        try:
            assert in_teardown.wait(5.0), "finalize did not reach teardown"
            assert backend_1._closed, "precondition: the owner is closed but not yet released"

            backend_2 = ExternalProcessBackend()
            fl_ctx = _make_fl_ctx(_make_engine(env.cell), env.app_dir)
            with pytest.raises(RuntimeError, match="already active"):
                backend_2.initialize(_make_context(), fl_ctx)
        finally:
            release_teardown.set()
            finalizer.join(10.0)

        # once teardown finished and released the slot, a fresh backend may claim it
        backend_3, _ = _initialized_backend(env)
        backend_3.finalize(FLContext())

    def test_finalize_idempotent_and_stops_process_tree(self, env):
        backend, _ = _initialized_backend(env)
        process = env.harness.processes[0]
        session = backend._session

        backend.finalize(FLContext())
        backend.finalize(FLContext())  # contract: idempotent, must not raise

        shutdowns = [f for f in env.cell.fired if f[0] == Topic.SHUTDOWN]
        assert len(shutdowns) == 1, "idempotency means no repeated side effects"
        assert process.returncode is not None
        # launch-scoped token invalidation
        assert session.token == "" and session.session_id is None
        assert backend._session is None

    def test_finalize_prefers_natural_exit_over_termination(self, env):
        backend, _ = _initialized_backend(env)
        process = env.harness.processes[0]

        # a well-behaved trainer exits on SHUTDOWN before the natural-exit bound
        def exit_on_shutdown(topic, targets, message):
            if topic == Topic.SHUTDOWN:
                process.exit(0)

        env.cell.on_fire = exit_on_shutdown
        backend.finalize(FLContext())

        assert process.returncode == 0
        assert env.harness.signals_sent() == [], "no signals for a trainer that exited naturally"

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
        backend, _ = _initialized_backend(env)
        process = env.harness.processes[0]
        process.exit(0)
        process.extra_group_members = True

        backend.finalize(FLContext())

        assert (process.pid, signal.SIGTERM) in env.harness.signals_sent()
        assert not process.extra_group_members, "surviving workers were signaled"
        # group confirmed gone -> breadcrumb removed
        assert not os.path.exists(os.path.join(env.app_dir, ebp.pgid_file_name(1)))

    @pytest.mark.skipif(os.name != "posix", reason="process-group semantics are POSIX")
    def test_pgid_breadcrumb_kept_while_group_survives(self, env, monkeypatch):
        """A tree that survives SIGKILL keeps its breadcrumb so the CP reaper (follow-up
        of the design's CJ Failure section) can still find and kill it."""
        backend, _ = _initialized_backend(env, stop_grace_period=0.01)
        # signals are recorded but never take the group down; probes report it alive
        monkeypatch.setattr(env.harness, "killpg", lambda pgid, sig: env.harness.killpg_calls.append((pgid, sig)))
        monkeypatch.setattr(ebp.os, "killpg", env.harness.killpg, raising=False)
        monkeypatch.setattr(ebp, "_LOG_THREAD_JOIN_TIMEOUT", 0.2)

        backend.finalize(FLContext())

        assert os.path.exists(os.path.join(env.app_dir, ebp.pgid_file_name(1)))

    def test_pgid_breadcrumbs_are_launch_scoped(self, env):
        """A group surviving launch N must keep ITS breadcrumb even after launch N+1
        writes its own — a shared fixed file name would truncate away exactly the
        recovery information the surviving group needs."""
        backend, fl_ctx = _initialized_backend(env, launch_once=False)
        try:
            _install_auto_result(env)
            assert backend.execute("train", Shareable(), fl_ctx, Signal()).get_return_code() == ReturnCode.OK
            assert backend.execute("train", Shareable(), fl_ctx, Signal()).get_return_code() == ReturnCode.OK
            # distinct per-launch paths (both removed here because both groups died)
            assert ebp.pgid_file_name(1) != ebp.pgid_file_name(2)
            assert not os.path.exists(os.path.join(env.app_dir, ebp.pgid_file_name(1)))
            assert not os.path.exists(os.path.join(env.app_dir, ebp.pgid_file_name(2)))
        finally:
            backend.finalize(FLContext())

    def test_command_split_is_platform_appropriate(self, monkeypatch):
        """POSIX shlex rules would corrupt backslashed Windows paths; on Windows the
        command string goes to CreateProcess unsplit."""
        monkeypatch.setattr(ebp.os, "name", "posix")
        assert ExternalProcessBackend._split_command("python -u custom/train.py") == [
            "python",
            "-u",
            "custom/train.py",
        ]
        monkeypatch.setattr(ebp.os, "name", "nt")
        win_command = "python C:\\work\\train.py"
        assert ExternalProcessBackend._split_command(win_command) == win_command

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

        def boom(channel, topic, targets, message, **kwargs):
            raise RuntimeError("cell send failed")

        env.cell.fire_and_forget = boom
        backend.finalize(FLContext())  # must not raise

        assert "cell send failed" in caplog.text
        assert process.returncode is not None, "a failing SHUTDOWN send must not skip process teardown"


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
            ({MsgKey.RANK: 1}, "rank"),
        ],
        ids=["bad_version", "bad_job_id", "nonzero_rank"],
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
        task = ebp._TaskContext(task_id="task-1", task_name="train", transfer_id="xfer-1")
        backend._current_task = task

        reply = env.cell.deliver(
            Topic.RESULT_READY,
            session.trainer_fqcn,
            {
                MsgKey.SESSION_ID: session.session_id,
                MsgKey.TASK_ID: "task-1",
                MsgKey.RESULT_ID: "res-1",
                MsgKey.TRANSFER_ID: "xfer-up-1",
                MsgKey.MANIFEST: {MsgKey.REF_IDS: ["r1"]},
            },
        )
        assert reply.payload[MsgKey.REPLY_TOPIC] == Topic.RESULT_REJECTED
        assert reply.payload[MsgKey.REASON] == "backend is closed"
        assert not task.result_ready.is_set()

    def test_closed_gate_ignores_task_failed(self, env):
        backend, session = self._finalized_backend_with_session(env)
        task = ebp._TaskContext(task_id="task-1", task_name="train", transfer_id="xfer-1")
        backend._current_task = task

        env.cell.deliver(
            Topic.TASK_FAILED,
            session.trainer_fqcn,
            {MsgKey.SESSION_ID: session.session_id, MsgKey.TASK_ID: "task-1", MsgKey.REASON: "late failure"},
        )
        assert not task.failed.is_set()

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
            assert payload[MsgKey.TASK_ID] and payload[MsgKey.TRANSFER_ID]
            attempt = FakeAttempt.instances[0]
            assert payload[MsgKey.MODEL] == {MsgKey.REF_IDS: [attempt.ref_id]}
            # the payload attempt targeted the trainer and carried the task shareable
            assert attempt.receiver_fqcn == session.trainer_fqcn
            assert attempt.obj is task
            assert task.get_header(FLMetaKey.JOB_ID) == "job-1"
            assert task.get_header(FLMetaKey.SITE_NAME) == "site-1"
            # RESULT_READY was control-acked, and the payload was pulled from the trainer
            assert seen.result_replies[0].payload[MsgKey.REPLY_TOPIC] == Topic.RESULT_ACCEPTED
            assert env.fetch_calls == [(session.trainer_fqcn, [f"res-ref-{payload[MsgKey.TASK_ID]}"])]
            # a delivered attempt is retired without termination
            assert not attempt.terminated
            assert backend._live_attempts == {}
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
                    MsgKey.TRANSFER_ID: "xfer-up-async",
                    MsgKey.MANIFEST: {MsgKey.REF_IDS: ["r-async"]},
                }
                threading.Timer(0.05, env.cell.deliver, args=(Topic.RESULT_READY, target, result_payload)).start()
                return _task_accepted_reply()

            env.cell.on_request = handler
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
            assert FakeAttempt.instances[0].terminated
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

    def test_abort_during_result_pull_returns_task_aborted(self, env):
        """An abort the pull observed (download_object fails the pull on a triggered
        signal) must be reported as TASK_ABORTED with the session ended — not as a
        generic failure a workflow might retry against a dead session."""
        backend, fl_ctx = _initialized_backend(env, result_wait_timeout=30.0)
        try:
            _install_auto_result(env)
            abort_signal = Signal()
            env.fetch_hook = lambda: abort_signal.trigger("stop")
            env.fetch_error = PayloadTransferError("pull aborted")

            result = backend.execute("train", Shareable(), fl_ctx, abort_signal)

            assert result.get_return_code() == ReturnCode.TASK_ABORTED
            assert [f for f in env.cell.fired if f[0] == Topic.ABORT]
            assert backend._abort, "the launch_once session is over after an aborted send"
        finally:
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
            FakeAttempt.instances[-1].mark_delivered()
            env.cell.deliver(
                Topic.RESULT_READY,
                target,
                {
                    MsgKey.SESSION_ID: payload[MsgKey.SESSION_ID],
                    MsgKey.TASK_ID: payload[MsgKey.TASK_ID],
                    MsgKey.RESULT_ID: "res-concurrent",
                    MsgKey.TRANSFER_ID: "xfer-up-concurrent",
                    MsgKey.MANIFEST: {MsgKey.REF_IDS: ["r-concurrent"]},
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

            # second task arrives while the first is active -> must be rejected, not admitted
            second = backend.execute("evaluate", Shareable(), fl_ctx, Signal())
            assert second.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
            assert backend._current_task is not None, "the rejected task must not clear the active one"
            assert backend._current_task.task_name == "train"

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
            FakeAttempt.instances[-1].mark_delivered()
            env.cell.deliver(
                Topic.RESULT_READY,
                target,
                {
                    MsgKey.SESSION_ID: payload[MsgKey.SESSION_ID],
                    MsgKey.TASK_ID: payload[MsgKey.TASK_ID],
                    MsgKey.RESULT_ID: "res-busy",
                    MsgKey.TRANSFER_ID: "xfer-busy",
                    MsgKey.MANIFEST: {MsgKey.REF_IDS: ["r-busy"]},
                },
            )
            return _task_accepted_reply()

        env.cell.on_request = handler
        try:
            t1 = threading.Thread(target=lambda: backend.execute("train", Shareable(), fl_ctx, Signal()))
            t1.start()
            assert first_in_flight.wait(5.0)

            aborted = Signal()
            aborted.trigger("stop")
            second = backend.execute("evaluate", Shareable(), fl_ctx, aborted)
            assert second.get_return_code() == ReturnCode.TASK_ABORTED
            # the active task's session was untouched: no ABORT was sent for the rejected call
            assert backend._current_task.task_name == "train"

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
            # the trainer was told to stop the task, and the payload attempt was terminated
            assert [f for f in env.cell.fired if f[0] == Topic.ABORT]
            assert FakeAttempt.instances[0].terminated
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

    def test_trainer_task_failed_fails_task_before_timeout(self, env):
        backend, fl_ctx = _initialized_backend(env, result_wait_timeout=30.0)
        try:

            def accept_then_fail(topic, target, request):
                payload = request.payload
                env.cell.deliver(
                    Topic.TASK_FAILED,
                    target,
                    {
                        MsgKey.SESSION_ID: payload[MsgKey.SESSION_ID],
                        MsgKey.TASK_ID: payload[MsgKey.TASK_ID],
                        MsgKey.REASON: "task payload download failed",
                    },
                )
                return _task_accepted_reply()

            env.cell.on_request = accept_then_fail
            start = time.monotonic()
            result = backend.execute("train", Shareable(), fl_ctx, Signal())
            elapsed = time.monotonic() - start

            assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
            assert elapsed < 10.0, "TASK_FAILED must not wait out the result bound"
        finally:
            backend.finalize(FLContext())

    def test_task_payload_delivery_failure_fails_task(self, env):
        backend, fl_ctx = _initialized_backend(env, result_wait_timeout=30.0)
        try:

            def accept_then_fail_transfer(topic, target, request):
                FakeAttempt.instances[-1].mark_failed("receiver idle budget exhausted")
                return _task_accepted_reply()

            env.cell.on_request = accept_then_fail_transfer
            start = time.monotonic()
            result = backend.execute("train", Shareable(), fl_ctx, Signal())
            elapsed = time.monotonic() - start

            assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
            assert elapsed < 10.0, "a failed task-payload attempt must not wait out the result bound"
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
            assert FakeAttempt.instances[0].terminated, "an unaccepted task must terminate its payload attempt"
        finally:
            backend.finalize(FLContext())

    def test_result_pull_failure_returns_execution_exception(self, env):
        backend, fl_ctx = _initialized_backend(env)
        try:
            _install_auto_result(env)
            env.fetch_error = PayloadTransferError("pull failed")
            result = backend.execute("train", Shareable(), fl_ctx, Signal())
            assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
        finally:
            backend.finalize(FLContext())

    def test_bad_result_type_returns_execution_exception(self, env):
        backend, fl_ctx = _initialized_backend(env)
        try:
            _install_auto_result(env)
            env.fetch_results = ["not-a-shareable"]
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
            # each round minted a fresh transfer_id and a fresh attempt (attempt-scoped ids)
            transfer_ids = [payload[MsgKey.TRANSFER_ID] for _, _, payload, _ in env.cell.sent]
            assert len(set(transfer_ids)) == 3
            assert len(FakeAttempt.instances) == 3
        finally:
            backend.finalize(FLContext())

    def test_unsafe_job_error_propagates(self, env, monkeypatch):
        backend, fl_ctx = _initialized_backend(env)
        try:
            monkeypatch.setattr(backend, "_create_task_attempt", Mock(side_effect=UnsafeJobError("unsafe")))
            with pytest.raises(UnsafeJobError, match="unsafe"):
                backend.execute("train", Shareable(), fl_ctx, Signal())
        finally:
            backend.finalize(FLContext())

    def test_execute_exception_returns_execution_exception(self, env, monkeypatch):
        backend, fl_ctx = _initialized_backend(env)
        try:
            monkeypatch.setattr(backend, "_create_task_attempt", Mock(side_effect=RuntimeError("boom")))
            result = backend.execute("train", Shareable(), fl_ctx, Signal())
            assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
        finally:
            backend.finalize(FLContext())

    def test_handle_event_is_noop(self):
        backend = ExternalProcessBackend()
        backend.handle_event("custom_event", FLContext())


class TestResultReadyHandler:
    def _task(self, backend):
        task = ebp._TaskContext(task_id="task-1", task_name="train", transfer_id="xfer-1")
        with backend._task_lock:
            backend._current_task = task
        return task

    def _payload(self, backend, **overrides):
        payload = {
            MsgKey.SESSION_ID: backend._session.session_id,
            MsgKey.TASK_ID: "task-1",
            MsgKey.RESULT_ID: "res-1",
            MsgKey.TRANSFER_ID: "xfer-up-1",
            MsgKey.MANIFEST: {MsgKey.REF_IDS: ["r1"]},
        }
        payload.update(overrides)
        return payload

    def test_duplicate_result_ready_is_idempotent(self, env):
        backend, _ = _initialized_backend(env)
        try:
            task = self._task(backend)
            origin = backend._session.trainer_fqcn
            first = env.cell.deliver(Topic.RESULT_READY, origin, self._payload(backend))
            second = env.cell.deliver(Topic.RESULT_READY, origin, self._payload(backend))

            assert first.payload[MsgKey.REPLY_TOPIC] == Topic.RESULT_ACCEPTED
            # retry rule: duplicate RESULT_READY replies the current state, no second result
            assert second.payload[MsgKey.REPLY_TOPIC] == Topic.RESULT_ACCEPTED
            assert task.result_ref_ids == ["r1"]

            different = env.cell.deliver(
                Topic.RESULT_READY, origin, self._payload(backend, **{MsgKey.RESULT_ID: "res-2"})
            )
            assert different.payload[MsgKey.REPLY_TOPIC] == Topic.RESULT_REJECTED
        finally:
            backend.finalize(FLContext())

    def test_result_ready_retry_after_task_cleared_is_idempotent(self, env):
        """The lost-RESULT_ACCEPTED-reply sequence: the CJ accepted and consumed the
        result, cleared _current_task, then the trainer retries the SAME result. It must
        get RESULT_ACCEPTED from the receipt ring — not "no current task", which would
        split the persistent trainer from the CJ (design's RESULT_READY retry rule)."""
        backend, _ = _initialized_backend(env)
        try:
            task = self._task(backend)
            origin = backend._session.trainer_fqcn
            accept = env.cell.deliver(Topic.RESULT_READY, origin, self._payload(backend))
            assert accept.payload[MsgKey.REPLY_TOPIC] == Topic.RESULT_ACCEPTED

            # simulate _run_task's finally: clear the current task AND record the receipt
            with backend._task_lock:
                backend._current_task = None
                backend._record_result_receipt_locked(task.task_id, task.result_id)

            # the trainer's retry (reply was lost) — same task_id + result_id
            retry = env.cell.deliver(Topic.RESULT_READY, origin, self._payload(backend))
            assert retry.payload[MsgKey.REPLY_TOPIC] == Topic.RESULT_ACCEPTED

            # a DIFFERENT result_id for the already-accepted task is still rejected
            other = env.cell.deliver(Topic.RESULT_READY, origin, self._payload(backend, **{MsgKey.RESULT_ID: "res-2"}))
            assert other.payload[MsgKey.REPLY_TOPIC] == Topic.RESULT_REJECTED
            assert "already accepted" in other.payload[MsgKey.REASON]

            # an UNKNOWN task id still gets the plain "no current task" rejection
            unknown = env.cell.deliver(Topic.RESULT_READY, origin, self._payload(backend, **{MsgKey.TASK_ID: "nope"}))
            assert unknown.payload[MsgKey.REPLY_TOPIC] == Topic.RESULT_REJECTED
            assert "no current task" in unknown.payload[MsgKey.REASON]
        finally:
            backend.finalize(FLContext())

    def test_result_ready_retry_after_full_execute_round_is_idempotent(self, env):
        """End-to-end: a real execute() round records the receipt via _run_task's finally
        (not a simulated clear), so a retry of that round's RESULT_READY after execute
        returned is answered RESULT_ACCEPTED. Exercises the actual finally path."""
        backend, fl_ctx = _initialized_backend(env)
        captured = {}

        def handler(topic, target, request):
            payload = request.payload
            result_payload = {
                MsgKey.SESSION_ID: payload[MsgKey.SESSION_ID],
                MsgKey.TASK_ID: payload[MsgKey.TASK_ID],
                MsgKey.RESULT_ID: "res-e2e",
                MsgKey.TRANSFER_ID: "xfer-up-e2e",
                MsgKey.MANIFEST: {MsgKey.REF_IDS: ["r-e2e"]},
            }
            captured["result_payload"] = result_payload
            captured["origin"] = target
            FakeAttempt.instances[-1].mark_delivered()
            env.cell.deliver(Topic.RESULT_READY, target, result_payload)
            return _task_accepted_reply()

        env.cell.on_request = handler
        try:
            assert backend.execute("train", Shareable(), fl_ctx, Signal()).get_return_code() == ReturnCode.OK
            assert backend._current_task is None, "precondition: the task was cleared"

            # the trainer never saw RESULT_ACCEPTED (reply lost) and retries the same result
            retry = env.cell.deliver(Topic.RESULT_READY, captured["origin"], captured["result_payload"])
            assert retry.payload[MsgKey.REPLY_TOPIC] == Topic.RESULT_ACCEPTED
        finally:
            backend.finalize(FLContext())

    def test_result_ready_validation_and_commit_are_atomic(self, env):
        """Round-7 race regression (DETERMINISTIC): teardown must not clear _current_task
        between the handler's current-task validation and its acceptance commit. This forces
        a teardown exactly at the commit boundary. Under the atomic fix (validate+commit in
        one _task_lock section) the hook is never reached, so the handler either commits
        fully (task still current) or would have seen the task cleared and rejected — never a
        RESULT_ACCEPTED with no receipt. The invariant asserted: an accepted result is always
        answerable on retry. The hook fires only if the handler commits under a separate
        per-task lock (the removed two-phase gap), so this is a red guard against
        reintroducing that split."""
        backend, _ = _initialized_backend(env)
        try:
            task = self._task(backend)  # installs _current_task
            origin = backend._session.trainer_fqcn

            def teardown():
                # exactly what _run_task's finally does — clear + conditional receipt
                with backend._task_lock:
                    if backend._current_task is task:
                        backend._current_task = None
                    if task.result_id is not None:
                        backend._record_result_receipt_locked(task.task_id, task.result_id)

            # inert under the atomic fix (the handler never takes a per-task lock); fires at
            # the commit boundary if a two-phase per-task-lock commit is reintroduced
            task.lock = _HookOnFirstEnter(teardown)

            reply = env.cell.deliver(Topic.RESULT_READY, origin, self._payload(backend))
            if reply.payload[MsgKey.REPLY_TOPIC] == Topic.RESULT_ACCEPTED:
                retry = env.cell.deliver(Topic.RESULT_READY, origin, self._payload(backend))
                assert (
                    retry.payload[MsgKey.REPLY_TOPIC] == Topic.RESULT_ACCEPTED
                ), "accepted result not answerable on retry: commit interleaved with teardown"
        finally:
            backend.finalize(FLContext())

    def test_result_receipt_ring_is_bounded(self, env):
        backend, _ = _initialized_backend(env)
        try:
            for i in range(ebp._RESULT_RECEIPT_RING + 5):
                with backend._task_lock:
                    backend._record_result_receipt_locked(f"task-{i}", f"res-{i}")
            assert len(backend._result_receipts) == ebp._RESULT_RECEIPT_RING
            # the oldest were evicted; the most recent are retained
            assert "task-0" not in backend._result_receipts
            assert f"task-{ebp._RESULT_RECEIPT_RING + 4}" in backend._result_receipts
        finally:
            backend.finalize(FLContext())

    @pytest.mark.parametrize(
        "origin_override,payload_overrides,expect",
        [
            (None, {MsgKey.TASK_ID: "task-99"}, "no current task"),
            (None, {MsgKey.SESSION_ID: "stale"}, "session id"),
            ("site-1.job-1.client_api_trainer_99", {}, "origin"),
            (None, {MsgKey.MANIFEST: {}}, "invalid result envelope"),
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


class TestOneLiveAttempt:
    def test_duplicate_transfer_id_raises(self, env):
        backend, _ = _initialized_backend(env)
        try:
            backend._create_task_attempt("xfer-1", Shareable(), "trainer-fqcn")
            with pytest.raises(ValueError, match="duplicate live payload attempt"):
                backend._create_task_attempt("xfer-1", Shareable(), "trainer-fqcn")
        finally:
            backend.finalize(FLContext())

    def test_retry_after_retire_is_a_new_attempt_with_new_tx_id(self, env):
        backend, _ = _initialized_backend(env)
        try:
            first = backend._create_task_attempt("xfer-1", Shareable(), "trainer-fqcn")
            backend._retire_task_attempt("xfer-1", first)
            # an undelivered attempt first gets the bounded settle wait, THEN termination
            assert first.wait_timeouts == [ebp._ATTEMPT_SETTLE_WAIT]
            assert first.terminated, "an undelivered attempt is terminated at retire"

            second = backend._create_task_attempt("xfer-1", Shareable(), "trainer-fqcn")
            assert second is not first and second.tx_id != first.tx_id, "a retry is a NEW attempt under a NEW tx_id"
        finally:
            backend.finalize(FLContext())

    def test_attempt_that_settles_during_retire_wait_is_not_terminated(self, env):
        """The settle wait exists so a transfer certifying within the window is never
        killed mid-certification."""
        backend, _ = _initialized_backend(env)
        try:
            attempt = backend._create_task_attempt("xfer-1", Shareable(), "trainer-fqcn")
            attempt.settle_on_wait = True
            backend._retire_task_attempt("xfer-1", attempt)
            assert attempt.wait_timeouts == [ebp._ATTEMPT_SETTLE_WAIT]
            assert not attempt.terminated
            assert backend._live_attempts == {}
        finally:
            backend.finalize(FLContext())

    def test_teardown_racing_attempt_creation_does_not_resurrect_the_attempt(self, env):
        """A sweep running between the registry reservation and the attempt's publication
        must win: the freshly created attempt is terminated instead of published, so
        nothing lives past END_RUN."""
        backend, _ = _initialized_backend(env)

        def teardown_mid_create():
            backend._closed = True
            backend._sweep_live_attempts()

        FakeAttempt.create_hook = teardown_mid_create
        with pytest.raises(RuntimeError, match="closed"):
            backend._create_task_attempt("xfer-race", Shareable(), "trainer-fqcn")

        assert FakeAttempt.instances[0].terminated, "the attempt teardown never saw must terminate itself"
        assert backend._live_attempts == {}

    def test_finalize_terminates_leaked_live_attempts(self, env):
        """Teardown with an attempt still live (task interrupted between creation and
        retire) must not leak the payload transaction."""
        backend, _ = _initialized_backend(env)
        attempt = backend._create_task_attempt("xfer-leak", Shareable(), "trainer-fqcn")

        backend.finalize(FLContext())

        assert attempt.terminated
        assert backend._live_attempts == {}

    def test_failed_attempt_creation_leaves_registry_clean(self, env):
        backend, _ = _initialized_backend(env)
        try:
            FakeAttempt.fail_on_create = RuntimeError("no payload layer")
            with pytest.raises(RuntimeError, match="no payload layer"):
                backend._create_task_attempt("xfer-1", Shareable(), "trainer-fqcn")
            FakeAttempt.fail_on_create = None
            assert backend._live_attempts == {}
            backend._create_task_attempt("xfer-1", Shareable(), "trainer-fqcn")  # reservation was released
        finally:
            backend.finalize(FLContext())

    def test_delivered_attempt_is_not_terminated_at_retire(self, env):
        backend, _ = _initialized_backend(env)
        try:
            attempt = backend._create_task_attempt("xfer-1", Shareable(), "trainer-fqcn")
            attempt.mark_delivered()
            backend._retire_task_attempt("xfer-1", attempt)
            assert not attempt.terminated
            assert attempt.wait_timeouts == [], "a certified attempt needs no settle wait"
            assert backend._live_attempts == {}
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
            _install_auto_result(env)

            first = backend.execute("train", Shareable(), fl_ctx, Signal())
            assert first.get_return_code() == ReturnCode.OK
            assert len(env.harness.processes) == 1
            assert env.harness.processes[0].returncode is not None, "per-task process stops with its task"
            assert backend._session is None

            second = backend.execute("train", Shareable(), fl_ctx, Signal())
            assert second.get_return_code() == ReturnCode.OK
            assert len(env.harness.processes) == 2

            config_1, config_2 = env.harness.bootstrap_configs
            assert config_1[BootstrapKey.LAUNCH_TOKEN] != config_2[BootstrapKey.LAUNCH_TOKEN]
            assert config_1[BootstrapKey.TRAINER_FQCN].endswith("_1")
            assert config_2[BootstrapKey.TRAINER_FQCN].endswith("_2")

            # bootstrap files are launch-scoped AND removed at stop: a survivor of launch 1
            # can never re-read a file and find launch 2's valid credentials
            path_1 = env.harness.processes[0].kwargs["env"][BOOTSTRAP_FILE_ENV_VAR]
            path_2 = env.harness.processes[1].kwargs["env"][BOOTSTRAP_FILE_ENV_VAR]
            assert path_1 != path_2
            assert not os.path.exists(path_1) and not os.path.exists(path_2)
        finally:
            backend.finalize(FLContext())

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
