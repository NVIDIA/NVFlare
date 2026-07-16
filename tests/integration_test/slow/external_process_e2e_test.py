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

"""End-to-end proof: a real subprocess FedAvg job on ClientAPIExecutor(external_process).

This is the "examples run correctly" regression guard for the external_process execution mode
(FLARE-2698): it runs an actual numpy FedAvg job in the simulator where each client launches a
REAL trainer subprocess that speaks the new Cell protocol (ExternalProcessBackend <-> the
trainer-side CellClientAPI) over real cellnet, pulls the global model, trains, and streams the
result back through the shared F3 payload layer — the whole stack, not fakes.

It is intentionally heavier than the unit tests (spawns processes and stands up cellnet), and it
points the launched subprocess at the nvflare under test via PYTHONPATH so it works whether or
not the editable install targets this tree.
"""

import os
import subprocess
import sys
import textwrap

import pytest

import nvflare

# repo root of the nvflare under test, so the launched trainer subprocess imports THIS tree
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(nvflare.__file__)))

_CLIENT_SCRIPT = textwrap.dedent(
    """
    import numpy as np
    import nvflare.client as flare
    from nvflare.app_common.np.constants import NPConstants

    flare.init()
    while flare.is_running():
        model = flare.receive()
        if model is None:
            break
        arr = model.params[NPConstants.NUMPY_KEY]
        new = arr + 1
        flare.log("weight_mean", np.mean(new), flare.AnalyticsDataType.SCALAR)
        flare.send(
            flare.FLModel(
                params={NPConstants.NUMPY_KEY: new},
                params_type=flare.ParamsType.FULL,
                metrics={"weight_mean": float(np.mean(new))},
                current_round=model.current_round,
            )
        )
    """
)

_JOB_SCRIPT = textwrap.dedent(
    """
    import sys
    from nvflare.app_common.np.np_model_persistor import NPModelPersistor
    from nvflare.app_common.workflows.fedavg import FedAvg
    from nvflare.fuel.utils.constants import FrameworkType
    from nvflare.job_config.base_fed_job import BaseFedJob
    from nvflare.job_config.script_runner import ScriptRunner
    from nvflare.recipe.utils import extract_persistor_id

    n_clients, num_rounds, workdir, command = 2, 2, sys.argv[1], sys.argv[2]
    job = BaseFedJob(name="ext-numpy-e2e", min_clients=n_clients, key_metric="weight_mean")
    pid = extract_persistor_id(job.to_server(NPModelPersistor(model=[[1, 2, 3], [4, 5, 6]]), id="persistor"))
    job.to_server(FedAvg(num_clients=n_clients, num_rounds=num_rounds, persistor_id=pid, task_name="train"))
    job.to_clients(
        ScriptRunner(
            script="client.py", execution_mode="external_process", command=command, framework=FrameworkType.NUMPY,
            # The client script intentionally does not call flare.shutdown(). Give the
            # trainer-side Cell watcher a bounded opportunity to reap F3 naturally.
            shutdown_timeout=5.0,
        ),
        tasks=["train"],
    )
    job.simulator_run(workdir, n_clients=n_clients, threads=n_clients)
    """
)


_PT_CLIENT_SCRIPT = textwrap.dedent(
    """
    import torch
    import nvflare.client as flare

    flare.init()
    while flare.is_running():
        model = flare.receive()
        if model is None:
            break
        params = model.params
        # trainer-side Client API conversion must hand the unchanged script torch tensors,
        # exactly as the legacy Client API did — even though the wire/server stay numpy
        assert params and all(isinstance(v, torch.Tensor) for v in params.values()), (
            "expected torch tensors at flare.receive(): " + str({k: type(v).__name__ for k, v in params.items()})
        )
        new = {k: v + 0.1 for k, v in params.items()}
        flare.send(
            flare.FLModel(params=new, params_type=flare.ParamsType.FULL, current_round=model.current_round)
        )
    """
)

_PT_MODEL_SCRIPT = "import torch.nn as nn\n\n\nclass TinyNet(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(4, 2)\n"

_PT_JOB_SCRIPT = textwrap.dedent(
    """
    import sys
    import torch  # noqa
    from model import TinyNet
    from nvflare.app_opt.pt.job_config.model import PTModel
    from nvflare.app_common.workflows.fedavg import FedAvg
    from nvflare.job_config.base_fed_job import BaseFedJob
    from nvflare.job_config.script_runner import ScriptRunner
    from nvflare.recipe.utils import extract_persistor_id

    n_clients, num_rounds, workdir, command = 2, 2, sys.argv[1], sys.argv[2]
    job = BaseFedJob(name="ext-pt-e2e", min_clients=n_clients)
    pid = extract_persistor_id(job.to_server(PTModel(TinyNet()), id="persistor"))
    job.to_server(FedAvg(num_clients=n_clients, num_rounds=num_rounds, persistor_id=pid, task_name="train"))
    # framework defaults to PYTORCH; ScriptRunner declares numpy<->torch API-boundary conversion
    job.to_clients(ScriptRunner(script="client.py", execution_mode="external_process", command=command), tasks=["train"])
    job.simulator_run(workdir, n_clients=n_clients, threads=n_clients)
    """
)


_SK_CLIENT_SCRIPT = textwrap.dedent(
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    import nvflare.client as flare
    from nvflare.app_common.np.constants import NPConstants

    rng = np.random.RandomState(0)
    X = rng.randn(200, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    flare.init()
    while flare.is_running():
        model = flare.receive()
        if model is None:
            break
        global_coef = np.asarray(model.params[NPConstants.NUMPY_KEY], dtype=np.float64)
        clf = LogisticRegression(max_iter=100).fit(X, y)  # real sklearn training
        new = clf.coef_.reshape(global_coef.shape).astype(np.float32)
        flare.send(
            flare.FLModel(params={NPConstants.NUMPY_KEY: new}, params_type=flare.ParamsType.FULL,
                         current_round=model.current_round)
        )
    """
)

_SK_JOB_SCRIPT = textwrap.dedent(
    """
    import sys
    from nvflare.app_common.np.np_model_persistor import NPModelPersistor
    from nvflare.app_common.workflows.fedavg import FedAvg
    from nvflare.fuel.utils.constants import FrameworkType
    from nvflare.job_config.base_fed_job import BaseFedJob
    from nvflare.job_config.script_runner import ScriptRunner
    from nvflare.recipe.utils import extract_persistor_id

    n_clients, num_rounds, workdir, command = 2, 2, sys.argv[1], sys.argv[2]
    job = BaseFedJob(name="ext-sklearn-e2e", min_clients=n_clients)
    pid = extract_persistor_id(job.to_server(NPModelPersistor(model=[[0.0, 0.0, 0.0, 0.0]]), id="persistor"))
    job.to_server(FedAvg(num_clients=n_clients, num_rounds=num_rounds, persistor_id=pid, task_name="train"))
    job.to_clients(
        ScriptRunner(script="client.py", execution_mode="external_process", command=command,
                     framework=FrameworkType.NUMPY),
        tasks=["train"],
    )
    job.simulator_run(workdir, n_clients=n_clients, threads=n_clients)
    """
)


def _torch_available():
    try:
        import torch  # noqa

        return True
    except Exception:
        return False


def _sklearn_available():
    try:
        import sklearn  # noqa

        return True
    except Exception:
        return False


def _lightning_available():
    try:
        import pytorch_lightning  # noqa

        import nvflare.client.lightning  # noqa

        return True
    except Exception:
        return False


def _tf_available():
    try:
        import tensorflow  # noqa

        return True
    except Exception:
        return False


def _torchvision_available():
    try:
        import torchvision  # noqa

        return True
    except Exception:
        return False


def test_external_process_numpy_fedavg_end_to_end(tmp_path):
    jobdir = tmp_path / "job"
    jobdir.mkdir()
    (jobdir / "client.py").write_text(_CLIENT_SCRIPT)
    (jobdir / "run_job.py").write_text(_JOB_SCRIPT)
    workdir = tmp_path / "sim"

    env = os.environ.copy()
    # the trainer subprocess (launched by the backend) must import the nvflare under test
    env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    # trainer launch command: the same interpreter, importing the tree under test
    command = f"{sys.executable} -u"

    proc = subprocess.run(
        [sys.executable, "-u", str(jobdir / "run_job.py"), str(workdir), command],
        cwd=str(jobdir),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    out = proc.stdout + proc.stderr

    assert "Finished FedAvg." in out, f"job did not finish FedAvg cleanly:\n{out[-3000:]}"
    # both rounds ran, the server aggregated real client results, and the model persisted
    assert "Round 1 started." in out, f"second round did not start:\n{out[-3000:]}"
    assert "Aggregated 2/2 results" in out, f"server did not aggregate both clients:\n{out[-3000:]}"
    assert "Saved numpy model" in out, f"server did not persist the aggregated model:\n{out[-3000:]}"
    # the LOG/metrics path works (numpy scalar coerced) — no analytics failures
    assert "failed to process trainer LOG data" not in out, f"metric logging failed:\n{out[-3000:]}"
    # real trainer subprocesses were launched and reaped by the backend
    assert "launching external trainer" in out, f"no external trainer was launched:\n{out[-3000:]}"
    assert "terminating trainer process tree" not in out, (
        "trainer did not exit naturally within the orderly shutdown bound:\n" + out[-3000:]
    )


@pytest.mark.skipif(not _torch_available(), reason="requires torch for the PyTorch example")
def test_external_process_pytorch_fedavg_end_to_end(tmp_path):
    """hello-pt-style: an UNCHANGED torch-expecting client. ScriptRunner declares trainer-side
    numpy<->torch API conversion, so flare.receive() hands the script torch
    tensors even though the wire/server stay numpy — the client asserts it received torch."""
    jobdir = tmp_path / "job"
    jobdir.mkdir()
    (jobdir / "client.py").write_text(_PT_CLIENT_SCRIPT)
    (jobdir / "model.py").write_text(_PT_MODEL_SCRIPT)
    (jobdir / "run_job.py").write_text(_PT_JOB_SCRIPT)
    workdir = tmp_path / "sim"

    env = os.environ.copy()
    env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    command = f"{sys.executable} -u"

    proc = subprocess.run(
        [sys.executable, "-u", str(jobdir / "run_job.py"), str(workdir), command],
        cwd=str(jobdir),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    out = proc.stdout + proc.stderr

    assert "Finished FedAvg." in out, f"PT job did not finish FedAvg cleanly:\n{out[-3000:]}"
    assert "Round 1 started." in out, f"second round did not start:\n{out[-3000:]}"
    assert "Aggregated 2/2 results" in out, f"server did not aggregate both clients:\n{out[-3000:]}"
    # the torch-expecting client's assertion (torch tensors at receive) did NOT fire
    assert "expected torch tensors" not in out, f"client did not receive torch tensors:\n{out[-3000:]}"
    assert "AssertionError" not in out, f"trainer script assertion failed:\n{out[-3000:]}"


@pytest.mark.skipif(not _sklearn_available(), reason="requires scikit-learn for the sklearn example")
def test_external_process_sklearn_fedavg_end_to_end(tmp_path):
    """A real scikit-learn (LogisticRegression) FedAvg job — the ML (non-DL) example
    category — runs as real subprocesses on external_process (numpy params path)."""
    jobdir = tmp_path / "job"
    jobdir.mkdir()
    (jobdir / "client.py").write_text(_SK_CLIENT_SCRIPT)
    (jobdir / "run_job.py").write_text(_SK_JOB_SCRIPT)
    workdir = tmp_path / "sim"

    env = os.environ.copy()
    env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    command = f"{sys.executable} -u"

    proc = subprocess.run(
        [sys.executable, "-u", str(jobdir / "run_job.py"), str(workdir), command],
        cwd=str(jobdir),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    out = proc.stdout + proc.stderr
    assert "Finished FedAvg." in out, f"sklearn job did not finish FedAvg cleanly:\n{out[-3000:]}"
    assert "Aggregated 2/2 results" in out, f"server did not aggregate both clients:\n{out[-3000:]}"
    assert "Traceback" not in out, f"sklearn job raised:\n{out[-3000:]}"


# --- Config-based integration-test job (job directory + .conf, not the FedJob API) ---------------
#
# The tests above build jobs with the FedJob API. The existing integration-test suite instead
# ships job DIRECTORIES with HOCON configs and a legacy workflow (ScatterAndGather), historically
# run on ClientAPILauncherExecutor + SubprocessLauncher + FilePipe. This test proves that same
# config-based shape runs on the new stack by swapping only the client executor to
# ClientAPIExecutor(external_process) — the exact "integration test jobs should run on the new
# ClientAPIExecutor with ExternalProcessBackend" acceptance path. It is derived from the repo's
# np_loop job (tests/integration_test/data/jobs/np_loop): NPModelPersistor + ScatterAndGather on
# the server, the np_loop train_loop.py (+1 mock training) on the client.

_SAG_TRAIN_SCRIPT = textwrap.dedent(
    """
    import copy
    import nvflare.client as flare

    flare.init()
    print(f"system info is: {flare.system_info()}", flush=True)
    while flare.is_running():
        input_model = flare.receive()
        if input_model is None:
            break
        arr = input_model.params["numpy_key"]
        out = copy.deepcopy(arr) + 1  # mock training
        print(f"finish round: {input_model.current_round}", flush=True)
        flare.send(
            flare.FLModel(
                params={"numpy_key": out},
                params_type="FULL",
                metrics={"accuracy": 100},
                current_round=input_model.current_round,
            )
        )
    """
)

_SAG_SERVER_CONF = textwrap.dedent(
    """
    {
      format_version = 2
      task_data_filters = []
      task_result_filters = []
      workflows = [
        {
          id = "scatter_and_gather"
          path = "nvflare.app_common.workflows.scatter_and_gather.ScatterAndGather"
          args {
            min_clients = 2
            num_rounds = 2
            start_round = 0
            wait_time_after_min_received = 0
            aggregator_id = "aggregator"
            persistor_id = "persistor"
            shareable_generator_id = "shareable_generator"
            train_task_name = "train"
            train_timeout = 0
          }
        }
      ]
      components = [
        { id = "persistor", path = "nvflare.app_common.np.np_model_persistor.NPModelPersistor" }
        {
          id = "shareable_generator"
          path = "nvflare.app_common.shareablegenerators.full_model_shareable_generator.FullModelShareableGenerator"
          args {}
        }
        {
          id = "aggregator"
          path = "nvflare.app_common.aggregators.intime_accumulate_model_aggregator.InTimeAccumulateWeightedAggregator"
          args { expected_data_kind = "WEIGHTS" }
        }
      ]
    }
    """
)

# __PYTHON__ is replaced with the interpreter under test so the launched trainer imports this tree.
_SAG_CLIENT_CONF = textwrap.dedent(
    """
    {
      format_version = 2
      app_script = "train_loop.py"
      app_config = ""
      executors = [
        {
          tasks = ["train"]
          executor {
            path = "nvflare.app_common.executors.client_api_executor.ClientAPIExecutor"
            args {
              execution_mode = "external_process"
              command = "__PYTHON__ -u custom/train_loop.py"
              launch_once = true
              stop_grace_period = 5.0
              train_with_evaluation = true
            }
          }
        }
      ]
      task_data_filters = []
      task_result_filters = []
      components = []
    }
    """
)

_SAG_META_CONF = textwrap.dedent(
    """
    {
      name = "np_loop_external_process"
      resource_spec {}
      deploy_map { app_server = ["server"], app_client = ["site-1", "site-2"] }
      min_clients = 2
      mandatory_clients = []
    }
    """
)


def test_external_process_scatter_and_gather_config_job_end_to_end(tmp_path):
    """A config-based integration-test job (ScatterAndGather, derived from np_loop) runs on the
    new ClientAPIExecutor(external_process) via `nvflare simulator` — real trainer subprocesses,
    real Cell protocol, real F3 payload transfer, legacy HOCON job layout."""
    jobdir = tmp_path / "np_loop_external_process"
    (jobdir / "app_server" / "config").mkdir(parents=True)
    (jobdir / "app_client" / "config").mkdir(parents=True)
    (jobdir / "app_client" / "custom").mkdir(parents=True)
    (jobdir / "app_server" / "config" / "config_fed_server.conf").write_text(_SAG_SERVER_CONF)
    (jobdir / "app_client" / "config" / "config_fed_client.conf").write_text(
        _SAG_CLIENT_CONF.replace("__PYTHON__", sys.executable)
    )
    (jobdir / "app_client" / "custom" / "train_loop.py").write_text(_SAG_TRAIN_SCRIPT)
    (jobdir / "meta.conf").write_text(_SAG_META_CONF)
    workdir = tmp_path / "sim"

    env = os.environ.copy()
    # both the simulator process and the trainer subprocess it launches import the tree under test
    env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "nvflare.private.fed.app.simulator.simulator",
            str(jobdir),
            "-w",
            str(workdir),
            "-n",
            "2",
            "-t",
            "2",
            "-c",
            "site-1,site-2",
        ],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    out = proc.stdout + proc.stderr

    assert "Finished ScatterAndGather Training." in out, f"SAG config job did not finish:\n{out[-4000:]}"
    assert "Round 1 finished." in out, f"second round did not finish:\n{out[-4000:]}"
    # both clients' results were aggregated each round, and the model persisted
    assert "aggregating 2 update(s)" in out, f"server did not aggregate both clients:\n{out[-4000:]}"
    assert "Saved numpy model" in out, f"server did not persist the aggregated model:\n{out[-4000:]}"
    # a real external trainer subprocess was launched by the backend
    assert "launching external trainer" in out, f"no external trainer was launched:\n{out[-4000:]}"
    assert "Traceback" not in out, f"SAG config job raised:\n{out[-4000:]}"


# --- Metrics federation (flare.log / MLflowWriter -> backend -> server receiver) -----------------
#
# Derived from the np_metrics integration job. The trainer streams metrics with MLflowWriter
# (flare.log under the hood). On the new stack there is no MetricRelay/metrics_pipe/CellPipe: the
# engine's log() sends a LOG message over the Cell, the backend fires it as a federation-scoped
# analytics event (fed.analytix_log_stats), and a server-side AnalyticsReceiver records it. This
# test proves that whole path with a tiny custom receiver that counts arrivals (no tensorboard /
# mlflow receiver dependency), so it can ASSERT the metrics actually reached the server.

_METRICS_TRAIN_SCRIPT = textwrap.dedent(
    """
    import copy
    import nvflare.client as flare
    from nvflare.client.tracking import MLflowWriter

    flare.init()
    writer = MLflowWriter()
    while flare.is_running():
        input_model = flare.receive()
        if input_model is None:
            break
        arr = input_model.params["numpy_key"]
        cr = input_model.current_round
        for step in range(5):  # a handful of metric records per round (trimmed from np_metrics)
            writer.log_metric(key="global_step", value=cr * 5 + step, step=cr * 5 + step)
        out = copy.deepcopy(arr) + 1
        flare.send(
            flare.FLModel(
                params={"numpy_key": out}, params_type="FULL",
                metrics={"accuracy": 100}, current_round=cr,
            )
        )
    """
)

_METRICS_RECEIVER = textwrap.dedent(
    """
    from nvflare.apis.analytix import AnalyticsData
    from nvflare.apis.dxo import from_shareable
    from nvflare.app_common.widgets.streaming import AnalyticsReceiver


    class CountingAnalyticsReceiver(AnalyticsReceiver):
        def __init__(self, events=None):
            super().__init__(events=events or ["fed.analytix_log_stats"])
            self.count = 0

        def initialize(self, fl_ctx):
            pass

        def save(self, fl_ctx, shareable, record_origin):
            dxo = from_shareable(shareable)
            ad = AnalyticsData.from_dxo(dxo)
            if not ad:
                return
            self.count += 1
            print(f"RECV_METRIC origin={record_origin} key={ad.tag} value={ad.value}", flush=True)

        def finalize(self, fl_ctx):
            print(f"RECV_METRIC_FINAL total={self.count}", flush=True)
    """
)

_METRICS_SERVER_CONF = textwrap.dedent(
    """
    {
      format_version = 2
      task_data_filters = []
      task_result_filters = []
      workflows = [
        {
          id = "scatter_and_gather"
          path = "nvflare.app_common.workflows.scatter_and_gather.ScatterAndGather"
          args {
            min_clients = 2
            num_rounds = 2
            start_round = 0
            wait_time_after_min_received = 0
            aggregator_id = "aggregator"
            persistor_id = "persistor"
            shareable_generator_id = "shareable_generator"
            train_task_name = "train"
            train_timeout = 0
          }
        }
      ]
      components = [
        { id = "persistor", path = "nvflare.app_common.np.np_model_persistor.NPModelPersistor" }
        {
          id = "shareable_generator"
          path = "nvflare.app_common.shareablegenerators.full_model_shareable_generator.FullModelShareableGenerator"
          args {}
        }
        {
          id = "aggregator"
          path = "nvflare.app_common.aggregators.intime_accumulate_model_aggregator.InTimeAccumulateWeightedAggregator"
          args { expected_data_kind = "WEIGHTS" }
        }
        {
          id = "analytics_receiver"
          path = "count_receiver.CountingAnalyticsReceiver"
          args { events = ["fed.analytix_log_stats"] }
        }
      ]
    }
    """
)

_METRICS_CLIENT_CONF = textwrap.dedent(
    """
    {
      format_version = 2
      app_script = "train_metrics.py"
      app_config = ""
      executors = [
        {
          tasks = ["train"]
          executor {
            path = "nvflare.app_common.executors.client_api_executor.ClientAPIExecutor"
            args {
              execution_mode = "external_process"
              command = "__PYTHON__ -u custom/train_metrics.py"
              launch_once = true
              stop_grace_period = 5.0
              train_with_evaluation = true
            }
          }
        }
      ]
      task_data_filters = []
      task_result_filters = []
      components = []
    }
    """
)

_METRICS_META_CONF = textwrap.dedent(
    """
    {
      name = "np_metrics_external_process"
      resource_spec {}
      deploy_map { app = ["@ALL"] }
      min_clients = 2
      mandatory_clients = []
    }
    """
)


def test_external_process_metrics_streaming_config_job_end_to_end(tmp_path):
    """A config-based job (derived from np_metrics) proves the metrics federation path on the new
    stack: MLflowWriter in the trainer subprocess -> engine log() over the Cell -> backend fires
    fed.analytix_log_stats -> a server-side AnalyticsReceiver records it. No MetricRelay/CellPipe."""
    jobdir = tmp_path / "np_metrics_external_process"
    (jobdir / "app" / "config").mkdir(parents=True)
    (jobdir / "app" / "custom").mkdir(parents=True)
    (jobdir / "app" / "config" / "config_fed_server.conf").write_text(_METRICS_SERVER_CONF)
    (jobdir / "app" / "config" / "config_fed_client.conf").write_text(
        _METRICS_CLIENT_CONF.replace("__PYTHON__", sys.executable)
    )
    (jobdir / "app" / "custom" / "train_metrics.py").write_text(_METRICS_TRAIN_SCRIPT)
    (jobdir / "app" / "custom" / "count_receiver.py").write_text(_METRICS_RECEIVER)
    (jobdir / "meta.conf").write_text(_METRICS_META_CONF)
    workdir = tmp_path / "sim"

    env = os.environ.copy()
    env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "nvflare.private.fed.app.simulator.simulator",
            str(jobdir),
            "-w",
            str(workdir),
            "-n",
            "2",
            "-t",
            "2",
            "-c",
            "site-1,site-2",
        ],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    out = proc.stdout + proc.stderr

    assert "Finished ScatterAndGather Training." in out, f"metrics job did not finish:\n{out[-4000:]}"
    # the trainer's MLflowWriter metrics reached the server-side receiver from BOTH clients
    assert "RECV_METRIC origin=site-1" in out, f"no metrics received from site-1:\n{out[-4000:]}"
    assert "RECV_METRIC origin=site-2" in out, f"no metrics received from site-2:\n{out[-4000:]}"
    # the LOG path did not error (numpy/enum serialization, rank gate)
    assert "failed to process trainer LOG" not in out, f"LOG path errored:\n{out[-4000:]}"
    assert "only rank 0 can call log" not in out, f"rank gate misfired on log:\n{out[-4000:]}"
    assert "Traceback" not in out, f"metrics config job raised:\n{out[-4000:]}"


# --- Config-based PyTorch job, torch-throughout RAW representation -------------------------------
#
# Derived from the pt_client_api_basic integration job (params_exchange_format="pytorch"). Unlike
# the FedJob-API PyTorch test above (numpy wire + trainer-side conversion), this keeps torch
# tensors on the wire end-to-end: PTFileModelPersistor registers the tensor decomposer server-side,
# the backend and trainer engine register it too, so the FLModel round-trips as torch with the
# Client API passing params through unconverted. Also exercises launch_once=false: a single-shot
# trainer script relaunched per task (per-launch token/FQCN rotation) across two rounds.

_PT_CONFIG_NET = "import torch.nn as nn\n\n\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc1 = nn.Linear(10, 5)\n\n    def forward(self, x):\n        return self.fc1(x)\n"  # noqa: E501

_PT_CONFIG_CLIENT_SCRIPT = textwrap.dedent(
    """
    import torch
    import nvflare.client as flare

    flare.init()
    input_model = flare.receive()
    cr = input_model.current_round
    site = input_model.meta.get("site_name", "site-0")
    mult = int("".join(ch for ch in site if ch.isdigit()) or "1")
    if cr == 0:
        weight = torch.zeros([5, 10], dtype=torch.float32)
        bias = torch.zeros([5], dtype=torch.float32)
    else:
        # round > 0: the server's aggregated model arrives as torch tensors (RAW representation,
        # no conversion) — a broken decomposer/pass-through path would not yield torch here
        weight = input_model.params.get("fc1.weight")
        bias = input_model.params.get("fc1.bias")
        assert isinstance(weight, torch.Tensor) and isinstance(bias, torch.Tensor), (
            "PT_PASSTHROUGH_BROKEN: got " + str({k: type(v).__name__ for k, v in input_model.params.items()})
        )
        print("PT_PASSTHROUGH_OK", flush=True)
    weight = torch.add(weight, 1) * mult
    bias = torch.add(bias, 1) * mult
    flare.send(
        flare.FLModel(
            params={"fc1.weight": weight, "fc1.bias": bias},
            metrics={"accuracy": 0.5},
            meta={"NUM_STEPS_CURRENT_ROUND": 2},
        )
    )
    """
)

_PT_CONFIG_SERVER_CONF = textwrap.dedent(
    """
    {
      format_version = 2
      task_data_filters = []
      task_result_filters = []
      model_class_path = "net.Net"
      workflows = [
        {
          id = "scatter_and_gather"
          path = "nvflare.app_common.workflows.scatter_and_gather.ScatterAndGather"
          args {
            min_clients = 2
            num_rounds = 2
            start_round = 0
            wait_time_after_min_received = 0
            aggregator_id = "aggregator"
            persistor_id = "persistor"
            shareable_generator_id = "shareable_generator"
            train_task_name = "train"
            train_timeout = 0
          }
        }
      ]
      components = [
        {
          id = "persistor"
          path = "nvflare.app_opt.pt.file_model_persistor.PTFileModelPersistor"
          args { model { path = "{model_class_path}" } }
        }
        {
          id = "shareable_generator"
          path = "nvflare.app_common.shareablegenerators.full_model_shareable_generator.FullModelShareableGenerator"
          args {}
        }
        {
          id = "aggregator"
          path = "nvflare.app_common.aggregators.intime_accumulate_model_aggregator.InTimeAccumulateWeightedAggregator"
          args { expected_data_kind = "WEIGHTS" }
        }
        {
          id = "model_selector"
          path = "nvflare.app_common.widgets.intime_model_selector.IntimeModelSelector"
          args { key_metric = "accuracy" }
        }
      ]
    }
    """
)

_PT_CONFIG_CLIENT_CONF = textwrap.dedent(
    """
    {
      format_version = 2
      app_script = "poc_client.py"
      app_config = ""
      executors = [
        {
          tasks = ["train"]
          executor {
            path = "nvflare.app_common.executors.client_api_executor.ClientAPIExecutor"
            args {
              execution_mode = "external_process"
              command = "__PYTHON__ -u custom/poc_client.py"
              launch_once = false
              stop_grace_period = 5.0
              train_with_evaluation = true
            }
          }
        }
      ]
      task_data_filters = []
      task_result_filters = []
      components = []
    }
    """
)

_PT_CONFIG_META_CONF = textwrap.dedent(
    """
    {
      name = "pt_client_api_basic_external_process"
      resource_spec {}
      deploy_map { app = ["@ALL"] }
      min_clients = 2
      mandatory_clients = []
    }
    """
)


@pytest.mark.skipif(not _torch_available(), reason="requires torch for the config-based PyTorch example")
def test_external_process_pytorch_config_job_passthrough_end_to_end(tmp_path):
    """A config-based PyTorch job (derived from pt_client_api_basic) with torch tensors on the wire
    end-to-end (RAW representation, no conversion): PTFileModelPersistor + backend + engine register the
    tensor decomposer, so the FLModel round-trips as torch. Also drives launch_once=false (per-task
    relaunch across two rounds). The trainer asserts it received torch tensors at round > 0."""
    jobdir = tmp_path / "pt_basic_external_process"
    (jobdir / "app" / "config").mkdir(parents=True)
    (jobdir / "app" / "custom").mkdir(parents=True)
    (jobdir / "app" / "config" / "config_fed_server.conf").write_text(_PT_CONFIG_SERVER_CONF)
    (jobdir / "app" / "config" / "config_fed_client.conf").write_text(
        _PT_CONFIG_CLIENT_CONF.replace("__PYTHON__", sys.executable)
    )
    (jobdir / "app" / "custom" / "poc_client.py").write_text(_PT_CONFIG_CLIENT_SCRIPT)
    (jobdir / "app" / "custom" / "net.py").write_text(_PT_CONFIG_NET)
    (jobdir / "meta.conf").write_text(_PT_CONFIG_META_CONF)
    workdir = tmp_path / "sim"

    env = os.environ.copy()
    env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "nvflare.private.fed.app.simulator.simulator",
            str(jobdir),
            "-w",
            str(workdir),
            "-n",
            "2",
            "-t",
            "2",
            "-c",
            "site-1,site-2",
        ],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    out = proc.stdout + proc.stderr

    assert "Finished ScatterAndGather Training." in out, f"PT config job did not finish:\n{out[-4000:]}"
    assert "aggregating 2 update(s) at round 1" in out, f"second round did not aggregate both clients:\n{out[-4000:]}"
    # torch tensors survived the RAW round-trip on the wire
    assert "PT_PASSTHROUGH_OK" in out, f"trainer did not receive torch tensors at round>0:\n{out[-4000:]}"
    assert "PT_PASSTHROUGH_BROKEN" not in out, f"pass-through delivered non-torch params:\n{out[-4000:]}"
    # launch_once=false relaunched the trainer per task
    assert "launching external trainer" in out, f"no external trainer was launched:\n{out[-4000:]}"
    assert "trainer SHUTDOWN was not acknowledged" not in out, f"per-task trainer did not stop cleanly:\n{out[-4000:]}"
    assert "terminating trainer process tree" not in out, f"per-task trainer required forced reaping:\n{out[-4000:]}"
    assert "Traceback" not in out, f"PT config job raised:\n{out[-4000:]}"


# --- Trainer-side DIFF calculation ---------------------------------------------------------------
#
# The original pt_client_api_basic job declares params_transfer_type=DIFF while its unchanged
# trainer returns a FULL PyTorch model.  The Client API boundary must subtract the received model,
# convert the resulting diff to the server representation, and put WEIGHT_DIFF on the wire.  A
# unit test of _prepare_param_diff cannot prove that last part, so this E2E installs an asserting
# aggregator at the server boundary.

_DIFF_ASSERTING_AGGREGATOR = textwrap.dedent(
    """
    from nvflare.apis.dxo import DataKind, from_shareable
    from nvflare.app_common.aggregators.intime_accumulate_model_aggregator import (
        InTimeAccumulateWeightedAggregator,
    )


    class DiffAssertingAggregator(InTimeAccumulateWeightedAggregator):
        def accept(self, shareable, fl_ctx):
            dxo = from_shareable(shareable)
            assert dxo.data_kind == DataKind.WEIGHT_DIFF, (
                "SERVER_EXPECTED_WEIGHT_DIFF: got " + str(dxo.data_kind)
            )
            print("SERVER_RECEIVED_DIFF", flush=True)
            return super().accept(shareable, fl_ctx)
    """
)

_DIFF_SERVER_CONF = _PT_CONFIG_SERVER_CONF.replace(
    "nvflare.app_common.aggregators.intime_accumulate_model_aggregator.InTimeAccumulateWeightedAggregator",
    "diff_aggregator.DiffAssertingAggregator",
).replace('expected_data_kind = "WEIGHTS"', 'expected_data_kind = "WEIGHT_DIFF"')

_DIFF_CLIENT_CONF = _PT_CONFIG_CLIENT_CONF.replace(
    "train_with_evaluation = true",
    (
        "train_with_evaluation = true\n"
        '              params_transfer_type = "DIFF"\n'
        '              params_exchange_format = "pytorch"\n'
        '              server_expected_format = "numpy"'
    ),
)

_DIFF_META_CONF = _PT_CONFIG_META_CONF.replace(
    "pt_client_api_basic_external_process", "pt_client_api_diff_external_process"
)


@pytest.mark.skipif(not _torch_available(), reason="requires torch for the config-based PyTorch DIFF example")
def test_external_process_pytorch_diff_reaches_server_end_to_end(tmp_path):
    """An unchanged PT trainer sends FULL, but ClientAPIExecutor declares DIFF.  Assert at the
    actual server aggregator boundary that both client results arrive as WEIGHT_DIFF."""
    jobdir = tmp_path / "pt_diff_external_process"
    (jobdir / "app" / "config").mkdir(parents=True)
    (jobdir / "app" / "custom").mkdir(parents=True)
    (jobdir / "app" / "config" / "config_fed_server.conf").write_text(_DIFF_SERVER_CONF)
    (jobdir / "app" / "config" / "config_fed_client.conf").write_text(
        _DIFF_CLIENT_CONF.replace("__PYTHON__", sys.executable)
    )
    (jobdir / "app" / "custom" / "poc_client.py").write_text(_PT_CONFIG_CLIENT_SCRIPT)
    (jobdir / "app" / "custom" / "net.py").write_text(_PT_CONFIG_NET)
    (jobdir / "app" / "custom" / "diff_aggregator.py").write_text(_DIFF_ASSERTING_AGGREGATOR)
    (jobdir / "meta.conf").write_text(_DIFF_META_CONF)
    workdir = tmp_path / "sim"

    env = os.environ.copy()
    env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "nvflare.private.fed.app.simulator.simulator",
            str(jobdir),
            "-w",
            str(workdir),
            "-n",
            "2",
            "-t",
            "2",
            "-c",
            "site-1,site-2",
        ],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    out = proc.stdout + proc.stderr

    assert "Finished ScatterAndGather Training." in out, f"PT DIFF job did not finish:\n{out[-4000:]}"
    # This marker is emitted only after from_shareable() at the server boundary verifies
    # DataKind.WEIGHT_DIFF.  Both clients must contribute in both rounds.
    assert out.count("SERVER_RECEIVED_DIFF") >= 4, f"server did not receive four DIFF results:\n{out[-4000:]}"
    assert "SERVER_EXPECTED_WEIGHT_DIFF" not in out, f"server received a FULL result:\n{out[-4000:]}"
    assert "aggregating 2 update(s) at round 1" in out, f"server did not aggregate round 1:\n{out[-4000:]}"
    assert "Traceback" not in out, f"PT DIFF job raised:\n{out[-4000:]}"


# --- Secret reference resolution in a real trainer launch ---------------------------------------
#
# Unit tests cover command tokenization.  This case proves the complete boundary: ScriptRunner
# stores the reference, the executing CJ resolves it immediately before shell-free Popen, a value
# with spaces/metacharacters remains one argv item, and the resolved value is not logged.

_SECRET_ENV_NAME = "NVFLARE_EXTERNAL_E2E_LAUNCH_SECRET"
_SECRET_REF = "${secret:" + _SECRET_ENV_NAME + "}"

_SECRET_CLIENT_SCRIPT = textwrap.dedent(
    """
    import argparse
    import os

    import nvflare.client as flare

    parser = argparse.ArgumentParser()
    parser.add_argument("--launch-secret", required=True)
    args = parser.parse_args()
    assert args.launch_secret == os.environ["NVFLARE_EXTERNAL_E2E_LAUNCH_SECRET"], "LAUNCH_SECRET_MISMATCH"
    print("LAUNCH_SECRET_ARG_OK", flush=True)

    flare.init()
    while flare.is_running():
        model = flare.receive()
        if model is None:
            break
        flare.send(
            flare.FLModel(
                params=model.params,
                params_type=flare.ParamsType.FULL,
                current_round=model.current_round,
            )
        )
    """
)

_SECRET_JOB_SCRIPT = textwrap.dedent(
    """
    import sys
    from nvflare.app_common.np.np_model_persistor import NPModelPersistor
    from nvflare.app_common.workflows.fedavg import FedAvg
    from nvflare.fuel.utils.constants import FrameworkType
    from nvflare.job_config.base_fed_job import BaseFedJob
    from nvflare.job_config.script_runner import ScriptRunner
    from nvflare.recipe.utils import extract_persistor_id

    workdir, command, secret_ref = sys.argv[1], sys.argv[2], sys.argv[3]
    job = BaseFedJob(name="external-secret-command-e2e", min_clients=1)
    pid = extract_persistor_id(job.to_server(NPModelPersistor(model=[[1.0]]), id="persistor"))
    job.to_server(FedAvg(num_clients=1, num_rounds=1, persistor_id=pid, task_name="train"))
    job.to_clients(
        ScriptRunner(
            script="client.py",
            script_args=f"--launch-secret {secret_ref}",
            execution_mode="external_process",
            command=command,
            framework=FrameworkType.NUMPY,
            shutdown_timeout=5.0,
        ),
        tasks=["train"],
    )
    job.simulator_run(workdir, n_clients=1, threads=1)
    """
)


def test_external_process_secret_command_launch_end_to_end(tmp_path):
    """A secret reference is resolved at the real CJ->trainer launch boundary, remains one argv
    item despite spaces and shell metacharacters, and is never written to captured logs."""
    jobdir = tmp_path / "job"
    jobdir.mkdir()
    (jobdir / "client.py").write_text(_SECRET_CLIENT_SCRIPT)
    (jobdir / "run_job.py").write_text(_SECRET_JOB_SCRIPT)
    workdir = tmp_path / "sim"

    # If the backend accidentally reparses this through a shell, the semicolon/dollar syntax
    # changes behavior.  With shell=False it is a single opaque argv value.
    secret_value = "synthetic value; $NOT_EXPANDED $(not-executed)"
    env = os.environ.copy()
    env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env[_SECRET_ENV_NAME] = secret_value
    command = f"{sys.executable} -u"

    proc = subprocess.run(
        [sys.executable, "-u", str(jobdir / "run_job.py"), str(workdir), command, _SECRET_REF],
        cwd=str(jobdir),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    out = proc.stdout + proc.stderr

    assert "Finished FedAvg." in out, f"secret-command job did not finish:\n{out[-4000:]}"
    assert "LAUNCH_SECRET_ARG_OK" in out, f"trainer did not receive the resolved secret argv:\n{out[-4000:]}"
    assert "LAUNCH_SECRET_MISMATCH" not in out, f"secret argv was changed during launch:\n{out[-4000:]}"
    assert secret_value not in out, "resolved launch secret leaked into simulator/trainer logs"
    assert "Traceback" not in out, f"secret-command job raised:\n{out[-4000:]}"


# --- Cyclic (relay) workflow ---------------------------------------------------------------------
#
# All the tests above drive the backend under broadcast controllers (FedAvg / ScatterAndGather).
# The pt_client_api_cyclic integration job instead uses CyclicController: the model is RELAYED
# client-to-client sequentially (site-1 -> site-2 -> ... within a round), not broadcast. This
# proves the backend/engine work under relay-style task dispatch too. Data-free numpy variant.

_CYCLIC_TRAIN_SCRIPT = textwrap.dedent(
    """
    import copy
    import nvflare.client as flare

    flare.init()
    while flare.is_running():
        input_model = flare.receive()
        if input_model is None:
            break
        arr = input_model.params["numpy_key"]
        out = copy.deepcopy(arr) + 1  # mock training
        print(f"CYCLIC_ROUND {input_model.current_round}", flush=True)
        flare.send(
            flare.FLModel(
                params={"numpy_key": out}, params_type="FULL",
                metrics={"accuracy": 100}, current_round=input_model.current_round,
            )
        )
    """
)

_CYCLIC_SERVER_CONF = textwrap.dedent(
    """
    {
      format_version = 2
      task_data_filters = []
      task_result_filters = []
      workflows = [
        {
          id = "cyclic"
          path = "nvflare.app_common.workflows.cyclic_ctl.CyclicController"
          args {
            num_rounds = 2
            task_assignment_timeout = 10
            persistor_id = "persistor"
            shareable_generator_id = "shareable_generator"
            task_name = "train"
          }
        }
      ]
      components = [
        { id = "persistor", path = "nvflare.app_common.np.np_model_persistor.NPModelPersistor" }
        {
          id = "shareable_generator"
          path = "nvflare.app_common.shareablegenerators.full_model_shareable_generator.FullModelShareableGenerator"
          args {}
        }
      ]
    }
    """
)

_CYCLIC_CLIENT_CONF = textwrap.dedent(
    """
    {
      format_version = 2
      app_script = "train_loop.py"
      app_config = ""
      executors = [
        {
          tasks = ["train"]
          executor {
            path = "nvflare.app_common.executors.client_api_executor.ClientAPIExecutor"
            args {
              execution_mode = "external_process"
              command = "__PYTHON__ -u custom/train_loop.py"
              launch_once = true
              stop_grace_period = 5.0
              train_with_evaluation = true
            }
          }
        }
      ]
      task_data_filters = []
      task_result_filters = []
      components = []
    }
    """
)

_CYCLIC_META_CONF = textwrap.dedent(
    """
    {
      name = "np_cyclic_external_process"
      resource_spec {}
      deploy_map { app = ["@ALL"] }
      min_clients = 2
      mandatory_clients = []
    }
    """
)


def test_external_process_cyclic_relay_config_job_end_to_end(tmp_path):
    """A config-based job with CyclicController (relay) — the model is passed client-to-client
    sequentially, not broadcast — runs on ClientAPIExecutor(external_process). Proves the backend
    works under relay-style task dispatch (derived from the pt_client_api_cyclic integration job)."""
    jobdir = tmp_path / "np_cyclic_external_process"
    (jobdir / "app" / "config").mkdir(parents=True)
    (jobdir / "app" / "custom").mkdir(parents=True)
    (jobdir / "app" / "config" / "config_fed_server.conf").write_text(_CYCLIC_SERVER_CONF)
    (jobdir / "app" / "config" / "config_fed_client.conf").write_text(
        _CYCLIC_CLIENT_CONF.replace("__PYTHON__", sys.executable)
    )
    (jobdir / "app" / "custom" / "train_loop.py").write_text(_CYCLIC_TRAIN_SCRIPT)
    (jobdir / "meta.conf").write_text(_CYCLIC_META_CONF)
    workdir = tmp_path / "sim"

    env = os.environ.copy()
    env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "nvflare.private.fed.app.simulator.simulator",
            str(jobdir),
            "-w",
            str(workdir),
            "-n",
            "2",
            "-t",
            "2",
            "-c",
            "site-1,site-2",
        ],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    out = proc.stdout + proc.stderr

    # both cyclic rounds ran; the relay reached BOTH clients and the server persisted each round
    assert "CYCLIC_ROUND 0" in out, f"first cyclic round did not run on a client:\n{out[-4000:]}"
    assert "CYCLIC_ROUND 1" in out, f"second cyclic round did not run on a client:\n{out[-4000:]}"
    assert "sending TASK_READY for 'train' to trainer site-1" in out, f"relay never reached site-1:\n{out[-4000:]}"
    assert "sending TASK_READY for 'train' to trainer site-2" in out, f"relay never reached site-2:\n{out[-4000:]}"
    assert "Saved numpy model" in out, f"server did not persist the relayed model:\n{out[-4000:]}"
    assert "Traceback" not in out, f"cyclic config job raised:\n{out[-4000:]}"


# --- Launcher-wrapped (grandchild) trainer topology ----------------------------------------------
#
# In the single-trainer tests the CJ launches the trainer script directly. Multi-worker launchers
# (torchrun, and any custom launcher) instead insert a launcher process between the CJ and the real
# trainer: CJ -> launcher -> trainer(s), so the trainer is a GRANDCHILD of the CJ. This test proves
# the full stack — including the F3 payload pull, not just control messages — works when the trainer
# is a grandchild reached through a plain subprocess wrapper. (It is the topology torchrun needs;
# torchrun itself is covered by the rank-contract unit tests because its gloo rendezvous hangs on
# macOS, but the grandchild payload path it depends on is exercised here rendezvous-free.)

_GRANDCHILD_CLIENT_SCRIPT = textwrap.dedent(
    """
    import copy
    import nvflare.client as flare

    flare.init()
    print("GRANDCHILD_CLIENT_START", flush=True)
    while flare.is_running():
        m = flare.receive()
        if m is None:
            break
        print("GRANDCHILD_RECEIVED_OK", flush=True)
        out = copy.deepcopy(m.params["numpy_key"]) + 1
        flare.send(
            flare.FLModel(params={"numpy_key": out}, params_type="FULL",
                         metrics={"accuracy": 1.0}, current_round=m.current_round)
        )
        print("GRANDCHILD_SENT_OK", flush=True)
    """
)

# a launcher process that execs the real trainer as its own child (CJ -> wrapper -> client.py)
_GRANDCHILD_WRAPPER_SCRIPT = textwrap.dedent(
    """
    import os
    import subprocess
    import sys

    here = os.path.dirname(os.path.abspath(__file__))
    sys.exit(subprocess.run([sys.executable, "-u", os.path.join(here, "client.py")], env=os.environ.copy()).returncode)
    """
)

_GRANDCHILD_SERVER_CONF = textwrap.dedent(
    """
    {
      format_version = 2
      task_data_filters = []
      task_result_filters = []
      workflows = [
        {
          id = "scatter_and_gather"
          path = "nvflare.app_common.workflows.scatter_and_gather.ScatterAndGather"
          args {
            min_clients = 2
            num_rounds = 1
            start_round = 0
            wait_time_after_min_received = 0
            aggregator_id = "aggregator"
            persistor_id = "persistor"
            shareable_generator_id = "shareable_generator"
            train_task_name = "train"
            train_timeout = 0
          }
        }
      ]
      components = [
        { id = "persistor", path = "nvflare.app_common.np.np_model_persistor.NPModelPersistor" }
        {
          id = "shareable_generator"
          path = "nvflare.app_common.shareablegenerators.full_model_shareable_generator.FullModelShareableGenerator"
          args {}
        }
        {
          id = "aggregator"
          path = "nvflare.app_common.aggregators.intime_accumulate_model_aggregator.InTimeAccumulateWeightedAggregator"
          args { expected_data_kind = "WEIGHTS" }
        }
      ]
    }
    """
)

_GRANDCHILD_CLIENT_CONF = textwrap.dedent(
    """
    {
      format_version = 2
      app_script = "wrapper.py"
      app_config = ""
      executors = [
        {
          tasks = ["train"]
          executor {
            path = "nvflare.app_common.executors.client_api_executor.ClientAPIExecutor"
            args {
              execution_mode = "external_process"
              command = "__PYTHON__ -u custom/wrapper.py"
              launch_once = true
              stop_grace_period = 5.0
              train_with_evaluation = true
            }
          }
        }
      ]
      task_data_filters = []
      task_result_filters = []
      components = []
    }
    """
)

_GRANDCHILD_META_CONF = textwrap.dedent(
    """
    {
      name = "grandchild_external_process"
      resource_spec {}
      deploy_map { app = ["@ALL"] }
      min_clients = 2
      mandatory_clients = []
    }
    """
)


def test_external_process_launcher_wrapped_grandchild_trainer_end_to_end(tmp_path):
    """The trainer is a GRANDCHILD of the CJ (CJ -> subprocess wrapper -> trainer), the topology a
    multi-worker launcher such as torchrun produces. Proves the whole stack — including the F3
    payload pull, not just control messages — works when the trainer is not a direct child."""
    jobdir = tmp_path / "grandchild_external_process"
    (jobdir / "app" / "config").mkdir(parents=True)
    (jobdir / "app" / "custom").mkdir(parents=True)
    (jobdir / "app" / "config" / "config_fed_server.conf").write_text(_GRANDCHILD_SERVER_CONF)
    (jobdir / "app" / "config" / "config_fed_client.conf").write_text(
        _GRANDCHILD_CLIENT_CONF.replace("__PYTHON__", sys.executable)
    )
    (jobdir / "app" / "custom" / "client.py").write_text(_GRANDCHILD_CLIENT_SCRIPT)
    (jobdir / "app" / "custom" / "wrapper.py").write_text(_GRANDCHILD_WRAPPER_SCRIPT)
    (jobdir / "meta.conf").write_text(_GRANDCHILD_META_CONF)
    workdir = tmp_path / "sim"

    env = os.environ.copy()
    env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "nvflare.private.fed.app.simulator.simulator",
            str(jobdir),
            "-w",
            str(workdir),
            "-n",
            "2",
            "-t",
            "2",
            "-c",
            "site-1,site-2",
        ],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    out = proc.stdout + proc.stderr

    assert "Finished ScatterAndGather Training." in out, f"grandchild job did not finish:\n{out[-4000:]}"
    # the grandchild trainer completed the full receive/send round-trip (F3 payload pull included)
    assert "GRANDCHILD_RECEIVED_OK" in out, f"grandchild did not pull the task payload:\n{out[-4000:]}"
    assert "GRANDCHILD_SENT_OK" in out, f"grandchild did not send its result:\n{out[-4000:]}"
    assert "aggregating 2 update(s)" in out, f"server did not aggregate both grandchild trainers:\n{out[-4000:]}"
    assert "Traceback" not in out, f"grandchild job raised:\n{out[-4000:]}"


# --- Large model: the chunked large-payload streaming path ---------------------------------------
#
# Derived from the pt_large_model_pass_through integration job. LargeNet (~8 MB of float32 params)
# is deliberately sized ABOVE the 2 MB ViaDownloaderDecomposer streaming threshold, so the model
# round-trips via the F3 chunked download path — the path all the small-model tests above stay
# below and never exercise. ScriptRunner declares trainer-side representation conversion: the wire
# carries large NUMPY arrays (streamed), and the Client API converts them to torch before returning
# from flare.receive(), so an unchanged torch script's net.load_state_dict() works.

_LARGE_MODEL_SCRIPT = textwrap.dedent(
    """
    import torch.nn as nn
    import torch.nn.functional as F


    class LargeNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(1024, 1024)  # ~4 MB, above the 2 MB streaming threshold
            self.fc2 = nn.Linear(1024, 1024)
            self.fc3 = nn.Linear(1024, 10)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)
    """
)

_LARGE_CLIENT_SCRIPT = textwrap.dedent(
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from model import LargeNet
    import nvflare.client as flare

    net = LargeNet()
    flare.init()
    while flare.is_running():
        m = flare.receive()
        if m is None:
            break
        # trainer-side conversion must deliver torch tensors even for the large (streamed) model
        assert all(isinstance(v, torch.Tensor) for v in m.params.values()), (
            "LARGE_NOT_TORCH:" + str({k: type(v).__name__ for k, v in m.params.items()})
        )
        net.load_state_dict(m.params)
        opt = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        x = torch.randn(4, 1024)
        y = torch.randint(0, 10, (4,))
        opt.zero_grad()
        nn.CrossEntropyLoss()(net(x), y).backward()
        opt.step()
        print(f"LARGE_ROUND {m.current_round} ok", flush=True)
        flare.send(
            flare.FLModel(params=net.cpu().state_dict(), params_type=flare.ParamsType.FULL,
                         current_round=m.current_round)
        )
    """
)

_LARGE_JOB_SCRIPT = textwrap.dedent(
    """
    import sys
    import torch  # noqa
    from model import LargeNet
    from nvflare.app_opt.pt.job_config.model import PTModel
    from nvflare.app_common.workflows.fedavg import FedAvg
    from nvflare.job_config.base_fed_job import BaseFedJob
    from nvflare.job_config.script_runner import ScriptRunner
    from nvflare.recipe.utils import extract_persistor_id

    n_clients, num_rounds, workdir, command = 2, 2, sys.argv[1], sys.argv[2]
    job = BaseFedJob(name="ext-large-e2e", min_clients=n_clients)
    pid = extract_persistor_id(job.to_server(PTModel(LargeNet()), id="persistor"))
    job.to_server(FedAvg(num_clients=n_clients, num_rounds=num_rounds, persistor_id=pid, task_name="train"))
    job.to_clients(ScriptRunner(script="client.py", execution_mode="external_process", command=command), tasks=["train"])
    job.simulator_run(workdir, n_clients=n_clients, threads=n_clients)
    """
)


@pytest.mark.skipif(not _torch_available(), reason="requires torch for the large-model example")
def test_external_process_large_model_streaming_end_to_end(tmp_path):
    """A ~8 MB model (above the 2 MB streaming threshold) round-trips via the F3 chunked download
    path — the large-payload path the small-model tests don't reach. Derived from the
    pt_large_model_pass_through integration job; ScriptRunner keeps large NUMPY arrays on the wire
    while trainer-side API conversion hands the script torch."""
    jobdir = tmp_path / "job"
    jobdir.mkdir()
    (jobdir / "client.py").write_text(_LARGE_CLIENT_SCRIPT)
    (jobdir / "model.py").write_text(_LARGE_MODEL_SCRIPT)
    (jobdir / "run_job.py").write_text(_LARGE_JOB_SCRIPT)
    workdir = tmp_path / "sim"

    env = os.environ.copy()
    env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    command = f"{sys.executable} -u"

    proc = subprocess.run(
        [sys.executable, "-u", str(jobdir / "run_job.py"), str(workdir), command],
        cwd=str(jobdir),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    out = proc.stdout + proc.stderr

    assert "Finished FedAvg." in out, f"large-model job did not finish FedAvg cleanly:\n{out[-4000:]}"
    assert "Aggregated 2/2 results" in out, f"server did not aggregate both clients:\n{out[-4000:]}"
    # the large model completed both rounds on both clients (streamed download + conversion)
    assert "LARGE_ROUND 0 ok" in out, f"round 0 did not complete on the large model:\n{out[-4000:]}"
    assert "LARGE_ROUND 1 ok" in out, f"round 1 did not complete on the large model:\n{out[-4000:]}"
    assert "LARGE_NOT_TORCH" not in out, f"trainer conversion failed for the streamed large model:\n{out[-4000:]}"
    assert "Traceback" not in out, f"large-model job raised:\n{out[-4000:]}"


# --- Multi-rank distributed trainer (torch.distributed rank contract, end-to-end) ----------------
#
# The pt_client_api_torchrun_cpu integration job launches the trainer under torchrun with two ranks
# and relies on the rank contract: only rank 0 opens a Client API session and talks to NVFlare;
# non-zero ranks stay passive and get the model via the framework's own collectives. torchrun's own
# gloo rendezvous hangs on macOS (c10d probes the hostname over IPv6), so this test uses an
# equivalent launcher that execs two independent worker processes with STATIC env-based rendezvous
# (MASTER_ADDR=127.0.0.1 + a free port) — the same process shape torchrun produces (CJ -> launcher
# -> workers), the same rank contract, and real torch.distributed collectives, minus the macOS
# rendezvous. Proves the full multi-rank path end-to-end: rank-0 Client API session + F3 payload
# pull, passive non-zero ranks, and a live all_reduce across ranks.

_MRANK_NET = "import torch.nn as nn\n\n\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc1 = nn.Linear(10, 5)\n\n    def forward(self, x):\n        return self.fc1(x)\n"  # noqa: E501

_MRANK_LAUNCH_SCRIPT = textwrap.dedent(
    """
    import os
    import socket
    import subprocess
    import sys


    def _free_port():
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        p = s.getsockname()[1]
        s.close()
        return p


    if __name__ == "__main__":
        port = _free_port()
        here = os.path.dirname(os.path.abspath(__file__))
        procs = []
        for rank in range(2):
            env = os.environ.copy()
            env["MASTER_ADDR"] = "127.0.0.1"
            env["MASTER_PORT"] = str(port)
            env["RANK"] = str(rank)
            env["WORLD_SIZE"] = "2"
            env["LOCAL_RANK"] = str(rank)
            procs.append(subprocess.Popen([sys.executable, "-u", os.path.join(here, "worker.py")], env=env))
        rc = 0
        for p in procs:
            rc = p.wait() or rc
        sys.exit(rc)
    """
)

_MRANK_WORKER_SCRIPT = textwrap.dedent(
    """
    import torch
    import torch.distributed as dist
    import nvflare.client as flare


    def _bcast_from_rank0(obj):
        box = [obj if dist.get_rank() == 0 else None]
        dist.broadcast_object_list(box, src=0)
        return box[0]


    def main():
        dist.init_process_group(backend="gloo")
        try:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            flare.init(rank=rank)
            received = flare.receive()
            # rank contract: only rank 0 gets the model from NVFlare; non-zero ranks stay passive
            if rank == 0:
                assert received is not None and received.params, "RANK0_GOT_NO_MODEL"
                print("MRANK_RANK0_RECEIVED_OK", flush=True)
            else:
                assert received is None, "NONZERO_RANK_GOT_MODEL"
                print("MRANK_NONZERO_PASSIVE_OK", flush=True)
            input_model = _bcast_from_rank0(received)  # framework collective distributes to all ranks
            assert input_model is not None and input_model.params is not None, "BROADCAST_FAILED"
            contrib = torch.tensor(float(rank + 1))
            dist.all_reduce(contrib, op=dist.ReduceOp.SUM)  # real torch.distributed collective
            assert contrib.item() == 3.0, "ALLREDUCE_" + str(contrib.item())
            print(f"MRANK_ALLREDUCE_OK rank={rank} sum={contrib.item()}", flush=True)
            if rank == 0:
                # echo the model back (type-agnostic: params may be numpy or torch on the wire)
                flare.send(
                    flare.FLModel(
                        params=input_model.params, params_type=flare.ParamsType.FULL,
                        metrics={"accuracy": 1.0}, meta={"NUM_STEPS_CURRENT_ROUND": world_size},
                    )
                )
                print("MRANK_RANK0_SENT_OK", flush=True)
        finally:
            dist.destroy_process_group()


    if __name__ == "__main__":
        main()
    """
)

_MRANK_SERVER_CONF = textwrap.dedent(
    """
    {
      format_version = 2
      task_data_filters = []
      task_result_filters = []
      model_class_path = "net.Net"
      workflows = [
        {
          id = "scatter_and_gather"
          path = "nvflare.app_common.workflows.scatter_and_gather.ScatterAndGather"
          args {
            min_clients = 2
            num_rounds = 1
            start_round = 0
            wait_time_after_min_received = 0
            aggregator_id = "aggregator"
            persistor_id = "persistor"
            shareable_generator_id = "shareable_generator"
            train_task_name = "train"
            train_timeout = 0
          }
        }
      ]
      components = [
        {
          id = "persistor"
          path = "nvflare.app_opt.pt.file_model_persistor.PTFileModelPersistor"
          args { model { path = "{model_class_path}" } }
        }
        {
          id = "shareable_generator"
          path = "nvflare.app_common.shareablegenerators.full_model_shareable_generator.FullModelShareableGenerator"
          args {}
        }
        {
          id = "aggregator"
          path = "nvflare.app_common.aggregators.intime_accumulate_model_aggregator.InTimeAccumulateWeightedAggregator"
          args { expected_data_kind = "WEIGHTS" }
        }
      ]
    }
    """
)

_MRANK_CLIENT_CONF = textwrap.dedent(
    """
    {
      format_version = 2
      app_script = "dist_launch.py"
      app_config = ""
      executors = [
        {
          tasks = ["train"]
          executor {
            path = "nvflare.app_common.executors.client_api_executor.ClientAPIExecutor"
            args {
              execution_mode = "external_process"
              command = "__PYTHON__ -u custom/dist_launch.py"
              # Mirrors pt_client_api_torchrun_cpu: this trainer handles one task and
              # returns, so the backend launches and owns one process tree per task.
              launch_once = false
              stop_grace_period = 5.0
              train_with_evaluation = true
            }
          }
        }
      ]
      task_data_filters = []
      task_result_filters = []
      components = []
    }
    """
)

_MRANK_META_CONF = textwrap.dedent(
    """
    {
      name = "mrank_external_process"
      resource_spec {}
      deploy_map { app = ["@ALL"] }
      min_clients = 2
      mandatory_clients = []
    }
    """
)


@pytest.mark.skipif(not _torch_available(), reason="requires torch for the multi-rank distributed example")
def test_external_process_multi_rank_distributed_end_to_end(tmp_path):
    """A multi-rank distributed trainer (two ranks per client via an exec'd-subprocess launcher with
    static gloo rendezvous — torchrun's process shape, minus its macOS-broken rendezvous) runs
    end-to-end: rank 0 drives the Client API session and F3 payload transfer, non-zero ranks stay
    passive (rank contract), and a real torch.distributed all_reduce runs across ranks."""
    jobdir = tmp_path / "mrank_external_process"
    (jobdir / "app" / "config").mkdir(parents=True)
    (jobdir / "app" / "custom").mkdir(parents=True)
    (jobdir / "app" / "config" / "config_fed_server.conf").write_text(_MRANK_SERVER_CONF)
    (jobdir / "app" / "config" / "config_fed_client.conf").write_text(
        _MRANK_CLIENT_CONF.replace("__PYTHON__", sys.executable)
    )
    (jobdir / "app" / "custom" / "net.py").write_text(_MRANK_NET)
    (jobdir / "app" / "custom" / "dist_launch.py").write_text(_MRANK_LAUNCH_SCRIPT)
    (jobdir / "app" / "custom" / "worker.py").write_text(_MRANK_WORKER_SCRIPT)
    (jobdir / "meta.conf").write_text(_MRANK_META_CONF)
    workdir = tmp_path / "sim"

    env = os.environ.copy()
    env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env["GLOO_SOCKET_IFNAME"] = "lo0" if sys.platform == "darwin" else "lo"

    proc = subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "nvflare.private.fed.app.simulator.simulator",
            str(jobdir),
            "-w",
            str(workdir),
            "-n",
            "2",
            "-t",
            "2",
            "-c",
            "site-1,site-2",
        ],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    out = proc.stdout + proc.stderr

    assert "Finished ScatterAndGather Training." in out, f"multi-rank job did not finish:\n{out[-4000:]}"
    # rank contract held on both clients: rank 0 received directly, non-zero ranks stayed passive
    assert "MRANK_RANK0_RECEIVED_OK" in out, f"rank 0 did not receive the model:\n{out[-4000:]}"
    assert "MRANK_NONZERO_PASSIVE_OK" in out, f"non-zero rank contract not observed:\n{out[-4000:]}"
    # a real torch.distributed collective ran across BOTH ranks
    assert "MRANK_ALLREDUCE_OK rank=0" in out, f"rank 0 did not complete the collective:\n{out[-4000:]}"
    assert "MRANK_ALLREDUCE_OK rank=1" in out, f"rank 1 did not complete the collective:\n{out[-4000:]}"
    assert "MRANK_RANK0_SENT_OK" in out, f"rank 0 did not send its result:\n{out[-4000:]}"
    assert "aggregating 2 update(s)" in out, f"server did not aggregate both clients:\n{out[-4000:]}"
    assert "AssertionError" not in out, f"a rank-contract assertion fired:\n{out[-4000:]}"
    assert "Traceback" not in out, f"multi-rank job raised:\n{out[-4000:]}"


# --- PyTorch Lightning integration (flare.patch) -------------------------------------------------
#
# Derived from the pt_lightning integration job. It uses the distinct Lightning client integration:
# `import nvflare.client.lightning as flare` + `flare.patch(trainer)`, which installs FL callbacks on
# the Lightning Trainer so trainer.fit()/validate() load the global model and send the result. Runs
# torch-throughout the trainer with numpy on the wire via representation formats declared in the
# CONFIG (the same TASK_EXCHANGE declaration ScriptRunner generates) — proving a config-based job
# gets the same trainer-side API conversion. Synthetic tensor data, CPU (no GPU/dataset).

_LIGHTNING_NET = (
    "import torch.nn as nn\n\n\n"
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.fc1 = nn.Linear(10, 5)\n\n"
    "    def forward(self, x):\n"
    "        return self.fc1(x)\n"
)

_LIGHTNING_PL_NET = textwrap.dedent(
    """
    import re
    import net
    import pytorch_lightning as L
    import torch
    from torch import nn, optim


    class PlNet(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.model = net.Net()
            self.site_name = "site-1"
            self.current_round = 0

        def training_step(self, batch, batch_idx):
            if self.current_round == 0:
                weight = torch.zeros([5, 10], dtype=torch.float32)
                bias = torch.zeros([5], dtype=torch.float32)
            else:
                weight = self.model.state_dict().get("fc1.weight")
                bias = self.model.state_dict().get("fc1.bias")
            multiplier = int(re.search(r"\\d+", self.site_name).group())
            weight = torch.add(weight, 1) * multiplier
            bias = torch.add(bias, 1) * multiplier
            self.model.load_state_dict({"fc1.weight": weight, "fc1.bias": bias})
            x = batch.view(1, -1)
            y = torch.ones(1, 10)
            loss = nn.functional.mse_loss(x, y)
            loss.requires_grad_(True)
            return loss

        def validation_step(self, batch, batch_idx):
            return torch.tensor([0.0])

        def configure_optimizers(self):
            return optim.Adam(self.parameters(), lr=1e-3)
    """
)

_LIGHTNING_CLIENT = textwrap.dedent(
    """
    import pl_net
    import pytorch_lightning as L
    import torch
    from torch import utils

    import nvflare.client.lightning as flare


    def main():
        plnet = pl_net.PlNet()
        dataset = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
             [4.0, 3.0, 2.0, 1.0, 2.0, 5.0, 6.0, 2.0, 1.0, 32.0]]
        )
        train_loader = utils.data.DataLoader(dataset)
        trainer = L.Trainer(
            limit_train_batches=1, max_epochs=1, accelerator="cpu",
            enable_progress_bar=False, logger=False,
        )
        flare.patch(trainer)
        print(f"LIGHTNING_PATCHED cb={len(trainer.callbacks)}", flush=True)
        site_name = flare.get_site_name()
        while flare.is_running():
            input_model = flare.receive()
            if input_model is None:
                break
            print(f"LIGHTNING_ROUND {input_model.current_round} site={site_name}", flush=True)
            plnet.current_round = input_model.current_round
            plnet.site_name = site_name
            trainer.validate(plnet, train_loader)
            trainer.fit(plnet, train_loader)
        print("LIGHTNING_DONE", flush=True)


    if __name__ == "__main__":
        main()
    """
)

_LIGHTNING_SERVER_CONF = textwrap.dedent(
    """
    {
      format_version = 2
      task_data_filters = []
      task_result_filters = []
      model_class_path = "pl_net.PlNet"
      workflows = [
        {
          id = "scatter_and_gather"
          path = "nvflare.app_common.workflows.scatter_and_gather.ScatterAndGather"
          args {
            min_clients = 2
            num_rounds = 2
            start_round = 0
            wait_time_after_min_received = 0
            aggregator_id = "aggregator"
            persistor_id = "persistor"
            shareable_generator_id = "shareable_generator"
            train_task_name = "train"
            train_timeout = 0
          }
        }
      ]
      components = [
        {
          id = "persistor"
          path = "nvflare.app_opt.pt.file_model_persistor.PTFileModelPersistor"
          args { model { path = "{model_class_path}" } }
        }
        {
          id = "shareable_generator"
          path = "nvflare.app_common.shareablegenerators.full_model_shareable_generator.FullModelShareableGenerator"
          args {}
        }
        {
          id = "aggregator"
          path = "nvflare.app_common.aggregators.intime_accumulate_model_aggregator.InTimeAccumulateWeightedAggregator"
          args { expected_data_kind = "WEIGHTS" }
        }
      ]
    }
    """
)

# Trainer/server representations declared in the executor config (what ScriptRunner generates
# automatically); __PYTHON__ is replaced with the interpreter under test.
_LIGHTNING_CLIENT_CONF = textwrap.dedent(
    """
    {
      format_version = 2
      app_script = "client.py"
      app_config = ""
      executors = [
        {
          tasks = ["train"]
          executor {
            path = "nvflare.app_common.executors.client_api_executor.ClientAPIExecutor"
            args {
              execution_mode = "external_process"
              command = "__PYTHON__ -u custom/client.py"
              launch_once = true
              stop_grace_period = 5.0
              train_with_evaluation = true
              params_exchange_format = "pytorch"
              server_expected_format = "numpy"
            }
          }
        }
      ]
      task_data_filters = []
      task_result_filters = []
      components = []
    }
    """
)

_LIGHTNING_META_CONF = textwrap.dedent(
    """
    {
      name = "pt_lightning_external_process"
      resource_spec {}
      deploy_map { app = ["@ALL"] }
      min_clients = 2
      mandatory_clients = []
    }
    """
)


@pytest.mark.skipif(
    not (_torch_available() and _lightning_available()),
    reason="requires torch + pytorch_lightning for the Lightning example",
)
def test_external_process_pytorch_lightning_config_job_end_to_end(tmp_path):
    """The pt_lightning integration job (nvflare.client.lightning flare.patch pattern) runs on
    ClientAPIExecutor(external_process). The trainer works in torch; declared numpy<->torch API
    conversion keeps the wire numpy and hands the Lightning module torch."""
    jobdir = tmp_path / "pt_lightning_external_process"
    (jobdir / "app" / "config").mkdir(parents=True)
    (jobdir / "app" / "custom").mkdir(parents=True)
    (jobdir / "app" / "config" / "config_fed_server.conf").write_text(_LIGHTNING_SERVER_CONF)
    (jobdir / "app" / "config" / "config_fed_client.conf").write_text(
        _LIGHTNING_CLIENT_CONF.replace("__PYTHON__", sys.executable)
    )
    (jobdir / "app" / "custom" / "net.py").write_text(_LIGHTNING_NET)
    (jobdir / "app" / "custom" / "pl_net.py").write_text(_LIGHTNING_PL_NET)
    (jobdir / "app" / "custom" / "client.py").write_text(_LIGHTNING_CLIENT)
    (jobdir / "meta.conf").write_text(_LIGHTNING_META_CONF)
    workdir = tmp_path / "sim"

    env = os.environ.copy()
    env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "nvflare.private.fed.app.simulator.simulator",
            str(jobdir),
            "-w",
            str(workdir),
            "-n",
            "2",
            "-t",
            "2",
            "-c",
            "site-1,site-2",
        ],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    out = proc.stdout + proc.stderr

    assert "Finished ScatterAndGather Training." in out, f"lightning job did not finish:\n{out[-4000:]}"
    # the Lightning trainer was patched and both rounds ran through fit()
    assert "LIGHTNING_PATCHED" in out, f"flare.patch did not install callbacks:\n{out[-4000:]}"
    assert "LIGHTNING_ROUND 1" in out, f"second round did not run through the Lightning trainer:\n{out[-4000:]}"
    assert "aggregating 2 update(s) at round 1" in out, f"server did not aggregate both clients:\n{out[-4000:]}"
    # trainer-side API conversion delivered torch to the Lightning module (no numpy load_state_dict error)
    assert "expected torch.Tensor" not in out, f"trainer conversion did not deliver torch:\n{out[-4000:]}"
    assert "Traceback" not in out, f"lightning job raised:\n{out[-4000:]}"


# --- TensorFlow (Keras) framework ----------------------------------------------------------------
#
# The fourth ML framework. TensorFlow has no wheel for Python 3.14, so this test self-skips on the
# 3.14 CI interpreter and runs wherever tensorflow is importable (e.g. a Python 3.13 environment).
# It mirrors the TF FedAvg examples: ScriptRunner declares numpy<->keras-layer-weights conversion
# (KERAS_LAYER_WEIGHTS exchange), so flare.receive() hands the script a
# {layer_name: [weight_arrays]} dict for model.get_layer(k).set_weights(v), and the wire/server
# stay numpy. Synthetic data, CPU.

_TF_MODEL_SCRIPT = textwrap.dedent(
    """
    from tensorflow.keras import layers, models


    def TinyTFNet():
        return models.Sequential([
            layers.Input(shape=(10,)),
            layers.Dense(8, activation="relu", name="d1"),
            layers.Dense(2, name="d2"),
        ])
    """
)

_TF_CLIENT_SCRIPT = textwrap.dedent(
    """
    import numpy as np
    import tensorflow as tf
    from model import TinyTFNet

    import nvflare.client as flare

    flare.init()
    model = TinyTFNet()
    model.compile(
        optimizer="sgd",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    rng = np.random.RandomState(0)
    X = rng.randn(64, 10).astype("float32")
    y = rng.randint(0, 2, size=(64,)).astype("int32")
    while flare.is_running():
        input_model = flare.receive()
        if input_model is None:
            break
        # trainer-side conversion delivers keras layer weights {layer_name: [arrays]}
        for k, v in input_model.params.items():
            model.get_layer(k).set_weights(v)
        model.fit(X, y, epochs=1, batch_size=16, verbose=0)
        print(f"TF_ROUND {input_model.current_round} ok", flush=True)
        flare.send(
            flare.FLModel(
                params={layer.name: layer.get_weights() for layer in model.layers},
                metrics={"accuracy": 0.5}, current_round=input_model.current_round,
            )
        )
    """
)

_TF_JOB_SCRIPT = textwrap.dedent(
    """
    import sys
    from nvflare.app_opt.tf.job_config.model import TFModel
    from nvflare.app_common.workflows.fedavg import FedAvg
    from nvflare.job_config.base_fed_job import BaseFedJob
    from nvflare.job_config.script_runner import ScriptRunner
    from nvflare.fuel.utils.constants import FrameworkType
    from nvflare.recipe.utils import extract_persistor_id

    n_clients, num_rounds, workdir, command = 2, 2, sys.argv[1], sys.argv[2]
    job = BaseFedJob(name="ext-tf-e2e", min_clients=n_clients)
    # dict-config model so the server app config is JSON-serializable (path resolved at runtime)
    pid = extract_persistor_id(job.to_server(TFModel(model={"path": "model.TinyTFNet", "args": {}}), id="persistor"))
    job.to_server(FedAvg(num_clients=n_clients, num_rounds=num_rounds, persistor_id=pid, task_name="train"))
    job.to_clients(
        ScriptRunner(
            script="client.py", execution_mode="external_process", command=command,
            framework=FrameworkType.TENSORFLOW,
        ),
        tasks=["train"],
    )
    job.simulator_run(workdir, n_clients=n_clients, threads=n_clients)
    """
)


@pytest.mark.skipif(not _tf_available(), reason="requires tensorflow (no wheel for Python 3.14)")
def test_external_process_tensorflow_fedavg_end_to_end(tmp_path):
    """A real TensorFlow/Keras FedAvg job runs as subprocesses on external_process. The ScriptRunner
    declares numpy<->keras-layer-weights API conversion, so the wire stays numpy while the TF
    script works with keras layer weights. Self-skips where tensorflow is unavailable (Python 3.14)."""
    jobdir = tmp_path / "job"
    jobdir.mkdir()
    (jobdir / "client.py").write_text(_TF_CLIENT_SCRIPT)
    (jobdir / "model.py").write_text(_TF_MODEL_SCRIPT)
    (jobdir / "run_job.py").write_text(_TF_JOB_SCRIPT)
    workdir = tmp_path / "sim"

    env = os.environ.copy()
    env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"  # quiet TF's native logging
    command = f"{sys.executable} -u"

    proc = subprocess.run(
        [sys.executable, "-u", str(jobdir / "run_job.py"), str(workdir), command],
        cwd=str(jobdir),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    out = proc.stdout + proc.stderr

    assert "Finished FedAvg." in out, f"TF job did not finish FedAvg cleanly:\n{out[-4000:]}"
    assert "Aggregated 2/2 results" in out, f"server did not aggregate both clients:\n{out[-4000:]}"
    # both rounds ran on both TF clients (keras set_weights/get_weights round-trip)
    assert "TF_ROUND 0 ok" in out, f"round 0 did not complete on the TF clients:\n{out[-4000:]}"
    assert "TF_ROUND 1 ok" in out, f"round 1 did not complete on the TF clients:\n{out[-4000:]}"
    assert "Traceback" not in out, f"TF job raised:\n{out[-4000:]}"


# --- Real downloaded dataset (data-dependent job) ------------------------------------------------
#
# The data-dependent examples (hello-pt, cifar10) load a real dataset and train a real network. The
# distinctive part (over the synthetic-data tests) is the dataset load + training loop; the backend
# is data-agnostic. This proves that path with a REAL downloaded dataset (MNIST — small; CIFAR10's
# host throttles to ~50 KB/s here) and a real CNN training loop over real image batches. The test
# pre-downloads MNIST in-process and self-skips if torchvision or the network is unavailable, so it
# never leaves a hard network dependency in CI.

_MNIST_MODEL_SCRIPT = textwrap.dedent(
    """
    import torch.nn as nn


    class MnistMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 64)
            self.fc2 = nn.Linear(64, 10)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            return self.fc2(nn.functional.relu(self.fc1(x)))
    """
)

# __DATAROOT__ is replaced with the pre-downloaded dataset dir; the client loads it with
# download=False so the trainer subprocess never depends on the network itself.
_MNIST_CLIENT_SCRIPT = textwrap.dedent(
    """
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as T
    from model import MnistMLP

    import nvflare.client as flare

    ds = torchvision.datasets.MNIST(root="__DATAROOT__", train=False, download=False, transform=T.ToTensor())
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)

    net = MnistMLP()
    flare.init()
    while flare.is_running():
        m = flare.receive()
        if m is None:
            break
        assert all(isinstance(v, torch.Tensor) for v in m.params.values()), "MNIST_NOT_TORCH"
        net.load_state_dict(m.params)
        opt = torch.optim.SGD(net.parameters(), lr=0.01)
        crit = nn.CrossEntropyLoss()
        for i, (x, y) in enumerate(loader):
            opt.zero_grad()
            crit(net(x), y).backward()
            opt.step()
            if i >= 4:  # a few real batches on real MNIST images
                break
        print(f"MNIST_ROUND {m.current_round} ok", flush=True)
        flare.send(
            flare.FLModel(params=net.cpu().state_dict(), params_type=flare.ParamsType.FULL,
                         current_round=m.current_round)
        )
    """
)

_MNIST_JOB_SCRIPT = textwrap.dedent(
    """
    import sys
    import torch  # noqa
    from model import MnistMLP
    from nvflare.app_opt.pt.job_config.model import PTModel
    from nvflare.app_common.workflows.fedavg import FedAvg
    from nvflare.job_config.base_fed_job import BaseFedJob
    from nvflare.job_config.script_runner import ScriptRunner
    from nvflare.recipe.utils import extract_persistor_id

    n_clients, num_rounds, workdir, command = 2, 2, sys.argv[1], sys.argv[2]
    job = BaseFedJob(name="ext-mnist-e2e", min_clients=n_clients)
    pid = extract_persistor_id(job.to_server(PTModel(MnistMLP()), id="persistor"))
    job.to_server(FedAvg(num_clients=n_clients, num_rounds=num_rounds, persistor_id=pid, task_name="train"))
    job.to_clients(ScriptRunner(script="client.py", execution_mode="external_process", command=command), tasks=["train"])
    job.simulator_run(workdir, n_clients=n_clients, threads=n_clients)
    """
)


@pytest.mark.skipif(
    not (_torch_available() and _torchvision_available()),
    reason="requires torch + torchvision for the data-dependent example",
)
def test_external_process_real_dataset_pytorch_end_to_end(tmp_path):
    """A data-dependent job (real downloaded MNIST + a real CNN training loop over real image
    batches) runs as subprocesses on external_process. Pre-downloads MNIST in-process and skips if
    the network is unavailable, so the committed test carries no hard network dependency."""
    import torchvision

    dataroot = tmp_path / "data"
    try:
        torchvision.datasets.MNIST(root=str(dataroot), train=False, download=True)
    except Exception as e:
        pytest.skip(f"MNIST dataset not downloadable in this environment: {e}")

    jobdir = tmp_path / "job"
    jobdir.mkdir()
    (jobdir / "client.py").write_text(_MNIST_CLIENT_SCRIPT.replace("__DATAROOT__", str(dataroot)))
    (jobdir / "model.py").write_text(_MNIST_MODEL_SCRIPT)
    (jobdir / "run_job.py").write_text(_MNIST_JOB_SCRIPT)
    workdir = tmp_path / "sim"

    env = os.environ.copy()
    env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    command = f"{sys.executable} -u"

    proc = subprocess.run(
        [sys.executable, "-u", str(jobdir / "run_job.py"), str(workdir), command],
        cwd=str(jobdir),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    out = proc.stdout + proc.stderr

    assert "Finished FedAvg." in out, f"MNIST job did not finish FedAvg cleanly:\n{out[-4000:]}"
    assert "Aggregated 2/2 results" in out, f"server did not aggregate both clients:\n{out[-4000:]}"
    # both rounds trained on real MNIST images on both clients
    assert "MNIST_ROUND 0 ok" in out, f"round 0 did not complete on real data:\n{out[-4000:]}"
    assert "MNIST_ROUND 1 ok" in out, f"round 1 did not complete on real data:\n{out[-4000:]}"
    assert "MNIST_NOT_TORCH" not in out, f"trainer conversion did not deliver torch:\n{out[-4000:]}"
    assert "Traceback" not in out, f"MNIST job raised:\n{out[-4000:]}"
