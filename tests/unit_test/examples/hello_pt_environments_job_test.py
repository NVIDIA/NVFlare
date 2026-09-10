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

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from nvflare.recipe.run import Run
from tests.hello_pt_test_utils import HELLO_PT_DIR, REPO_ROOT, load_hello_pt_module

HAS_PT = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not HAS_PT, reason="PyTorch is not installed")
ADVANCED_DIR = REPO_ROOT / "examples" / "advanced" / "hello-pt-environments"


def _load_job_module():
    with load_hello_pt_module("job.py", example_dir=ADVANCED_DIR) as module:
        return module


def test_defaults_preserve_the_beginner_application():
    job_module = _load_job_module()

    args = job_module.parse_args([])

    assert args.env == "sim"
    assert args.n_clients == 2
    assert args.num_rounds == 3
    assert args.dataset == "synthetic"
    assert args.data_root == "/tmp/nvflare/data"
    assert args.evaluation == "final"
    assert args.experiment_tracking == "none"
    assert args.batch_size is args.epochs is args.learning_rate is args.num_workers is None


def test_production_only_arguments_are_rejected_elsewhere(capsys):
    job_module = _load_job_module()

    with pytest.raises(SystemExit, match="2"):
        job_module.parse_args(["--env", "prod"])
    assert "--startup-kit is required with --env prod" in capsys.readouterr().err

    with pytest.raises(SystemExit, match="2"):
        job_module.parse_args(["--startup-kit", "/tmp/admin"])
    assert "--startup-kit can only be used with --env prod" in capsys.readouterr().err

    with pytest.raises(SystemExit, match="2"):
        job_module.parse_args(["--username", "researcher@example.com"])
    assert "--username can only be used with --env prod" in capsys.readouterr().err


def test_help_includes_recipe_export_options():
    help_text = _load_job_module().define_parser().format_help()

    assert "--export" in help_text
    assert "--export-dir EXPORT_DIR" in help_text


def test_environment_selection_uses_the_same_client_count(tmp_path):
    job_module = _load_job_module()

    sim_env = job_module.create_environment(job_module.parse_args(["--n_clients", "3"]))
    poc_env = job_module.create_environment(job_module.parse_args(["--env", "poc", "--n_clients", "3"]))
    startup_kit = tmp_path / "admin-kit"
    startup_kit.mkdir()
    prod_env = job_module.create_environment(
        job_module.parse_args(
            [
                "--env",
                "prod",
                "--startup-kit",
                str(startup_kit),
                "--username",
                "researcher@example.com",
            ]
        )
    )

    assert isinstance(sim_env, job_module.SimEnv)
    assert isinstance(poc_env, job_module.PocEnv)
    assert isinstance(prod_env, job_module.ProdEnv)
    assert sim_env.num_clients == poc_env.num_clients == 3
    assert prod_env.startup_kit_location == str(startup_kit)
    assert prod_env.username == "researcher@example.com"


def test_default_recipe_forwards_only_application_selection(monkeypatch):
    job_module = _load_job_module()
    calls = []
    recipe = SimpleNamespace(enable_log_streaming=lambda: calls.append("log-streaming"))
    recipe_kwargs = {}
    monkeypatch.setattr(job_module, "FedAvgRecipe", lambda **kwargs: recipe_kwargs.update(kwargs) or recipe)
    monkeypatch.setattr(job_module, "create_model", lambda: "model")
    monkeypatch.setattr(job_module, "add_final_global_evaluation", lambda value: calls.append(("final", value)))
    monkeypatch.setattr(job_module, "add_cross_site_evaluation", lambda value: calls.append(("cross-site", value)))
    monkeypatch.setattr(
        job_module,
        "add_experiment_tracking",
        lambda value, tracking_type: calls.append(("tracking", value, tracking_type)),
    )

    result = job_module.create_recipe(job_module.parse_args([]))

    assert result is recipe
    assert recipe_kwargs["model"] == "model"
    assert recipe_kwargs["min_clients"] == 2
    assert recipe_kwargs["num_rounds"] == 3
    assert Path(recipe_kwargs["train_script"]).resolve() == HELLO_PT_DIR / "client.py"
    assert recipe_kwargs["train_args"] == ["--dataset", "synthetic"]
    assert calls == [("final", recipe)]


def test_advanced_recipe_controls_are_explicit(monkeypatch):
    job_module = _load_job_module()
    calls = []
    recipe = SimpleNamespace(enable_log_streaming=lambda: calls.append("log-streaming"))
    recipe_kwargs = {}
    monkeypatch.setattr(job_module, "FedAvgRecipe", lambda **kwargs: recipe_kwargs.update(kwargs) or recipe)
    monkeypatch.setattr(job_module, "create_model", lambda: "model")
    monkeypatch.setattr(job_module, "add_final_global_evaluation", lambda value: calls.append(("final", value)))
    monkeypatch.setattr(job_module, "add_cross_site_evaluation", lambda value: calls.append(("cross-site", value)))
    monkeypatch.setattr(
        job_module,
        "add_experiment_tracking",
        lambda value, tracking_type: calls.append(("tracking", value, tracking_type)),
    )
    args = job_module.parse_args(
        [
            "--dataset",
            "cifar10",
            "--data_root",
            "/data/cifar cache",
            "--batch_size",
            "16",
            "--epochs",
            "2",
            "--learning_rate",
            "0.01",
            "--num_workers",
            "2",
            "--evaluation",
            "cross-site",
            "--experiment_tracking",
            "tensorboard",
            "--enable_log_streaming",
            "--launch_external_process",
            "--client_memory_gc_rounds",
            "1",
        ]
    )

    job_module.create_recipe(args)

    assert recipe_kwargs["train_args"] == [
        "--dataset",
        "cifar10",
        "--data_root",
        "/data/cifar cache",
        "--batch_size",
        "16",
        "--epochs",
        "2",
        "--learning_rate",
        "0.01",
        "--num_workers",
        "2",
    ]
    assert recipe_kwargs["launch_external_process"] is True
    assert recipe_kwargs["client_memory_gc_rounds"] == 1
    assert calls == [("tracking", recipe, "tensorboard"), ("cross-site", recipe), "log-streaming"]


def test_main_preserves_successful_poc_result(tmp_path, monkeypatch, capsys):
    job_module = _load_job_module()
    result_dir = tmp_path / "poc-result"
    result_dir.mkdir()
    calls = []
    run = SimpleNamespace(
        get_result=lambda clean_up: calls.append(("get-result", clean_up)) or str(result_dir),
        get_status=lambda: calls.append(("get-status",)) or "FINISHED:COMPLETED",
    )
    env = SimpleNamespace(stop=lambda clean_up: calls.append(("stop", clean_up)), poc_workspace=str(tmp_path))
    recipe = SimpleNamespace(execute=lambda value: calls.append(("execute", value)) or run)
    monkeypatch.setattr(job_module, "create_recipe", lambda args: recipe)
    monkeypatch.setattr(job_module, "create_environment", lambda args: env)

    result = job_module.main(["--env", "poc"])

    assert result == str(result_dir)
    assert calls == [("execute", env), ("get-result", False), ("get-status",)]
    output = capsys.readouterr().out
    assert "Job Status is: FINISHED:COMPLETED" in output
    assert f"Result can be found in: {result_dir}" in output


def test_main_accepts_legacy_production_success_status(tmp_path, monkeypatch, capsys):
    job_module = _load_job_module()
    result_dir = tmp_path / "production-result"
    result_dir.mkdir()
    run = SimpleNamespace(
        get_result=lambda clean_up: str(result_dir),
        get_status=lambda: "FINISHED_OK",
    )
    env = SimpleNamespace()
    recipe = SimpleNamespace(execute=lambda value: run)
    monkeypatch.setattr(job_module, "create_recipe", lambda args: recipe)
    monkeypatch.setattr(job_module, "create_environment", lambda args: env)

    result = job_module.main(["--env", "prod", "--startup-kit", "/tmp/admin"])

    assert result == str(result_dir)
    assert "Job Status is: FINISHED_OK" in capsys.readouterr().out


def test_main_turns_real_run_monitoring_failure_into_error(monkeypatch):
    job_module = _load_job_module()
    stop_calls = []

    class FailingPocEnv:
        def get_job_result(self, job_id, timeout):
            raise RuntimeError("monitor unavailable")

        def get_job_status(self, job_id):
            return None

        def stop(self, clean_up):
            stop_calls.append(clean_up)

    env = FailingPocEnv()
    run = Run(env, "job-id")
    recipe = SimpleNamespace(execute=lambda value: run)
    monkeypatch.setattr(job_module, "create_recipe", lambda args: recipe)
    monkeypatch.setattr(job_module, "create_environment", lambda args: env)

    with pytest.raises(RuntimeError, match="did not return a result"):
        job_module.main(["--env", "poc"])

    assert stop_calls == [False, True]


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_main_leaves_failed_deployment_cleanup_to_environment(monkeypatch, error_type):
    job_module = _load_job_module()
    env = SimpleNamespace(stop=lambda **kwargs: pytest.fail("caller must not repeat deployment cleanup"))

    def execute(value):
        assert value is env
        raise error_type("deployment failed")

    monkeypatch.setattr(job_module, "create_recipe", lambda args: SimpleNamespace(execute=execute))
    monkeypatch.setattr(job_module, "create_environment", lambda args: env)
    with pytest.raises(error_type, match="deployment failed"):
        job_module.main(["--env", "poc"])


@pytest.mark.parametrize("status", ["FINISHED:ABORTED", "FINISHED:EXECUTION_EXCEPTION", None])
def test_main_retains_result_and_logs_on_unsuccessful_poc_status(tmp_path, monkeypatch, capsys, status):
    job_module = _load_job_module()
    stop_calls = []
    result_dir = tmp_path / "failed-result"
    result_dir.mkdir()
    log = tmp_path / "poc_console.log"
    log.write_text("client failure details")
    run = SimpleNamespace(get_result=lambda clean_up: str(result_dir), get_status=lambda: status)

    def stop(clean_up):
        stop_calls.append(clean_up)
        if clean_up:
            pytest.fail("downloaded results must be retained")

    env = SimpleNamespace(stop=stop, poc_workspace=str(tmp_path))
    recipe = SimpleNamespace(execute=lambda value: run)
    monkeypatch.setattr(job_module, "create_recipe", lambda args: recipe)
    monkeypatch.setattr(job_module, "create_environment", lambda args: env)

    with pytest.raises(RuntimeError, match="unsuccessful status:"):
        job_module.main(["--env", "poc"])

    assert stop_calls == [False]
    assert result_dir.is_dir()
    output = capsys.readouterr().err
    assert str(result_dir) in output
    assert str(log) in output


def test_exported_job_uses_the_bundled_shared_application(tmp_path, monkeypatch):
    with load_hello_pt_module("job.py", example_dir=ADVANCED_DIR) as job_module:
        args = job_module.parse_args([])
        recipe = job_module.create_recipe(args)
        env = job_module.create_environment(args)
        export_root = tmp_path / "job-config"
        monkeypatch.chdir(ADVANCED_DIR)
        recipe.export(job_dir=str(export_root), env=env)

    job_dir = export_root / "hello-pt"
    assert (job_dir / "meta.json").is_file()
    exported_python_files = {path.name for path in job_dir.rglob("*.py")}
    assert {"client.py", "model.py", "prepare_data.py"} <= exported_python_files
    assert (job_dir / "app" / "custom" / "client.py").is_file()

    client_config = json.loads((job_dir / "app" / "config" / "config_fed_client.json").read_text())
    executor_args = client_config["executors"][0]["executor"]["args"]
    assert executor_args["task_script_path"] == "client.py"


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit, GeneratorExit, pytest.fail.Exception])
def test_monitoring_interruption_stops_poc(monkeypatch, error_type):
    job_module = _load_job_module()
    calls = []

    def interrupt(clean_up):
        raise error_type("monitor interrupted")

    run = SimpleNamespace(get_result=interrupt)
    env = SimpleNamespace(stop=lambda clean_up: calls.append(clean_up))
    monkeypatch.setattr(job_module, "create_recipe", lambda args: SimpleNamespace(execute=lambda env: run))
    monkeypatch.setattr(job_module, "create_environment", lambda args: env)
    with pytest.raises(error_type, match="monitor interrupted"):
        job_module.main(["--env", "poc"])
    assert calls == [True]


@pytest.mark.parametrize("env_name", ["sim", "poc"])
@pytest.mark.parametrize("empty_files", [False, True])
def test_local_cifar_preflight_runs_before_recipe_or_environment(tmp_path, monkeypatch, env_name, empty_files):
    job_module = _load_job_module()
    if empty_files:
        batches = tmp_path / "cifar-10-batches-py"
        batches.mkdir()
        for name in [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch", "batches.meta"]:
            (batches / name).touch()
    monkeypatch.setattr(job_module, "create_recipe", lambda args: pytest.fail("recipe must not be created"))
    monkeypatch.setattr(job_module, "create_environment", lambda args: pytest.fail("environment must not start"))
    with pytest.raises(SystemExit, match="python ../../hello-world/hello-pt/prepare_data.py"):
        job_module.main(["--env", env_name, "--dataset", "cifar10", "--data_root", str(tmp_path)])


def test_cifar_production_does_not_check_admin_local_cache(monkeypatch):
    job_module = _load_job_module()
    monkeypatch.setattr(
        job_module, "validate_cifar10", lambda *args, **kwargs: pytest.fail("cache belongs to remote clients")
    )
    run = SimpleNamespace(get_result=lambda clean_up: "/tmp/result", get_status=lambda: "FINISHED_OK")
    monkeypatch.setattr(job_module, "create_recipe", lambda args: SimpleNamespace(execute=lambda env: run))
    monkeypatch.setattr(job_module, "create_environment", lambda args: object())
    assert job_module.main(["--env", "prod", "--startup-kit", "/tmp/admin", "--dataset", "cifar10"]) == "/tmp/result"


def test_negative_memory_gc_interval_is_rejected_before_execution(monkeypatch, capsys):
    job_module = _load_job_module()
    monkeypatch.setattr(job_module, "create_recipe", lambda args: pytest.fail("recipe must not be created"))
    with pytest.raises(SystemExit, match="2"):
        job_module.main(["--client_memory_gc_rounds", "-1"])
    assert "--client_memory_gc_rounds must be >= 0" in capsys.readouterr().err


@pytest.mark.parametrize("example_dir", [HELLO_PT_DIR, ADVANCED_DIR])
def test_cifar_cli_error_has_preparation_command_without_traceback(tmp_path, example_dir):
    result = subprocess.run(
        [sys.executable, "job.py", "--dataset", "cifar10", "--data_root", str(tmp_path)],
        cwd=example_dir,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "prepare_data.py --data_root" in result.stderr
    assert "Traceback" not in result.stderr


def test_copied_example_reports_missing_shared_application(tmp_path):
    script = tmp_path / "job.py"
    script.write_text((ADVANCED_DIR / "job.py").read_text())
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert "full NVFlare checkout" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("env_name", ["sim", "poc", "prod"])
def test_cifar_cli_export_does_not_require_local_cache(tmp_path, env_name):
    command = [
        sys.executable,
        "job.py",
        "--env",
        env_name,
        "--dataset",
        "cifar10",
        "--data_root",
        str(tmp_path / "missing"),
        "--export",
        "--export-dir",
        str(tmp_path / "export"),
    ]
    if env_name == "prod":
        kit = tmp_path / "admin"
        kit.mkdir()
        command.extend(["--startup-kit", str(kit)])
    subprocess.run(
        command,
        cwd=ADVANCED_DIR,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert (tmp_path / "export" / "hello-pt" / "meta.json").is_file()
