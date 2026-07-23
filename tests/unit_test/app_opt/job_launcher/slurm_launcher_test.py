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

from types import SimpleNamespace

import pytest

from nvflare.apis.app_validation import AppValidationKey
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, JobConstants, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.job_launcher_spec import JobProcessArgs, JobProcessEnv
from nvflare.apis.workspace import Workspace
from nvflare.app_opt.job_launcher.slurm.config import (
    SLURM_CHILD_PROCESS_ENV,
    SlurmLauncherError,
    _validate_mount_destination,
)
from nvflare.app_opt.job_launcher.slurm.launcher import (
    ClientSlurmJobLauncher,
    ServerSlurmJobLauncher,
    _resolve_parent_host,
    _resolve_resources,
    _rewrite_parent_url,
)
from nvflare.app_opt.job_launcher.slurm.manager import _job_key
from nvflare.app_opt.job_launcher.study_runtime import SecretEnvRef, StudyRuntime


def _workspace(tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / "startup").mkdir(parents=True)
    (workspace / "local").mkdir()
    (workspace / "job-1").mkdir()
    return workspace


def _launcher(tmp_path, workspace, sandbox="none", image=None, launcher_class=ClientSlurmJobLauncher):
    return launcher_class(
        workspace_path=str(workspace),
        sandbox=sandbox,
        image=image,
        python_path="/usr/bin/python3",
        executables={name: "/usr/bin/true" for name in ("sbatch", "squeue", "sacct", "scancel")},
        parent_host="compute.example",
    )


def _fl_ctx(workspace, module="ignored.by.slurm"):
    context = FLContext()
    context.set_prop(ReservedKey.IDENTITY_NAME, "site-1", private=False, sticky=True)
    context.set_prop(
        FLContextKey.WORKSPACE_OBJECT,
        Workspace(str(workspace), site_name="site-1"),
        private=True,
        sticky=False,
    )
    context.set_prop(
        FLContextKey.JOB_PROCESS_ARGS,
        {
            JobProcessArgs.EXE_MODULE: ("-m", module),
            JobProcessArgs.JOB_ID: ("-n", "job-1"),
            JobProcessArgs.PARENT_URL: ("-p", "tcp://old-host:8102"),
            JobProcessArgs.PARENT_CONN_SEC: ("--parent_conn_sec", "clear"),
            JobProcessArgs.AUTH_TOKEN: ("-t", "secret-token"),
            JobProcessArgs.TOKEN_SIGNATURE: ("-ts", "secret-signature"),
            JobProcessArgs.SSID: ("-d", "secret-ssid"),
        },
        private=True,
        sticky=False,
    )
    return context


def test_resource_resolution_combines_portable_gpu_total():
    job_meta = {JobMetaKey.RESOURCE_SPEC.value: {"site-1": {"num_of_gpus": 2}}}

    resources = _resolve_resources(job_meta, "site-1", "none", 600, spec={})

    assert resources.gpus_per_node == 2
    assert resources.nodes == 1


@pytest.mark.parametrize(
    "spec, message",
    [
        ({"unknown": 1}, "unsupported"),
        ({"nodes": 2}, "multi-node Slurm jobs require"),
        ({"pending_timeout": 601}, "may only reduce"),
    ],
)
def test_resource_resolution_rejects_invalid_job_policy(spec, message):
    with pytest.raises(SlurmLauncherError, match=message):
        _resolve_resources({}, "site-1", "apptainer", 600, spec=spec)


def test_time_value_is_passed_to_slurm_without_custom_grammar():
    resources = _resolve_resources({}, "site-1", "none", 600, spec={"time": "site-defined"})

    assert resources.time_limit == "site-defined"


def test_explicit_parent_host_always_wins(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.setenv("SLURMD_NODENAME", "compute-1")

    assert _resolve_parent_host("login.example") == "login.example"


def test_allocated_parent_uses_slurmd_node_name(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.setenv("SLURMD_NODENAME", "compute-1")

    assert _resolve_parent_host(None) == "compute-1"


def test_unallocated_parent_requires_configuration(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)

    with pytest.raises(SlurmLauncherError, match="parent_host is required"):
        _resolve_parent_host(None)


def test_parent_url_rewrite_is_shallow_and_preserves_other_entries():
    nested = {"value": 1}
    args = {JobProcessArgs.PARENT_URL: ("-p", "tcp://old:8102"), "nested": nested}

    rewritten = _rewrite_parent_url(args, "new-host", 8102)

    assert rewritten[JobProcessArgs.PARENT_URL][1] == "tcp://new-host:8102"
    assert rewritten["nested"] is nested
    assert args[JobProcessArgs.PARENT_URL][1] == "tcp://old:8102"


def test_parent_url_rewrite_formats_ipv6_host():
    args = {JobProcessArgs.PARENT_URL: ("-p", "tcp://[2001:db8::1]:8102")}

    rewritten = _rewrite_parent_url(args, "2001:db8::2", 8102)

    assert rewritten[JobProcessArgs.PARENT_URL][1] == "tcp://[2001:db8::2]:8102"


@pytest.mark.parametrize(
    "entry, message",
    [
        (None, "missing or malformed"),
        (("-p", "tcp://old:not-a-port"), "malformed parent URL"),
        (("-p", "http://old:8102"), "must use tcp"),
        (("-p", "tcp://old:9000"), "configured internal_port"),
    ],
)
def test_parent_url_rewrite_rejects_malformed_or_incompatible_value(entry, message):
    args = {} if entry is None else {JobProcessArgs.PARENT_URL: entry}

    with pytest.raises(SlurmLauncherError, match=message):
        _rewrite_parent_url(args, "new-host", 8102)


@pytest.mark.parametrize("path", ["relative", "//double", "/tmp/../bad", "/proc/value", "/sys"])
def test_mount_destination_rejects_unsafe_shapes(path):
    with pytest.raises(SlurmLauncherError):
        _validate_mount_destination(path, "mount")


def test_mount_destination_accepts_normal_application_path():
    assert _validate_mount_destination("/data/study/input", "mount") == "/data/study/input"


def test_launch_plan_uses_fixed_worker_and_one_resolved_job_spec(tmp_path):
    workspace = _workspace(tmp_path)
    launcher = _launcher(tmp_path, workspace)

    plan = launcher._build_launch_plan({JobConstants.JOB_ID: "job-1"}, _fl_ctx(workspace))

    assert plan.exe_module == ClientSlurmJobLauncher.EXE_MODULE
    assert plan.site_name == "site-1"
    parent_url_index = plan.module_args.index("-p") + 1
    assert plan.module_args[parent_url_index] == "tcp://compute.example:8102"
    assert "secret-token" not in plan.module_args
    assert plan.study_secret_env[JobProcessEnv.AUTH_TOKEN] == "secret-token"
    assert plan.study_secret_env[JobProcessEnv.TOKEN_SIGNATURE] == "secret-signature"
    assert plan.study_secret_env[JobProcessEnv.SSID] == "secret-ssid"
    assert plan.resources.nodes == 1


def test_concrete_launchers_select_client_and_server_module_arguments(tmp_path):
    workspace = _workspace(tmp_path)
    job_args = {
        JobProcessArgs.AUTH_TOKEN: ("-t", "client-token"),
        JobProcessArgs.ROOT_URL: ("-r", "server-root"),
        JobProcessArgs.OPTIONS: ("--set", "alpha=1 'message=two words'"),
    }

    client_args = _launcher(tmp_path, workspace).get_module_args(job_args)
    server_args = _launcher(tmp_path, workspace, launcher_class=ServerSlurmJobLauncher).get_module_args(job_args)

    assert client_args == ("--set", "alpha=1", "message=two words")
    assert server_args == ("-r", "server-root", "--set", "alpha=1", "message=two words")


def test_launch_plan_rejects_job_id_with_trailing_newline(tmp_path):
    workspace = _workspace(tmp_path)
    launcher = _launcher(tmp_path, workspace)

    with pytest.raises(SlurmLauncherError, match="invalid job ID"):
        launcher._build_launch_plan({JobConstants.JOB_ID: "job-1\n"}, _fl_ctx(workspace))


def test_launch_plan_rejects_different_context_workspace(tmp_path):
    configured_workspace = _workspace(tmp_path / "configured")
    context_workspace = _workspace(tmp_path / "context")
    launcher = _launcher(tmp_path, configured_workspace)

    with pytest.raises(SlurmLauncherError, match="workspace does not match"):
        launcher._build_launch_plan({JobConstants.JOB_ID: "job-1"}, _fl_ctx(context_workspace))


def test_non_clear_internal_connection_is_rejected(tmp_path):
    workspace = _workspace(tmp_path)
    launcher = _launcher(tmp_path, workspace)
    context = _fl_ctx(workspace)
    context.get_prop(FLContextKey.JOB_PROCESS_ARGS)[JobProcessArgs.PARENT_CONN_SEC] = ("--parent_conn_sec", "tls")

    with pytest.raises(SlurmLauncherError, match="requires clear"):
        launcher._build_launch_plan({JobConstants.JOB_ID: "job-1"}, context)


def test_study_environment_requires_secret_source(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path)
    launcher = _launcher(tmp_path, workspace)
    monkeypatch.delenv("MISSING_STUDY_SECRET", raising=False)
    runtime = StudyRuntime(
        study="study-a",
        secret_env=[SecretEnvRef(name="DB_PASSWORD", source="MISSING_STUDY_SECRET")],
    )

    with pytest.raises(SlurmLauncherError, match="requires parent environment variable"):
        launcher._study_environment(runtime)


def test_legacy_study_data_file_is_rejected(tmp_path):
    workspace = _workspace(tmp_path)
    (workspace / "local" / "study_data.yaml").write_text("studies: {}\n", encoding="utf-8")
    launcher = _launcher(tmp_path, workspace)

    with pytest.raises(SlurmLauncherError, match="legacy study data file"):
        launcher._load_study_runtime(None)


def test_job_image_overrides_study_and_site_after_byoc_authorization(tmp_path):
    workspace = _workspace(tmp_path)
    for name in ("site.sif", "study.sif", "job.sif"):
        (tmp_path / name).write_text("image", encoding="utf-8")
    (workspace / "local" / "study_runtime.yaml").write_text(
        """format_version: 2
studies:
  study-a:
    container:
      image: %s
    slurm:
      sandbox: apptainer
"""
        % (tmp_path / "study.sif"),
        encoding="utf-8",
    )
    launcher = _launcher(tmp_path, workspace, sandbox="apptainer", image=str(tmp_path / "site.sif"))
    meta = {
        JobConstants.JOB_ID: "job-1",
        AppValidationKey.BYOC: True,
        "study": "study-a",
        "launcher_spec": {"site-1": {"slurm": {"image": str(tmp_path / "job.sif")}}},
    }

    plan = launcher._build_launch_plan(meta, _fl_ctx(workspace))

    assert plan.image == str(tmp_path / "job.sif")


def test_sandbox_none_rejects_byoc_authorized_job_image(tmp_path):
    workspace = _workspace(tmp_path)
    image = tmp_path / "job.sif"
    image.write_text("image", encoding="utf-8")
    launcher = _launcher(tmp_path, workspace)
    meta = {
        JobConstants.JOB_ID: "job-1",
        AppValidationKey.BYOC: True,
        "launcher_spec": {"site-1": {"slurm": {"image": str(image)}}},
    }

    with pytest.raises(SlurmLauncherError, match="effective sandbox 'none'"):
        launcher._build_launch_plan(meta, _fl_ctx(workspace))


def test_study_mount_overlap_reports_validation_error(tmp_path):
    workspace = _workspace(tmp_path)
    image = tmp_path / "image.sif"
    dataset = tmp_path / "dataset"
    secret = tmp_path / "secret"
    image.write_text("image", encoding="utf-8")
    dataset.mkdir()
    secret.mkdir()
    (workspace / "local" / "study_runtime.yaml").write_text(
        f"""format_version: 2
studies:
  study-a:
    container:
      image: {image}
    datasets:
      input:
        source: {dataset}
        mode: ro
    secret_mounts:
      duplicate:
        source: {secret}
        mount_path: /data/study-a/input
""",
        encoding="utf-8",
    )
    launcher = _launcher(tmp_path, workspace, sandbox="apptainer", image=str(image))

    with pytest.raises(SlurmLauncherError, match="overlaps another study mount"):
        launcher._build_launch_plan({JobConstants.JOB_ID: "job-1", "study": "study-a"}, _fl_ctx(workspace))


def test_public_events_only_initialize_register_and_shutdown(tmp_path):
    workspace = _workspace(tmp_path)
    launcher = _launcher(tmp_path, workspace)
    calls = []
    launcher.manager = SimpleNamespace(
        initialize=lambda: calls.append(("initialize", None)),
        shutdown=lambda: calls.append(("shutdown", None)),
    )
    context = FLContext()

    launcher.handle_event(EventType.SYSTEM_BOOTSTRAP, context)
    launcher.handle_event(EventType.BEFORE_JOB_LAUNCH, context)
    launcher.handle_event(EventType.SYSTEM_END, context)

    assert [name for name, _ in calls] == ["initialize", "shutdown"]
    assert context.get_prop(FLContextKey.JOB_LAUNCHER) == [launcher]


def test_child_process_is_inert_and_cannot_nest_slurm(monkeypatch, tmp_path):
    workspace = _workspace(tmp_path)
    monkeypatch.setenv(SLURM_CHILD_PROCESS_ENV, "1")
    launcher = _launcher(tmp_path, workspace)

    assert launcher.manager is None
    with pytest.raises(SlurmLauncherError, match="nested"):
        launcher.launch_job({}, FLContext())


def test_manager_bootstrap_keeps_existing_job_artifacts(tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / "startup").mkdir(parents=True)
    (workspace / "local").mkdir()
    workspace.chmod(0o700)
    control = workspace / ".nvflare_slurm"
    control.mkdir()
    old_job = control / "jobs" / _job_key("stale-job")
    old_job.mkdir(parents=True)
    secret = old_job / "secret.env"
    secret.write_text("export TEST_SECRET=value\n", encoding="utf-8")
    launcher = _launcher(tmp_path, workspace)
    launcher.manager._require_accounting = lambda: None
    launcher.manager.adapter = SimpleNamespace()

    launcher.manager.initialize()

    assert launcher.manager.jobs_dir == str(control / "jobs")
    assert secret.read_text(encoding="utf-8") == "export TEST_SECRET=value\n"
