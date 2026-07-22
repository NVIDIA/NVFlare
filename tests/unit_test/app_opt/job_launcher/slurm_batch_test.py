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

import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from nvflare.apis.job_launcher_spec import JobProcessEnv
from nvflare.app_opt.job_launcher.slurm.batch import (
    _render_batch_script,
    _render_node_script,
    _render_secret_file,
    _submission_argv,
)
from nvflare.app_opt.job_launcher.slurm.config import BindMount, JobResources, LaunchPlan, SlurmConfig


def _job_dir(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    return str(job_dir)


def _config(tmp_path, sandbox="none"):
    return SlurmConfig(
        workspace_path=str(tmp_path / "workspace"),
        sandbox=sandbox,
        python_path="/usr/bin/python3",
        executables={
            "sbatch": "/usr/bin/sbatch",
            "squeue": "/usr/bin/squeue",
            "sacct": "/usr/bin/sacct",
            "scancel": "/usr/bin/scancel",
            "apptainer": "apptainer",
            "srun": "srun",
        },
    )


def _plan(tmp_path, sandbox="none", resources=None, mounts=(), forward_env=(), node_command=(), node_app_dir=None):
    run_dir = tmp_path / "workspace" / "job-1"
    run_dir.mkdir(parents=True, exist_ok=True)
    return LaunchPlan(
        job_id="job-1",
        site_name="site-1",
        run_dir=str(run_dir),
        exe_module="worker.module",
        module_args=("-n", "job-1"),
        resources=resources or JobResources(),
        directives={},
        sandbox=sandbox,
        image=None if sandbox == "none" else "/images/python.sif",
        setup="",
        study_env={"PLAIN": "value"},
        study_secret_env={
            "STUDY_SECRET": "hidden",
            JobProcessEnv.AUTH_TOKEN: "secret-token",
            JobProcessEnv.TOKEN_SIGNATURE: "secret-signature",
            JobProcessEnv.SSID: "secret-ssid",
        },
        mounts=mounts,
        python_path="/usr/bin/python3",
        python_env="/custom",
        forward_env=forward_env,
        node_command=node_command,
        node_app_dir=node_app_dir,
    )


def _multinode_plan(tmp_path, sandbox="none", node_command=("python3", "-m", "trainer", "--epochs", "2")):
    app_dir = tmp_path / "workspace" / "job-1" / "app_site-1"
    app_dir.mkdir(parents=True, exist_ok=True)
    return _plan(
        tmp_path,
        sandbox=sandbox,
        resources=JobResources(nodes=2, gpus_per_node=1),
        node_command=node_command,
        node_app_dir=str(app_dir),
    )


def test_bare_renderer_has_no_slurm_exit_file_or_compute_preflight(tmp_path):
    plan = _plan(tmp_path)

    script, secrets = _render_batch_script(plan, _job_dir(tmp_path), _config(tmp_path))

    assert 'exec "${_nvfl_command[@]}"' in script
    assert "exit_code" not in script
    assert "command -v" not in script
    assert "set -a" not in script
    assert secrets["STUDY_SECRET"] == "hidden"
    assert secrets[JobProcessEnv.AUTH_TOKEN] == "secret-token"


@pytest.mark.parametrize("sandbox", ["none", "apptainer", "pyxis"])
def test_renderer_delivers_credentials_through_environment(tmp_path, sandbox):
    plan = _plan(tmp_path, sandbox=sandbox)

    script, secrets = _render_batch_script(
        plan,
        _job_dir(tmp_path),
        _config(tmp_path, sandbox),
    )
    secret_file = _render_secret_file(secrets)

    for value in ("hidden", "secret-token", "secret-signature", "secret-ssid"):
        assert value not in script
    assert "worker.module" in script
    command_line = next(line for line in script.splitlines() if line.startswith("_nvfl_command="))
    assert "-t" not in command_line
    assert "-ts" not in command_line
    assert "-d" not in command_line
    assert "export STUDY_SECRET=hidden" in secret_file
    assert f"export {JobProcessEnv.AUTH_TOKEN}=secret-token" in secret_file
    assert f"export {JobProcessEnv.TOKEN_SIGNATURE}=secret-signature" in secret_file
    assert f"export {JobProcessEnv.SSID}=secret-ssid" in secret_file


@pytest.mark.parametrize("worker_exists", [True, False])
def test_rendered_batch_propagates_final_exec_status_and_removes_secret(tmp_path, worker_exists):
    worker = tmp_path / "worker"
    if worker_exists:
        worker.write_text(
            "#!/usr/bin/env bash\n"
            "[[ $NVFLARE_JOB_AUTH_TOKEN == secret-token ]]\n"
            "[[ $NVFLARE_JOB_TOKEN_SIGNATURE == secret-signature ]]\n"
            "[[ $NVFLARE_JOB_SSID == secret-ssid ]]\n",
            encoding="utf-8",
        )
        worker.chmod(0o700)
    plan = replace(_plan(tmp_path), python_path=str(worker))
    job_dir = Path(_job_dir(tmp_path))
    script, secrets = _render_batch_script(plan, str(job_dir), _config(tmp_path))
    secret_path = job_dir / "secret.env"
    secret_path.write_text(_render_secret_file(secrets), encoding="utf-8")
    batch_path = job_dir / "batch.sh"
    batch_path.write_text(script, encoding="utf-8")
    batch_path.chmod(0o700)

    completed = subprocess.run([str(batch_path)], capture_output=True, text=True)

    if worker_exists:
        assert completed.returncode == 0
    else:
        assert completed.returncode != 0
    assert not secret_path.exists()


def test_multinode_batch_exports_node_group_contract_and_delegates_to_srun(tmp_path):
    plan = _multinode_plan(tmp_path)
    job_dir = _job_dir(tmp_path)

    script, _ = _render_batch_script(plan, job_dir, _config(tmp_path))

    assert 'export NVFL_NNODES="${SLURM_JOB_NUM_NODES:?}"' in script
    assert 'export NVFL_MASTER_ADDR="${SLURMD_NODENAME:?}"' in script
    assert 'export NVFL_MASTER_PORT="$((29400 + 10#${SLURM_JOB_ID} % 1000))"' in script
    assert 'export NVFL_RUN_ID="${SLURM_JOB_ID}"' in script
    assert '[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]' in script
    assert "NVFL_SRUN=srun" in script
    command_line = next(line for line in script.splitlines() if line.startswith("_nvfl_command="))
    assert "--nodes=2" in command_line
    assert "--ntasks-per-node=1" in command_line
    assert "--kill-on-bad-exit=1" in command_line
    assert "--wait=0" in command_line
    assert "--label" in command_line
    assert f"{job_dir}/node.sh" in command_line
    assert "worker.module" not in command_line


def test_site_port_range_overrides_the_default_rendezvous_ports(tmp_path):
    from dataclasses import replace as dc_replace

    config = dc_replace(_config(tmp_path), multi_node_port_range=(29500, 29599))

    script, _ = _render_batch_script(_multinode_plan(tmp_path), _job_dir(tmp_path), config)

    assert 'export NVFL_MASTER_PORT="$((29500 + 10#${SLURM_JOB_ID} % 100))"' in script


def test_apptainer_node_group_containerizes_each_rank_on_its_node(tmp_path):
    plan = _multinode_plan(tmp_path, sandbox="apptainer")
    job_dir = _job_dir(tmp_path)

    batch, _ = _render_batch_script(plan, job_dir, _config(tmp_path, "apptainer"))
    node = _render_node_script(plan, _config(tmp_path, "apptainer"))

    batch_command = next(line for line in batch.splitlines() if line.startswith("_nvfl_command="))
    assert "--ntasks-per-node=1" in batch_command
    assert f"{job_dir}/node.sh" in batch_command
    assert "NVFL_APPTAINER" not in batch_command
    assert 'export APPTAINERENV_NVFL_NNODES="${NVFL_NNODES}"' in batch
    assert 'export APPTAINERENV_NVFL_MASTER_ADDR="${NVFL_MASTER_ADDR}"' in batch

    assert 'export APPTAINERENV_NVFL_NODE_RANK="${NVFL_NODE_RANK}"' in node
    assert node.count('"${NVFL_APPTAINER}"') == 2
    assert f"--pwd {plan.run_dir}" in node
    assert f"--pwd {plan.node_app_dir}" in node
    assert "cd " not in node
    assert "worker.module" in node
    assert "python3 -m trainer --epochs 2" in node
    assert "unset NVFLARE_JOB_AUTH_TOKEN" in node
    assert "APPTAINERENV_NVFLARE_JOB_AUTH_TOKEN" in node
    assert "SINGULARITYENV_NVFLARE_JOB_AUTH_TOKEN" in node
    assert "node group topology mismatch" in node


def test_pyxis_node_group_fans_out_containers_through_one_srun(tmp_path):
    plan = _multinode_plan(tmp_path, sandbox="pyxis")
    job_dir = _job_dir(tmp_path)

    batch, _ = _render_batch_script(plan, job_dir, _config(tmp_path, "pyxis"))
    node = _render_node_script(plan, _config(tmp_path, "pyxis"))

    env_line = next(line for line in batch.splitlines() if line.startswith("_nvfl_container_env="))
    for name in ("NVFL_NNODES", "NVFL_MASTER_ADDR", "NVFL_MASTER_PORT", "NVFL_RUN_ID"):
        assert name in env_line
    batch_command = next(line for line in batch.splitlines() if line.startswith("_nvfl_command="))
    assert "--nodes=2" in batch_command
    assert "--kill-on-bad-exit=1" in batch_command
    assert "--container-image=/images/python.sif" in batch_command
    assert batch_command.rstrip(")").endswith(f"{job_dir}/node.sh")
    assert "worker.module" not in batch_command

    assert "apptainer" not in node
    assert f"cd {plan.node_app_dir}" in node
    assert "worker.module" in node


@pytest.mark.parametrize("node_rank, expected", [("0", "worker-ran token=cj-secret"), ("1", "rank=1 token=scrubbed")])
def test_rendered_node_script_executes_by_rank_and_scrubs_worker_credentials(tmp_path, node_rank, expected):
    worker = tmp_path / "worker"
    worker.write_text(
        '#!/usr/bin/env bash\necho "worker-ran token=${NVFLARE_JOB_AUTH_TOKEN:-missing}"\n', encoding="utf-8"
    )
    worker.chmod(0o700)
    plan = replace(
        _multinode_plan(
            tmp_path,
            node_command=("bash", "-c", 'echo "rank=${NVFL_NODE_RANK} token=${NVFLARE_JOB_AUTH_TOKEN:-scrubbed}"; pwd'),
        ),
        python_path=str(worker),
    )
    node_path = Path(_job_dir(tmp_path)) / "node.sh"
    node_path.write_text(_render_node_script(plan, _config(tmp_path)), encoding="utf-8")
    node_path.chmod(0o700)

    completed = subprocess.run(
        [str(node_path)],
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "SLURM_NODEID": node_rank,
            "SLURM_JOB_NUM_NODES": "2",
            "NVFLARE_JOB_AUTH_TOKEN": "cj-secret",
        },
    )

    assert completed.returncode == 0
    assert expected in completed.stdout
    if node_rank == "1":
        assert plan.node_app_dir in completed.stdout


def test_apptainer_renderer_keeps_isolation_contract(tmp_path):
    mount = BindMount("/data/source", "/data/study/data", "ro")
    plan = _plan(tmp_path, sandbox="apptainer", resources=JobResources(gpus_per_node=1), mounts=(mount,))

    script, _ = _render_batch_script(plan, _job_dir(tmp_path), _config(tmp_path, "apptainer"))

    for option in ("--userns", "--containall", "--no-privs", "--nv"):
        assert option in script
    assert "NVFL_APPTAINER=apptainer" in script
    assert "/data/source:/data/study/data:ro" in script


def test_pyxis_renderer_keeps_read_only_no_home_contract(tmp_path):
    mount = BindMount("/data/source", "/data/study/data", "ro")
    plan = _plan(tmp_path, sandbox="pyxis", mounts=(mount,))

    script, _ = _render_batch_script(plan, _job_dir(tmp_path), _config(tmp_path, "pyxis"))

    assert "--container-readonly" in script
    assert "--no-container-mount-home" in script
    assert "--no-container-entrypoint" in script
    assert "/data/source:/data/study/data:ro" in script


@pytest.mark.parametrize(
    "sandbox, expected",
    [
        ("none", "then export FORWARD_ME; fi"),
        ("apptainer", 'then export APPTAINERENV_FORWARD_ME="${FORWARD_ME}"; fi'),
        ("pyxis", 'then _nvfl_container_env="${_nvfl_container_env},FORWARD_ME"; fi'),
    ],
)
def test_renderer_forwards_selected_environment_for_each_backend(tmp_path, sandbox, expected):
    plan = _plan(tmp_path, sandbox=sandbox, forward_env=("FORWARD_ME",))

    script, _ = _render_batch_script(plan, _job_dir(tmp_path), _config(tmp_path, sandbox))

    assert expected in script


def test_submission_argv_is_structured_and_resource_complete(tmp_path):
    resources = JobResources(nodes=2, gpus_per_node=1, cpus_per_node=4, mem_per_node=1024, time_limit="1:00")
    plan = _plan(tmp_path, resources=resources)
    job_dir = _job_dir(tmp_path)

    argv = _submission_argv(plan, job_dir, "job-name", "marker", _config(tmp_path))

    for expected in (
        "--parsable",
        "--no-requeue",
        "--export=NIL",
        f"--chdir={plan.run_dir}",
        "--nodes=2",
        "--ntasks=2",
        "--ntasks-per-node=1",
        "--gres=gpu:1",
        "--cpus-per-task=4",
        "--mem=1024M",
        "--time=1:00",
        "--job-name=job-name",
        "--comment=marker",
        f"--output={plan.run_dir}/slurm-%j.out",
    ):
        assert expected in argv
