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

import builtins
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_EXAMPLE_PATH = Path(__file__).resolve().parents[3] / "examples" / "advanced" / "slurm" / "slurm_torchrun_node.py"
_SPEC = importlib.util.spec_from_file_location("slurm_torchrun_node_example", _EXAMPLE_PATH)
helper = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = helper
_SPEC.loader.exec_module(helper)


def test_public_cli_preserves_every_argument_after_boundary():
    options = helper._parse_public_cli(
        [
            "--nproc-per-node",
            "4",
            "--rdzv-port-base",
            "31000",
            "--rdzv-port-span",
            "17",
            "--",
            "train.py",
            "--nproc-per-node",
            "different",
            "two words",
            "",
            "--",
        ]
    )

    assert options == helper._PublicOptions(
        nproc_per_node=4,
        rdzv_port_base=31000,
        rdzv_port_span=17,
        training_argv=("train.py", "--nproc-per-node", "different", "two words", "", "--"),
    )


@pytest.mark.parametrize("argv", [[], ["train.py"], ["--"]])
def test_public_cli_requires_boundary_and_training_script(argv):
    with pytest.raises(helper.SlurmTorchrunError):
        helper._parse_public_cli(argv)


def test_private_node_mode_refuses_options():
    with pytest.raises(helper.SlurmTorchrunError, match="does not accept"):
        helper._parse_node_cli(["--nproc-per-node", "8", "--", "train.py"])


def test_main_dispatches_private_node_mode_with_exact_training_argv(monkeypatch):
    calls = []
    environ = {"marker": "value"}
    monkeypatch.setattr(helper, "_run_node", lambda argv, env: calls.append((argv, env)) or 23)

    assert helper.main([helper._PRIVATE_NODE_MODE, "--", "train.py", "--x"], environ) == 23
    assert calls == [(("train.py", "--x"), environ)]


def test_main_returns_configuration_error_status(capsys):
    assert helper.main(["train.py"], {}) == 2
    assert "requires '--'" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("job_id", "base", "span", "expected"),
    [(0, 29400, 1000, 29400), (1234, 29400, 1000, 29634), (20, 65535, 1, 65535)],
)
def test_compute_master_port_is_deterministic(job_id, base, span, expected):
    assert helper._compute_master_port(job_id, base, span) == expected


@pytest.mark.parametrize(("base", "span"), [(1023, 1), (29400, 0), (29400, -1), (65535, 2)])
def test_compute_master_port_rejects_invalid_range(base, span):
    with pytest.raises(helper.SlurmTorchrunError):
        helper._compute_master_port(1, base, span)


def test_discover_hosts_uses_exact_scontrol_command(monkeypatch):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout="node-a\nnode-b\n", stderr="")

    monkeypatch.setattr(helper.subprocess, "run", fake_run)

    assert helper._discover_hosts("/opt/slurm/bin/scontrol", "node-[1-2]", 2) == ("node-a", "node-b")
    assert calls == [
        (
            ["/opt/slurm/bin/scontrol", "show", "hostnames", "node-[1-2]"],
            {"shell": False, "capture_output": True, "text": True, "check": False},
        )
    ]


@pytest.mark.parametrize("stdout", ["node-a\n", "node-a\nnode-a\n", "node-a\n\n"])
def test_discover_hosts_requires_exact_unique_nonempty_count(monkeypatch, stdout):
    monkeypatch.setattr(
        helper.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=stdout, stderr=""),
    )

    with pytest.raises(helper.SlurmTorchrunError, match="exactly 2 unique nonempty"):
        helper._discover_hosts("/scontrol", "nodes", 2)


def test_discover_hosts_reports_scontrol_failure(monkeypatch):
    monkeypatch.setattr(
        helper.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=3, stdout="", stderr="bad nodelist\n"),
    )

    with pytest.raises(helper.SlurmTorchrunError, match="exit code 3: bad nodelist"):
        helper._discover_hosts("/scontrol", "nodes", 2)


def test_explicit_nproc_does_not_probe_gpus():
    assert helper._resolve_nproc_per_node(3, {}, lambda: pytest.fail("must not probe CUDA")) == 3


@pytest.mark.parametrize("value", ["8", "0002"])
def test_auto_nproc_prefers_positive_slurm_gpu_count(value):
    assert helper._resolve_nproc_per_node(
        "auto", {"SLURM_GPUS_ON_NODE": value}, lambda: pytest.fail("must not probe CUDA")
    ) == int(value)


@pytest.mark.parametrize(
    ("slurm_value", "cuda_count", "expected"),
    [(None, 4, 4), ("0", 2, 2), ("invalid", 0, 1), ("-1", 0, 1)],
)
def test_auto_nproc_falls_back_to_cuda_then_cpu(slurm_value, cuda_count, expected):
    environ = {} if slurm_value is None else {"SLURM_GPUS_ON_NODE": slurm_value}
    assert helper._resolve_nproc_per_node("auto", environ, lambda: cuda_count) == expected


def test_cuda_probe_reports_missing_pytorch_as_configuration_error(monkeypatch):
    real_import = builtins.__import__

    def import_without_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch is unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_torch)

    with pytest.raises(helper.SlurmTorchrunError, match="PyTorch is required"):
        helper._cuda_device_count()


def test_srun_argv_is_exact_and_repeats_positive_cpu_count():
    assert helper._build_srun_argv("/srun", "/python", 2, 12, ["train.py", "--epochs", "3"]) == [
        "/srun",
        "--nodes=2",
        "--ntasks=2",
        "--ntasks-per-node=1",
        "--distribution=block:block",
        "--cpus-per-task=12",
        "--kill-on-bad-exit=1",
        "--export=ALL",
        "/python",
        helper._SCRIPT_PATH,
        "__nvflare_slurm_node__",
        "--",
        "train.py",
        "--epochs",
        "3",
    ]


def test_srun_argv_omits_cpu_flag_without_positive_count():
    argv = helper._build_srun_argv("/srun", "/python", 2, None, ["train.py"])
    assert not any(arg.startswith("--cpus-per-task=") for arg in argv)


def test_torchrun_argv_is_exact():
    assert helper._build_torchrun_argv("/python", 3, 1, "node-a", 29417, 4, ["train.py", "--x"]) == [
        "/python",
        "-m",
        "torch.distributed.run",
        "--nnodes=3",
        "--nproc-per-node=4",
        "--node-rank=1",
        "--master-addr=node-a",
        "--master-port=29417",
        "--max-restarts=0",
        "--",
        "train.py",
        "--x",
    ]


def _node_environ(node_id="1"):
    return {
        "SLURM_NODEID": node_id,
        helper._ENV_NNODES: "3",
        helper._ENV_MASTER_ADDR: "node-a",
        helper._ENV_MASTER_PORT: "29417",
        helper._ENV_NPROC: "4",
    }


def test_private_node_mode_execs_current_python_with_exact_training_argv(monkeypatch):
    calls = []
    monkeypatch.setattr(helper.os, "execv", lambda executable, argv: calls.append((executable, argv)))

    helper._run_node(["train.py", "two words", "", "--flag"], _node_environ("0"))
    assert calls == [
        (
            sys.executable,
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--nnodes=3",
                "--nproc-per-node=4",
                "--node-rank=0",
                "--master-addr=node-a",
                "--master-port=29417",
                "--max-restarts=0",
                "--",
                "train.py",
                "two words",
                "",
                "--flag",
            ],
        )
    ]


@pytest.mark.parametrize("node_id", [None, "", "-1", "+1", "1.0", "3"])
def test_private_node_mode_requires_rank_in_allocation(node_id):
    environ = _node_environ()
    if node_id is None:
        environ.pop("SLURM_NODEID")
    else:
        environ["SLURM_NODEID"] = node_id

    with pytest.raises(helper.SlurmTorchrunError, match="SLURM_NODEID"):
        helper._run_node(["train.py"], environ)


def _public_environ():
    return {
        "SLURM_JOB_ID": "41",
        "SLURM_NNODES": "2",
        "SLURM_JOB_NODELIST": "node-[a-b]",
        "SLURMD_NODENAME": "node-a",
        "SLURM_CPUS_PER_TASK": "7",
    }


def test_public_mode_runs_exact_srun(monkeypatch):
    environ = _public_environ()
    monkeypatch.setattr(helper, "_discover_hosts", lambda *args: ("node-a", "node-b"))
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        child_env = kwargs["env"]
        assert child_env[helper._ENV_NNODES] == "2"
        assert child_env[helper._ENV_MASTER_ADDR] == "node-a"
        assert child_env[helper._ENV_MASTER_PORT] == "29441"
        assert child_env[helper._ENV_NPROC] == "2"
        return SimpleNamespace(returncode=19)

    monkeypatch.setattr(helper.subprocess, "run", fake_run)

    options = helper._PublicOptions(2, 29400, 1000, ("train.py", "--epochs", "3"))
    assert helper._run_public(options, environ) == 19
    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert argv == helper._build_srun_argv("srun", sys.executable, 2, 7, ["train.py", "--epochs", "3"])
    assert set(kwargs) == {"shell", "env", "check"}
    assert kwargs["shell"] is False
    assert kwargs["check"] is False
    assert helper._ENV_NNODES not in environ


def test_public_mode_requires_execution_on_ordered_node_zero(monkeypatch):
    environ = _public_environ()
    environ["SLURMD_NODENAME"] = "node-b"
    monkeypatch.setattr(helper, "_discover_hosts", lambda *args: ("node-a", "node-b"))

    with pytest.raises(helper.SlurmTorchrunError, match="allocation node 0"):
        helper._run_public(helper._PublicOptions(1, 29400, 1000, ("train.py",)), environ)


@pytest.mark.parametrize(
    ("name", "value", "error"),
    [
        ("SLURM_JOB_ID", "not-numeric", "SLURM_JOB_ID"),
        ("SLURM_NNODES", "0", "SLURM_NNODES"),
        ("SLURM_JOB_NODELIST", "", "SLURM_JOB_NODELIST"),
        ("SLURMD_NODENAME", "", "SLURMD_NODENAME"),
    ],
)
def test_public_mode_requires_valid_slurm_allocation_environment(name, value, error):
    environ = _public_environ()
    environ[name] = value

    with pytest.raises(helper.SlurmTorchrunError, match=error):
        helper._run_public(helper._PublicOptions(1, 29400, 1000, ("train.py",)), environ)


@pytest.mark.parametrize(("returncode", "expected"), [(0, 0), (7, 7), (-15, 143), (-9, 137)])
def test_public_mode_normalizes_srun_status(monkeypatch, returncode, expected):
    monkeypatch.setattr(helper, "_discover_hosts", lambda *args: ("node-a", "node-b"))
    monkeypatch.setattr(helper.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=returncode))
    assert helper._run_public(helper._PublicOptions(1, 29400, 1000, ("train.py",)), _public_environ()) == expected
