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
"""Real Flower runner checks. They need Flower's full dependency set next to NVFlare and skip otherwise.

CI installs Flower with --no-deps for the unit suite, so these run only where Flower is fully installed (see the
"Flower integration" step of the premerge workflow).
"""

import os
import shlex
import subprocess
import sys
import textwrap

import pytest

from nvflare.app_opt.flower.applet import FlowerClientApplet, FlowerServerApplet
from tests.flower_test_utils import build_superlink_command, build_supernode_command
from tests.process_test_utils import run_owned_process

_BIN_DIR = os.path.dirname(sys.executable)
_HELP_ENV = {**os.environ, "COLUMNS": "250", "NO_COLOR": "1", "TERM": "dumb"}


def _flower_runnable() -> bool:
    exe = os.path.join(_BIN_DIR, "flower-superlink")
    if not os.path.isfile(exe):
        return False
    try:
        return subprocess.run([exe, "--help"], capture_output=True, timeout=120, env=_HELP_ENV).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


pytestmark = pytest.mark.skipif(
    not _flower_runnable(), reason="Flower is not fully installed in this environment (flower-superlink --help fails)"
)


def _help_text(*args) -> str:
    result = subprocess.run(
        [os.path.join(_BIN_DIR, args[0]), *args[1:], "--help"],
        capture_output=True,
        text=True,
        timeout=120,
        env=_HELP_ENV,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout + result.stderr


def _flags(command: str):
    return sorted({token.split("=")[0] for token in shlex.split(command) if token.startswith("--")})


def test_every_flag_the_applet_passes_exists_in_the_installed_flower(tmp_path):
    """Catches the next flag removal (as with --serverappio-api-address in 1.36 and --grpc-adapter in 1.43).

    Each command is built by the applet itself, so a flag added to the applet is checked automatically.
    """
    supernode = build_supernode_command(FlowerClientApplet(allow_runtime_dependency_installation=True))
    superlink_flags = set()
    for allow in (True, False):  # the two settings pass different, mutually exclusive flags
        applet = FlowerServerApplet(
            database=str(tmp_path / "state.db"),
            superlink_ready_timeout=1.0,
            allow_runtime_dependency_installation=allow,
        )
        superlink_flags.update(_flags(build_superlink_command(applet, tmp_path).cmd))
    server_applet = FlowerServerApplet(database="", superlink_ready_timeout=1.0, run_config={"a": 1})
    server_applet.superlink_api_addr = "127.0.0.1:9091"
    server_applet.flwr_home_dir = str(tmp_path)

    checks = {
        ("flower-supernode",): _flags(supernode.cmd),
        ("flower-superlink",): sorted(superlink_flags),
    }
    for sub in ("run", "stop", "list"):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("nvflare.app_opt.flower.applet._validate_flower_executable", lambda *a: None)
            cmd = server_applet._flower_command(sub, "123" if sub == "stop" else "")
        checks[("flwr", sub)] = _flags(cmd)

    # sanity: the checks are not vacuous
    assert "--port" in checks[("flower-supernode",)] and "--port" in checks[("flower-superlink",)]

    missing = {}
    for command, flags in checks.items():
        text = _help_text(*command)
        absent = [f for f in flags if f not in text]
        if absent:
            missing[" ".join(command)] = absent
    assert not missing, f"flags passed by the applet but unknown to the installed Flower: {missing}"


_PYPROJECT = """
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "bridgeapp"
version = "1.0.0"
license = "Apache-2.0"
# Unresolvable on purpose: Flower 1.43's SuperLink runs "uv sync" on the app's dependencies unless runtime
# dependency installation is disabled, so the run can only complete if NVFlare disables it.
dependencies = ["flwr>=1.43", "nvflare-test-package-that-does-not-exist==0.0.1"]

[tool.hatch.build.targets.wheel]
packages = ["bridgeapp"]

[tool.flwr.app]
publisher = "nvflare-test"

[tool.flwr.app.components]
serverapp = "bridgeapp.server:app"
clientapp = "bridgeapp.client:app"

[tool.flwr.app.config]
num-server-rounds = 2
"""

_SERVER = """
import numpy as np
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg


def server_fn(context: Context):
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_available_clients=2,
        initial_parameters=ndarrays_to_parameters([np.zeros((100, 100), dtype=np.float32)]),
    )
    return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=context.run_config["num-server-rounds"]))


app = ServerApp(server_fn=server_fn)
"""

_CLIENT = """
from flwr.client import ClientApp, NumPyClient


class Client(NumPyClient):
    def fit(self, parameters, config):
        return [p + 1 for p in parameters], 10, {}


app = ClientApp(client_fn=lambda context: Client().to_client())
"""

_JOB = """
import sys

from nvflare.app_opt.flower.recipe import FlowerRecipe
from nvflare.recipe import SimEnv

recipe = FlowerRecipe(name="bridge", flower_content=sys.argv[1], min_clients=2, run_config={"num-server-rounds": 2})
run = recipe.execute(SimEnv(num_clients=2, workspace_root=sys.argv[2]))
print("JOB STATUS:", run.get_status())
"""


def test_flower_app_completes_all_rounds_in_the_simulator_through_the_bridge(tmp_path):
    """Real SuperLink, two real SuperNodes and ClientApps, the real flwr CLI, all through the FLARE tunnel.

    The app is numpy-only so no dataset is downloaded. run_owned_process reaps every descendant, including
    Flower subprocesses that detach from their parent.
    """
    app = tmp_path / "app"
    (app / "bridgeapp").mkdir(parents=True)
    (app / "pyproject.toml").write_text(textwrap.dedent(_PYPROJECT))
    (app / "bridgeapp" / "__init__.py").write_text("")
    (app / "bridgeapp" / "server.py").write_text(textwrap.dedent(_SERVER))
    (app / "bridgeapp" / "client.py").write_text(textwrap.dedent(_CLIENT))
    job = tmp_path / "job.py"
    job.write_text(textwrap.dedent(_JOB))

    result = run_owned_process(
        [sys.executable, "-I", str(job), str(app), str(tmp_path / "workspace")], cwd=str(tmp_path), timeout=420
    )

    assert result.returncode == 0, result.stdout[-5000:] + result.stderr[-5000:]
    assert "JOB STATUS: FINISHED:COMPLETED" in result.stdout
