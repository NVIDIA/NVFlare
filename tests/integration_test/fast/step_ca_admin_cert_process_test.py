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

import json
import socket
import subprocess
import sys
import time
from contextlib import suppress

import psutil
import pytest

from nvflare.fuel.sec.admin_cert_provider import AdminCertProviderError
from nvflare.fuel.sec.step_ca_admin_cert import obtain_step_ca_admin_cert_files


@pytest.mark.parametrize("interrupted", [False, True])
def test_step_ca_source_stops_login_child_before_retry(monkeypatch, tmp_path, interrupted):
    ready = tmp_path / "login.json"
    child_code = (
        "import json, os, socket, sys, time\n"
        "from pathlib import Path\n"
        "listener = socket.socket()\n"
        "listener.bind(('127.0.0.1', 0))\n"
        "listener.listen()\n"
        "ready = Path(sys.argv[1])\n"
        "staging = ready.with_suffix('.tmp')\n"
        "staging.write_text(json.dumps({'pid': os.getpid(), 'port': listener.getsockname()[1]}))\n"
        "staging.replace(ready)\n"
        "time.sleep(60)\n"
    )
    fake_step = tmp_path / "step"
    fake_step.write_text(
        f"#!{sys.executable}\n"
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}, {str(ready)!r}])\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    fake_step.chmod(0o755)
    popen = subprocess.Popen
    parents = []
    children = []

    def start_step(*args, **kwargs):
        process = popen(*args, **kwargs)
        parents.append((process, psutil.Process(process.pid)))
        # Start the timeout only once the child has acquired its callback port.
        deadline = time.monotonic() + 10
        while not ready.exists():
            assert process.poll() is None, "fake step exited before login was ready"
            assert time.monotonic() < deadline, "login child did not become ready"
            time.sleep(0.01)
        children.append(psutil.Process(json.loads(ready.read_text())["pid"]))
        if interrupted:
            wait = process.wait

            def interrupt_wait(*_args, **_kwargs):
                process.wait = wait
                raise KeyboardInterrupt

            process.wait = interrupt_wait
        return process

    monkeypatch.setattr(subprocess, "Popen", start_step)
    try:
        error = KeyboardInterrupt if interrupted else AdminCertProviderError
        with pytest.raises(error, match=None if interrupted else "timed out after 0.01 seconds"):
            obtain_step_ca_admin_cert_files(
                config={
                    "ca_url": "https://step-ca.example.com",
                    "provisioner": "nvflare-admin-oidc",
                    "step_bin": str(fake_step),
                    "command_timeout": 0.01,
                },
                root_ca_file="unused-root.pem",
            )

        assert all(process.poll() is not None for process, _identity in parents)
        with suppress(psutil.NoSuchProcess):
            assert children[0].status() in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD)
        with socket.socket() as retry:
            retry.bind(("127.0.0.1", json.loads(ready.read_text())["port"]))
    finally:
        # Discover descendants even when readiness fails partway through startup.
        for _process, identity in parents:
            with suppress(psutil.NoSuchProcess):
                children.extend(identity.children(recursive=True))
        for child in children:
            with suppress(psutil.NoSuchProcess):
                child.kill()
        for process, _identity in parents:
            process.kill()
            popen.wait(process, timeout=5)
        _gone, alive = psutil.wait_procs(children, timeout=5)
        for child in alive:
            with suppress(psutil.NoSuchProcess):
                assert child.status() in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD)
