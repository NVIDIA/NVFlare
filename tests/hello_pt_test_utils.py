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
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HELLO_PT_DIR = REPO_ROOT / "examples" / "hello-world" / "hello-pt"

# Arm the watchdog before importing PyTorch/NVFlare, including during startup.
# Restore the script's argv and import path so CLI parsing and sibling imports
# behave as they do with ``python job.py ...``.
_EXPORT_BOOTSTRAP = """
import faulthandler
import os
import runpy
import sys

faulthandler.dump_traceback_later(float(sys.argv[1]), repeat=True)
sys.argv = sys.argv[2:]
sys.path[0] = os.path.dirname(os.path.abspath(sys.argv[0]))
runpy.run_path(sys.argv[0], run_name="__main__")
"""


def run_hello_pt_export(command, *, cwd, env, timeout=120, traceback_delay=90):
    """Allow cold framework imports in CI and expose diagnostics on failure."""
    try:
        return subprocess.run(
            [command[0], "-u", "-c", _EXPORT_BOOTSTRAP, str(traceback_delay), *command[1:]],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout,
        )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
        # TimeoutExpired can contain bytes even when text=True. Print captured
        # output because the exception's message alone omits the stack dump.
        for stream, output in (("stdout", error.stdout), ("stderr", error.stderr)):
            if output:
                if isinstance(output, bytes):
                    output = output.decode("utf-8", errors="replace")
                print(f"{stream} from {command!r}:\n{output}", file=sys.stderr)
        raise


@contextmanager
def load_hello_pt_module(file_name, example_dir=HELLO_PT_DIR):
    """Load a Hello PyTorch example without sharing sibling imports with other tests."""
    module_path = Path(example_dir) / file_name
    spec = importlib.util.spec_from_file_location(f"hello_pt_{module_path.stem}", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None

    sibling_names = ("model", "prepare_data")
    original_sys_path = list(sys.path)
    original_modules = {name: sys.modules[name] for name in sibling_names if name in sys.modules}
    for name in sibling_names:
        sys.modules.pop(name, None)
    sys.path[:0] = [str(example_dir), str(HELLO_PT_DIR)]
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path[:] = original_sys_path
        for name in sibling_names:
            sys.modules.pop(name, None)
        sys.modules.update(original_modules)
