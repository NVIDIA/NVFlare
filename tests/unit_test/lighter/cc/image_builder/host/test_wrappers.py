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

"""Exercise interpreter selection and launch path handling without a VM or sudo."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from cvm.build.config import SOURCE


def executable(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    path.chmod(0o755)


class WrapperTests(unittest.TestCase):
    def test_interpreter_order_and_system_fallback_warning(self):
        for selected in ("system", "local", "explicit"):
            with self.subTest(selected=selected), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                builder = root / "builder space"
                builder.mkdir()
                wrapper = builder / "cvmctl"
                shutil.copyfile(SOURCE / "cvmctl", wrapper)
                wrapper.chmod(0o755)
                python = "#!/bin/sh\nprintf '%s\\n' SELECTED\nprintf 'arg=%s\\n' \"$@\"\n"
                executable(root / "bin/python3", python.replace("SELECTED", "system"))
                if selected != "system":
                    executable(builder / ".venv/bin/python", python.replace("SELECTED", "local"))
                env = dict(os.environ, PATH=str(root / "bin") + os.pathsep + os.defpath)
                env.pop("CVM_BUILDER_PYTHON", None)
                if selected == "explicit":
                    custom = root / "explicit environment/python"
                    executable(custom, python.replace("SELECTED", "explicit"))
                    env["CVM_BUILDER_PYTHON"] = str(custom)
                result = subprocess.run([str(wrapper), "--help"], env=env, capture_output=True, text=True, check=True)
                self.assertEqual(result.stdout.splitlines()[0], selected)
                self.assertIn("arg=--help", result.stdout)
                self.assertEqual("using system python3" in result.stderr, selected == "system")

    def test_launch_wrapper_preserves_caller_for_relative_bundle_arguments(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            delivery = root / "delivery space"
            caller = root / "caller space"
            delivery.mkdir()
            caller.mkdir()
            wrapper = delivery / "launch_cvm.sh"
            shutil.copyfile(SOURCE / "templates/launch_cvm.sh.tmpl", wrapper)
            wrapper.chmod(0o755)
            executable(
                root / "bin/python3",
                '#!/bin/sh\nprintf "cwd=%s\\n" "$PWD"\nprintf "arg=%s\\n" "$@"\n'
                'printf "pythonpath=%s\\n" "$PYTHONPATH"\nprintf "safe=%s\\n" "$PYTHONSAFEPATH"\n',
            )
            # Isolate path handling from the real wrapper's root-ownership check.
            executable(root / "bin/find", '#!/bin/sh\nprintf "%s" "$1" > "$FIND_TARGET"\n')
            env = dict(
                os.environ,
                PATH=str(root / "bin") + os.pathsep + os.defpath,
                FIND_TARGET=str(root / "find-target"),
            )
            for arguments in (["--cvm-bundle", "../generic bundle"], ["--cvm-bundle=../generic bundle"]):
                with self.subTest(arguments=arguments):
                    result = subprocess.run(
                        [str(wrapper), *arguments], cwd=caller, env=env, capture_output=True, text=True, check=True
                    )
                    self.assertIn(f"cwd={caller.resolve()}\n", result.stdout)
                    self.assertIn(f"pythonpath={delivery.resolve()}\n", result.stdout)
                    self.assertIn(f"arg={delivery.resolve()}\n", result.stdout)
                    self.assertIn("safe=1\n", result.stdout)
                    self.assertEqual((root / "find-target").read_text(), str(delivery.resolve()))
                    for argument in arguments:
                        self.assertIn(f"arg={argument}\n", result.stdout)
