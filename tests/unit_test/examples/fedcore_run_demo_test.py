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

from pathlib import Path

from tests.unit_test.examples.fedcore_test_utils import fedcore_import_context


def test_full_mode_default_defers_gpu_allocation_to_qwen_example():
    with fedcore_import_context():
        import run_demo

        args = run_demo.define_parser().parse_args(["--mode", "full"])
        command = run_demo._build_predictor_command(
            args,
            Path("/repo/examples/advanced/qwen3-vl"),
            Path("/tmp/fedcore data"),
            Path("/tmp/fedcore workspace"),
        )

    assert args.gpu is None
    assert run_demo._first_gpu(args.gpu) == "0"
    assert "--gpu" not in command


def test_full_mode_forwards_explicit_gpu_allocation():
    with fedcore_import_context():
        import run_demo

        args = run_demo.define_parser().parse_args(["--mode", "full", "--gpu", "[3],[4],[5]"])
        command = run_demo._build_predictor_command(args, Path("/repo/qwen"), Path("/tmp/data"), Path("/tmp/work"))

    assert command[command.index("--gpu") + 1] == "[3],[4],[5]"
