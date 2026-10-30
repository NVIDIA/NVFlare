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

"""Process targets and cleanup for resource-statistics CPU integration tests."""

import subprocess
from decimal import Decimal
from types import SimpleNamespace

from nvflare.private.fed.resource_stats import cpu_consumed as cpu_module
from nvflare.private.fed.resource_stats.cpu_consumed import CpuConsumedAccountant


def cleanup_subprocess(process):
    """Reap one owned interpreter, including an assertion or startup failure."""
    if process.poll() is None:
        process.terminate()
    try:
        process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate(timeout=5)


def exercise_forked_cpu_hooks(connection):
    try:
        inherited_discarded = cpu_module._active is None
        cpu_module.mark_unwaited_managed_children()
        cpu_module.register_managed_child(SimpleNamespace(poll=lambda: 0))
        fresh = CpuConsumedAccountant(usage_reader=lambda: (Decimal(0), Decimal(0)))
        cpu_module.install_cpu_accountant(fresh)
        cpu_module.register_managed_child(SimpleNamespace(poll=lambda: 0))
        connection.send((inherited_discarded, cpu_module._active is fresh, fresh.finish()))
    finally:
        connection.close()
