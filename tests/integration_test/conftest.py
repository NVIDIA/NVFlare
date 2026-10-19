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

"""Apply CI's gRPC launch defaults before importing integration test modules."""

import os

# Direct pytest runs launch trainer subprocesses after starting Cell threads.
# Match ci/run_integration.sh so gRPC's fork hooks do not stall those launches.
# Preserve explicit settings for tests that deliberately exercise another mode.
os.environ.setdefault("GRPC_POLL_STRATEGY", "poll")
os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "False")
