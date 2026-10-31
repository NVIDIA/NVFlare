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

"""NVFlare execution runtime (nvflare >= 2.8). All nvflare imports are lazy
and confined to this package — importing it without nvflare installed is fine;
running it is not (`pip install -e ".[nvflare]"`).

The server is a ModelController whose run() is the shared `runtime/server_session.py`
orchestration over an NVFlareCommunicator; the client is a Client-API script
looping over the shared `runtime/client_session.py`. Same strategies, same
checkpoints/resume, same partitioning as the local plane.
"""
