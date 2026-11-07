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

"""Compose readiness includes this startup's site preparation and FLARE registration."""
import json
import os
import socket

from common import WORKSPACE


def main():
    state = json.loads((WORKSPACE / ".startup.json").read_text())
    os.kill(state["pid"], 0)
    role = os.environ["FLARE_ROLE"]
    if role == "server":
        with socket.create_connection(("127.0.0.1", 8002), timeout=2):
            pass
    else:
        with (WORKSPACE / "log.txt").open() as stream:
            stream.seek(state["log_offset"])
            if f"Successfully registered client:{role}" not in stream.read():
                raise RuntimeError("site not registered yet")


if __name__ == "__main__":
    main()
