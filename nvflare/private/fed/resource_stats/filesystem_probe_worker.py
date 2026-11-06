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

"""Run filesystem syscalls away from the job's finalization thread."""

import json
import os
import select
import sys
import threading

from .collector import observe_retained_content, observe_workspace_filesystem


def _watch_owner(parent_pid):
    # Watch the owning process, since a forked descendant can inherit stdin's
    # writer. Unlike PDEATHSIG, this does not follow the creating thread.
    try:
        while os.getppid() == parent_pid:
            if select.select([sys.stdin.fileno()], [], [], 0.1)[0]:
                break
    except (OSError, ValueError):
        pass
    finally:
        os._exit(0)


def main(run_dir, parent_pid):
    if os.getppid() != parent_pid:
        return
    threading.Thread(target=_watch_owner, args=(parent_pid,), daemon=True, name="filesystem-probe-owner").start()
    # Publish dimensions independently: a blocked scan cannot erase a completed
    # capacity observation, and no incomplete JSON frame is accepted by the owner.
    for name, observer in (
        ("workspace_filesystem", observe_workspace_filesystem),
        ("retained_content", observe_retained_content),
    ):
        try:
            value = observer(run_dir)
        except Exception:
            value = {"status": "unavailable", "issues": ["observation_incomplete"]}
        print(json.dumps({name: value}, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
