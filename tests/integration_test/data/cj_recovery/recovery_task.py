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

"""Repeated worker loss must replace CJ without closing its server assignment."""

import json
import os
import signal
from pathlib import Path

import nvflare.client as flare

flare.init()
model = flare.receive()
checkpoint = Path("recovery_checkpoint.json")
state = json.loads(checkpoint.read_text()) if checkpoint.exists() else {"value": 0, "runs": []}
state["runs"].append({"pid": os.getpid(), "cj_pid": os.getppid(), "job": flare.get_job_id()})
state["value"] += len(state["runs"])
pending = checkpoint.with_suffix(".tmp")
pending.write_text(json.dumps(state))
pending.replace(checkpoint)
if len(state["runs"]) <= model.meta["crashes"]:
    os.kill(os.getpid(), signal.SIGKILL)
flare.send(flare.FLModel(params=model.params, meta={"recovery_state": state}))
