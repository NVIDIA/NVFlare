# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

import os
import uuid
from datetime import datetime, timedelta

from nvflare.lighter.utils import load_yaml

OVERSEER_STORE = os.environ.get("OVERSEER_STORE")

if OVERSEER_STORE == "REDIS":
    from .redis_store import do_refresh, get_all_sp, get_primary_sp, get_sp_by, update_sp
elif OVERSEER_STORE == "SQL":
    from .sql_store import do_refresh, get_all_sp, get_primary_sp, get_sp_by, update_sp
elif OVERSEER_STORE == "MEM":
    from .mem_store import do_refresh, get_all_sp, get_primary_sp, get_sp_by, update_sp
else:
    print("Using default STORE (MEM)")
    from .mem_store import do_refresh, get_all_sp, get_primary_sp, get_sp_by, update_sp  # noqa


def load_privilege():
    privilege_file = os.environ.get("AUTHZ_FILE", "privilege.yml")
    try:
        privilege = load_yaml(privilege_file)
    except:
        privilege = dict()
    return privilege


def update_sp_state(project, now, heartbeat_timeout=10):
    valid_starting = now - timedelta(seconds=heartbeat_timeout)
    # mark all late SP as offline and not primary
    # print(f"{now=} {valid_starting=}")
    for sp in get_all_sp(project):
        if datetime.fromisoformat(sp["last_heartbeat"]) < valid_starting:
            sp["state"] = "offline"
            sp["primary"] = False
        else:
            sp["state"] = "online"
        update_sp(sp)


def simple_PSP_policy(incoming_sp, now):
    """Find the primary SP (PSP).

    If there is no PSP or current PSP timeout, choose one without heartbeat timeout.
    """
    project = incoming_sp["project"]
    sp = get_sp_by(dict(project=project, sp_end_point=incoming_sp["sp_end_point"]))
    if sp:
        sp["last_heartbeat"] = now.isoformat()
        update_sp(sp)
    else:
        update_sp(
            dict(
                project=incoming_sp["project"],
                sp_end_point=incoming_sp["sp_end_point"],
                last_heartbeat=now.isoformat(),
                state="online",
                primary=False,
            )
        )

    psp = get_primary_sp(project)
    if not psp:
        psp = get_sp_by(dict(project=project, state="online"))
        if psp:
            print(f"{psp['sp_end_point']} online")
            psp["primary"] = True
            psp["service_session_id"] = str(uuid.uuid4())
            update_sp(psp)

    return psp


def promote_sp(sp):
    psp = get_sp_by(sp)
    project = sp["project"]
    sp_end_point = sp["sp_end_point"]
    if psp and psp["state"] == "online":
        current_psp = get_primary_sp(project)
        if all(current_psp[k] == v for k, v in sp.items()):
            return True, f"Same sp_end_point, no need to promote {sp_end_point}."
        psp["primary"] = True
        current_psp["primary"] = False
        psp["service_session_id"] = str(uuid.uuid4())
        print(f"{psp['sp_end_point']} promoted")
        print(f"{current_psp['sp_end_point']} demoted")
        update_sp(psp)
        update_sp(current_psp)
        return False, psp
    else:
        return True, f"Unable to promote {sp_end_point}, either offline or not registered."
