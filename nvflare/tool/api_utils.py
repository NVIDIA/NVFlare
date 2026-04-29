# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import time
from typing import List, Optional

from nvflare.fuel.flare_api.api_spec import JobNotFound, NoConnection, TargetType
from nvflare.fuel.flare_api.flare_api import Session


class SystemStartTimeout(RuntimeError):
    pass


def shutdown_system(
    prod_dir: str,
    username: str = "admin@nvidia.com",
    secure_mode: bool = True,
    timeout_in_sec: int = 30,
    wait: bool = True,
):
    from nvflare.tool.cli_output import print_human

    admin_user_dir = os.path.join(prod_dir, username)
    print_human("connect to nvflare server")
    sess = None
    conn_timeout = 10
    try:
        sess = Session(username=username, startup_path=admin_user_dir, secure_mode=secure_mode)
        sess.try_connect(conn_timeout)
        shutdown_system_by_session(sess=sess, timeout_in_sec=timeout_in_sec, wait=wait)
    except NoConnection:
        # system is already shutdown
        return
    finally:
        if sess:
            sess.close()


def shutdown_system_by_session(sess: Session, timeout_in_sec: int = 20, wait: bool = True):
    from nvflare.tool.cli_output import print_human

    print_human("checking running jobs")
    jobs = sess.list_jobs()
    active_job_ids = get_running_job_ids(jobs)
    if len(active_job_ids) > 0:
        print_human("Warning: current running jobs will be aborted")
        abort_jobs(sess, active_job_ids)
    if wait:
        print_human("shutdown NVFLARE and wait for completion")
    else:
        print_human("shutdown NVFLARE")
    sess.shutdown(TargetType.ALL, wait=wait, timeout=timeout_in_sec)


def get_running_job_ids(jobs: list) -> List[str]:
    if len(jobs) > 0:
        running_job_ids = [job for job in jobs if job["status"] == "RUNNING"]
        return running_job_ids
    else:
        return []


def abort_jobs(sess, job_ids):
    for job_id in job_ids:
        try:
            sess.abort_job(job_id)
        except JobNotFound:
            # ignore invalid job id
            pass


def wait_for_system_start(
    num_clients: int,
    prod_dir: str,
    username: str = "admin",
    secure_mode: bool = False,
    second_to_wait: int = 10,
    timeout_in_sec: int = 30,
    poll_interval: float = 2.0,
    conn_timeout: float = 10.0,
    expected_clients: Optional[List[str]] = None,
):
    from nvflare.tool.cli_output import print_human

    if second_to_wait > 0:
        print_human(f"wait for {second_to_wait} seconds before FL system is up")
        time.sleep(second_to_wait)
    # just in case try to connect before server started
    flare_not_ready = True
    expected_client_set = set(expected_clients or [])
    start = time.time()
    deadline = start + timeout_in_sec
    admin_user_dir = os.path.join(prod_dir, username)
    last_error = None
    while flare_not_ready and time.time() < deadline:
        print_human("trying to connect to server")
        sess = None
        try:
            sess = Session(username=username, startup_path=admin_user_dir, secure_mode=secure_mode)
            remaining = max(deadline - time.time(), 0.1)
            sess.try_connect(min(conn_timeout, remaining))
            sys_info = sess.get_system_info()
            print_human(f"Server info:\n{sys_info.server_info}")
            print_human("\nClient info")
            for client in sys_info.client_info:
                print_human(client)
            if expected_client_set:
                registered_clients = {getattr(client, "name", None) for client in sys_info.client_info}
                missing_clients = sorted(expected_client_set - registered_clients)
                flare_not_ready = bool(missing_clients)
            else:
                missing_clients = []
                flare_not_ready = len(sys_info.client_info) < num_clients
            if flare_not_ready:
                if missing_clients:
                    last_error = f"waiting for clients: {', '.join(missing_clients)}"
                else:
                    last_error = f"{len(sys_info.client_info)} of {num_clients} clients registered"
            else:
                print_human("ready to go")
                return sys_info
        except NoConnection:
            # server is not up yet
            last_error = "server is not reachable"
        except Exception as e:
            last_error = str(e)
            print_human("failure", e)
        finally:
            if sess:
                try:
                    sess.close()
                except Exception as e:
                    print_human("failure", e)

        remaining = deadline - time.time()
        if flare_not_ready and remaining > 0:
            time.sleep(min(poll_interval, remaining))

    detail = f"; last error: {last_error}" if last_error else ""
    raise SystemStartTimeout(f"cannot connect to server with {num_clients} clients within {timeout_in_sec} sec{detail}")
