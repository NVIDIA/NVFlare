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
from typing import List

from nvflare.fuel.flare_api.api_spec import JobNotFound, NoConnection
from nvflare.fuel.flare_api.flare_api import Session


def shutdown_system(prod_dir: str, username: str = "admin", secure_mode: bool = False, timeout_in_sec: int = 30):
    admin_user_dir = os.path.join(prod_dir, username)
    print("connect to nvflare server")
    sess = None
    conn_timeout = 10
    try:
        sess = Session(username=username, startup_path=admin_user_dir, secure_mode=secure_mode)
        sess.try_connect(conn_timeout)
        shutdown_system_by_session(sess=sess, timeout_in_sec=timeout_in_sec)
    except NoConnection:
        # system is already shutdown
        return
    finally:
        if sess:
            sess.close()


def shutdown_system_by_session(sess: Session, timeout_in_sec: int = 20):

    print("checking running jobs")
    jobs = sess.list_jobs()
    active_job_ids = get_running_job_ids(jobs)
    if len(active_job_ids) > 0:
        print("Warning: current running jobs will be aborted")
        abort_jobs(sess, active_job_ids)
    print("shutdown NVFLARE")
    sess.api.do_command("shutdown all")
    wait_for_system_shutdown(sess, timeout_in_sec=timeout_in_sec)


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


def wait_for_system_shutdown(sess: Session, timeout_in_sec: int = 30):
    start = time.time()
    duration = 0
    cnt = 0
    status = None
    while (status is None or status == "started") and duration < timeout_in_sec:
        try:
            print("trying to connect to NVFLARE server")
            sys_info = sess.get_system_info()
            status = sys_info.server_info.status
            curr = time.time()
            duration = curr - start
            if cnt % 25 == 0:
                print("waiting system to shutdown")
            cnt += 1
            time.sleep(0.1)
        except Exception:
            # Server is already shutdown
            return


def wait_for_system_start(
    num_clients: int,
    prod_dir: str,
    username: str = "admin",
    secure_mode: bool = False,
    second_to_wait: int = 10,
    timeout_in_sec: int = 30,
):
    print(f"wait for {second_to_wait} seconds before FL system is up")
    time.sleep(second_to_wait)
    # just in case try to connect before server started
    flare_not_ready = True
    start = time.time()
    duration = 0
    admin_user_dir = os.path.join(prod_dir, username)
    conn_timeout = 10.0
    while flare_not_ready and duration < timeout_in_sec:
        print("trying to connect to server")
        sess = None
        try:
            sess = Session(username=username, startup_path=admin_user_dir, secure_mode=secure_mode)
            sess.try_connect(conn_timeout)
            sys_info = sess.get_system_info()
            print(f"Server info:\n{sys_info.server_info}")
            print("\nClient info")
            for client in sys_info.client_info:
                print(client)
            flare_not_ready = len(sys_info.client_info) < num_clients
            curr = time.time()
            duration = curr - start
            time.sleep(2)
        except NoConnection:
            # server is not up yet
            pass
        except Exception as e:
            print("failure", e)
        finally:
            if sess:
                sess.close()

    if flare_not_ready:
        raise RuntimeError("can't not connect to server within {timeout_in_sec} sec")
    else:
        print("ready to go")
