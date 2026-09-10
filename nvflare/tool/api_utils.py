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
import threading
import time
from typing import List, Optional, Tuple

from nvflare.fuel.flare_api.api_spec import JobNotFound, NoConnection, TargetType
from nvflare.fuel.flare_api.flare_api import Session


class SystemStartTimeout(RuntimeError):
    pass


def _client_names(client_info: list) -> List[str]:
    return [name for name in (getattr(client, "name", None) for client in client_info) if name]


def _format_ready_clients(client_names: List[str], ready_count: int, expected_count: int) -> str:
    names = f" ({', '.join(client_names)})" if client_names else ""
    return f"Clients ready: {ready_count}/{expected_count}{names}"


def _close_session_before_deadline(sess: Session, deadline: float) -> Tuple[Optional[str], bool]:
    """Close a readiness-probe session without exceeding its overall deadline."""
    close_errors = []
    close_finished = threading.Event()

    def _close():
        try:
            sess.close()
        except Exception as e:
            close_errors.append(str(e))
        finally:
            close_finished.set()

    close_thread = threading.Thread(target=_close, name="poc-readiness-session-close", daemon=True)
    close_thread.start()
    remaining = max(deadline - time.monotonic(), 0.0)
    if not close_finished.wait(remaining):
        return "admin session cleanup did not finish before the readiness deadline", True
    return (close_errors[0] if close_errors else None), False


def shutdown_system(
    prod_dir: str,
    username: str = "admin@nvidia.com",
    secure_mode: bool = True,
    timeout_in_sec: int = 30,
    wait: bool = True,
    verbose: bool = True,
) -> dict:
    from nvflare.tool.cli_output import print_human

    admin_user_dir = os.path.join(prod_dir, username)
    if verbose:
        print_human("connect to nvflare server")
    sess = None
    conn_timeout = 10
    try:
        sess = Session(username=username, startup_path=admin_user_dir, secure_mode=secure_mode)
        sess.try_connect(conn_timeout)
        return shutdown_system_by_session(sess=sess, timeout_in_sec=timeout_in_sec, wait=wait, verbose=verbose)
    except NoConnection:
        # system is already shutdown
        return {"server_reachable": False, "already_stopped": True, "active_job_ids": [], "wait": wait}
    finally:
        if sess:
            sess.close()


def shutdown_system_by_session(
    sess: Session, timeout_in_sec: int = 20, wait: bool = True, verbose: bool = True
) -> dict:
    from nvflare.tool.cli_output import print_human

    if verbose:
        print_human("checking running jobs")
    jobs = sess.list_jobs()
    active_job_ids = get_running_job_ids(jobs)
    if len(active_job_ids) > 0:
        if verbose:
            print_human("Warning: current running jobs will be aborted")
        abort_jobs(sess, active_job_ids)
    if wait:
        if verbose:
            print_human("shutdown NVFLARE and wait for completion")
    else:
        if verbose:
            print_human("shutdown NVFLARE")
    sess.shutdown(TargetType.ALL, wait=wait, timeout=timeout_in_sec)
    return {
        "server_reachable": True,
        "already_stopped": False,
        "active_job_ids": active_job_ids,
        "active_jobs_aborted": bool(active_job_ids),
        "wait": wait,
    }


def get_running_job_ids(jobs: list) -> List[str]:
    running_job_ids = []
    for job in jobs or []:
        if not isinstance(job, dict) or job.get("status") != "RUNNING":
            continue
        job_id = job.get("job_id") or job.get("id")
        if job_id:
            running_job_ids.append(job_id)
    return running_job_ids


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
    deadline = time.monotonic() + timeout_in_sec
    admin_user_dir = os.path.join(prod_dir, username)
    last_error = None
    while flare_not_ready and time.monotonic() < deadline:
        sess = None
        ready_sys_info = None
        cleanup_timed_out = False
        try:
            sess = Session(username=username, startup_path=admin_user_dir, secure_mode=secure_mode)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("readiness deadline reached before connecting to the admin server")
            sess.try_connect(remaining, connect_timeout=conn_timeout)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("readiness deadline reached before checking system status")
            sess.api.set_command_timeout(remaining)
            sys_info = sess.get_system_info()
            if time.monotonic() >= deadline:
                raise TimeoutError("readiness deadline reached while checking system status")
            client_names = _client_names(sys_info.client_info)
            ready_count = len(sys_info.client_info)
            expected_count = len(expected_client_set) if expected_client_set else num_clients
            if expected_client_set:
                registered_clients = set(client_names)
                missing_clients = sorted(expected_client_set - registered_clients)
                flare_not_ready = bool(missing_clients)
            else:
                missing_clients = []
                flare_not_ready = ready_count < num_clients
            if flare_not_ready:
                if missing_clients:
                    last_error = f"waiting for clients: {', '.join(missing_clients)}"
                    print_human(
                        f"Waiting for clients: {', '.join(missing_clients)} ({ready_count}/{expected_count} ready)"
                    )
                else:
                    last_error = f"{ready_count} of {num_clients} clients registered"
                    print_human(f"Waiting for clients: {ready_count}/{expected_count} ready")
            else:
                ready_sys_info = sys_info
        except NoConnection:
            # server is not up yet
            last_error = "server is not reachable"
        except Exception as e:
            last_error = str(e)
        finally:
            if sess:
                close_error, cleanup_timed_out = _close_session_before_deadline(sess, deadline)
                if close_error:
                    last_error = f"{last_error}; {close_error}" if last_error else close_error

        if ready_sys_info is not None:
            print_human(_format_ready_clients(client_names, ready_count, expected_count))
            print_human("\nReady to go.")
            return ready_sys_info
        if cleanup_timed_out:
            break

        remaining = deadline - time.monotonic()
        if flare_not_ready and remaining > 0:
            time.sleep(min(poll_interval, remaining))

    detail = f"; last error: {last_error}" if last_error else ""
    client_target = (
        f"expected clients {', '.join(sorted(expected_client_set))}"
        if expected_client_set
        else f"{num_clients} clients"
    )
    raise SystemStartTimeout(f"cannot connect to server with {client_target} within {timeout_in_sec} sec{detail}")
