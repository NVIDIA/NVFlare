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
import math
import os
import threading
import time
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import List, Optional

from nvflare.fuel.common.excepts import ConfigError
from nvflare.fuel.flare_api.api_spec import JobNotFound, NoConnection, TargetType
from nvflare.fuel.flare_api.flare_api import Session
from nvflare.fuel.hci.client.api import AdminCertAcquisitionError


class SystemStartTimeout(RuntimeError):
    pass


def _client_names(client_info: list) -> List[str]:
    return [name for name in (getattr(client, "name", None) for client in client_info) if name]


def _format_ready_clients(client_names: List[str], ready_count: int, expected_count: int) -> str:
    names = f" ({', '.join(client_names)})" if client_names else ""
    return f"Clients ready: {ready_count}/{expected_count}{names}"


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
    """Wait for readiness for up to timeout_in_sec seconds after second_to_wait.

    conn_timeout is passed to transport authentication (default ten seconds).
    The caller's wait covers session creation, login, status requests and cleanup,
    even when an underlying idle timeout is extended by network progress. One
    daemon worker owns the session; it is not forcibly cancelled on timeout and
    closes the session when the pending operation returns. Confirmed readiness
    is returned without waiting for cleanup. No general Session API is changed.
    """
    from nvflare.tool.cli_output import print_human

    # Reject caller errors before sleeping, creating a session, or retrying.
    for name, value, allow_zero in (
        ("timeout_in_sec", timeout_in_sec, False),
        ("conn_timeout", conn_timeout, False),
        ("poll_interval", poll_interval, True),
        ("second_to_wait", second_to_wait, True),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            or (value == 0 and not allow_zero)
        ):
            required = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be a finite {required} number of seconds")

    if second_to_wait > 0:
        print_human(f"wait for {second_to_wait} seconds before FL system is up")
        time.sleep(second_to_wait)
    deadline = time.monotonic() + timeout_in_sec
    outcome = Future()
    stopped = threading.Event()
    expected_client_set = set(expected_clients or [])
    expected_count = len(expected_client_set) if expected_client_set else num_clients

    last_error = None

    def remaining_time():
        return 0.0 if stopped.is_set() else max(0.0, deadline - time.monotonic())

    def probe():
        nonlocal last_error
        # Keep the whole session lifecycle in one worker: transport and streamed
        # requests use idle timeouts and cannot enforce a total elapsed limit.
        while remaining_time() > 0:
            sess = None
            try:
                print_human(
                    f"Connecting and logging in to the admin server "
                    f"(up to {remaining_time():.1f} seconds remaining)..."
                )
                try:
                    sess = Session(
                        username=username, startup_path=os.path.join(prod_dir, username), secure_mode=secure_mode
                    )
                except AdminCertAcquisitionError:
                    # Initial certificate issuance can perform provider I/O here.
                    raise
                except (AssertionError, ValueError, TypeError, OSError, RuntimeError, ConfigError) as e:
                    outcome.set_exception(ConfigError(f"Cannot initialize the admin session: {e}"))
                    return
                remaining = remaining_time()
                if remaining <= 0:
                    return
                sess.try_connect(min(conn_timeout, remaining))
                remaining = remaining_time()
                if remaining <= 0:
                    return
                sess.api.set_command_timeout(remaining)
                sys_info = sess.get_system_info()
                if remaining_time() <= 0:
                    return
                client_names = _client_names(sys_info.client_info)
                ready_count = len(sys_info.client_info)
                missing = sorted(expected_client_set - set(client_names))
                if not missing and (expected_client_set or ready_count >= num_clients):
                    outcome.set_result(
                        (time.monotonic(), sys_info, _format_ready_clients(client_names, ready_count, expected_count))
                    )
                    return
                waiting = (
                    f"Waiting for clients: {', '.join(missing)} ({ready_count}/{expected_count} ready)"
                    if missing
                    else f"Waiting for clients: {ready_count}/{expected_count} ready"
                )
                last_error = waiting
                print_human(waiting)
            except Exception as e:
                last_error = str(e)
            except BaseException as e:
                outcome.set_exception(e)
                return
            finally:
                if sess is not None:
                    try:
                        sess.close()
                    except Exception as e:
                        if not stopped.is_set():
                            print_human(f"Warning: could not close the admin session: {e}")
            stopped.wait(min(poll_interval, remaining_time()))

    # ThreadPoolExecutor joins workers at interpreter exit, even with shutdown(wait=False).
    # A daemon worker lets the CLI exit if the underlying operation never returns.
    threading.Thread(target=probe, name="poc-readiness", daemon=True).start()
    try:
        try:
            observed_at, sys_info, ready_message = outcome.result(timeout=remaining_time())
        except FutureTimeoutError:
            pass
        else:
            if observed_at < deadline:
                print_human(ready_message)
                print_human("\nReady to go.")
                return sys_info
    finally:
        # Never start another attempt after the caller returns. A blocked worker
        # owns its session and closes it when the underlying operation returns.
        stopped.set()

    detail = f" Last observation: {last_error}" if last_error else ""
    client_target = (
        f"expected clients {', '.join(sorted(expected_client_set))}"
        if expected_client_set
        else f"{num_clients} clients"
    )
    raise SystemStartTimeout(
        f"Could not confirm that the server and {client_target} were ready within {timeout_in_sec} seconds.{detail}"
    )
