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

import logging
import os
import re
import time
from typing import Dict

from nvflare.apis.job_def import RunStatus
from nvflare.fuel.hci.client.api_status import APIStatus
from nvflare.fuel.hci.client.fl_admin_api import FLAdminAPI
from nvflare.fuel.hci.client.fl_admin_api_constants import FLDetailKey
from nvflare.fuel.hci.client.fl_admin_api_spec import TargetType
from nvflare.ha.dummy_overseer_agent import DummyOverseerAgent
from nvflare.ha.overseer_agent import HttpOverseerAgent
from tests.integration_test.utils import run_admin_api_tests


def process_logs(logs, run_state):
    # regex to extract run_state from logs

    prev_run_state = run_state.copy()

    # matches the latest instance of "wf={workflow}," or "wf={workflow}]"
    match = re.search("(wf=)([^,\\]]+)(,|\\])(?!.*(wf=)([^,\\]]+)(,|\\]))", logs)
    if match:
        run_state["workflow"] = match.group(2)

    # matches the latest instance of "task_name={validate}, or "task_name={validate}"
    match = re.search("(task_name=)([^,\\]]+)(,|\\])(?!.*(task_name=)([^,\\]]+)(,|\\]))", logs)
    if match:
        run_state["task"] = match.group(2)

    # matches the latest instance of "Round {0-999} started."
    match = re.search(
        "Round ([0-9]|[1-9][0-9]|[1-9][0-9][0-9]) started\\.(?!.*Round ([0-9]|[1-9][0-9]|[1-9][0-9][0-9]) started\\.)",
        logs,
    )
    if match:
        run_state["round_number"] = int(match.group(1))

    return run_state != prev_run_state, run_state


def _parse_job_run_statuses(list_jobs_string: str) -> Dict[str, str]:
    """Parse the list_jobs string to return job run status."""
    job_statuses = {}
    # first three is table start and header row, last one is table end
    rows = list_jobs_string.splitlines()[3:-1]
    for r in rows:
        segments = [s.strip() for s in r.split("|")]
        job_statuses[segments[1]] = segments[3]
    return job_statuses


def _parse_workflow_states(stats_message: dict):
    workflow_states = {}
    if not stats_message:
        return workflow_states
    for k, v in stats_message.items():
        # each controller inherit from nvflare/apis/impl/controller has tasks
        if v.get("tasks"):
            workflow_states[k] = v.copy()
            workflow_states[k].pop("tasks")
    return workflow_states


def _print_state(state: dict, length: int = 30):
    print("\n" + "-" * length)
    for k, v in state.items():
        print(f"{k}: {v}")
    print("-" * length + "\n", flush=True)


def _update_run_state(stats: dict, run_state: dict, job_run_status: str):
    # extract run_state from stats
    # {'status': <APIStatus.SUCCESS: 'SUCCESS'>,
    #        'details': {
    #           'message': {
    #               'ScatterAndGather': {
    #                   'tasks': {'train': []},
    #                   'phase': 'train',
    #                   'current_round': 0,
    #                   'num_rounds': 2},
    #               'CrossSiteModelEval':
    #                   {'tasks': {}},
    #               'ServerRunner': {
    #                   'job_id': XXX,
    #                   'status': 'started',
    #                   'workflow': 'scatter_and_gather'
    #                }
    #            }
    #       },
    # 'raw': {'time': '2022-04-04 15:13:09.367350', 'data': [{'type': 'dict', 'data': {'ScatterAndGather': {'tasks': {'train': []}, 'phase': 'train', 'current_round': 0, 'num_rounds': 2}, 'CrossSiteModelEval': {'tasks': {}}, 'ServerRunner': {'job_id': XXX, 'status': 'started', 'workflow': 'scatter_and_gather'}}}], 'status': <APIStatus.SUCCESS: 'SUCCESS'>}}
    prev_run_state = run_state.copy()

    # parse stats
    if (
        stats
        and "status" in stats
        and stats["status"] == APIStatus.SUCCESS
        and "details" in stats
        and "message" in stats["details"]
    ):
        run_state["workflows"] = _parse_workflow_states(stats_message=stats["details"]["message"])

    # parse job status
    run_state["run_finished"] = job_run_status == RunStatus.FINISHED_COMPLETED.value

    print("Prev state:")
    _print_state(state=prev_run_state)
    print(f"STATS: {stats}", flush=True)
    print("Current state:")
    _print_state(state=run_state)

    return run_state != prev_run_state, run_state


def _check_dict_b_value_same_as_dict_a_for_keys_in_dict_a(dict_a: dict, dict_b: dict):
    if not dict_a and not dict_b:
        return True
    if dict_a and not dict_b:
        return False
    for k in dict_a:
        if isinstance(dict_a[k], dict):
            if not _check_dict_b_value_same_as_dict_a_for_keys_in_dict_a(dict_a[k], dict_b[k]):
                return False
        elif dict_b.get(k) != dict_a[k]:
            return False
    return True


def _check_run_state(state, expected_state):
    for k, v in expected_state.items():
        print(f"ASSERT Expected State {k}: {v} is part of Current State {k}: {state[k]}")
        if isinstance(v, dict):
            assert _check_dict_b_value_same_as_dict_a_for_keys_in_dict_a(v, state[k])
        else:
            assert state[k] == v
    print("\n")


def _check_event_trigger(event_trigger: dict, run_state: dict):
    """check if a run state trigger an event trigger."""
    event_triggered = _check_dict_b_value_same_as_dict_a_for_keys_in_dict_a(event_trigger, run_state)
    return event_triggered


class AdminController:
    ADMIN_USER_NAME = "admin"

    def __init__(self, upload_root_dir: str, download_root_dir: str, ha: bool, poll_period=1):
        """

        Args:
            upload_root_dir: the root dir to look for folders to upload
            download_root_dir: the root dir to download things to
        """
        self.poll_period = poll_period

        if ha:
            overseer_agent = HttpOverseerAgent(
                role="admin", overseer_end_point="http://127.0.0.1:5000/api/v1", project="example_project", name="admin"
            )
        else:
            overseer_agent = DummyOverseerAgent(sp_end_point="localhost:8002:8003")

        self.download_root_dir = download_root_dir
        self.admin_api = FLAdminAPI(
            upload_dir=upload_root_dir,
            download_dir=download_root_dir,
            overseer_agent=overseer_agent,
            poc=True,
            debug=False,
            user_name=AdminController.ADMIN_USER_NAME,
        )

        self.logger = logging.getLogger(self.__class__.__name__)
        self.job_id = None
        self.last_job_name = ""

    def initialize(self):
        success = False

        try:
            response = None
            timeout = 100
            start_time = time.time()
            while time.time() - start_time <= timeout:
                response = self.admin_api.login_with_poc(username="admin", poc_key="admin")
                if response["status"] == APIStatus.SUCCESS:
                    success = True
                    break
                time.sleep(1.0)

            if not success:
                details = response.get("details") if response else "No details"
                raise ValueError(f"Login to admin api failed: {details}")
            else:
                print("Admin successfully logged into server.")
        except Exception as e:
            print(f"Exception in logging in to admin: {e.__str__()}")
        return success

    def get_job_result(self):
        command_name = "download_job"
        response = self.admin_api.do_command(f"{command_name} {self.job_id}")
        if response["status"] != APIStatus.SUCCESS:
            raise RuntimeError(f"{command_name} failed: {response}")
        run_data = {
            "job_id": self.job_id,
            "workspace_root": os.path.join(self.download_root_dir, self.job_id, "workspace"),
        }

        return run_data

    def ensure_clients_started(self, num_clients):
        timeout = 1000
        start_time = time.time()
        clients_up = False
        while not clients_up:
            if time.time() - start_time > timeout:
                raise ValueError(f"Clients could not be started in {timeout} seconds.")

            time.sleep(0.5)
            response = self.admin_api.check_status(target_type=TargetType.CLIENT)
            if response["status"] == APIStatus.SUCCESS:
                # print(f"check client status response {response}")
                if "details" not in response:  # clients not ready....
                    # client response would be: {'status': <APIStatus.SUCCESS: 'SUCCESS'>, 'raw': {'time': '2021-10-29 00:09:06.220615', 'data': [{'type': 'error', 'data': 'no clients available'}], 'status': <APIStatus.SUCCESS: 'SUCCESS'>}}
                    # How can the user know if clients are ready or not.....
                    continue
                for row in response["details"]["client_statuses"][1:]:
                    if row[3] != "not started":
                        continue
                # wait for all clients to come up
                if len(response["details"]["client_statuses"]) < num_clients + 1:
                    continue
                clients_up = True
                print("All clients are up.")
            time.sleep(1.0)

        return clients_up

    def server_status(self):
        response = self.admin_api.check_status(target_type=TargetType.SERVER)
        if response and "status" in response and response["status"] == APIStatus.SUCCESS and "details" in response:
            return response["details"]
        return None

    def client_status(self):
        response = self.admin_api.check_status(target_type=TargetType.CLIENT)
        if response and "status" in response and response["status"] == APIStatus.SUCCESS and "details" in response:
            return response["details"]
        return None

    def submit_job(self, job_name) -> bool:
        response = self.admin_api.submit_job(job_name)
        if response["status"] != APIStatus.SUCCESS:
            raise RuntimeError(f"submit_job failed: {response}")
        self.job_id = response["details"]["job_id"]
        self.last_job_name = job_name
        return True

    def _check_job_done(self):
        response = self.admin_api.check_status(target_type=TargetType.SERVER)
        if response and "status" in response:
            if response["status"] != APIStatus.SUCCESS:
                print(f"Check server status failed: {response}.")
                return False
            else:
                if "details" not in response:
                    print(f"Check server status missing details: {response}.")
                    return False
                else:
                    # check if run is stopped
                    if (
                        FLDetailKey.SERVER_ENGINE_STATUS in response["details"]
                        and response["details"][FLDetailKey.SERVER_ENGINE_STATUS] == "stopped"
                    ):
                        response = self.admin_api.check_status(target_type=TargetType.CLIENT)
                        if response["status"] != APIStatus.SUCCESS:
                            print(f"CHECK client status failed: {response}")
                            return False
                        if "details" not in response:
                            print(f"Check client status missing details: {response}.")
                            return False
                        else:
                            job_run_statuses = self._get_job_run_statuses()
                            for row in response["details"]["client_statuses"]:
                                if row[3] != "stopped":
                                    continue
                            # check if the current job is completed
                            if job_run_statuses[self.job_id] in (
                                RunStatus.FINISHED_COMPLETED.value,
                                RunStatus.FINISHED_ABORTED.value,
                            ):
                                return True
        return False

    def wait_for_job_done(self):
        # TODO:: Is it possible to get the training log after training is done?
        training_done = False
        while not training_done:
            training_done = self._check_job_done()
            time.sleep(self.poll_period)

    def _get_stats(self, target):
        return self.admin_api.show_stats(self.job_id, target)

    def _get_job_log(self, target):
        job_log_file = os.path.join(self.job_id, "log.txt")
        logs = self.admin_api.cat_target(target, file=job_log_file)["details"]["message"].splitlines()
        return logs

    def _get_job_run_statuses(self):
        list_jobs_result = self.admin_api.list_jobs()
        if list_jobs_result["status"] == APIStatus.SUCCESS:
            list_jobs_string = list_jobs_result["details"]["message"]
            job_run_statuses = _parse_job_run_statuses(list_jobs_string)
            return job_run_statuses
        else:
            return {self.job_id: "List Job command failed"}

    def run_event_sequence(self, site_launcher, event_sequence):
        run_state = {"run_finished": None, "workflows": None}

        last_read_line = 0
        event_idx = 0
        ha_events = event_sequence["events"]
        event_test_status = [False for _ in range(len(ha_events))]  # whether event has been successfully triggered

        training_done = False
        while not training_done:
            server_logs = self._get_job_log(target=TargetType.SERVER)
            server_logs = server_logs[last_read_line:]
            last_read_line += len(server_logs)
            server_logs_string = "\n".join(server_logs)

            print("\n")
            print(f"Job name: {self.last_job_name}")
            print(f"HA test: {event_sequence['description']}")

            # update run_state
            job_run_statuses = self._get_job_run_statuses()
            print(f"job run status: {job_run_statuses}")
            stats = self._get_stats(TargetType.SERVER)
            changed, run_state = _update_run_state(
                stats=stats, run_state=run_state, job_run_status=job_run_statuses[self.job_id]
            )

            # check if event is triggered -> then execute the corresponding actions
            if event_idx < len(ha_events) and not event_test_status[event_idx]:
                event_trigger = ha_events[event_idx]["trigger"]
                event_triggered = False

                if isinstance(event_trigger, dict):
                    event_triggered = _check_event_trigger(event_trigger, run_state)
                elif isinstance(event_trigger, str):
                    if event_trigger in server_logs_string:
                        event_triggered = True
                else:
                    raise RuntimeError(f"event_trigger type {type(event_trigger)} is not supported.")

                if event_triggered:
                    print(f"EVENT TRIGGER '{event_trigger}' is TRIGGERED.")
                    event_test_status[event_idx] = True
                    self.execute_actions(site_launcher, ha_events[event_idx]["actions"])
                    continue

            # check result state only when server is up and running
            if self.server_status():
                # compare run_state to expected result_state from the test case
                if event_idx < len(ha_events) and event_test_status[event_idx]:
                    result_state = ha_events[event_idx]["result_state"]
                    if any(list(run_state.values())):
                        _check_run_state(state=run_state, expected_state=result_state)
                        event_idx += 1

            training_done = self._check_job_done()
            time.sleep(self.poll_period)

        assert all(event_test_status), "Test failed: not all test events were triggered"

    def execute_actions(self, site_launcher, actions):
        for action in actions:
            tokens = action.split(" ")
            command = tokens[0]
            args = tokens[1:]

            print(f"ACTION: {action}")

            if command == "sleep":
                time.sleep(int(args[0]))

            elif command == "kill":
                if args[0] == "server":
                    if len(args) == 2:
                        # if server id is provided
                        server_id = int(args[1])
                    else:
                        server_id = site_launcher.get_active_server_id(self.admin_api.port)
                    self.admin_api.logout()
                    site_launcher.stop_server(server_id)
                elif args[0] == "overseer":
                    site_launcher.stop_overseer()
                elif args[0] == "client":  # TODO fix client kill & restart during run
                    if len(args) == 2:
                        # if client id is provided
                        client_ids = [int(args[1])]
                    else:
                        # close all clients
                        client_ids = list(site_launcher.client_properties.keys())
                    for cid in client_ids:
                        self.admin_api.remove_client([site_launcher.client_properties[cid]["name"]])
                        site_launcher.stop_client(cid)

            elif command == "start":
                if args[0] == "server":
                    if len(args) == 2:
                        # if server id is provided
                        server_ids = [int(args[1])]
                    else:
                        # start all servers
                        server_ids = list(site_launcher.server_properties.keys())
                    for sid in server_ids:
                        site_launcher.start_server(sid)
                    self.admin_api.login(username=AdminController.ADMIN_USER_NAME)
                elif args[0] == "overseer":
                    site_launcher.start_overseer()
                elif args[0] == "client":  # TODO fix client kill & restart during run
                    if len(args) == 2:
                        # if client id is provided
                        client_ids = [int(args[1])]
                    else:
                        # start all clients
                        client_ids = list(site_launcher.client_properties.keys())
                    for cid in client_ids:
                        site_launcher.start_client(cid)
            elif command == "test":
                if args[0] == "admin_commands":
                    run_admin_api_tests(self.admin_api)
            else:
                raise RuntimeError(f"Command {command} is not supported.")

    def finalize(self):
        if self.job_id:
            self.admin_api.abort_job(self.job_id)
        self.admin_api.overseer_agent.end()
        self.admin_api.shutdown(target_type=TargetType.ALL)
        time.sleep(10)
