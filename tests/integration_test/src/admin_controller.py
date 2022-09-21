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

import json
import logging
import os
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


def _parse_job_run_statuses(list_jobs_string: str) -> Dict[str, RunStatus]:
    """Parse the list_jobs string to return job run status."""
    job_statuses = {}
    # first three is table start and header row, last one is table end
    rows = list_jobs_string.splitlines()[3:-1]
    for r in rows:
        segments = [s.strip() for s in r.split("|")]
        job_statuses[segments[1]] = RunStatus(segments[3])
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
            assert _check_dict_b_value_same_as_dict_a_for_keys_in_dict_a(dict_a=v, dict_b=state[k])
        else:
            assert state[k] == v
    print("\n")


def _check_event_trigger(
    event_trigger,
    string_to_check: str = None,
    run_state: dict = None,
) -> bool:
    """check if a run state trigger an event trigger."""
    if isinstance(event_trigger, dict):
        if run_state is None:
            raise RuntimeError("Event trigger is of dict type but run_state is not provided.")
        return _check_dict_b_value_same_as_dict_a_for_keys_in_dict_a(event_trigger, run_state)
    elif isinstance(event_trigger, str):
        if string_to_check is None:
            raise RuntimeError("Event trigger is of str type but string_to_check is not provided.")
        return event_trigger in string_to_check
    else:
        raise RuntimeError(f"event_trigger type {type(event_trigger)} is not supported.")


def _get_admin_full_path(workspace_root, admin_user_name, file_name: str):
    return os.path.join(workspace_root, admin_user_name, "startup", file_name)


class AdminController:
    def __init__(self, download_root_dir: str, poll_period=5):
        """

        Args:
            download_root_dir: the root dir to download things to
            poll_period (int): note that this value can't be too small,
            otherwise will have resource issue
        """
        self.poll_period = poll_period

        self.download_root_dir = download_root_dir
        self.admin_api = None
        self.admin_user_name = ""
        self.admin_api_response = None

        self.logger = logging.getLogger(self.__class__.__name__)
        self.test_done = False
        self.job_id = None
        self.last_job_name = ""

    def initialize(self, workspace_root_dir: str, upload_root_dir: str, ha: bool):
        success = False
        self.admin_user_name = "admin@nvidia.com" if ha else "admin"
        admin_json_file = os.path.join(workspace_root_dir, self.admin_user_name, "startup", "fed_admin.json")
        if not os.path.exists(admin_json_file):
            raise RuntimeError("Missing admin json file.")
        with open(admin_json_file, "r") as f:
            admin_json = json.load(f)

        if ha:
            admin_config = admin_json["admin"]
            overseer_agent = HttpOverseerAgent(**admin_json["admin"]["overseer_agent"]["args"])
            self.admin_api = FLAdminAPI(
                upload_dir=upload_root_dir,
                download_dir=self.download_root_dir,
                overseer_agent=overseer_agent,
                poc=False,
                debug=False,
                user_name=self.admin_user_name,
                ca_cert=_get_admin_full_path(workspace_root_dir, self.admin_user_name, admin_config["ca_cert"]),
                client_key=_get_admin_full_path(workspace_root_dir, self.admin_user_name, admin_config["client_key"]),
                client_cert=_get_admin_full_path(workspace_root_dir, self.admin_user_name, admin_config["client_cert"]),
            )
        else:
            overseer_agent = DummyOverseerAgent(**admin_json["admin"]["overseer_agent"]["args"])
            self.admin_api = FLAdminAPI(
                upload_dir=upload_root_dir,
                download_dir=self.download_root_dir,
                overseer_agent=overseer_agent,
                poc=True,
                debug=False,
                user_name=self.admin_user_name,
            )

        try:
            response = None
            timeout = 30
            start_time = time.time()
            while time.time() - start_time <= timeout:
                if ha:
                    response = self.admin_api.login(username=self.admin_user_name)
                else:
                    response = self.admin_api.login_with_poc(username=self.admin_user_name, poc_key="admin")
                if response["status"] == APIStatus.SUCCESS:
                    success = True
                    break
                time.sleep(1.0)

            if not success:
                details = response.get("details") if response else "No details"
                raise RuntimeError(f"Login to admin api failed: {details}")
            else:
                print("Admin successfully logged into server.")
        except Exception as e:
            print(f"Exception in logging in to admin: {e.__str__()}")
        return success

    def get_job_result(self, job_id: str):
        command_name = "download_job"
        response = self.admin_api.do_command(f"{command_name} {job_id}")
        if response["status"] != APIStatus.SUCCESS:
            raise RuntimeError(f"{command_name} failed: {response}")
        run_data = {
            "job_id": job_id,
            "workspace_root": os.path.join(self.download_root_dir, job_id, "workspace"),
        }

        return run_data

    def ensure_clients_started(self, num_clients):
        timeout = 1000
        start_time = time.time()
        clients_up = False
        while not clients_up:
            if time.time() - start_time > timeout:
                raise RuntimeError(f"Clients could not be started in {timeout} seconds.")

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

    def submit_job(self, job_name: str):
        response = self.admin_api.submit_job(job_name)
        self.admin_api_response = response["raw"]["data"]
        if response["status"] == APIStatus.SUCCESS:
            self.job_id = response["details"]["job_id"]
            self.last_job_name = job_name

    def _check_job_done(self, job_id: str):
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
                            print(f"response is {response}")
                            print(f"job_run_statuses is {job_run_statuses}")
                            for row in response["details"]["client_statuses"]:
                                if row[3] != "stopped":
                                    continue
                            # check if the current job is completed
                            if job_run_statuses[job_id] in (
                                RunStatus.FINISHED_COMPLETED.value,
                                RunStatus.FINISHED_ABORTED.value,
                            ):
                                return True
        return False

    def _get_stats(self, target: str, job_id: str):
        return self.admin_api.show_stats(job_id, target)

    def _get_job_log(self, target: str, job_id: str):
        job_log_file = os.path.join(job_id, "log.txt")
        logs = self.admin_api.cat_target(target, file=job_log_file)["details"]["message"].splitlines()
        return logs

    def _get_site_log(self, target: str):
        logs = self.admin_api.cat_target(target, file="log.txt")["details"]["message"].splitlines()
        return logs

    def _get_job_run_statuses(self):
        list_jobs_result = self.admin_api.list_jobs()
        if list_jobs_result["status"] == APIStatus.SUCCESS:
            list_jobs_string = list_jobs_result["details"]["message"]
            job_run_statuses = _parse_job_run_statuses(list_jobs_string)
            return job_run_statuses
        else:
            return {self.job_id: "List Job command failed"}

    def _print_state(self, state: dict, length: int = 30):
        self.logger.debug("\n" + "-" * length)
        for k, v in state.items():
            self.logger.debug(f"{k}: {v}")
        self.logger.debug("-" * length + "\n")

    def _update_run_state(self, stats: dict, run_state: dict, job_run_status: str):
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

        self.logger.debug("Prev state:")
        self._print_state(state=prev_run_state)
        self.logger.debug(f"STATS: {stats}")
        self.logger.debug("Current state:")
        self._print_state(state=run_state)

        return run_state != prev_run_state, run_state

    def run_event_sequence(self, site_launcher, event_sequence):
        run_state = {"run_finished": None, "workflows": None}

        event_idx = 0
        # whether event has been successfully triggered
        event_triggered = [False for _ in range(len(event_sequence))]

        self.test_done = False
        while not self.test_done:
            job_run_statuses = self._get_job_run_statuses()
            self.logger.debug(f"job run status: {job_run_statuses}")

            if self.job_id:
                stats = self._get_stats(target=TargetType.SERVER, job_id=self.job_id)
                # update run_state
                changed, run_state = self._update_run_state(
                    stats=stats, run_state=run_state, job_run_status=job_run_statuses[self.job_id]
                )

            # check if event is triggered -> then execute the corresponding actions
            if event_idx < len(event_sequence) and not event_triggered[event_idx]:
                event_trigger = event_sequence[event_idx]["trigger"]

                strings_to_check = None
                # prepare to check for trigger for different trigger type
                if event_trigger["type"] == "server_log":
                    server_logs = self._get_site_log(target=TargetType.SERVER)
                    strings_to_check = "\n".join(server_logs)
                elif event_trigger["type"] == "client_log":
                    client_logs = self._get_site_log(target=event_trigger["args"]["target"])
                    strings_to_check = "\n".join(client_logs)
                elif event_trigger["type"] == "server_job_log":
                    if not self.job_id:
                        raise RuntimeError("No submitted jobs.")
                    server_logs = self._get_job_log(target=TargetType.SERVER, job_id=self.job_id)
                    strings_to_check = "\n".join(server_logs)
                elif event_trigger["type"] == "client_job_log":
                    if not self.job_id:
                        raise RuntimeError("No submitted jobs.")
                    client_logs = self._get_job_log(target=event_trigger["args"]["target"], job_id=self.job_id)
                    strings_to_check = "\n".join(client_logs)
                elif event_trigger["type"] != "run_state":
                    raise RuntimeError(f"This trigger type {event_trigger['type']} is not supported.")

                trigger_data = event_trigger["data"]
                if _check_event_trigger(
                    event_trigger=trigger_data, string_to_check=strings_to_check, run_state=run_state
                ):
                    print(f"EVENT TRIGGER '{trigger_data}' is TRIGGERED.")
                    event_triggered[event_idx] = True
                    self.execute_actions(site_launcher, event_sequence[event_idx]["actions"])

            if event_idx < len(event_sequence) and event_triggered[event_idx]:
                result = event_sequence[event_idx]["result"]
                if result["type"] == "run_state":
                    # check result state only when server is up and running
                    if self.server_status():
                        # compare run_state to expected result data from the test case
                        if any(list(run_state.values())):
                            _check_run_state(state=run_state, expected_state=result["data"])
                            event_idx += 1
                elif result["type"] == "admin_api_response":
                    if not self.admin_api_response:
                        raise RuntimeError("Missing admin_api_response.")
                    assert self.admin_api_response == result["data"]
                    event_idx += 1

            if self.job_id:
                self.test_done = self._check_job_done(job_id=self.job_id)
            time.sleep(self.poll_period)

        assert all(event_triggered), "Test failed: not all test events were triggered"

    def execute_actions(self, site_launcher, actions):
        for action in actions:
            tokens = action.split(" ")
            command = tokens[0]
            args = tokens[1:]

            print(f"ACTION: {action}")

            if command == "submit_job":
                self.submit_job(job_name=str(args[0]))
            elif command == "sleep":
                time.sleep(int(args[0]))
            elif command == "kill":
                _handle_kill(args, self.admin_api, site_launcher)
            elif command == "start":
                self._handle_start(args, site_launcher)
            elif command == "test":
                if args[0] == "admin_commands":
                    run_admin_api_tests(self.admin_api)
            elif command == "no_op":
                pass
            elif command == "mark_test_done":
                self.test_done = True
            else:
                raise RuntimeError(f"Action {command} is not supported.")

    def finalize(self):
        if self.job_id:
            self.admin_api.abort_job(self.job_id)
        self.admin_api.shutdown(target_type=TargetType.ALL)
        self.admin_api.close()
        time.sleep(10)

    def _handle_start(self, args, site_launcher):
        if args[0] == "server":
            if len(args) == 2:
                # if server id is provided
                server_ids = [int(args[1])]
            else:
                # start all servers
                server_ids = list(site_launcher.server_properties.keys())
            for sid in server_ids:
                site_launcher.start_server(sid)
            self.admin_api.login(username=self.admin_user_name)
        elif args[0] == "client":
            if len(args) == 2:
                # if client id is provided
                client_ids = [int(args[1])]
            else:
                # start all clients
                client_ids = list(site_launcher.client_properties.keys())
            for cid in client_ids:
                site_launcher.start_client(cid)
        elif args[0] == "overseer":
            site_launcher.start_overseer()
        else:
            raise RuntimeError(f"Target {args[0]} is not supported.")


def _handle_kill(args, admin_api, site_launcher):
    if args[0] == "server":
        if len(args) == 2:
            # if server id is provided
            server_id = int(args[1])
        else:
            # kill active server
            server_id = site_launcher.get_active_server_id(admin_api.port)
        site_launcher.stop_server(server_id)
    elif args[0] == "overseer":
        site_launcher.stop_overseer()
    elif args[0] == "client":
        if len(args) == 2:
            # if client id is provided
            client_ids = [int(args[1])]
        else:
            # close all clients
            client_ids = list(site_launcher.client_properties.keys())
        for cid in client_ids:
            admin_api.remove_client([site_launcher.client_properties[cid]["name"]])
            site_launcher.stop_client(cid)
