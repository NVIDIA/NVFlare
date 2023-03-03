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

from nvflare.apis.job_def import RunStatus
from nvflare.fuel.hci.client.api_status import APIStatus
from nvflare.fuel.hci.client.fl_admin_api import FLAdminAPI
from nvflare.fuel.hci.client.fl_admin_api_spec import TargetType
from nvflare.fuel.utils.class_utils import instantiate_class
from tests.integration_test.src.action_handlers import (
    _AbortJobHandler,
    _AdminCommandsHandler,
    _CheckJobHandler,
    _CloneJobHandler,
    _KillHandler,
    _ListJobHandler,
    _NoopHandler,
    _ShellCommandHandler,
    _SleepHandler,
    _StartHandler,
    _SubmitJobHandler,
    _TestDoneHandler,
)
from tests.integration_test.src.site_launcher import SiteLauncher
from tests.integration_test.src.utils import get_job_meta


class FLTestError(Exception):
    pass


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


def _read_admin_json_file(admin_json_file) -> dict:
    if not os.path.exists(admin_json_file):
        raise RuntimeError("Missing admin json file.")
    with open(admin_json_file, "r") as f:
        admin_json = json.load(f)
    return admin_json


def _create_admin_api(workspace_root_dir, upload_root_dir, download_root_dir, admin_user_name, poc):
    admin_startup_folder = os.path.join(workspace_root_dir, admin_user_name, "startup")
    admin_json_file = os.path.join(admin_startup_folder, "fed_admin.json")
    admin_json = _read_admin_json_file(admin_json_file)
    overseer_agent = instantiate_class(
        class_path=admin_json["admin"]["overseer_agent"]["path"],
        init_params=admin_json["admin"]["overseer_agent"]["args"],
    )

    ca_cert = ""
    client_key = ""
    client_cert = ""
    if not poc:
        ca_cert = os.path.join(admin_startup_folder, admin_json["admin"]["ca_cert"])
        client_key = os.path.join(admin_startup_folder, admin_json["admin"]["client_key"])
        client_cert = os.path.join(admin_startup_folder, admin_json["admin"]["client_cert"])

    admin_api = FLAdminAPI(
        upload_dir=upload_root_dir,
        download_dir=download_root_dir,
        overseer_agent=overseer_agent,
        poc=poc,
        user_name=admin_user_name,
        ca_cert=ca_cert,
        client_key=client_key,
        client_cert=client_cert,
    )
    return admin_api


def _admin_api_login(admin_api, admin_user_name, poc):
    login_success = False
    try:
        response = None
        timeout = 30
        start_time = time.time()
        while time.time() - start_time <= timeout:
            if poc:
                response = admin_api.login_with_poc(username=admin_user_name, poc_key="admin")
            else:
                response = admin_api.login(username=admin_user_name)
            if response["status"] == APIStatus.SUCCESS:
                login_success = True
                break
            time.sleep(0.1)

        if not login_success:
            details = response.get("details") if response else "No details"
            print(f"Login to admin api failed: {details}")
        else:
            print("Admin successfully logged into server.")
    except Exception as e:
        print(f"Exception in logging in to admin: {e.__str__()}")
    return login_success


class FLTestDriver:
    def __init__(self, download_root_dir: str, site_launcher: SiteLauncher, poll_period=1):
        """FL system test driver.

        Args:
            download_root_dir: the root dir to download things to
            site_launcher (SiteLauncher): a SiteLauncher object
            poll_period (int): note that this value can't be too small,
                otherwise will have resource issue
        """
        self.download_root_dir = download_root_dir
        self.site_launcher = site_launcher
        self.poll_period = poll_period

        self.super_admin_api = None
        self.super_admin_user_name = None
        self.admin_api_response = None
        self.admin_apis = {}

        self.logger = logging.getLogger(self.__class__.__name__)
        self.test_done = False
        self.job_id = None
        self.last_job_name = ""

        self.action_handlers = {
            "start": _StartHandler(),
            "kill": _KillHandler(),
            "sleep": _SleepHandler(),
            "no_op": _NoopHandler(),
            "mark_test_done": _TestDoneHandler(),
            "run_admin_commands": _AdminCommandsHandler(),
            "submit_job": _SubmitJobHandler(),
            "clone_job": _CloneJobHandler(),
            "abort_job": _AbortJobHandler(),
            "list_job": _ListJobHandler(),
            "shell_commands": _ShellCommandHandler(),
            "ensure_current_job_done": _CheckJobHandler(),
        }

    def initialize_super_user(self, workspace_root_dir: str, upload_root_dir: str, poc: bool, super_user_name: str):
        self.super_admin_user_name = super_user_name
        admin_api = _create_admin_api(
            workspace_root_dir=workspace_root_dir,
            upload_root_dir=upload_root_dir,
            download_root_dir=self.download_root_dir,
            admin_user_name=super_user_name,
            poc=poc,
        )
        if _admin_api_login(admin_api, super_user_name, poc):
            self.super_admin_api = admin_api
            return True

        return False

    def initialize_admin_users(self, workspace_root_dir: str, upload_root_dir: str, poc: bool, admin_user_names: list):
        for user_name in admin_user_names:
            if user_name == self.super_admin_user_name:
                continue
            admin_api = _create_admin_api(
                workspace_root_dir=workspace_root_dir,
                upload_root_dir=upload_root_dir,
                download_root_dir=self.download_root_dir,
                admin_user_name=user_name,
                poc=poc,
            )
            login_result = _admin_api_login(admin_api, user_name, poc)
            if not login_result:
                return False
            self.admin_apis[user_name] = admin_api
        return True

    def get_job_result(self, job_id: str):
        command_name = "download_job"
        response = self.super_admin_api.do_command(f"{command_name} {job_id}")
        if response["status"] != APIStatus.SUCCESS:
            raise RuntimeError(f"{command_name} failed: {response}")
        run_data = {
            "job_id": job_id,
            "workspace_root": os.path.join(self.download_root_dir, job_id, "workspace"),
        }

        return run_data

    @staticmethod
    def _check_client_status_ready(response: dict):
        if "details" not in response:
            return False

        data = response.get("raw", {}).get("data", [])
        if data:
            for d in data:
                if d.get("type") == "error":
                    return False

        return True

    def ensure_clients_started(self, num_clients):
        timeout = 1000
        start_time = time.time()
        clients_up = False
        while not clients_up:
            if time.time() - start_time > timeout:
                raise RuntimeError(f"Clients could not be started in {timeout} seconds.")

            time.sleep(0.5)
            response = self.super_admin_api.check_status(target_type=TargetType.CLIENT)
            if response["status"] == APIStatus.SUCCESS:
                if not self._check_client_status_ready(response):
                    # clients not ready
                    continue
                for row in response["details"]["client_statuses"][1:]:
                    if row[3] != "not started":
                        continue
                # wait for all clients to come up
                if len(response["details"]["client_statuses"]) < num_clients + 1:
                    continue
                clients_up = True
                print("All clients are up.")

        return clients_up

    def server_status(self):
        response = self.super_admin_api.check_status(target_type=TargetType.SERVER)
        if response and "status" in response and response["status"] == APIStatus.SUCCESS and "details" in response:
            return response["details"]
        return None

    def client_status(self):
        response = self.super_admin_api.check_status(target_type=TargetType.CLIENT)
        if response and "status" in response and response["status"] == APIStatus.SUCCESS and "details" in response:
            return response["details"]
        return None

    def _get_stats(self, target: str, job_id: str):
        return self.super_admin_api.show_stats(job_id, target)

    def _get_job_log(self, target: str, job_id: str):
        job_log_file = os.path.join(job_id, "log.txt")
        logs = self.super_admin_api.cat_target(target, file=job_log_file)["details"]["message"].splitlines()
        return logs

    def _get_site_log(self, target: str):
        logs = self.super_admin_api.cat_target(target, file="log.txt")["details"]["message"].splitlines()
        return logs

    def _print_state(self, state: dict, length: int = 30):
        self.logger.info("\n" + "-" * length)
        for k, v in state.items():
            self.logger.info(f"{k}: {v}")
        self.logger.info("-" * length + "\n")

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
        # 'raw': {'time': '2022-04-04 15:13:09.367350', 'data': [xxx], 'status': <APIStatus.SUCCESS: 'SUCCESS'>}}
        prev_run_state = run_state.copy()

        # parse stats
        if (
            stats
            and "status" in stats
            and stats["status"] == APIStatus.SUCCESS
            and "details" in stats
            and "message" in stats["details"]
            and isinstance(stats["details"]["message"], dict)
        ):
            run_state["workflows"] = _parse_workflow_states(stats_message=stats["details"]["message"])

        # parse job status
        run_state["run_finished"] = job_run_status == RunStatus.FINISHED_COMPLETED.value

        return run_state != prev_run_state, run_state

    def reset_test_info(self, reset_job_info=False):
        self.test_done = False
        self.admin_api_response = None
        if reset_job_info:
            self.job_id = None
            self.last_job_name = ""

    def run_event_sequence(self, event_sequence):
        run_state = {"run_finished": None, "workflows": None}

        event_idx = 0
        # whether event has been successfully triggered
        event_triggered = [False for _ in range(len(event_sequence))]

        self.test_done = False
        while not self.test_done:

            if self.job_id:
                job_meta = get_job_meta(self.super_admin_api, job_id=self.job_id)
                job_run_status = job_meta.get("status")
                stats = self._get_stats(target=TargetType.SERVER, job_id=self.job_id)
                # update run_state
                changed, run_state = self._update_run_state(
                    stats=stats, run_state=run_state, job_run_status=job_run_status
                )

            if event_idx < len(event_sequence):
                if not event_triggered[event_idx]:
                    # check if event is triggered -> then execute the corresponding actions
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
                        self.execute_actions(
                            actions=event_sequence[event_idx]["actions"],
                            admin_user_name=event_sequence[event_idx].get("admin_user_name"),
                        )

                if event_triggered[event_idx]:
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

            time.sleep(self.poll_period)

        assert all(event_triggered), "Test failed: not all test events were triggered"

    def execute_actions(self, actions, admin_user_name):
        for action in actions:
            tokens = action.split(" ")
            command = tokens[0]
            args = tokens[1:]

            print(f"ACTION: {action} ADMIN_USER_NAME: {admin_user_name}")

            if command not in self.action_handlers:
                raise RuntimeError(f"Action {command} is not supported.")

            if admin_user_name is None:
                admin_api = self.super_admin_api
            else:
                admin_api = self.admin_apis[admin_user_name]

            self.action_handlers[command].handle(command_args=args, admin_controller=self, admin_api=admin_api)

    def finalize(self):
        if self.job_id:
            self.super_admin_api.abort_job(self.job_id)
        for k in self.admin_apis:
            self.admin_apis[k].close()
        self.super_admin_api.shutdown(target_type=TargetType.ALL)
        self.super_admin_api.close()
        time.sleep(1)
