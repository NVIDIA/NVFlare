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

import json
import logging
import os
import time

from nvflare.apis.job_def import RunStatus
from nvflare.fuel.flare_api.api_spec import JobNotFound, JobNotRunning
from nvflare.fuel.flare_api.flare_api import new_secure_session, new_insecure_session
from nvflare.fuel.hci.client.api_status import APIStatus
from nvflare.fuel.hci.client.fl_admin_api_constants import FLDetailKey
from nvflare.fuel.hci.client.fl_admin_api_spec import TargetType
from tests.integration_test.src.action_handlers_flare_api import (
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
from tests.integration_test.src.nvf_test_driver import NVFTestError
from tests.integration_test.src.site_launcher import SiteLauncher
from tests.integration_test.src.utils import (
    check_client_status_ready_FLARE_API,
    ensure_FLARE_API_admin_logged_in,
    get_job_meta_FLARE_API,
)

# This is copied from nvf_test_driver.py and modified to use the new FLARE API

def _parse_workflow_states(stats_message: dict):
    # {
    #     'ScatterAndGather':
    #         {'tasks': {'train': []}, 'phase': 'train', 'current_round': 1, 'num_rounds': 2},
    #     'ServerRunner':
    #         {'job_id': 'xxx', 'status': 'started', 'workflow': 'scatter_and_gather'}
    # }
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
            raise NVFTestError("Event trigger is of dict type but run_state is not provided.")
        return _check_dict_b_value_same_as_dict_a_for_keys_in_dict_a(event_trigger, run_state)
    elif isinstance(event_trigger, str):
        if string_to_check is None:
            raise NVFTestError("Event trigger is of str type but string_to_check is not provided.")
        return event_trigger in string_to_check
    else:
        raise NVFTestError(f"event_trigger type {type(event_trigger)} is not supported.")


def _update_run_state(stats: dict, run_state: dict, job_run_status: str):
    # extract run_state from stats
    # for stats structure please refer to "nvflare/private/fed/server/info_coll_cmd.py"
    # {'status': <APIStatus.SUCCESS: 'SUCCESS'>,
    #        'details': {
    #           'message': {
    #               'server': {
    #                   'ScatterAndGather': {
    #                       'tasks': {'train': []},
    #                       'phase': 'train',
    #                       'current_round': 0,
    #                       'num_rounds': 2},
    #                   'CrossSiteModelEval':
    #                       {'tasks': {}},
    #                   'ServerRunner': {
    #                       'job_id': XXX,
    #                       'status': 'started',
    #                       'workflow': 'scatter_and_gather'
    #                    }
    #                }
    #            }
    #       },
    # 'raw': {'time': '2022-04-04 15:13:09.367350', 'data': [xxx], 'status': <APIStatus.SUCCESS: 'SUCCESS'>}}

    # reply.get("data")["data"] = object after message
    prev_run_state = run_state.copy()

    # parse stats
    if stats:
        run_state["workflows"] = _parse_workflow_states(stats_message=stats["server"])

    # parse job status
    run_state["run_finished"] = job_run_status == RunStatus.FINISHED_COMPLETED.value

    return run_state != prev_run_state, run_state


class NVFFLAREAPITestDriver:
    def __init__(self, download_root_dir: str, site_launcher: SiteLauncher, poll_period=1):
        """FL system test driver using the new FLARE API.

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
        self.last_job_name = None

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
        admin_json_file = os.path.join(workspace_root_dir, super_user_name, "startup", "fed_admin.json")
        if not os.path.exists(admin_json_file):
            raise RuntimeError("Missing admin json file.")
        with open(admin_json_file, "r") as f:
            admin_config = json.load(f)
        admin_config["admin"]["upload_dir"] = os.path.join(os.path.dirname(__file__), "..", upload_root_dir)
        admin_config["admin"]["download_dir"] = self.download_root_dir
        with open(admin_json_file, "w") as f:
            json.dump(admin_config, f)
        self.super_admin_user_name = super_user_name
        try:
            if not poc:
                admin_api = new_secure_session(username=super_user_name, startup_kit_location=os.path.join(workspace_root_dir, super_user_name))
            else:
                admin_api = new_insecure_session(startup_kit_location=os.path.join(workspace_root_dir, super_user_name))
            login_result = ensure_FLARE_API_admin_logged_in(admin_api)
        except Exception as e:
            raise NVFTestError(f"create and login to admin failed: {e}")
        if not login_result:
            raise NVFTestError(f"initialize_super_user {super_user_name} failed.")

        self.super_admin_api = admin_api

    def initialize_admin_users(self, workspace_root_dir: str, upload_root_dir: str, poc: bool, admin_user_names: list):
        for user_name in admin_user_names:
            if user_name == self.super_admin_user_name:
                continue
            try:
                admin_json_file = os.path.join(workspace_root_dir, user_name, "startup", "fed_admin.json")
                if not os.path.exists(admin_json_file):
                    raise RuntimeError("Missing admin json file.")
                with open(admin_json_file, "r") as f:
                    admin_config = json.load(f)
                admin_config["admin"]["upload_dir"] = os.path.join(os.path.dirname(__file__), "..", upload_root_dir)
                admin_config["admin"]["download_dir"] = self.download_root_dir
                with open(admin_json_file, "w") as f:
                    json.dump(admin_config, f)
                if not poc:
                    admin_api = new_secure_session(username=user_name, startup_kit_location=os.path.join(workspace_root_dir, user_name))
                else:
                    admin_api = new_insecure_session(startup_kit_location=os.path.join(workspace_root_dir, user_name))
                login_result = ensure_FLARE_API_admin_logged_in(admin_api)
            except Exception as e:
                self.admin_apis = None
                raise NVFTestError(f"Create and login to admin failed: {e}")
            if not login_result:
                self.admin_apis = None
                raise NVFTestError(f"Initialize_admin_users failed for user: {user_name}.")
            self.admin_apis[user_name] = admin_api

    def get_job_result(self, job_id: str):
        command_name = "download_job"
        response = self.super_admin_api.api.do_command(f"{command_name} {job_id}")
        run_data = {
            "job_id": job_id,
            "workspace_root": os.path.join(self.download_root_dir, job_id, "workspace"),
        }

        return run_data

    def ensure_clients_started(self, num_clients: int, timeout: int):
        start_time = time.time()
        clients_up = False
        while not clients_up:
            if time.time() - start_time > timeout:
                raise NVFTestError(f"Clients could not be started in {timeout} seconds.")

            time.sleep(0.5)
            details = self.client_status()
            print(f"Check client status response is {details}")
            if not details:
                # clients not ready
                continue

            # this coming from private/fed/server/training_cmds.py
            for row in details["client_statuses"][1:]:
                if row[3] != "No Jobs":
                    raise NVFTestError("Clients started with left-over jobs.")

            # wait for all clients to come up
            if len(details["client_statuses"]) < num_clients + 1:
                continue
            clients_up = True
            print("All clients are up.")

    def _parse_section_of_response_text(
        self, data, start_string: str, offset: int = None, end_string: str = None, end_index=None
    ) -> str:
        """Convenience method to get portion of string based on parameters. Copied from fl_admin_api.py"""
        if not offset:
            offset = len(start_string) + 1
        if end_string:
            return data[data.find(start_string) + offset : data.find(end_string)]
        if end_index:
            return data[data.find(start_string) + offset : end_index]
        return data[data.find(start_string) + offset :]

    def _parse_section_of_response_text_as_int(
        self, data, start_string: str, offset: int = None, end_string: str = None, end_index=None
    ) -> int:
        """Copied from fl_admin_api.py"""
        try:
            return int(
                self._parse_section_of_response_text(
                    data=data, start_string=start_string, offset=offset, end_string=end_string, end_index=end_index
                )
            )
        except ValueError:
            return -1

    # server_status and client_status seem to only be used for display for now
    def server_status(self):
        reply = self.super_admin_api.api.do_command("check_status server")
        details = {}
        if reply.get("data"):
            for data in reply["data"]:
                if data["type"] == "string":
                    if data["data"].find("Engine status:") != -1:
                        details[FLDetailKey.SERVER_ENGINE_STATUS] = self._parse_section_of_response_text(
                            data=data["data"], start_string="Engine status:"
                        )
                    if data["data"].find("Registered clients:") != -1:
                        details[FLDetailKey.REGISTERED_CLIENTS] = self._parse_section_of_response_text_as_int(
                            data=data["data"], start_string="Registered clients:"
                        )
                if data["type"] == "table":
                    details[FLDetailKey.STATUS_TABLE] = data["rows"]
        return details

    def client_status(self):
        reply = self.super_admin_api.api.do_command("check_status client")
        details = {}
        if reply.get("data"):
            for data in reply["data"]:
                if data["type"] == "table":
                    details["client_statuses"] = data["rows"]
        return details

    def _get_stats(self, target: str, job_id: str):
        print("getting stats for: "+target+": "+job_id)
        try:
            return self.super_admin_api.show_stats(job_id, target)
        except JobNotFound:
            return False

    def _get_job_log(self, target: str, job_id: str):
        job_log_file = os.path.join(job_id, "log.txt")
        logs = self.super_admin_api.cat_target(target, file=job_log_file).splitlines()
        return logs

    def _get_site_log(self, target: str):
        logs = self.super_admin_api.cat_target(target, file="log.txt").splitlines()
        return logs

    def _print_state(self, state: dict, length: int = 30):
        self.logger.info("\n" + "-" * length)
        for k, v in state.items():
            self.logger.info(f"{k}: {v}")
        self.logger.info("-" * length + "\n")

    def reset_test_info(self, reset_job_info=False):
        self.test_done = False
        self.admin_api_response = None
        if reset_job_info:
            self.job_id = None
            self.last_job_name = None

    def run_event_sequence(self, event_sequence):
        run_state = {"run_finished": None, "workflows": None}

        event_idx = 0
        # whether event has been successfully triggered
        event_triggered = [False for _ in range(len(event_sequence))]

        self.test_done = False
        while not self.test_done:

            if self.job_id:
                job_meta = get_job_meta_FLARE_API(self.super_admin_api, job_id=self.job_id)
                job_run_status = job_meta.get("status")
                try:
                    stats = self._get_stats(target=TargetType.SERVER, job_id=self.job_id)
                    # update run_state
                    changed, run_state = _update_run_state(stats=stats, run_state=run_state, job_run_status=job_run_status)
                except JobNotRunning:
                    print("JOB NOT FOUND FOR STATS: " + self.job_id)
                    run_state["run_finished"] = job_run_status == RunStatus.FINISHED_COMPLETED.value

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
                            raise NVFTestError("No submitted jobs.")
                        server_logs = self._get_job_log(target=TargetType.SERVER, job_id=self.job_id)
                        strings_to_check = "\n".join(server_logs)
                    elif event_trigger["type"] == "client_job_log":
                        if not self.job_id:
                            raise NVFTestError("No submitted jobs.")
                        client_logs = self._get_job_log(target=event_trigger["args"]["target"], job_id=self.job_id)
                        strings_to_check = "\n".join(client_logs)
                    elif event_trigger["type"] != "run_state":
                        raise NVFTestError(f"This trigger type {event_trigger['type']} is not supported.")

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
                        if self.server_status() is not None:
                            # compare run_state to expected result data from the test case
                            _check_run_state(state=run_state, expected_state=result["data"])
                            event_idx += 1
                    elif result["type"] == "admin_api_response":
                        if self.admin_api_response is None:
                            raise NVFTestError("Missing admin_api_response.")
                        assert self.admin_api_response == result["data"]
                        event_idx += 1
                    elif result["type"] == "job_submit_success":
                        if self.job_id is None or self.last_job_name is None:
                            raise NVFTestError(f"Job submission failed with: {self.admin_api_response}")
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
                raise NVFTestError(f"Action {command} is not supported.")

            if admin_user_name is None:
                admin_api = self.super_admin_api
            else:
                admin_api = self.admin_apis[admin_user_name]

            self.action_handlers[command].handle(command_args=args, admin_controller=self, admin_api=admin_api)

    def finalize(self):
        if self.super_admin_api:
            if self.job_id:
                self.super_admin_api.abort_job(self.job_id)
            for k in self.admin_apis:
                self.admin_apis[k].close()
            self.super_admin_api.shutdown(target_type=TargetType.ALL)
            self.super_admin_api.close()
        time.sleep(1)
