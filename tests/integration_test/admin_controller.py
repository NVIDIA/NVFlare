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

from nvflare.fuel.hci.client.api_status import APIStatus
from nvflare.fuel.hci.client.fl_admin_api import FLAdminAPI
from nvflare.fuel.hci.client.fl_admin_api_constants import FLDetailKey
from nvflare.fuel.hci.client.fl_admin_api_spec import TargetType
from nvflare.ha.dummy_overseer_agent import DummyOverseerAgent
from nvflare.ha.overseer_agent import HttpOverseerAgent


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


def process_stats(stats, run_state):
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
    #                   'run_number': 1,
    #                   'status': 'started',
    #                   'workflow': 'scatter_and_gather'
    #                }
    #            }
    #       },
    # 'raw': {'time': '2022-04-04 15:13:09.367350', 'data': [{'type': 'dict', 'data': {'ScatterAndGather': {'tasks': {'train': []}, 'phase': 'train', 'current_round': 0, 'num_rounds': 2}, 'CrossSiteModelEval': {'tasks': {}}, 'ServerRunner': {'run_number': 1, 'status': 'started', 'workflow': 'scatter_and_gather'}}}], 'status': <APIStatus.SUCCESS: 'SUCCESS'>}}
    wfs = {}
    prev_run_state = run_state.copy()
    print(f"run_state {prev_run_state}", flush=True)
    print(f"stats {stats}", flush=True)
    if stats and "status" in stats and "details" in stats and stats["status"] == APIStatus.SUCCESS:
        if "message" in stats["details"]:
            wfs = stats["details"]["message"]
            if wfs:
                run_state["workflow"], run_state["task"], run_state["round_number"] = None, None, None
            for item in wfs:
                if wfs[item].get("tasks"):
                    run_state["workflow"] = item
                    run_state["task"] = list(wfs[item].get("tasks").keys())[0]
                    if "current_round" in wfs[item]:
                        run_state["round_number"] = wfs[item]["current_round"]
                # if wfs[item].get("workflow"):
                #     workflow = wfs[item].get("workflow")

    if stats["status"] == APIStatus.SUCCESS:
        run_state["run_finished"] = "ServerRunner" not in wfs.keys()
    else:
        run_state["run_finished"] = False

    wfs = [wf for wf in list(wfs.items()) if "tasks" in wf[1]]

    return run_state != prev_run_state, wfs, run_state


class AdminController:
    def __init__(self, jobs_root_dir, ha, poll_period=10):
        """
        This class runs an app on a given server and clients.
        """
        super().__init__()

        self.jobs_root_dir = jobs_root_dir
        self.poll_period = poll_period

        if ha:
            overseer_agent = HttpOverseerAgent(
                role="admin", overseer_end_point="http://127.0.0.1:5000/api/v1", project="example_project", name="admin"
            )
        else:
            overseer_agent = DummyOverseerAgent(sp_end_point="localhost:8002:8003")

        self.admin_api: FLAdminAPI = FLAdminAPI(
            upload_dir=self.jobs_root_dir,
            download_dir=self.jobs_root_dir,
            overseer_agent=overseer_agent,
            poc=True,
            debug=False,
            user_name="admin",
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

    def get_run_data(self):
        run_data = {"job_id": self.job_id, "jobs_root_dir": self.jobs_root_dir}

        return run_data

    def ensure_clients_started(self, num_clients):
        if not self.admin_api:
            return False

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
        if not self.admin_api:
            return ""

        response = self.admin_api.check_status(target_type=TargetType.SERVER)
        if response["status"] == APIStatus.SUCCESS:
            if "details" in response:
                return response["details"]
        return ""

    def client_status(self):
        if not self.admin_api:
            return ""

        response = self.admin_api.check_status(target_type=TargetType.CLIENT)
        if response["status"] == APIStatus.SUCCESS:
            if "details" in response:
                return response["details"]
        return ""

    def submit_job(self, job_name) -> bool:
        if not self.admin_api:
            raise RuntimeError("Missing admin_api in admin_controller.")

        response = self.admin_api.submit_job(job_name)
        if response["status"] != APIStatus.SUCCESS:
            raise RuntimeError(f"submit_job failed: {response}")
        self.job_id = response["details"]["job_id"]
        self.last_job_name = job_name
        return True

    def wait_for_job_done(self):
        # TODO:: Is it possible to get the training log after training is done?
        training_done = False
        while not training_done:
            time.sleep(self.poll_period)
            response = self.admin_api.check_status(target_type=TargetType.SERVER)
            if response["status"] != APIStatus.SUCCESS:
                raise RuntimeError(f"check_status failed: {response}")
            if not response["details"]:
                raise RuntimeError(f"response {response} does not have details.")
            if response["details"][FLDetailKey.SERVER_ENGINE_STATUS] == "stopped":
                response = self.admin_api.check_status(target_type=TargetType.CLIENT)
                if response["status"] != APIStatus.SUCCESS:
                    raise RuntimeError(f"check_status failed: {response}")
                for row in response["details"]["client_statuses"]:
                    if row[3] != "stopped":
                        continue
                training_done = True

    def _get_stats(self, target):
        return self.admin_api.show_stats(self.job_id, target)

    def _get_job_log(self, target):
        job_log_file = os.path.join(self.job_id, "log.txt")
        logs = self.admin_api.cat_target(target, file=job_log_file)["details"]["message"].splitlines()
        return logs

    def run_event_sequence(self, site_launcher, event_sequence):
        run_state = {"workflow": None, "task": None, "round_number": None, "run_finished": None}

        last_read_line = 0
        event_idx = 0
        ha_events = event_sequence["events"]
        event_test_status = [False for _ in range(len(ha_events))]  # whether event has been successfully triggered

        i = 0
        training_done = False
        while not training_done:
            i += 1

            server_logs = self._get_job_log(target=TargetType.SERVER)
            server_logs = server_logs[last_read_line:]
            last_read_line += len(server_logs)
            server_logs_string = "\n".join(server_logs)

            stats = self._get_stats(TargetType.SERVER)

            # update run_state
            changed, wfs, run_state = process_stats(stats, run_state)

            if changed or i % (10 / self.poll_period) == 0:
                i = 0
                print("STATS: ", stats)
                self.print_state(event_sequence["description"], run_state)

            # check if event is triggered -> then execute the corresponding actions
            if event_idx < len(ha_events) and not event_test_status[event_idx]:
                event_trigger = ha_events[event_idx]["trigger"]
                event_triggered = True

                if isinstance(event_trigger, dict):
                    for k, v in event_trigger.items():
                        if k == "workflow":
                            print(run_state)
                            print(wfs)
                            if run_state[k] != wfs[v][0]:
                                event_triggered = False
                                break
                        else:
                            if run_state[k] != v:
                                event_triggered = False
                                break
                elif isinstance(event_trigger, str):
                    if event_trigger not in server_logs_string:
                        event_triggered = False
                else:
                    raise RuntimeError(f"event_trigger type {type(event_trigger)} is not supported.")

                if event_triggered:
                    print(f"EVENT TRIGGER '{event_trigger}' is TRIGGERED.")
                    event_test_status[event_idx] = True
                    self.execute_actions(site_launcher, ha_events[event_idx]["actions"])
                    continue

            response = self.admin_api.check_status(target_type=TargetType.SERVER)
            if response and "status" in response and response["status"] != APIStatus.SUCCESS:
                print("NO ACTIVE SERVER!")

            elif (
                response and "status" in response and "details" in response and response["status"] == APIStatus.SUCCESS
            ):

                # compare run_state to expected result_state from the test case
                if (
                    event_idx < len(ha_events)
                    and event_test_status[event_idx]
                    and response["status"] == APIStatus.SUCCESS
                ):
                    result_state = ha_events[event_idx]["result_state"]
                    if any(list(run_state.values())):
                        if result_state == "unchanged":
                            result_state = ha_events[event_idx]["trigger"]
                        for k, v in result_state.items():
                            if k == "workflow":
                                print(f"ASSERT Current {k}: {run_state[k]} == Expected {k}: {wfs[v][0]}")
                                assert run_state[k] == wfs[v][0]
                            else:
                                print(f"ASSERT Current {k}: {run_state[k]} == Expected {k}: {v}")
                                assert run_state[k] == v
                        print("\n")
                        event_idx += 1

                # check if run is stopped
                if (
                    FLDetailKey.SERVER_ENGINE_STATUS in response["details"]
                    and response["details"][FLDetailKey.SERVER_ENGINE_STATUS] == "stopped"
                ):
                    response = self.admin_api.check_status(target_type=TargetType.CLIENT)
                    if response["status"] != APIStatus.SUCCESS:
                        print(f"CHECK status failed: {response}")
                    for row in response["details"]["client_statuses"]:
                        if row[3] != "stopped":
                            continue
                    training_done = True
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
                    active_server_id = site_launcher.get_active_server_id(self.admin_api.port)
                    site_launcher.stop_server(active_server_id)
                elif args[0] == "overseer":
                    site_launcher.stop_overseer()
                elif args[0] == "client":  # TODO fix client kill & restart during run
                    if len(args) == 2:
                        client_id = int(args[1])
                    else:
                        client_id = list(site_launcher.client_properties.keys())[0]
                    self.admin_api.remove_client([site_launcher.client_properties[client_id]["name"]])
                    site_launcher.stop_client(client_id)

            elif command == "restart":
                if args[0] == "server":
                    if len(args) == 2:
                        server_id = int(args[1])
                    else:
                        print(site_launcher.server_properties)
                        server_id = list(site_launcher.server_properties.keys())[0]
                    site_launcher.start_server()
                elif args[0] == "overseer":
                    site_launcher.start_overseer()
                elif args[0] == "client":  # TODO fix client kill & restart during run
                    if len(args) == 2:
                        client_id = int(args[1])
                    else:
                        client_id = list(site_launcher.client_properties.keys())[0]
                    site_launcher.start_client(client_id)
            else:
                raise RuntimeError(f"Command {command} is not supported.")

    def print_state(self, test_description: str, state: dict):
        print("\n")
        print(f"Job name: {self.last_job_name}")
        print(f"HA test: {test_description}")
        print("-" * 30)
        for k, v in state.items():
            print(f"{k}: {v}")
        print("-" * 30 + "\n")

    def finalize(self):
        if self.job_id:
            self.admin_api.abort_job(self.job_id)
        self.admin_api.overseer_agent.end()
        self.admin_api.shutdown(target_type=TargetType.ALL)
        time.sleep(10)
