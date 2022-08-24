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

# TODO:: remove this file

# HA test cases (referenced from NVFlare 2.1 Test Plan- High Availability)

# supported trigger events:
#   - match a string based on log output from server
#   - state based on predefined tracked state variables (workflow, task, round_number, run_finished etc.)
# supported actions:
#   - kill {target} [id]
#   - sleep {seconds}
#   - restart {target} [id]


# 14 upload a job, kill the server during training, restart it should pick up the work
test_case_14 = {
    "name": "test_case_14",
    "description": "upload a job, kill the server during training, restart it should pick up the work",
    "setup": {
        "n_servers": 1,
        "n_clients": 2,
    },  # TODO potentially add ability to start overseer & choose order of starting overseer/server/client?
    "events": [
        {
            "trigger": {"workflow": 0, "round_number": 1, "task": "train"},
            "actions": [
                "sleep 5",
                "kill server",
                "sleep 10",
                "restart server",
            ],
            "result_state": "unchanged",  # "unchanged" is same as trigger state
        },
    ],
}

# 15 upload a job, kill the server after we start training but no round is completed, restart it should pick up the work
test_case_15 = {
    "name": "test_case_15",
    "description": "upload a job, kill the server after we start training but no round is completed, restart it should pick up the work",
    "setup": {"n_servers": 1, "n_clients": 2},
    "events": [
        {
            "trigger": {"workflow": 0, "round_number": 0, "task": "train"},
            "actions": [
                "kill server",
                "sleep 15",
                "restart server",
            ],
            "result_state": {"workflow": 0, "round_number": 0, "task": "train"},
        },
        # {
        #     "trigger": "Round 1 started.",
        #     "actions": [
        #         "sleep 5",
        #         "kill server",
        #         "sleep 10",
        #         "restart server",
        #     ],
        #     "result_state": {"workflow": 0, "round_number": 1, "task": "train"},
        # },
    ],
}

# 16 upload a job, kill the server during sending models to clients, restart it should pick up the work
test_case_16 = {
    "name": "test_case_16",
    "description": "upload a job, kill the server during sending models to clients, restart it should pick up the work",
    "setup": {"n_servers": 1, "n_clients": 2},
    "events": [
        {
            "trigger": "sent task assignment to client",
            "actions": ["kill server", "sleep 10", "restart server"],
            "result_state": {"workflow": 0, "round_number": 0},
        },
    ],
}

# 17 upload a job, kill the primary server during training, the second one should pick up the work
test_case_17 = {
    "name": "test_case_17",
    "description": "upload a job, kill the primary server during training, the second one should pick up the work",
    "setup": {"n_servers": 2, "n_clients": 2},
    "events": [
        {
            "trigger": {"workflow": 0, "round_number": 1, "task": "train"},
            "actions": [
                "sleep 5",
                "kill server",
                "sleep 10",
            ],
            "result_state": "unchanged",
        },
    ],
}

# 18 upload a job, kill the primary server after we start training but no round is completed, the second one should start from round 0
test_case_18 = {
    "name": "test_case_18",
    "description": "upload a job, kill the primary server after we start training but no round is completed, the second one should start from round 0",
    "setup": {"n_servers": 2, "n_clients": 2},
    "events": [
        {
            "trigger": {"workflow": 0, "round_number": 0, "task": "train"},
            "actions": [
                "sleep 5",
                "kill server",
                "sleep 10",
            ],
            "result_state": "unchanged",
        },
    ],
}

# 19 upload a job, kill the primary server during sending models to clients, the second one should start from round 0
test_case_19 = {
    "name": "test_case_16",
    "description": "upload a job, kill the primary server during sending models to clients, the second one should start from round 0",
    "setup": {"n_servers": 2, "n_clients": 2},
    "events": [
        {
            "trigger": "sent task assignment to client",
            "actions": [
                "kill server",
                "sleep 10",
            ],
            "result_state": {"workflow": 0, "round_number": 0, "task": "train"},
        },
    ],
}

# 20 upload a job that has multiple workflows, kill the primary server when the first workflow is completed, the second one should start with the second workflow
test_case_20 = {
    "name": "test_case_20",
    "description": "upload a job that has multiple workflows, kill the primary server when the first workflow is completed, the second one should start with the second workflow",
    "setup": {"n_servers": 2, "n_clients": 2},
    "events": [
        {"trigger": {"workflow": 1}, "actions": ["sleep 5", "kill server", "sleep 10"], "result_state": "unchanged"},
    ],
}

# 21 upload a job, kill the OverSeer, since the primary server is still running, things should run into completion (if client sites already got the old SP info.)

# 22 kill overseer with old information
test_case_22 = {
    "name": "test_case_22",
    "description": "kill overseer with old information",
    "setup": {"n_servers": 2, "n_clients": 2},
    "events": [
        {
            "trigger": {"workflow": 0, "task": "train"},
            "actions": ["sleep 5", "kill overseer", "sleep 10"],
            "result_state": "unchanged",
        },
    ],
}

# 23
# kill overseer with no information

# 24
# start overseer, client, then server

# 25
# overseer returns there is no hot endpoint available? -> fallback to previous SP

# 26
# overseer gives wrong information? -> keep trying

# 27
# sequence of kills and restarts of server

# 28
# kill client during training, change server, restart client
test_case_28 = {  # FAILS: gets stuck "Communicator - ERROR - Action: getTask grpc communication error."
    "name": "test_case_28",
    "description": "kill client during training, change server, restart client",
    "setup": {"n_servers": 2, "n_clients": 1},
    "events": [
        {
            "trigger": {"workflow": 0, "round_number": 0, "task": "train"},
            "actions": ["sleep 10", "kill client", "sleep 10", "kill server", "sleep 15", "restart client"],
            "result_state": "unchanged",
        },
    ],
}

# 28
# kill client during training, restart client
test_case_28b = {  # FAILS: gets stuck " - Communicator - ERROR - Action: getTask grpc communication error."
    "name": "test_case_28b",
    "description": "kill client during training, restart client",
    "setup": {"n_servers": 2, "n_clients": 1},
    "events": [
        {
            "trigger": {"workflow": 0, "round_number": 0, "task": "train"},
            "actions": ["sleep 10", "kill client", "sleep 15", "restart client"],
            "result_state": "unchanged",
        },
    ],
}

# 29
# overseer dies, will job submit fail?

# 30
# no overseer section in fed_client.json

# 31 After the successful run is completed, kill the server, restart it, it should NOT go into training phase
test_case_31 = {  # FAILS: still in cross validate workflow even though cross validate workflow finished (relook into "run_finished" state variable)
    "name": "test_case_31",
    "description": "After the successful run is completed, kill the server, restart it, it should NOT go into training phase",
    "setup": {"n_servers": 1, "n_clients": 2},
    "events": [
        {
            "trigger": {"run_finished": True},
            "actions": ["sleep 1", "kill server", "sleep 5", "restart server"],
            "result_state": {"run_finished": True},
        },
    ],
}

# 32 After the successful run is completed, kill the 1st server, the second server should NOT go into training phase
test_case_32 = {  # FAILS
    "name": "test_case_32",
    "description": "After the successful run is completed, kill the server, restart it, it should NOT go into training phase",
    "setup": {"n_servers": 2, "n_clients": 2},
    "events": [
        {
            "trigger": {"run_finished": True},
            "actions": [
                "kill server",
                "sleep 5",
            ],
            "result_state": {"run_finished": True},
        },
    ],
}

# add kills and restarts at various times for stress testing?

sag_tests = [test_case_15]  # , test_case_16, test_case_17, test_case_18, test_case_19, test_case_20, test_case_22]
cyclic_tests = [test_case_22]  # note: some tests are not applicable to cyclic_tests, as cyclic app only has 1 workflow

ha_tests = {"pt": sag_tests, "cyclic": cyclic_tests}
