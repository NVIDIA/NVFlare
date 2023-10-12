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

import argparse
import logging

from nvflare.client.defs import RC, MetaKey, AgentClosed, TaskResult
from nvflare.client.ipc_agent import IPCAgent

NUMPY_KEY = "numpy_key"


def main():

    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", "-w", type=str, help="workspace folder", required=False, default=".")
    parser.add_argument("--site_name", "-s", type=str, help="flare site name", required=True)
    parser.add_argument("--agent_id", "-a", type=str, help="agent id", required=True)
    parser.add_argument("--job_id", "-j", type=str, help="flare job id", required=False, default="")
    parser.add_argument("--site_url", "-u", type=str, help="flare site url", required=False, default="")

    args = parser.parse_args()

    agent = IPCAgent(
        root_url="grpc://server:8002",
        flare_site_name=args.site_name,
        agent_id=args.agent_id,
        workspace_dir=args.workspace,
        secure_mode=True,
        submit_result_timeout=2.0,
        flare_site_heartbeat_timeout=120.0,
        job_id=args.job_id,
        flare_site_url=args.site_url,
    )

    agent.start()

    while True:
        print("getting task ...")
        try:
            task = agent.get_task()
        except AgentClosed:
            print("agent closed - exit")
            break

        print(f"got task: {task}")
        rc, meta, result = train(task.meta, task.data)
        submitted = agent.submit_result(TaskResult(data=result, meta=meta, return_code=rc))
        print(f"result submitted: {submitted}")

    agent.stop()


def train(meta, model):
    current_round = meta.get(MetaKey.CURRENT_ROUND)
    total_rounds = meta.get(MetaKey.TOTAL_ROUND)

    # Ensure that data is of type weights. Extract model data
    np_data = model

    # Display properties.
    print(f"Model: \n{np_data}")
    print(f"Current Round: {current_round}")
    print(f"Total Rounds: {total_rounds}")

    # Doing some dummy training.
    if np_data:
        if NUMPY_KEY in np_data:
            np_data[NUMPY_KEY] += 1.0
        else:
            print("error: numpy_key not found in model.")
            return RC.BAD_TASK_DATA, None, None
    else:
        print("No model weights found in shareable.")
        return RC.BAD_TASK_DATA, None, None

    # Save local numpy model
    print(f"Model after training: {np_data}")

    # Prepare a DXO for our updated model. Create shareable and return
    return RC.OK, {MetaKey.NUM_STEPS_CURRENT_ROUND: 1}, np_data


if __name__ == "__main__":
    main()
