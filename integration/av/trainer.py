# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from integration.av.defs import RC, ModelMetaKey
from integration.av.flare import FlareAgent

NUMPY_KEY = "numpy_key"


def main():

    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", "-w", type=str, help="workspace folder", required=False, default=".")
    parser.add_argument("--site_name", "-s", type=str, help="flare site name", required=True)
    parser.add_argument("--agent_id", "-a", type=str, help="agent id", required=True)
    args = parser.parse_args()

    agent = FlareAgent(
        root_url="grpc://server:8002",
        flare_site_name=args.site_name,
        agent_id=args.agent_id,
        workspace_dir=args.workspace,
        secure_mode=True,
        submit_result_timeout=2.0,
        flare_site_ready_timeout=120.0,
    )

    agent.start()

    while True:
        print("try to get a new task ...")
        task = agent.get_task()
        if not task:
            # done
            print("no more task - exit")
            break

        name, tid, meta, model = task
        print(f"got task: {name=} {tid=}")
        rc, meta, result = train(meta, model)
        submitted = agent.submit_result(task_id=tid, model=result, meta=meta, rc=rc)
        print(f"result submitted: {submitted}")

    agent.stop()


def train(meta, model):
    current_round = meta.get(ModelMetaKey.CURRENT_ROUND)
    total_rounds = meta.get(ModelMetaKey.TOTAL_ROUND)

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
    return RC.OK, {ModelMetaKey.NUM_STEPS_CURRENT_ROUND: 1}, np_data


if __name__ == "__main__":
    main()
