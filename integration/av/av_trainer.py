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
import os.path
import pickle

from integration.av.av_model import META_IS_DIFF, AVModel, AVModelDecomposer
from nvflare.client.defs import RC, AgentClosed, MetaKey, Task, TaskResult
from nvflare.client.ipc_agent import IPCAgent


def main():

    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", "-w", type=str, help="workspace folder", required=False, default=".")
    parser.add_argument("--site_name", "-s", type=str, help="flare site name", required=True)
    parser.add_argument("--agent_id", "-a", type=str, help="agent id", required=True)
    parser.add_argument("--site_url", "-u", type=str, help="flare site url", required=True)

    args = parser.parse_args()

    agent = IPCAgent(
        flare_site_url=args.site_url,
        flare_site_name=args.site_name,
        agent_id=args.agent_id,
        workspace_dir=args.workspace,
        secure_mode=True,
        submit_result_timeout=2.0,
        flare_site_heartbeat_timeout=None,  # waiting forever
    )
    AVModelDecomposer.register_decomposers()

    snapshot_file_name = f"{args.site_name}_{args.agent_id}_snapshot.dat"

    agent.start()
    if os.path.exists(snapshot_file_name):
        # finish previous round
        print(f"recover training from {snapshot_file_name}")
        task = pickle.load(open(snapshot_file_name, "rb"))
        rc, meta, result = train(task)
        agent.submit_result(TaskResult(data=result, meta=meta, return_code=rc))
        os.remove(snapshot_file_name)

    done = False
    while not done:
        print("getting task ...")
        try:
            task = agent.get_task()
        except AgentClosed:
            print("agent closed")
            break

        print(f"got task: {task}")

        # create a snapshot, so we can recover in case the training fails
        pickle.dump(task, open(snapshot_file_name, "wb"))

        current_round = task.meta.get(MetaKey.CURRENT_ROUND)

        # simulate crash
        if current_round == 10:
            print(f"training crashed at round {current_round}")
            done = True
            continue

        rc, meta, result = train(task)
        if current_round == 20:
            rc = RC.EARLY_TERMINATION
            print(f"Early termination at round {current_round}")
            done = True
        submitted = agent.submit_result(TaskResult(data=result, meta=meta, return_code=rc))
        os.remove(snapshot_file_name)
        print(f"result submitted: {submitted}")

    print("stopping agent")
    agent.stop()
    print("TRAINER DONE")


def train(task: Task):
    print(f"got task: {task.meta=} {task.data=}")
    meta = task.meta
    model = task.data

    if not isinstance(model, AVModel):
        raise ValueError(f"task data must be AVModel but got {type(model)}")
    layers = model.free_layers

    current_round = meta.get(MetaKey.CURRENT_ROUND)
    total_rounds = meta.get(MetaKey.TOTAL_ROUND)

    print(f"Layers: \n{layers}")
    print(f"Current Round: {current_round}")
    print(f"Total Rounds: {total_rounds}")

    # the "layers" is a dict of layer_name => list of numbers
    # Doing some dummy training.
    if layers:
        for _, w in layers.items():
            for i, v in enumerate(w):
                w[i] += 1.0
    else:
        print("No layers found.")
        return RC.BAD_TASK_DATA, None, None

    # Save local numpy model
    print(f"Layers after training: {layers}")

    meta = {MetaKey.NUM_STEPS_CURRENT_ROUND: 1, META_IS_DIFF: False}
    return RC.OK, meta, AVModel({}, {}, layers)


if __name__ == "__main__":
    main()
