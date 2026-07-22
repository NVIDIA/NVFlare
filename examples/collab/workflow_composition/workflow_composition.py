# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Composing workflows, resource directories, and artifacts.

A single ``@collab.main`` method chains two workflows — cyclic relay first,
then parallel FedAvg seeded with the relay's result. Along the way it shows:

  - ``@collab.init``: the initial model is loaded from a data resource dir
    shipped with the job (``set_server_resource_dirs``) before the main
    workflow starts
  - workspace artifacts: intermediate and final models are saved under
    ``collab.workspace.get_work_dir()``
  - ``@collab.final``: a hook that runs after the main workflow completes

Run:
    python -m collab.workflow_composition.workflow_composition --num-clients 3
"""

import os

import numpy as np
from collab.workflow_composition.avg_para import NPFedAvgParallel
from collab.workflow_composition.cyclic import NPCyclic
from collab.workflow_composition.np_utils import load_np_model, save_np_model
from collab.workflow_composition.runner import make_parser, run_recipe
from collab.workflow_composition.trainer import NPTrainer

from nvflare.collab import CollabRecipe, collab
from nvflare.fuel.utils.log_utils import get_obj_logger

INITIAL_MODEL_FILE = "initial_model.npy"


class Controller:

    def __init__(self, initial_model, cyclic_rounds, avg_rounds):
        self.initial_model = initial_model
        self.cyclic_rounds = cyclic_rounds
        self.avg_rounds = avg_rounds
        self.logger = get_obj_logger(self)

    @collab.init
    def load_initial_model(self):
        # initial_model is the name of a file in the "data" resource dir
        # shipped with the job; load it before the workflow starts.
        if isinstance(self.initial_model, str):
            file_name = os.path.join(collab.workspace.get_resource_dir("data"), self.initial_model)
            self.initial_model = load_np_model(file_name)
            self.logger.info(f"loaded initial model from {file_name}: {self.initial_model}")

    @collab.main
    def run(self):
        self.logger.info("running cyclic ...")
        ctl = NPCyclic(self.initial_model, num_rounds=self.cyclic_rounds)
        result = ctl.execute()

        file_name = os.path.join(collab.workspace.get_work_dir(), "cyclic_model.npy")
        save_np_model(result, file_name)
        self.logger.info(f"[{collab.call_info}]: saved cyclic model {result} to {file_name}")

        self.logger.info("running fed-avg ...")
        ctl = NPFedAvgParallel(initial_model=result, num_rounds=self.avg_rounds)
        return ctl.execute()

    @collab.final
    def save_result(self):
        final_result = collab.get_result()
        file_name = os.path.join(collab.workspace.get_work_dir(), "final_model.npy")
        save_np_model(final_result, file_name)
        self.logger.info(f"[{collab.call_info}]: saved final model {final_result} to {file_name}")


def prepare_data_dir(root: str) -> str:
    data_dir = os.path.join(root, "workflow_composition_data")
    os.makedirs(data_dir, exist_ok=True)
    save_np_model(np.array([1.0, 2.0, 3.0]), os.path.join(data_dir, INITIAL_MODEL_FILE))
    return data_dir


def make_recipe(args):
    recipe = CollabRecipe(
        job_name="collab_workflow_composition",
        # The initial model is a file name resolved against the "data"
        # resource dir shipped with the job.
        server=Controller(initial_model=INITIAL_MODEL_FILE, cyclic_rounds=2, avg_rounds=3),
        client=NPTrainer(delta=1.0),
        min_clients=args.num_clients,
        sync_task_timeout=60,
    )
    recipe.set_server_resource_dirs({"data": prepare_data_dir(args.job_root)})
    return recipe


def main():
    parser = make_parser("Workflow composition: cyclic relay feeding parallel FedAvg")
    parser.set_defaults(num_clients=3)
    args = parser.parse_args()
    run_recipe(make_recipe(args), args)


if __name__ == "__main__":
    main()
