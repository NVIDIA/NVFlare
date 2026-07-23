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

"""Composing workflows and artifacts.

A single ``@collab.main`` method chains two workflows — cyclic relay first,
then parallel FedAvg seeded with the relay's result. Along the way it shows:

  - ``@collab.init``: the initial model is prepared before the main workflow
  - workspace artifacts: intermediate and final models are saved under
    the standard NVFlare run directory
  - ``@collab.final``: a hook that runs after the main workflow completes

Run:
    python -m collab.workflow_composition.workflow_composition --num-clients 3
"""

import argparse
import os

import numpy as np
from collab.workflow_composition.avg_para import NPFedAvgParallel
from collab.workflow_composition.cyclic import NPCyclic
from collab.workflow_composition.np_utils import save_np_model
from collab.workflow_composition.trainer import NPTrainer

from nvflare.collab import CollabRecipe, collab, simple_logging
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.recipe import SimEnv


class Controller:

    def __init__(self, initial_model, cyclic_rounds, avg_rounds):
        self.initial_model = initial_model
        self.cyclic_rounds = cyclic_rounds
        self.avg_rounds = avg_rounds
        self.logger = get_obj_logger(self)

    @collab.init
    def prepare_initial_model(self):
        self.initial_model = np.asarray(self.initial_model, dtype=np.float64)
        self.logger.info(f"prepared initial model: {self.initial_model}")

    @collab.main
    def run(self):
        self.logger.info("running cyclic ...")
        ctl = NPCyclic(self.initial_model, num_rounds=self.cyclic_rounds)
        result = ctl.execute()

        run_dir = collab.workspace.get_run_dir(collab.fl_ctx.get_job_id())
        file_name = os.path.join(run_dir, "cyclic_model.npy")
        save_np_model(result, file_name)
        self.logger.info(f"[{collab.call_info}]: saved cyclic model {result} to {file_name}")

        self.logger.info("running fed-avg ...")
        ctl = NPFedAvgParallel(initial_model=result, num_rounds=self.avg_rounds)
        return ctl.execute()

    @collab.final
    def save_result(self):
        final_result = collab.get_result()
        run_dir = collab.workspace.get_run_dir(collab.fl_ctx.get_job_id())
        file_name = os.path.join(run_dir, "final_model.npy")
        save_np_model(final_result, file_name)
        self.logger.info(f"[{collab.call_info}]: saved final model {final_result} to {file_name}")


def make_recipe(args):
    return CollabRecipe(
        job_name="collab_workflow_composition",
        server=Controller(initial_model=[1.0, 2.0, 3.0], cyclic_rounds=2, avg_rounds=3),
        client=NPTrainer(delta=1.0),
        min_clients=args.num_clients,
        sync_task_timeout=60,
    )


def main():
    parser = argparse.ArgumentParser(description="Workflow composition: cyclic relay feeding parallel FedAvg")
    parser.add_argument("--num-clients", type=int, default=3)
    args = parser.parse_args()
    simple_logging()
    run = make_recipe(args).execute(SimEnv(num_clients=args.num_clients))
    print("Job Status:", run.get_status())
    print("Results at:", run.get_result())


if __name__ == "__main__":
    main()
