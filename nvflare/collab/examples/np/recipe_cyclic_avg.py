# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import os

from nvflare.collab import fox
from nvflare.collab.examples import export_recipe
from nvflare.collab.examples.np.algos.client import NPTrainer
from nvflare.collab.examples.np.algos.strategies.avg_para import NPFedAvgParallel
from nvflare.collab.examples.np.algos.strategies.cyclic import NPCyclic
from nvflare.collab.examples.np.algos.utils import save_np_model
from nvflare.collab.sys.recipe import CollabRecipe
from nvflare.fuel.utils.log_utils import get_obj_logger


class Controller:

    def __init__(
        self,
        initial_model,
        cyclic_rounds,
        avg_rounds,
    ):
        self.initial_model = initial_model
        self.cyclic_rounds = cyclic_rounds
        self.avg_rounds = avg_rounds
        self.logger = get_obj_logger(self)

    @fox.algo
    def run(self):
        self.logger.info("running cyclic ...")
        ctl = NPCyclic(self.initial_model, num_rounds=self.cyclic_rounds)
        result = ctl.execute()

        file_name = os.path.join(fox.workspace.get_work_dir(), "cyclic_model.npy")
        save_np_model(result, file_name)
        self.logger.info(f"[{fox.call_info}]: saved cyclic model {result} to {file_name}")

        self.logger.info("running fed-avg ...")
        ctl = NPFedAvgParallel(initial_model=result, num_rounds=self.avg_rounds)
        return ctl.execute()

    @fox.final
    def save_result(self):
        final_result = fox.get_result()
        file_name = os.path.join(fox.workspace.get_work_dir(), "final_model.npy")
        save_np_model(final_result, file_name)
        self.logger.info(f"[{fox.call_info}]: saved final model {final_result} to {file_name}")


def main():
    export_recipe("fox_cyclic_avg", _make_recipe)


def _make_recipe(job_name):
    return CollabRecipe(
        job_name=job_name,
        server=Controller(
            initial_model=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            cyclic_rounds=2,
            avg_rounds=3,
        ),
        client=NPTrainer(delta=1.0),
    )


if __name__ == "__main__":
    main()
