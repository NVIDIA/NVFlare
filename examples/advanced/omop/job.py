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

"""Fit the GI hemorrhage model across the OMOP CDM sites with federated Newton-Raphson logistic regression."""

import argparse

from cohort import FEATURES
from prepare_data import DATA_ROOT, SITE_FRACTIONS

from nvflare.app_common.np.recipes.lr.fedavg import FedAvgLrRecipe
from nvflare.recipe import SimEnv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default=DATA_ROOT)
    args = parser.parse_args()

    recipe = FedAvgLrRecipe(
        name="omop_gi_bleed",
        min_clients=len(SITE_FRACTIONS),
        num_rounds=15,
        num_features=len(FEATURES),
        train_script="client.py",
        train_args=f"--data_root {args.data_root}",
    )
    run = recipe.execute(SimEnv(num_clients=len(SITE_FRACTIONS)))
    print("result location =", run.get_result())


if __name__ == "__main__":
    main()
