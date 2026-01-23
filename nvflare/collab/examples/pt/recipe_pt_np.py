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
from nvflare.app_common.decomposers.numpy_decomposers import NumpyArrayDecomposer
from nvflare.app_opt.pt.decomposers import TensorDecomposer
from nvflare.collab.examples import export_recipe
from nvflare.collab.examples.pt.pt_np import PTFedAvgMixed, PTTrainer
from nvflare.collab.sys.recipe import CollabRecipe


def main():
    export_recipe("collab_pt_np", _make_recipe)


def _make_recipe(job_name):
    init_model = {
        "x": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        "y": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        "z": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    }

    recipe = CollabRecipe(
        job_name=job_name,
        server=PTFedAvgMixed(
            pt_model=init_model,
            np_model=init_model,
            num_rounds=2,
        ),
        client=PTTrainer(delta=1.0),
    )
    recipe.add_decomposers([TensorDecomposer(), NumpyArrayDecomposer()])
    return recipe


if __name__ == "__main__":
    main()
