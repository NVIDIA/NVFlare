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
from nvflare.collab.examples.np.mains.strategies.cyclic import NPCyclic
from nvflare.collab.sys.recipe import CollabRecipe


class CyclicRecipe(CollabRecipe):

    def __init__(
        self,
        job_name,
        initial_model,
        num_rounds,
        client,
    ):
        CollabRecipe.__init__(
            self,
            job_name,
            server=NPCyclic(initial_model, num_rounds),
            client=client,
        )
