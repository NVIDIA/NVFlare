# Copyright (c) 2021, NVIDIA CORPORATION.
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

import sys

from nvflare.app_common.executors.multi_process_executor import MultiProcessExecutor


class PTMultiProcessExecutor(MultiProcessExecutor):
    def __init__(self, executor_id=None, num_of_processes=1, components=None):
        super().__init__(executor_id, num_of_processes, components)

    def get_multi_process_command(self) -> str:
        return (
            f"{sys.executable} -m torch.distributed.run --nproc_per_node="
            + str(self.num_of_processes)
            + " --nnodes=1 --node_rank=0"
            + ' --master_addr="localhost" --master_port=1234'
        )
