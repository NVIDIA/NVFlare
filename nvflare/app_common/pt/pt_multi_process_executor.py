# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.apis.utils.common_utils import get_open_ports
from nvflare.app_common.executors.multi_process_executor import MultiProcessExecutor


class PTMultiProcessExecutor(MultiProcessExecutor):
    def get_multi_process_command(self) -> str:
        return (
            f"{sys.executable} -m torch.distributed.run --nproc_per_node="
            + str(self.num_of_processes)
            + " --nnodes=1 --node_rank=0"
            + ' --master_addr="localhost" --master_port='
            + str(get_open_ports(1)[0])
        )
