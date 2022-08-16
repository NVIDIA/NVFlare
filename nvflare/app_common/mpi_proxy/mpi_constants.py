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

MPI_PROXY_TOPIC = "_mpi_proxy"


class MpiFields:
    BUFFER = "_buffer"
    DATA_TYPE = "_data_type"
    SEQUENCE_NUMBER = "_sequence_number"
    RANK = "_rank"
    ROOT = "_root"
    WORLD_SIZE = "_world_size"
    REDUCE_OPERATION = "_reduce_operation"
    MPI_FUNC = "_mpi_func"


class MpiFunctions:
    ALL_REDUCE = "_all_reduce"
    ALL_GATHER = "_all_gather"
    BROADCAST = "_broadcast"

