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


class CollectiveCommandKey:
    TIMEOUT = "_timeout"


class CollectiveCommEvent:
    FAILED = "_failed"


class CollectiveCommShareableHeader:
    ALL_REQUESTS = "_all_requests"
    IS_COLLECTIVE_AUX = "_is_collective_aux"
    WORLD_SIZE = "_world_size"
    TIMEOUT = "_timeout"

    BUFFER = "_buffer"
    SEQUENCE_NUMBER = "_sequence_number"
    RANK = "_rank"
    COLLECTIVE_FUNC = "_collective_func"
    REDUCE_FUNCTION = "_reduce_function"
    ROOT = "_root"


class CollectiveCommRequestTopic:
    ALL_REDUCE = "_all_reduce"
    ALL_GATHER = "_all_gather"
    BROADCAST = "_broadcast"


class CollectiveCommHandleError(Exception):
    pass
