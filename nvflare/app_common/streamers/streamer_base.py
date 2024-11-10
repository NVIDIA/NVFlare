# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from nvflare.apis.stream_shareable import StreamMeta, StreamMetaKey


class StreamerBase:
    @staticmethod
    def get_channel(meta: StreamMeta):
        return meta.get(StreamMetaKey.CHANNEL)

    @staticmethod
    def get_topic(meta: StreamMeta):
        return meta.get(StreamMetaKey.TOPIC)

    @staticmethod
    def get_rc(meta: StreamMeta):
        return meta.get(StreamMetaKey.RC)
