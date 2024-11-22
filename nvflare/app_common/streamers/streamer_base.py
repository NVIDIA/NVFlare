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
from nvflare.apis.streaming import StreamContext, StreamContextKey


class StreamerBase:
    """This is the base class for all future streamers.
    This base class provides methods for accessing common properties in the StreamContext.
    When a streamer class is defined as a subclass of this base, then all such StreamContext accessing methods
    will be inherited.
    """

    @staticmethod
    def get_channel(ctx: StreamContext):
        return ctx.get(StreamContextKey.CHANNEL)

    @staticmethod
    def get_topic(ctx: StreamContext):
        return ctx.get(StreamContextKey.TOPIC)

    @staticmethod
    def get_rc(ctx: StreamContext):
        return ctx.get(StreamContextKey.RC)
