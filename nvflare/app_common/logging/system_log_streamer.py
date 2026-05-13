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
import warnings

from nvflare.app_common.logging.site_log_streamer import SiteLogStreamer


class SystemLogStreamer(SiteLogStreamer):
    """Backward-compatible alias for :class:`SiteLogStreamer`."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "SystemLogStreamer is deprecated and will be removed in a future release; "
            "use SiteLogStreamer from nvflare.app_common.logging.site_log_streamer instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


__all__ = ["SystemLogStreamer"]
