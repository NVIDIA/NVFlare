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

from nvflare.fuel.f3.streaming.byte_streamer import ByteStreamer, reliable_retry_scheduler
from nvflare.fuel.f3.streaming.download_service import DownloadService
from nvflare.fuel.f3.streaming.stream_utils import stream_shutdown


def shutdown_f3_streaming() -> None:
    """Stop process-global F3 services in dependency order.

    Keep the Cell alive while this function runs: an already-admitted retry or
    callback can still need it. Every stage is attempted and is idempotent so a
    partially completed shutdown can be retried safely.
    """
    errors = []
    for name, shutdown in (
        ("download service", DownloadService.shutdown),
        ("active byte streams", ByteStreamer.shutdown),
        ("reliable retry scheduler", reliable_retry_scheduler.shutdown),
        ("stream executors", stream_shutdown),
    ):
        try:
            shutdown()
        except Exception as e:
            errors.append((name, e))
    if errors:
        names = ", ".join(name for name, _ in errors)
        raise RuntimeError(f"failed to stop F3 streaming services: {names}") from errors[0][1]
