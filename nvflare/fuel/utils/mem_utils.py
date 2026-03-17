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

import logging
import os

# Evaluated once at import time — restart the process to toggle.
# Set NVFLARE_CLIENT_MEMORY_PROFILE=1 (or "true"/"yes") to enable RSS logging.
# The name is intentionally client-scoped: only the client-side APIs call log_rss,
# so server processes are unaffected even if the env var is set process-wide.
_ENABLED = os.environ.get("NVFLARE_CLIENT_MEMORY_PROFILE", "").lower() in ("1", "true", "yes")

_logger = logging.getLogger(__name__)


def log_rss(tag: str) -> None:
    """Log process RSS memory if NVFLARE_CLIENT_MEMORY_PROFILE env var is set.

    Zero-overhead no-op when not enabled — only cost is a single boolean check.
    psutil is imported lazily so it is not a hard dependency.
    In subprocess mode, ExProcessClientAPI.init() loads log_config.json before
    any receive/send calls, so logger.info() is fully configured by the time
    this function runs.

    Usage::

        NVFLARE_CLIENT_MEMORY_PROFILE=1 python train.py

    Args:
        tag: label for the log line, e.g. ``"round=3 after_send"``
    """
    if not _ENABLED:
        return
    try:
        import psutil

        rss_mb = psutil.Process().memory_info().rss / 1024 / 1024
        _logger.info(f"[RSS] {tag}: {rss_mb:.1f} MB")
    except Exception:
        pass
