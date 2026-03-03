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

"""Unit tests for ViaDownloaderDecomposer._create_downloader() (Fix 9).

Fix 9 — Make _MIN_DOWNLOAD_TIMEOUT configurable via job config:
    Root Cause: _MIN_DOWNLOAD_TIMEOUT was hardcoded to 60s.  For large models
    (5 GiB+) with slow networks, the gap between chunk N completing and chunk
    N+1 being requested can exceed 60s, causing _monitor_tx to kill the
    transaction mid-download ("ref not found" errors).
    Fix: read min_download_timeout from job config via acu.get_positive_float_var()
    using the same ConfigVarName + config_var_prefix pattern as DOWNLOAD_CHUNK_SIZE
    and STREAMING_PER_REQUEST_TIMEOUT.  The module-level constant _MIN_DOWNLOAD_TIMEOUT
    remains as the fallback default (60s) when the key is absent.

CONTRACT:
- When no job config value is set → use _MIN_DOWNLOAD_TIMEOUT (60s) as min_timeout
- When job config sets np_min_download_timeout=600 → min_timeout=600
- msg_root_ttl > min_timeout → timeout = msg_root_ttl (not clamped up)
- msg_root_ttl < min_timeout → timeout floored to min_timeout
- msg_root_ttl absent → timeout = min_timeout
- ObjectDownloader receives the computed timeout value
"""

from unittest.mock import MagicMock, patch

from nvflare.apis.fl_constant import ConfigVarName
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.fobs.decomposers.via_downloader import _MIN_DOWNLOAD_TIMEOUT, ViaDownloaderDecomposer, _CtxKey

# ---------------------------------------------------------------------------
# Minimal concrete subclass — only _create_downloader is under test
# ---------------------------------------------------------------------------


class _FakeDecomposer(ViaDownloaderDecomposer):
    """Concrete stub of ViaDownloaderDecomposer for testing _create_downloader."""

    def __init__(self, config_var_prefix="np_"):
        super().__init__(max_chunk_size=1024 * 1024, config_var_prefix=config_var_prefix)

    def to_downloadable(self, items, max_chunk_size, fobs_ctx):
        return MagicMock()

    def download(self, from_fqcn, ref_id, per_request_timeout, cell, secure=False, optional=False, abort_signal=None):
        return None, {}

    def get_download_dot(self):
        return 99

    def native_decompose(self, target, manager=None):
        return b""

    def native_recompose(self, data, manager=None):
        return data

    def supported_type(self):
        return object


def _make_fobs_ctx(msg_root_id=None, msg_root_ttl=None, cell=None):
    ctx = {}
    if msg_root_id is not None:
        ctx[_CtxKey.MSG_ROOT_ID] = msg_root_id
    if msg_root_ttl is not None:
        ctx[_CtxKey.MSG_ROOT_TTL] = msg_root_ttl
    ctx[fobs.FOBSContextKey.CELL] = cell or MagicMock()
    return ctx


# ---------------------------------------------------------------------------
# Fix 9: min_timeout comes from job config via acu.get_positive_float_var
# ---------------------------------------------------------------------------


class TestCreateDownloaderFix9:

    def test_default_min_timeout_used_when_no_config(self, monkeypatch):
        """When job config has no np_min_download_timeout, default 60s is used as floor."""
        decomposer = _FakeDecomposer()
        captured = []

        def fake_object_downloader(**kwargs):
            captured.append(kwargs["timeout"])
            od = MagicMock()
            od.add_object = MagicMock()
            return od

        # acu returns default (no job config value set)
        monkeypatch.setattr(
            "nvflare.fuel.utils.fobs.decomposers.via_downloader.acu.get_positive_float_var",
            lambda name, default: default,
        )
        with patch(
            "nvflare.fuel.utils.fobs.decomposers.via_downloader.ObjectDownloader",
            side_effect=fake_object_downloader,
        ):
            ctx = _make_fobs_ctx()
            decomposer._create_downloader(ctx)

        assert len(captured) == 1
        assert captured[0] == _MIN_DOWNLOAD_TIMEOUT

    def test_job_config_min_timeout_overrides_default(self, monkeypatch):
        """When job config sets np_min_download_timeout=600, ObjectDownloader gets 600."""
        decomposer = _FakeDecomposer()
        captured = []

        def fake_object_downloader(**kwargs):
            captured.append(kwargs["timeout"])
            od = MagicMock()
            od.add_object = MagicMock()
            return od

        # Simulate job config returning 600 for the np_min_download_timeout key
        def fake_get_positive_float_var(name, default):
            if name == "np_min_download_timeout":
                return 600.0
            return default

        monkeypatch.setattr(
            "nvflare.fuel.utils.fobs.decomposers.via_downloader.acu.get_positive_float_var",
            fake_get_positive_float_var,
        )
        with patch(
            "nvflare.fuel.utils.fobs.decomposers.via_downloader.ObjectDownloader",
            side_effect=fake_object_downloader,
        ):
            ctx = _make_fobs_ctx()
            decomposer._create_downloader(ctx)

        assert len(captured) == 1
        assert captured[0] == 600.0

    def test_msg_root_ttl_above_min_not_clamped(self, monkeypatch):
        """msg_root_ttl larger than min_timeout is passed through as-is."""
        decomposer = _FakeDecomposer()
        captured = []

        def fake_object_downloader(**kwargs):
            captured.append(kwargs["timeout"])
            od = MagicMock()
            od.add_object = MagicMock()
            return od

        monkeypatch.setattr(
            "nvflare.fuel.utils.fobs.decomposers.via_downloader.acu.get_positive_float_var",
            lambda name, default: default,  # min_timeout = 60
        )
        with patch(
            "nvflare.fuel.utils.fobs.decomposers.via_downloader.ObjectDownloader",
            side_effect=fake_object_downloader,
        ):
            ctx = _make_fobs_ctx(msg_root_ttl=300.0)
            decomposer._create_downloader(ctx)

        assert captured[0] == 300.0  # not raised to min_timeout

    def test_msg_root_ttl_below_min_is_floored(self, monkeypatch):
        """msg_root_ttl smaller than min_timeout is floored to min_timeout."""
        decomposer = _FakeDecomposer()
        captured = []

        def fake_object_downloader(**kwargs):
            captured.append(kwargs["timeout"])
            od = MagicMock()
            od.add_object = MagicMock()
            return od

        # Simulate job config setting min_timeout to 600
        monkeypatch.setattr(
            "nvflare.fuel.utils.fobs.decomposers.via_downloader.acu.get_positive_float_var",
            lambda name, default: 600.0 if "min_download_timeout" in name else default,
        )
        with patch(
            "nvflare.fuel.utils.fobs.decomposers.via_downloader.ObjectDownloader",
            side_effect=fake_object_downloader,
        ):
            ctx = _make_fobs_ctx(msg_root_ttl=30.0)  # much lower than min
            decomposer._create_downloader(ctx)

        assert captured[0] == 600.0  # floored to min_timeout

    def test_no_msg_root_ttl_uses_min_timeout(self, monkeypatch):
        """When msg_root_ttl is absent, ObjectDownloader gets min_timeout."""
        decomposer = _FakeDecomposer()
        captured = []

        def fake_object_downloader(**kwargs):
            captured.append(kwargs["timeout"])
            od = MagicMock()
            od.add_object = MagicMock()
            return od

        monkeypatch.setattr(
            "nvflare.fuel.utils.fobs.decomposers.via_downloader.acu.get_positive_float_var",
            lambda name, default: 120.0 if "min_download_timeout" in name else default,
        )
        with patch(
            "nvflare.fuel.utils.fobs.decomposers.via_downloader.ObjectDownloader",
            side_effect=fake_object_downloader,
        ):
            ctx = _make_fobs_ctx()  # no msg_root_ttl
            decomposer._create_downloader(ctx)

        assert captured[0] == 120.0

    def test_config_var_name_uses_prefix(self, monkeypatch):
        """acu.get_positive_float_var must be called with the prefixed key."""
        decomposer = _FakeDecomposer(config_var_prefix="np_")
        names_queried = []

        def tracking_get(name, default):
            names_queried.append(name)
            return default

        monkeypatch.setattr(
            "nvflare.fuel.utils.fobs.decomposers.via_downloader.acu.get_positive_float_var",
            tracking_get,
        )
        with patch("nvflare.fuel.utils.fobs.decomposers.via_downloader.ObjectDownloader", return_value=MagicMock()):
            ctx = _make_fobs_ctx()
            decomposer._create_downloader(ctx)

        assert f"np_{ConfigVarName.MIN_DOWNLOAD_TIMEOUT}" in names_queried
