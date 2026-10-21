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

from types import SimpleNamespace

import pytest

from tests.integration_test.fast import client_api_external_process_lifecycle_test as lifecycle


def test_paused_download_is_released_when_test_assertion_fails(monkeypatch):
    env = SimpleNamespace(site=SimpleNamespace(send_request=lambda **kwargs: object()))

    def materialize(result, site, abort_signal):
        site.send_request(channel=lifecycle.OBJ_DOWNLOADER_CHANNEL, topic=lifecycle.OBJ_DOWNLOADER_TOPIC)
        if abort_signal.triggered:
            raise RuntimeError("download cancelled")

    monkeypatch.setattr(lifecycle, "materialize_lazy_download_refs", materialize)

    with pytest.raises(AssertionError, match="injected test failure"):
        with lifecycle._paused_download(env, object(), monkeypatch) as download:
            raise AssertionError("injected test failure")

    assert download.release.is_set()
    assert download.abort.triggered
    assert download.done.is_set()
    assert not download.thread.is_alive()
    assert len(download.errors) == 1
    assert str(download.errors[0]) == "download cancelled"
