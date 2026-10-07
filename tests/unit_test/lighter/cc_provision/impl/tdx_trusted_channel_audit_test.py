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


import inspect
import json
import os
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "examples/devops/coco/acceptance"))
from trusted_channel_audit import install  # noqa: E402

SECRET = "PRIVATE_KEY_JWT_ATTESTED_EVIDENCE_SECRET"


class Secret:
    def __str__(self):
        raise AssertionError("Never stringify secret object")

    def __eq__(self, other):
        raise AssertionError("Never compare secret object")


class Response:
    def __init__(self, code="ok", error=None, payload=None):
        self.code = code
        self.error = error
        self.payload = payload

    def get_header(self, name):
        if name == "cn__return_code":
            return self.code
        if name == "cn__error":
            return self.error
        raise AssertionError("Unexpected header")


def setup(tmp_path, response=None, error=None):
    tmp_path.chmod(0o700)
    calls = []

    class Cell:
        def _send_request(
            self,
            channel,
            target,
            topic,
            request,
            timeout=10.0,
            secure=False,
            optional=False,
            abort_signal=None,
            progress_wait_cb=None,
            num_receivers=1,
            receiver_ids=None,
            fobs_ctx_props=None,
        ):
            calls.append(
                (
                    channel,
                    target,
                    topic,
                    request,
                    timeout,
                    secure,
                    optional,
                    abort_signal,
                    progress_wait_cb,
                    num_receivers,
                    receiver_ids,
                    fobs_ctx_props,
                )
            )
            if error is not None:
                raise error
            return response

    original = Cell._send_request
    path = tmp_path / "stream.jsonl"
    session, saved = install(Cell, path, ["site-1", "site-2"])
    assert saved is original
    return Cell(), calls, path, session, saved


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize(
    "topic,target,payload",
    [
        ("request_fresh_token", "site-2", {"site_name": "site-2", "cc_info": [{"_cc_token": Secret()}]}),
        ("get_sites", "server", {"sites": [["PRIVATE_ROUTE_1", "site-1"], ["PRIVATE_ROUTE_2", "site-2"]]}),
    ],
)
def test_all_positional_and_keyword_arguments_response_signature_preserved(tmp_path, topic, target, payload):
    response = Response(payload=payload)
    cell, calls, path, session, saved = setup(tmp_path, response)
    request = Secret()
    abort = object()

    def progress():
        return False

    ids = object()
    ctx = object()
    assert inspect.signature(cell._send_request) == inspect.signature(saved.__get__(cell))
    assert (
        cell._send_request("cc_validation", target, topic, request, 45.0, True, True, abort, progress, 2, ids, ctx)
        is response
    )
    assert calls == [("cc_validation", target, topic, request, 45.0, True, True, abort, progress, 2, ids, ctx)]
    assert (
        cell._send_request(
            channel="cc_validation",
            target=target,
            topic=topic,
            request=request,
            timeout=45.0,
            secure=True,
            optional=True,
            abort_signal=abort,
            progress_wait_cb=progress,
            num_receivers=2,
            receiver_ids=ids,
            fobs_ctx_props=ctx,
        )
        is response
    )
    assert len(calls) == 2
    session.close()
    data = rows(path)
    assert data[1]["request_id"] == data[2]["request_id"] == 1 and data[3]["request_id"] == data[4]["request_id"] == 2
    assert data[2]["topic"] == topic and data[2]["return_code"] == "ok"
    assert SECRET not in path.read_text() and "PRIVATE_ROUTE" not in path.read_text()
    assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    "code,error,categories",
    [
        ("process_exception", "Failed to generate tokens " + SECRET, ["generation_failed"]),
        ("process_exception", "Failed to generate token: " + SECRET, ["generation_handler_exception"]),
        ("timeout", "timed out " + SECRET, ["timeout_literal"]),
        ("PRIVATE_CODE", SECRET, []),
        ("ok", "", []),
        ("ok", SECRET * 100, []),
    ],
)
def test_closed_codes_literal_categories_no_error_values_exported(tmp_path, code, error, categories):
    cell, calls, path, session, saved = setup(
        tmp_path, Response(code, error, {"site_name": Secret(), "cc_info": [Secret()]})
    )
    assert cell._send_request("cc_validation", "site-1", "request_fresh_token", Secret()) is not None
    session.close()
    row = rows(path)[2]
    assert row["error_literal_categories"] == categories
    assert row["reply_structure"]["site_name_matches_target"] is False
    assert row["reply_structure"]["cc_info_count"] == 1
    assert row["return_code"] == (code if code != "PRIVATE_CODE" else "unknown_return_code")
    assert SECRET not in path.read_text() and "PRIVATE_CODE" not in path.read_text()


@pytest.mark.parametrize("error", [RuntimeError(SECRET), ValueError(SECRET), KeyboardInterrupt(SECRET)])
def test_exception_identity_preserved_without_messages(tmp_path, error):
    cell, calls, path, session, saved = setup(tmp_path, error=error)
    with pytest.raises(type(error)) as caught:
        cell._send_request("cc_validation", "site-1", "request_fresh_token", Secret())
    assert caught.value is error
    session.close()
    assert rows(path)[2]["exception_class"] == (
        type(error).__name__ if isinstance(error, (RuntimeError, ValueError)) else "other"
    )
    assert SECRET not in path.read_text()


@pytest.mark.parametrize(
    "channel,target,topic",
    [
        ("other", "site-1", "request_fresh_token"),
        ("cc_validation", "SECRET_TARGET", "request_fresh_token"),
        ("cc_validation", "site-1", "other"),
        ("cc_validation", "site-1", "get_sites"),
    ],
)
def test_unrelated_requests_unobserved(tmp_path, channel, target, topic):
    response = Response()
    cell, calls, path, session, saved = setup(tmp_path, response)
    assert cell._send_request(channel, target, topic, Secret()) is response
    session.close()
    assert len(rows(path)) == 2


def test_fork_closed_output_or_write_failure_never_changes_return(tmp_path, monkeypatch):
    response = Response()
    cell, calls, path, session, saved = setup(tmp_path, response)
    with monkeypatch.context() as m:
        m.setattr(os, "getpid", lambda: session.pid + 1)
        assert cell._send_request("cc_validation", "site-1", "request_fresh_token", Secret()) is response
    assert len(rows(path)) == 1
    os.close(session.fd)
    assert cell._send_request("cc_validation", "site-1", "request_fresh_token", Secret()) is response
    assert session.failed
    session.fd = None
    session.close()
    assert cell._send_request("cc_validation", "site-1", "request_fresh_token", Secret()) is response


def test_safe_structure_bounds_unreadable_payload_and_unknown_site(tmp_path):
    from trusted_channel_audit import reply_structure

    class Unreadable:
        @property
        def payload(self):
            raise RuntimeError(SECRET)

    assert reply_structure(Unreadable(), "get_sites", "server", {"site-1", "site-2"}) == {"payload_kind": "unreadable"}
    assert reply_structure(None, "get_sites", "server", {"site-1", "site-2"}) == {"payload_kind": "no_response"}
    for payload in [
        {"sites": [["SECRET_ROUTE", "SECRET_SITE"]]},
        {"sites": [[Secret(), "site-1"]]},
        {"sites": [[]] * 33},
    ]:
        result = reply_structure(Response(payload=payload), "get_sites", "server", {"site-1", "site-2"})
        assert result["required_site_names_present"] is False
        assert "SECRET" not in json.dumps(result)
    assert (
        reply_structure(
            Response(payload={"site_name": "site-1", "cc_info": [Secret()] * 33}),
            "request_fresh_token",
            "site-1",
            {"site-1", "site-2"},
        )["cc_info_count"]
        is None
    )


def test_pinned_real_cell_stream_dispatch_reaches_wrapper_without_core_send(tmp_path):
    from nvflare.fuel.f3.cellnet.cell import Cell as RealCell
    from nvflare.fuel.f3.cellnet.cell import _is_stream_channel

    assert _is_stream_channel("cc_validation")
    cell, calls, path, session, saved = setup(
        tmp_path, Response(payload={"sites": [["site-1", "site-1"], ["site-2", "site-2"]]})
    )

    class Adapter(type(cell)):
        __getattr__ = RealCell.__getattr__

    adapter = Adapter()
    adapter.logger = types.SimpleNamespace(debug=lambda *a: None)
    adapter.core_cell = types.SimpleNamespace(
        send_request=lambda **kwargs: (_ for _ in ()).throw(AssertionError("Core path must be bypassed"))
    )
    result = adapter.send_request(
        channel="cc_validation", target="server", topic="get_sites", request=Secret(), timeout=45.0
    )
    assert result is not None and len(calls) == 1
    session.close()
    assert rows(path)[2]["topic"] == "get_sites"


def test_bad_signature_or_defaults_and_output_overwrite_refused(tmp_path):
    tmp_path.chmod(0o700)

    class Bad:
        def _send_request(self, channel, target):
            pass

    with pytest.raises(ValueError):
        install(Bad, tmp_path / "bad.jsonl", ["site-1", "site-2"])
    cell, calls, path, session, saved = setup(tmp_path, Response())
    session.close()
    with pytest.raises(FileExistsError):
        install(type(cell), path, ["site-1", "site-2"])
