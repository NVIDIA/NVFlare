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

import json

from nvflare.private.fed.server.sys_cmd import SystemCommandModule


class _MockConnection:
    def __init__(self):
        self._props = {}
        self.errors = []
        self.strings = []
        self.tables = []
        self.dicts = []

    def get_prop(self, key, default=None):
        return self._props.get(key, default)

    def append_error(self, msg, meta=None):
        self.errors.append((msg, meta))

    def append_string(self, msg, meta=None):
        self.strings.append((msg, meta))

    def append_table(self, headers, name=None):
        table = _MockTable(headers=headers, name=name)
        self.tables.append(table)
        return table

    def append_dict(self, data, meta=None):
        self.dicts.append((data, meta))


class _MockTable:
    def __init__(self, headers, name=None):
        self.headers = headers
        self.name = name
        self.rows = []

    def add_row(self, row, meta=None):
        self.rows.append((row, meta))


class _MockReplyMessage:
    def __init__(self, body):
        self.body = body

    def get_header(self, name):
        return None


class _MockClientReply:
    def __init__(self, client_name, body):
        self.client_name = client_name
        self.reply = _MockReplyMessage(body)


def test_report_resources_all_includes_server_and_clients(monkeypatch):
    module = SystemCommandModule()
    conn = _MockConnection()
    monkeypatch.setattr(
        module,
        "send_request_to_clients",
        lambda conn, message: [_MockClientReply("site-1", json.dumps({"gpu": "1"}))],
    )

    module.report_resources(conn, ["report_resources", "all"])

    assert conn.errors == []
    assert len(conn.tables) == 1
    rows = [row for row, _meta in conn.tables[0].rows]
    assert ["server", "unlimited"] in rows
    assert ["site-1", "{'gpu': '1'}"] in rows


def test_report_resources_all_keeps_server_when_no_clients_reply(monkeypatch):
    module = SystemCommandModule()
    conn = _MockConnection()
    monkeypatch.setattr(module, "send_request_to_clients", lambda conn, message: [])

    module.report_resources(conn, ["report_resources", "all"])

    assert conn.errors == []
    assert len(conn.tables) == 1
    rows = [row for row, _meta in conn.tables[0].rows]
    assert rows == [["server", "unlimited"]]


def test_report_version_all_includes_server_and_clients(monkeypatch):
    module = SystemCommandModule()
    conn = _MockConnection()
    monkeypatch.setattr(
        module,
        "send_request_to_clients",
        lambda conn, message: [_MockClientReply("site-1", json.dumps({"version": "2.8.0"}))],
    )

    module.report_version(conn, ["report_version", "all"])

    assert conn.errors == []
    assert len(conn.dicts) == 1
    payload, _meta = conn.dicts[0]
    assert payload["server"]["version"]
    assert payload["site-1"] == {"version": "2.8.0"}


def test_report_version_invalid_target_emits_error_string():
    module = SystemCommandModule()
    conn = _MockConnection()

    module.report_version(conn, ["report_version", "bogus"])

    assert conn.dicts == []
    assert conn.strings
    assert "invalid target type" in conn.strings[0][0]
