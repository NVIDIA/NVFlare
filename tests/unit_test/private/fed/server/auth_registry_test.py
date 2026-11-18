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

from nvflare.fuel.f3.endpoint import Endpoint, EndpointState
from nvflare.private.fed.server.auth_registry import NonRegularAuthRegistry


def test_re_registration_replaces_old_token_but_keeps_separate_admin_sessions():
    registry = NonRegularAuthRegistry()
    first, second = Endpoint("_admin_first"), Endpoint("_admin_second")
    registry.register("lead@test.org", "other-session", second.name, second)
    for n in range(100):
        registry.register("lead@test.org", str(n), first.name, first)
    assert registry.resolve("lead@test.org", "0") is None
    assert registry.resolve("lead@test.org", "99") == first.name
    assert registry.resolve("lead@test.org", "other-session") == second.name
    assert registry.resolve("other@test.org", "99") is None
    assert len(registry._bindings) == 2


def test_disconnect_removes_all_bindings_received_through_that_transport():
    registry = NonRegularAuthRegistry()
    relay, admin = Endpoint("relay-1"), Endpoint("_admin_session")
    for name, token, fqcn in [("relay-1", "outer", "relay-1"), ("relay-2", "inner", "relay-1.relay-2")]:
        registry.register(name, token, fqcn, relay)
    registry.register("lead@test.org", "admin", admin.name, admin)
    relay.state = EndpointState.DISCONNECTED
    registry.state_change(relay)
    assert registry.resolve("relay-1", "outer") is None
    assert registry.resolve("relay-2", "inner") is None
    assert registry.resolve("lead@test.org", "admin") == admin.name
    admin.state = EndpointState.DISCONNECTED
    registry.state_change(admin)
    assert not registry._bindings


def test_old_connection_disconnect_does_not_remove_replacement_registration():
    registry = NonRegularAuthRegistry()
    old, new = Endpoint("relay-1"), Endpoint("relay-1")
    registry.register("relay-1", "old", old.name, old)
    registry.register("relay-1", "new", new.name, new)
    old.state = EndpointState.DISCONNECTED
    registry.state_change(old)
    assert registry.resolve("relay-1", "old") is None
    assert registry.resolve("relay-1", "new") == new.name
    new.state = EndpointState.READY
    registry.state_change(new)
    assert registry.resolve("relay-1", "new") == new.name


def test_registration_finishing_after_disconnect_cannot_resurrect_or_replace_a_binding():
    registry = NonRegularAuthRegistry()
    old, new = Endpoint("relay-1"), Endpoint("relay-1")
    registry.register("relay-1", "old", old.name, old)
    old.state = EndpointState.DISCONNECTED
    registry.state_change(old)
    registry.register("relay-1", "late-old", old.name, old)
    assert not registry._bindings
    registry.register("relay-1", "new", new.name, new)
    registry.register("relay-1", "late-old", old.name, old)
    assert registry.resolve("relay-1", "late-old") is None
    assert registry.resolve("relay-1", "new") == new.name


def test_transport_replacement_prunes_old_instances_and_rejects_late_registration():
    old, new = Endpoint("_admin_first"), Endpoint("_admin_first")
    other = Endpoint("_admin_second")
    current = {old.name: old, other.name: other}
    registry = NonRegularAuthRegistry(current.get)
    registry.register("lead@test.org", "old", old.name, old)
    registry.register("lead@test.org", "other", other.name, other)
    # ConnManager replaces the endpoint before closing the old connection;
    # that old object can remain READY and receive no DISCONNECTED event.
    current[new.name] = new
    new.state = EndpointState.READY
    registry.state_change(new)
    assert registry.resolve("lead@test.org", "old") is None
    registry.register("lead@test.org", "new", new.name, new)
    registry.register("lead@test.org", "late-old", old.name, old)
    old.state = EndpointState.READY
    registry.state_change(old)
    assert registry.resolve("lead@test.org", "late-old") is None
    assert registry.resolve("lead@test.org", "new") == new.name
    new.state = EndpointState.DISCONNECTED
    registry.state_change(new)
    assert registry.resolve("lead@test.org", "new") is None
    assert registry.resolve("lead@test.org", "other") == other.name
    assert len(registry._bindings) == 1
