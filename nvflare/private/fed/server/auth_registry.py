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

import threading

from nvflare.fuel.f3.endpoint import EndpointMonitor, EndpointState


class NonRegularAuthRegistry(EndpointMonitor):
    """Track active delegated tokens and preserve live bindings during renewal."""

    def __init__(self, endpoint_lookup=None):
        self._endpoint_lookup = endpoint_lookup
        self._lock = threading.Lock()
        self._bindings = {}

    def register(self, name, token, fqcn, peer):
        with self._lock:
            if peer is None or peer.state == EndpointState.DISCONNECTED:
                return
            if self._endpoint_lookup and self._endpoint_lookup(peer.name) is not peer:
                return
            for existing_token, binding in self._bindings.items():
                if binding == (name, fqcn, peer):
                    # Keep in-flight messages valid until the renewal reply is
                    # received. Reconnects still retire the old transport's token.
                    return existing_token
            # Admin sessions have distinct FQCNs even when they share a user.
            self._bindings = {k: v for k, v in self._bindings.items() if v[1] != fqcn}
            self._bindings[token] = (name, fqcn, peer)
            return token

    def resolve(self, name, token):
        with self._lock:
            binding = self._bindings.get(token)
            return binding[1] if binding and binding[0] == name else None

    def state_change(self, endpoint):
        if endpoint.state not in (EndpointState.READY, EndpointState.DISCONNECTED):
            return
        with self._lock:
            if endpoint.state == EndpointState.READY:
                if self._endpoint_lookup and self._endpoint_lookup(endpoint.name) is not endpoint:
                    return
                # A replacement transport may retire the old Endpoint without a
                # DISCONNECTED event. Remove bindings from that old instance.
                self._bindings = {
                    k: v for k, v in self._bindings.items() if v[2].name != endpoint.name or v[2] is endpoint
                }
            else:
                # A late disconnect must not remove the replacement's binding.
                self._bindings = {k: v for k, v in self._bindings.items() if v[2] is not endpoint}
