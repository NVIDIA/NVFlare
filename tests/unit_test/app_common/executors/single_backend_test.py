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

"""Tests for SingleBackendGuard itself. The backend test modules cover the guard as wired
into each backend's initialize/finalize; this module covers the registry semantics directly,
including the weak-owner self-healing that the backend paths cannot exercise."""

import gc

import pytest

from nvflare.app_common.executors.client_api.single_backend import SingleBackendGuard


class _Resource:
    pass


class _Backend:
    pass


class TestSingleBackendGuard:
    def test_claim_is_idempotent_for_same_owner(self):
        guard = SingleBackendGuard(mode="test", remedy="fix it")
        resource, owner = _Resource(), _Backend()
        guard.claim(resource, owner)
        guard.claim(resource, owner)
        assert guard.owner_of(resource) is owner

    def test_second_owner_is_rejected_and_message_names_scope(self):
        guard = SingleBackendGuard(mode="test", remedy="fix it", scope="process")
        resource = _Resource()
        winner, loser = _Backend(), _Backend()
        guard.claim(resource, winner)
        with pytest.raises(RuntimeError, match="one test executor per process"):
            guard.claim(resource, loser)
        assert guard.owner_of(resource) is winner

    def test_default_scope_is_client_job(self):
        guard = SingleBackendGuard(mode="test", remedy="fix it")
        resource = _Resource()
        winner = _Backend()  # must stay referenced: an unreferenced owner is (correctly) reclaimed
        guard.claim(resource, winner)
        with pytest.raises(RuntimeError, match="one test executor per client job"):
            guard.claim(resource, _Backend())

    def test_losers_release_does_not_evict_winner(self):
        guard = SingleBackendGuard(mode="test", remedy="fix it")
        resource = _Resource()
        winner, loser = _Backend(), _Backend()
        guard.claim(resource, winner)
        guard.release(resource, loser)
        assert guard.owner_of(resource) is winner

    def test_release_none_resource_is_noop(self):
        guard = SingleBackendGuard(mode="test", remedy="fix it")
        guard.release(None, _Backend())

    def test_release_frees_slot_for_next_owner(self):
        guard = SingleBackendGuard(mode="test", remedy="fix it")
        resource = _Resource()
        first, second = _Backend(), _Backend()
        guard.claim(resource, first)
        guard.release(resource, first)
        assert guard.owner_of(resource) is None
        guard.claim(resource, second)
        assert guard.owner_of(resource) is second

    def test_dead_owner_claim_self_heals(self):
        """A backend that leaked its claim (teardown never ran) must stop blocking the slot
        once it is collected — the process-scoped DataBus guard would otherwise lock out
        every later in_process backend in the process."""
        guard = SingleBackendGuard(mode="test", remedy="fix it")
        resource = _Resource()
        leaked = _Backend()
        guard.claim(resource, leaked)
        del leaked
        gc.collect()
        assert guard.owner_of(resource) is None
        replacement = _Backend()
        guard.claim(resource, replacement)  # must not raise
        assert guard.owner_of(resource) is replacement

    def test_live_owner_is_not_reclaimed(self):
        """Weak owner is a backstop for LEAKED backends only: a reachable owner, even one
        that never releases, keeps its claim."""
        guard = SingleBackendGuard(mode="test", remedy="fix it")
        resource = _Resource()
        owner = _Backend()
        guard.claim(resource, owner)
        gc.collect()
        with pytest.raises(RuntimeError, match="already active"):
            guard.claim(resource, _Backend())
        assert guard.owner_of(resource) is owner

    def test_guard_does_not_pin_resource(self):
        """The registry must never extend a per-job Cell's lifetime."""
        import weakref

        guard = SingleBackendGuard(mode="test", remedy="fix it")
        resource = _Resource()
        resource_ref = weakref.ref(resource)
        guard.claim(resource, _Backend())
        del resource
        gc.collect()
        assert resource_ref() is None
