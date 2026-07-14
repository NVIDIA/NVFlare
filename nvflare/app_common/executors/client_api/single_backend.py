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

"""One-backend-per-resource guard shared by the Client API execution-mode backends.

Both out-of-the-CJ-thread modes drive a shared control plane keyed by a single well-known
identity, and a Client API job may legally wire more than one executor (TaskRouter accepts
multiple executors for disjoint task names). Two ClientAPIExecutors of the SAME mode would
then silently overwrite each other on that shared identity (last-writer-wins), cross-wiring
both trainers:

- ``in_process``: the process-singleton DataBus — both trainers' ``flare.init()`` resolve
  the single CLIENT_API_KEY to the last-installed API, and one trainer's result fires both
  backends' fixed-topic callbacks. Because the DataBus is a process singleton, the guard's
  effective scope in this mode is the process, not the job.
- ``external_process``: the job-scoped CJ Cell — ``register_request_cb`` keeps one callback
  per ``(channel, topic)``, so the second backend replaces the first's protocol handlers,
  and both launches collide on the same trainer FQCN leaf and bootstrap/pgid artifacts.

This is a misconfiguration, not a normal-operation path (the legacy InProcessClientAPI /
ClientAPILauncher executors have the identical latent collision and never guarded it), so
the remedy is to fail fast and deterministically at START_RUN — the executor converts the
raised error into system_panic — instead of hanging cross-wired at the first task.

The claim is held until the owner's ``release()`` (the end of finalize/unwind), NOT merely
until the owner marks itself closed: a backend that has begun teardown is still using the
shared resource (stopping its trainer, holding its handlers/subscriptions and launch
artifacts). Sequential reuse in one process (e.g. the simulator running jobs back to back)
reclaims cleanly because teardown releases before the next job initializes.
"""

import threading
import weakref
from typing import Optional


class SingleBackendGuard:
    """A process-wide registry enforcing one live backend per shared control-plane resource.

    Both sides of an entry are weak: the resource key, so the registry never extends a
    per-job Cell's lifetime, and the owner value, so a backend that leaked its claim (its
    teardown never ran) stops blocking the slot as soon as the leaked backend is collected.
    A claim whose owner is still reachable is honored — a live backend, even a wedged one,
    is still using the resource. Explicit ``release()`` in teardown remains the primary
    cleanup; the weak owner is the self-healing backstop, not the mechanism.
    """

    def __init__(self, mode: str, remedy: str, scope: str = "client job"):
        """
        Args:
            mode: the execution mode this guard protects (used in the rejection message).
            remedy: a one-line "how to fix your job config" clause appended to the message.
            scope: what one live backend is enforced per, for the rejection message —
                "client job" for a job-scoped resource (the CJ Cell); the in_process guard
                passes "process" because its resource (the DataBus) is process-scoped.
        """
        self._mode = mode
        self._remedy = remedy
        self._scope = scope
        self._active = weakref.WeakKeyDictionary()  # resource -> weakref.ref(owner backend)
        self._lock = threading.Lock()

    def claim(self, resource, owner) -> None:
        """Claims the slot for ``resource`` on behalf of ``owner``.

        Idempotent for the same owner. Raises RuntimeError if a different, still-live
        backend already holds it (rejecting the second backend without disturbing the
        first's claim). A dead owner's stale claim is reclaimed silently.
        """
        with self._lock:
            current = self._current_owner_locked(resource)
            if current is not None and current is not owner:
                raise RuntimeError(
                    f"another {self._mode} Client API backend is already active: V1 supports one "
                    f"{self._mode} executor per {self._scope} — {self._remedy}"
                )
            self._active[resource] = weakref.ref(owner)

    def release(self, resource, owner) -> None:
        """Releases the slot only if ``owner`` still holds it. Never raises.

        The ownership check is what keeps a rejected second backend's teardown from evicting
        the live first backend's claim (the loser never became the owner).
        """
        if resource is None:
            return
        with self._lock:
            if self._current_owner_locked(resource) is owner:
                del self._active[resource]

    def owner_of(self, resource) -> Optional[object]:
        """The live backend currently holding ``resource`` (or None). For assertions/diagnostics."""
        with self._lock:
            return self._current_owner_locked(resource)

    def _current_owner_locked(self, resource) -> Optional[object]:
        owner_ref = self._active.get(resource)
        return owner_ref() if owner_ref is not None else None
