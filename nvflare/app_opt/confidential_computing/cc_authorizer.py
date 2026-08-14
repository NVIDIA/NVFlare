# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from abc import ABC, abstractmethod


class CCAuthorizer(ABC):
    """Abstract base class for confidential computing authorizers"""

    @abstractmethod
    def get_namespace(self) -> str:
        """Returns the namespace of the CCAuthorizer.

        Returns:
            namespace string

        """
        pass

    @abstractmethod
    def generate(self, challenge: str | None = None) -> str:
        """Generates and returns the active CCAuthorizer token.

        Args:
            challenge: Request-scoped verifier challenge, when supported.

        Returns:
            token string

        """
        pass

    @abstractmethod
    def verify(self, token: str, challenge: str | None = None) -> bool:
        """Returns the token verification result.

        Args:
            token: str
            challenge: Request-scoped verifier challenge, when supported.

        Returns:
            a boolean value indicating the token verification result
        """
        pass

    def supports_challenge_binding(self) -> bool:
        """Whether this authorizer cryptographically binds a verifier-supplied
        ``challenge`` into the evidence returned by ``generate``/``verify``.

        Authorizers that return ``False`` cannot prove that a given token was
        produced for a specific request: a previously accepted token remains
        valid evidence forever, regardless of the challenge passed in. Callers
        relying on ``challenge`` for freshness (e.g. ``CCManager``) must not
        treat such authorizers as offering verifier-bound freshness and should
        compensate, for example with bounded replay detection.

        Returns:
            True if a passed-in ``challenge`` is bound into signed evidence.
        """
        return False


class CCTokenGenerateError(Exception):
    """Raised when a CC token generation failed"""

    pass


class CCTokenVerifyError(Exception):
    """Raised when a CC token verification failed"""

    pass
