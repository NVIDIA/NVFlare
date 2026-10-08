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
from threading import Event


class CCAuthorizer(ABC):
    """Abstract base class for confidential computing authorizers"""

    supports_site_binding = False

    @abstractmethod
    def get_namespace(self) -> str:
        """Returns the namespace of the CCAuthorizer.

        Returns:
            namespace string

        """
        pass

    @abstractmethod
    def generate(self) -> str:
        """Generates and returns the active CCAuthorizer token.

        Returns:
            token string

        """
        pass

    @abstractmethod
    def verify(self, token: str) -> bool:
        """Returns the token verification result.

        Args:
            token: str

        Returns:
            a boolean value indicating the token verification result
        """
        pass

    def verify_for_site(self, token: str, site_name: str) -> bool:
        """Verify for a transport-authenticated site.

        Legacy attesters without a signed site claim retain their existing
        verification semantics. Site-aware attesters must override this method.
        """
        return self.verify(token)

    def generate_with_retry(self, timeout: float, cancel_event: Event) -> str:
        """Generate within a caller budget when supported by the authorizer.

        Legacy authorizers retain single-attempt behavior. Implementations that
        opt into retries must preserve verification and honor cancellation.
        This compatibility adapter cannot interrupt a blocking legacy generate()
        call; timeout is advisory here, not a promised wall-clock bound.
        """
        if cancel_event.is_set():
            raise CCTokenGenerateError("Token generation cancelled")
        return self.generate()


class CCTokenGenerateError(Exception):
    """Raised when a CC token generation failed"""

    pass


class CCTokenVerifyError(Exception):
    """Raised when a CC token verification failed"""

    pass
