# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from datetime import datetime, timedelta, timezone

import jwt
from app.core.config import settings


def create_token():
    """Get the test token

    Creates a JWT token just for testing the REST API service.

    Returns:
        A encoded JWT token.
    """
    payload = {
        "sub": settings.test_user,  # Subject of the token (typically the user identifier)
        "iat": datetime.now(timezone.utc),  # Issued at time
        "exp": datetime.now(timezone.utc) + timedelta(days=settings.key_expiry_days),  # Expiration time
    }
    token = jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)
    return token


if __name__ == "__main__":
    token = create_token()
    print(token)
