# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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
import logging
import re
import ssl

NO_VERSION = -1

log = logging.getLogger(__name__)


def parse_openssl_version(version: str) -> (int, int, int):

    pattern = r"[a-zA-Z ]*([\d.]+).*"
    matches = re.match(pattern, version)
    if not matches:
        log.error(f"Invalid OpenSSL version: {version}")
        return None, None, None

    parts = matches.group(1).split(".")
    major = int(parts[0]) if len(parts) > 0 else NO_VERSION
    minor = int(parts[1]) if len(parts) > 1 else NO_VERSION
    patch = int(parts[2]) if len(parts) > 2 else NO_VERSION

    return major, minor, patch


def check_openssl(version: str = ssl.OPENSSL_VERSION) -> bool:
    """Check if vulnerabilities exist in this version. It only checks against
    most recent CVEs.

    Args:
        version: OpenSSL version string

    Returns: True if vulnerabilities are found
    """
    triple = parse_openssl_version(version)

    # Check CVE https://www.openssl.org/news/secadv/20221101.txt

    # OpenSSL versions 3.0.0 to 3.0.6 are vulnerable to this issue.
    if triple[0] == 3:
        return (3, 0, 0) <= triple <= (3, 0, 6)

    # OpenSSL 1.1.1 and 1.0.2 are not affected by this issue.
    if triple[0] == 1:
        return triple != (1, 1, 1) and triple != (1, 0, 2)

    return False


def warn_openssl(version: str = ssl.OPENSSL_VERSION):
    log.debug(f"OpenSSL version: {version}")

    if check_openssl(version):
        log.error(f"WARNING! The OpenSSL version '{version}' has known vulnerabilities, please upgrade!")


def warn():
    """Print warnings in the log if recent CVEs are found in any libraries that may affect NVFlare"""

    warn_openssl()


def check() -> bool:
    """Check if recent discovered vulnerabilities exist in any of the libraries.
    Note:
        This is not an exhaustive check for all CVEs in all the libraries, just most recent ones that may
        affect NVFlare
    Returns:
        True if vulnerabilities are found
    """
    return check_openssl()
