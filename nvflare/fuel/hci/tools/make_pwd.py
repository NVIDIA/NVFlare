# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import getpass

from nvflare.fuel.hci.security import hash_password


def main():
    """
    TODO: should this file be removed?

    """
    user_name = input("User Name: ")

    pwd = getpass.getpass("Password (8 or more chars): ")

    if len(pwd) < 8:
        print("Invalid password - must have at least 8 chars")
        return

    pwd2 = getpass.getpass("Confirm Password: ")

    if pwd != pwd2:
        print("Passwords mismatch")
        return

    result = hash_password(user_name + pwd)
    print("Password Hash: {}".format(result))


if __name__ == "__main__":
    main()
