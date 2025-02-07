# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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


import os
import sys

import requests


def save_access_token(access_token: str, destination_path):

    # Ensure the destination directory exists
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)

    with open(destination_path, "w") as f:
        f.write(access_token)
    print(f"Access token saved to {destination_path}")


def get_keycloak_acces_token(username, password, client_id, keycloak_url) -> str:

    # Request payload
    data = {"username": username, "password": password, "grant_type": "password", "client_id": client_id}

    try:
        # Make a POST request to get the access token
        response = requests.post(keycloak_url, data=data, headers={"Content-Type": "application/x-www-form-urlencoded"})
        response_data = response.json()

        # Extract the access token
        access_token = response_data.get("access_token")

        if not access_token:
            print("Failed to retrieve access token.")
        else:
            return access_token

    except Exception as e:
        print(f"Error fetching access token: {e}")


if __name__ == "__main__":

    # Define variables
    keycloak_url = "http://localhost:8080/realms/master/protocol/openid-connect/token"
    username = "admin"
    password = "admin123"
    client_id = "admin-cli"
    destination_path = sys.argv[1]

    token = get_keycloak_acces_token(
        username=username, password=password, client_id=client_id, keycloak_url=keycloak_url
    )

    print("token=", token)

    save_access_token(token, destination_path=destination_path)
