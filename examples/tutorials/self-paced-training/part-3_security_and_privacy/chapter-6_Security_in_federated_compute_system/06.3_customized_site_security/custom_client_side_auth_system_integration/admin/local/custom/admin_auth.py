# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import json

import requests

from nvflare.fuel.hci.client.event import EventContext, EventHandler, EventPropKey, EventType


class AdminAuth(EventHandler):
    def __init__(self, orgs: dict):
        # orgs is a dict of name => endpoint of the org's auth service
        self.orgs = orgs
        self.auth_tokens = {}
        self.authentication_done = False
        self.passwords = {}

    def _get_passwords_to_all_sites(self):
        # This example asks the admin user to type in the password to each org.
        for org_name, _ in self.orgs.items():
            password = getpass.getpass(f"Password to {org_name}:")
            self.passwords[org_name] = password

    def _auth_org(self, user_name: str, org_name: str, endpoint: str) -> str:
        try:
            # The access token query depending on the KeyCloak user and client set up.
            # We set up the user using the same admin user name for demonstrating.
            payload = {
                "client_id": "myclient",
                "username": user_name,
                "password": self.passwords[org_name],
                "grant_type": "password",
            }
            response = requests.post(endpoint, data=payload)
            token = json.loads(response.text).get("access_token")
        except:
            token = None
        # If raising an exception here, it will prevent the admin tool connecting to the admin server
        # and terminating the admin tool.
        return f"{user_name}:{token}"

    def _authenticate_user_to_all_sites(self, ctx: EventContext):
        user_name = ctx.get_prop(EventPropKey.USER_NAME)
        for org_name, ep in self.orgs.items():
            access_token = self._auth_org(user_name, org_name, ep)
            self.auth_tokens[org_name] = access_token

    def handle_event(self, event_type: str, ctx: EventContext):
        if event_type == EventType.LOGIN_SUCCESS:
            # called after the user is logged in successfully
            # print("got event: LOGIN_SUCCESS")
            if not self.authentication_done:
                # print("authenticating user to orgs ...")
                self._authenticate_user_to_all_sites(ctx)
        elif event_type == EventType.BEFORE_EXECUTE_CMD:
            cmd_name = ctx.get_prop(EventPropKey.CMD_NAME)
            # print(f"got event: BEFORE_EXECUTE_CMD for cmd {cmd_name}")
            if cmd_name == "submit_job":
                # print(f"adding auth_tokens: {self.auth_tokens}")
                ctx.set_custom_prop("auth_tokens", self.auth_tokens)
                # print("added custom prop!")
        elif event_type == EventType.BEFORE_LOGIN:
            self._get_passwords_to_all_sites()
