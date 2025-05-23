#!/bin/bash

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

# Wait for Keycloak to be ready
echo "Waiting for Keycloak to be ready..."
until curl -sf http://keycloak:8080/realms/master > /dev/null; do
    printf '.'
    sleep 5
done


echo "Keycloak is ready!"

# Get Admin Token
ACCESS_TOKEN=$(curl -s -X POST "http://keycloak:8080/realms/master/protocol/openid-connect/token" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "username=admin" \
     -d "password=admin123" \
     -d "grant_type=password" \
     -d "client_id=admin-cli" | jq -r .access_token)

# Create Realm
curl -X POST "http://keycloak:8080/admin/realms" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer $ACCESS_TOKEN" \
     -d '{"realm": "myrealm", "enabled": true}'

# Create User
curl -X POST "http://keycloak:8080/admin/realms/myrealm/users" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer $ACCESS_TOKEN" \
     -d '{
           "username": "myuser@example.com",
           "enabled": true,
           "email": "myuser@example.com",
           "firstName": "My",
           "lastName": "User",
           "credentials": [{
               "type": "password",
               "value": "password123",
               "temporary": false
           }]
         }'

# Create Client
curl -X POST "http://keycloak:8080/admin/realms/myrealm/clients" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer $ACCESS_TOKEN" \
     -d '{
           "clientId": "myclient",
           "enabled": true,
           "protocol": "openid-connect",
           "publicClient": true,
           "redirectUris": ["http://localhost:8080/*"]
         }'

echo "Setup completed!"
