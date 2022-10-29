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

from typing import List


class AttestationHelper(object):

    def __init__(self,
                 site_name: str,
                 attestation_service_endpoint: str,
                 orchestration_server_endpoint: str):
        """Create an AttestationHelper instance

        Args:
            site_name: name of the site
            attestation_service_endpoint: endpoint of the attestation service
            orchestration_server_endpoint: endpoint of the orchestration server
        """
        self.site_name = site_name
        self.attestation_service_endpoint = attestation_service_endpoint
        self.orchestration_server_endpoint = orchestration_server_endpoint

    def reset_participant(self, participant_name: str):
        pass

    def prepare(
            self,
            claim_policy_file_path: str,
            requirement_policy_file_path: str) -> str:
        """Prepare for attestation process
        This is a complex step that performs multiple interactions with the attestation service
        and the orchestration server:
            - load claim policy and requirement policy
            - register the claim policy with the attestation service
            - get a CC token from the attestation service.
            - register the token with the orchestration server

        Args:
            claim_policy_file_path: path to the claim policy
            requirement_policy_file_path: path to the requirement policy

        Returns: error if any

        """
        pass

    def validate_participants(
            self,
            participants: List[str]) -> str:
        """Validate CC policies of specified participants against the requirement policy of the site.
            - get CC tokens of the participants from the Orchestration Server
            - get claim policies from the Attestation Service based of the participants using their tokens
            - check the participants claim policies against the requirement policy of the site to see
            whether requirements are satisfied.

        Args:
            participants: list of participant names

        Returns: error if any

        """
        pass
