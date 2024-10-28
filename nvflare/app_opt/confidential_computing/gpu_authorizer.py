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


from nvflare.app_opt.confidential_computing.cc_authorizer import CCAuthorizer

GPU_NAMESPACE = "x-nv-gpu-"


class GPUAuthorizer(CCAuthorizer):
    """Note: This is just a fake implementation for GPU authorizer. It will be replaced later
    with the real implementation.

    """

    def __init__(self, verifiers: list) -> None:
        """

        Args:
            verifiers (list):
                each element in this list is a dictionary and the keys of dictionary are
                "devices", "env", "url", "appraisal_policy_file" and "result_policy_file."

                the values of devices are "gpu" and "cpu"
                the values of env are "local" and "test"
                currently, valid combination is gpu + local

                url must be an empty string
                appraisal_policy_file must point to an existing file
                currently supports an empty file only

                result_policy_file must point to an existing file
                currently supports the following content only

                .. code-block:: json

                    {
                        "version":"1.0",
                        "authorization-rules":{
                            "x-nv-gpu-available":true,
                            "x-nv-gpu-attestation-report-available":true,
                            "x-nv-gpu-info-fetched":true,
                            "x-nv-gpu-arch-check":true,
                            "x-nv-gpu-root-cert-available":true,
                            "x-nv-gpu-cert-chain-verified":true,
                            "x-nv-gpu-ocsp-cert-chain-verified":true,
                            "x-nv-gpu-ocsp-signature-verified":true,
                            "x-nv-gpu-cert-ocsp-nonce-match":true,
                            "x-nv-gpu-cert-check-complete":true,
                            "x-nv-gpu-measurement-available":true,
                            "x-nv-gpu-attestation-report-parsed":true,
                            "x-nv-gpu-nonce-match":true,
                            "x-nv-gpu-attestation-report-driver-version-match":true,
                            "x-nv-gpu-attestation-report-vbios-version-match":true,
                            "x-nv-gpu-attestation-report-verified":true,
                            "x-nv-gpu-driver-rim-schema-fetched":true,
                            "x-nv-gpu-driver-rim-schema-validated":true,
                            "x-nv-gpu-driver-rim-cert-extracted":true,
                            "x-nv-gpu-driver-rim-signature-verified":true,
                            "x-nv-gpu-driver-rim-driver-measurements-available":true,
                            "x-nv-gpu-driver-vbios-rim-fetched":true,
                            "x-nv-gpu-vbios-rim-schema-validated":true,
                            "x-nv-gpu-vbios-rim-cert-extracted":true,
                            "x-nv-gpu-vbios-rim-signature-verified":true,
                            "x-nv-gpu-vbios-rim-driver-measurements-available":true,
                            "x-nv-gpu-vbios-index-conflict":true,
                            "x-nv-gpu-measurements-match":true
                        }
                    }

        """
        super().__init__()
        self.verifiers = verifiers

    def get_namespace(self) -> str:
        return GPU_NAMESPACE

    def generate(self) -> str:
        raise NotImplementedError

    def verify(self, token: str) -> bool:
        raise NotImplementedError
