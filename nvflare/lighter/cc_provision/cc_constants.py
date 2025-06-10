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


class CCConfigKey:
    COMPUTE_ENV = "compute_env"
    CC_CPU_MECHANISM = "cc_cpu_mechanism"
    CC_GPU_MECHANISM = "cc_gpu_mechanism"
    CC_ISSUERS = "cc_issuers"
    CC_ATTESTATION_CONFIG = "cc_attestation"
    CVM_IMAGE_NAME = "cvm_image_name"


class CCConfigValue:
    # Compute environments
    AZURE_CVM = "azure_cvm"
    AZURE_CONFIDENTIAL_CONTAINER = "azure_confidential_container"
    ONPREM_CVM = "onprem_cvm"
    MOCK = "mock"

    # CC CPU mechanisms
    AMD_SEV_SNP = "amd_sev_snp"
    INTEL_TDX = "intel_tdx"

    # CC GPU mechanisms
    NVIDIA_CC = "nvidia_cc"


# CC Manager constants
CC_AUTHORIZERS_KEY = "cc_authorizers"


class CCManagerArgs:
    CC_ISSUERS_CONF = "cc_issuers_conf"
    CC_VERIFIER_IDS = "cc_verifier_ids"
    VERIFY_FREQUENCY = "verify_frequency"
    CRITICAL_LEVEL = "critical_level"
    CC_ENABLED_SITES = "cc_enabled_sites"


class CCIssuerConfig:
    ID = "id"
    TOKEN_EXPIRATION = "token_expiration"
    PATH = "path"
    ARGS = "args"
