# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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


"""Factories shared by the split CoCo contract suites."""

from nvflare.lighter.cc_provision import workload_release as release

RUNTIMES = list(release.RUNTIME_TARGETS)

DIGEST = "ab" * 32

IMAGE = "secure-services.example.com:5000/workload@sha256:" + "c" * 64

PATHS = ["default/" + kind + "/demo" for kind in ("image-key", "sig-public-key", "security-policy")]


def authorization(runtime):
    return release.build_authorization(runtime, "demo", IMAGE, ["/bin/workload", "--silent"], DIGEST, PATHS)


def legacy_authorization():
    auth = authorization(RUNTIMES[0])
    auth["schema"] = release.V1
    auth["snp_init_data_sha256"] = auth.pop("init_data_sha256")
    auth["snp_init_data_encoding_in_trustee_v0_21"] = "lowercase-hex"
    for key in ("cpu_tee", "gpu", "init_data_claim"):
        del auth[key]
    return auth
